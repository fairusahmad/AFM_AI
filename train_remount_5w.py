"""
train_remount_5w.py — 大样本 Remount 变换回归训练（5万样本/锚点）

方案：
  1. 以每次 save region 保存的 reference_template 为局部锚点
     （不依赖其绝对物理位姿）
  2. 对每个锚点图生成 50,000 张带已知相对仿射变换的歪图
  3. 用 ResNet18 提取 (锚点, 歪图) 的差分特征
  4. 训练 MLP 回归器预测相对偏移 (dx_px, dy_px, angle_deg)
  5. 模型保存为 deep_remount_predictor_5w.pkl

与旧版区别：
  - 旧版: overview 图像, 12 样本/站点, 输出 dx_um/dy_um
  - 新版: reference_template 锚点, 50000 样本/锚点, 输出 dx_px/dy_px/angle_deg
          （像素空间，调用方自行 × scale 转 um）

使用:
  python train_remount_5w.py [--samples 50000] [--batch 256] [--device cuda]
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.neural_network import MLPRegressor

# 复用现有模块
from afm_ml_recognition import DeepFeatureExtractor, deep_pair_features, FEATURE_DIM
from afm_phase2_ml import _load_site_memories
from afm_relocation import apply_affine, rotation_translation_affine, to_grayscale_u8

BASE_DIR = Path(__file__).resolve().parent
SITE_MEMORY_ROOT = BASE_DIR / "collected_data" / "site_memories"
DEFAULT_SAMPLES_PER_ANCHOR = 50000


def generate_warped_image(anchor_gray, rng):
    """对锚点灰度图施加随机旋转+平移+缩放，返回 (warped, dx_px, dy_px, angle_deg, scale)"""
    h, w = anchor_gray.shape[:2]

    # 变换参数范围（比旧版 ±8°/±60px 更宽，覆盖真实 remount 场景）
    angle_deg = float(rng.uniform(-15, 15))
    shift_x_px = float(rng.uniform(-0.3 * w, 0.3 * w))
    shift_y_px = float(rng.uniform(-0.3 * h, 0.3 * h))
    scale = float(rng.uniform(0.85, 1.15))

    matrix = rotation_translation_affine(
        w, h,
        angle_deg=angle_deg,
        shift_x_px=shift_x_px,
        shift_y_px=shift_y_px,
        scale=scale,
    )

    warped = apply_affine(anchor_gray, matrix, output_shape=(h, w))

    # 添加轻微亮度/对比度抖动（模拟光照变化）
    alpha = float(rng.uniform(0.92, 1.08))
    beta = float(rng.uniform(-10, 10))
    warped = np.clip(warped.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return warped, shift_x_px, shift_y_px, angle_deg


def build_5w_dataset(
    site_memory_root,
    extractor,
    samples_per_anchor=50000,
    batch_size=256,
):
    """构建大样本 remount 回归数据集。

    Returns:
        X: (N, 1541) 差分特征矩阵
        y: (N, 3) 标签 [dx_px, dy_px, angle_deg]
        anchor_count: 使用的锚点图数量
    """
    site_memories = _load_site_memories(site_memory_root)
    if not site_memories:
        raise ValueError("No saved site memories found.")

    # 收集所有锚点图 (reference_template)
    anchors = []
    for site_dir, memory in site_memories:
        template = memory.get("reference_template")
        if template is not None:
            gray = to_grayscale_u8(template)
            if gray is not None and gray.size > 0:
                anchors.append((site_dir.name, gray))

    # 去重：相同图像内容只保留一份（MD5哈希）
    seen = set()
    unique_anchors = []
    for name, gray in anchors:
        h = hashlib.md5(gray.tobytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_anchors.append((name, gray))
    if len(unique_anchors) < len(anchors):
        print(f"去重: {len(anchors)} → {len(unique_anchors)} 个唯一锚点"
              f"（跳过 {len(anchors) - len(unique_anchors)} 个重复）")
    anchors = unique_anchors

    if not anchors:
        raise ValueError("No valid reference_template found in any site_memory.")

    print(f"找到 {len(anchors)} 个锚点图")
    for name, img in anchors:
        print(f"  - {name}: {img.shape[1]}x{img.shape[0]}")

    total_samples = len(anchors) * samples_per_anchor
    print(f"\n目标总样本数: {total_samples:,} ({len(anchors)} 锚点 × {samples_per_anchor:,})")

    # 预分配（1541 = 512*3 + 5）
    feat_dim = FEATURE_DIM * 3 + 5  # deep_pair_features 输出维度
    X_all = np.zeros((total_samples, feat_dim), dtype=np.float32)
    y_all = np.zeros((total_samples, 3), dtype=np.float32)

    rng = np.random.RandomState(42)
    sample_idx = 0
    t_start = time.time()

    for anchor_idx, (anchor_name, anchor_gray) in enumerate(anchors):
        print(f"\n[锚点 {anchor_idx + 1}/{len(anchors)}] {anchor_name}")

        # 预提取锚点特征（复用，避免重复计算）
        anchor_feat = extractor.extract(anchor_gray)
        anchor_feat_t = torch.from_numpy(anchor_feat).unsqueeze(0).to(extractor.device)

        t_anchor_start = time.time()
        for batch_start in range(0, samples_per_anchor, batch_size):
            batch_end = min(batch_start + batch_size, samples_per_anchor)
            batch_count = batch_end - batch_start

            # 生成歪图批次
            warped_batch = []
            labels_batch = []
            for _ in range(batch_count):
                warped, dx, dy, angle = generate_warped_image(anchor_gray, rng)
                warped_batch.append(warped)
                labels_batch.append([dx, dy, angle])

            # 批量提取歪图特征
            warped_feats = extractor.extract_batch(warped_batch)

            # 计算 deep_pair_features（用预计算的 anchor 特征加速）
            for k in range(batch_count):
                wf = warped_feats[k]
                diff = anchor_feat - wf

                # 统计特征
                cosine_sim = float(np.dot(anchor_feat, wf))
                l1_dist = float(np.sum(np.abs(diff)))
                l2_dist = float(np.linalg.norm(diff))
                stat = np.array([1.0, 1.0, cosine_sim, l1_dist, l2_dist], dtype=np.float32)

                # 拼接: ref_feat(512) + cand_feat(512) + diff_feat(512) + stat(5)
                X_all[sample_idx] = np.concatenate([anchor_feat, wf, diff, stat])
                y_all[sample_idx] = labels_batch[k]
                sample_idx += 1

            # 进度
            if (batch_end % 5000) == 0 or batch_end == samples_per_anchor:
                elapsed = time.time() - t_anchor_start
                rate = batch_end / max(elapsed, 0.1)
                eta = (samples_per_anchor - batch_end) / max(rate, 1)
                print(f"  {batch_end:,}/{samples_per_anchor:,} "
                      f"({batch_end * 100 // samples_per_anchor}%) "
                      f"速率: {rate:.0f} 样本/秒, 预计剩余: {eta:.0f}秒")

        t_anchor_elapsed = time.time() - t_anchor_start
        print(f"  完成, 耗时: {t_anchor_elapsed:.0f}秒 "
              f"({samples_per_anchor / max(t_anchor_elapsed, 0.1):.0f} 样本/秒)")

    t_total = time.time() - t_start
    print(f"\n总耗时: {t_total:.0f}秒 ({t_total / 60:.1f}分钟)")
    print(f"总样本: {sample_idx:,}")
    print(f"特征维度: {X_all.shape[1]}")

    return X_all, y_all, len(anchors)


def train_remount_5w(
    site_memory_root=None,
    output_path=None,
    samples_per_anchor=50000,
    batch_size=256,
    device="cpu",
):
    """训练 5w 样本 Remount MLP 回归器"""
    if site_memory_root is None:
        site_memory_root = SITE_MEMORY_ROOT
    if output_path is None:
        output_path = BASE_DIR / "collected_data" / "models" / "deep_remount_predictor_5w.pkl"

    site_memory_root = Path(site_memory_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("大样本 Remount 回归训练 (5万样本/锚点)")
    print("=" * 60)
    print(f"数据源: {site_memory_root}")
    print(f"输出:   {output_path}")
    print(f"设备:   {device}")
    print()

    # 初始化特征提取器
    print("初始化 ResNet18 特征提取器 ...")
    extractor = DeepFeatureExtractor(device=device)
    print(f"特征提取器就绪 (设备: {extractor.device})")
    print()

    # 构建数据集
    print("构建 5w 样本数据集 ...")
    X, y, anchor_count = build_5w_dataset(
        site_memory_root,
        extractor,
        samples_per_anchor=samples_per_anchor,
        batch_size=batch_size,
    )

    print(f"\n最终数据集: {X.shape[0]:,} 样本, {X.shape[1]} 特征")
    print(f"标签范围:")
    print(f"  dx_px:    [{y[:, 0].min():.1f}, {y[:, 0].max():.1f}]")
    print(f"  dy_px:    [{y[:, 1].min():.1f}, {y[:, 1].max():.1f}]")
    print(f"  angle_deg: [{y[:, 2].min():.1f}, {y[:, 2].max():.1f}]")

    # 归一化标签（稳定训练）
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y_std = np.maximum(y_std, 1e-6)
    y_norm = (y - y_mean) / y_std

    scale_factors = {
        "dx_mean": float(y_mean[0]),
        "dx_std": float(y_std[0]),
        "dy_mean": float(y_mean[1]),
        "dy_std": float(y_std[1]),
        "angle_mean": float(y_mean[2]),
        "angle_std": float(y_std[2]),
        "model_space": "pixel",  # 标记为像素空间
    }

    # 训练 MLP（更大网络 + 更多迭代 → 适配大数据量）
    print("\n训练 MLP 回归器 ...")
    model = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42,
        verbose=True,
    )

    t_train_start = time.time()
    model.fit(X, y_norm)
    t_train = time.time() - t_train_start

    train_score = model.score(X, y_norm)
    print(f"\n训练完成, 耗时: {t_train:.0f}秒 ({t_train / 60:.1f}分钟)")
    print(f"训练 R²: {train_score:.4f}")
    print(f"最终迭代: {model.n_iter_}")
    print(f"损失: {model.loss_:.6f}")

    # 保存
    bundle = {
        "model_type": "deep_remount_predictor_5w",
        "feature_dim": FEATURE_DIM,
        "model": model,
        "scale_factors": scale_factors,
        "anchor_count": anchor_count,
        "samples_per_anchor": samples_per_anchor,
        "total_samples": int(X.shape[0]),
    }
    joblib.dump(bundle, output_path)
    print(f"\n模型已保存到: {output_path}")
    print(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return bundle


def main():
    parser = argparse.ArgumentParser(
        description="大样本 Remount 回归训练 (5万样本/锚点)"
    )
    parser.add_argument(
        "--samples", type=int, default=50000,
        help=f"每个锚点的合成样本数 (默认: {DEFAULT_SAMPLES_PER_ANCHOR})"
    )
    parser.add_argument(
        "--batch", type=int, default=256,
        help="特征提取批次大小 (默认: 256)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="计算设备: cpu 或 cuda (默认: cpu)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出模型路径 (默认: collected_data/models/deep_remount_predictor_5w.pkl)"
    )
    args = parser.parse_args()

    try:
        train_remount_5w(
            samples_per_anchor=args.samples,
            batch_size=args.batch,
            device=args.device,
            output_path=args.output,
        )
    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n训练失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
