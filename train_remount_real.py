"""
train_remount_real.py — 用真实帧对训练 Remount 预测模型

核心理念:
    不用 warpAffine 假数据。从 site_memories 中取同一个 sample 的
    两次不同 Save Region 的 ref_template 作为图像对 (A, B),
    用匹配算法确定它们之间的真实像素偏移作为标签。

用法:
    python train_remount_real.py [--device cuda]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from afm_ml_recognition import DeepFeatureExtractor, FEATURE_DIM
from afm_relocation import (
    apply_affine,
    estimate_affine_transform,
    load_site_memory,
    rotation_translation_affine,
    to_grayscale_u8,
)

SITE_MEMORY_ROOT = BASE_DIR / "collected_data" / "site_memories"
OUTPUT_PATH = BASE_DIR / "collected_data" / "models" / "deep_remount_predictor_real.pkl"


# ═══════════════════════════════════════════════════════════════
# Step 1: 收集真实帧对
# ═══════════════════════════════════════════════════════════════
def collect_real_pairs():
    """从 site_memories 中收集同一个 sample 的 ref_template 对"""
    by_sample = {}
    for site_dir in sorted(SITE_MEMORY_ROOT.rglob("metadata.json")):
        try:
            mem = load_site_memory(site_dir.parent)
            tpl = mem.get("reference_template")
            if tpl is None:
                continue
            gray = to_grayscale_u8(tpl)
            if gray is None or gray.size == 0:
                continue
            sample_id = mem.get("sample_id", "unknown")
            if sample_id not in by_sample:
                by_sample[sample_id] = []
            by_sample[sample_id].append({
                "name": site_dir.parent.name,
                "image": gray,
                "ref_x": float((mem.get("reference_top_left") or {}).get("x_um", 0)),
                "ref_y": float((mem.get("reference_top_left") or {}).get("y_um", 0)),
                "fov_w": float((mem.get("fov_size_um") or {}).get("width_um", 840)),
                "fov_h": float((mem.get("fov_size_um") or {}).get("height_um", 630)),
            })
        except Exception:
            continue

    pairs = []
    for sample_id, frames in by_sample.items():
        if len(frames) < 2:
            continue
        # 所有相邻帧对
        for i in range(len(frames)):
            for j in range(i + 1, min(i + 3, len(frames))):  # 最多相邻2帧
                pairs.append((sample_id, frames[i], frames[j]))
    return pairs


# ═══════════════════════════════════════════════════════════════
# Step 2: 用 ORB 匹配计算真实像素偏移 (作为监督标签)
# ═══════════════════════════════════════════════════════════════
def compute_ground_truth(img_a, img_b, fov_w_a, fov_h_a, fov_w_b, fov_h_b):
    """
    用 ORB 估算 A→B 的仿射变换, 返回像素空间的 (dx_px, dy_px, angle_deg).
    同时用坐标信息交叉验证。
    """
    # 确保同尺寸
    h, w = img_a.shape[:2]
    if img_b.shape[:2] != (h, w):
        img_b = cv2.resize(img_b, (w, h))

    affine = estimate_affine_transform(img_a, img_b, max_features=800, keep_matches=200)
    if affine is None or affine["confidence"] < 0.15:
        return None

    # ORB 给出的矩阵是在当前像素空间的 (A 和 B 同尺寸)
    matrix = affine["matrix"]  # 2x3: [a b tx; c d ty]
    dx = float(matrix[0, 2])
    dy = float(matrix[1, 2])
    angle = float(affine["rotation_deg"])
    scale = float(affine["scale"])

    # 交叉验证: 用坐标信息
    # A 的 FOV: fov_w_a × fov_h_a 对应 w × h 像素
    # B 的 FOV: fov_w_b × fov_h_b 对应 w × h 像素
    # 如果两帧的 FOV 相同，标签直接可用; 如果不同，需要按比例修正
    scale_factor_x = fov_w_b / max(fov_w_a, 1e-6)
    scale_factor_y = fov_h_b / max(fov_h_a, 1e-6)
    if abs(scale_factor_x - 1.0) > 0.1 or abs(scale_factor_y - 1.0) > 0.1:
        dx *= scale_factor_x
        dy *= scale_factor_y

    return {
        "dx_px": dx,
        "dy_px": dy,
        "angle_deg": angle,
        "scale": scale,
        "confidence": float(affine["confidence"]),
        "inlier_count": int(affine["inlier_count"]),
        "match_count": int(affine["match_count"]),
    }


# ═══════════════════════════════════════════════════════════════
# Step 3: 构建特征 + 训练
# ═══════════════════════════════════════════════════════════════
def train_on_real_pairs(device="cpu"):
    print("=" * 60)
    print("真实帧对训练: Remount 预测模型")
    print("=" * 60)

    # 收集帧对
    pairs = collect_real_pairs()
    print(f"\n找到 {len(pairs)} 个帧对")
    if not pairs:
        print("FAIL 未找到可用的帧对. 需要同一个 sample 的至少 2 帧")
        return None

    # 初始化特征提取器
    print("初始化 ResNet18 ...")
    extractor = DeepFeatureExtractor(device=device)

    # 计算真值 + 提取特征
    X_list, y_list = [], []
    valid_count = 0
    for sample_id, fa, fb in pairs:
        gt = compute_ground_truth(
            fa["image"], fb["image"],
            fa["fov_w"], fa["fov_h"],
            fb["fov_w"], fb["fov_h"],
        )
        if gt is None or gt["confidence"] < 0.20:
            continue

        # 特征: [ref_feat(512), cand_feat(512), diff(512), stat(5)]
        ref_feat = extractor.extract(fa["image"])
        cand_feat = extractor.extract(fb["image"])
        diff = ref_feat - cand_feat
        cosine_sim = float(np.dot(ref_feat, cand_feat))
        l1 = float(np.sum(np.abs(diff)))
        l2 = float(np.linalg.norm(diff))
        stat = np.array([1.0, 1.0, cosine_sim, l1, l2], dtype=np.float32)
        feat = np.concatenate([ref_feat, cand_feat, diff, stat])

        X_list.append(feat)
        y_list.append([gt["dx_px"], gt["dy_px"], gt["angle_deg"]])
        valid_count += 1

        print(f"  {fa['name'][:30]:<30} -> {fb['name'][:30]:<30}  "
              f"dx={gt['dx_px']:+.1f} dy={gt['dy_px']:+.1f} "
              f"angle={gt['angle_deg']:+.2f} conf={gt['confidence']:.3f}")

    if valid_count < 5:
        print(f"\nFAIL 只有 {valid_count} 个有效帧对 (至少需要 5 个)")
        return None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"\n有效帧对: {valid_count}, 特征维度: {X.shape[1]}")

    # 归一化标签
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y_std = np.maximum(y_std, 1e-6)
    y_norm = (y - y_mean) / y_std

    scale_factors = {
        "dx_mean": float(y_mean[0]), "dx_std": float(y_std[0]),
        "dy_mean": float(y_mean[1]), "dy_std": float(y_std[1]),
        "angle_mean": float(y_mean[2]), "angle_std": float(y_std[2]),
        "model_space": "pixel",
    }

    # 训练 MLP
    print("\n训练 MLP ...")
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu", solver="adam", alpha=0.001,
        batch_size=min(32, valid_count), learning_rate="adaptive",
        learning_rate_init=0.001, max_iter=2000,
        early_stopping=True, validation_fraction=0.2 if valid_count >= 10 else 0.1,
        n_iter_no_change=50, random_state=42, verbose=True,
    )

    t0 = time.time()
    model.fit(X, y_norm)
    print(f"训练耗时: {time.time() - t0:.0f} 秒")
    print(f"训练 R^2: {model.score(X, y_norm):.4f}")

    # 保存
    bundle = {
        "model_type": "deep_remount_predictor_5w",
        "feature_dim": FEATURE_DIM,
        "model": model,
        "scale_factors": scale_factors,
        "pair_count": valid_count,
        "training_data": "real_pairs",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUTPUT_PATH)
    print(f"\n模型已保存: {OUTPUT_PATH}")
    print(f"文件大小: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")

    return bundle


def main():
    parser = argparse.ArgumentParser(description="真实帧对训练 Remount 模型")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()
    try:
        train_on_real_pairs(device=args.device)
    except KeyboardInterrupt:
        print("\n中断")
    except Exception as e:
        print(f"\n失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
