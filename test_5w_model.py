"""
test_5w_model.py — 独立验证 5w Remount 预测模型

用法:
    python test_5w_model.py

功能:
    1. 加载 deep_remount_predictor_5w.pkl
    2. 从 site_memories 中读取 reference_template 作为锚点
    3. 对锚点施加 已知 的旋转+平移，生成测试图
    4. 用模型预测偏移量，对比真值计算误差
    5. (可选) 在两帧真实 site_memory 间测试
"""

import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

# ── 复用项目模块 ──
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from afm_ml_recognition import (
    DeepFeatureExtractor,
    MLTransformPredictor5W,
    load_deep_remount_predictor_5w,
    deep_pair_features,
    FEATURE_DIM,
)
from afm_relocation import apply_affine, rotation_translation_affine, to_grayscale_u8, load_site_memory

MODEL_PATH = BASE_DIR / "collected_data" / "models" / "deep_remount_predictor_5w.pkl"
SITE_MEMORY_ROOT = BASE_DIR / "collected_data" / "site_memories"


# ═══════════════════════════════════════════════════════════════
# 测试 1: 合成数据 — 用模型训练完全一致的方式
# ═══════════════════════════════════════════════════════════════
def test_synthetic(extractor, predictor):
    """对锚点图施加已知变换，验证模型预测精度"""
    print("\n" + "=" * 60)
    print("测试 1: 合成数据 (已知真值)")
    print("=" * 60)

    # 找一个真实锚点
    anchors = []
    for site_dir in sorted(SITE_MEMORY_ROOT.rglob("metadata.json")):
        try:
            memory = load_site_memory(site_dir.parent)
            tpl = memory.get("reference_template")
            if tpl is not None:
                gray = to_grayscale_u8(tpl)
                if gray is not None and gray.size > 0:
                    anchors.append((site_dir.parent.name, gray))
        except Exception:
            continue

    if not anchors:
        print("[FAIL] 未找到任何 site_memory 锚点图")
        return

    anchor_name, anchor = anchors[0]
    h, w = anchor.shape[:2]
    print(f"锚点: {anchor_name}  ({w}x{h})")

    # 测试多个随机变换
    rng = np.random.RandomState(42)
    test_cases = []

    # 小偏移 (模拟微小 remount)
    for _ in range(3):
        angle = float(rng.uniform(-3, 3))
        dx = float(rng.uniform(-60, 60))
        dy = float(rng.uniform(-60, 60))
        test_cases.append(("小偏移", dx, dy, angle))

    # 中等偏移
    for _ in range(3):
        angle = float(rng.uniform(-8, 8))
        dx = float(rng.uniform(-200, 200))
        dy = float(rng.uniform(-200, 200))
        test_cases.append(("中偏移", dx, dy, angle))

    # 大偏移
    for _ in range(3):
        angle = float(rng.uniform(-12, 12))
        dx = float(rng.uniform(-0.25 * w, 0.25 * w))
        dy = float(rng.uniform(-0.25 * h, 0.25 * h))
        test_cases.append(("大偏移", dx, dy, angle))

    print(f"\n{'类型':<8} {'真值 dx':>8} {'真值 dy':>8} {'真值 deg':>8} | "
          f"{'预测 dx':>8} {'预测 dy':>8} {'预测 deg':>8} | "
          f"{'Δdx':>8} {'Δdy':>8} {'Δdeg':>6}")
    print("-" * 90)

    errors_dx, errors_dy, errors_deg = [], [], []

    for label, dx_true, dy_true, angle_true in test_cases:
        matrix = rotation_translation_affine(
            w, h, angle_deg=angle_true, shift_x_px=dx_true, shift_y_px=dy_true, scale=1.0
        )
        warped = apply_affine(anchor, matrix, output_shape=(h, w))

        # 预测
        result = predictor.predict(anchor, warped)
        if result is None:
            print(f"{label:<8} {'FAIL':>60}")
            continue

        dx_pred = result["dx_px"]
        dy_pred = result["dy_px"]
        angle_pred = result["angle_deg"]

        e_dx = dx_pred - dx_true
        e_dy = dy_pred - dy_true
        e_deg = angle_pred - angle_true
        errors_dx.append(abs(e_dx))
        errors_dy.append(abs(e_dy))
        errors_deg.append(abs(e_deg))

        print(f"{label:<8} {dx_true:>+8.1f} {dy_true:>+8.1f} {angle_true:>+8.2f} | "
              f"{dx_pred:>+8.1f} {dy_pred:>+8.1f} {angle_pred:>+8.2f} | "
              f"{e_dx:>+8.1f} {e_dy:>+8.1f} {e_deg:>+6.2f}")

    print("-" * 90)
    if errors_dx:
        print(f"平均绝对误差: dx={np.mean(errors_dx):.1f} px, "
              f"dy={np.mean(errors_dy):.1f} px, "
              f"angle={np.mean(errors_deg):.2f} deg")
        print(f"最大误差:     dx={np.max(errors_dx):.1f} px, "
              f"dy={np.max(errors_dy):.1f} px, "
              f"angle={np.max(errors_deg):.2f} deg")
    print()


# ═══════════════════════════════════════════════════════════════
# 测试 2: 同一 site 的两帧真实图像
# ═══════════════════════════════════════════════════════════════
def test_real_pair(extractor, predictor):
    """找一个 site 下的两帧图像，测试模型在真实数据上的表现"""
    print("=" * 60)
    print("测试 2: 真实图像对 (如有两帧以上)")
    print("=" * 60)

    # 按 sample_id 分组
    sites_by_sample = {}
    for site_dir in sorted(SITE_MEMORY_ROOT.rglob("metadata.json")):
        try:
            memory = load_site_memory(site_dir.parent)
            tpl = memory.get("reference_template")
            if tpl is not None:
                gray = to_grayscale_u8(tpl)
                if gray is not None and gray.size > 0:
                    sid = memory.get("sample_id", "unknown")
                    if sid not in sites_by_sample:
                        sites_by_sample[sid] = []
                    sites_by_sample[sid].append((site_dir.parent.name, gray))
        except Exception:
            continue

    found_pair = False
    for sid, frames in sites_by_sample.items():
        if len(frames) < 2:
            continue
        found_pair = True
        name_a, img_a = frames[0]
        name_b, img_b = frames[1]
        print(f"\nSample: {sid}")
        print(f"  帧 A: {name_a}")
        print(f"  帧 B: {name_b}")

        if img_a.shape != img_b.shape:
            # resize 到相同尺寸
            print(f"  尺寸不同: {img_a.shape} vs {img_b.shape} → resize")
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

        result_ab = predictor.predict(img_a, img_b)
        result_ba = predictor.predict(img_b, img_a)

        if result_ab and result_ba:
            print(f"  A→B: dx={result_ab['dx_px']:+.1f} px, "
                  f"dy={result_ab['dy_px']:+.1f} px, "
                  f"angle={result_ab['angle_deg']:+.2f} deg")
            print(f"  B→A: dx={result_ba['dx_px']:+.1f} px, "
                  f"dy={result_ba['dy_px']:+.1f} px, "
                  f"angle={result_ba['angle_deg']:+.2f} deg")

            # 一致性检查: A→B 和 B→A 应互为相反数
            consistency = np.hypot(
                result_ab["dx_px"] + result_ba["dx_px"],
                result_ab["dy_px"] + result_ba["dy_px"],
            )
            angle_consistency = abs(result_ab["angle_deg"] + result_ba["angle_deg"])
            print(f"  一致性: offset={consistency:.1f} px  angle={angle_consistency:.2f} deg"
                  f"{'  [OK]' if consistency < 20 and angle_consistency < 2 else '  [WARN]'}")
        break

    if not found_pair:
        print("  未找到同一 sample 的多帧数据，跳过")


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("5w Remount 预测模型 — 独立验证")
    print("=" * 60)

    # 加载模型
    print(f"\n模型路径: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"[FAIL] 模型文件不存在: {MODEL_PATH}")
        sys.exit(1)

    predictor = load_deep_remount_predictor_5w(MODEL_PATH)
    if predictor is None:
        print("[FAIL] 模型加载失败")
        sys.exit(1)
    print("[OK] 模型加载成功")

    extractor = predictor.extractor
    print(f"   设备: {extractor.device}")
    print(f"   特征维度: {FEATURE_DIM}")
    if predictor.scale_factors:
        sf = predictor.scale_factors
        print(f"   归一化: dx_mean={sf.get('dx_mean',0):.1f} dx_std={sf.get('dx_std',0):.1f}")
        print(f"           dy_mean={sf.get('dy_mean',0):.1f} dy_std={sf.get('dy_std',0):.1f}")
        print(f"           angle_mean={sf.get('angle_mean',0):.2f} angle_std={sf.get('angle_std',0):.2f}")

    # 运行测试
    t0 = time.time()
    test_synthetic(extractor, predictor)
    test_real_pair(extractor, predictor)
    print(f"总耗时: {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
