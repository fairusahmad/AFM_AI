"""
train_ml_models.py — 一键训练所有 ML 模型（深度特征 + MLP）

训练流程:
  1. 从保存的 site_memories 生成合成训练数据
  2. 训练 deep_same_site_classifier（MLP二分类：判断是否同一site）
  3. 训练 deep_remount_predictor（MLP回归：预测dx, dy, dtheta）
  4. 模型保存到 collected_data/models/

使用方法:
  python training/train_ml_models.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from afm_ml_recognition import (
    DeepFeatureExtractor,
    train_deep_same_site_classifier,
    train_deep_remount_predictor,
)

SITE_MEMORY_ROOT = PROJECT_ROOT / "collected_data" / "site_memories"


def main():
    if not SITE_MEMORY_ROOT.exists():
        print(f"ERROR: No site memories found at {SITE_MEMORY_ROOT}")
        print("Please save at least one reference region first:")
        print("  1. Run: python afm_control_panel.py")
        print("  2. Navigate to a region of interest")
        print("  3. Click 'Save Region'")
        print("  4. Repeat for a few different sites (optional)")
        return

    models_dir = PROJECT_ROOT / "collected_data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training ML models from {SITE_MEMORY_ROOT}")
    print(f"Models will be saved to {models_dir}")
    print()
    print("Initializing ResNet18 feature extractor (downloads weights on first run)...")

    device = "cpu"  # 使用 CPU；如有 CUDA 可改为 "cuda"
    extractor = DeepFeatureExtractor(device=device)
    print(f"Feature extractor ready on {device}")
    print()

    # ── 1. 训练 same-site 分类器 ──
    print("=" * 60)
    print("STEP 1: Training Deep Same-Site Classifier (MLP)")
    print("=" * 60)
    same_site_path = models_dir / "deep_same_site_classifier.pkl"
    try:
        train_deep_same_site_classifier(SITE_MEMORY_ROOT, same_site_path, device=device)
        print(f"  → Saved to {same_site_path}")
    except Exception as exc:
        print(f"  [FAILED]: {exc}")

    print()

    # ── 2. 训练 remount 变换预测器 ──
    print("=" * 60)
    print("STEP 2: Training Deep Remount Predictor (MLP)")
    print("=" * 60)
    remount_path = models_dir / "deep_remount_predictor.pkl"
    try:
        train_deep_remount_predictor(SITE_MEMORY_ROOT, remount_path, device=device)
        print(f"  → Saved to {remount_path}")
    except Exception as exc:
        print(f"  [FAILED]: {exc}")

    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Restart: python afm_control_panel.py")
    print("  2. Save a reference region (1. Save Region)")
    print("  3. Simulate remount (2. Remount)")
    print("  4. Click 'AI Recall' — ML recognition will be used automatically")


if __name__ == "__main__":
    main()
