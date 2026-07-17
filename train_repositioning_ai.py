from pathlib import Path

from afm_phase2_ml import (
    train_lowmag_embedding_index,
    train_remount_transform_predictor,
    train_same_site_classifier,
)


BASE_DIR = Path(__file__).resolve().parent
SITE_MEMORY_ROOT = BASE_DIR / "collected_data" / "site_memories"


def main():
    if not SITE_MEMORY_ROOT.exists():
        raise FileNotFoundError(f"No site memories found at {SITE_MEMORY_ROOT}")

    models_dir = BASE_DIR / "collected_data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training Phase 2 models from {SITE_MEMORY_ROOT}")

    same_site_path = models_dir / "same_site_classifier.pkl"
    remount_path = models_dir / "remount_transform_predictor.pkl"
    lowmag_path = models_dir / "lowmag_embedding_index.pkl"

    try:
        train_same_site_classifier(SITE_MEMORY_ROOT, same_site_path)
        print(f"Saved same-site classifier to {same_site_path}")
    except Exception as exc:
        print(f"Same-site classifier training skipped: {exc}")

    try:
        train_remount_transform_predictor(SITE_MEMORY_ROOT, remount_path)
        print(f"Saved remount transform predictor to {remount_path}")
    except Exception as exc:
        print(f"Remount transform predictor training skipped: {exc}")

    try:
        train_lowmag_embedding_index(SITE_MEMORY_ROOT, lowmag_path)
        print(f"Saved low-mag embedding index to {lowmag_path}")
    except Exception as exc:
        print(f"Low-mag embedding index training skipped: {exc}")


if __name__ == "__main__":
    main()
