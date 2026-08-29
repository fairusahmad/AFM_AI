import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from afm_phase2_ml import (
    train_lowmag_embedding_index,
    train_remount_transform_predictor,
    train_same_site_classifier,
)


SITE_MEMORY_ROOT = PROJECT_ROOT / "collected_data" / "site_memories"


def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 relocation models from site memories.")
    parser.add_argument(
        "--site-memory-root",
        default=str(SITE_MEMORY_ROOT),
        help="Root directory containing saved site memories.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(PROJECT_ROOT / "collected_data" / "models"),
        help="Output directory for trained model bundles.",
    )
    args = parser.parse_args()

    site_memory_root = Path(args.site_memory_root)
    models_dir = Path(args.models_dir)

    if not site_memory_root.exists():
        raise FileNotFoundError(f"No site memories found at {site_memory_root}")

    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training Phase 2 models from {site_memory_root}")

    same_site_path = models_dir / "same_site_classifier.pkl"
    remount_path = models_dir / "remount_transform_predictor.pkl"
    lowmag_path = models_dir / "lowmag_embedding_index.pkl"

    try:
        train_same_site_classifier(site_memory_root, same_site_path)
        print(f"Saved same-site classifier to {same_site_path}")
    except Exception as exc:
        print(f"Same-site classifier training skipped: {exc}")

    try:
        train_remount_transform_predictor(site_memory_root, remount_path)
        print(f"Saved remount transform predictor to {remount_path}")
    except Exception as exc:
        print(f"Remount transform predictor training skipped: {exc}")

    try:
        train_lowmag_embedding_index(site_memory_root, lowmag_path)
        print(f"Saved low-mag embedding index to {lowmag_path}")
    except Exception as exc:
        print(f"Low-mag embedding index training skipped: {exc}")


if __name__ == "__main__":
    main()
