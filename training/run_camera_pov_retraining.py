import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.prepare_camera_fov_dataset import (
    inspect_site_memory,
    iter_site_memory_dirs,
    stage_camera_only_site_memories,
    write_manifest,
)
from training.normalize_site_memory_metadata_paths import normalize_site_memory_root
from training.train_remount_real import train_on_real_pairs
from training.train_repositioning_ai import (
    train_lowmag_embedding_index,
    train_remount_transform_predictor,
    train_same_site_classifier,
)


def prepare_camera_only_root(source_root, prepared_root):
    source_root = Path(source_root)
    prepared_root = Path(prepared_root)
    rows = [inspect_site_memory(site_dir) for site_dir in iter_site_memory_dirs(source_root)]
    prepared_root.mkdir(parents=True, exist_ok=True)
    manifest_path = prepared_root / "camera_fov_training_manifest.csv"
    write_manifest(rows, manifest_path)
    staged_root = prepared_root / "site_memories_camera_only"
    staged_count = stage_camera_only_site_memories(rows, staged_root)
    total = len(rows)
    camera_ready = sum(1 for row in rows if row["has_live_camera_view"])
    return {
        "manifest_path": manifest_path,
        "staged_root": staged_root,
        "total": total,
        "camera_ready": camera_ready,
        "legacy": total - camera_ready,
        "staged_count": staged_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare and retrain camera-POV relocation models.")
    parser.add_argument(
        "--source-root",
        default=str(PROJECT_ROOT / "collected_data" / "site_memories"),
        help="Original site memory root.",
    )
    parser.add_argument(
        "--prepared-root",
        default=str(PROJECT_ROOT / "collected_data" / "prepared_training"),
        help="Prepared dataset output root.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(PROJECT_ROOT / "collected_data" / "models"),
        help="Model output directory.",
    )
    parser.add_argument("--device", default="cpu", help="Training device for deep feature extraction.")
    parser.add_argument(
        "--skip-real-remount",
        action="store_true",
        help="Skip retraining deep_remount_predictor_real.pkl.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    prepared_root = Path(args.prepared_root)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    normalized_count = normalize_site_memory_root(source_root)
    print(f"Normalized metadata files before staging: {normalized_count}")

    summary = prepare_camera_only_root(source_root, prepared_root)
    print(f"Manifest written to: {summary['manifest_path']}")
    print(f"Total site memories: {summary['total']}")
    print(f"Camera-FOV-ready site memories: {summary['camera_ready']}")
    print(f"Legacy site memories: {summary['legacy']}")
    print(f"Staged camera-only site memories: {summary['staged_count']}")
    print(f"Staged root: {summary['staged_root']}")

    staged_root = summary["staged_root"]
    if not staged_root.exists():
        raise FileNotFoundError(f"Staged camera-only root missing: {staged_root}")
    staged_normalized_count = normalize_site_memory_root(staged_root)
    print(f"Normalized metadata files after staging: {staged_normalized_count}")

    print("\nTraining Phase 2 camera-only models...")
    train_same_site_classifier(staged_root, models_dir / "same_site_classifier.pkl")
    print(f"Saved same-site classifier to {models_dir / 'same_site_classifier.pkl'}")
    train_remount_transform_predictor(staged_root, models_dir / "remount_transform_predictor.pkl")
    print(f"Saved remount transform predictor to {models_dir / 'remount_transform_predictor.pkl'}")
    train_lowmag_embedding_index(staged_root, models_dir / "lowmag_embedding_index.pkl")
    print(f"Saved low-mag embedding index to {models_dir / 'lowmag_embedding_index.pkl'}")

    if not args.skip_real_remount:
        print("\nTraining camera-only real-pair remount model...")
        train_on_real_pairs(
            device=args.device,
            site_memory_root=staged_root,
            output_path=models_dir / "deep_remount_predictor_real.pkl",
        )


if __name__ == "__main__":
    main()
