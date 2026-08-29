import argparse
import csv
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SITE_MEMORY_ROOT = PROJECT_ROOT / "collected_data" / "site_memories"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "collected_data" / "prepared_training"


def iter_site_memory_dirs(site_memory_root):
    root = Path(site_memory_root)
    if not root.exists():
        return []
    return sorted({path.parent for path in root.rglob("metadata.json")})


def load_metadata(site_dir):
    metadata_path = Path(site_dir) / "metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def inspect_site_memory(site_dir):
    site_dir = Path(site_dir)
    metadata = load_metadata(site_dir)
    has_live_camera_view = bool(metadata.get("live_camera_view_path"))
    live_camera_path = site_dir / metadata["live_camera_view_path"] if has_live_camera_view else None
    overview_path = None
    overview = metadata.get("overview") or {}
    if overview.get("image_path"):
        overview_path = site_dir / overview["image_path"]
    reference_path = site_dir / metadata["reference_template_path"] if metadata.get("reference_template_path") else None
    return {
        "site_dir": site_dir,
        "sample_id": metadata.get("sample_id", "unknown"),
        "site_id": metadata.get("site_id", site_dir.name),
        "captured_at": metadata.get("captured_at", ""),
        "has_live_camera_view": has_live_camera_view,
        "live_camera_view_path": None if live_camera_path is None else str(live_camera_path),
        "reference_template_path": None if reference_path is None else str(reference_path),
        "overview_path": None if overview_path is None else str(overview_path),
        "coarse_zoom_level": metadata.get("coarse_zoom_level"),
        "final_zoom_level": metadata.get("final_zoom_level", metadata.get("zoom_level")),
        "source_mode": "camera_fov" if has_live_camera_view else "legacy_reference_only",
    }


def write_manifest(rows, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "site_id",
        "captured_at",
        "source_mode",
        "has_live_camera_view",
        "live_camera_view_path",
        "reference_template_path",
        "overview_path",
        "coarse_zoom_level",
        "final_zoom_level",
        "site_dir",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "site_id": row["site_id"],
                    "captured_at": row["captured_at"],
                    "source_mode": row["source_mode"],
                    "has_live_camera_view": row["has_live_camera_view"],
                    "live_camera_view_path": row["live_camera_view_path"],
                    "reference_template_path": row["reference_template_path"],
                    "overview_path": row["overview_path"],
                    "coarse_zoom_level": row["coarse_zoom_level"],
                    "final_zoom_level": row["final_zoom_level"],
                    "site_dir": str(row["site_dir"]),
                }
            )


def stage_camera_only_site_memories(rows, staged_root):
    staged_root = Path(staged_root)
    if staged_root.exists():
        shutil.rmtree(staged_root)
    staged_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for row in rows:
        if not row["has_live_camera_view"]:
            continue
        sample_dir = staged_root / row["sample_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        destination = sample_dir / row["site_dir"].name
        shutil.copytree(row["site_dir"], destination)
        copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser(description="Prepare camera-FOV-only training dataset manifest and staging area.")
    parser.add_argument("--site-memory-root", default=str(DEFAULT_SITE_MEMORY_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stage-camera-only", action="store_true")
    args = parser.parse_args()

    site_memory_root = Path(args.site_memory_root)
    output_dir = Path(args.output_dir)

    rows = [inspect_site_memory(site_dir) for site_dir in iter_site_memory_dirs(site_memory_root)]
    manifest_path = output_dir / "camera_fov_training_manifest.csv"
    write_manifest(rows, manifest_path)

    total = len(rows)
    strict_camera = sum(1 for row in rows if row["has_live_camera_view"])
    legacy = total - strict_camera

    staged_count = 0
    if args.stage_camera_only:
        staged_count = stage_camera_only_site_memories(rows, output_dir / "site_memories_camera_only")

    print(f"Manifest written to: {manifest_path}")
    print(f"Total site memories: {total}")
    print(f"Camera-FOV-ready site memories: {strict_camera}")
    print(f"Legacy site memories: {legacy}")
    if args.stage_camera_only:
        print(f"Staged camera-only site memories: {staged_count}")
        print(f"Staged root: {output_dir / 'site_memories_camera_only'}")


if __name__ == "__main__":
    main()
