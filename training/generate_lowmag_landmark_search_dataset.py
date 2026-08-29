import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from afm_relocation import (
    apply_affine,
    estimate_landmark_consensus,
    load_site_memory,
    rotation_translation_affine,
    to_grayscale_u8,
)


DEFAULT_SITE_MEMORY_ROOT = PROJECT_ROOT / "collected_data" / "site_memories"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "collected_data" / "lowmag_landmark_search_training"


def iter_site_memory_dirs(site_memory_root):
    root = Path(site_memory_root)
    if not root.exists():
        return []
    return sorted({path.parent for path in root.rglob("metadata.json")})


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    gray = to_grayscale_u8(image)
    if gray is None or gray.size == 0:
        return None
    cv2.imwrite(str(path), gray)
    return path


def simulate_lowmag_variant(reference_image, *, shift_x_px, shift_y_px, rotation_deg):
    height, width = reference_image.shape[:2]
    matrix = rotation_translation_affine(
        width,
        height,
        angle_deg=float(rotation_deg),
        shift_x_px=float(shift_x_px),
        shift_y_px=float(shift_y_px),
    )
    current_image = apply_affine(reference_image, matrix, output_shape=reference_image.shape[:2])
    return to_grayscale_u8(current_image)


def evaluate_landmark_search(site_memory, current_image):
    overview = site_memory.get("overview") or {}
    lowmag_landmarks = site_memory.get("lowmag_landmarks") or []
    if not lowmag_landmarks or overview.get("image") is None:
        return None
    return estimate_landmark_consensus(
        lowmag_landmarks,
        current_image,
        search_origin_x_um=float(overview.get("top_left_x_um", 0.0)),
        search_origin_y_um=float(overview.get("top_left_y_um", 0.0)),
        scale_x_um_per_px=float(overview["scale_x_um_per_px"]),
        scale_y_um_per_px=float(overview["scale_y_um_per_px"]),
        min_score=0.30,
        min_gap=0.01,
        max_residual_um=120.0,
    )


def build_dataset(site_memory_root, output_dir, augmentations_per_site, seed):
    rng = random.Random(seed)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    evaluation_errors_x = []
    evaluation_errors_y = []
    usable_matches = 0
    total_pairs = 0

    for site_dir in iter_site_memory_dirs(site_memory_root):
        site_memory = load_site_memory(site_dir)
        overview = site_memory.get("overview") or {}
        reference_image = to_grayscale_u8(overview.get("image"))
        lowmag_landmarks = site_memory.get("lowmag_landmarks") or []
        if reference_image is None or reference_image.size == 0 or not lowmag_landmarks:
            continue

        sample_id = str(site_memory.get("sample_id", site_dir.parent.name))
        site_id = str(site_memory.get("site_id", site_dir.name))
        scale_x = float(overview["scale_x_um_per_px"])
        scale_y = float(overview["scale_y_um_per_px"])

        reference_rel_path = Path(sample_id) / site_id / "reference_overview.png"
        write_image(images_dir / reference_rel_path, reference_image)

        for augmentation_index in range(max(int(augmentations_per_site), 1)):
            shift_x_px = rng.uniform(-60.0, 60.0)
            shift_y_px = rng.uniform(-60.0, 60.0)
            rotation_deg = rng.uniform(-8.0, 8.0)
            current_image = simulate_lowmag_variant(
                reference_image,
                shift_x_px=shift_x_px,
                shift_y_px=shift_y_px,
                rotation_deg=rotation_deg,
            )
            current_rel_path = Path(sample_id) / site_id / f"current_{augmentation_index:04d}.png"
            write_image(images_dir / current_rel_path, current_image)

            consensus = evaluate_landmark_search(site_memory, current_image)
            total_pairs += 1

            recovered_dx_um = None
            recovered_dy_um = None
            support_count = 0
            confidence = 0.0
            if consensus is not None:
                recovered_dx_um = float(consensus.get("offset_x_um", 0.0))
                recovered_dy_um = float(consensus.get("offset_y_um", 0.0))
                support_count = int(consensus.get("support_count", 0))
                confidence = float(consensus.get("confidence", 0.0))
                evaluation_errors_x.append(abs(recovered_dx_um - shift_x_px * scale_x))
                evaluation_errors_y.append(abs(recovered_dy_um - shift_y_px * scale_y))
                usable_matches += 1

            rows.append(
                {
                    "sample_id": sample_id,
                    "site_id": site_id,
                    "reference_overview_path": str(reference_rel_path).replace("\\", "/"),
                    "current_overview_path": str(current_rel_path).replace("\\", "/"),
                    "label_dx_um": float(shift_x_px * scale_x),
                    "label_dy_um": float(shift_y_px * scale_y),
                    "label_rotation_deg": float(rotation_deg),
                    "scale_x_um_per_px": scale_x,
                    "scale_y_um_per_px": scale_y,
                    "support_count": support_count,
                    "confidence": confidence,
                    "recovered_dx_um": recovered_dx_um,
                    "recovered_dy_um": recovered_dy_um,
                    "usable_landmark_match": bool(consensus is not None),
                    "site_memory_dir": str(site_dir),
                }
            )

    manifest_path = output_dir / "lowmag_landmark_search_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else [
            "sample_id",
            "site_id",
            "reference_overview_path",
            "current_overview_path",
            "label_dx_um",
            "label_dy_um",
            "label_rotation_deg",
            "scale_x_um_per_px",
            "scale_y_um_per_px",
            "support_count",
            "confidence",
            "recovered_dx_um",
            "recovered_dy_um",
            "usable_landmark_match",
            "site_memory_dir",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "site_memory_root": str(site_memory_root),
        "output_dir": str(output_dir),
        "total_pairs": int(total_pairs),
        "usable_landmark_matches": int(usable_matches),
        "mean_abs_error_dx_um": None if not evaluation_errors_x else float(np.mean(evaluation_errors_x)),
        "mean_abs_error_dy_um": None if not evaluation_errors_y else float(np.mean(evaluation_errors_y)),
        "seed": int(seed),
        "augmentations_per_site": int(augmentations_per_site),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return manifest_path, summary_path, summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate low-magnification landmark-search training pairs and evaluate landmark recovery."
    )
    parser.add_argument("--site-memory-root", default=str(DEFAULT_SITE_MEMORY_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--augmentations-per-site", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path, summary_path, summary = build_dataset(
        Path(args.site_memory_root),
        Path(args.output_dir),
        args.augmentations_per_site,
        args.seed,
    )
    print(f"Low-mag training manifest: {manifest_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Total pairs: {summary['total_pairs']}")
    print(f"Usable landmark matches: {summary['usable_landmark_matches']}")
    print(f"Mean |dx error| (um): {summary['mean_abs_error_dx_um']}")
    print(f"Mean |dy error| (um): {summary['mean_abs_error_dy_um']}")


if __name__ == "__main__":
    main()
