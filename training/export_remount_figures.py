import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from afm_phase2_ml import _load_site_memories
from afm_relocation import load_site_memory, to_grayscale_u8
from training.train_remount_real import collect_real_pairs, compute_ground_truth


def resolve_project_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return PROJECT_ROOT / path


def save_figure(fig, output_base, dpi=300):
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")


def build_site_memory_summary(site_memory_root):
    rows = []
    unique_anchor_hashes = set()
    first_site_dir = None
    first_memory = None
    for site_dir, memory in _load_site_memories(site_memory_root):
        if first_site_dir is None:
            first_site_dir = site_dir
            first_memory = memory
        ref = memory.get("reference_template")
        ref_gray = to_grayscale_u8(ref) if ref is not None else None
        if ref_gray is not None and ref_gray.size > 0:
            unique_anchor_hashes.add(hashlib.md5(ref_gray.tobytes()).hexdigest())
            ref_shape = ref_gray.shape
        else:
            ref_shape = (0, 0)
        rows.append(
            {
                "sample_id": memory.get("sample_id", "unknown"),
                "site_id": memory.get("site_id", site_dir.name),
                "lowmag_landmark_count": len(memory.get("lowmag_landmarks") or []),
                "highmag_landmark_count": len(memory.get("highmag_landmarks") or []),
                "reference_width_px": int(ref_shape[1]),
                "reference_height_px": int(ref_shape[0]),
                "zoom_level": float(memory.get("zoom_level", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows), len(unique_anchor_hashes), first_site_dir, first_memory


def export_site_memory_composite(first_site_dir, first_memory, output_dir):
    if first_site_dir is None or first_memory is None:
        return

    overview_path = first_site_dir / "lowmag_overview.png"
    reference_path = first_site_dir / "reference_template.png"
    overview = cv2.imread(str(overview_path), cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.0, 1.0], height_ratios=[1.0, 1.0])

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(overview, cmap="gray")
    ax0.set_title("Low-magnification overview")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(reference, cmap="gray")
    ax1.set_title("Reference template")
    ax1.axis("off")

    lowmag_landmarks = first_memory.get("lowmag_landmarks") or []
    highmag_landmarks = first_memory.get("highmag_landmarks") or []

    def make_patch_mosaic(landmarks, title, ax):
        tiles = []
        for item in landmarks[:6]:
            patch = item.get("patch")
            if patch is None:
                continue
            patch = to_grayscale_u8(patch)
            if patch is not None and patch.size > 0:
                tiles.append(patch)
        if not tiles:
            ax.text(0.5, 0.5, "No patches", ha="center", va="center")
            ax.axis("off")
            ax.set_title(title)
            return
        max_h = max(tile.shape[0] for tile in tiles)
        max_w = max(tile.shape[1] for tile in tiles)
        padded = []
        for tile in tiles:
            canvas = np.zeros((max_h, max_w), dtype=np.uint8)
            canvas[: tile.shape[0], : tile.shape[1]] = tile
            padded.append(canvas)
        while len(padded) < 6:
            padded.append(np.zeros((max_h, max_w), dtype=np.uint8))
        top = np.hstack(padded[:3])
        bottom = np.hstack(padded[3:6])
        mosaic = np.vstack([top, bottom])
        ax.imshow(mosaic, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    ax2 = fig.add_subplot(gs[1, 1])
    make_patch_mosaic(lowmag_landmarks, "Low-mag landmarks", ax2)

    ax3 = fig.add_subplot(gs[:, 2])
    make_patch_mosaic(highmag_landmarks, "High-mag landmarks", ax3)

    fig.suptitle("Structured site memory example", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir / "fig3_site_memory_composite")
    plt.close(fig)


def export_dataset_summary(df, unique_anchor_count, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    sample_counts = df["sample_id"].value_counts().sort_values(ascending=False)
    axes[0, 0].bar(range(len(sample_counts)), sample_counts.values, color="#4c72b0")
    axes[0, 0].set_title("Saved site memories by sample")
    axes[0, 0].set_xlabel("Sample index")
    axes[0, 0].set_ylabel("Memory count")
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    axes[0, 1].hist(df["reference_width_px"], bins=min(10, max(len(df), 3)), alpha=0.8, color="#55a868")
    axes[0, 1].set_title("Reference-template width distribution")
    axes[0, 1].set_xlabel("Width (px)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].grid(True, alpha=0.3)

    x = np.arange(len(df))
    axes[1, 0].bar(x - 0.18, df["lowmag_landmark_count"], width=0.36, label="Low-mag")
    axes[1, 0].bar(x + 0.18, df["highmag_landmark_count"], width=0.36, label="High-mag")
    axes[1, 0].set_title("Landmark counts per site memory")
    axes[1, 0].set_xlabel("Site memory index")
    axes[1, 0].set_ylabel("Landmark count")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    axes[1, 1].axis("off")
    summary = pd.DataFrame(
        [
            ["Saved site memories", len(df)],
            ["Unique sample IDs", df["sample_id"].nunique()],
            ["Unique reference anchors", unique_anchor_count],
            ["Mean low-mag landmarks", round(float(df["lowmag_landmark_count"].mean()), 2)],
            ["Mean high-mag landmarks", round(float(df["highmag_landmark_count"].mean()), 2)],
        ],
        columns=["Metric", "Value"],
    )
    table = axes[1, 1].table(cellText=summary.values, colLabels=summary.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    axes[1, 1].set_title("Dataset summary")

    fig.suptitle("Site-memory dataset summary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir / "fig3_site_memory_dataset_summary")
    plt.close(fig)

    df.to_csv(output_dir / "fig3_site_memory_dataset_summary.csv", index=False)


def export_real_pair_summary(output_dir):
    rows = []
    for sample_id, fa, fb in collect_real_pairs():
        gt = compute_ground_truth(
            fa["image"],
            fb["image"],
            fa["fov_w"],
            fa["fov_h"],
            fb["fov_w"],
            fb["fov_h"],
        )
        if gt is None:
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "pair_a": fa["name"],
                "pair_b": fb["name"],
                "dx_px": gt["dx_px"],
                "dy_px": gt["dy_px"],
                "angle_deg": gt["angle_deg"],
                "confidence": gt["confidence"],
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].scatter(df["dx_px"], df["dy_px"], c=df["confidence"], cmap="viridis", s=35)
    axes[0].set_title("Real-pair translation distribution")
    axes[0].set_xlabel("dx (px)")
    axes[0].set_ylabel("dy (px)")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(df["angle_deg"], bins=min(12, max(len(df), 4)), color="#c44e52", alpha=0.85)
    axes[1].set_title("Rotation-angle distribution")
    axes[1].set_xlabel("Angle (deg)")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)

    pair_counts = df["sample_id"].value_counts().sort_values(ascending=False)
    axes[2].bar(range(len(pair_counts)), pair_counts.values, color="#8172b3")
    axes[2].set_title("Valid real pairs by sample")
    axes[2].set_xlabel("Sample index")
    axes[2].set_ylabel("Pair count")
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Real-pair remount-training summary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir / "fig4_real_pair_training_summary")
    plt.close(fig)

    df.to_csv(output_dir / "fig4_real_pair_training_summary.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Export manuscript-ready remount/site-memory figures")
    parser.add_argument("--site-memory-root", default="collected_data/site_memories")
    parser.add_argument("--output-dir", default="manuscript/generated_figures/remount")
    args = parser.parse_args()

    site_memory_root = resolve_project_path(args.site_memory_root)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, unique_anchor_count, first_site_dir, first_memory = build_site_memory_summary(site_memory_root)
    export_site_memory_composite(first_site_dir, first_memory, output_dir)
    export_dataset_summary(df, unique_anchor_count, output_dir)
    real_pair_df = export_real_pair_summary(output_dir)

    manifest = {
        "site_memory_root": str(site_memory_root),
        "output_dir": str(output_dir),
        "site_memory_count": int(len(df)),
        "unique_sample_ids": int(df["sample_id"].nunique()) if not df.empty else 0,
        "unique_reference_anchors": int(unique_anchor_count),
        "valid_real_pairs": int(len(real_pair_df)) if real_pair_df is not None else 0,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
