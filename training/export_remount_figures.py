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

from afm_phase2_ml import _load_site_memories, preferred_training_view
from afm_relocation import load_site_memory, to_grayscale_u8
from training.train_remount_real import collect_real_pairs, compute_ground_truth


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)


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


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)


def annotate_bar_values(ax, bars, fmt="{:.0f}"):
    ymax = 0.0
    for bar in bars:
        ymax = max(ymax, float(bar.get_height()))
    for bar in bars:
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() * 0.5,
            value + max(ymax * 0.02, 0.1),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def build_site_memory_summary(site_memory_root):
    rows = []
    unique_anchor_hashes = set()
    first_site_dir = None
    first_memory = None
    for site_dir, memory in _load_site_memories(site_memory_root):
        if first_site_dir is None:
            first_site_dir = site_dir
            first_memory = memory
        ref, _ = preferred_training_view(memory)
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
    reference_path = first_site_dir / "live_camera_view.png"
    if not reference_path.exists():
        reference_path = first_site_dir / "reference_template.png"
    overview = cv2.imread(str(overview_path), cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.0, 1.0], height_ratios=[1.0, 1.0])

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(overview, cmap="gray")
    ax0.set_title("A. Low-magnification overview")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(reference, cmap="gray")
    ax1.set_title("B. Saved live camera frame")
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
    make_patch_mosaic(lowmag_landmarks, "C. Low-mag landmark patches", ax2)

    ax3 = fig.add_subplot(gs[:, 2])
    make_patch_mosaic(highmag_landmarks, "D. High-mag landmark patches", ax3)

    fig.suptitle("Structured camera-POV site memory used for relocation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir / "fig3_site_memory_composite")
    plt.close(fig)


def export_dataset_summary(df, unique_anchor_count, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    sample_counts = df["sample_id"].value_counts().sort_values(ascending=False)
    top_counts = sample_counts.head(8)
    labels = [f"Sample {idx + 1}" for idx in range(len(top_counts))]
    bars = axes[0, 0].bar(labels, top_counts.values, color="#2f6db2")
    annotate_bar_values(axes[0, 0], bars)
    axes[0, 0].set_title("A. Site memories available per sample")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Number of saved sites")
    axes[0, 0].tick_params(axis="x", rotation=25)
    style_axis(axes[0, 0])

    width_counts = df["reference_width_px"].value_counts().sort_index()
    bars = axes[0, 1].bar(
        [f"{int(width)} px" for width in width_counts.index],
        width_counts.values,
        color="#2b9c78",
    )
    annotate_bar_values(axes[0, 1], bars)
    axes[0, 1].set_title("B. Saved camera-frame resolution")
    axes[0, 1].set_xlabel("Reference width")
    axes[0, 1].set_ylabel("Number of site memories")
    style_axis(axes[0, 1])

    landmark_summary = pd.DataFrame(
        {
            "View": ["Low magnification", "High magnification"],
            "Mean count": [
                float(df["lowmag_landmark_count"].mean()),
                float(df["highmag_landmark_count"].mean()),
            ],
        }
    )
    bars = axes[1, 0].bar(
        landmark_summary["View"],
        landmark_summary["Mean count"],
        color=["#f28e2b", "#7b61b3"],
    )
    annotate_bar_values(axes[1, 0], bars, fmt="{:.1f}")
    axes[1, 0].set_title("C. Mean landmark count per saved site")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Average landmark count")
    style_axis(axes[1, 0])

    axes[1, 1].axis("off")
    summary = pd.DataFrame(
        [
            ["Saved site memories", len(df)],
            ["Samples represented", df["sample_id"].nunique()],
            ["Unique camera anchors", unique_anchor_count],
            ["Mean low-mag landmarks", round(float(df["lowmag_landmark_count"].mean()), 2)],
            ["Mean high-mag landmarks", round(float(df["highmag_landmark_count"].mean()), 2)],
        ],
        columns=["Metric", "Value"],
    )
    table = axes[1, 1].table(cellText=summary.values, colLabels=summary.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    axes[1, 1].set_title("D. Camera-POV dataset summary")

    fig.suptitle("Camera-POV relocation dataset summary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir / "fig3_site_memory_dataset_summary")
    plt.close(fig)

    df.to_csv(output_dir / "fig3_site_memory_dataset_summary.csv", index=False)


def export_real_pair_summary(site_memory_root, output_dir):
    rows = []
    for sample_id, fa, fb in collect_real_pairs(site_memory_root=site_memory_root):
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
    scatter = axes[0].scatter(df["dx_px"], df["dy_px"], c=df["confidence"], cmap="viridis", s=45)
    axes[0].axhline(0.0, color="0.5", linewidth=0.8)
    axes[0].axvline(0.0, color="0.5", linewidth=0.8)
    axes[0].set_title("A. Translation labels from real saved pairs")
    axes[0].set_xlabel("Horizontal shift, dx (pixels)")
    axes[0].set_ylabel("Vertical shift, dy (pixels)")
    axes[0].grid(True, alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("Match confidence")

    bins = min(12, max(len(df), 4))
    axes[1].hist(df["angle_deg"], bins=bins, color="#c44e52", alpha=0.85)
    mean_angle = float(df["angle_deg"].mean())
    axes[1].axvline(mean_angle, color="black", linestyle="--", linewidth=1.0, label=f"Mean = {mean_angle:.1f}°")
    axes[1].set_title("B. Rotation labels from real saved pairs")
    axes[1].set_xlabel("Relative rotation (degrees)")
    axes[1].set_ylabel("Number of valid pairs")
    axes[1].legend(frameon=False, loc="upper right")
    style_axis(axes[1])

    pair_counts = df["sample_id"].value_counts().sort_values(ascending=False).head(8)
    labels = [f"Sample {idx + 1}" for idx in range(len(pair_counts))]
    bars = axes[2].bar(labels, pair_counts.values, color="#7b61b3")
    annotate_bar_values(axes[2], bars)
    axes[2].set_title("C. Real-pair count per sample")
    axes[2].set_xlabel("Sample")
    axes[2].set_ylabel("Number of valid pairs")
    axes[2].tick_params(axis="x", rotation=25)
    style_axis(axes[2])

    fig.suptitle("Real-pair training labels for remount prediction", fontsize=14, fontweight="bold")
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
    real_pair_df = export_real_pair_summary(site_memory_root, output_dir)

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
