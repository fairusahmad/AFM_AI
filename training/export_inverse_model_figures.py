import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hysteresis import NanoPositioner
from training.train_inverse_model import predict_command


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


def load_hysteresis_data(csv_path=None):
    if csv_path is not None:
        path = resolve_project_path(csv_path)
    else:
        candidates = sorted((PROJECT_ROOT / "collected_data").glob("hysteresis_data_*.csv"))
        if not candidates:
            raise FileNotFoundError("No hysteresis_data_*.csv found in collected_data/")
        path = candidates[-1]
    df = pd.read_csv(path)
    return path, df


def export_hysteresis_characterization(df, output_dir):
    labels = list(df["label"].dropna().unique())
    color_map = plt.cm.get_cmap("tab10", max(len(labels), 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for idx, label in enumerate(labels):
        group = df[df["label"] == label].copy()
        color = color_map(idx)
        axes[0, 0].plot(group["cmd_x"], group["actual_x"], label=label, alpha=0.85, lw=1.5, color=color)
        axes[0, 1].plot(group["cmd_x"], group["error"], label=label, alpha=0.85, lw=1.3, color=color)

        for direction, marker in [("ascending", "o"), ("descending", "x")]:
            sub = group[group["direction"] == direction]
            if not sub.empty:
                axes[1, 0].scatter(
                    sub["cmd_x"],
                    sub["actual_x"],
                    s=10,
                    alpha=0.45,
                    color=color,
                    marker=marker,
                    label=f"{label} {direction}" if idx == 0 else None,
                )

    identity_min = float(min(df["cmd_x"].min(), df["actual_x"].min()))
    identity_max = float(max(df["cmd_x"].max(), df["actual_x"].max()))
    axes[0, 0].plot([identity_min, identity_max], [identity_min, identity_max], "k--", lw=1.2, label="Ideal 1:1")
    axes[0, 0].set_title("Command vs actual displacement")
    axes[0, 0].set_xlabel("Command x (um)")
    axes[0, 0].set_ylabel("Actual x (um)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=2)

    axes[0, 1].axhline(0.0, color="k", linestyle="--", lw=1.0)
    axes[0, 1].set_title("Positioning error vs command")
    axes[0, 1].set_xlabel("Command x (um)")
    axes[0, 1].set_ylabel("Error (um)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Ascending/descending hysteresis loop")
    axes[1, 0].set_xlabel("Command x (um)")
    axes[1, 0].set_ylabel("Actual x (um)")
    axes[1, 0].grid(True, alpha=0.3)

    summary = (
        df.groupby("label")
        .agg(
            samples=("label", "size"),
            cmd_min=("cmd_x", "min"),
            cmd_max=("cmd_x", "max"),
            mae_um=("error", lambda s: float(np.mean(np.abs(s)))),
            max_abs_error_um=("error", lambda s: float(np.max(np.abs(s)))),
        )
        .reset_index()
    )

    axes[1, 1].axis("off")
    axes[1, 1].set_title("Sweep summary")
    table = axes[1, 1].table(
        cellText=np.round(summary[["samples", "cmd_min", "cmd_max", "mae_um", "max_abs_error_um"]].values, 2),
        colLabels=["N", "Cmd min", "Cmd max", "MAE", "Max |err|"],
        rowLabels=summary["label"].tolist(),
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)

    fig.suptitle("Hysteresis and creep characterization", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir / "fig2_hysteresis_characterization")
    plt.close(fig)

    summary_path = output_dir / "fig2_hysteresis_summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary


def export_inverse_model_performance(output_dir, model_path=None):
    if model_path is None:
        model_path = PROJECT_ROOT / "inverse_model.pkl"
    else:
        model_path = resolve_project_path(model_path)
    if not model_path.exists():
        return None

    bundle = joblib.load(model_path)

    stage = NanoPositioner()
    stage.reset(0, 0)

    desired_positions = np.linspace(200, 1400, 50)
    actual_no_comp = []
    for cmd in desired_positions:
        actual, _ = stage.move_to(cmd, 0)
        actual_no_comp.append(actual)

    stage.reset(0, 0)
    actual_with_comp = []
    cmd_with_comp = []
    desired_history = [0.0, 0.0]
    for desired in desired_positions:
        cmd_pred = predict_command(
            bundle,
            desired=float(desired),
            prev_1=float(desired_history[-1]),
            prev_2=float(desired_history[-2]),
        )
        cmd_with_comp.append(cmd_pred)
        desired_history.append(float(desired))
        desired_history = desired_history[-2:]
        actual, _ = stage.move_to(cmd_pred, 0)
        actual_with_comp.append(actual)

    actual_no_comp = np.asarray(actual_no_comp, dtype=float)
    actual_with_comp = np.asarray(actual_with_comp, dtype=float)
    cmd_with_comp = np.asarray(cmd_with_comp, dtype=float)

    error_no_comp = actual_no_comp - desired_positions
    error_with_comp = actual_with_comp - desired_positions
    rmse_no_comp = float(np.sqrt(np.mean(error_no_comp**2)))
    rmse_with_comp = float(np.sqrt(np.mean(error_with_comp**2)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].plot(desired_positions, desired_positions, "k--", lw=1.2, label="Ideal 1:1")
    axes[0].plot(desired_positions, actual_no_comp, lw=2, label="Without compensation")
    axes[0].plot(desired_positions, actual_with_comp, lw=2, label="With inverse model")
    axes[0].set_title("Tracking improvement")
    axes[0].set_xlabel("Desired position (um)")
    axes[0].set_ylabel("Actual position (um)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(desired_positions, error_no_comp, lw=2, label="Without compensation")
    axes[1].plot(desired_positions, error_with_comp, lw=2, label="With inverse model")
    axes[1].axhline(0.0, color="k", linestyle="--", lw=1.0)
    axes[1].set_title("Residual error")
    axes[1].set_xlabel("Desired position (um)")
    axes[1].set_ylabel("Error (um)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].bar(["No comp", "Inverse model"], [rmse_no_comp, rmse_with_comp], color=["#4c72b0", "#dd8452"])
    axes[2].set_title("RMSE comparison")
    axes[2].set_ylabel("RMSE (um)")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Inverse-model compensation performance", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir / "fig2_inverse_model_compensation")
    plt.close(fig)

    summary = {
        "rmse_no_comp_um": rmse_no_comp,
        "rmse_with_comp_um": rmse_with_comp,
        "rmse_reduction_percent": float((1.0 - rmse_with_comp / rmse_no_comp) * 100.0) if rmse_no_comp else 0.0,
        "mean_abs_error_no_comp_um": float(np.mean(np.abs(error_no_comp))),
        "mean_abs_error_with_comp_um": float(np.mean(np.abs(error_with_comp))),
    }
    (output_dir / "fig2_inverse_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Export manuscript-ready hysteresis and inverse-model figures")
    parser.add_argument("--hysteresis-csv", default=None)
    parser.add_argument("--inverse-model", default=None)
    parser.add_argument("--output-dir", default="manuscript/generated_figures/inverse_model")
    args = parser.parse_args()

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path, df = load_hysteresis_data(args.hysteresis_csv)
    summary = export_hysteresis_characterization(df, output_dir)
    inverse_summary = export_inverse_model_performance(output_dir, args.inverse_model)

    manifest = {
        "hysteresis_csv": str(csv_path),
        "output_dir": str(output_dir),
        "hysteresis_summary_rows": int(len(summary)),
        "inverse_model_exported": inverse_summary is not None,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
