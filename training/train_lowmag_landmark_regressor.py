import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from afm_phase2_ml import PAIR_FEATURE_NAMES, pair_features
from afm_relocation import to_grayscale_u8

import cv2


DEFAULT_DATASET_DIR = PROJECT_ROOT / "collected_data" / "lowmag_landmark_search_training_test"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "collected_data" / "models"


GEOMETRY_FEATURE_NAMES = [
    "support_count",
    "confidence",
    "recovered_dx_um",
    "recovered_dy_um",
    "recovered_distance_um",
]


def _to_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_gray_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return to_grayscale_u8(image)


def _safe_float(value, default=0.0):
    if value in ("", None, "None"):
        return float(default)
    return float(value)


def load_manifest_rows(manifest_path):
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_training_table(dataset_dir, min_support, min_confidence):
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "lowmag_landmark_search_manifest.csv"
    rows = load_manifest_rows(manifest_path)

    features = []
    labels = []
    kept_rows = []
    for row in rows:
        if not _to_bool(row.get("usable_landmark_match")):
            continue
        support_count = int(_safe_float(row.get("support_count"), 0))
        confidence = float(_safe_float(row.get("confidence"), 0.0))
        if support_count < int(min_support) or confidence < float(min_confidence):
            continue

        recovered_dx_um = float(_safe_float(row.get("recovered_dx_um"), 0.0))
        recovered_dy_um = float(_safe_float(row.get("recovered_dy_um"), 0.0))
        label_dx_um = float(_safe_float(row.get("label_dx_um"), 0.0))
        label_dy_um = float(_safe_float(row.get("label_dy_um"), 0.0))
        reference_path = dataset_dir / "images" / Path(row["reference_overview_path"])
        current_path = dataset_dir / "images" / Path(row["current_overview_path"])
        reference_image = _load_gray_image(reference_path)
        current_image = _load_gray_image(current_path)
        if reference_image is None or current_image is None:
            continue

        geom_features = np.array(
            [
                float(support_count),
                float(confidence),
                float(recovered_dx_um),
                float(recovered_dy_um),
                float(np.hypot(recovered_dx_um, recovered_dy_um)),
            ],
            dtype=np.float32,
        )
        img_features = pair_features(reference_image, current_image).astype(np.float32)
        combined = np.concatenate([geom_features, img_features]).astype(np.float32)
        target = np.array(
            [
                float(label_dx_um - recovered_dx_um),
                float(label_dy_um - recovered_dy_um),
            ],
            dtype=np.float32,
        )

        features.append(combined)
        labels.append(target)
        kept_rows.append(dict(row))

    if not features:
        raise ValueError("No usable rows passed the low-mag regressor filters.")
    return (
        np.vstack(features),
        np.vstack(labels),
        kept_rows,
        manifest_path,
    )


def train_regressor(X, y, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=int(random_state),
    )
    model = RandomForestRegressor(
        n_estimators=320,
        max_depth=14,
        min_samples_leaf=2,
        random_state=int(random_state),
    )
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    corrected_test = pred_test + X_test[:, 2:4]
    baseline_test = X_test[:, 2:4]
    label_test = y_test + X_test[:, 2:4]

    metrics = {
        "test_rows": int(X_test.shape[0]),
        "baseline_mae_dx_um": float(mean_absolute_error(label_test[:, 0], baseline_test[:, 0])),
        "baseline_mae_dy_um": float(mean_absolute_error(label_test[:, 1], baseline_test[:, 1])),
        "corrected_mae_dx_um": float(mean_absolute_error(label_test[:, 0], corrected_test[:, 0])),
        "corrected_mae_dy_um": float(mean_absolute_error(label_test[:, 1], corrected_test[:, 1])),
        "correction_target_mae_dx_um": float(mean_absolute_error(y_test[:, 0], pred_test[:, 0])),
        "correction_target_mae_dy_um": float(mean_absolute_error(y_test[:, 1], pred_test[:, 1])),
        "r2_dx": float(r2_score(y_test[:, 0], pred_test[:, 0])),
        "r2_dy": float(r2_score(y_test[:, 1], pred_test[:, 1])),
    }
    return model, metrics


def write_filtered_manifest(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Train a low-mag landmark correction regressor on filtered landmark-search data."
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.38)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    X, y, kept_rows, manifest_path = build_training_table(
        dataset_dir,
        args.min_support,
        args.min_confidence,
    )
    model, metrics = train_regressor(X, y, args.random_state)

    feature_names = GEOMETRY_FEATURE_NAMES + list(PAIR_FEATURE_NAMES)
    bundle = {
        "model_type": "lowmag_landmark_regressor",
        "feature_names": feature_names,
        "target_names": ["delta_dx_um", "delta_dy_um"],
        "baseline_feature_slice": [2, 4],
        "min_support": int(args.min_support),
        "min_confidence": float(args.min_confidence),
        "source_manifest": str(manifest_path),
        "model": model,
        "metrics": metrics,
    }

    model_path = models_dir / "lowmag_landmark_regressor.pkl"
    joblib.dump(bundle, model_path)

    filtered_manifest_path = dataset_dir / "lowmag_landmark_search_manifest.filtered.csv"
    write_filtered_manifest(kept_rows, filtered_manifest_path)

    summary = {
        "dataset_dir": str(dataset_dir),
        "source_manifest": str(manifest_path),
        "filtered_manifest": str(filtered_manifest_path),
        "kept_rows": int(len(kept_rows)),
        "feature_count": int(X.shape[1]),
        "metrics": metrics,
    }
    summary_path = dataset_dir / "lowmag_landmark_regressor_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Filtered rows: {len(kept_rows)}")
    print(f"Saved regressor: {model_path}")
    print(f"Filtered manifest: {filtered_manifest_path}")
    print(f"Summary JSON: {summary_path}")
    print(
        "Baseline MAE (um): "
        f"dx={metrics['baseline_mae_dx_um']:.3f}, dy={metrics['baseline_mae_dy_um']:.3f}"
    )
    print(
        "Corrected MAE (um): "
        f"dx={metrics['corrected_mae_dx_um']:.3f}, dy={metrics['corrected_mae_dy_um']:.3f}"
    )


if __name__ == "__main__":
    main()
