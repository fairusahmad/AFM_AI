"""
preprocess_data.py - Data preprocessing for AI training
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FORWARD_FEATURES = ["cmd_x", "cmd_prev_1", "cmd_prev_2", "direction", "velocity"]
INVERSE_FEATURES = ["actual_x", "actual_prev_1", "actual_prev_2", "direction_actual", "velocity_actual"]
BASE_DIR = Path(__file__).resolve().parent


def resolve_project_path(path_str):
    """Resolve paths relative to the script directory when needed."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return BASE_DIR / path


def load_and_merge_data(data_dir="collected_data"):
    """Load and merge all collected CSV files."""
    data_path = resolve_project_path(data_dir)
    if not data_path.exists():
        print(f"Directory not found: {data_path}")
        return None

    all_files = sorted(path.name for path in data_path.glob("*.csv"))
    if not all_files:
        print(f"No CSV files found in {data_path}")
        return None

    dfs = []
    for file in all_files:
        filepath = data_path / file
        df = pd.read_csv(filepath)
        df["source_file"] = file
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(all_files)} files from {data_path}, total {len(combined)} records")
    return combined


def create_command_features(df):
    """Create forward-model features from commanded motion."""
    df = df.copy()
    df["cmd_prev_1"] = df["cmd_x"].shift(1)
    df["cmd_prev_2"] = df["cmd_x"].shift(2)
    df["cmd_prev_3"] = df["cmd_x"].shift(3)
    df["direction"] = np.sign(df["cmd_x"] - df["cmd_prev_1"]).fillna(0)
    df["velocity"] = (df["cmd_x"] - df["cmd_prev_1"]).fillna(0)
    return df.dropna().reset_index(drop=True)


def create_inverse_features(df):
    """Create inverse-model features from actual motion history."""
    df = df.copy()
    df["actual_prev_1"] = df["actual_x"].shift(1)
    df["actual_prev_2"] = df["actual_x"].shift(2)
    df["direction_actual"] = np.sign(df["actual_x"] - df["actual_prev_1"]).fillna(0)
    df["velocity_actual"] = (df["actual_x"] - df["actual_prev_1"]).fillna(0)
    return df.dropna().reset_index(drop=True)


def prepare_training_data(df, target, feature_cols):
    """Split, scale, and package training data."""
    X = df[feature_cols].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "feature_names": list(feature_cols),
        "target_name": target,
    }


def save_preprocessed_data(data_dict, filename):
    """Save array-only data for quick inspection or compatibility."""
    output_path = resolve_project_path(filename)
    np.savez(
        output_path,
        X_train=data_dict["X_train"],
        X_test=data_dict["X_test"],
        y_train=data_dict["y_train"],
        y_test=data_dict["y_test"],
        feature_names=np.array(data_dict["feature_names"], dtype=object),
        target_name=data_dict["target_name"],
    )
    print(f"Preprocessed data saved to {output_path}")


def save_training_bundle(data_dict, filename):
    """Save model-training data including scalers."""
    output_path = resolve_project_path(filename)
    bundle = {
        "X_train": data_dict["X_train"],
        "X_test": data_dict["X_test"],
        "y_train": data_dict["y_train"],
        "y_test": data_dict["y_test"],
        "scaler_X": data_dict["scaler_X"],
        "scaler_y": data_dict["scaler_y"],
        "feature_names": data_dict["feature_names"],
        "target_name": data_dict["target_name"],
    }
    joblib.dump(bundle, output_path)
    print(f"Training bundle saved to {output_path}")


def plot_data_distribution(df):
    """Visualize data distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(df["cmd_x"], bins=50, alpha=0.7)
    axes[0, 0].set_xlabel("Command (um)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Command Distribution")

    axes[0, 1].hist(df["error"], bins=50, alpha=0.7, color="orange")
    axes[0, 1].set_xlabel("Error (um)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Error Distribution")

    axes[1, 0].scatter(df["cmd_x"], df["actual_x"], s=1, alpha=0.3)
    axes[1, 0].plot([df["cmd_x"].min(), df["cmd_x"].max()], [df["cmd_x"].min(), df["cmd_x"].max()], "r--")
    axes[1, 0].set_xlabel("Command (um)")
    axes[1, 0].set_ylabel("Actual (um)")
    axes[1, 0].set_title("Command vs Actual")

    for direction, group in df.groupby("direction"):
        label = {1: "Increasing", -1: "Decreasing", 0: "Stationary"}.get(direction, "Unknown")
        axes[1, 1].plot(group["cmd_x"], group["error"], ".", alpha=0.3, label=label)
    axes[1, 1].set_xlabel("Command (um)")
    axes[1, 1].set_ylabel("Error (um)")
    axes[1, 1].set_title("Error by Direction")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def main():
    print("=" * 50)
    print("Data Preprocessing")
    print("=" * 50)

    raw_df = load_and_merge_data("collected_data")
    if raw_df is None:
        return 1

    forward_df = create_command_features(raw_df)
    inverse_df = create_inverse_features(raw_df)

    plot_data_distribution(forward_df)

    train_data_forward = prepare_training_data(
        forward_df,
        target="actual_x",
        feature_cols=FORWARD_FEATURES,
    )
    save_preprocessed_data(train_data_forward, "preprocessed_data_forward.npz")
    save_training_bundle(train_data_forward, "forward_model_data.pkl")

    train_data_inverse = prepare_training_data(
        inverse_df,
        target="cmd_x",
        feature_cols=INVERSE_FEATURES,
    )
    save_preprocessed_data(train_data_inverse, "preprocessed_data_inverse.npz")
    save_training_bundle(train_data_inverse, "inverse_model_data.pkl")

    print(f"\nTraining set size (forward): {len(train_data_forward['X_train'])}")
    print(f"Test set size (forward): {len(train_data_forward['X_test'])}")
    print(f"Features (forward): {train_data_forward['feature_names']}")
    print(f"\nTraining set size (inverse): {len(train_data_inverse['X_train'])}")
    print(f"Test set size (inverse): {len(train_data_inverse['X_test'])}")
    print(f"Features (inverse): {train_data_inverse['feature_names']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
