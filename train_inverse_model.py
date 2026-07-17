"""
train_inverse_model.py - Train AI inverse model for hysteresis compensation
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor


def load_data(filename="inverse_model_data.pkl"):
    """Load preprocessed training data and scalers."""
    if not os.path.exists(filename):
        print(f"File {filename} does not exist, please run data preprocessing first")
        return None

    if filename.endswith(".pkl"):
        return joblib.load(filename)

    data = np.load(filename, allow_pickle=True)
    return {
        "X_train": data["X_train"],
        "X_test": data["X_test"],
        "y_train": data["y_train"],
        "y_test": data["y_test"],
        "feature_names": data["feature_names"].tolist() if "feature_names" in data.files else [],
    }


def train_mlp_model(data, hidden_layers=(50, 25, 10), max_iter=2000):
    """Train the inverse MLP model."""
    print("\n" + "=" * 50)
    print("Training MLP inverse model")
    print("=" * 50)
    print(f"Hidden layer structure: {hidden_layers}")
    print(f"Max iterations: {max_iter}")

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=42,
        verbose=True,
    )

    print("\nStarting training...")
    model.fit(data["X_train"], data["y_train"])

    y_train_pred = model.predict(data["X_train"])
    y_test_pred = model.predict(data["X_test"])

    train_mse = mean_squared_error(data["y_train"], y_train_pred)
    test_mse = mean_squared_error(data["y_test"], y_test_pred)
    train_r2 = r2_score(data["y_train"], y_train_pred)
    test_r2 = r2_score(data["y_test"], y_test_pred)

    print(f"\nTraining MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Training R^2: {train_r2:.4f}")
    print(f"Test R^2: {test_r2:.4f}")

    metrics = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_r2": train_r2,
        "test_r2": test_r2,
    }
    return model, metrics


def plot_training_results(data, model):
    """Plot training and validation diagnostics."""
    y_train_pred = model.predict(data["X_train"])
    y_test_pred = model.predict(data["X_test"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].scatter(data["y_train"], y_train_pred, alpha=0.5, s=1)
    axes[0, 0].plot([data["y_train"].min(), data["y_train"].max()], [data["y_train"].min(), data["y_train"].max()], "r--")
    axes[0, 0].set_xlabel("Actual")
    axes[0, 0].set_ylabel("Predicted")
    axes[0, 0].set_title("Training set: Predicted vs Actual")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(data["y_test"], y_test_pred, alpha=0.5, s=1)
    axes[0, 1].plot([data["y_test"].min(), data["y_test"].max()], [data["y_test"].min(), data["y_test"].max()], "r--")
    axes[0, 1].set_xlabel("Actual")
    axes[0, 1].set_ylabel("Predicted")
    axes[0, 1].set_title("Test set: Predicted vs Actual")
    axes[0, 1].grid(True, alpha=0.3)

    train_error = y_train_pred - data["y_train"]
    test_error = y_test_pred - data["y_test"]
    axes[1, 0].hist(train_error, bins=50, alpha=0.5, label="Training", color="blue")
    axes[1, 0].hist(test_error, bins=50, alpha=0.5, label="Test", color="orange")
    axes[1, 0].set_xlabel("Prediction error")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Error distribution")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].text(
        0.1,
        0.5,
        f"Training R^2 = {r2_score(data['y_train'], y_train_pred):.4f}\n"
        f"Test R^2 = {r2_score(data['y_test'], y_test_pred):.4f}",
        fontsize=14,
        transform=axes[1, 1].transAxes,
    )
    axes[1, 1].set_title("Model performance")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


def predict_command(model_bundle, desired, prev_1, prev_2):
    """Predict the compensated command for one axis."""
    scaler_X = model_bundle["scaler_X"]
    scaler_y = model_bundle["scaler_y"]
    model = model_bundle["model"]

    velocity = desired - prev_1
    direction = 0.0 if np.isclose(velocity, 0.0) else float(np.sign(velocity))
    features = np.array([[desired, prev_1, prev_2, direction, velocity]], dtype=float)
    features_scaled = scaler_X.transform(features)
    cmd_scaled = model.predict(features_scaled).reshape(-1, 1)
    return float(scaler_y.inverse_transform(cmd_scaled).ravel()[0])


def test_compensation(model_bundle):
    """
    Test the compensation effect of the inverse model.
    Simulation: desired position -> model predicts command -> hysteresis -> actual position
    """
    print("\n" + "=" * 50)
    print("Testing inverse model compensation effect")
    print("=" * 50)

    from hysteresis import NanoPositioner

    stage = NanoPositioner()
    stage.reset(0, 0)

    desired_positions = np.linspace(200, 1400, 50)

    cmd_no_comp = desired_positions
    actual_no_comp = []
    for cmd in cmd_no_comp:
        actual, _ = stage.move_to(cmd, 0)
        actual_no_comp.append(actual)

    stage.reset(0, 0)
    cmd_with_comp = []
    actual_with_comp = []
    desired_history = [0.0, 0.0]

    for desired in desired_positions:
        cmd_pred = predict_command(
            model_bundle,
            desired=desired,
            prev_1=desired_history[-1],
            prev_2=desired_history[-2],
        )
        cmd_with_comp.append(cmd_pred)
        desired_history.append(float(desired))
        desired_history = desired_history[-2:]

        actual, _ = stage.move_to(cmd_pred, 0)
        actual_with_comp.append(actual)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(desired_positions, desired_positions, "k--", label="Ideal (1:1)")
    axes[0].plot(desired_positions, actual_no_comp, "b-", label="Without compensation", alpha=0.7)
    axes[0].plot(desired_positions, actual_with_comp, "r-", label="With AI compensation", alpha=0.7)
    axes[0].set_xlabel("Desired position (um)")
    axes[0].set_ylabel("Actual position (um)")
    axes[0].set_title("Compensation effect comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis("equal")

    error_no_comp = np.array(actual_no_comp) - desired_positions
    error_with_comp = np.array(actual_with_comp) - desired_positions
    axes[1].plot(desired_positions, error_no_comp, "b-", label="Without compensation", alpha=0.7)
    axes[1].plot(desired_positions, error_with_comp, "r-", label="With AI compensation", alpha=0.7)
    axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Desired position (um)")
    axes[1].set_ylabel("Error (um)")
    axes[1].set_title("Error comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    rmse_no_comp = np.sqrt(np.mean(error_no_comp**2))
    rmse_with_comp = np.sqrt(np.mean(error_with_comp**2))
    axes[2].bar(
        ["Without compensation", "With AI compensation"],
        [rmse_no_comp, rmse_with_comp],
        color=["blue", "red"],
    )
    axes[2].set_ylabel("RMSE (um)")
    axes[2].set_title(f"RMSE: {rmse_no_comp:.2f} -> {rmse_with_comp:.2f}")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    print(f"\nRMSE without compensation: {rmse_no_comp:.2f} um")
    print(f"RMSE with AI compensation: {rmse_with_comp:.2f} um")
    print(f"Error reduction: {(1 - rmse_with_comp / rmse_no_comp) * 100:.1f}%")


def save_model(model, training_data, metrics, filename="inverse_model.pkl"):
    """Save the trained model package for UI integration."""
    model_data = {
        "model": model,
        "scaler_X": training_data["scaler_X"],
        "scaler_y": training_data["scaler_y"],
        "feature_names": training_data.get("feature_names", []),
        "target_name": training_data.get("target_name", "cmd_x"),
        "metrics": metrics,
    }
    joblib.dump(model_data, filename)
    print(f"\nModel saved to: {filename}")


def main():
    print("=" * 60)
    print("Training AI inverse model")
    print("=" * 60)

    print("\nLoading preprocessed data...")
    data = load_data("inverse_model_data.pkl")
    if data is None:
        print("Please run preprocess_data.py first to preprocess data")
        return 1

    required_keys = {"X_train", "X_test", "y_train", "y_test", "scaler_X", "scaler_y"}
    missing_keys = sorted(required_keys - set(data))
    if missing_keys:
        print(f"Training bundle is missing required keys: {missing_keys}")
        return 1

    print(f"Training set: {len(data['X_train'])} samples")
    print(f"Test set: {len(data['X_test'])} samples")

    model, metrics = train_mlp_model(data, hidden_layers=(100, 50, 25), max_iter=2000)
    plot_training_results(data, model)

    trained_bundle = dict(data)
    trained_bundle["model"] = model
    test_compensation(trained_bundle)

    save_model(model, data, metrics, "inverse_model.pkl")
    print("\nModel training complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
