"""
hysteresis.py - PI hysteresis model for AFM stage simulation.

This module implements a standard multi-operator Prandtl-Ishlinskii (PI)
hysteresis model for the X and Y scanner axes. The defaults are tuned to
produce a visibly stronger loop than the earlier mild 4.1 um configuration.
"""

import csv
import time

import matplotlib.pyplot as plt
import numpy as np


def play_operator_sequence(u, r):
    """
    Apply the play operator over a 1D input sequence.

    P_r[u](t_k) = max(u(t_k) - r, min(u(t_k) + r, P_r[u](t_{k-1})))
    with P_r[u](t_0) = u(t_0).
    """
    values = np.asarray(u, dtype=float).reshape(-1)
    if values.size == 0:
        return values.copy()

    output = np.empty_like(values)
    output[0] = values[0]
    radius = float(r)
    for idx in range(1, len(values)):
        output[idx] = max(values[idx] - radius, min(values[idx] + radius, output[idx - 1]))
    return output


def pi_model(u, thresholds, weights, linear_gain=0.0):
    """Standard PI model: y = a*u + sum_i w_i * P_{r_i}[u]."""
    values = np.asarray(u, dtype=float).reshape(-1)
    thresholds = np.asarray(thresholds, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if thresholds.shape != weights.shape:
        raise ValueError("thresholds and weights must have the same shape")

    if values.size == 0:
        return values.copy()

    output = float(linear_gain) * values
    for threshold, weight in zip(thresholds, weights):
        output = output + float(weight) * play_operator_sequence(values, threshold)
    return output


def hysteresis_model(u, params=None):
    """Compatibility wrapper that evaluates the PI model only."""
    params = {} if params is None else dict(params)
    thresholds = np.asarray(params.get("thresholds", _default_thresholds()), dtype=float)
    weights = np.asarray(params.get("weights", _default_weights(thresholds)), dtype=float)
    linear_gain = float(params.get("linear_gain", 0.0))
    return pi_model(u, thresholds, weights, linear_gain=linear_gain)


def validation_metrics(y_measured, y_predicted, direction_labels=None):
    """Compute RMSE, MAE, max error, and optional direction-split metrics."""
    measured = np.asarray(y_measured, dtype=float).reshape(-1)
    predicted = np.asarray(y_predicted, dtype=float).reshape(-1)
    if measured.shape != predicted.shape:
        raise ValueError("y_measured and y_predicted must have the same shape")

    residual = measured - predicted
    metrics = {
        "rmse": float(np.sqrt(np.mean(residual ** 2))) if residual.size else 0.0,
        "mae": float(np.mean(np.abs(residual))) if residual.size else 0.0,
        "max_error": float(np.max(np.abs(residual))) if residual.size else 0.0,
    }

    if direction_labels is not None:
        labels = np.asarray(direction_labels)
        if labels.shape[0] != residual.shape[0]:
            raise ValueError("direction_labels must match the number of samples")
        for label in np.unique(labels):
            mask = labels == label
            if not np.any(mask):
                continue
            label_residual = residual[mask]
            metrics[f"{label}_rmse"] = float(np.sqrt(np.mean(label_residual ** 2)))
            metrics[f"{label}_mae"] = float(np.mean(np.abs(label_residual)))
            metrics[f"{label}_max_error"] = float(np.max(np.abs(label_residual)))
    return metrics


class AxisPIState:
    """Stateful single-axis PI evaluator for real-time stage motion."""

    def __init__(self, thresholds, initial_value=0.0):
        self.thresholds = np.asarray(thresholds, dtype=float).reshape(-1)
        self.operator_states = np.full_like(self.thresholds, float(initial_value), dtype=float)

    def reset(self, value=0.0):
        self.operator_states.fill(float(value))

    def step(self, u, weights, linear_gain=0.0):
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape != self.thresholds.shape:
            raise ValueError("weights must match thresholds")

        total = float(linear_gain) * float(u)
        for idx, threshold in enumerate(self.thresholds):
            updated = max(float(u) - threshold, min(float(u) + threshold, self.operator_states[idx]))
            self.operator_states[idx] = updated
            total += float(weights[idx]) * updated
        return float(total)


class NanoPositioner:
    """
    Stateful XY scanner using a PI hysteresis model per axis.

    Backward compatibility:
    - move_to(target_x, target_y) -> (actual_x, actual_y)
    - move(dx, dy)
    - reset(x, y)
    - plot_hysteresis(...)
    """

    def __init__(
        self,
        r_list=None,
        w_list=None,
        log_file="movement_log.csv",
        linear_gain=0.0,
        creep_gain=0.18,
        creep_decay=0.14,
        creep_nonlinearity_um=40.0,
    ):
        self.r_list = np.asarray(_default_thresholds() if r_list is None else r_list, dtype=float)
        self.w_list = np.asarray(_default_weights(self.r_list) if w_list is None else w_list, dtype=float)
        if self.r_list.shape != self.w_list.shape:
            raise ValueError("r_list and w_list must have the same shape")

        self.n_ops = len(self.r_list)
        self.linear_gain_x = float(linear_gain)
        self.linear_gain_y = float(linear_gain)
        self.axis_x = AxisPIState(self.r_list)
        self.axis_y = AxisPIState(self.r_list)
        self.creep_gain = float(max(creep_gain, 0.0))
        self.creep_decay = float(np.clip(creep_decay, 0.0, 1.0))
        self.creep_nonlinearity_um = float(max(creep_nonlinearity_um, 1e-6))

        self.x = 0.0
        self.y = 0.0
        self.cmd_x = 0.0
        self.cmd_y = 0.0
        self._prev_cmd_x = 0.0
        self._prev_cmd_y = 0.0
        self._base_x = 0.0
        self._base_y = 0.0
        self._creep_x = 0.0
        self._creep_y = 0.0

        self.log_file = log_file
        with open(self.log_file, "w", newline="") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(["time_s", "cmd_x", "cmd_y", "actual_x", "actual_y"])
        self.start_time = time.time()

        self.history_cmd = []
        self.history_actual = []
        self.history_cmd_y = []
        self.history_actual_y = []
        self.history_time = []

        print("PI hysteresis model initialized")
        print(f"Number of operators: {self.n_ops}")
        print(f"Threshold range: [{self.r_list[0]:.1f}, {self.r_list[-1]:.1f}]")
        print(f"Weight sum: {np.sum(self.w_list):.4f}")
        print(f"Approximate saturated hysteresis offset: {float(np.sum(self.r_list * self.w_list)):.2f} um")
        print(
            "Creep model:"
            f" gain={self.creep_gain:.3f}, decay/frame={self.creep_decay:.3f},"
            f" nonlinearity={self.creep_nonlinearity_um:.1f} um"
        )

    def _creep_increment(self, delta_cmd):
        magnitude = abs(float(delta_cmd))
        if np.isclose(magnitude, 0.0):
            return 0.0
        saturation = 1.0 - np.exp(-magnitude / self.creep_nonlinearity_um)
        return -np.sign(delta_cmd) * self.creep_gain * magnitude * saturation

    def _decay_creep(self):
        keep = 1.0 - self.creep_decay
        self._creep_x *= keep
        self._creep_y *= keep

    def _update_position(self):
        self._base_x = self.axis_x.step(self.cmd_x, self.w_list, self.linear_gain_x)
        self._base_y = self.axis_y.step(self.cmd_y, self.w_list, self.linear_gain_y)

        delta_cmd_x = self.cmd_x - self._prev_cmd_x
        delta_cmd_y = self.cmd_y - self._prev_cmd_y
        self._prev_cmd_x = self.cmd_x
        self._prev_cmd_y = self.cmd_y
        self._creep_x += self._creep_increment(delta_cmd_x)
        self._creep_y += self._creep_increment(delta_cmd_y)
        self._decay_creep()

        self.x = self._base_x + self._creep_x
        self.y = self._base_y + self._creep_y
        self._log_motion()
        self._record_history()
        return self.x, self.y

    def _log_motion(self):
        t_now = time.time() - self.start_time
        with open(self.log_file, "a", newline="") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(
                [
                    f"{t_now:.6f}",
                    f"{self.cmd_x:.3f}",
                    f"{self.cmd_y:.3f}",
                    f"{self.x:.3f}",
                    f"{self.y:.3f}",
                ]
            )

    def _record_history(self):
        self.history_cmd.append(self.cmd_x)
        self.history_actual.append(self.x)
        self.history_cmd_y.append(self.cmd_y)
        self.history_actual_y.append(self.y)
        self.history_time.append(time.time() - self.start_time)
        if len(self.history_cmd) > 2000:
            self.history_cmd.pop(0)
            self.history_actual.pop(0)
            self.history_cmd_y.pop(0)
            self.history_actual_y.pop(0)
            self.history_time.pop(0)

    def move_to(self, target_x, target_y):
        self.cmd_x = float(target_x)
        self.cmd_y = float(target_y)
        return self._update_position()

    def move(self, dx, dy):
        self.cmd_x += float(dx)
        self.cmd_y += float(dy)
        return self._update_position()

    def reset(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)
        self.cmd_x = float(x)
        self.cmd_y = float(y)
        self.axis_x.reset(x)
        self.axis_y.reset(y)
        self._prev_cmd_x = float(x)
        self._prev_cmd_y = float(y)
        self._base_x = float(x)
        self._base_y = float(y)
        self._creep_x = 0.0
        self._creep_y = 0.0
        self._log_motion()

    def clear_history(self):
        self.history_cmd = []
        self.history_actual = []
        self.history_cmd_y = []
        self.history_actual_y = []
        self.history_time = []

    def plot_hysteresis(self, title="Hysteresis (PI Model)"):
        if len(self.history_cmd) == 0:
            print("No data to plot")
            return

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history_cmd, self.history_actual, "b-", linewidth=2, label="Actual X")
        plt.plot(self.history_cmd, self.history_cmd, "r--", linewidth=1.5, label="Ideal X")
        plt.fill_between(self.history_cmd, self.history_actual, self.history_cmd, alpha=0.2, color="blue")
        plt.xlabel("Command X (um)")
        plt.ylabel("Actual X (um)")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis("equal")

        plt.subplot(1, 2, 2)
        plt.plot(self.history_cmd_y, self.history_actual_y, "g-", linewidth=2, label="Actual Y")
        plt.plot(self.history_cmd_y, self.history_cmd_y, "r--", linewidth=1.5, label="Ideal Y")
        plt.fill_between(self.history_cmd_y, self.history_actual_y, self.history_cmd_y, alpha=0.2, color="green")
        plt.xlabel("Command Y (um)")
        plt.ylabel("Actual Y (um)")
        plt.title("Vertical PI Hysteresis")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis("equal")
        plt.tight_layout()
        plt.show()


def _default_thresholds():
    return np.linspace(0.0, 96.0, 9)


def _default_weights(thresholds):
    thresholds = np.asarray(thresholds, dtype=float)
    weights = np.exp(-thresholds / 24.0)
    total = float(np.sum(weights))
    if np.isclose(total, 0.0):
        return np.ones_like(thresholds) / max(len(thresholds), 1)
    return weights / total


if __name__ == "__main__":
    stage = NanoPositioner()
    u_up = np.linspace(0, 200, 200)
    u_down = np.linspace(200, 0, 200)
    u_seq = np.concatenate([u_up, u_down])
    actual = []
    for u_value in u_seq:
        x_value, _ = stage.move_to(u_value, 0)
        actual.append(x_value)
        time.sleep(0.005)

    plt.plot(u_seq, actual, "b-", label="Actual")
    plt.plot(u_seq, u_seq, "r--", label="Ideal")
    plt.xlabel("Command (um)")
    plt.ylabel("Actual (um)")
    plt.title("PI Model - Hysteresis Loop")
    plt.grid(True)
    plt.legend()
    plt.show()
