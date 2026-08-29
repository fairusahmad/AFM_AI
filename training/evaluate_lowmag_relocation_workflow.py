import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from afm_callbacks import AFMCallbacks
from afm_data import TrajectoryData
from afm_state import AFMState
from afm_utils import create_stage_fov, get_tip_position, render_camera_frame
from hysteresis import NanoPositioner
from sample_generation import artifact_layer, height_um, sample as stage_surface_image, width_um


def make_dummy_patch():
    class DummyPatch:
        def __init__(self):
            self._xy = np.array([[0.0, 0.0], [0.0, 0.0], [0.5, 0.5]], dtype=float)
            self._transform = None

        def set_xy(self, *args, **kwargs):
            if args:
                self._xy = np.asarray(args[0], dtype=float)
            return None

        def set_width(self, *args, **kwargs):
            return None

        def set_height(self, *args, **kwargs):
            return None

        def set_visible(self, *args, **kwargs):
            return None

        def set_transform(self, transform):
            self._transform = transform
            return None

        def get_xy(self):
            return self._xy

        def get_transform(self):
            return self._transform if self._transform is not None else plt.gca().transAxes

    return DummyPatch()


def build_callbacks():
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_axes([0.08, 0.12, 0.80, 0.78])
    ax.set_xticks([])
    ax.set_yticks([])

    state = AFMState(stage_surface_image, width_um, height_um)
    state.sample_source = "synthetic-surface"
    state.sample_path = None
    state.show_artifact = False
    stage = NanoPositioner(log_file="workflow_eval_movement_log.csv")
    data = TrajectoryData()

    initial_fov, outside_mask, ix, iy = create_stage_fov(
        state.surface_image,
        artifact_layer,
        state.show_artifact,
        state.x,
        state.y,
        state.fov_width,
        state.fov_height,
        valid_mask=state.surface_valid_mask,
    )
    state.current_fov_raw = initial_fov.copy()
    display_fov, _ = render_camera_frame(
        initial_fov,
        state.camera_resolution,
        outside_mask=outside_mask,
        focus_model=state.get_focus_model(),
    )
    img = ax.imshow(
        display_fov,
        cmap="gray",
        extent=[ix, ix + state.fov_width, iy + state.fov_height, iy],
        origin="upper",
    )

    tip = make_dummy_patch()
    cantilever = make_dummy_patch()
    rod = make_dummy_patch()
    callbacks = AFMCallbacks(
        state,
        stage,
        fig,
        ax,
        tip,
        cantilever,
        rod,
        0.50,
        data,
        lambda: None,
        lambda: (float(state.probe_tip_x), float(state.probe_tip_y)),
        {},
        artifact_layer,
    )
    callbacks.img = img
    logs = []
    callbacks.set_log_callback(logs.append)
    callbacks._refresh_current_view()
    return callbacks, state, stage, fig, logs


def prepare_reference_site(callbacks, state, *, final_zoom=5.0, manual_lowmag_count=4):
    reference_tip_x = float(state.probe_tip_x)
    reference_tip_y = float(state.probe_tip_y)
    coarse_zoom = float(min(state.zoom_levels))

    callbacks._reset_view_to_zoom(coarse_zoom, center_x_um=reference_tip_x, center_y_um=reference_tip_y)
    callbacks._refresh_current_view()
    _overview, merged_lowmag_landmarks, capture_report = callbacks._capture_reference_lowmag_landmark_map()

    manual_lowmag_landmarks = []
    for landmark in list(merged_lowmag_landmarks or [])[: max(int(manual_lowmag_count), 0)]:
        item = dict(landmark)
        item["manual"] = True
        item["capture_zoom_level"] = coarse_zoom
        manual_lowmag_landmarks.append(item)
    state.manual_reference_landmarks = manual_lowmag_landmarks

    callbacks._reset_view_to_zoom(final_zoom, center_x_um=reference_tip_x, center_y_um=reference_tip_y)
    callbacks._refresh_current_view()
    callbacks.save_reference(None)
    return capture_report


def run_trial(callbacks, state, trial_index, shift_x_um, shift_y_um, rotation_deg):
    capture_report = prepare_reference_site(callbacks, state, final_zoom=5.0, manual_lowmag_count=4)
    reference_tip_x = float(state.probe_tip_x)
    reference_tip_y = float(state.probe_tip_y)

    callbacks._apply_simulated_sample_remount(shift_x_um, shift_y_um, rotation_deg)
    callbacks._reset_view_to_zoom(
        min(state.zoom_levels),
        center_x_um=reference_tip_x,
        center_y_um=reference_tip_y,
    )
    callbacks._refresh_current_view()
    callbacks.relocate(None)

    report = state.lowmag_guidance_report or {}
    verification = (state.last_relocation_report or {}).get("verification") or {}
    ml_correction = report.get("ml_correction") or {}
    return {
        "trial_index": int(trial_index),
        "shift_x_um": float(shift_x_um),
        "shift_y_um": float(shift_y_um),
        "rotation_deg": float(rotation_deg),
        "saved_lowmag_frames": 0 if capture_report is None else int(capture_report.get("frame_count", 0)),
        "saved_lowmag_landmarks": 0 if state.site_memory is None else int(len(state.site_memory.get("lowmag_landmarks", []))),
        "support_count": int(report.get("support_count", 0)),
        "confidence": float(report.get("confidence", 0.0)),
        "search_frames": int(report.get("search_frames", 0)),
        "raw_dx_um": None if report.get("estimated_tip_dx_um") is None else float(report["estimated_tip_dx_um"]),
        "raw_dy_um": None if report.get("estimated_tip_dy_um") is None else float(report["estimated_tip_dy_um"]),
        "ml_dx_um": None if ml_correction.get("predicted_dx_um") is None else float(ml_correction["predicted_dx_um"]),
        "ml_dy_um": None if ml_correction.get("predicted_dy_um") is None else float(ml_correction["predicted_dy_um"]),
        "verified": bool(verification.get("verified", False)),
        "reference_score": float(verification.get("reference_score", 0.0)),
        "reference_gap": float(verification.get("reference_score_gap", 0.0)),
    }


def summarize(trials):
    raw_dx_errors = []
    raw_dy_errors = []
    ml_dx_errors = []
    ml_dy_errors = []
    for row in trials:
        if row["raw_dx_um"] is not None:
            raw_dx_errors.append(abs(float(row["raw_dx_um"]) - float(row["shift_x_um"])))
            raw_dy_errors.append(abs(float(row["raw_dy_um"]) - float(row["shift_y_um"])))
        if row["ml_dx_um"] is not None:
            ml_dx_errors.append(abs(float(row["ml_dx_um"]) - float(row["shift_x_um"])))
            ml_dy_errors.append(abs(float(row["ml_dy_um"]) - float(row["shift_y_um"])))
    return {
        "trial_count": int(len(trials)),
        "verified_count": int(sum(1 for row in trials if row["verified"])),
        "mean_raw_dx_error_um": None if not raw_dx_errors else float(np.mean(raw_dx_errors)),
        "mean_raw_dy_error_um": None if not raw_dy_errors else float(np.mean(raw_dy_errors)),
        "mean_ml_dx_error_um": None if not ml_dx_errors else float(np.mean(ml_dx_errors)),
        "mean_ml_dy_error_um": None if not ml_dy_errors else float(np.mean(ml_dy_errors)),
    }


def main():
    parser = argparse.ArgumentParser(description="Headless evaluation of low-mag relocation workflow.")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    callbacks, state, _stage, fig, logs = build_callbacks()
    trial_rows = []
    try:
        for trial_index in range(int(args.trials)):
            shift_x_um = float(rng.uniform(-160.0, 160.0))
            shift_y_um = float(rng.uniform(-160.0, 160.0))
            rotation_deg = float(rng.uniform(-5.0, 5.0))
            trial_rows.append(
                run_trial(callbacks, state, trial_index, shift_x_um, shift_y_um, rotation_deg)
            )
    finally:
        plt.close(fig)

    summary = summarize(trial_rows)
    print("Low-mag workflow evaluation")
    for row in trial_rows:
        print(
            f"trial={row['trial_index']} shift=({row['shift_x_um']:+.1f},{row['shift_y_um']:+.1f}) "
            f"saved_lowmag={row['saved_lowmag_landmarks']} "
            f"raw=({row['raw_dx_um'] if row['raw_dx_um'] is not None else 'n/a'},"
            f"{row['raw_dy_um'] if row['raw_dy_um'] is not None else 'n/a'}) "
            f"ml=({row['ml_dx_um'] if row['ml_dx_um'] is not None else 'n/a'},"
            f"{row['ml_dy_um'] if row['ml_dy_um'] is not None else 'n/a'}) "
            f"support={row['support_count']} conf={row['confidence']:.3f} "
            f"frames={row['search_frames']} verified={row['verified']}"
        )
    print(summary)
    if logs:
        print("Recent logs:")
        for line in logs[-12:]:
            print(line)


if __name__ == "__main__":
    main()
