import random
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np

from afm_phase2_ml import (
    predict_remount_transform,
    retrieve_lowmag_candidates,
    score_same_site_probability,
)
from afm_ml_recognition import (
    DeepFeatureExtractor,
    MLPatternMatcher,
    MLSameSiteClassifier,
    MLTransformPredictor,
    MLTransformPredictor5W,
    load_deep_same_site_classifier,
    load_deep_remount_predictor,
    load_deep_remount_predictor_5w,
)
from afm_relocation import (
    analyze_landmark_geometry,
    apply_affine,
    build_overview,
    build_site_memory,
    estimate_landmark_consensus,
    expanded_rotation_affine,
    extract_landmarks,
    find_latest_site_memory,
    invert_affine,
    load_site_memory,
    match_template_candidates,
    persist_site_memory,
    rotation_translation_affine,
    transform_point,
    translate_image,
)
from afm_utils import create_stage_fov, render_camera_frame, rotate_camera_frame
from afm_utils import render_camera_recognition_frame
from artefact_detector import ArtefactDetector
from image_matching import match_reference_template
from sample_generation import load_real_sample_image, load_real_sample_image_from_scale

BASE_DIR = Path(__file__).resolve().parent


class AFMCallbacks:
    SCALE_BAR_CHOICES_UM = (50.0, 100.0, 200.0, 500.0)
    PROBE_TIP_REL_X = 0.50
    PROBE_TIP_REL_Y = 0.50
    PROBE_BODY_WIDTH_UM = 1600.0
    PROBE_BODY_LENGTH_UM = 3400.0
    PROBE_TIP_WIDTH_UM = 35.0
    PROBE_TIP_TOTAL_LENGTH_UM = 125.0
    PROBE_TRIANGULAR_TIP_LENGTH_UM = 15.0
    PROBE_VISIBLE_BODY_DEPTH_UM = 3400
    VIEWPORT_ARROW_HIT_RADIUS_AX = 0.07

    def __init__(
        self,
        state,
        stage,
        fig,
        ax,
        tip,
        cantilever,
        rod,
        center_x_ax,
        data,
        update_title_func,
        get_tip_func,
        buttons,
        artifact_layer,
    ):
        self.state = state
        self.stage = stage
        self.fig = fig
        self.ax = ax
        self.tip = tip
        self.cantilever = cantilever
        self.rod = rod
        self.center_x_ax = center_x_ax
        self.data = data
        self.update_title = update_title_func
        self.get_tip = get_tip_func
        self.buttons = buttons
        self.artifact_layer = artifact_layer
        self.img = None
        self.ai_mode = False
        self.ai_compensator = None
        self.same_site_classifier = None
        self.remount_transform_predictor = None
        self.lowmag_embedding_index = None
        self.busy_actions = set()
        self.log_callback = None
        self.status_callback = None
        self.persist_default_callback = None

        self.artefact_detector = None
        try:
            self.artefact_detector = ArtefactDetector()
            if getattr(self.artefact_detector, "model_loaded", False):
                self.log("Artefact detector ready")
            else:
                self.artefact_detector = None
                self.log("Artefact detector unavailable")
        except Exception as e:
            self.log(f"Failed to load artefact detector: {e}")

        inverse_model_path = self._resolve_project_path("inverse_model.pkl")
        try:
            self.ai_compensator = joblib.load(inverse_model_path)
            self.ai_mode = True
            self.log(f"AI inverse model loaded: {inverse_model_path}")
        except Exception as e:
            self.log(f"AI inverse model not available at {inverse_model_path}: {e}")

        phase2_models_dir = self._resolve_project_path("collected_data/models")
        self.same_site_classifier = self._load_optional_model(
            phase2_models_dir / "same_site_classifier.pkl",
            "same-site classifier",
        )
        self.remount_transform_predictor = self._load_optional_model(
            phase2_models_dir / "remount_transform_predictor.pkl",
            "remount transform predictor",
        )
        self.lowmag_embedding_index = self._load_optional_model(
            phase2_models_dir / "lowmag_embedding_index.pkl",
            "low-mag embedding index",
        )

        # ── Deep ML 模型 (ResNet18 特征 + MLP) ──
        self.deep_feature_extractor = None
        self.ml_pattern_matcher = None
        self.deep_classifier = None
        self.deep_regressor = None
        self.deep_regressor_is_5w = False
        self._ml_ready = False
        try:
            self.deep_feature_extractor = DeepFeatureExtractor()
            self.ml_pattern_matcher = MLPatternMatcher(extractor=self.deep_feature_extractor)
            self.deep_classifier = load_deep_same_site_classifier(
                phase2_models_dir / "deep_same_site_classifier.pkl"
            )
            if self.deep_classifier is not None:
                self.deep_classifier.extractor = self.deep_feature_extractor
                self.log("Deep same-site classifier loaded")

            # 优先加载 5w 大样本模型，回退到旧版
            self.deep_regressor = load_deep_remount_predictor_5w(
                phase2_models_dir / "deep_remount_predictor_real.pkl"
            )
            if self.deep_regressor is None:
                self.deep_regressor = load_deep_remount_predictor_5w(
                    phase2_models_dir / "deep_remount_predictor_5w.pkl"
                )
            if self.deep_regressor is None:
                # 回退: _final.pkl (用户手动重命名的副本)
                self.deep_regressor = load_deep_remount_predictor_5w(
                    phase2_models_dir / "deep_remount_predictor_final.pkl"
                )
            if self.deep_regressor is not None:
                self.deep_regressor.extractor = self.deep_feature_extractor
                self.deep_regressor_is_5w = True
                self.log("Deep remount predictor loaded (5w large-sample model)")
            else:
                # 最后回退: 旧版模型
                self.deep_regressor = load_deep_remount_predictor(
                    phase2_models_dir / "deep_remount_predictor.pkl"
                )
                if self.deep_regressor is not None:
                    self.deep_regressor.extractor = self.deep_feature_extractor
                    self.log("Deep remount predictor loaded (legacy model)")

            if self.deep_regressor is not None:
                pass  # 已在上面各自记录

            self._ml_ready = self.deep_classifier is not None or self.deep_regressor is not None
            if self._ml_ready:
                self.log("Deep ML recognition ENABLED")
        except Exception as e:
            self.log(f"Deep ML models not available: {e}")

    def _use_ml(self):
        """检查 ML 模型是否全部就绪"""
        return self._ml_ready and self.deep_classifier is not None

    def set_log_callback(self, callback):
        self.log_callback = callback

    def set_status_callback(self, callback):
        self.status_callback = callback

    def set_persist_default_callback(self, callback):
        self.persist_default_callback = callback

    def log(self, message):
        if self.log_callback is not None:
            self.log_callback(message)
        else:
            print(message)
        if self.status_callback is not None:
            self.status_callback()

    def _resolve_project_path(self, path_str):
        path = Path(path_str)
        if path.is_absolute():
            return path
        cwd_candidate = Path.cwd() / path
        if cwd_candidate.exists():
            return cwd_candidate
        return BASE_DIR / path

    def _load_optional_model(self, model_path, label):
        try:
            if Path(model_path).exists():
                model = joblib.load(model_path)
                self.log(f"Loaded {label}: {model_path}")
                return model
        except Exception as e:
            self.log(f"Failed to load {label} from {model_path}: {e}")
        return None

    def _begin_action(self, action_name, message):
        if action_name in self.busy_actions:
            self.log(message)
            return False
        self.busy_actions.add(action_name)
        return True

    def _end_action(self, action_name):
        self.busy_actions.discard(action_name)

    def _start_smooth_move(self, target_x, target_y):
        self.state.smooth_move_active = True
        self.state.smooth_move_target_x = target_x
        self.state.smooth_move_target_y = target_y

    def _cancel_active_motion(self):
        self.state.auto_scan_active = False
        self.state.smooth_move_active = False
        self.state.smooth_move_target_x = float(self.state.x)
        self.state.smooth_move_target_y = float(self.state.y)
        self.state.target_x = float(self.state.x)
        self.state.target_y = float(self.state.y)
        if self.state.pi_mode:
            self.stage.reset(self.state.x, self.state.y)
            self.stage.cmd_x = float(self.state.x)
            self.stage.cmd_y = float(self.state.y)

    def _jump_view_to_target(self, target_x=None, target_y=None):
        target_x = float(self.state.target_x if target_x is None else target_x)
        target_y = float(self.state.target_y if target_y is None else target_y)
        target_x, target_y = self._clamp_to_stage_margin(target_x, target_y)
        delta_x = float(target_x - self.state.x)
        delta_y = float(target_y - self.state.y)
        self.state.auto_scan_active = False
        self.state.smooth_move_active = False
        self.state.smooth_move_target_x = target_x
        self.state.smooth_move_target_y = target_y
        self.state.x = target_x
        self.state.y = target_y
        self.state.target_x = target_x
        self.state.target_y = target_y
        self._translate_probe_tip(delta_x, delta_y)
        if self.state.pi_mode:
            self.stage.reset(self.state.x, self.state.y)
            self.stage.cmd_x = float(self.state.x)
            self.stage.cmd_y = float(self.state.y)
        self._refresh_current_view()
        self.update_title()
        return delta_x, delta_y

    def _get_zoom_level_index(self, zoom_level):
        levels = np.array(self.state.zoom_levels, dtype=float)
        return int(np.argmin(np.abs(levels - float(zoom_level))))

    def _wait_for_zoom_complete(self, timeout_sec=2.0, poll_interval=0.05):
        """Block until the current zoom animation finishes, or timeout."""
        import time as _time
        start = _time.time()
        while self.state.zooming:
            if _time.time() - start > timeout_sec:
                self.log("Zoom wait timed out — forcing zoom completion")
                self.state.current_zoom_level = float(self.state.target_zoom_level)
                self.state.fov_width, self.state.fov_height = self.state.get_fov_for_zoom_level(self.state.current_zoom_level)
                self.state.zooming = False
                self.state.zoom_progress = 0
                break
            _time.sleep(poll_interval)
        # Force a view refresh
        self._refresh_current_view()

    def _begin_quantized_zoom(self, target_zoom_level):
        levels = list(self.state.zoom_levels)
        target_zoom_level = float(levels[self._get_zoom_level_index(target_zoom_level)])
        if self.state.zooming:
            self.log("Zoom already in progress")
            return
        if np.isclose(target_zoom_level, float(self.state.current_zoom_level)):
            self.log(f"Zoom already at {target_zoom_level:g}x")
            return

        self.state.zooming = True
        self.state.zoom_progress = 0
        self.state.zoom_direction = 1 if target_zoom_level > self.state.current_zoom_level else -1
        self.state.zoom_base_width = self.state.fov_width
        self.state.zoom_base_height = self.state.fov_height
        self.state.zoom_target_width, self.state.zoom_target_height = self.state.get_fov_for_zoom_level(target_zoom_level)
        self.state.target_zoom_level = target_zoom_level
        self.state.zoom_center_x = float(self.state.probe_tip_x)
        self.state.zoom_center_y = float(self.state.probe_tip_y)
        self.log(
            f"Zoom transition: {self.state.current_zoom_level:g}x -> {target_zoom_level:g}x "
            f"around tip X={self.state.zoom_center_x:.1f} um, Y={self.state.zoom_center_y:.1f} um"
        )

    def _set_probe_tip_to_view_center(self):
        self.state.probe_tip_x = float(self.state.x + self.state.fov_width * self.PROBE_TIP_REL_X)
        self.state.probe_tip_y = float(self.state.y + self.state.fov_height * self.PROBE_TIP_REL_Y)

    def _set_random_start_view(self):
        self.state.current_zoom_level = 0.25
        self.state.target_zoom_level = 0.25
        self.state.fov_width, self.state.fov_height = self.state.get_fov_for_zoom_level(self.state.current_zoom_level)
        max_x = max(float(self.state.width_um) - float(self.state.fov_width), 0.0)
        max_y = max(float(self.state.height_um) - float(self.state.fov_height), 0.0)
        self.state.x = float(random.uniform(0.0, max_x)) if max_x > 0.0 else 0.0
        self.state.y = float(random.uniform(0.0, max_y)) if max_y > 0.0 else 0.0
        self.state.target_x = self.state.x
        self.state.target_y = self.state.y
        self._set_probe_tip_to_view_center()

    def _translate_probe_tip(self, delta_x, delta_y):
        self.state.probe_tip_x = float(self.state.probe_tip_x + delta_x)
        self.state.probe_tip_y = float(self.state.probe_tip_y + delta_y)

    def _pan_viewport_without_moving_probe(self, delta_x, delta_y):
        new_x, new_y = self._clamp_to_stage_margin(self.state.x + delta_x, self.state.y + delta_y)
        actual_dx = float(new_x - self.state.x)
        actual_dy = float(new_y - self.state.y)
        if np.isclose(actual_dx, 0.0) and np.isclose(actual_dy, 0.0):
            self.log("Viewport pan reached the stage limit")
            return
        self.state.auto_scan_active = False
        self.state.smooth_move_active = False
        self.state.x = new_x
        self.state.y = new_y
        self.state.target_x = new_x
        self.state.target_y = new_y
        if self.state.pi_mode:
            self.stage.reset(self.state.x, self.state.y)
            self.stage.cmd_x = self.state.x
            self.stage.cmd_y = self.state.y
        self.update_probe_visuals()
        if self.status_callback is not None:
            self.status_callback()
        self.log(
            f"Viewport panned by dX={actual_dx:+.1f} um, dY={actual_dy:+.1f} um while keeping the cantilever fixed on the sample stage"
        )

    def handle_viewport_arrow_click(self, event):
        if event.inaxes != self.ax or event.x is None or event.y is None:
            return False
        if getattr(event, "button", None) != 1:
            return False
        if self.state.zooming:
            self.log("Wait for zoom to finish before panning the viewport.")
            return True

        x_ax, y_ax = self.ax.transAxes.inverted().transform((event.x, event.y))
        hit = float(self.VIEWPORT_ARROW_HIT_RADIUS_AX)
        step = float(self.state.current_step)

        if abs(x_ax - 0.50) <= hit and abs(y_ax - 0.95) <= hit:
            self._pan_viewport_without_moving_probe(0.0, -step)
            return True
        if abs(x_ax - 0.50) <= hit and abs(y_ax - 0.03) <= hit:
            self._pan_viewport_without_moving_probe(0.0, step)
            return True
        if abs(x_ax - 0.03) <= hit and abs(y_ax - 0.50) <= hit:
            self._pan_viewport_without_moving_probe(-step, 0.0)
            return True
        if abs(x_ax - 0.97) <= hit and abs(y_ax - 0.50) <= hit:
            self._pan_viewport_without_moving_probe(step, 0.0)
            return True
        return False

    def update_probe_visuals(self):
        tip_x = float(self.state.probe_tip_x)
        tip_y = float(self.state.probe_tip_y)
        zoom_level = max(float(self.state.current_zoom_level), 1e-6)
        camera_fixed_scale = 1.0 / zoom_level if zoom_level < 1.0 else 1.0
        geometry_scale = camera_fixed_scale
        tip_half_width_um = (self.PROBE_TIP_WIDTH_UM / 2.0) * geometry_scale
        tip_total_length_um = self.PROBE_TIP_TOTAL_LENGTH_UM * geometry_scale
        triangular_tip_length_um = self.PROBE_TRIANGULAR_TIP_LENGTH_UM * geometry_scale
        visible_body_depth_um = self.PROBE_VISIBLE_BODY_DEPTH_UM * geometry_scale
        body_half_width_um = (self.PROBE_BODY_WIDTH_UM / 2.0) * geometry_scale

        triangle_base_y = tip_y + triangular_tip_length_um
        body_top_y = tip_y + tip_total_length_um
        body_bottom_y = body_top_y + visible_body_depth_um

        self.cantilever.set_transform(self.ax.transData)
        self.rod.set_transform(self.ax.transData)
        self.tip.set_transform(self.ax.transData)

        self.cantilever.set_xy(
            [
                [tip_x - body_half_width_um, body_bottom_y],
                [tip_x + body_half_width_um, body_bottom_y],
                [tip_x + body_half_width_um, body_top_y],
                [tip_x + tip_half_width_um, body_top_y],
                [tip_x + tip_half_width_um, triangle_base_y],
                [tip_x, tip_y],
                [tip_x - tip_half_width_um, triangle_base_y],
                [tip_x - tip_half_width_um, body_top_y],
                [tip_x - body_half_width_um, body_top_y],
            ]
        )

        self.rod.set_xy((tip_x, tip_y))
        self.rod.set_width(0.0)
        self.rod.set_height(0.0)

        self.tip.set_xy(
            [
                [tip_x - tip_half_width_um, triangle_base_y],
                [tip_x + tip_half_width_um, triangle_base_y],
                [tip_x, tip_y],
            ]
        )
        self.fig.canvas.draw_idle()

    def _sync_hud_master_state(self):
        self.state.show_probe_hud = bool(self.state.show_hud_detection or self.state.show_hud_distance)

    def get_hud_button_label(self):
        detection_on = bool(self.state.show_hud_detection)
        distance_on = bool(self.state.show_hud_distance)
        if detection_on and distance_on:
            mode = "ALL"
        elif detection_on:
            mode = "DET"
        elif distance_on:
            mode = "DIST"
        else:
            mode = "OFF"
        return f"HUD: {mode}"

    def toggle_probe_hud(self, event):
        modes = [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ]
        current = (bool(self.state.show_hud_detection), bool(self.state.show_hud_distance))
        try:
            current_index = modes.index(current)
        except ValueError:
            current_index = 0
        next_detection, next_distance = modes[(current_index + 1) % len(modes)]
        self.state.show_hud_detection = next_detection
        self.state.show_hud_distance = next_distance
        self._sync_hud_master_state()
        if "hud" in self.buttons:
            self.buttons["hud"].label.set_text(self.get_hud_button_label())
        if self.status_callback is not None:
            self.status_callback()
        self.fig.canvas.draw_idle()
        self.log(
            "Probe HUD mode: "
            f"detection={'ON' if self.state.show_hud_detection else 'OFF'}, "
            f"distance={'ON' if self.state.show_hud_distance else 'OFF'}"
        )

    def _clamp_z_stage(self, z_position_um):
        half_travel = float(self.state.z_stage_travel_um) / 2.0
        return float(np.clip(z_position_um, -half_travel, half_travel))

    def get_position_center(self):
        return (
            float(self.state.x + self.state.fov_width / 2.0),
            float(self.state.y + self.state.fov_height / 2.0),
        )

    def get_target_center(self):
        return (
            float(self.state.target_x + self.state.fov_width / 2.0),
            float(self.state.target_y + self.state.fov_height / 2.0),
        )

    def set_origin(self, x_um, y_um, label=None):
        self.state.origin_x = float(x_um)
        self.state.origin_y = float(y_um)
        if label:
            self.state.origin_label = str(label)
        elif not self.state.origin_label:
            self.state.origin_label = "Origin"
        self.state.origin_defined = True
        self._capture_origin_template(self.state.origin_x, self.state.origin_y)
        if self.status_callback is not None:
            self.status_callback()
        self.fig.canvas.draw_idle()
        self.log(
            f"Origin set: {self.state.origin_label} at "
            f"X={self.state.origin_x:.1f} um, Y={self.state.origin_y:.1f} um"
        )

    def clear_origin(self):
        self.state.origin_defined = False
        self.state.origin_template = None
        if self.status_callback is not None:
            self.status_callback()
        self.fig.canvas.draw_idle()
        self.log("Origin cleared")

    def _capture_origin_template(self, abs_x, abs_y):
        source_view = self.state.current_camera_view if self.state.current_camera_view is not None else self.state.current_fov_raw
        if source_view is None:
            self.state.origin_template = None
            return
        rel_x = int(round(abs_x - self.state.x))
        rel_y = int(round(abs_y - self.state.y))
        half_size = int(self.state.origin_template_half_size)
        x0 = max(0, rel_x - half_size)
        x1 = min(source_view.shape[1], rel_x + half_size)
        y0 = max(0, rel_y - half_size)
        y1 = min(source_view.shape[0], rel_y + half_size)
        if x1 - x0 < 16 or y1 - y0 < 16:
            self.state.origin_template = None
            self.log("Origin template too small in current view; move origin away from the edge and set it again.")
            return
        self.state.origin_template = source_view[y0:y1, x0:x1].copy()

    def _load_sample_into_state(self, sample, width_um, height_um, sample_source, sample_path=None):
        self.state.surface_image = sample
        self.state.sample = self.state.surface_image
        self.state.surface_valid_mask = np.ones(sample.shape[:2], dtype=bool)
        self.state.width_um = float(width_um)
        self.state.height_um = float(height_um)
        self.state.sample_source = sample_source
        self.state.sample_path = str(sample_path) if sample_path else None
        if sample_path:
            self.state.default_image_path = str(sample_path)

        self._set_random_start_view()
        self.state.stage_margin_um = max(2000.0, float(max(self.state.width_um, self.state.height_um)) * 1.5)
        self.state.sample_removed = False
        self.state.show_artifact = False
        self.state.surface_tilt_angle = float(random.uniform(-10.0, 10.0))
        self.state.z_stage_position_um = float(
            self.state.get_effective_camera_stage_position_um() - self.state.focus_z_um
        )
        self.state.simulated_sample_shift_x_um = 0.0
        self.state.simulated_sample_shift_y_um = 0.0
        self.state.simulated_sample_rotation_deg = 0.0
        self.state.simulated_sample_tilt_deg = float(self.state.surface_tilt_angle)
        self.state.ai_desired_history_x = [self.state.x, self.state.x]
        self.state.ai_desired_history_y = [self.state.y, self.state.y]
        self._clear_reference_data()

        if self.artifact_layer is not None:
            self.artifact_layer.reset_canvas(sample.shape[1], sample.shape[0])

        self.stage.reset(self.state.x, self.state.y)
        self._refresh_current_view()
        self._try_load_latest_site_memory()
        self.update_title()

    def load_default_image(self, event=None):
        if not self._begin_action("load_default_image", "Default image loading is already running"):
            return
        try:
            candidate_paths = []
            if self.state.default_image_path:
                candidate_paths.append(Path(self.state.default_image_path))
            candidate_paths.extend(
                [
                    self._resolve_project_path("afm_ideal_scan.png"),
                    self._resolve_project_path("../gambar/figure4_vision.png"),
                ]
            )
            image_path = next((path for path in candidate_paths if path.exists()), None)
            if image_path is None:
                self.log("No default image found. Use Load Image to choose one.")
                return

            raw = plt.imread(image_path)
            if raw.ndim == 3:
                raw = raw[..., 0]
            scale_bar_um = 500.0
            scale_bar_px = 140.0
            self.state.default_scale_um_per_px = scale_bar_um / scale_bar_px
            default_width_um = self.state.default_image_width_um
            default_height_um = self.state.default_image_height_um
            if default_width_um and default_height_um:
                width_um = float(default_width_um)
                height_um = float(default_height_um)
            else:
                width_um = float(raw.shape[1])
                height_um = float(raw.shape[0])
            sample, width_um, height_um = load_real_sample_image(
                str(image_path),
                width_um,
                height_um,
            )
            self._load_sample_into_state(sample, width_um, height_um, "default-image", image_path)
            self.log(f"Loaded default image: {image_path}")
            if default_width_um and default_height_um:
                self.log(
                    f"Loaded saved default calibration: {width_um / 1000.0:.3f} mm x "
                    f"{height_um / 1000.0:.3f} mm"
                )
            else:
                self.log(
                    f"Default image scale estimate: {self.state.default_scale_um_per_px:.3f} um/px "
                    f"from 500 um over ~140 px"
                )
        except Exception as e:
            self.log(f"Failed to load default image: {e}")
        finally:
            self._end_action("load_default_image")

    def _clear_reference_data(self):
        self.state.ref_template = None
        self.state.ref_artefacts = []
        self.state.ref_x = 0.0
        self.state.ref_y = 0.0
        self.state.site_memory = None
        self.state.last_saved_site_dir = None
        self.state.last_relocation_report = None
        self.state.last_affine_transform_report = None

    def _refresh_current_view(self):
        fov, outside_mask, ix, iy = create_stage_fov(
            self.state.surface_image,
            self.artifact_layer,
            self.state.show_artifact,
            self.state.x,
            self.state.y,
            self.state.fov_width,
            self.state.fov_height,
            valid_mask=self.state.surface_valid_mask,
        )
        self.state.current_fov_raw = fov.copy()
        self.state.current_camera_view, _ = render_camera_recognition_frame(
            fov,
            outside_mask=outside_mask,
            focus_model=self.state.get_focus_model(),
            fov_width_um=self.state.fov_width,
            fov_height_um=self.state.fov_height,
            tip_rel_x=self.PROBE_TIP_REL_X,
            tip_rel_y=self.PROBE_TIP_REL_Y,
            body_width_um=self.state.probe_body_width_um,
            tip_width_um=self.state.probe_tip_width_um,
            tip_total_length_um=self.state.probe_tip_total_length_um,
            triangular_tip_length_um=self.state.probe_triangular_tip_length_um,
            visible_body_depth_um=self.state.probe_visible_body_depth_um,
        )
        display_fov, focus_metrics = render_camera_frame(
            fov,
            self.state.camera_resolution,
            outside_mask=outside_mask,
            focus_model=self.state.get_focus_model(),
        )
        self.state.last_blur_diameter_um = focus_metrics["blur_diameter_um"]
        self.state.last_blur_sigma_px = focus_metrics["sigma_px"]
        self.state.last_dof_camera_um = focus_metrics["dof_camera_um"]
        display_fov = rotate_camera_frame(display_fov, self.state.surface_tilt_angle)
        self.img.set_data(display_fov)
        self.img.set_extent([ix, ix + self.state.fov_width, iy + self.state.fov_height, iy])
        self.ax.set_xlim(ix, ix + self.state.fov_width)
        self.ax.set_ylim(iy + self.state.fov_height, iy)
        self.update_probe_visuals()
        self.fig.canvas.draw_idle()

    def _clamp_to_stage_margin(self, x, y):
        min_x = -self.state.stage_margin_um
        min_y = -self.state.stage_margin_um
        max_x = self.state.width_um + self.state.stage_margin_um - self.state.fov_width
        max_y = self.state.height_um + self.state.stage_margin_um - self.state.fov_height
        return float(np.clip(x, min_x, max_x)), float(np.clip(y, min_y, max_y))

    def move_to_clicked_point(self, event):
        if getattr(self.state, "detection_roi_draw_mode", False) or getattr(self.state, "detection_roi_drag_active", False):
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if self.handle_viewport_arrow_click(event):
            return

        # Page 2/3: click-to-move correction mode (AI relocation fallback)
        if self.state.ai_relocate_awaiting_click:
            if getattr(event, "button", None) == 3:
                # Right-click: cancel correction mode
                self.state.ai_relocate_awaiting_click = False
                self.state.ai_relocate_pending_target_x = None
                self.state.ai_relocate_pending_target_y = None
                self.log("Click-to-move correction cancelled.")
                return
            if getattr(event, "button", None) != 1:
                return
            clicked_x = float(event.xdata)
            clicked_y = float(event.ydata)
            # Target viewport top-left: center the view on clicked point
            target_tl_x = clicked_x - self.state.fov_width / 2.0
            target_tl_y = clicked_y - self.state.fov_height / 2.0
            target_tl_x, target_tl_y = self._clamp_to_stage_margin(target_tl_x, target_tl_y)
            self.state.ai_relocate_awaiting_click = False
            self.state.ai_relocate_pending_target_x = None
            self.state.ai_relocate_pending_target_y = None
            self._start_smooth_move(target_tl_x, target_tl_y)
            self.state.sample_removed = False
            self.log(
                f"📍 Manual correction: cantilever moved to clicked position "
                f"({clicked_x:.1f}, {clicked_y:.1f}) → viewport top-left=({target_tl_x:.1f}, {target_tl_y:.1f})"
            )
            return

        if getattr(event, "button", None) != 1:
            return
        if self.state.zooming:
            self.log("Wait for zoom to finish before selecting a new scan area.")
            return

        clicked_x = float(event.xdata)
        clicked_y = float(event.ydata)
        delta_x = clicked_x - float(self.state.probe_tip_x)
        delta_y = clicked_y - float(self.state.probe_tip_y)
        target_x = self.state.target_x + delta_x
        target_y = self.state.target_y + delta_y
        target_x, target_y = self._clamp_to_stage_margin(target_x, target_y)
        self.state.auto_scan_active = False

        if self.state.pi_mode:
            self._start_smooth_move(target_x, target_y)
        else:
            self.state.target_x = target_x
            self.state.target_y = target_y

        self.log(
            f"AOI selected at X={clicked_x:.1f} um, Y={clicked_y:.1f} um -> "
            f"moving cantilever tip by dX={delta_x:+.1f} um, dY={delta_y:+.1f} um"
        )

    def _append_desired_history(self, axis, value):
        history_attr = f"ai_desired_history_{axis}"
        history = list(getattr(self.state, history_attr))
        history.append(float(value))
        setattr(self.state, history_attr, history[-2:])

    def _predict_compensated_axis(self, desired, axis):
        if not self.ai_mode or not self.ai_compensator:
            return float(desired)

        required_keys = {"model", "scaler_X", "scaler_y"}
        if not required_keys.issubset(self.ai_compensator):
            return float(desired)

        history = getattr(self.state, f"ai_desired_history_{axis}", [float(desired), float(desired)])
        prev_1 = float(history[-1])
        prev_2 = float(history[-2])
        velocity = float(desired - prev_1)
        direction = 0.0 if np.isclose(velocity, 0.0) else float(np.sign(velocity))

        features = np.array([[desired, prev_1, prev_2, direction, velocity]], dtype=float)
        scaler_X = self.ai_compensator["scaler_X"]
        scaler_y = self.ai_compensator["scaler_y"]
        model = self.ai_compensator["model"]

        features_scaled = scaler_X.transform(features)
        prediction_scaled = model.predict(features_scaled).reshape(-1, 1)
        command = float(scaler_y.inverse_transform(prediction_scaled).ravel()[0])

        self._append_desired_history(axis, desired)
        return command

    def load_sample_image(self, event):
        if not self._begin_action("load_sample_image", "Image loading is already running"):
            return
        root = tk.Tk()
        try:
            root.withdraw()
            image_path = filedialog.askopenfilename(
                title="Select Microscope Image",
                filetypes=[
                    ("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),
                    ("All files", "*.*"),
                ],
            )
            if not image_path:
                return

            use_scale_bar = messagebox.askyesno(
                "Scale Calibration",
                "Do you want to calibrate using the scale bar embedded in the image?\n\n"
                "Choose Yes to click the two ends of the original scale bar.\n"
                "Choose No to enter the full image width and height manually.",
            )

            if use_scale_bar:
                raw_image = plt.imread(image_path)
                if raw_image.ndim == 3:
                    raw_image = raw_image[..., 0]

                fig, ax = plt.subplots(figsize=(10, 7))
                ax.imshow(raw_image, cmap="gray")
                ax.set_title("Click the LEFT and RIGHT ends of the original scale bar, then close this window")
                points = plt.ginput(2, timeout=-1)
                plt.close(fig)

                if len(points) != 2:
                    self.log("Scale bar calibration cancelled")
                    return

                (x1, y1), (x2, y2) = points
                scale_length_px = float(np.hypot(x2 - x1, y2 - y1))
                scale_length_um = simpledialog.askfloat(
                    "Scale Bar Length",
                    "Enter the real scale bar length (um):",
                    initialvalue=500.0,
                    minvalue=0.001,
                )
                if scale_length_um is None:
                    return

                sample, width_um, height_um = load_real_sample_image_from_scale(
                    image_path,
                    scale_bar_length_um=scale_length_um,
                    scale_bar_length_px=scale_length_px,
                )
                self.log(
                    f"Scale calibration from image bar: {scale_length_um:.3f} um "
                    f"over {scale_length_px:.2f} px"
                )
            else:
                width_mm = simpledialog.askfloat("Sample Width", "Enter physical image width (mm):", initialvalue=2.0, minvalue=0.01)
                if width_mm is None:
                    return
                height_mm = simpledialog.askfloat("Sample Height", "Enter physical image height (mm):", initialvalue=2.0, minvalue=0.01)
                if height_mm is None:
                    return

                sample, width_um, height_um = load_real_sample_image(image_path, width_mm * 1000.0, height_mm * 1000.0)
                self.log(f"Manual calibration: {width_mm:.3f} mm x {height_mm:.3f} mm")

            self._load_sample_into_state(sample, width_um, height_um, "image", image_path)
            self.log(f"Loaded microscope image: {image_path}")
            self.log(f"Calibrated sample size: {self.state.width_um / 1000.0:.3f} mm x {self.state.height_um / 1000.0:.3f} mm")
            self.log("Internal scale: 1 pixel = 1 um after calibration resize")
        except Exception as e:
            self.log(f"Failed to load microscope image: {e}")
        finally:
            root.destroy()
            self._end_action("load_sample_image")

    def save_current_as_default(self, event=None):
        if not self.state.sample_path:
            self.log("Load an image first before saving it as the default.")
            return
        default_path = Path(self.state.sample_path)
        if not default_path.exists():
            self.log(f"Current image path no longer exists: {default_path}")
            return

        self.state.default_image_path = str(default_path)
        self.state.default_image_width_um = float(self.state.width_um)
        self.state.default_image_height_um = float(self.state.height_um)
        if self.persist_default_callback is not None:
            try:
                self.persist_default_callback(
                    {
                        "path": default_path,
                        "width_um": float(self.state.width_um),
                        "height_um": float(self.state.height_um),
                        "scale_um_per_px": float(self.state.default_scale_um_per_px),
                    }
                )
            except Exception as e:
                self.log(f"Failed to save default image setting: {e}")
                return
        self.log(
            f"Saved default image: {default_path} "
            f"with calibration {self.state.width_um / 1000.0:.3f} mm x {self.state.height_um / 1000.0:.3f} mm"
        )

    def _estimate_relocation_offset(self, reference_detections, current_detections):
        candidates = []
        for ref in reference_detections:
            for curr in current_detections:
                if curr["class_name"] != ref["class_name"]:
                    continue
                offset_x = curr["abs_coord"][0] - ref["abs_coord"][0]
                offset_y = curr["abs_coord"][1] - ref["abs_coord"][1]
                candidates.append(
                    {
                        "ref": ref,
                        "curr": curr,
                        "offset_x": offset_x,
                        "offset_y": offset_y,
                    }
                )

        if not candidates:
            return None

        offsets = np.array([[c["offset_x"], c["offset_y"]] for c in candidates], dtype=float)
        median_offset = np.median(offsets, axis=0)

        supporters = []
        for candidate in candidates:
            residual = float(
                np.hypot(
                    candidate["offset_x"] - median_offset[0],
                    candidate["offset_y"] - median_offset[1],
                )
            )
            if residual <= 75.0:
                candidate["residual"] = residual
                supporters.append(candidate)

        if not supporters:
            supporters = candidates
            for candidate in supporters:
                candidate["residual"] = float(
                    np.hypot(
                        candidate["offset_x"] - median_offset[0],
                        candidate["offset_y"] - median_offset[1],
                    )
                )

        offset_x = float(np.mean([c["offset_x"] for c in supporters]))
        offset_y = float(np.mean([c["offset_y"] for c in supporters]))
        best_match = min(supporters, key=lambda item: item["residual"])

        return {
            "offset_x": offset_x,
            "offset_y": offset_y,
            "best_match": best_match,
            "support_count": len(supporters),
        }

    def _persist_site_memory(self, site_memory):
        output_dir = persist_site_memory(
            site_memory,
            self._resolve_project_path("collected_data/site_memories"),
        )
        self.state.last_saved_site_dir = str(output_dir)
        self.state.saved_site_memories.append(str(output_dir))
        return output_dir

    def _activate_site_memory(self, site_memory, source_dir=None):
        self.state.site_memory = site_memory
        self.state.ref_template = site_memory.get("reference_template")
        self.state.ref_artefacts = list(site_memory.get("highmag_landmarks", []))
        reference_top_left = site_memory.get("reference_top_left") or {}
        self.state.ref_x = float(reference_top_left.get("x_um", 0.0))
        self.state.ref_y = float(reference_top_left.get("y_um", 0.0))
        origin_info = site_memory.get("origin")
        if origin_info:
            self.state.origin_label = str(origin_info.get("label", self.state.origin_label or "Origin"))
            self.state.origin_x = float(origin_info.get("x_um", self.state.origin_x))
            self.state.origin_y = float(origin_info.get("y_um", self.state.origin_y))
        self.state.origin_template = site_memory.get("origin_template")
        # Page 3: restore the zoom_level that was saved with this site memory
        saved_zoom = site_memory.get("zoom_level")
        if saved_zoom is not None:
            self._begin_quantized_zoom(float(saved_zoom))
            self.log(f"Restored zoom level to {float(saved_zoom):.2f}x from site memory")
        if source_dir is not None:
            self.state.last_saved_site_dir = str(source_dir)
            if str(source_dir) not in self.state.saved_site_memories:
                self.state.saved_site_memories.append(str(source_dir))

    def _try_load_latest_site_memory(self):
        sample_id = Path(self.state.sample_path).stem if self.state.sample_path else self.state.sample_source
        latest_dir = find_latest_site_memory(
            self._resolve_project_path("collected_data/site_memories"),
            sample_id,
        )
        if latest_dir is None:
            return False
        try:
            site_memory = load_site_memory(latest_dir)
        except Exception as e:
            self.log(f"Failed to load latest site memory from {latest_dir}: {e}")
            return False
        self._activate_site_memory(site_memory, source_dir=latest_dir)
        self.log(
            f"Loaded saved site memory: {site_memory.get('site_id', 'unknown')} "
            f"from {latest_dir}"
        )
        return True

    def _estimate_coarse_affine_transform(self):
        site_memory = self.state.site_memory or {}
        ref_tpl = self.state.ref_template
        if ref_tpl is None:
            self.log("No ref_template available")
            return None

        ref_tl = site_memory.get("reference_top_left") or {}
        ref_x = float(ref_tl.get("x_um", self.state.ref_x))
        ref_y = float(ref_tl.get("y_um", self.state.ref_y))
        search_half = max(self.state.relocation_fine_half_range_um * 6, 4000)
        tpl_h, tpl_w = ref_tpl.shape[:2]

        # -- ML 预估旋转, 缩小 NCC 搜索范围 --
        ml_angle = None
        if self.deep_regressor is not None:
            try:
                crop = self.state.surface_image[int(ref_y):int(ref_y)+tpl_h,
                                                int(ref_x):int(ref_x)+tpl_w]
                if crop.shape == ref_tpl.shape:
                    result = self.deep_regressor.predict(ref_tpl, crop)
                    if result is not None:
                        ml_angle = float(result["angle_deg"])
                        self.log(f"[ML] rotation estimate: {ml_angle:+.2f} deg")
            except:
                pass
        use_ml_narrow = self.state.force_ml_mode and ml_angle is not None
        if use_ml_narrow:
            sweep = list(range(int(ml_angle)-4, int(ml_angle)+5, 2))
            if 0 not in sweep: sweep.append(0)
            angles_to_try = sorted(set(sweep))
            self.log(f"[ML mode ON] narrow sweep around {ml_angle:+.1f} deg: {angles_to_try}")
        else:
            angles_to_try = list(range(-10, 11, 2))
            if ml_angle is not None:
                self.log(f"[ML mode OFF] full sweep (ML says {ml_angle:+.1f} deg but force_ml_mode=False)")
        ncc_match = None
        best_angle = 0.0
        self.log(f"[NCC] trying angles: {angles_to_try}")

        for try_angle in angles_to_try:
            if abs(try_angle) > 0.5:
                rm = cv2.getRotationMatrix2D((tpl_w / 2, tpl_h / 2), -try_angle, 1.0)
                tpl = cv2.warpAffine(ref_tpl, rm, (tpl_w, tpl_h), borderMode=cv2.BORDER_REPLICATE)
            else:
                tpl = ref_tpl
            match = match_reference_template(self.state.surface_image, tpl, ref_x, ref_y, half_range=search_half)
            if match is not None and (ncc_match is None or match["score"] > ncc_match["score"]):
                ncc_match = match
                best_angle = try_angle
                self.log(f"[NCC] angle={try_angle:+.1f} deg score={match['score']:.3f} at ({match['x']:.0f},{match['y']:.0f})")

        if ncc_match is None:
            self.log("[NCC] no match found on surface")
            return None

        dx_ncc = ncc_match["x"] - ref_x
        dy_ncc = ncc_match["y"] - ref_y
        final_angle = best_angle
        self.log(f"[NCC] result: dX={dx_ncc:+.0f} dY={dy_ncc:+.0f} um angle={final_angle:+.2f} deg score={ncc_match['score']:.3f}")

        # -- Step 3: ML verification (optional) --
        ml_prediction = None
        if self.deep_regressor is not None:
            try:
                mx2, my2 = int(round(ncc_match["x"])), int(round(ncc_match["y"]))
                sh2, sw2 = self.state.surface_image.shape[:2]
                cx2, cy2 = min(sw2, mx2 + tpl_w), min(sh2, my2 + tpl_h)
                if cx2 - mx2 >= tpl_w // 2 and cy2 - my2 >= tpl_h // 2:
                    vc = self.state.surface_image[my2:cy2, mx2:cx2]
                    if vc.shape[0] != tpl_h or vc.shape[1] != tpl_w:
                        fv = int(np.median(self.state.surface_image))
                        vc = cv2.copyMakeBorder(vc, 0, tpl_h - vc.shape[0], 0, tpl_w - vc.shape[1],
                                                borderType=cv2.BORDER_CONSTANT, value=fv)
                    mr = self.deep_regressor.predict(ref_tpl, vc)
                    if mr is not None:
                        ml_prediction = {"dx_um": mr["dx_px"], "dy_um": mr["dy_px"], "dtheta_deg": mr["angle_deg"]}
            except Exception as e:
                self.log(f"[ML verify] failed: {e}")

        # -- Build report --
        theta_rad = np.radians(final_angle)
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
        full_matrix = np.array([[cos_t, -sin_t, dx_ncc], [sin_t, cos_t, dy_ncc]], dtype=np.float32)
        affine = {"matrix": full_matrix.copy(), "rotation_deg": final_angle, "scale": 1.0,
                  "translation_px": (dx_ncc, dy_ncc), "match_count": 1, "inlier_count": 1,
                  "confidence": float(ncc_match["score"])}

        reference_overview = site_memory.get("overview")
        current_overview = build_overview(self.state.surface_image)
        if current_overview is None:
            current_overview = {}
        if reference_overview is None:
            reference_overview = {}
        report = dict(affine)
        report["current_overview"] = current_overview
        report["full_matrix"] = full_matrix
        report["ml_remount_prediction"] = ml_prediction
        cur_img = current_overview.get("image")
        ref_img = reference_overview.get("image")
        report["retrieval_candidates"] = (
            retrieve_lowmag_candidates(self.lowmag_embedding_index, cur_img, top_k=3)
            if cur_img is not None else []
        )
        report["predicted_remount_transform"] = (
            predict_remount_transform(self.remount_transform_predictor, ref_img, cur_img)
            if ref_img is not None and cur_img is not None else None
        )
        self.state.last_affine_transform_report = report
        return report


    def _apply_simulated_sample_translation(self, shift_x_um, shift_y_um):
        self.state.surface_image = translate_image(self.state.surface_image, shift_x_um, shift_y_um)
        self.state.sample = self.state.surface_image
        self.state.surface_valid_mask = translate_image(
            self.state.surface_valid_mask.astype(np.uint8),
            shift_x_um,
            shift_y_um,
            border_value=0,
        ) > 0
        if self.artifact_layer is not None and hasattr(self.artifact_layer, "layer"):
            self.artifact_layer.layer = translate_image(
                self.artifact_layer.layer,
                shift_x_um,
                shift_y_um,
                border_value=0,
            )
        self.state.simulated_sample_shift_x_um += float(shift_x_um)
        self.state.simulated_sample_shift_y_um += float(shift_y_um)

    def _apply_simulated_sample_remount(self, shift_x_um, shift_y_um, rotation_deg):
        h, w = self.state.surface_image.shape[:2]

        # -- compute bounding box of rotated+translated content --
        center = (w / 2.0, h / 2.0)
        rot = cv2.getRotationMatrix2D(center, float(rotation_deg), 1.0)
        corners = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.float32)
        rotated = cv2.transform(corners, rot).reshape(-1, 2)
        rotated[:, 0] += float(shift_x_um)
        rotated[:, 1] += float(shift_y_um)

        min_x = int(np.floor(np.min(rotated[:, 0])))
        max_x = int(np.ceil(np.max(rotated[:, 0])))
        min_y = int(np.floor(np.min(rotated[:, 1])))
        max_y = int(np.ceil(np.max(rotated[:, 1])))

        new_w = max_x - min_x
        new_h = max_y - min_y

        # -- adjust matrix: shift origin by (-min_x, -min_y) --
        matrix = rot.copy()
        matrix[0, 2] += float(shift_x_um) - float(min_x)
        matrix[1, 2] += float(shift_y_um) - float(min_y)

        output_shape = (new_h, new_w)
        median_val = int(np.median(self.state.surface_image))

        self.state.surface_image = apply_affine(
            self.state.surface_image, matrix,
            output_shape=output_shape,
        )
        self.state.sample = self.state.surface_image

        self.state.surface_valid_mask = apply_affine(
            self.state.surface_valid_mask.astype(np.uint8) * 255,
            matrix,
            output_shape=output_shape,
            border_value=0,
        ) > 0

        if self.artifact_layer is not None and hasattr(self.artifact_layer, "layer"):
            self.artifact_layer.layer = apply_affine(
                self.artifact_layer.layer, matrix,
                output_shape=output_shape, border_value=0,
            )

        self.state.width_um = float(new_w)
        self.state.height_um = float(new_h)
        self.state.stage_margin_um = max(2000.0, float(max(new_w, new_h)) * 1.5)

        self.state.simulated_sample_shift_x_um += float(shift_x_um)
        self.state.simulated_sample_shift_y_um += float(shift_y_um)
        self.state.simulated_sample_rotation_deg += float(rotation_deg)

    def _extract_surface_crop(self, top_left_x, top_left_y, margin_um, surface_image=None):
        if self.state.ref_template is None:
            return None, 0, 0
        surface_image = self.state.surface_image if surface_image is None else surface_image
        if surface_image is None:
            return None, 0, 0
        template_h, template_w = self.state.ref_template.shape[:2]
        sample_h, sample_w = surface_image.shape[:2]
        x0 = max(0, int(round(top_left_x - margin_um)))
        y0 = max(0, int(round(top_left_y - margin_um)))
        x1 = min(sample_w, int(round(top_left_x + template_w + margin_um)))
        y1 = min(sample_h, int(round(top_left_y + template_h + margin_um)))
        if x1 <= x0 or y1 <= y0:
            return None, x0, y0
        return surface_image[y0:y1, x0:x1], x0, y0

    def _verify_relocation(self, top_left_x, top_left_y, surface_image=None, site_memory=None):
        surface_image = self.state.surface_image if surface_image is None else surface_image
        site_memory = self.state.site_memory if site_memory is None else site_memory
        template_h, template_w = self.state.ref_template.shape[:2]

        # Pad surface with median color if template is larger than surface
        sh, sw = surface_image.shape[:2]
        verify_margin = int(self.state.relocation_verify_half_range_um * 2 + 40)
        pad_x = max(0, template_w + verify_margin - sw)
        pad_y = max(0, template_h + verify_margin - sh)
        if pad_x > 0 or pad_y > 0:
            fill_val = float(np.median(surface_image))
            surface_image = cv2.copyMakeBorder(
                surface_image, 0, pad_y, 0, pad_x,
                borderType=cv2.BORDER_CONSTANT, value=fill_val,
            )

        scaled_top_left_x = top_left_x * verify_scale
        scaled_top_left_y = top_left_y * verify_scale
        verify_match = match_reference_template(
            surface_image,
            self.state.ref_template,
            top_left_x,
            top_left_y,
            half_range=self.state.relocation_verify_half_range_um,
        )
        if verify_match is None:
            verify_match = {
                "score": 0.0,
                "score_gap": 0.0,
                "x": float(top_left_x),
                "y": float(top_left_y),
            }

        landmark_consensus = None
        geometry_check = None
        site_memory = site_memory or {}
        if site_memory.get("highmag_landmarks"):
            crop, crop_x0, crop_y0 = self._extract_surface_crop(
                top_left_x,
                top_left_y,
                self.state.relocation_verify_half_range_um,
                surface_image=surface_image,
            )
            if crop is not None:
                landmark_consensus = estimate_landmark_consensus(
                    site_memory.get("highmag_landmarks", []),
                    crop,
                    search_origin_x_um=float(crop_x0),
                    search_origin_y_um=float(crop_y0),
                    min_score=self.state.relocation_min_match_score,
                    min_gap=self.state.relocation_min_score_gap,
                    max_residual_um=50.0,
                )
                reference_top_left = site_memory.get("reference_top_left") or {}
                reference_tip = site_memory.get("reference_tip") or {}
                tip_x_um = None
                tip_y_um = None
                if (
                    reference_top_left.get("x_um") is not None
                    and reference_top_left.get("y_um") is not None
                    and reference_tip.get("x_um") is not None
                    and reference_tip.get("y_um") is not None
                ):
                    tip_x_um = float(crop_x0 + (float(reference_tip["x_um"]) - float(reference_top_left["x_um"])))
                    tip_y_um = float(crop_y0 + (float(reference_tip["y_um"]) - float(reference_top_left["y_um"])))
                geometry_check = analyze_landmark_geometry(
                    site_memory.get("highmag_landmarks", []),
                    crop,
                    view_origin_x_um=float(crop_x0),
                    view_origin_y_um=float(crop_y0),
                    tip_x_um=tip_x_um,
                    tip_y_um=tip_y_um,
                    min_score=self.state.relocation_min_match_score,
                    min_gap=self.state.relocation_min_score_gap,
                )

        match_ok = (
            verify_match["score"] >= self.state.relocation_min_match_score
            and verify_match.get("score_gap", 0.0) >= self.state.relocation_min_score_gap
        )
        landmark_ok = (
            landmark_consensus is None
            or (
                landmark_consensus.get("support_count", 0) >= self.state.relocation_min_landmark_support
                and landmark_consensus["confidence"] >= self.state.relocation_min_match_score
            )
        )
        geometry_ok = (
            geometry_check is None
            or (
                geometry_check.get("matched_count", 0) >= self.state.relocation_min_landmark_support
                and geometry_check.get("geometry_confidence", 0.0) >= self.state.relocation_min_match_score
            )
        )
        sample_h, sample_w = surface_image.shape[:2]
        x0 = int(np.clip(round(verify_match["x"]), 0, max(sample_w - template_w, 0)))
        y0 = int(np.clip(round(verify_match["y"]), 0, max(sample_h - template_h, 0)))
        candidate_patch = surface_image[y0 : y0 + template_h, x0 : x0 + template_w]
        same_site_probability = score_same_site_probability(
            self.same_site_classifier,
            self.state.ref_template,
            candidate_patch,
        )
        same_site_ok = same_site_probability is None or same_site_probability >= 0.50
        return {
            "verified": bool(match_ok and landmark_ok and geometry_ok and same_site_ok),
            "reference_score": float(verify_match["score"]),
            "reference_score_gap": float(verify_match.get("score_gap", 0.0)),
            "reference_match_x_um": float(verify_match["x"]),
            "reference_match_y_um": float(verify_match["y"]),
            "affine_verify": None,
            "landmark_consensus": landmark_consensus,
            "geometry_check": geometry_check,
            "same_site_probability": same_site_probability,
        }

    def set_step(self, step):
        self.state.current_step = step
        self.log(f"Step size set to {step} um")

    def move_up(self, event):
        new_target_x, new_target_y = self._clamp_to_stage_margin(self.state.target_x, self.state.target_y - self.state.current_step)
        if self.state.pi_mode:
            self._start_smooth_move(new_target_x, new_target_y)
        else:
            self.state.target_x = new_target_x
            self.state.target_y = new_target_y
        self.log(f"Move up {self.state.current_step} um -> target center Y={new_target_y + self.state.fov_height / 2.0:.1f}")

    def move_down(self, event):
        new_target_x, new_target_y = self._clamp_to_stage_margin(self.state.target_x, self.state.target_y + self.state.current_step)
        if self.state.pi_mode:
            self._start_smooth_move(new_target_x, new_target_y)
        else:
            self.state.target_x = new_target_x
            self.state.target_y = new_target_y
        self.log(f"Move down {self.state.current_step} um -> target center Y={new_target_y + self.state.fov_height / 2.0:.1f}")

    def move_left(self, event):
        new_target_x, new_target_y = self._clamp_to_stage_margin(self.state.target_x - self.state.current_step, self.state.target_y)
        if self.state.pi_mode:
            self._start_smooth_move(new_target_x, new_target_y)
        else:
            self.state.target_x = new_target_x
            self.state.target_y = new_target_y
        self.log(f"Move left {self.state.current_step} um -> target center X={new_target_x + self.state.fov_width / 2.0:.1f}")

    def move_right(self, event):
        new_target_x, new_target_y = self._clamp_to_stage_margin(self.state.target_x + self.state.current_step, self.state.target_y)
        if self.state.pi_mode:
            self._start_smooth_move(new_target_x, new_target_y)
        else:
            self.state.target_x = new_target_x
            self.state.target_y = new_target_y
        self.log(f"Move right {self.state.current_step} um -> target center X={new_target_x + self.state.fov_width / 2.0:.1f}")

    def toggle_pause(self, event):
        self.state.paused = not self.state.paused
        self.buttons["pause_motion"].label.set_text("Motion: OFF" if self.state.paused else "Motion: ON")
        if not self.state.paused:
            self.state.target_x, self.state.target_y = self.state.x, self.state.y
        self.update_title()
        self.log(f"Motion pause: {'ON' if self.state.paused else 'OFF'}")

    def stop_motion(self, event):
        had_motion = (
            self.state.auto_scan_active
            or self.state.smooth_move_active
            or not np.isclose(float(self.state.target_x), float(self.state.x))
            or not np.isclose(float(self.state.target_y), float(self.state.y))
        )
        self._cancel_active_motion()
        self._refresh_current_view()
        self.update_title()
        if had_motion:
            self.log("Active motion cancelled; destination snapped to the current position.")
        else:
            self.log("No active motion to stop.")

    def jump_to_destination(self, event):
        if np.isclose(float(self.state.target_x), float(self.state.x)) and np.isclose(float(self.state.target_y), float(self.state.y)):
            self.log("Already at the current destination.")
            return
        delta_x, delta_y = self._jump_view_to_target()
        self.log(
            f"Jumped immediately to destination by dX={delta_x:+.1f} um, dY={delta_y:+.1f} um"
        )

    def toggle_pi(self, event):
        self.state.pi_mode = not self.state.pi_mode
        # Switching compensation mode should not itself trigger a motion jump.
        # Snap the active target to the current position so any new motion only
        # begins after the next explicit user command.
        self.state.smooth_move_active = False
        self.state.smooth_move_target_x = float(self.state.x)
        self.state.smooth_move_target_y = float(self.state.y)
        self.state.target_x = float(self.state.x)
        self.state.target_y = float(self.state.y)
        self.data.clear()
        self.stage.clear_history()
        if self.state.pi_mode:
            self.stage.reset(self.state.x, self.state.y)
            self.stage.cmd_x = float(self.state.x)
            self.stage.cmd_y = float(self.state.y)
            self.buttons["pi"].label.set_text("PI Compensation: ON")
            self.log("PI mode activated from current position; previous motion history cleared")
        else:
            self.buttons["pi"].label.set_text("PI Compensation Mode")
            self.log("PI mode deactivated; previous motion history cleared")
        self.update_title()

    def zoom_in(self, event):
        current_index = self._get_zoom_level_index(self.state.current_zoom_level)
        levels = list(self.state.zoom_levels)
        self._begin_quantized_zoom(levels[min(current_index + 1, len(levels) - 1)])

    def zoom_out(self, event):
        current_index = self._get_zoom_level_index(self.state.current_zoom_level)
        levels = list(self.state.zoom_levels)
        self._begin_quantized_zoom(levels[max(current_index - 1, 0)])

    def move_z_up(self, event):
        self.state.z_stage_position_um = self._clamp_z_stage(self.state.z_stage_position_um + self.state.z_stage_step_um)
        self._refresh_current_view()
        self.log(
            f"Sample Z stage moved to {self.state.z_stage_position_um:+.1f} um "
            f"(probe gap {self.state.get_probe_sample_gap_um():+.1f} um, focus offset {self.state.get_focus_offset_um():+.1f} um)"
        )

    def move_z_down(self, event):
        self.state.z_stage_position_um = self._clamp_z_stage(self.state.z_stage_position_um - self.state.z_stage_step_um)
        self._refresh_current_view()
        self.log(
            f"Sample Z stage moved to {self.state.z_stage_position_um:+.1f} um "
            f"(probe gap {self.state.get_probe_sample_gap_um():+.1f} um, focus offset {self.state.get_focus_offset_um():+.1f} um)"
        )

    def reset_focus(self, event):
        self.state.z_stage_position_um = float(
            self.state.get_effective_camera_stage_position_um() - self.state.focus_z_um
        )
        self._refresh_current_view()
        self.log(
            f"Sample Z stage returned to best focus at {self.state.z_stage_position_um:.1f} um "
            f"for camera/cantilever stage {self.state.get_effective_camera_stage_position_um():.1f} um"
        )

    def _get_active_dof_value(self):
        if self.state.manual_dof_camera_um is not None:
            return float(self.state.manual_dof_camera_um)
        return float(self.state.last_dof_camera_um)

    def increase_dof(self, event):
        next_dof = max(0.1, self._get_active_dof_value() + float(self.state.dof_step_um))
        self.state.manual_dof_camera_um = next_dof
        self._refresh_current_view()
        self.log(f"Manual DOF set to {next_dof:.2f} um")

    def decrease_dof(self, event):
        next_dof = max(0.1, self._get_active_dof_value() - float(self.state.dof_step_um))
        self.state.manual_dof_camera_um = next_dof
        self._refresh_current_view()
        self.log(f"Manual DOF set to {next_dof:.2f} um")

    def reset_dof_auto(self, event):
        self.state.manual_dof_camera_um = None
        self._refresh_current_view()
        self.log("DOF control returned to automatic optics mode")

    def show_tip_coord(self, event):
        tip_x, tip_y = self.get_tip()
        if self.state.origin_defined:
            self.log(
                f"Tip position: X={tip_x:.2f} um, Y={tip_y:.2f} um "
                f"(relative to {self.state.origin_label}: "
                f"dX={tip_x - self.state.origin_x:+.2f} um, dY={tip_y - self.state.origin_y:+.2f} um)"
            )
        else:
            self.log(f"Tip position: X={tip_x:.2f} um, Y={tip_y:.2f} um")

    def cycle_scale_bar_length(self, event):
        choices = list(self.SCALE_BAR_CHOICES_UM)
        try:
            current_index = choices.index(float(self.state.scale_bar_total_um))
        except ValueError:
            current_index = 0
        next_value = choices[(current_index + 1) % len(choices)]
        self.state.scale_bar_total_um = float(next_value)
        if "scale_bar" in self.buttons:
            self.buttons["scale_bar"].label.set_text(f"Scale Bar: {int(next_value)} um")
        self.fig.canvas.draw_idle()
        self.log(f"Viewport scale bar length set to {int(next_value)} um")

    def auto_origin_unsupervised(self, event):
        current_view = self.state.current_camera_view if self.state.current_camera_view is not None else self.state.current_fov_raw
        if current_view is None or current_view.size == 0:
            self.log("No current viewport image available for unsupervised origin detection.")
            return

        landmarks = extract_landmarks(
            current_view,
            base_x_um=float(self.state.x),
            base_y_um=float(self.state.y),
            patch_half=max(18, int(self.state.origin_template_half_size // 2)),
            max_landmarks=5,
            min_distance_px=20,
        )
        if not landmarks:
            self.log("Unsupervised origin detection could not find a distinctive pattern in the current viewport.")
            return

        best_landmark = max(landmarks, key=lambda item: item["score"])
        abs_x = float(best_landmark["abs_x_um"])
        abs_y = float(best_landmark["abs_y_um"])
        self.set_origin(abs_x, abs_y, label="Auto Origin")
        self.log(
            f"Unsupervised origin selected at X={abs_x:.1f} um, Y={abs_y:.1f} um "
            f"(distinctiveness score {best_landmark['score']:.3f}, "
            f"{len(landmarks)} candidate landmarks stored in the current view)"
        )

    def ml_find_origin(self, event):
        if not self._begin_action("ml_find_origin", "ML origin search is already running"):
            return
        try:
            if self.state.origin_template is None:
                self.log("Set an origin in the viewport first so the supervised ML search has a labeled pattern.")
                return

            template = self.state.origin_template
            surface = self.state.surface_image
            if template.ndim == 3:
                template = template[..., 0]
            if surface.ndim == 3:
                surface = surface[..., 0]

            candidates = match_template_candidates(surface, template, top_k=3)
            if not candidates:
                self.log("ML origin search could not find the labeled pattern.")
                return
            best = candidates[0]
            score_gap = float(best["score"] - candidates[1]["score"]) if len(candidates) > 1 else float(best["score"])
            if best["score"] < self.state.relocation_min_match_score or score_gap < self.state.relocation_min_score_gap:
                self.log(
                    f"ML origin search confidence too low: score={best['score']:.3f}, gap={score_gap:.3f}"
                )
                return

            tpl_h, tpl_w = template.shape[:2]
            origin_x = float(best["x"] + tpl_w / 2.0)
            origin_y = float(best["y"] + tpl_h / 2.0)
            self.state.origin_x = origin_x
            self.state.origin_y = origin_y
            self.state.origin_defined = True
            center_target_x, center_target_y = self._clamp_to_stage_margin(
                origin_x - self.state.fov_width / 2.0,
                origin_y - self.state.fov_height / 2.0,
            )
            if self.state.pi_mode:
                self._start_smooth_move(center_target_x, center_target_y)
            else:
                self.state.target_x = center_target_x
                self.state.target_y = center_target_y
            if self.status_callback is not None:
                self.status_callback()
            self.fig.canvas.draw_idle()
            self.log(
                f"ML origin recognized {self.state.origin_label} at "
                f"X={origin_x:.1f} um, Y={origin_y:.1f} um "
                f"(confidence {best['score']:.3f}, gap {score_gap:.3f})"
            )
        finally:
            self._end_action("ml_find_origin")

    def set_tilt(self, event):
        if self.state.tilting:
            self.log("Tilt adjustment already open")
            return
        self.state.tilting = True
        root = tk.Tk()
        try:
            root.withdraw()
            angle = simpledialog.askfloat(
                "Stage Surface Tilt",
                "Enter stage surface rotation angle (degrees):",
                initialvalue=float(self.state.surface_tilt_angle),
                minvalue=-10,
                maxvalue=10,
            )
            if angle is not None:
                self.state.surface_tilt_angle = float(angle)
                self._refresh_current_view()
                self.log(f"Stage surface tilt set to {self.state.surface_tilt_angle:.1f} degrees")
        finally:
            root.destroy()
            self.state.tilting = False
            self.update_title()

    def clear_trails(self, event):
        self.data.clear()
        self.stage.clear_history()
        self.log("Trails cleared")

    def start_auto_scan(self, event):
        if self.state.auto_scan_active:
            self.log("Auto scan is already running")
            return
        self.state.auto_scan_start_x = self.state.target_x
        self.state.auto_scan_end_x = self.state.target_x + 1000
        self.state.auto_scan_step = 0
        self.state.auto_scan_direction = 1
        self.state.auto_scan_active = True
        self.log(f"Auto scan: {self.state.auto_scan_start_x:.0f} -> {self.state.auto_scan_end_x:.0f}")

    def show_hysteresis_curve(self, event):
        if not self._begin_action("show_hysteresis", "Hysteresis plot is already opening"):
            return
        if self.state.pi_mode and len(self.stage.history_cmd) > 0:
            try:
                self.stage.plot_hysteresis("Hysteresis (PI Model)")
            finally:
                self._end_action("show_hysteresis")
        else:
            self._end_action("show_hysteresis")
            self.log("Please enable PI mode and move the tip first")

    def start_data_collection(self, event):
        if not self._begin_action("data_collection", "Data collection is already running"):
            return
        self.log("=== Starting data collection ===")
        try:
            from data_collection import DataCollector

            collector = DataCollector(self.stage, self.state, self.get_tip)
            configs = [
                {"start_x": 200, "end_x": 600, "steps": 80, "speed_factor": 1, "label": "Range_400um_Slow"},
                {"start_x": 200, "end_x": 600, "steps": 80, "speed_factor": 2, "label": "Range_400um_Fast"},
                {"start_x": 200, "end_x": 1000, "steps": 120, "speed_factor": 1, "label": "Range_800um_Slow"},
                {"start_x": 200, "end_x": 1000, "steps": 120, "speed_factor": 2, "label": "Range_800um_Fast"},
                {"start_x": 200, "end_x": 1400, "steps": 160, "speed_factor": 1, "label": "Range_1200um_Slow"},
                {"start_x": 200, "end_x": 1400, "steps": 160, "speed_factor": 2, "label": "Range_1200um_Fast"},
            ]
            collector.collect_multi_configurations(configs, base_wait_time=0.05)
            filename = collector.save_all_to_csv()
            collector.plot_collected_data()
            self.log(f"Data collection completed. File saved: {filename}")
        except Exception as e:
            self.log(f"Data collection failed: {e}")
        finally:
            self._end_action("data_collection")

    def save_reference(self, event):
        """Save the current site memory, reference view, and multi-landmark relocation data."""
        if not self._begin_action("save_reference", "Reference capture is already running"):
            return
        try:
            if self.img is None:
                self.log("Cannot get current image")
                return

            fov = self.state.current_camera_view if self.state.current_camera_view is not None else self.state.current_fov_raw
            if fov is None:
                fov = self.img.get_array()
            self.state.ref_template = fov.copy()
            self.state.ref_artefacts = []
            self.state.ref_x = self.state.x
            self.state.ref_y = self.state.y
            self.state.ai_desired_history_x = [self.state.x, self.state.x]
            self.state.ai_desired_history_y = [self.state.y, self.state.y]
            self.state.site_memory = build_site_memory(self.state, stage_history=self.stage.history_cmd)
            self.state.ref_artefacts = list(self.state.site_memory.get("highmag_landmarks", []))
            output_dir = self._persist_site_memory(self.state.site_memory)
            self.log(f"Reference position saved: ({self.state.ref_x:.1f}, {self.state.ref_y:.1f})")
            self.log(f"Saved reference template size: {self.state.ref_template.shape[1]} x {self.state.ref_template.shape[0]} px")
            self.log(
                "Structured site memory saved with "
                f"{len(self.state.site_memory.get('lowmag_landmarks', []))} low-mag landmarks and "
                f"{len(self.state.site_memory.get('highmag_landmarks', []))} high-mag landmarks"
            )
            self.log(f"Site memory folder: {output_dir}")
        finally:
            self._end_action("save_reference")

    def research_patterns(self, event):
        site_memory = self.state.site_memory or {}
        current_view = self.state.current_camera_view if self.state.current_camera_view is not None else self.state.current_fov_raw
        if not site_memory.get("highmag_landmarks"):
            self.log("No saved site memory landmarks available. Save site memory first.")
            return
        if current_view is None or current_view.size == 0:
            self.log("No current camera view available for pattern re-search.")
            return
        tip_x, tip_y = self.get_tip()
        report = analyze_landmark_geometry(
            site_memory.get("highmag_landmarks", []),
            current_view,
            view_origin_x_um=float(self.state.x),
            view_origin_y_um=float(self.state.y),
            tip_x_um=float(tip_x),
            tip_y_um=float(tip_y),
            min_score=max(0.38, float(self.state.relocation_min_match_score) - 0.04),
            min_gap=max(0.01, float(self.state.relocation_min_score_gap) - 0.01),
        )
        self.state.hud_landmark_report = report
        self.state.hud_landmark_matches = list(report.get("matches", []))
        if self.status_callback is not None:
            self.status_callback()
        self.fig.canvas.draw_idle()
        self.log(
            "Pattern re-search: "
            f"matched={report.get('matched_count', 0)}, "
            f"pair_error={0.0 if report.get('mean_pair_error_um') is None else report['mean_pair_error_um']:.1f} um, "
            f"tip_error={0.0 if report.get('mean_distance_error_um') is None else report['mean_distance_error_um']:.1f} um, "
            f"geometry={report.get('geometry_confidence', 0.0):.3f}"
        )

    def remove_sample(self, event):
        """Simulate sample removal by translating and rotating the sample relative to the stage."""
        self.state.sample_removed = True
        dx = random.uniform(-500, 500)
        dy = random.uniform(-500, 500)
        rotation_deg = random.uniform(-8.0, 8.0)
        tilt_deg = random.uniform(-10.0, 10.0)
        self._apply_simulated_sample_remount(dx, dy, rotation_deg)
        self.state.surface_tilt_angle = float(tilt_deg)
        self.state.simulated_sample_tilt_deg = float(tilt_deg)
        self.state.target_x = float(self.state.x)
        self.state.target_y = float(self.state.y)
        if self.state.pi_mode:
            self.stage.reset(self.state.x, self.state.y)
            self.stage.cmd_x = self.state.x
            self.stage.cmd_y = self.state.y
        self.state.origin_defined = False
        self._refresh_current_view()
        self.log(
            "Sample removal simulation: sample remounted relative to the stage by "
            f"dX={dx:+.1f} um, dY={dy:+.1f} um, dTheta={rotation_deg:+.2f} deg, "
            f"dTilt={tilt_deg:+.2f} deg"
        )
        if self.state.origin_template is not None:
            self.log("Stored origin template kept for later re-identification after remount.")

    def relocate(self, event):
        """Relocate using coarse-to-fine matching, landmark consensus, and final verification."""
        if not self._begin_action("relocate", "Relocation is already running"):
            return
        try:
            if self.state.site_memory is None:
                self._try_load_latest_site_memory()
            if self.state.site_memory is not None and self.state.ref_template is None:
                self._activate_site_memory(self.state.site_memory, source_dir=self.state.last_saved_site_dir)
            if self.state.ref_template is None:
                self.log("Please save reference position first")
                return

            self.log("Starting coarse-to-fine relocation...")
            site_memory = self.state.site_memory or {}
            coarse_target_x = float(self.state.x)
            coarse_target_y = float(self.state.y)
            coarse_result = None
            affine_report = None
            fine_search_surface = self.state.surface_image
            fine_search_target_x = float(self.state.x)
            fine_search_target_y = float(self.state.y)
            fine_to_current_matrix = None

            if site_memory.get("reference_top_left"):
                coarse_target_x = float(site_memory["reference_top_left"]["x_um"])
                coarse_target_y = float(site_memory["reference_top_left"]["y_um"])
                fine_search_target_x = coarse_target_x
                fine_search_target_y = coarse_target_y

            affine_report = self._estimate_coarse_affine_transform()
            has_rotation = (
                affine_report is not None
                and abs(affine_report.get("rotation_deg", 0.0)) > 1.0
            )
            use_de_rotation = (
                has_rotation
                and affine_report["confidence"] >= max(self.state.relocation_min_affine_confidence, 0.30)
            )

            if use_de_rotation:
                # 有旋转且置信度高 → de-rotate surface 后细搜索
                fine_to_current_matrix = affine_report["full_matrix"]
                inv_matrix = invert_affine(fine_to_current_matrix)
                ref_x = float(self.state.ref_x)
                ref_y = float(self.state.ref_y)
                template_h, template_w = self.state.ref_template.shape[:2]
                search_margin = (
                    self.state.relocation_fine_half_range_um * 4.0
                    + max(float(template_w), float(template_h))
                )

                # Handle negative reference coordinates: the reference may be
                # outside the warped image bounds.  Expand canvas and shift so
                # the reference lands in a searchable positive region.
                offset_x = (-ref_x + search_margin) if ref_x < 0 else 0.0
                offset_y = (-ref_y + search_margin) if ref_y < 0 else 0.0
                surface_h, surface_w = self.state.surface_image.shape[:2]

                if offset_x > 0 or offset_y > 0:
                    new_w = int(surface_w + offset_x)
                    new_h = int(surface_h + offset_y)
                    adjusted_inv = inv_matrix.copy()
                    adjusted_inv[0, 2] = (
                        inv_matrix[0, 2]
                        - (inv_matrix[0, 0] * offset_x + inv_matrix[0, 1] * offset_y)
                    )
                    adjusted_inv[1, 2] = (
                        inv_matrix[1, 2]
                        - (inv_matrix[1, 0] * offset_x + inv_matrix[1, 1] * offset_y)
                    )
                    fine_search_surface = apply_affine(
                        self.state.surface_image, adjusted_inv,
                        output_shape=(new_h, new_w),
                    )
                else:
                    fine_search_surface = apply_affine(
                        self.state.surface_image, inv_matrix,
                        output_shape=(surface_h, surface_w),
                    )
                fine_search_target_x = ref_x + offset_x
                fine_search_target_y = ref_y + offset_y
                self.log(
                    f"Fine search surface: shape={fine_search_surface.shape}, "
                    f"min={np.min(fine_search_surface):.0f}, max={np.max(fine_search_surface):.0f}, "
                    f"mean={np.mean(fine_search_surface):.1f}, "
                    f"offset=({offset_x:.0f},{offset_y:.0f})"
                )
                self.log(
                    f"Invert affine matrix: "
                    f"[{inv_matrix[0,0]:.4f},{inv_matrix[0,1]:.4f},{inv_matrix[0,2]:.1f}; "
                    f"{inv_matrix[1,0]:.4f},{inv_matrix[1,1]:.4f},{inv_matrix[1,2]:.1f}]"
                )
                coarse_target_x, coarse_target_y = transform_point(
                    fine_to_current_matrix,
                    self.state.ref_x,
                    self.state.ref_y,
                )
                self.log(
                    (
                        "[ML mode] Coarse localization: "
                        if affine_report.get("ml_remount_prediction")
                        else "Coarse affine recovery: "
                    )
                    + f"dTheta={affine_report['rotation_deg']:+.2f} deg, "
                    + f"inliers={affine_report['inlier_count']}/{affine_report['match_count']}, "
                    + f"confidence={affine_report['confidence']:.3f}"
                )
                predicted_transform = affine_report.get("predicted_remount_transform")
                if predicted_transform is not None:
                    self.log(
                        "Phase 2 remount predictor: "
                        f"dX={predicted_transform['dx_um']:+.1f} um, "
                        f"dY={predicted_transform['dy_um']:+.1f} um, "
                        f"dTheta={predicted_transform['dtheta_deg']:+.2f} deg"
                    )
                ml_remount = affine_report.get("ml_remount_prediction")
                if ml_remount is not None:
                    self.log(
                        "[ML mode] 5w model prediction: "
                        f"dX={ml_remount['dx_um']:+.1f} um, "
                        f"dY={ml_remount['dy_um']:+.1f} um, "
                        f"dTheta={ml_remount['dtheta_deg']:+.2f} deg"
                    )
                retrieval_candidates = affine_report.get("retrieval_candidates") or []
                if retrieval_candidates:
                    best_candidate = retrieval_candidates[0]
                    self.log(
                        "Phase 2 low-mag retrieval best match: "
                        f"{best_candidate.get('site_id', 'unknown')} "
                        f"(distance {best_candidate['distance']:.3f})"
                    )
            elif affine_report is not None:
                # NCC 粗定位 (无旋转) → 用匹配位置做细搜索起点
                dx_total = float(affine_report["translation_px"][0])
                dy_total = float(affine_report["translation_px"][1])
                fine_search_target_x = float(self.state.ref_x + dx_total)
                fine_search_target_y = float(self.state.ref_y + dy_total)
                self.log(
                    "Coarse NCC localization: "
                    f"dX={dx_total:+.1f} um, dY={dy_total:+.1f} um, "
                    f"confidence={affine_report['confidence']:.3f}"
                )

            if fine_to_current_matrix is None and site_memory.get("lowmag_landmarks"):
                overview = build_overview(self.state.surface_image)
                if overview is not None:
                    coarse_result = estimate_landmark_consensus(
                        site_memory.get("lowmag_landmarks", []),
                        overview["image"],
                        scale_x_um_per_px=overview["scale_x_um_per_px"],
                        scale_y_um_per_px=overview["scale_y_um_per_px"],
                        min_score=self.state.relocation_min_match_score,
                        min_gap=self.state.relocation_min_score_gap,
                        max_residual_um=120.0,
                    )
                if coarse_result is not None and coarse_result["support_count"] >= self.state.relocation_min_landmark_support:
                    coarse_target_x = float(self.state.ref_x + coarse_result["offset_x_um"])
                    coarse_target_y = float(self.state.ref_y + coarse_result["offset_y_um"])
                    fine_search_target_x = coarse_target_x
                    fine_search_target_y = coarse_target_y
                    self.log(
                        "Coarse low-mag localization: "
                        f"dX={coarse_result['offset_x_um']:+.1f} um, "
                        f"dY={coarse_result['offset_y_um']:+.1f} um, "
                        f"support={coarse_result['support_count']}, "
                        f"confidence={coarse_result['confidence']:.3f}"
                    )
                else:
                    self.log("Coarse low-mag localization was ambiguous; falling back to the last known reference region.")

            desired_x = float(fine_search_target_x)
            desired_y = float(fine_search_target_y)
            # If the reference template is larger than the search surface
            # (can happen after remount cropping), pad the surface instead of
            # scaling the template — this preserves coordinate accuracy.
            template_h, template_w = self.state.ref_template.shape[:2]
            surface_h, surface_w = fine_search_surface.shape[:2]
            max_range = self.state.relocation_fine_half_range_um * 4.0
            pad_x = int(max(0, template_w + 2 * max_range + 40 - surface_w))
            pad_y = int(max(0, template_h + 2 * max_range + 40 - surface_h))
            if pad_x > 0 or pad_y > 0:
                fill_val = float(np.median(fine_search_surface))
                fine_search_surface = cv2.copyMakeBorder(
                    fine_search_surface, 0, pad_y, 0, pad_x,
                    borderType=cv2.BORDER_CONSTANT, value=fill_val,
                )
                self.log(
                    f"Padded search surface by (+{pad_x}, +{pad_y}) px "
                    f"to fit {template_w}x{template_h} template"
                )
            fine_match = None
            search_half_range = self.state.relocation_fine_half_range_um
            max_search_half_range = self.state.relocation_fine_half_range_um * 4.0
            self.log(
                f"Fine search starting at X={desired_x:.1f} um, Y={desired_y:.1f} um, "
                f"half_range={search_half_range:.0f} um"
            )
            while True:
                for iteration in range(int(self.state.relocation_max_iterations)):
                    fine_match = match_reference_template(
                        fine_search_surface,
                        self.state.ref_template,
                        desired_x,
                        desired_y,
                        half_range=search_half_range,
                    )
                    if fine_match is not None:
                        desired_x = float(fine_match["x"])
                        desired_y = float(fine_match["y"])
                        self.log(
                            f"Fine relocation pass {iteration + 1}: "
                            f"X={desired_x:.1f} um, Y={desired_y:.1f} um, "
                            f"score={fine_match['score']:.3f}, gap={fine_match.get('score_gap', 0.0):.3f}"
                        )
                        break
                if fine_match is not None:
                    if (
                        fine_match is not None
                        and fine_match["score"] >= self.state.relocation_min_match_score
                        and fine_match.get("score_gap", 0.0) >= self.state.relocation_min_score_gap
                    ):
                        break
                    self.log("Fine match found but quality insufficient; expanding range")
                    fine_match = None
                search_half_range *= 2.0
                if search_half_range > max_search_half_range:
                    self.log(
                        f"Reference template could not be matched "
                        f"(tried up to {max_search_half_range:.0f} um half-range)"
                    )
                    return
                self.log(
                    f"No match at current range, expanding to half_range={search_half_range:.0f} um"
                )

            if fine_match is None:
                self.log("Fine relocation did not produce a usable reference match.")
                return

            if fine_to_current_matrix is not None:
                desired_current_x, desired_current_y = transform_point(
                    fine_to_current_matrix,
                    desired_x,
                    desired_y,
                )
            else:
                desired_current_x = desired_x
                desired_current_y = desired_y

            verification = self._verify_relocation(
                desired_x,
                desired_y,
                surface_image=fine_search_surface,
                site_memory=site_memory,
            )
            self.state.last_relocation_report = {
                "affine": affine_report,
                "coarse": coarse_result,
                "fine_affine": fine_affine,
                "fine": fine_match,
                "predicted_current_top_left": {"x_um": float(desired_current_x), "y_um": float(desired_current_y)},
                "verification": verification,
            }
            if not verification.get("verified", False):
                self.log(
                    "Relocation verification failed: "
                    f"reference score={verification.get('reference_score', 0.0):.3f}, "
                    f"gap={verification.get('reference_score_gap', 0.0):.3f}"
                )
                landmark_consensus = verification.get("landmark_consensus")
                if landmark_consensus is not None:
                    self.log(
                        "High-mag landmark verification: "
                        f"support={landmark_consensus['support_count']}, "
                        f"confidence={landmark_consensus['confidence']:.3f}"
                    )
                geometry_check = verification.get("geometry_check")
                if geometry_check is not None:
                    self.log(
                        "HUD landmark geometry check: "
                        f"matched={geometry_check.get('matched_count', 0)}, "
                        f"pair_error={0.0 if geometry_check.get('mean_pair_error_um') is None else geometry_check['mean_pair_error_um']:.1f} um, "
                        f"tip_error={0.0 if geometry_check.get('mean_distance_error_um') is None else geometry_check['mean_distance_error_um']:.1f} um, "
                        f"confidence={geometry_check.get('geometry_confidence', 0.0):.3f}"
                    )
                affine_verify = verification.get("affine_verify")
                if affine_verify is not None:
                    self.log(
                        "Affine verification: "
                        f"dTheta={affine_verify['rotation_deg']:+.2f} deg, "
                        f"inliers={affine_verify['inlier_count']}/{affine_verify['match_count']}, "
                        f"confidence={affine_verify['confidence']:.3f}"
                    )
                if verification.get("same_site_probability") is not None:
                    self.log(
                        f"Phase 2 same-site probability: {verification['same_site_probability']:.3f}"
                    )
                return

            if self.ai_compensator is not None and self.ai_mode:
                cmd_x = self._predict_compensated_axis(desired_current_x, "x")
                cmd_y = self._predict_compensated_axis(desired_current_y, "y")
                self.log("AI compensation applied to relocation command")
            else:
                cmd_x = float(desired_current_x)
                cmd_y = float(desired_current_y)

            cmd_x, cmd_y = self._clamp_to_stage_margin(cmd_x, cmd_y)

            self._start_smooth_move(cmd_x, cmd_y)
            self.state.sample_removed = False
            self.log(
                "Relocation verified and accepted: "
                f"reference score={verification['reference_score']:.3f}, "
                f"gap={verification['reference_score_gap']:.3f}"
            )
            landmark_consensus = verification.get("landmark_consensus")
            if landmark_consensus is not None:
                self.log(
                    "High-mag landmark verification: "
                    f"support={landmark_consensus['support_count']}, "
                    f"confidence={landmark_consensus['confidence']:.3f}"
                )
            geometry_check = verification.get("geometry_check")
            if geometry_check is not None:
                self.log(
                    "HUD landmark geometry check: "
                    f"matched={geometry_check.get('matched_count', 0)}, "
                    f"pair_error={0.0 if geometry_check.get('mean_pair_error_um') is None else geometry_check['mean_pair_error_um']:.1f} um, "
                    f"tip_error={0.0 if geometry_check.get('mean_distance_error_um') is None else geometry_check['mean_distance_error_um']:.1f} um, "
                    f"confidence={geometry_check.get('geometry_confidence', 0.0):.3f}"
                )
            affine_verify = verification.get("affine_verify")
            if affine_verify is not None:
                self.log(
                    "Affine verification: "
                    f"dTheta={affine_verify['rotation_deg']:+.2f} deg, "
                    f"inliers={affine_verify['inlier_count']}/{affine_verify['match_count']}, "
                    f"confidence={affine_verify['confidence']:.3f}"
                )
            if verification.get("same_site_probability") is not None:
                self.log(
                    f"Phase 2 same-site probability: {verification['same_site_probability']:.3f}"
                )
            self.log(f"Relocation complete, moved to ({cmd_x:.1f}, {cmd_y:.1f})")
        finally:
            self._end_action("relocate")

    # ------------------------------------------------------------------
    # ML Recognition Helpers
    # ------------------------------------------------------------------
    def _ml_recognize_pattern(self, search_image, ref_template, center_x=None, center_y=None, half_range_um=None):
        """用 ML 特征匹配在 search_image 中找到 ref_template 的最佳匹配位置。

        Returns:
            (x: int, y: int, score: float) 或 None
        """
        if not self._use_ml() or self.ml_pattern_matcher is None:
            return None
        try:
            # 如果提供了搜索范围，裁剪搜索图像以加速
            if center_x is not None and center_y is not None and half_range_um is not None:
                tpl_h, tpl_w = ref_template.shape[:2]
                sch_h, sch_w = search_image.shape[:2]
                x0 = int(max(0, center_x - half_range_um))
                y0 = int(max(0, center_y - half_range_um))
                x1 = int(min(sch_w, center_x + half_range_um + tpl_w))
                y1 = int(min(sch_h, center_y + half_range_um + tpl_h))
                if x1 <= x0 or y1 <= y0:
                    return None
                cropped = search_image[y0:y1, x0:x1]
                matches = self.ml_pattern_matcher.match(
                    ref_template, cropped, top_k=1, stride_frac=0.25, min_score=0.35
                )
                if matches:
                    best = matches[0]
                    return (best["x"] + x0, best["y"] + y0, best["score"])
                return None

            matches = self.ml_pattern_matcher.match(
                ref_template, search_image, top_k=1, stride_frac=0.30, min_score=0.35
            )
            if matches:
                best = matches[0]
                return (best["x"], best["y"], best["score"])
        except Exception as e:
            self.log(f"ML pattern recognition failed: {e}")
        return None

    def _ml_predict_remount(self, ref_overview_image, cur_overview_image,
                            scale_x_um_per_px=1.0, scale_y_um_per_px=1.0):
        """用 ML 回归器预测 remount 变换 (dx, dy, dtheta)。

        Returns:
            dict: {"dx_um": float, "dy_um": float, "dtheta_deg": float} 或 None
        """
        if self.deep_regressor is None:
            return None
        try:
            if self.deep_regressor_is_5w:
                result = self.deep_regressor.predict_um(
                    ref_overview_image, cur_overview_image,
                    scale_x_um_per_px=scale_x_um_per_px,
                    scale_y_um_per_px=scale_y_um_per_px,
                )
            else:
                result = self.deep_regressor.predict(ref_overview_image, cur_overview_image)
            return result
        except Exception as e:
            self.log(f"ML remount prediction failed: {e}")
        return None

    def _ml_verify_same_site(self, ref_image, candidate_image):
        """用 ML 分类器验证两张图是否同一 site。

        Returns:
            (is_same: bool, probability: float) 或 (False, None)
        """
        if not self._use_ml() or self.deep_classifier is None:
            return False, None
        try:
            return self.deep_classifier.classify(ref_image, candidate_image, threshold=0.5)
        except Exception as e:
            self.log(f"ML site verification failed: {e}")
        return False, None

    # ------------------------------------------------------------------
    # Page 2 – AI Recall & Recover (PPT slide 2)
    # ------------------------------------------------------------------
    def ai_recall_and_recover(self, event):
        """One-button AI relocation: load last site_memory, restore zoom,
        run coarse-to-fine AI recognition, move cantilever, verify.
        On verification failure, enter click-to-move correction mode."""
        if not self._begin_action("ai_recall", "AI recall is already running"):
            return
        try:
            self.log("===== AI Recall & Recover =====")
            # 1. Machine recall the last measurement region
            if self.state.site_memory is None:
                self._try_load_latest_site_memory()
            if self.state.site_memory is not None and self.state.ref_template is None:
                self._activate_site_memory(self.state.site_memory, source_dir=self.state.last_saved_site_dir)
            if self.state.ref_template is None:
                self.log("No saved site memory found. Save a reference region first (1. Save Region).")
                return

            # Wait for zoom restoration to complete before recognition
            self._wait_for_zoom_complete()

            site_memory = self.state.site_memory or {}
            self.log("Site memory loaded. Running AI recognition...")

            # 2. AI recognize the current surrounding pattern:
            #    estimate rotation angle and distance from origin
            affine_report = self._estimate_coarse_affine_transform()
            if affine_report is not None and affine_report["confidence"] >= self.state.relocation_min_affine_confidence:
                self.log(
                    "AI pattern recognition: "
                    f"rotation={affine_report['rotation_deg']:+.2f} deg, "
                    f"inliers={affine_report['inlier_count']}/{affine_report['match_count']}, "
                    f"confidence={affine_report['confidence']:.3f}"
                )
            else:
                self.log(
                    "AI pattern recognition confidence low "
                    f"({affine_report['confidence']:.3f if affine_report else 'N/A'}), "
                    "proceeding with landmark-only fallback."
                )

            # 3. Run the existing coarse-to-fine relocation (without moving yet)
            #    We reuse the relocate logic but intercept the verification result
            self.log("Computing relocation offset...")
            coarse_target_x = float(self.state.x)
            coarse_target_y = float(self.state.y)
            coarse_result = None
            fine_search_surface = self.state.surface_image
            fine_search_target_x = float(self.state.x)
            fine_search_target_y = float(self.state.y)
            fine_to_current_matrix = None

            if site_memory.get("reference_top_left"):
                coarse_target_x = float(site_memory["reference_top_left"]["x_um"])
                coarse_target_y = float(site_memory["reference_top_left"]["y_um"])
                fine_search_target_x = coarse_target_x
                fine_search_target_y = coarse_target_y

            if affine_report is not None and affine_report["confidence"] >= self.state.relocation_min_affine_confidence:
                fine_to_current_matrix = affine_report["full_matrix"]
                fine_search_surface = apply_affine(
                    self.state.surface_image,
                    invert_affine(fine_to_current_matrix),
                    output_shape=self.state.surface_image.shape[:2],
                )
                fine_search_target_x = float(self.state.ref_x)
                fine_search_target_y = float(self.state.ref_y)
                coarse_target_x, coarse_target_y = transform_point(
                    fine_to_current_matrix, self.state.ref_x, self.state.ref_y,
                )

            if fine_to_current_matrix is None and site_memory.get("lowmag_landmarks"):
                overview = build_overview(self.state.surface_image)
                if overview is not None:
                    coarse_result = estimate_landmark_consensus(
                        site_memory.get("lowmag_landmarks", []),
                        overview["image"],
                        scale_x_um_per_px=overview["scale_x_um_per_px"],
                        scale_y_um_per_px=overview["scale_y_um_per_px"],
                        min_score=max(0.30, float(self.state.relocation_min_match_score) - 0.12),
                        min_gap=max(0.01, float(self.state.relocation_min_score_gap) - 0.01),
                        max_residual_um=100.0,
                    )
                    if coarse_result is not None:
                        fine_search_target_x = float(coarse_result["offset_x_um"] + self.state.ref_x)
                        fine_search_target_y = float(coarse_result["offset_y_um"] + self.state.ref_y)

            # ── ML Remount fallback for coarse positioning ──
            ml_remount = None
            if fine_to_current_matrix is None and coarse_result is None:
                cur_overview = build_overview(self.state.surface_image)
                if cur_overview is not None and site_memory.get("overview", {}).get("image") is not None:
                    ml_remount = self._ml_predict_remount(
                        site_memory.get("overview", {}).get("image"),
                        cur_overview.get("image"),
                        scale_x_um_per_px=cur_overview.get("scale_x_um_per_px", 1.0),
                        scale_y_um_per_px=cur_overview.get("scale_y_um_per_px", 1.0),
                    )
                    if ml_remount is not None:
                        fine_search_target_x = float(self.state.ref_x + ml_remount["dx_um"])
                        fine_search_target_y = float(self.state.ref_y + ml_remount["dy_um"])
                        self.log(
                            f"🤖 ML remount guides coarse: "
                            f"dX={ml_remount['dx_um']:+.1f} um, "
                            f"dY={ml_remount['dy_um']:+.1f} um, "
                            f"dTheta={ml_remount['dtheta_deg']:+.2f} deg"
                        )
                else:
                    ml_remount = None

            # ── Fine relocation: ML path first, fall back to CV ──
            ml_match_result = self._ml_recognize_pattern(
                fine_search_surface, self.state.ref_template,
                center_x=fine_search_target_x, center_y=fine_search_target_y,
                half_range_um=self.state.relocation_fine_half_range_um * 2,
            )
            # If coarse positioning already succeeded (affine/landmarks), still try ML remount for logging
            if ml_remount is None:
                cur_overview = build_overview(self.state.surface_image)
                if cur_overview is not None and site_memory.get("overview", {}).get("image") is not None:
                    ml_remount = self._ml_predict_remount(
                        site_memory.get("overview", {}).get("image"),
                        cur_overview.get("image"),
                        scale_x_um_per_px=cur_overview.get("scale_x_um_per_px", 1.0),
                        scale_y_um_per_px=cur_overview.get("scale_y_um_per_px", 1.0),
                    )

            fine_match = None
            ml_used = False

            if ml_match_result is not None:
                ml_x, ml_y, ml_score = ml_match_result
                fine_match = {"x": float(ml_x), "y": float(ml_y), "score": ml_score, "score_gap": ml_score}
                ml_used = True
                self.log(
                    f"🤖 ML pattern matched: position=({ml_x}, {ml_y}), "
                    f"score={ml_score:.3f}"
                )
                if ml_remount is not None:
                    self.log(
                        f"🤖 ML remount prediction: "
                        f"dX={ml_remount['dx_um']:+.1f} um, "
                        f"dY={ml_remount['dy_um']:+.1f} um, "
                        f"dTheta={ml_remount['dtheta_deg']:+.2f} deg"
                    )
            else:
                # CV fallback: search de-rotated surface around expected position
                fine_match = match_reference_template(
                    fine_search_surface, self.state.ref_template,
                    fine_search_target_x, fine_search_target_y,
                    half_range=self.state.relocation_fine_half_range_um,
                )

            # ── Fallback: if de-rotated surface search failed, try original surface ──
            if fine_match is None and fine_to_current_matrix is not None:
                self.log("De-rotated surface search failed, trying on original surface ...")
                # Map ref position to current surface coordinates via the affine
                raw_search_x, raw_search_y = transform_point(
                    fine_to_current_matrix, self.state.ref_x, self.state.ref_y,
                )
                # Wider search on original (non-de-rotated) surface
                fine_match = match_reference_template(
                    self.state.surface_image, self.state.ref_template,
                    raw_search_x, raw_search_y,
                    half_range=self.state.relocation_fine_half_range_um * 2,
                )
                if fine_match is not None:
                    self.log("Original surface search succeeded after de-rotated surface failed")
                    # Update fine_search_surface to original for downstream verification
                    fine_search_surface = self.state.surface_image
                    fine_search_target_x = raw_search_x
                    fine_search_target_y = raw_search_y
                    if fine_match is not None:
                        self.log(
                            f"Template matched on original surface: "
                            f"score={fine_match['score']:.3f}, gap={fine_match.get('score_gap', 0):.3f}"
                        )

            if fine_match is None:
                self.log("AI relocation could not produce a usable reference match.")
                self._enter_click_to_move_correction()
                return

            desired_x = float(fine_match["x"])
            desired_y = float(fine_match["y"])

            # -- Iterative refinement after ML coarse match --
            for iteration in range(int(self.state.relocation_max_iterations)):
                fine_match = match_reference_template(
                    fine_search_surface,
                    self.state.ref_template,
                    desired_x,
                    desired_y,
                    half_range=self.state.relocation_fine_half_range_um,
                )
                if fine_match is not None:
                    desired_x = float(fine_match["x"])
                    desired_y = float(fine_match["y"])
                    self.log(
                        f"ML/CV refinement pass {iteration + 1}: "
                        f"X={desired_x:.1f} um, Y={desired_y:.1f} um, "
                        f"score={fine_match['score']:.3f}"
                    )
                else:
                    break

            if fine_to_current_matrix is not None:
                desired_current_x, desired_current_y = transform_point(
                    fine_to_current_matrix, desired_x, desired_y,
                )
            else:
                desired_current_x = desired_x
                desired_current_y = desired_y

            # 4. Verify: ML verification + traditional verification
            verification = self._verify_relocation(
                desired_x, desired_y,
                surface_image=fine_search_surface,
                site_memory=site_memory,
            )
            # ML site verification as additional check
            ml_verified, ml_prob = self._ml_verify_same_site(
                self.state.ref_template,
                fine_search_surface[
                    int(desired_y) : int(desired_y) + self.state.ref_template.shape[0],
                    int(desired_x) : int(desired_x) + self.state.ref_template.shape[1],
                ] if (
                    0 <= int(desired_y) < fine_search_surface.shape[0] - self.state.ref_template.shape[0]
                    and 0 <= int(desired_x) < fine_search_surface.shape[1] - self.state.ref_template.shape[1]
                ) else fine_search_surface,
            )
            if ml_prob is not None:
                verification["ml_same_site_probability"] = ml_prob
                verification["ml_verified"] = ml_verified
                self.log(f"🤖 ML site verification: probability={ml_prob:.3f}, same_site={'YES' if ml_verified else 'NO'}")

            final_verified = verification.get("verified", False)
            if ml_verified and not final_verified:
                self.log("Traditional verification failed but ML classifier confirms same-site — accepting.")
                final_verified = True
                verification["verified"] = True

            self.state.last_relocation_report = {
                "affine": affine_report,
                "coarse": coarse_result,
                "fine_affine": fine_affine,
                "fine": fine_match,
                "ml_used": ml_used,
                "ml_remount": ml_remount,
                "predicted_current_top_left": {"x_um": float(desired_current_x), "y_um": float(desired_current_y)},
                "verification": verification,
            }

            if final_verified:
                # Move cantilever to the recovered position
                cmd_x, cmd_y = self._clamp_to_stage_margin(desired_current_x, desired_current_y)
                self._start_smooth_move(cmd_x, cmd_y)
                self.state.sample_removed = False
                self.log(
                    "✅ AI Recall SUCCESS: "
                    f"reference score={verification['reference_score']:.3f}, "
                    f"gap={verification['reference_score_gap']:.3f}, "
                    f"{'ML' if ml_used else 'CV'} recognition, "
                    f"moved to ({cmd_x:.1f}, {cmd_y:.1f})"
                )
            else:
                # 5. Verification failed → enter click-to-move correction mode
                self.log(
                    "⚠️ AI Recall verification FAILED: "
                    f"reference score={verification.get('reference_score', 0.0):.3f}, "
                    f"gap={verification.get('reference_score_gap', 0.0):.3f}"
                )
                geometry_check = verification.get("geometry_check")
                if geometry_check is not None:
                    self.log(
                        "Landmark geometry: "
                        f"matched={geometry_check.get('matched_count', 0)}, "
                        f"confidence={geometry_check.get('geometry_confidence', 0.0):.3f}"
                    )
                self._enter_click_to_move_correction(desired_current_x, desired_current_y)
        finally:
            self._end_action("ai_recall")

    def _enter_click_to_move_correction(self, target_x=None, target_y=None):
        """Enter click-to-move mode: user clicks the correct position on the viewport
        to manually guide the cantilever."""
        self.state.ai_relocate_awaiting_click = True
        self.state.ai_relocate_pending_target_x = target_x
        self.state.ai_relocate_pending_target_y = target_y
        if target_x is not None and target_y is not None:
            self.log(
                "👉 Click-to-move correction: click the CORRECT position on the viewport.\n"
                f"   AI best guess was ({target_x:.1f}, {target_y:.1f}).\n"
                "   Right-click to cancel correction mode."
            )
        else:
            self.log(
                "👉 Click-to-move: click the target position on the viewport.\n"
                "   Right-click to cancel."
            )

    # ------------------------------------------------------------------
    # Page 3 – AI Zoom & Recover (PPT slide 3)
    # ------------------------------------------------------------------
    def ai_zoom_recover(self, event):
        """AI Zoom: recall last zoom value, recognize surrounding pattern.
        If not recognized, zoom out and in to search. Then move + verify."""
        if not self._begin_action("ai_zoom", "AI zoom recovery is already running"):
            return
        try:
            self.log("===== AI Zoom & Recover =====")
            # 1. Machine recall the last zoom value
            if self.state.site_memory is None:
                self._try_load_latest_site_memory()
            if self.state.site_memory is not None and self.state.ref_template is None:
                self._activate_site_memory(self.state.site_memory, source_dir=self.state.last_saved_site_dir)

            site_memory = self.state.site_memory or {}
            saved_zoom = site_memory.get("zoom_level")
            if saved_zoom is not None:
                self.state.ai_zoom_recalled_level = float(saved_zoom)
                self._begin_quantized_zoom(float(saved_zoom))
                self.log(f"Recalled zoom: {float(saved_zoom):.2f}x. AI analyzing pattern...")
            else:
                self.log("No saved zoom level in site memory. Using current zoom for recognition.")

            # 2. AI recognize the current surrounding pattern at recalled zoom
            #    Wait for zoom animation to complete + view refresh
            self._wait_for_zoom_complete()

            recognized = self._attempt_pattern_recognition_at_current_zoom()
            if recognized is not None:
                rx, ry, rconf = recognized
                self.log(
                    f"AI recognized pattern at zoom {self.state.current_zoom_level:.2f}x: "
                    f"position=({rx:.1f}, {ry:.1f}), confidence={rconf:.3f}"
                )
                # 4. Move to recognized position
                self._finish_ai_zoom_move(rx, ry, rconf)
            else:
                # 3. Not recognized → zoom out search, then zoom in
                self.log("Pattern NOT recognized at current zoom. Starting zoom-out search...")
                self.state.ai_zoom_search_active = True
                self.state.ai_zoom_search_stage = "zooming_out"
                self._start_zoom_out_search()
        finally:
            self._end_action("ai_zoom")

    def _attempt_pattern_recognition_at_current_zoom(self):
        """Try to recognize the saved high-mag landmarks in the current viewport.
        Uses ML first, falls back to CV.
        Returns (target_x_um, target_y_um, confidence) or None."""
        site_memory = self.state.site_memory or {}
        current_view = self.state.current_camera_view if self.state.current_camera_view is not None else self.state.current_fov_raw
        if current_view is None or current_view.size == 0:
            return None

        # ── ML path: ResNet + sliding window ──
        if self._use_ml() and self.state.ref_template is not None:
            ml_result = self._ml_recognize_pattern(current_view, self.state.ref_template)
            if ml_result is not None:
                ml_x, ml_y, ml_score = ml_result
                tpl_h, tpl_w = self.state.ref_template.shape[:2]
                target_x = float(self.state.x + ml_x + tpl_w / 2.0)
                target_y = float(self.state.y + ml_y + tpl_h / 2.0)
                self.log(f"🤖 ML recognized pattern at zoom {self.state.current_zoom_level:.2f}x: score={ml_score:.3f}")
                return (target_x, target_y, float(ml_score))

        # ── CV fallback: landmark geometry + template matching ──
        highmag_landmarks = site_memory.get("highmag_landmarks", [])
        if not highmag_landmarks:
            self.log("No saved high-mag landmarks to recognize.")
            return None

        report = analyze_landmark_geometry(
            highmag_landmarks,
            current_view,
            view_origin_x_um=float(self.state.x),
            view_origin_y_um=float(self.state.y),
            tip_x_um=float(self.state.probe_tip_x),
            tip_y_um=float(self.state.probe_tip_y),
            min_score=max(0.35, float(self.state.relocation_min_match_score) - 0.07),
            min_gap=max(0.01, float(self.state.relocation_min_score_gap) - 0.01),
        )
        matched_count = report.get("matched_count", 0)
        geometry_conf = report.get("geometry_confidence", 0.0)
        if matched_count >= self.state.relocation_min_landmark_support and geometry_conf >= 0.30:
            # Use the matched landmarks to compute target position
            matches = report.get("matches", [])
            if matches:
                ref_tl = site_memory.get("reference_top_left") or {}
                ref_tip = site_memory.get("reference_tip") or {}
                avg_dx = float(np.mean([m.get("tip_dx_um", 0.0) or 0.0 for m in matches]))
                avg_dy = float(np.mean([m.get("tip_dy_um", 0.0) or 0.0 for m in matches]))
                target_x = float(self.state.probe_tip_x + avg_dx)
                target_y = float(self.state.probe_tip_y + avg_dy)
                return (target_x, target_y, float(geometry_conf))

        # Even with fewer matches, try template matching on current view
        if self.state.ref_template is not None:
            candidates = match_template_candidates(current_view, self.state.ref_template, top_k=1)
            if candidates and candidates[0]["score"] >= 0.40:
                best = candidates[0]
                tpl_h, tpl_w = self.state.ref_template.shape[:2]
                target_x = float(self.state.x + best["x"] + tpl_w / 2.0)
                target_y = float(self.state.y + best["y"] + tpl_h / 2.0)
                return (target_x, target_y, float(best["score"]))

        return None

    def _start_zoom_out_search(self):
        """Zoom out step by step, searching for landmarks at each level."""
        levels = list(self.state.zoom_levels)
        current_idx = self._get_zoom_level_index(self.state.current_zoom_level)
        # Find the next wider zoom level
        search_idx = max(0, current_idx - 1)
        if search_idx == current_idx:
            self.log("Already at widest zoom. Cannot zoom out further.")
            self.state.ai_zoom_search_active = False
            self.state.ai_zoom_search_stage = "idle"
            return
        self.state.ai_zoom_search_stage = "zooming_out"
        self.state._ai_zoom_search_idx = search_idx
        self._begin_quantized_zoom(levels[search_idx])
        self.log(f"Zooming out to {levels[search_idx]:.2f}x for search...")
        self._wait_for_zoom_complete()
        self._continue_zoom_search()

    def _continue_zoom_search(self):
        """Continue the zoom search: test recognition at current zoom level."""
        if not self.state.ai_zoom_search_active:
            return

        self.state.ai_zoom_search_stage = "searching"
        recognized = self._attempt_pattern_recognition_at_current_zoom()
        if recognized is not None:
            rx, ry, rconf = recognized
            self.log(
                f"Pattern FOUND during zoom search at {self.state.current_zoom_level:.2f}x: "
                f"confidence={rconf:.3f}"
            )
            self.state.ai_zoom_search_stage = "zooming_in"
            recalled_zoom = self.state.ai_zoom_recalled_level
            if recalled_zoom is not None and recalled_zoom != self.state.current_zoom_level:
                self._begin_quantized_zoom(float(recalled_zoom))
                self.log(f"Zooming back in to recalled level {float(recalled_zoom):.2f}x...")
                self._wait_for_zoom_complete()
            self.state.ai_zoom_search_active = False
            self.state.ai_zoom_search_stage = "done"
            self._finish_ai_zoom_move(rx, ry, rconf)
        else:
            # Try zooming out one more step
            levels = list(self.state.zoom_levels)
            current_idx = self._get_zoom_level_index(self.state.current_zoom_level)
            search_idx = getattr(self.state, "_ai_zoom_search_idx", current_idx - 1)
            next_idx = max(0, current_idx - 1)
            if next_idx < current_idx:
                self.state._ai_zoom_search_idx = next_idx
                self._begin_quantized_zoom(levels[next_idx])
                self.log(f"Still not found. Zooming further out to {levels[next_idx]:.2f}x...")
                self._wait_for_zoom_complete()
                self._continue_zoom_search()
            else:
                self.log("Zoom search exhausted. Pattern not found at any zoom level.")
                self.state.ai_zoom_search_active = False
                self.state.ai_zoom_search_stage = "idle"
                self._enter_click_to_move_correction()

    def _finish_ai_zoom_move(self, target_x, target_y, confidence):
        """Move cantilever to the AI-recognized position and verify."""
        # Convert from absolute coordinates to viewport top-left
        target_tl_x = target_x - self.state.fov_width / 2.0
        target_tl_y = target_y - self.state.fov_height / 2.0
        target_tl_x, target_tl_y = self._clamp_to_stage_margin(target_tl_x, target_tl_y)

        site_memory = self.state.site_memory or {}
        verification = self._verify_relocation(
            target_tl_x, target_tl_y,
            surface_image=self.state.surface_image,
            site_memory=site_memory,
        )
        if verification.get("verified", False):
            self._start_smooth_move(target_tl_x, target_tl_y)
            self.state.sample_removed = False
            self.log(
                "✅ AI Zoom SUCCESS: "
                f"confidence={confidence:.3f}, "
                f"verify_score={verification['reference_score']:.3f}, "
                f"moved to ({target_tl_x:.1f}, {target_tl_y:.1f})"
            )
        else:
            self.log(
                "⚠️ AI Zoom verification FAILED: "
                f"recognition_confidence={confidence:.3f}, "
                f"verify_score={verification.get('reference_score', 0.0):.3f}"
            )
            self._enter_click_to_move_correction(target_tl_x, target_tl_y)

    def toggle_ai_mode(self, event):
        """Toggle AI compensation mode."""
        self.ai_mode = not self.ai_mode
        self.log(f"AI compensation mode: {'ON' if self.ai_mode else 'OFF'}")
