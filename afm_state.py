import numpy as np
from afm_optics_model import (
    DEFAULT_ZOOM_OUT_LIFT_AT_MIN_UM,
    DEFAULT_ZOOM_OUT_LIFT_CURVE_POWER,
    DEFAULT_ZOOM_OUT_LIFT_START,
    compute_zoom_out_camera_lift_um,
)


class AFMState:
    def __init__(self, sample, width_um, height_um):
        self.surface_image = sample
        self.sample = self.surface_image
        self.surface_valid_mask = np.ones(sample.shape[:2], dtype=bool)
        self.width_um = width_um
        self.height_um = height_um

        self.fov_width = 840
        self.fov_height = 630
        self.x = 0.0
        self.y = 0.0
        self.target_x = self.x
        self.target_y = self.y
        self.probe_tip_x = self.x + self.fov_width / 2.0
        self.probe_tip_y = self.y + self.fov_height / 2.0
        self.stage_margin_um = 2000.0
        self.max_zoom_out_scale = 4.0

        self.step_speed = 5
        self.move_step = 200
        self.current_step = 5
        self.animation_interval_ms = 30
        self.paused = False

        self.zooming = False
        self.zoom_progress = 0
        self.zoom_steps = 20
        self.zoom_direction = 0
        self.zoom_anchor_tip_x = None
        self.zoom_anchor_tip_y = None
        self.zoom_anchor_rel_x = None
        self.zoom_anchor_rel_y = None
        self.zoom_base_width = None
        self.zoom_base_height = None
        self.zoom_center_x = None
        self.zoom_center_y = None
        self.zoom_target_width = None
        self.zoom_target_height = None
        self.zoom_levels = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        self.current_zoom_level = 0.25
        self.target_zoom_level = 0.25
        self.min_zoom_level = min(self.zoom_levels)
        self.max_zoom_level = max(self.zoom_levels)
        self.current_fov_raw = None
        self.current_camera_view = None

        self.pi_mode = False

        self.auto_scan_active = False
        self.auto_scan_step = 0
        self.auto_scan_total_steps = 200
        self.auto_scan_start_x = 0.0
        self.auto_scan_end_x = 0.0
        self.auto_scan_direction = 1

        self.show_artifact = False
        self.sample_source = "synthetic-surface"
        self.sample_path = None
        self.default_image_path = None
        self.default_image_width_um = None
        self.default_image_height_um = None

        self.camera_resolution = (1024, 768)
        self.camera_reference_resolution = (2592, 1944)
        self.sample_view_camera_resolution = (4208, 3120)
        self.camera_mode = "Park FX40 On-Axis Optics"
        self.base_objective_magnification = 10.0
        self.objective_numerical_aperture = 0.21
        self.illumination_wavelength_um = 0.55
        self.sensor_pixel_size_um = 3.45
        self.on_axis_fov_width_um = 840.0
        self.on_axis_fov_height_um = 630.0
        self.sample_view_fov_width_mm = 172.0
        self.sample_view_fov_height_mm = 97.0
        self.z_stage_travel_um = 22000.0
        self.z_stage_position_um = 0.0
        self.camera_stage_position_um = 0.0
        self.focus_z_um = 0.0
        self.z_stage_step_um = 5.0
        self.probe_lift_start_zoom = DEFAULT_ZOOM_OUT_LIFT_START
        self.probe_lift_at_min_zoom_um = DEFAULT_ZOOM_OUT_LIFT_AT_MIN_UM
        self.probe_lift_curve_power = DEFAULT_ZOOM_OUT_LIFT_CURVE_POWER
        self.probe_full_exit_gap_um = 320.0
        self.probe_exit_travel_fov_heights = 1.25
        self.afm_z_scanner_range_options_um = (15.0, 30.0)
        self.xy_scanner_range_options_um = ((100.0, 100.0), (50.0, 50.0), (5.0, 5.0))
        self.z_stage_position_um = float(self.get_effective_camera_stage_position_um() - self.focus_z_um)
        self.fov_width, self.fov_height = self.get_fov_for_zoom_level(self.current_zoom_level)
        max_x = max(float(self.width_um) - float(self.fov_width), 0.0)
        max_y = max(float(self.height_um) - float(self.fov_height), 0.0)
        self.x = float(np.random.uniform(0.0, max_x)) if max_x > 0.0 else 0.0
        self.y = float(np.random.uniform(0.0, max_y)) if max_y > 0.0 else 0.0
        self.target_x = self.x
        self.target_y = self.y
        self.probe_tip_x = self.x + self.fov_width / 2.0
        self.probe_tip_y = self.y + self.fov_height / 2.0
        self.last_blur_diameter_um = 0.0
        self.last_blur_sigma_px = 0.0
        self.last_dof_camera_um = 0.0
        self.manual_dof_camera_um = None
        self.dof_step_um = 0.5

        self.default_scale_um_per_px = 1.0
        self.scale_bar_total_um = 200.0
        self.scale_bar_segments = 2
        self.show_probe_hud = False
        self.show_hud_detection = False
        self.show_hud_distance = False
        self.scan_region_size_um = 100.0
        self.probe_body_width_um = 1600.0
        self.probe_tip_width_um = 35.0
        self.probe_tip_total_length_um = 125.0
        self.probe_triangular_tip_length_um = 15.0
        self.probe_visible_body_depth_um = 3400.0
        self.hud_landmark_overlay_enabled = True
        self.hud_landmark_matches = []
        self.hud_landmark_report = None
        self.detection_rois = []
        self.detection_roi_draw_mode = False
        self.detection_roi_drag_active = False
        self.detection_roi_center_x_um = None
        self.detection_roi_center_y_um = None
        self.detection_roi_radius_um = 0.0
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_label = "Origin"
        self.origin_defined = False
        self.origin_template = None
        self.origin_template_half_size = 48

        self.ref_artefacts = []
        self.ref_template = None
        self.ref_x = 0.0
        self.ref_y = 0.0
        self.site_memory = None
        self.saved_site_memories = []
        self.last_saved_site_dir = None
        self.last_relocation_report = None
        self.last_affine_transform_report = None
        self.relocation_min_match_score = 0.42
        self.relocation_min_score_gap = 0.02
        self.relocation_min_landmark_support = 2
        self.relocation_min_affine_confidence = 0.12
        self.relocation_min_affine_inliers = 12
        self.force_ml_mode = False  # 强制使用5w ML模型（跳过ORB）
        self.relocation_fine_half_range_um = 700.0
        self.relocation_verify_half_range_um = 120.0
        self.relocation_max_iterations = 3
        self.simulated_sample_shift_x_um = 0.0
        self.simulated_sample_shift_y_um = 0.0
        self.simulated_sample_rotation_deg = 0.0
        self.sample_removed = False
        self.ai_desired_history_x = [self.x, self.x]
        self.ai_desired_history_y = [self.y, self.y]

        # Page 2 (AI Relocation): click-to-move correction mode
        self.ai_relocate_awaiting_click = False
        self.ai_relocate_pending_target_x = None
        self.ai_relocate_pending_target_y = None

        # Page 3 (AI Zoom): zoom search state
        self.ai_zoom_search_active = False
        self.ai_zoom_search_stage = "idle"  # idle / zooming_out / searching / zooming_in / done
        self.ai_zoom_recalled_level = None
        self.ai_zoom_search_pending_target_x = None
        self.ai_zoom_search_pending_target_y = None

        self.smooth_move_active = False
        self.smooth_move_target_x = None
        self.smooth_move_target_y = None
        self.smooth_move_step = 20.0
        self.smooth_move_min_step = 1.0
        self.smooth_move_max_step = 160.0

    def get_optical_zoom_ratio(self):
        return float(np.clip(self.current_zoom_level, self.min_zoom_level, self.max_zoom_level))

    def get_digital_zoom_level(self):
        return float(np.clip(self.current_zoom_level, self.min_zoom_level, self.max_zoom_level))

    def get_current_objective_magnification(self):
        return float(self.base_objective_magnification) * self.get_optical_zoom_ratio()

    def get_fov_for_zoom_level(self, zoom_level):
        zoom_level = float(np.clip(float(zoom_level), self.min_zoom_level, self.max_zoom_level))
        width = max(50, int(round(self.on_axis_fov_width_um / float(zoom_level))))
        height = max(50, int(round(self.on_axis_fov_height_um / float(zoom_level))))
        return width, height

    def get_zoom_out_camera_lift_um(self, zoom_level=None):
        zoom_level = float(self.current_zoom_level if zoom_level is None else zoom_level)
        return compute_zoom_out_camera_lift_um(
            zoom_level=zoom_level,
            zoom_levels=self.zoom_levels,
            lift_start_zoom=self.probe_lift_start_zoom,
            lift_at_min_zoom_um=self.probe_lift_at_min_zoom_um,
            curve_power=self.probe_lift_curve_power,
        )

    def get_effective_camera_stage_position_um(self, zoom_level=None):
        return float(self.camera_stage_position_um + self.get_zoom_out_camera_lift_um(zoom_level))

    def get_probe_sample_gap_um(self, zoom_level=None):
        return float(self.get_effective_camera_stage_position_um(zoom_level) - self.z_stage_position_um)

    def get_focus_offset_um(self, zoom_level=None):
        return float(self.get_probe_sample_gap_um(zoom_level) - self.focus_z_um)

    def get_focus_model(self, zoom_level=None, fov_width_um=None, fov_height_um=None):
        return {
            "z_position_um": float(self.get_probe_sample_gap_um(zoom_level)),
            "focus_z_um": float(self.focus_z_um),
            "numerical_aperture": float(self.objective_numerical_aperture),
            "wavelength_um": float(self.illumination_wavelength_um),
            "sensor_pixel_size_um": float(self.sensor_pixel_size_um),
            "manual_dof_camera_um": (
                None if self.manual_dof_camera_um is None else float(self.manual_dof_camera_um)
            ),
            "objective_magnification": float(self.base_objective_magnification * np.clip(
                float(self.current_zoom_level if zoom_level is None else zoom_level),
                self.min_zoom_level,
                self.max_zoom_level,
            )),
            "fov_width_um": float(self.fov_width if fov_width_um is None else fov_width_um),
            "fov_height_um": float(self.fov_height if fov_height_um is None else fov_height_um),
        }
