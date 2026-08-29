import numpy as np
from afm_utils import create_stage_fov, get_scale_bar_geometry, render_camera_frame, render_camera_matching_frame, render_camera_recognition_frame, update_title

class AFMAnimation:
    def __init__(self, state, stage, data, ax, img, ideal_line, hyst_line,
                 artifact_layer, fig, get_tip_func, scale_bar_black=None, scale_bar_white=None,
                 scale_bar_text=None,
                 status_update_func=None,
                 probe_update_func=None):
        self.state = state
        self.stage = stage
        self.data = data
        self.ax = ax
        self.img = img
        self.ideal_line = ideal_line
        self.hyst_line = hyst_line
        self.artifact_layer = artifact_layer
        self.fig = fig
        self.get_tip = get_tip_func
        self.scale_bar_black = scale_bar_black
        self.scale_bar_white = scale_bar_white
        self.scale_bar_text = scale_bar_text
        self.status_update = status_update_func
        self.probe_update = probe_update_func

    def _sync_best_focus_for_zoom(self, zoom_level=None):
        self.state.z_stage_position_um = float(
            self.state.get_effective_camera_stage_position_um(zoom_level) - self.state.focus_z_um
        )
    
    def update(self, frame):
        # 自动扫描
        if self.state.auto_scan_active and not self.state.paused and not self.state.zooming:
            frac = self.state.auto_scan_step / self.state.auto_scan_total_steps
            if self.state.auto_scan_direction == 1:
                self.state.target_x = self.state.auto_scan_start_x + (self.state.auto_scan_end_x - self.state.auto_scan_start_x) * frac
            else:
                self.state.target_x = self.state.auto_scan_end_x - (self.state.auto_scan_end_x - self.state.auto_scan_start_x) * frac
            self.state.auto_scan_step += 1
            if self.state.auto_scan_step > self.state.auto_scan_total_steps:
                if self.state.auto_scan_direction == 1:
                    self.state.auto_scan_direction = -1
                    self.state.auto_scan_step = 0
                else:
                    self.state.auto_scan_active = False
        
        # 平滑移动处理
        if self.state.smooth_move_active and not self.state.paused and not self.state.zooming:
            dx = self.state.smooth_move_target_x - self.state.target_x
            dy = self.state.smooth_move_target_y - self.state.target_y
            step = self.state.smooth_move_step
            
            if abs(dx) > step:
                self.state.target_x += step if dx > 0 else -step
            else:
                self.state.target_x = self.state.smooth_move_target_x
            
            if abs(dy) > step:
                self.state.target_y += step if dy > 0 else -step
            else:
                self.state.target_y = self.state.smooth_move_target_y
            
            if (self.state.target_x == self.state.smooth_move_target_x and 
                self.state.target_y == self.state.smooth_move_target_y):
                self.state.smooth_move_active = False
        
        # 位置更新
        if not self.state.paused and not self.state.zooming:
            old_x = float(self.state.x)
            old_y = float(self.state.y)
            if self.state.pi_mode:
                x_new, y_new = self.stage.move_to(self.state.target_x, self.state.target_y)
                self.state.x = x_new
                self.state.y = y_new
            else:
                if self.state.x < self.state.target_x:
                    self.state.x = min(self.state.x + self.state.step_speed, self.state.target_x)
                elif self.state.x > self.state.target_x:
                    self.state.x = max(self.state.x - self.state.step_speed, self.state.target_x)
                if self.state.y < self.state.target_y:
                    self.state.y = min(self.state.y + self.state.step_speed, self.state.target_y)
                elif self.state.y > self.state.target_y:
                    self.state.y = max(self.state.y - self.state.step_speed, self.state.target_y)
            delta_x = float(self.state.x - old_x)
            delta_y = float(self.state.y - old_y)
            if not np.isclose(delta_x, 0.0) or not np.isclose(delta_y, 0.0):
                self.state.probe_tip_x = float(self.state.probe_tip_x + delta_x)
                self.state.probe_tip_y = float(self.state.probe_tip_y + delta_y)
        
        # 缩放
        if self.state.zooming:
            self._handle_zoom()
            return self.img, self.ideal_line, self.hyst_line
        
        # 视野更新
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
            camera_resolution=self.state.camera_resolution,
            outside_mask=outside_mask,
            focus_model=self.state.get_focus_model(),
            fov_width_um=self.state.fov_width,
            fov_height_um=self.state.fov_height,
            body_width_um=self.state.probe_body_width_um,
            tip_width_um=self.state.probe_tip_width_um,
            tip_total_length_um=self.state.probe_tip_total_length_um,
            triangular_tip_length_um=self.state.probe_triangular_tip_length_um,
            visible_body_depth_um=self.state.probe_visible_body_depth_um,
        )
        self.state.current_matching_view, _ = render_camera_matching_frame(
            fov,
            camera_resolution=(512, 384),
            outside_mask=outside_mask,
            focus_model=self.state.get_focus_model(),
            fov_width_um=self.state.fov_width,
            fov_height_um=self.state.fov_height,
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
        self.img.set_data(display_fov)
        self.img.set_extent([ix, ix+self.state.fov_width, iy+self.state.fov_height, iy])
        self.ax.set_xlim(ix, ix + self.state.fov_width)
        self.ax.set_ylim(iy + self.state.fov_height, iy)
        self._update_scale_bar(ix, iy, self.state.fov_width, self.state.fov_height)
        if self.probe_update is not None:
            self.probe_update()
        
        # 记录轨迹
        tip_x, tip_y = self.get_tip()
        if self.state.pi_mode:
            self.data.add_hyst(tip_x, tip_y, self.state.target_x, self.state.target_y)
        else:
            self.data.add_ideal(tip_x, tip_y, self.state.target_x, self.state.target_y)
        
        # 更新线条和标题
        self.ideal_line.set_data(self.data.ideal_tip_x, self.data.ideal_tip_y)
        self.hyst_line.set_data(self.data.hyst_tip_x, self.data.hyst_tip_y)
        
        tip_x, tip_y = self.get_tip()
        target_center_x = float(self.state.target_x + self.state.fov_width / 2.0)
        target_center_y = float(self.state.target_y + self.state.fov_height / 2.0)
        update_title(self.ax, self.fig, self.state.pi_mode, target_center_x, target_center_y, tip_x, tip_y)
        if self.status_update is not None:
            self.status_update()

        return self.img, self.ideal_line, self.hyst_line
    
    def _handle_zoom(self):
        self.state.zoom_progress += 1
        alpha = self.state.zoom_progress / self.state.zoom_steps

        if self.state.zoom_center_x is None or self.state.zoom_center_y is None:
            self.state.zoom_center_x = self.state.x + self.state.fov_width / 2.0
            self.state.zoom_center_y = self.state.y + self.state.fov_height / 2.0
            self.state.zoom_base_width = self.state.fov_width
            self.state.zoom_base_height = self.state.fov_height

        base_width = self.state.zoom_base_width or self.state.fov_width
        base_height = self.state.zoom_base_height or self.state.fov_height
        target_width = self.state.zoom_target_width or self.state.fov_width
        target_height = self.state.zoom_target_height or self.state.fov_height
        center_x = self.state.zoom_center_x
        center_y = self.state.zoom_center_y

        new_width = max(50, int(round(base_width + (target_width - base_width) * alpha)))
        new_height = max(50, int(round(base_height + (target_height - base_height) * alpha)))
        interpolated_zoom = float(self.state.current_zoom_level) + (
            float(self.state.target_zoom_level) - float(self.state.current_zoom_level)
        ) * alpha
        self._sync_best_focus_for_zoom(interpolated_zoom)
        x_new = center_x - new_width / 2.0
        y_new = center_y - new_height / 2.0
        fov, outside_mask, ix, iy = create_stage_fov(
            self.state.surface_image,
            self.artifact_layer,
            self.state.show_artifact,
            x_new,
            y_new,
            new_width,
            new_height,
            valid_mask=self.state.surface_valid_mask,
        )
        self.state.current_fov_raw = fov.copy()
        self.state.current_camera_view, _ = render_camera_recognition_frame(
            fov,
            camera_resolution=self.state.camera_resolution,
            outside_mask=outside_mask,
            focus_model=self.state.get_focus_model(
                zoom_level=interpolated_zoom,
                fov_width_um=new_width,
                fov_height_um=new_height,
            ),
            fov_width_um=new_width,
            fov_height_um=new_height,
            body_width_um=self.state.probe_body_width_um,
            tip_width_um=self.state.probe_tip_width_um,
            tip_total_length_um=self.state.probe_tip_total_length_um,
            triangular_tip_length_um=self.state.probe_triangular_tip_length_um,
            visible_body_depth_um=self.state.probe_visible_body_depth_um,
        )
        self.state.current_matching_view, _ = render_camera_matching_frame(
            fov,
            camera_resolution=(512, 384),
            outside_mask=outside_mask,
            focus_model=self.state.get_focus_model(
                zoom_level=interpolated_zoom,
                fov_width_um=new_width,
                fov_height_um=new_height,
            ),
            fov_width_um=new_width,
            fov_height_um=new_height,
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
            focus_model=self.state.get_focus_model(
                zoom_level=interpolated_zoom,
                fov_width_um=new_width,
                fov_height_um=new_height,
            ),
        )
        self.state.last_blur_diameter_um = focus_metrics["blur_diameter_um"]
        self.state.last_blur_sigma_px = focus_metrics["sigma_px"]
        self.state.last_dof_camera_um = focus_metrics["dof_camera_um"]
        self.img.set_data(display_fov)
        self.img.set_extent([x_new, x_new+new_width, y_new+new_height, y_new])
        self.ax.set_xlim(x_new, x_new + new_width)
        self.ax.set_ylim(y_new + new_height, y_new)
        self._update_scale_bar(x_new, y_new, new_width, new_height)
        if self.probe_update is not None:
            original_zoom = self.state.current_zoom_level
            self.state.current_zoom_level = interpolated_zoom
            self.probe_update()
            self.state.current_zoom_level = original_zoom
        
        if self.state.zoom_progress >= self.state.zoom_steps:
            self.state.zooming = False
            self.state.zoom_progress = 0
            self.state.x, self.state.y = x_new, y_new
            self.state.fov_width, self.state.fov_height = new_width, new_height
            self.state.current_zoom_level = float(self.state.target_zoom_level)
            self._sync_best_focus_for_zoom(self.state.current_zoom_level)
            # Optical zoom should not trigger any mechanical catch-up motion afterward.
            self.state.target_x = self.state.x
            self.state.target_y = self.state.y
            self.state.smooth_move_active = False
            if self.state.pi_mode:
                self.stage.reset(self.state.x, self.state.y)
                self.stage.cmd_x = self.state.x
                self.stage.cmd_y = self.state.y
            self.state.zoom_anchor_tip_x = None
            self.state.zoom_anchor_tip_y = None
            self.state.zoom_anchor_rel_x = None
            self.state.zoom_anchor_rel_y = None
            self.state.zoom_base_width = None
            self.state.zoom_base_height = None
            self.state.zoom_center_x = None
            self.state.zoom_center_y = None
            self.state.zoom_target_width = None
            self.state.zoom_target_height = None

    def _update_scale_bar(self, x_origin, y_origin, fov_width, fov_height):
        if self.scale_bar_black is None or self.scale_bar_white is None or self.scale_bar_text is None:
            return

        geometry = get_scale_bar_geometry(
            x_origin,
            y_origin,
            fov_width,
            fov_height,
            self.state.scale_bar_total_um,
            self.state.scale_bar_segments,
        )
        black_segment = geometry["segments"][0]
        self.scale_bar_black.set_data([black_segment[0], black_segment[1]], [geometry["y"], geometry["y"]])

        if len(geometry["segments"]) > 1:
            white_segment = geometry["segments"][1]
            self.scale_bar_white.set_data([white_segment[0], white_segment[1]], [geometry["y"], geometry["y"]])
        else:
            self.scale_bar_white.set_data([], [])

        self.scale_bar_text.set_position(geometry["text_pos"])
        self.scale_bar_text.set_text(geometry["label"])
