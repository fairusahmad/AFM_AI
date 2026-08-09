import matplotlib.pyplot as plt
import cv2
import numpy as np
from collections import deque
import json
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
from pathlib import Path
import tkinter as tk
from tkinter import scrolledtext, simpledialog

from afm_animation import AFMAnimation
from afm_callbacks import AFMCallbacks
from afm_data import TrajectoryData
from afm_relocation import analyze_landmark_geometry
from afm_state import AFMState
from afm_ui import setup_dashboard, setup_figure, setup_probe_graphics
from afm_utils import create_stage_fov, get_defocus_metrics, get_tip_position, render_camera_frame, render_camera_recognition_frame, rotate_camera_frame, update_title
from hysteresis import NanoPositioner
from sample_generation import artifact_layer, height_um, sample as stage_surface_image, width_um

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SETTINGS_PATH = BASE_DIR / "afm_default_settings.json"
DOCK_LAYOUT_PATH = BASE_DIR / "afm_dock_layout.json"
DEFAULT_IMAGE_CONFIG = {
    "path": Path(r"c:\Users\fairu\Downloads\To delete\ChatGPT Image May 20, 2026, 02_20_13 PM.png"),
    "autoload": True,
}


class StatusTextDock:
    def __init__(self, fig, manager, panel, text_artist):
        self.fig = fig
        self.manager = manager
        self.panel = panel
        self.text_artist = text_artist
        self.host_widget = None
        self.container = None
        self.text_widget = None
        self.entries = []
        self._line_tooltips = {}
        self.tooltip_window = None
        self._hovered_line = None
        self._last_yview = (0.0, 1.0)
        self._build_widget()

    def _build_widget(self):
        root = getattr(self.manager, "window", None)
        tk_widget_getter = getattr(self.fig.canvas, "get_tk_widget", None)
        if root is None or tk_widget_getter is None:
            self.text_artist.set_visible(True)
            return
        try:
            host_widget = tk_widget_getter()
            self.host_widget = host_widget
            self.text_artist.set_visible(False)

            container = tk.Frame(host_widget, bg="#f6f9fe", highlightbackground="#c7d8ec", highlightthickness=1, bd=0)
            text_widget = scrolledtext.ScrolledText(
                container,
                wrap=tk.NONE,
                bg="#f6f9fe",
                fg="#24384f",
                insertbackground="#24384f",
                font=("Consolas", 8),
                relief=tk.FLAT,
                borderwidth=0,
                padx=6,
                pady=6,
            )
            text_widget.pack(fill=tk.BOTH, expand=True)
            text_widget.configure(state=tk.DISABLED)
            text_widget.tag_configure("heading", font=("Segoe UI", 9, "bold"), foreground="#27476a")
            text_widget.tag_configure("body", font=("Consolas", 8), foreground="#24384f")
            text_widget.bind("<Motion>", self._on_motion, add="+")
            text_widget.bind("<Leave>", self._on_leave, add="+")
            text_widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
            text_widget.bind("<Button-4>", self._on_mousewheel_linux, add="+")
            text_widget.bind("<Button-5>", self._on_mousewheel_linux, add="+")

            self.container = container
            self.text_widget = text_widget

            self.fig.canvas.mpl_connect("draw_event", self._sync_geometry)
            host_widget.bind("<Configure>", self._sync_geometry, add="+")
            root.bind("<Configure>", self._sync_geometry, add="+")
            self._sync_geometry()
        except Exception:
            self.container = None
            self.text_widget = None
            self.host_widget = None
            self.text_artist.set_visible(True)

    def _sync_geometry(self, event=None):
        if self.container is None or self.host_widget is None:
            return
        bbox = self.panel.ax.get_window_extent()
        host_height = self.host_widget.winfo_height()
        x0 = int(round(bbox.x0 + 6))
        y0 = int(round(host_height - bbox.y1 + self.panel.header_frac * bbox.height + 4))
        width = int(round(bbox.width - 12))
        height = int(round(bbox.height * (1.0 - self.panel.header_frac) - 8))
        if width <= 20 or height <= 20:
            self.container.place_forget()
            return
        self.container.place(x=x0, y=y0, width=width, height=height)

    def _hide_tooltip(self):
        if self.tooltip_window is not None:
            self.tooltip_window.destroy()
            self.tooltip_window = None
        self._hovered_line = None

    def _show_tooltip(self, text, x_root, y_root):
        if not text:
            self._hide_tooltip()
            return
        if self.tooltip_window is not None:
            self.tooltip_window.destroy()
        tooltip = tk.Toplevel(self.text_widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{x_root + 14}+{y_root + 14}")
        label = tk.Label(
            tooltip,
            text=text,
            justify=tk.LEFT,
            bg="#fffde8",
            fg="#202020",
            relief=tk.SOLID,
            borderwidth=1,
            padx=8,
            pady=5,
            font=("Segoe UI", 9),
            wraplength=360,
        )
        label.pack()
        self.tooltip_window = tooltip

    def _on_motion(self, event):
        if self.text_widget is None:
            return
        index = self.text_widget.index(f"@{event.x},{event.y}")
        line_no = int(index.split(".")[0])
        tooltip = self._line_tooltips.get(line_no)
        if not tooltip:
            self._hide_tooltip()
            return
        if self._hovered_line == line_no and self.tooltip_window is not None:
            return
        self._hovered_line = line_no
        self._show_tooltip(tooltip, event.x_root, event.y_root)

    def _on_leave(self, event):
        self._hide_tooltip()

    def _on_mousewheel(self, event):
        if self.text_widget is None:
            return
        delta = -1 if event.delta > 0 else 1
        self.text_widget.yview_scroll(delta, "units")
        self._last_yview = self.text_widget.yview()
        return "break"

    def _on_mousewheel_linux(self, event):
        if self.text_widget is None:
            return
        delta = -1 if getattr(event, "num", None) == 4 else 1
        self.text_widget.yview_scroll(delta, "units")
        self._last_yview = self.text_widget.yview()
        return "break"

    def set_entries(self, entries):
        self.entries = list(entries)
        if self.text_widget is None:
            self.text_artist.set_text("\n".join(str(entry.get("text", "")) for entry in self.entries))
            return
        previous_yview = self.text_widget.yview()
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self._line_tooltips = {}
        for line_no, entry in enumerate(self.entries, start=1):
            text = str(entry.get("text", ""))
            tag = "heading" if entry.get("is_heading") else "body"
            self.text_widget.insert(tk.END, text, (tag,))
            if line_no < len(self.entries):
                self.text_widget.insert(tk.END, "\n")
            tooltip = entry.get("tooltip")
            if tooltip:
                self._line_tooltips[line_no] = str(tooltip)
        self.text_widget.configure(state=tk.DISABLED)
        target_top = previous_yview[0] if previous_yview != (0.0, 1.0) else self._last_yview[0]
        self.text_widget.yview_moveto(target_top)
        self._last_yview = self.text_widget.yview()

    def set_lines(self, lines):
        self.set_entries([{"text": str(line)} for line in lines])


class TraceTextDock:
    def __init__(self, fig, manager, panel, text_artist, history_lines=250):
        self.fig = fig
        self.manager = manager
        self.panel = panel
        self.text_artist = text_artist
        self.history_lines = history_lines
        self.host_widget = None
        self.container = None
        self.text_widget = None
        self.lines = deque(maxlen=history_lines)
        self._build_widget()

    def _build_widget(self):
        root = getattr(self.manager, "window", None)
        tk_widget_getter = getattr(self.fig.canvas, "get_tk_widget", None)
        if root is None or tk_widget_getter is None:
            self.text_artist.set_visible(True)
            return
        try:
            host_widget = tk_widget_getter()
            self.host_widget = host_widget
            self.text_artist.set_visible(False)

            container = tk.Frame(host_widget, bg="#101824", highlightbackground="#2b405c", highlightthickness=1, bd=0)
            text_widget = scrolledtext.ScrolledText(
                container,
                wrap=tk.WORD,
                bg="#0b1220",
                fg="#cde7ff",
                insertbackground="#cde7ff",
                font=("Consolas", 8),
                relief=tk.FLAT,
                borderwidth=0,
                padx=6,
                pady=6,
            )
            text_widget.pack(fill=tk.BOTH, expand=True)
            text_widget.configure(state=tk.DISABLED)

            self.container = container
            self.text_widget = text_widget

            self.fig.canvas.mpl_connect("draw_event", self._sync_geometry)
            host_widget.bind("<Configure>", self._sync_geometry, add="+")
            root.bind("<Configure>", self._sync_geometry, add="+")
            self._sync_geometry()
        except Exception:
            self.container = None
            self.text_widget = None
            self.host_widget = None
            self.text_artist.set_visible(True)

    def _sync_geometry(self, event=None):
        if self.container is None or self.host_widget is None:
            return
        bbox = self.panel.ax.get_window_extent()
        host_height = self.host_widget.winfo_height()
        x0 = int(round(bbox.x0 + 6))
        y0 = int(round(host_height - bbox.y1 + self.panel.header_frac * bbox.height + 4))
        width = int(round(bbox.width - 12))
        height = int(round(bbox.height * (1.0 - self.panel.header_frac) - 8))
        if width <= 20 or height <= 20:
            self.container.place_forget()
            return
        self.container.place(x=x0, y=y0, width=width, height=height)

    def append(self, message):
        self.lines.append(str(message))
        if self.text_widget is None:
            self.text_artist.set_text("\n".join(self.lines))
            return
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert(tk.END, "\n".join(self.lines))
        self.text_widget.see(tk.END)
        self.text_widget.configure(state=tk.DISABLED)


def load_default_settings():
    if not DEFAULT_SETTINGS_PATH.exists():
        return dict(DEFAULT_IMAGE_CONFIG)
    try:
        data = json.loads(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
        path_str = data.get("path")
        autoload = bool(data.get("autoload", True))
        if path_str:
            return {
                "path": Path(path_str),
                "autoload": autoload,
                "width_um": data.get("width_um"),
                "height_um": data.get("height_um"),
                "scale_um_per_px": data.get("scale_um_per_px"),
            }
    except Exception:
        pass
    return dict(DEFAULT_IMAGE_CONFIG)


def save_default_settings(default_info, autoload=True):
    path = Path(default_info["path"])
    payload = {
        "path": str(path),
        "autoload": bool(autoload),
        "width_um": default_info.get("width_um"),
        "height_um": default_info.get("height_um"),
        "scale_um_per_px": default_info.get("scale_um_per_px"),
    }
    DEFAULT_SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


runtime_default_config = load_default_settings()

state = AFMState(stage_surface_image, width_um, height_um)
state.default_image_path = str(runtime_default_config["path"])
state.default_image_width_um = runtime_default_config.get("width_um")
state.default_image_height_um = runtime_default_config.get("height_um")
if runtime_default_config.get("scale_um_per_px") is not None:
    state.default_scale_um_per_px = float(runtime_default_config["scale_um_per_px"])
stage = NanoPositioner(log_file="movement_log.csv")
data = TrajectoryData()

fig, ax = setup_figure()
fig.suptitle("AFM Hysteresis Simulation Control Panel", fontsize=14, fontweight="bold", color="#22364d")

manager = getattr(fig.canvas, "manager", None)
if manager is not None:
    try:
        manager.set_window_title("AFM Hysteresis Simulation Control Panel")
    except Exception:
        pass
    try:
        manager.window.wm_geometry("1600x900")
    except Exception:
        try:
            manager.resize(1600, 900)
        except Exception:
            pass

initial_fov, initial_outside_mask, initial_ix, initial_iy = create_stage_fov(
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
state.current_camera_view, _ = render_camera_recognition_frame(
    initial_fov,
    outside_mask=initial_outside_mask,
    focus_model=state.get_focus_model(),
    fov_width_um=state.fov_width,
    fov_height_um=state.fov_height,
    body_width_um=state.probe_body_width_um,
    tip_width_um=state.probe_tip_width_um,
    tip_total_length_um=state.probe_tip_total_length_um,
    triangular_tip_length_um=state.probe_triangular_tip_length_um,
    visible_body_depth_um=state.probe_visible_body_depth_um,
)
img = ax.imshow(
    rotate_camera_frame(
        render_camera_frame(
            initial_fov,
            state.camera_resolution,
            outside_mask=initial_outside_mask,
            focus_model=state.get_focus_model(),
        )[0],
        state.surface_tilt_angle,
    ),
    cmap="gray",
    extent=[initial_ix, initial_ix + state.fov_width, initial_iy + state.fov_height, initial_iy],
    origin="upper",
)

ideal_line, = ax.plot([], [], "g-", linewidth=2, alpha=0.7)
hyst_line, = ax.plot([], [], "b-", linewidth=2, alpha=0.7)
scale_bar_black, = ax.plot([], [], color="black", linewidth=6.0, solid_capstyle="butt", zorder=25)
scale_bar_white, = ax.plot([], [], color="white", linewidth=6.0, solid_capstyle="butt", zorder=26)
scale_bar_text = ax.text(
    0,
    0,
    "",
    color="white",
    fontsize=9,
    ha="center",
    va="bottom",
    zorder=25,
    bbox=dict(boxstyle="round,pad=0.15", facecolor=(0, 0, 0, 0.35), edgecolor="none"),
)
origin_marker, = ax.plot([], [], marker="+", markersize=14, markeredgewidth=2.0, color="#ffbf00", linestyle="None", zorder=27)
origin_text = ax.text(
    0,
    0,
    "",
    color="#ffdd57",
    fontsize=9,
    ha="left",
    va="bottom",
    zorder=27,
    bbox=dict(boxstyle="round,pad=0.15", facecolor=(0, 0, 0, 0.35), edgecolor="none"),
)
tip_marker, = ax.plot([], [], marker="o", markersize=5, markeredgewidth=1.2, markerfacecolor="#8ef9f3", markeredgecolor="#0f766e", linestyle="None", zorder=28)
tip_text = ax.text(
    0,
    0,
    "",
    color="#8ef9f3",
    fontsize=9,
    ha="left",
    va="bottom",
    zorder=28,
    bbox=dict(boxstyle="round,pad=0.15", facecolor=(0, 0, 0, 0.35), edgecolor="none"),
)
scan_region = Rectangle(
    (0, 0),
    state.scan_region_size_um,
    state.scan_region_size_um,
    linewidth=1.6,
    edgecolor="#8ef9f3",
    facecolor="none",
    linestyle="--",
    zorder=28,
)
ax.add_patch(scan_region)
hud_landmark_boxes = []
hud_landmark_outlines = []
hud_landmark_lines = []
hud_landmark_labels = []
for _ in range(6):
    landmark_box = Rectangle(
        (0, 0),
        0,
        0,
        linewidth=1.5,
        edgecolor="#ff4d4d",
        facecolor="none",
        linestyle="-",
        zorder=29,
    )
    ax.add_patch(landmark_box)
    hud_landmark_boxes.append(landmark_box)
    outline, = ax.plot([], [], color="#ff4d4d", linewidth=1.5, alpha=0.95, zorder=29)
    hud_landmark_outlines.append(outline)
    distance_line, = ax.plot([], [], color="#3da5ff", linewidth=1.2, alpha=0.95, zorder=29)
    hud_landmark_lines.append(distance_line)
    label = ax.text(
        0,
        0,
        "",
        color="#7dd3ff",
        fontsize=8,
        ha="left",
        va="bottom",
        zorder=30,
        bbox=dict(boxstyle="round,pad=0.12", facecolor=(0, 0, 0, 0.25), edgecolor="none"),
    )
    hud_landmark_labels.append(label)
hud_landmark_summary = ax.text(
    0,
    0,
    "",
    color="#ffefd5",
    fontsize=8,
    ha="left",
    va="top",
    zorder=30,
    bbox=dict(boxstyle="round,pad=0.18", facecolor=(0, 0, 0, 0.28), edgecolor="none"),
)
detection_roi_circle = Circle(
    (0, 0),
    0.0,
    linewidth=1.8,
    edgecolor="#f59e0b",
    facecolor=(245 / 255.0, 158 / 255.0, 11 / 255.0, 0.08),
    linestyle="--",
    zorder=31,
    visible=False,
)
ax.add_patch(detection_roi_circle)
detection_roi_patches = []
detection_roi_labels = []
for _ in range(8):
    roi_patch = Circle(
        (0, 0),
        0.0,
        linewidth=1.8,
        edgecolor="#f59e0b",
        facecolor=(245 / 255.0, 158 / 255.0, 11 / 255.0, 0.05),
        linestyle="-",
        zorder=31,
        visible=False,
    )
    ax.add_patch(roi_patch)
    detection_roi_patches.append(roi_patch)
    roi_label = ax.text(
        0,
        0,
        "",
        color="#fbbf24",
        fontsize=8,
        ha="left",
        va="bottom",
        zorder=32,
        bbox=dict(boxstyle="round,pad=0.12", facecolor=(0, 0, 0, 0.28), edgecolor="none"),
        visible=False,
    )
    detection_roi_labels.append(roi_label)

cantilever, rod, tip, center_x_ax = setup_probe_graphics(ax)
button_objects, radio_step, status_text, activity_text, dock_manager = setup_dashboard(fig, layout_path=DOCK_LAYOUT_PATH)
relocation_panel = next((panel for panel in dock_manager.panels if panel.panel_id == "relocation"), None)
status_panel = next((panel for panel in dock_manager.panels if panel.panel_id == "status_activity"), None)
trace_panel = next((panel for panel in dock_manager.panels if panel.panel_id == "trace"), None)
status_dock = None if status_panel is None else StatusTextDock(fig, manager, status_panel, status_text)
trace_dock = None if trace_panel is None else TraceTextDock(fig, manager, trace_panel, activity_text)
relocation_help_text = None if relocation_panel is None else relocation_panel.children_by_role["relocation_help"]["artist"]
relocation_tooltips = {
    "save_ref": "Save Region: store the current low-mag context, high-mag template, zoom context, and landmark memory for later recovery.",
    "remove_sample": "Remount: simulate taking the sample out and putting it back with shift, rotation, and tilt changes.",
    "auto_origin": "Pick Origin: choose the strongest distinctive local landmark in the current viewport as the working origin reference.",
    "ml_origin": "Find Origin: search the full sample for the saved origin pattern and move toward the recognized origin if confidence is good.",
    "relocate": "Recover Site: run coarse low-mag recall, estimate rotation and offset, refine with high-mag matching, and propose the recovered site.",
    "research_patterns": "Verify Tip: re-match multiple remembered landmark patterns around the tip to confirm whether the cantilever is at the correct place.",
    "ai_recall": "AI Recall: one-click AI relocation \u2014 load site memory, recognize pattern with rotation, move cantilever, verify. Click to correct if needed.",
    "ai_zoom": "AI Zoom: recall saved zoom level, AI-recognize pattern, auto zoom-out search if not found, then move + verify.",
}


def get_hud_mode_text():
    detection_on = bool(state.show_hud_detection)
    distance_on = bool(state.show_hud_distance)
    if detection_on and distance_on:
        return "Detection + Distance"
    if detection_on:
        return "Detection Only"
    if distance_on:
        return "Distance Only"
    return "Off"


def update_detection_roi_overlay():
    for index, (patch, roi, label) in enumerate(zip(detection_roi_patches, state.detection_rois, detection_roi_labels), start=1):
        patch.center = (float(roi["center_x_um"]), float(roi["center_y_um"]))
        patch.set_radius(max(0.0, float(roi["radius_um"])))
        patch.set_visible(True)
        label.set_position(
            (
                float(roi["center_x_um"]) + max(8.0, float(roi["radius_um"]) * 0.25),
                float(roi["center_y_um"]) - max(8.0, float(roi["radius_um"]) * 0.25),
            )
        )
        label.set_text(f"R{index}")
        label.set_visible(True)
    for patch in detection_roi_patches[len(state.detection_rois):]:
        patch.set_visible(False)
        patch.set_radius(0.0)
    for label in detection_roi_labels[len(state.detection_rois):]:
        label.set_visible(False)
        label.set_text("")

    if state.detection_roi_drag_active and state.detection_roi_center_x_um is not None and state.detection_roi_center_y_um is not None:
        detection_roi_circle.center = (
            float(state.detection_roi_center_x_um),
            float(state.detection_roi_center_y_um),
        )
        detection_roi_circle.set_radius(max(0.0, float(state.detection_roi_radius_um)))
        detection_roi_circle.set_visible(True)
        return
    detection_roi_circle.set_visible(False)
    detection_roi_circle.set_radius(0.0)


def begin_detection_roi_draw():
    state.detection_roi_draw_mode = True
    state.detection_roi_drag_active = False
    log_message("Detection ROI draw mode: click and drag on the viewport to add one circular ROI.")


def clear_detection_roi(log_change=True):
    state.detection_rois = []
    state.detection_roi_draw_mode = False
    state.detection_roi_drag_active = False
    state.detection_roi_center_x_um = None
    state.detection_roi_center_y_um = None
    state.detection_roi_radius_um = 0.0
    update_detection_roi_overlay()
    refresh_status_panel()
    if log_change:
        log_message("All detection ROIs cleared.")


def point_inside_detection_roi(x_um, y_um, roi):
    dx = float(x_um) - float(roi["center_x_um"])
    dy = float(y_um) - float(roi["center_y_um"])
    return float(dx * dx + dy * dy) <= float(roi["radius_um"]) ** 2


def select_best_match_per_roi(matches):
    rois = list(state.detection_rois)
    if not rois:
        return list(matches)

    remaining_matches = list(matches)
    selected_matches = []
    for roi_index, roi in enumerate(rois, start=1):
        candidates = [
            match for match in remaining_matches
            if point_inside_detection_roi(match["center_x_um"], match["center_y_um"], roi)
        ]
        if not candidates:
            continue
        best_match = max(
            candidates,
            key=lambda match: (
                float(match.get("score", 0.0)),
                -float(np.hypot(match["center_x_um"] - roi["center_x_um"], match["center_y_um"] - roi["center_y_um"])),
            ),
        )
        best_match = dict(best_match)
        best_match["roi_index"] = roi_index
        selected_matches.append(best_match)
        remaining_matches = [
            match for match in remaining_matches
            if not (
                np.isclose(match["center_x_um"], best_match["center_x_um"])
                and np.isclose(match["center_y_um"], best_match["center_y_um"])
                and int(match["index"]) == int(best_match["index"])
            )
        ]
    return selected_matches


def find_nearest_detection_roi(x_um, y_um, margin_um=30.0):
    best_index = None
    best_distance = None
    for index, roi in enumerate(state.detection_rois):
        distance_to_center = float(np.hypot(float(x_um) - float(roi["center_x_um"]), float(y_um) - float(roi["center_y_um"])))
        edge_distance = abs(distance_to_center - float(roi["radius_um"]))
        inside_distance = float(roi["radius_um"]) - distance_to_center
        hit_distance = edge_distance if inside_distance < 0.0 else min(edge_distance, inside_distance)
        if hit_distance <= float(margin_um) and (best_distance is None or hit_distance < best_distance):
            best_index = index
            best_distance = hit_distance
    return best_index


def remove_detection_roi(index, log_change=True):
    if index < 0 or index >= len(state.detection_rois):
        return
    removed_roi = state.detection_rois.pop(index)
    update_detection_roi_overlay()
    refresh_status_panel()
    if log_change:
        log_message(
            f"Detection ROI {index + 1} removed: "
            f"center=({removed_roi['center_x_um']:.1f}, {removed_roi['center_y_um']:.1f}) um, "
            f"radius={removed_roi['radius_um']:.1f} um"
        )


def summarize_filtered_landmark_report(report, matches):
    filtered_matches = list(matches)
    pair_errors_um = []
    for i in range(len(filtered_matches)):
        for j in range(i + 1, len(filtered_matches)):
            match_a = filtered_matches[i]
            match_b = filtered_matches[j]
            reference_distance = float(
                np.hypot(
                    match_a["reference_abs_x_um"] - match_b["reference_abs_x_um"],
                    match_a["reference_abs_y_um"] - match_b["reference_abs_y_um"],
                )
            )
            current_distance = float(
                np.hypot(
                    match_a["center_x_um"] - match_b["center_x_um"],
                    match_a["center_y_um"] - match_b["center_y_um"],
                )
            )
            pair_errors_um.append(abs(current_distance - reference_distance))

    distance_errors_um = [item["distance_error_um"] for item in filtered_matches if item.get("distance_error_um") is not None]
    angle_errors_deg = [item["angle_error_deg"] for item in filtered_matches if item.get("angle_error_deg") is not None]
    mean_pair_error_um = float(np.mean(pair_errors_um)) if pair_errors_um else None
    mean_distance_error_um = float(np.mean(distance_errors_um)) if distance_errors_um else None
    mean_angle_error_deg = float(np.mean(angle_errors_deg)) if angle_errors_deg else None
    mean_score = float(np.mean([item["score"] for item in filtered_matches])) if filtered_matches else 0.0
    mean_gap = float(np.mean([item["score_gap"] for item in filtered_matches])) if filtered_matches else 0.0
    geometry_confidence = 0.0
    distance_confidence = 0.0
    if filtered_matches:
        pair_term = 1.0 if mean_pair_error_um is None else float(np.clip(1.0 - mean_pair_error_um / 80.0, 0.0, 1.0))
        distance_term = 1.0 if mean_distance_error_um is None else float(np.clip(1.0 - mean_distance_error_um / 80.0, 0.0, 1.0))
        score_term = float(np.clip(0.75 * mean_score + 0.25 * min(mean_gap * 5.0, 1.0), 0.0, 1.0))
        geometry_confidence = float(np.clip(0.45 * pair_term + 0.35 * distance_term + 0.20 * score_term, 0.0, 1.0))
        distance_confidence = float(np.clip(0.65 * distance_term + 0.35 * score_term, 0.0, 1.0))
    return {
        **report,
        "matches": filtered_matches,
        "matched_count": len(filtered_matches),
        "pair_count": len(pair_errors_um),
        "mean_score": mean_score,
        "mean_score_gap": mean_gap,
        "mean_pair_error_um": mean_pair_error_um,
        "mean_distance_error_um": mean_distance_error_um,
        "mean_angle_error_deg": mean_angle_error_deg,
        "geometry_confidence": geometry_confidence,
        "distance_confidence": distance_confidence,
    }


def extract_landmark_outline(match, source_view):
    if source_view is None or source_view.size == 0:
        return None

    local_x0 = int(round(float(match["top_left_x_um"]) - float(state.x)))
    local_y0 = int(round(float(match["top_left_y_um"]) - float(state.y)))
    local_w = max(1, int(round(float(match["width_um"]))))
    local_h = max(1, int(round(float(match["height_um"]))))

    pad_x = max(4, int(round(local_w * 0.15)))
    pad_y = max(4, int(round(local_h * 0.15)))
    x0 = max(0, local_x0 - pad_x)
    y0 = max(0, local_y0 - pad_y)
    x1 = min(source_view.shape[1], local_x0 + local_w + pad_x)
    y1 = min(source_view.shape[0], local_y0 + local_h + pad_y)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None

    roi = source_view[y0:y1, x0:x1]
    if roi.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = np.asarray(roi, dtype=np.uint8)
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    masks = []
    for thresh_mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, thresholded = cv2.threshold(blurred, 0, 255, thresh_mode | cv2.THRESH_OTSU)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
        masks.append(thresholded)

    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    masks.append(edges)

    roi_center = np.array([(x1 - x0) / 2.0, (y1 - y0) / 2.0], dtype=float)
    best_contour = None
    best_score = -np.inf
    min_area = max(12.0, 0.015 * float((x1 - x0) * (y1 - y0)))

    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            centroid = contour.reshape(-1, 2).mean(axis=0)
            center_distance = float(np.linalg.norm(centroid - roi_center))
            score = area - 2.5 * center_distance
            if score > best_score:
                best_score = score
                best_contour = contour

    if best_contour is None:
        return None

    points = best_contour.reshape(-1, 2).astype(float)
    abs_x = float(state.x) + float(x0) + points[:, 0]
    abs_y = float(state.y) + float(y0) + points[:, 1]
    if len(abs_x) > 1:
        abs_x = np.append(abs_x, abs_x[0])
        abs_y = np.append(abs_y, abs_y[0])
    return abs_x, abs_y


def refresh_status_panel():
    tip_x, tip_y = get_tip_wrapper()
    target_center_x, target_center_y = callbacks.get_target_center()
    site_memory = state.site_memory or {}
    low_landmark_count = len(site_memory.get("lowmag_landmarks", [])) if site_memory else 0
    high_landmark_count = len(site_memory.get("highmag_landmarks", [])) if site_memory else 0
    site_memory_line = (
        f"Saved Site Memory: {site_memory.get('site_id', 'not saved')}"
        if site_memory
        else "Saved Site Memory: not saved"
    )
    remount_shift_line = (
        f"Simulated Remount Shift: dX={state.simulated_sample_shift_x_um:+7.1f}  dY={state.simulated_sample_shift_y_um:+7.1f}"
    )
    remount_rotation_line = f"Simulated Remount Rotation: {state.simulated_sample_rotation_deg:+7.2f} deg"
    remount_tilt_line = f"Simulated Remount Tilt: {state.simulated_sample_tilt_deg:+7.2f} deg"
    if state.last_affine_transform_report:
        affine_line = (
            f"Last Affine Recovery: dTheta={state.last_affine_transform_report.get('rotation_deg', 0.0):+6.2f}  "
            f"conf={state.last_affine_transform_report.get('confidence', 0.0):.3f}"
        )
    else:
        affine_line = "Last Affine Recovery: not run"
    if state.last_relocation_report:
        verification = state.last_relocation_report.get("verification") or {}
        relocation_line = (
            f"Last Relocation: {'Verified' if verification.get('verified') else 'Not Verified'}  "
            f"score={verification.get('reference_score', 0.0):.3f}  "
            f"gap={verification.get('reference_score_gap', 0.0):.3f}"
        )
    else:
        relocation_line = "Last Relocation: not run"
    phase2_line = (
        f"Phase 2 Models: same-site={'Yes' if callbacks.same_site_classifier is not None else 'No'}  "
        f"remount={'Yes' if callbacks.remount_transform_predictor is not None else 'No'}  "
        f"retrieval={'Yes' if callbacks.lowmag_embedding_index is not None else 'No'}"
    )
    if state.origin_defined:
        origin_line = f"{state.origin_label}: X={state.origin_x:7.1f}  Y={state.origin_y:7.1f}"
        pos_rel_line = f"Probe Tip Relative To Origin: dX={tip_x - state.origin_x:+7.1f}  dY={tip_y - state.origin_y:+7.1f}"
        tgt_rel_line = f"Target Relative To Origin: dX={target_center_x - state.origin_x:+7.1f}  dY={target_center_y - state.origin_y:+7.1f}"
    else:
        origin_line = "Named Origin: not set"
        pos_rel_line = "Probe Tip Relative To Origin: not set"
        tgt_rel_line = "Target Relative To Origin: not set"

    digital_zoom = state.get_digital_zoom_level()
    optical_mag = state.get_current_objective_magnification()
    focus_metrics = get_defocus_metrics(state.get_focus_model(), (state.camera_resolution[1], state.camera_resolution[0]))
    effective_camera_stage_um = state.get_effective_camera_stage_position_um()
    probe_gap_um = state.get_probe_sample_gap_um()
    dof_mode = "Manual" if state.manual_dof_camera_um is not None else "Automatic"
    hud_landmark_report = state.hud_landmark_report or {}
    hud_mode_line = f"HUD Overlay Mode: {get_hud_mode_text()}"
    if state.detection_rois:
        detection_area_line = f"Detection ROIs: {len(state.detection_rois)} active  |  one ROI -> one pattern"
    elif state.detection_roi_draw_mode:
        detection_area_line = "Detection ROIs: drawing mode active"
    else:
        detection_area_line = "Detection ROIs: full viewport"
    if hud_landmark_report:
        hud_landmark_line = (
            f"HUD Landmark Matches: {hud_landmark_report.get('matched_count', 0)}  "
            f"geometry={hud_landmark_report.get('geometry_confidence', 0.0):.3f}"
        )
        if hud_landmark_report.get("mean_angle_error_deg") is not None:
            hud_landmark_line += f"  angle err={hud_landmark_report['mean_angle_error_deg']:.1f} deg"
    else:
        hud_landmark_line = "HUD Landmark Matches: not active"

    entries = [
        {"text": "Surface And Camera", "is_heading": True},
        {
            "text": f"Surface Source: {state.sample_source}",
            "tooltip": "Shows where the current sample surface came from, such as a synthetic surface, a default image, or a user-loaded microscope image.",
        },
        {
            "text": f"Image File: {Path(state.sample_path).name if state.sample_path else 'synthetic surface'}",
            "tooltip": "Shows the filename of the loaded sample image. If no file was loaded, the simulation is using the built-in synthetic surface.",
        },
        {
            "text": f"Camera Mode: {state.camera_mode}",
            "tooltip": "Shows which camera rendering mode is currently active for the viewport display.",
        },
        {
            "text": f"Digital Zoom: {digital_zoom:7g}x",
            "tooltip": "Shows the current digital zoom multiplier applied to the viewport.",
        },
        {
            "text": f"Objective Magnification: {optical_mag:7g}x",
            "tooltip": "Shows the simulated microscope objective magnification currently associated with the selected zoom level.",
        },
        {
            "text": f"Live Camera Resolution: {state.camera_resolution[0]} x {state.camera_resolution[1]} pixels",
            "tooltip": "Shows the pixel resolution used for the live viewport image.",
        },
        {
            "text": f"Reference Camera Resolution: {state.camera_reference_resolution[0]} x {state.camera_reference_resolution[1]} pixels",
            "tooltip": "Shows the resolution used when saving or comparing reference camera images.",
        },
        {
            "text": f"Auxiliary Camera Resolution: {state.sample_view_camera_resolution[0]} x {state.sample_view_camera_resolution[1]} pixels",
            "tooltip": "Shows the resolution for the auxiliary sample-view camera display.",
        },
        {"text": "", "tooltip": ""},
        {"text": "Position And Motion", "is_heading": True},
        {
            "text": f"Probe Tip Position: X={tip_x:7.1f}  Y={tip_y:7.1f}",
            "tooltip": "Shows the current cantilever tip position on the stage in micrometers.",
        },
        {
            "text": f"Target Center: X={target_center_x:7.1f}  Y={target_center_y:7.1f}",
            "tooltip": "Shows the target viewport center that motion control is trying to reach.",
        },
        {
            "text": origin_line,
            "tooltip": "Shows the named origin point used as the reference for relative coordinates. Set this from the viewport when you want meaningful local offsets.",
        },
        {
            "text": pos_rel_line,
            "tooltip": "Shows the current probe tip position relative to the named origin. Positive and negative values indicate offset direction from that reference.",
        },
        {
            "text": tgt_rel_line,
            "tooltip": "Shows the target position center relative to the named origin.",
        },
        {
            "text": f"Step Size: {state.current_step:7.1f} micrometers",
            "tooltip": "Shows how far one manual movement command shifts the target in the selected direction.",
        },
        {
            "text": f"Smooth Move Speed: {state.smooth_move_step:7.1f} micrometers per frame",
            "tooltip": "Shows the step size used for smooth animated motion between the current position and the target.",
        },
        {
            "text": f"Automatic Scan: {'On' if state.auto_scan_active else 'Off'}",
            "tooltip": "Shows whether the automatic scan routine is currently active.",
        },
        {
            "text": f"Proportional-Integral Compensation: {'On' if state.pi_mode else 'Off'}",
            "tooltip": "Shows whether PI-based hysteresis compensation is enabled for motion commands.",
        },
        {
            "text": f"Motion Pause: {'Yes' if state.paused else 'No'}",
            "tooltip": "Shows whether motion updates are currently paused.",
        },
        {"text": "", "tooltip": ""},
        {"text": "Repositioning Memory", "is_heading": True},
        {
            "text": site_memory_line,
            "tooltip": "Shows whether a structured scan-site memory has been saved for relocation. The saved memory includes the reference image, overview, and landmark patches.",
        },
        {
            "text": f"Low-Mag Landmarks Saved: {low_landmark_count}",
            "tooltip": "Shows how many coarse low-magnification landmarks are stored in the current site memory for coarse localization after remounting.",
        },
        {
            "text": f"High-Mag Landmarks Saved: {high_landmark_count}",
            "tooltip": "Shows how many high-magnification landmarks are stored near the scan site for fine relocation and verification.",
        },
        {
            "text": remount_shift_line,
            "tooltip": "Shows the simulated sample shift relative to the stage after the most recent remount action.",
        },
        {
            "text": remount_rotation_line,
            "tooltip": "Shows the simulated in-plane sample rotation introduced by the most recent remount action.",
        },
        {
            "text": remount_tilt_line,
            "tooltip": "Shows the simulated sample tilt angle introduced by the most recent remount action.",
        },
        {
            "text": affine_line,
            "tooltip": "Shows the latest low-magnification affine remount estimate, including the recovered rotation and its confidence.",
        },
        {
            "text": relocation_line,
            "tooltip": "Shows the outcome of the most recent relocation attempt, including whether verification passed and the final reference-match quality.",
        },
        {
            "text": phase2_line,
            "tooltip": "Shows whether the optional Phase 2 AI models are currently available for same-site scoring, remount prediction, and low-magnification retrieval.",
        },
        {"text": "", "tooltip": ""},
        {"text": "Focus And Stage Geometry", "is_heading": True},
        {
            "text": f"Sample Stage Z Position: {state.z_stage_position_um:+7.1f} micrometers",
            "tooltip": "Shows the vertical position of the sample stage relative to its nominal center position.",
        },
        {
            "text": f"Camera Stage Z Position: {effective_camera_stage_um:+7.1f} micrometers",
            "tooltip": "Shows the effective vertical position of the camera and cantilever assembly used by the focus model.",
        },
        {
            "text": f"Probe-To-Sample Gap: {probe_gap_um:+7.1f} micrometers",
            "tooltip": "Shows the current vertical separation between the cantilever tip reference plane and the sample surface stage position.",
        },
        {
            "text": f"Focus Plane Position: {state.focus_z_um:+7.1f} micrometers",
            "tooltip": "Shows the best-focus plane used by the optical focus model.",
        },
        {
            "text": f"Depth Of Field: {focus_metrics['dof_camera_um']:5.2f} micrometers ({dof_mode})",
            "tooltip": "Shows the current depth of field. Automatic means it is derived from the optics model, while Manual means it was overridden by the user.",
        },
        {
            "text": f"Blur Diameter: {focus_metrics['blur_diameter_um']:7.2f} micrometers  {focus_metrics['blur_diameter_px']:6.2f} pixels",
            "tooltip": "Shows the estimated blur spot size caused by defocus, expressed both in stage-space micrometers and in image pixels.",
        },
        {
            "text": f"Surface Tilt Angle: {state.surface_tilt_angle:7.1f} degrees",
            "tooltip": "Shows the rotation angle applied to the stage surface image for the viewport rendering.",
        },
        {
            "text": f"Field Of View Size: {state.fov_width} x {state.fov_height} micrometers",
            "tooltip": "Shows the current physical width and height covered by the viewport.",
        },
        {"text": "", "tooltip": ""},
        {"text": "Display And Overlay", "is_heading": True},
        {
            "text": hud_mode_line,
            "tooltip": "Shows whether the HUD is displaying landmark detection outlines, tip-to-landmark distance geometry, both overlays, or neither.",
        },
        {
            "text": f"Detection Outlines: {'On' if state.show_hud_detection else 'Off'}  |  Distance Geometry: {'On' if state.show_hud_distance else 'Off'}",
            "tooltip": "Shows the two independently controlled HUD overlay layers. The HUD button cycles through both on, detection only, distance only, and off.",
        },
        {
            "text": detection_area_line,
            "tooltip": "Shows the current user-drawn detection ROIs. Right-click the viewport and choose Draw Detection ROI to add circular regions that each keep only one best detection pattern.",
        },
        {
            "text": hud_landmark_line,
            "tooltip": "Shows how many remembered high-magnification landmark patterns are currently re-matched in the HUD and the geometry-consistency confidence built from their relative spacing.",
        },
    ]
    if status_dock is not None:
        status_dock.set_entries(entries)
    else:
        status_text.set_text("\n".join(entry["text"] for entry in entries))
    update_origin_overlay()
    update_tip_overlay()


def log_message(message):
    print(message)
    if trace_dock is not None:
        trace_dock.append(message)
    elif activity_text is not None:
        activity_text.set_text(str(message))
    refresh_status_panel()
    fig.canvas.draw_idle()


def log_button_press(label):
    log_message(f"[Button] {label}")


def bind_logged_button(button_key, label, callback):
    def handle_click(event):
        log_button_press(label)
        callback(event)

    button_objects[button_key].on_clicked(handle_click)


def update_origin_overlay():
    if not state.origin_defined:
        origin_marker.set_data([], [])
        origin_text.set_text("")


def update_relocation_hover_help(event):
    if relocation_help_text is None:
        return
    hovered_key = None
    for key, tooltip in relocation_tooltips.items():
        if event.inaxes == button_objects[key].ax:
            hovered_key = key
            relocation_help_text.set_text(tooltip)
            fig.canvas.draw_idle()
            break
    if hovered_key is None and relocation_help_text.get_text() != "Hover over a relocation button to see what it does.":
        relocation_help_text.set_text("Hover over a relocation button to see what it does.")
        fig.canvas.draw_idle()
        return

    ox = float(state.origin_x)
    oy = float(state.origin_y)
    in_view = state.x <= ox <= state.x + state.fov_width and state.y <= oy <= state.y + state.fov_height
    if in_view:
        origin_marker.set_data([ox], [oy])
        origin_text.set_position((ox + state.fov_width * 0.015, oy - state.fov_height * 0.02))
        origin_text.set_text(state.origin_label)
    else:
        origin_marker.set_data([], [])
        origin_text.set_text("")


def update_tip_overlay():
    tip_x, tip_y = get_tip_wrapper()
    hud_visible = bool(state.show_probe_hud)
    update_detection_roi_overlay()
    tip_marker.set_visible(hud_visible)
    tip_text.set_visible(hud_visible)
    scan_region.set_visible(hud_visible)

    if not hud_visible:
        tip_marker.set_data([], [])
        tip_text.set_text("")
        update_landmark_overlay(tip_x, tip_y, hud_visible)
        return

    half_scan = float(state.scan_region_size_um) / 2.0
    tip_marker.set_data([tip_x], [tip_y])
    tip_text.set_position((tip_x + state.fov_width * 0.015, tip_y - state.fov_height * 0.02))
    tip_text.set_text("Cantilever Tip")
    scan_region.set_xy((tip_x - half_scan, tip_y - half_scan))
    scan_region.set_width(state.scan_region_size_um)
    scan_region.set_height(state.scan_region_size_um)
    update_landmark_overlay(tip_x, tip_y, hud_visible)


def update_landmark_overlay(tip_x, tip_y, hud_visible):
    for box in hud_landmark_boxes:
        box.set_visible(False)
    for outline in hud_landmark_outlines:
        outline.set_visible(False)
        outline.set_data([], [])
    for line in hud_landmark_lines:
        line.set_visible(False)
        line.set_data([], [])
    for label in hud_landmark_labels:
        label.set_visible(False)
        label.set_text("")
    hud_landmark_summary.set_visible(False)
    hud_landmark_summary.set_text("")

    if not hud_visible or not state.hud_landmark_overlay_enabled:
        state.hud_landmark_matches = []
        state.hud_landmark_report = None
        return

    site_memory = state.site_memory or {}
    reference_landmarks = site_memory.get("highmag_landmarks") or []
    current_view = state.current_camera_view if state.current_camera_view is not None else state.current_fov_raw
    if not reference_landmarks or current_view is None:
        state.hud_landmark_matches = []
        state.hud_landmark_report = None
        return

    report = analyze_landmark_geometry(
        reference_landmarks,
        current_view,
        view_origin_x_um=float(state.x),
        view_origin_y_um=float(state.y),
        tip_x_um=float(tip_x),
        tip_y_um=float(tip_y),
        min_score=max(0.38, float(state.relocation_min_match_score) - 0.04),
        min_gap=max(0.01, float(state.relocation_min_score_gap) - 0.01),
    )
    if state.detection_rois:
        filtered_matches = select_best_match_per_roi(report.get("matches", []))
        report = summarize_filtered_landmark_report(report, filtered_matches)
    state.hud_landmark_matches = list(report.get("matches", []))
    state.hud_landmark_report = report
    matches = state.hud_landmark_matches
    if not matches:
        return

    outline_view = state.current_fov_raw if state.current_fov_raw is not None else current_view
    show_detection = bool(state.show_hud_detection)
    show_distance = bool(state.show_hud_distance)

    for index, match in enumerate(matches[: len(hud_landmark_boxes)]):
        box = hud_landmark_boxes[index]
        outline = hud_landmark_outlines[index]
        line = hud_landmark_lines[index]
        label = hud_landmark_labels[index]
        x0 = float(match["top_left_x_um"])
        y0 = float(match["top_left_y_um"])
        width = float(match["width_um"])
        height = float(match["height_um"])
        center_x = float(match["center_x_um"])
        center_y = float(match["center_y_um"])
        if show_detection:
            outline_points = extract_landmark_outline(match, outline_view)
            if outline_points is None:
                box.set_xy((x0, y0))
                box.set_width(width)
                box.set_height(height)
                box.set_visible(True)
            else:
                outline.set_data(outline_points[0], outline_points[1])
                outline.set_visible(True)
        if show_distance:
            line.set_data([tip_x, center_x], [tip_y, center_y])
            line.set_visible(True)
        label.set_position((center_x + state.fov_width * 0.01, center_y - state.fov_height * 0.015))
        label_parts = [f"R{match.get('roi_index', index + 1)}", f"L{match['index']}"]
        if show_distance and match.get("tip_distance_um") is not None:
            distance_text = f"{match['tip_distance_um']:.0f} um"
            if match.get("tip_angle_deg") is not None:
                distance_text += f"  {match['tip_angle_deg']:+.1f} deg"
            if match.get("distance_error_um") is not None:
                distance_text += f"  dE={match['distance_error_um']:.0f}"
            if match.get("angle_error_deg") is not None:
                distance_text += f"  aE={match['angle_error_deg']:.1f}"
            label_parts.append(distance_text)
        elif show_detection:
            label_parts.append("outline")
        label.set_text("  ".join(label_parts).rstrip())
        label.set_visible(show_detection or show_distance)

    summary_parts = [f"HUD landmarks: {report['matched_count']}"]
    if report.get("mean_pair_error_um") is not None:
        summary_parts.append(f"pair err {report['mean_pair_error_um']:.1f} um")
    if report.get("mean_distance_error_um") is not None:
        summary_parts.append(f"tip err {report['mean_distance_error_um']:.1f} um")
    if report.get("mean_angle_error_deg") is not None:
        summary_parts.append(f"angle err {report['mean_angle_error_deg']:.1f} deg")
    summary_parts.append(f"geom {report['geometry_confidence']:.2f}")
    hud_landmark_summary.set_position((state.x + state.fov_width * 0.02, state.y + state.fov_height * 0.06))
    hud_landmark_summary.set_text(" | ".join(summary_parts))
    hud_landmark_summary.set_visible(True)


def show_viewport_context_menu(event):
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return
    if getattr(event, "button", None) != 3:
        return
    if manager is None or getattr(manager, "window", None) is None:
        return

    clicked_x = float(event.xdata)
    clicked_y = float(event.ydata)
    nearest_roi_index = find_nearest_detection_roi(clicked_x, clicked_y)
    menu = tk.Menu(manager.window, tearoff=0)

    def set_origin_here():
        label = simpledialog.askstring(
            "Origin Label",
            "Enter origin label:",
            initialvalue=state.origin_label if state.origin_defined else "Origin",
            parent=manager.window,
        )
        callbacks.set_origin(clicked_x, clicked_y, label=label or "Origin")

    def remove_clicked_roi():
        if nearest_roi_index is not None:
            remove_detection_roi(nearest_roi_index)

    menu.add_command(label="Draw Detection ROI", command=begin_detection_roi_draw)
    menu.add_command(label="Set Origin Here", command=set_origin_here)
    if nearest_roi_index is not None:
        menu.add_command(label=f"Delete Detection ROI {nearest_roi_index + 1}", command=remove_clicked_roi)
    if state.detection_rois or state.detection_roi_draw_mode:
        menu.add_command(label="Clear Detection ROIs", command=clear_detection_roi)
    if state.origin_defined:
        menu.add_command(label="Clear Origin", command=callbacks.clear_origin)
    try:
        menu.tk_popup(int(event.guiEvent.x_root), int(event.guiEvent.y_root))
    finally:
        menu.grab_release()


def get_tip_wrapper():
    return get_tip_position(tip, ax)


def update_title_wrapper():
    tip_x, tip_y = get_tip_wrapper()
    target_center_x = float(state.target_x + state.fov_width / 2.0)
    target_center_y = float(state.target_y + state.fov_height / 2.0)
    update_title(ax, fig, state.pi_mode, target_center_x, target_center_y, tip_x, tip_y)


def handle_detection_roi_press(event):
    if not state.detection_roi_draw_mode:
        return
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return
    if getattr(event, "button", None) != 1:
        return
    state.detection_roi_drag_active = True
    state.detection_roi_center_x_um = float(event.xdata)
    state.detection_roi_center_y_um = float(event.ydata)
    state.detection_roi_radius_um = 0.0
    update_detection_roi_overlay()
    fig.canvas.draw_idle()


def handle_detection_roi_motion(event):
    if not state.detection_roi_drag_active:
        return
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return
    dx = float(event.xdata) - float(state.detection_roi_center_x_um)
    dy = float(event.ydata) - float(state.detection_roi_center_y_um)
    state.detection_roi_radius_um = float(np.hypot(dx, dy))
    update_detection_roi_overlay()
    fig.canvas.draw_idle()


def handle_detection_roi_release(event):
    if not state.detection_roi_drag_active:
        return
    state.detection_roi_drag_active = False
    state.detection_roi_draw_mode = False
    if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
        dx = float(event.xdata) - float(state.detection_roi_center_x_um)
        dy = float(event.ydata) - float(state.detection_roi_center_y_um)
        state.detection_roi_radius_um = float(np.hypot(dx, dy))
    if float(state.detection_roi_radius_um) < 5.0:
        state.detection_roi_center_x_um = None
        state.detection_roi_center_y_um = None
        state.detection_roi_radius_um = 0.0
        update_detection_roi_overlay()
        log_message("Detection ROI draw cancelled because the circle was too small.")
        return
    state.detection_rois.append(
        {
            "center_x_um": float(state.detection_roi_center_x_um),
            "center_y_um": float(state.detection_roi_center_y_um),
            "radius_um": float(state.detection_roi_radius_um),
        }
    )
    state.detection_roi_center_x_um = None
    state.detection_roi_center_y_um = None
    state.detection_roi_radius_um = 0.0
    update_detection_roi_overlay()
    refresh_status_panel()
    log_message(
        f"Detection ROI {len(state.detection_rois)} set: "
        f"center=({state.detection_rois[-1]['center_x_um']:.1f}, {state.detection_rois[-1]['center_y_um']:.1f}) um, "
        f"radius={state.detection_rois[-1]['radius_um']:.1f} um"
    )
    fig.canvas.draw_idle()


callbacks = AFMCallbacks(
    state,
    stage,
    fig,
    ax,
    tip,
    cantilever,
    rod,
    center_x_ax,
    data,
    update_title_wrapper,
    get_tip_wrapper,
    button_objects,
    artifact_layer,
)
callbacks.img = img
callbacks.set_log_callback(log_message)
callbacks.set_status_callback(refresh_status_panel)
callbacks.set_persist_default_callback(lambda default_info: save_default_settings(default_info, autoload=True))
callbacks.update_probe_visuals()

bind_logged_button("up", "Move Up", callbacks.move_up)
bind_logged_button("down", "Move Down", callbacks.move_down)
bind_logged_button("left", "Move Left", callbacks.move_left)
bind_logged_button("right", "Move Right", callbacks.move_right)
bind_logged_button("pause_motion", "Toggle Motion Pause", callbacks.toggle_pause)
bind_logged_button("stop_move", "Stop Motion Here", callbacks.stop_motion)
bind_logged_button("jump_target", "Jump To Destination", callbacks.jump_to_destination)

bind_logged_button("pi", "Toggle PI Compensation", callbacks.toggle_pi)
bind_logged_button("auto", "Start Auto Scan", callbacks.start_auto_scan)
bind_logged_button("clear", "Clear Path", callbacks.clear_trails)

def _toggle_ml_mode(event=None):
    if event is not None and event.key != 'm':
        return
    state.force_ml_mode = not state.force_ml_mode
    log_message(f"[ML Mode] {'ON — ML narrows NCC sweep (+-4 deg)' if state.force_ml_mode else 'OFF — NCC full sweep (+-10 deg)'}")

bind_logged_button("save_ref", "Save Region Memory", callbacks.save_reference)
bind_logged_button("remove_sample", "Remount Sample", callbacks.remove_sample)
bind_logged_button("auto_origin", "Pick Local Origin", callbacks.auto_origin_unsupervised)
bind_logged_button("ml_origin", "Find Saved Origin", callbacks.ml_find_origin)
bind_logged_button("relocate", "Recover Site", callbacks.relocate)
bind_logged_button("ai_recall", "AI Recall & Recover", callbacks.ai_recall_and_recover)
bind_logged_button("ai_zoom", "AI Zoom & Recover", callbacks.ai_zoom_recover)

bind_logged_button("zoom_in", "Zoom In", callbacks.zoom_in)
bind_logged_button("zoom_out", "Zoom Out", callbacks.zoom_out)
bind_logged_button("z_down", "Move Z Down", callbacks.move_z_down)
bind_logged_button("z_up", "Move Z Up", callbacks.move_z_up)
bind_logged_button("focus_reset", "Best Focus", callbacks.reset_focus)
bind_logged_button("load_default", "Load Default Image", callbacks.load_default_image)
bind_logged_button("load_image", "Load Image", callbacks.load_sample_image)
bind_logged_button("save_default", "Save As Default", callbacks.save_current_as_default)
bind_logged_button("save_layout", "Save Dock Layout", lambda event: log_message(f"Dock layout saved to {dock_manager.save_layout()}"))
bind_logged_button("research_patterns", "Verify Tip With Landmarks", callbacks.research_patterns)
button_objects["scale_bar"].label.set_text(f"Scale Bar: {int(state.scale_bar_total_um)} um")
bind_logged_button("scale_bar", "Cycle Scale Bar", callbacks.cycle_scale_bar_length)
bind_logged_button("coord", "Show Tip Position", callbacks.show_tip_coord)
button_objects["hud"].label.set_text(callbacks.get_hud_button_label())
bind_logged_button("hud", "Toggle HUD", callbacks.toggle_probe_hud)
bind_logged_button("tilt", "Stage Tilt", callbacks.set_tilt)
fig.canvas.mpl_connect("button_press_event", handle_detection_roi_press)
fig.canvas.mpl_connect("button_press_event", callbacks.move_to_clicked_point)
fig.canvas.mpl_connect("button_press_event", show_viewport_context_menu)
fig.canvas.mpl_connect("motion_notify_event", handle_detection_roi_motion)
fig.canvas.mpl_connect("motion_notify_event", update_relocation_hover_help)
fig.canvas.mpl_connect("button_release_event", handle_detection_roi_release)


def on_step_selected(label):
    step = int(label.split()[0])
    callbacks.set_step(step)


radio_step.on_clicked(on_step_selected)

animation = AFMAnimation(
    state,
    stage,
    data,
    ax,
    img,
    ideal_line,
    hyst_line,
    artifact_layer,
    fig,
    get_tip_wrapper,
    scale_bar_black,
    scale_bar_white,
    scale_bar_text,
    refresh_status_panel,
    callbacks.update_probe_visuals,
)
ani = FuncAnimation(fig, animation.update, interval=state.animation_interval_ms, cache_frame_data=False)

is_closing = False


def on_close(event):
    global is_closing
    if is_closing:
        return
    is_closing = True
    try:
        data.save()
        data.plot()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        plt.close("all")


fig.canvas.mpl_connect("close_event", on_close)

update_title_wrapper()
update_detection_roi_overlay()
refresh_status_panel()
if dock_manager.load_layout():
    log_message(f"Loaded dock layout: {DOCK_LAYOUT_PATH.name}")
log_message("AFM control panel ready.")
log_message(
    "Start state: "
    f"zoom={state.get_digital_zoom_level():.2f}x, "
    f"HUD={callbacks.get_hud_button_label().split(': ', 1)[1]}, "
    f"tilt={state.surface_tilt_angle:+.1f} deg, "
    f"focus_offset={state.get_focus_offset_um():+.1f} um, "
    f"X={state.x:.1f} um, Y={state.y:.1f} um"
)
log_message("Definition: sample = the surface image mounted on the stage.")
log_message("Position and Target now mean viewport center on the stage, not the top-left corner.")
log_message("Right-click in the Viewport to set a named origin for relative coordinates.")
log_message("Repositioning plan: before removal, capture low-mag overview, move to the ROI, define a local origin, and save structured site memory at the exact scan site.")
log_message("Saved site memory includes the named origin, scan position, current high-mag reference image, nearby landmark patches, and the low-mag to high-mag relationship.")
log_message("After replacement, recovery follows a coarse-to-fine path: low-mag localization, regional refinement with landmarks, final high-mag verification, then acceptance only if confidence passes.")
log_message("Auto Origin is a local landmark helper in the current viewport. Find Labeled Origin is a supervised full-sample search for a previously defined origin pattern.")
log_message("High-mag pattern recognition now uses the camera-visible view, with cantilever-body occlusion applied so hidden texture under the probe is not used for matching.")
log_message("Press 'M' key to toggle ML mode (5w model vs ORB+RANSAC for coarse localization).")
fig.canvas.mpl_connect("key_press_event", _toggle_ml_mode)
log_message("Named regions: Viewport, FOV, Relocation Trace Dock, Navigation Dock, Motion Dock, Status Dock, Relocation Dock, Utility Dock.")
log_message("Suggested order: load image at low mag -> move to the region of interest -> zoom in to the scan site -> set the named origin -> save site memory -> remove and replace sample -> optionally guide near the old region -> run recovery and verification.")
if runtime_default_config.get("autoload"):
    callbacks.load_default_image()
plt.show()
