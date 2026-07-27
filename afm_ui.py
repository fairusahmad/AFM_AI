import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.widgets import Button, RadioButtons


PANEL_FACE = "#e8edf5"
PANEL_EDGE = "#9fb9d8"
HEADER_FACE = "#d7e5f6"
TEXT_COLOR = "#202020"
TITLE_COLOR = "#27476a"
SUBTITLE_COLOR = "#527294"
BUTTON_FACE = "#ffffff"
BUTTON_HOVER = "#f5f8fc"
BUTTON_EDGE = "#202020"
BUTTON_TEXT = "#202020"
BUTTON_FONT_SIZE = 10.0
BUTTON_HEIGHT_WIDE = 0.11
BUTTON_HEIGHT_COMPACT = 0.08
DEFAULT_LAYOUT_PATH = Path(__file__).resolve().parent / "afm_dock_layout.json"


class DockablePanel:
    def __init__(
        self,
        fig,
        panel_id,
        bounds,
        title,
        subtitle,
        header_frac=0.16,
        content_gap_frac=0.07,
        min_size=(0.14, 0.12),
        resize_grip_frac=0.16,
    ):
        self.fig = fig
        self.panel_id = panel_id
        self.title = title
        self.subtitle = subtitle
        self.bounds = list(bounds)
        self.home_bounds = list(bounds)
        self.header_frac = header_frac
        self.content_gap_frac = content_gap_frac
        self.button_gap_frac = 0.002
        self.min_width, self.min_height = min_size
        self.resize_grip_frac = resize_grip_frac
        self.children = []
        self.children_by_role = {}
        self.layout_callback = None

        self.ax = fig.add_axes(bounds, zorder=1)
        self.ax.set_facecolor(PANEL_FACE)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(PANEL_EDGE)
            spine.set_linewidth(1.2)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.header = Rectangle(
            (0, 1.0 - self.header_frac),
            1.0,
            self.header_frac,
            transform=self.ax.transAxes,
            facecolor=HEADER_FACE,
            edgecolor="none",
            zorder=0,
        )
        self.ax.add_patch(self.header)
        self.ax.text(0.04, 0.98, title, fontsize=10, fontweight="bold", color=TITLE_COLOR, va="top")
        self.ax.text(0.04, 0.90, subtitle, fontsize=7.9, color=SUBTITLE_COLOR, va="top")
        self.ax.text(0.96, 0.98, "Drag", fontsize=8.4, color=SUBTITLE_COLOR, ha="right", va="top")
        self.ax.text(0.975, 0.03, "Resize", fontsize=7.8, color=SUBTITLE_COLOR, ha="right", va="bottom")

    def set_layout_callback(self, callback):
        self.layout_callback = callback
        self._apply_bounds()

    def _register_child(self, child):
        self.children.append(child)
        role = child.get("role")
        if role:
            self.children_by_role[role] = child

    def _to_absolute_bounds(self, rel_bounds):
        x0, y0, w, h = self.bounds
        return [
            x0 + rel_bounds[0] * w,
            y0 + rel_bounds[1] * h,
            rel_bounds[2] * w,
            rel_bounds[3] * h,
        ]

    def _apply_inner_padding(self, rel_bounds, pad_x=None, pad_y=None):
        x, y, w, h = [float(value) for value in rel_bounds]
        pad_x = self.button_gap_frac if pad_x is None else float(pad_x)
        pad_y = self.button_gap_frac if pad_y is None else float(pad_y)
        max_pad_x = max(0.0, w / 2.0 - 0.002)
        max_pad_y = max(0.0, h / 2.0 - 0.002)
        pad_x = min(pad_x, max_pad_x)
        pad_y = min(pad_y, max_pad_y)
        return [
            x + pad_x,
            y + pad_y,
            max(0.004, w - 2.0 * pad_x),
            max(0.004, h - 2.0 * pad_y),
        ]

    def _fit_button_font(self, child):
        widget = child["widget"]
        label = getattr(widget, "label", None)
        if label is None:
            return
        label.set_fontsize(BUTTON_FONT_SIZE)

    def get_content_top(self):
        return max(0.0, 1.0 - self.header_frac - self.content_gap_frac)

    def _clamp_rel_bounds_to_content(self, rel_bounds):
        x, y, w, h = [float(value) for value in rel_bounds]
        content_top = self.get_content_top()
        max_y = max(0.0, content_top - h)
        y = min(max(0.0, y), max_y)
        return [x, y, w, h]

    def _clamp_text_y_to_content(self, rel_y):
        return min(float(rel_y), self.get_content_top())

    def add_button(self, key, label, facecolor=BUTTON_FACE, hovercolor=BUTTON_HOVER, fontsize=BUTTON_FONT_SIZE, role=None):
        default_bounds = self._apply_inner_padding([0.0, 0.0, 0.2, 0.2])
        axis = self.fig.add_axes(self._to_absolute_bounds(default_bounds), zorder=2)
        axis.set_facecolor(PANEL_FACE)
        button = Button(axis, label, color=facecolor, hovercolor=hovercolor)
        for spine in axis.spines.values():
            spine.set_edgecolor(BUTTON_EDGE)
            spine.set_linewidth(1.0)
        button.label.set_color(BUTTON_TEXT)
        button.label.set_fontsize(fontsize)
        button.label.set_fontfamily("Segoe UI")
        self._register_child(
            {
                "type": "axes",
                "kind": "button",
                "key": key,
                "role": role or key,
                "axes": axis,
                "widget": button,
                "base_fontsize": float(fontsize),
                "rel_bounds": default_bounds,
            }
        )
        return key, button

    def add_radio(self, title, labels, active=0, fontsize=BUTTON_FONT_SIZE, role=None):
        axis = self.fig.add_axes(self._to_absolute_bounds([0.0, 0.0, 0.2, 0.2]), zorder=2)
        axis.set_facecolor(PANEL_FACE)
        axis.set_title(title, fontsize=fontsize)
        radio = RadioButtons(axis, labels, active=active)
        for label in radio.labels:
            label.set_fontsize(BUTTON_FONT_SIZE)
            label.set_fontfamily("Segoe UI")
        self._register_child(
            {
                "type": "axes",
                "kind": "radio",
                "role": role or title.lower().replace(" ", "_"),
                "axes": axis,
                "widget": radio,
                "rel_bounds": [0.0, 0.0, 0.2, 0.2],
            }
        )
        return radio

    def add_text_block(self, rel_x, rel_y, text="", fontsize=8.4, family=None, linespacing=1.3, weight=None, role=None):
        text_artist = self.ax.text(
            rel_x,
            rel_y,
            text,
            transform=self.ax.transAxes,
            fontsize=fontsize,
            color=TEXT_COLOR,
            va="top",
            family=family,
            linespacing=linespacing,
            fontweight=weight,
        )
        self._register_child(
            {
                "type": "text",
                "role": role,
                "artist": text_artist,
                "rel_pos": [rel_x, rel_y],
            }
        )
        return text_artist

    def set_child_bounds(self, role, rel_bounds):
        child = self.children_by_role.get(role)
        if child and child["type"] == "axes":
            adjusted_bounds = self._clamp_rel_bounds_to_content(rel_bounds)
            if child.get("kind") == "button":
                adjusted_bounds = self._apply_inner_padding(adjusted_bounds)
            child["rel_bounds"] = adjusted_bounds

    def set_text_position(self, role, rel_x, rel_y):
        child = self.children_by_role.get(role)
        if child and child["type"] == "text":
            child["rel_pos"] = [rel_x, self._clamp_text_y_to_content(rel_y)]

    def _relayout_content(self):
        if self.layout_callback is not None:
            self.layout_callback(self)

    def _apply_bounds(self):
        self._relayout_content()
        self.ax.set_position(self.bounds)
        for child in self.children:
            if child["type"] == "axes":
                child["axes"].set_position(self._to_absolute_bounds(child["rel_bounds"]))
                if child.get("kind") == "button":
                    self._fit_button_font(child)
            elif child["type"] == "text":
                child["artist"].set_position(child["rel_pos"])

    def move_to(self, x0, y0):
        self.bounds[0] = x0
        self.bounds[1] = y0
        self._apply_bounds()

    def resize_to(self, width, height):
        self.bounds[2] = width
        self.bounds[3] = height
        self._apply_bounds()

    def clamp(self, margin=0.01):
        self.bounds[2] = min(max(self.bounds[2], self.min_width), 1.0 - 2 * margin)
        self.bounds[3] = min(max(self.bounds[3], self.min_height), 1.0 - 2 * margin)
        width = self.bounds[2]
        height = self.bounds[3]
        x0 = min(max(self.bounds[0], margin), 1.0 - width - margin)
        y0 = min(max(self.bounds[1], margin), 1.0 - height - margin)
        self.bounds[0] = x0
        self.bounds[1] = y0
        self._apply_bounds()

    def snap(self, margin=0.01, threshold=0.02):
        x0, y0, width, height = self.bounds
        candidates_x = [margin, self.home_bounds[0], 1.0 - width - margin]
        candidates_y = [margin, self.home_bounds[1], 1.0 - height - margin]

        best_x = min(candidates_x, key=lambda value: abs(value - x0))
        best_y = min(candidates_y, key=lambda value: abs(value - y0))
        if abs(best_x - x0) <= threshold:
            x0 = best_x
        if abs(best_y - y0) <= threshold:
            y0 = best_y

        self.move_to(x0, y0)
        self.clamp(margin=margin)

    def header_contains(self, event):
        if event.x is None or event.y is None:
            return False
        bbox = self.ax.get_window_extent()
        if not (bbox.x0 <= event.x <= bbox.x1 and bbox.y0 <= event.y <= bbox.y1):
            return False
        header_height_px = max((bbox.y1 - bbox.y0) * self.header_frac, 18.0)
        return event.y >= bbox.y1 - header_height_px

    def resize_grip_contains(self, event):
        if event.x is None or event.y is None:
            return False
        bbox = self.ax.get_window_extent()
        if not (bbox.x0 <= event.x <= bbox.x1 and bbox.y0 <= event.y <= bbox.y1):
            return False
        grip_width_px = max((bbox.x1 - bbox.x0) * self.resize_grip_frac, 18.0)
        grip_height_px = max((bbox.y1 - bbox.y0) * self.resize_grip_frac, 18.0)
        return event.x >= bbox.x1 - grip_width_px and event.y <= bbox.y0 + grip_height_px

    def serialize(self):
        return {"bounds": [float(value) for value in self.bounds]}

    def apply_layout(self, layout_dict):
        bounds = layout_dict.get("bounds")
        if not bounds or len(bounds) != 4:
            return
        self.bounds = [float(value) for value in bounds]
        self.clamp()


class DockManager:
    def __init__(self, fig, panels, layout_path=None):
        self.fig = fig
        self.panels = panels
        self.layout_path = Path(layout_path) if layout_path else DEFAULT_LAYOUT_PATH
        self.active_panel = None
        self.active_mode = None
        self.drag_offset = (0.0, 0.0)
        self.resize_anchor = (0.0, 0.0)

        canvas = self.fig.canvas
        canvas.mpl_connect("button_press_event", self.on_press)
        canvas.mpl_connect("motion_notify_event", self.on_motion)
        canvas.mpl_connect("button_release_event", self.on_release)

    def _event_to_figure(self, event):
        return self.fig.transFigure.inverted().transform((event.x, event.y))

    def on_press(self, event):
        for panel in reversed(self.panels):
            if event.inaxes is not panel.ax:
                continue
            if panel.resize_grip_contains(event):
                self.active_panel = panel
                self.active_mode = "resize"
                self.resize_anchor = (panel.bounds[0], panel.bounds[1])
                return
            if panel.header_contains(event):
                self.active_panel = panel
                self.active_mode = "drag"
                fig_x, fig_y = self._event_to_figure(event)
                self.drag_offset = (fig_x - panel.bounds[0], fig_y - panel.bounds[1])
                return

    def on_motion(self, event):
        if self.active_panel is None:
            return
        fig_x, fig_y = self._event_to_figure(event)
        if self.active_mode == "resize":
            anchor_x, anchor_y = self.resize_anchor
            self.active_panel.resize_to(fig_x - anchor_x, fig_y - anchor_y)
        else:
            new_x = fig_x - self.drag_offset[0]
            new_y = fig_y - self.drag_offset[1]
            self.active_panel.move_to(new_x, new_y)
        self.active_panel.clamp()
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.active_panel is None:
            return
        self.active_panel.snap()
        self.active_panel = None
        self.active_mode = None
        self.fig.canvas.draw_idle()

    def serialize_layout(self):
        return {panel.panel_id: panel.serialize() for panel in self.panels}

    def save_layout(self):
        self.layout_path.write_text(json.dumps(self.serialize_layout(), indent=2), encoding="utf-8")
        return self.layout_path

    def load_layout(self):
        if not self.layout_path.exists():
            return False
        payload = json.loads(self.layout_path.read_text(encoding="utf-8"))
        for panel in self.panels:
            if panel.panel_id in payload:
                panel.apply_layout(payload[panel.panel_id])
        self.fig.canvas.draw_idle()
        return True


def setup_figure():
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#f3f6fb")
    ax = fig.add_axes([0.05, 0.22, 0.60, 0.70])
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Distance (um)")
    ax.set_ylabel("Distance (um)")
    ax.set_title("Viewport", loc="left", fontsize=10, fontweight="bold", color=TITLE_COLOR, pad=10)
    ax.set_title("FOV", loc="right", fontsize=9, color=SUBTITLE_COLOR, pad=10)
    return fig, ax


def setup_probe_graphics(ax):
    center_x_ax = 0.50
    arrow_style = dict(ha="center", va="center", fontsize=22, color="white", weight="bold", zorder=20)

    cantilever = Polygon(
        [
            [center_x_ax - 0.50 / 2, 1.09],
            [center_x_ax + 0.50 / 2, 1.09],
            [center_x_ax + 0.50 / 2, 1.00],
            [center_x_ax + 0.035 / 2, 1.00],
            [center_x_ax + 0.035 / 2, 0.89],
            [center_x_ax - 0.035 / 2, 0.89],
            [center_x_ax - 0.035 / 2, 1.00],
            [center_x_ax - 0.50 / 2, 1.00],
        ],
        closed=True,
        transform=ax.transAxes,
        facecolor="0.72",
        edgecolor="black",
        linewidth=2,
        zorder=20,
    )
    ax.add_patch(cantilever)

    rod = Rectangle(
        (center_x_ax, 0.5),
        0.0,
        0.0,
        transform=ax.transAxes,
        facecolor="none",
        edgecolor="none",
        linewidth=0,
        zorder=19,
    )
    ax.add_patch(rod)

    tip = Polygon(
        [[center_x_ax - 0.035 / 2, 0.89], [center_x_ax + 0.035 / 2, 0.89], [center_x_ax, 0.875]],
        closed=True,
        transform=ax.transAxes,
        facecolor="none",
        edgecolor="none",
        linewidth=0,
        zorder=21,
    )
    ax.add_patch(tip)

    ax.text(0.5, 0.95, "^", transform=ax.transAxes, **arrow_style)
    ax.text(0.5, 0.03, "v", transform=ax.transAxes, **arrow_style)
    ax.text(0.03, 0.5, "<", transform=ax.transAxes, **arrow_style)
    ax.text(0.97, 0.5, ">", transform=ax.transAxes, **arrow_style)

    return cantilever, rod, tip, center_x_ax


def _layout_navigation(panel):
    width, height = panel.bounds[2], panel.bounds[3]
    if width >= height * 1.15:
        panel.set_child_bounds("step_size", [0.05, 0.07, 0.34, 0.38])
        button_w = 0.17
        button_h = BUTTON_HEIGHT_WIDE
        center_x = 0.65
        center_y = 0.16
        gap = 0.03
        panel.set_child_bounds("up", [center_x - button_w / 2, center_y + button_h + gap, button_w, button_h])
        panel.set_child_bounds("left", [center_x - button_w - gap, center_y, button_w, button_h])
        panel.set_child_bounds("down", [center_x - button_w / 2, center_y - button_h - gap, button_w, button_h])
        panel.set_child_bounds("right", [center_x + gap, center_y, button_w, button_h])
    else:
        panel.set_child_bounds("step_size", [0.07, 0.48, 0.86, 0.11])
        button_w = 0.26
        button_h = BUTTON_HEIGHT_COMPACT
        center_x = 0.50
        center_y = 0.10
        gap = 0.03
        panel.set_child_bounds("up", [center_x - button_w / 2, center_y + button_h + gap, button_w, button_h])
        panel.set_child_bounds("left", [center_x - button_w - gap / 2, center_y, button_w, button_h])
        panel.set_child_bounds("down", [center_x - button_w / 2, center_y - button_h - gap, button_w, button_h])
        panel.set_child_bounds("right", [center_x + gap / 2, center_y, button_w, button_h])


def _layout_motion(panel):
    width, height = panel.bounds[2], panel.bounds[3]
    if width >= height * 1.4:
        panel.set_child_bounds("zoom_in", [0.05, 0.08, 0.17, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("zoom_out", [0.24, 0.08, 0.17, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("z_down", [0.54, 0.08, 0.12, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("z_up", [0.68, 0.08, 0.12, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("stop_move", [0.05, 0.31, 0.20, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("jump_target", [0.26, 0.31, 0.20, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("focus_reset", [0.54, 0.31, 0.27, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("pause_motion", [0.05, 0.54, 0.20, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("pi", [0.26, 0.54, 0.20, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("auto", [0.54, 0.54, 0.27, BUTTON_HEIGHT_WIDE])
    else:
        panel.set_child_bounds("z_down", [0.06, 0.08, 0.40, BUTTON_HEIGHT_COMPACT])
        panel.set_child_bounds("z_up", [0.52, 0.08, 0.40, BUTTON_HEIGHT_COMPACT])
        panel.set_child_bounds("zoom_in", [0.06, 0.19, 0.40, BUTTON_HEIGHT_COMPACT])
        panel.set_child_bounds("zoom_out", [0.52, 0.19, 0.40, BUTTON_HEIGHT_COMPACT])
        panel.set_child_bounds("focus_reset", [0.06, 0.30, 0.86, BUTTON_HEIGHT_COMPACT])
        panel.set_child_bounds("stop_move", [0.06, 0.41, 0.40, BUTTON_HEIGHT_COMPACT])
        panel.set_child_bounds("jump_target", [0.52, 0.41, 0.40, BUTTON_HEIGHT_COMPACT])
        panel.set_child_bounds("pause_motion", [0.06, 0.52, 0.40, BUTTON_HEIGHT_COMPACT])
        panel.set_child_bounds("pi", [0.52, 0.52, 0.40, BUTTON_HEIGHT_COMPACT])
        panel.set_child_bounds("auto", [0.06, 0.63, 0.86, BUTTON_HEIGHT_COMPACT])


def _layout_status(panel):
    panel.set_text_position("status_label", 0.04, 0.82)
    panel.set_text_position("status_text", 0.04, 0.75)


def _layout_trace(panel):
    panel.set_text_position("trace_label", 0.04, 0.82)
    panel.set_text_position("trace_text", 0.04, 0.75)


def _layout_relocation(panel):
    width, height = panel.bounds[2], panel.bounds[3]
    if width >= height * 3.0:
        button_w = 0.11
        button_h = BUTTON_HEIGHT_WIDE
        gap = 0.008
        start_x = 0.03
        y = 0.10
        roles = ["save_ref", "remove_sample", "auto_origin", "ml_origin",
                 "relocate", "research_patterns", "ai_recall", "ai_zoom"]
        for index, role in enumerate(roles):
            panel.set_child_bounds(role, [start_x + index * (button_w + gap), y, button_w, button_h])
    elif width >= height * 1.1:
        panel.set_child_bounds("research_patterns", [0.05, 0.08, 0.43, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("relocate", [0.52, 0.08, 0.43, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("auto_origin", [0.05, 0.27, 0.43, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("ml_origin", [0.52, 0.27, 0.43, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("save_ref", [0.05, 0.46, 0.43, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("remove_sample", [0.52, 0.46, 0.43, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("ai_recall", [0.05, 0.65, 0.43, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("ai_zoom", [0.52, 0.65, 0.43, BUTTON_HEIGHT_WIDE])
    else:
        y_inc = 0.105
        y_start = 0.06
        roles_vert = ["save_ref", "remove_sample", "auto_origin", "ml_origin",
                      "relocate", "research_patterns", "ai_recall", "ai_zoom"]
        for idx, role in enumerate(roles_vert):
            panel.set_child_bounds(role, [0.08, y_start + idx * y_inc, 0.84, BUTTON_HEIGHT_COMPACT])
    panel.set_text_position("relocation_help", 0.04, 0.10)


def _layout_utility(panel):
    width, height = panel.bounds[2], panel.bounds[3]
    if width >= height * 1.45:
        placements = {
            "coord": [0.07, 0.08, 0.24, BUTTON_HEIGHT_WIDE],
            "hud": [0.35, 0.08, 0.24, BUTTON_HEIGHT_WIDE],
            "clear": [0.63, 0.08, 0.22, BUTTON_HEIGHT_WIDE],
            "scale_bar": [0.07, 0.20, 0.38, BUTTON_HEIGHT_WIDE],
            "research_patterns": [0.07, 0.32, 0.78, BUTTON_HEIGHT_WIDE],
            "save_layout": [0.07, 0.44, 0.78, BUTTON_HEIGHT_WIDE],
            "load_default": [0.07, 0.56, 0.38, BUTTON_HEIGHT_WIDE],
            "load_image": [0.55, 0.56, 0.30, BUTTON_HEIGHT_WIDE],
            "save_default": [0.07, 0.68, 0.78, BUTTON_HEIGHT_WIDE],
        }
    else:
        placements = {
            "clear": [0.07, 0.02, 0.86, BUTTON_HEIGHT_COMPACT],
            "coord": [0.07, 0.12, 0.40, BUTTON_HEIGHT_COMPACT],
            "hud": [0.53, 0.12, 0.40, BUTTON_HEIGHT_COMPACT],
            "scale_bar": [0.07, 0.22, 0.40, BUTTON_HEIGHT_COMPACT],
            "research_patterns": [0.07, 0.32, 0.86, BUTTON_HEIGHT_COMPACT],
            "save_layout": [0.07, 0.42, 0.86, BUTTON_HEIGHT_COMPACT],
            "save_default": [0.07, 0.52, 0.86, BUTTON_HEIGHT_COMPACT],
            "load_default": [0.07, 0.62, 0.40, BUTTON_HEIGHT_COMPACT],
            "load_image": [0.53, 0.62, 0.40, BUTTON_HEIGHT_COMPACT],
        }
    for role, rel_bounds in placements.items():
        panel.set_child_bounds(role, rel_bounds)


def setup_dashboard(fig, layout_path=None):
    button_objects = {}

    navigation_panel = DockablePanel(
        fig,
        "navigation",
        [0.69, 0.51, 0.28, 0.22],
        "Navigation Dock",
        "Panel: step-size + motion pad",
        header_frac=0.19,
        content_gap_frac=0.11,
    )
    radio_step = navigation_panel.add_radio("", ["1 um", "5 um", "50 um", "200 um"], active=1, role="step_size")
    for key, button in [
        navigation_panel.add_button("up", "Up"),
        navigation_panel.add_button("left", "Left"),
        navigation_panel.add_button("down", "Down"),
        navigation_panel.add_button("right", "Right"),
    ]:
        button_objects[key] = button
    navigation_panel.set_layout_callback(_layout_navigation)

    motion_panel = DockablePanel(fig, "motion_view", [0.69, 0.29, 0.28, 0.18], "Motion Dock", "Panel: scan, PI, zoom, focus")
    motion_buttons = [
        ("pi", "PI Compensation Mode", 8.4),
        ("auto", "Auto Scan", 8.7),
        ("pause_motion", "Motion: ON", 8.7),
        ("stop_move", "Stop Here", 8.3),
        ("jump_target", "Go Now", 8.7),
        ("focus_reset", "Best Focus", 8.3),
        ("zoom_in", "Zoom +", 8.7),
        ("zoom_out", "Zoom -", 8.7),
        ("z_down", "Z -", 8.7),
        ("z_up", "Z +", 8.7),
    ]
    for key, label, fontsize in motion_buttons:
        button_key, button = motion_panel.add_button(key, label, fontsize=fontsize, role=key)
        button_objects[button_key] = button
    motion_panel.set_layout_callback(_layout_motion)

    status_panel = DockablePanel(fig, "status_activity", [0.69, 0.04, 0.28, 0.22], "Status Dock", "Panel: live parameters", min_size=(0.22, 0.18))
    status_panel.add_text_block(0.04, 0.82, "Parameters", fontsize=8.7, weight="bold", role="status_label")
    status_text = status_panel.add_text_block(0.04, 0.75, "", fontsize=8.2, family="monospace", linespacing=1.3, role="status_text")
    status_panel.set_layout_callback(_layout_status)

    trace_panel = DockablePanel(fig, "trace", [0.69, 0.77, 0.28, 0.17], "Relocation Trace Dock", "Panel: movement, zoom, click history", min_size=(0.22, 0.16))
    trace_panel.add_text_block(0.04, 0.82, "Recent Trace", fontsize=8.7, weight="bold", role="trace_label")
    trace_text = trace_panel.add_text_block(
        0.04,
        0.75,
        "",
        fontsize=8.0,
        family="monospace",
        linespacing=1.28,
        role="trace_text",
    )
    trace_panel.set_layout_callback(_layout_trace)

    relocation_panel = DockablePanel(fig, "relocation", [0.05, 0.04, 0.44, 0.13], "Relocation Dock", "Panel: low-mag recall + high-mag verification", min_size=(0.40, 0.20))
    for key, button in [
        relocation_panel.add_button("save_ref", "1. Save Region", fontsize=9.0),
        relocation_panel.add_button("remove_sample", "2. Remount", fontsize=9.1),
        relocation_panel.add_button("auto_origin", "3. Pick Origin", fontsize=9.0),
        relocation_panel.add_button("ml_origin", "4. Find Origin", fontsize=9.0),
        relocation_panel.add_button("relocate", "5. Recover Site", fontsize=9.1),
        relocation_panel.add_button("research_patterns", "6. Verify Tip", fontsize=9.0),
        relocation_panel.add_button("ai_recall", "AI Recall", fontsize=8.8),
        relocation_panel.add_button("ai_zoom", "AI Zoom", fontsize=8.8),
    ]:
        button_objects[key] = button
    relocation_panel.add_text_block(
        0.04,
        0.12,
        "Hover over a relocation button to see what it does.",
        fontsize=8.1,
        linespacing=1.2,
        role="relocation_help",
    )
    relocation_panel.set_layout_callback(_layout_relocation)

    utility_panel = DockablePanel(fig, "utility", [0.51, 0.04, 0.16, 0.22], "Utility Dock", "Panel: surface image, plots, data")
    utility_specs = [
        ("load_default", "Load Default", 8.4),
        ("load_image", "Load Image", 8.6),
        ("save_default", "Save As Default", 8.5),
        ("save_layout", "Save Dock Layout", 8.4),
        ("scale_bar", "Scale Bar: 200 um", 8.1),
        ("clear", "Clear Path", 8.6),
        ("coord", "Tip Position", 8.5),
        ("hud", "HUD: ON", 8.5),
    ]
    for key, label, fontsize in utility_specs:
        button_key, button = utility_panel.add_button(key, label, fontsize=fontsize, role=key)
        button_objects[button_key] = button
    utility_panel.set_layout_callback(_layout_utility)

    panels = [
        trace_panel,
        navigation_panel,
        motion_panel,
        status_panel,
        relocation_panel,
        utility_panel,
    ]
    dock_manager = DockManager(fig, panels, layout_path=layout_path)
    return button_objects, radio_step, status_text, trace_text, dock_manager
