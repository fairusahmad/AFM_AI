import json
from pathlib import Path
from types import SimpleNamespace

import tkinter as tk
from tkinter import scrolledtext, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from matplotlib.widgets import Button, RadioButtons

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None


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


class TkTextProxy:
    def __init__(self, widget, history_lines=None):
        self.widget = widget
        self.history_lines = history_lines
        self._lines = []

    def _write(self, text):
        self.widget.configure(state=tk.NORMAL)
        self.widget.delete("1.0", tk.END)
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.configure(state=tk.DISABLED)

    def set_text(self, text):
        value = str(text)
        if self.history_lines is None:
            self._write(value)
            return
        self._lines = value.splitlines()[-self.history_lines :]
        self._write("\n".join(self._lines))

    def append(self, text):
        if self.history_lines is None:
            self.set_text(text)
            return
        self._lines.append(str(text))
        self._lines = self._lines[-self.history_lines :]
        self._write("\n".join(self._lines))

    def get_text(self):
        return self.widget.get("1.0", tk.END).strip()


class TkLabelProxy:
    def __init__(self, widget):
        self.widget = widget

    def set_text(self, text):
        self.widget.configure(text=str(text))

    def get_text(self):
        return str(self.widget.cget("text"))


class TkImagePreviewProxy:
    def __init__(self, widget, caption_widget=None, max_size=(180, 180), empty_text="No image"):
        self.widget = widget
        self.caption_widget = caption_widget
        self.max_size = tuple(max_size)
        self.empty_text = str(empty_text)
        self._photo = None
        self.clear()

    def clear(self, caption=None):
        self._photo = None
        self.widget.configure(image="", text=self.empty_text, compound="center")
        if self.caption_widget is not None and caption is not None:
            self.caption_widget.configure(text=str(caption))

    def set_image(self, image, caption=None):
        if image is None or getattr(image, "shape", None) is None or Image is None or ImageTk is None:
            self.clear(caption=caption)
            return
        array = np.asarray(image)
        if array.ndim == 2:
            rgb = np.stack([array] * 3, axis=-1)
        elif array.ndim == 3 and array.shape[2] >= 3:
            rgb = array[..., :3]
        else:
            self.clear(caption=caption)
            return
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(rgb)
        resampling = getattr(Image, "Resampling", None)
        resample_filter = resampling.LANCZOS if resampling is not None else Image.LANCZOS
        pil_image.thumbnail(self.max_size, resample_filter)
        frame_width, frame_height = self.max_size
        square_frame = Image.new("RGB", (frame_width, frame_height), color=(246, 249, 254))
        offset_x = max(0, (frame_width - pil_image.width) // 2)
        offset_y = max(0, (frame_height - pil_image.height) // 2)
        square_frame.paste(pil_image, (offset_x, offset_y))
        self._photo = ImageTk.PhotoImage(square_frame)
        self.widget.configure(image=self._photo, text="", compound="center")
        if self.caption_widget is not None and caption is not None:
            self.caption_widget.configure(text=str(caption))


class TkButtonProxy:
    def __init__(self, widget):
        self.widget = widget
        self.label = TkLabelProxy(widget)
        self.ax = None
        self._callbacks = []
        self.widget.configure(command=self._dispatch)

    def _dispatch(self):
        event = SimpleNamespace(widget=self.widget, inaxes=None)
        for callback in list(self._callbacks):
            callback(event)

    def on_clicked(self, callback):
        self._callbacks.append(callback)
        return len(self._callbacks)

    def bind_hover_text(self, target, message, default_text):
        def _on_enter(event):
            target.set_text(message)

        def _on_leave(event):
            target.set_text(default_text)

        self.widget.bind("<Enter>", _on_enter, add="+")
        self.widget.bind("<Leave>", _on_leave, add="+")


class TkRadioProxy:
    def __init__(self, variable, labels):
        self.variable = variable
        self.labels = list(labels)
        self._callbacks = []

    def on_clicked(self, callback):
        self._callbacks.append(callback)
        return len(self._callbacks)

    def _dispatch(self):
        label = self.variable.get()
        for callback in list(self._callbacks):
            callback(label)

    def set_active(self, index):
        if 0 <= index < len(self.labels):
            self.variable.set(self.labels[index])
            self._dispatch()


class TkPanelProxy:
    def __init__(self, panel_id):
        self.panel_id = panel_id
        self.children_by_role = {}


class TkDockWindowManager:
    def __init__(self, root, panels, notebook, window, layout_path=None):
        self.root = root
        self.panels = panels
        self.notebook = notebook
        self.window = window
        self.layout_path = Path(layout_path) if layout_path else DEFAULT_LAYOUT_PATH
        self.uses_external_window = True
        self._position_window()
        self.window.protocol("WM_DELETE_WINDOW", self.window.withdraw)

    def _position_window(self):
        try:
            self.root.update_idletasks()
            root_x = self.root.winfo_rootx()
            root_y = self.root.winfo_rooty()
            root_w = self.root.winfo_width()
            self.window.geometry(f"430x760+{root_x + root_w + 18}+{root_y + 40}")
        except Exception:
            self.window.geometry("430x760")

    def save_layout(self):
        payload = {
            "mode": "external_window",
            "window_geometry": self.window.geometry(),
            "selected_tab": int(self.notebook.index(self.notebook.select())),
        }
        self.layout_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.layout_path

    def load_layout(self):
        if not self.layout_path.exists():
            return False
        try:
            payload = json.loads(self.layout_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        geometry = payload.get("window_geometry")
        if geometry:
            self.window.geometry(str(geometry))
        selected_tab = payload.get("selected_tab")
        if isinstance(selected_tab, int) and 0 <= selected_tab < len(self.panels):
            self.notebook.select(selected_tab)
        return bool(geometry is not None or selected_tab is not None)


def _create_external_dock_dashboard(fig, layout_path=None):
    manager = getattr(fig.canvas, "manager", None)
    root = getattr(manager, "window", None)
    if root is None or not hasattr(root, "winfo_exists"):
        return None

    dock_window = tk.Toplevel(root)
    dock_window.title("AFM Dock Group")
    dock_window.configure(bg="#e8edf5")

    style = ttk.Style(dock_window)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    notebook = ttk.Notebook(dock_window)
    notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    button_objects = {}
    panel_map = {}

    def create_tab(panel_id, title, subtitle):
        frame = tk.Frame(notebook, bg=PANEL_FACE, padx=10, pady=10)
        frame.columnconfigure(0, weight=1)
        notebook.add(frame, text=title)
        heading = tk.Label(frame, text=title, bg=PANEL_FACE, fg=TITLE_COLOR, font=("Segoe UI", 10, "bold"), anchor="w")
        heading.grid(row=0, column=0, sticky="ew")
        subheading = tk.Label(frame, text=subtitle, bg=PANEL_FACE, fg=SUBTITLE_COLOR, font=("Segoe UI", 8), anchor="w")
        subheading.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        panel = TkPanelProxy(panel_id)
        panel_map[panel_id] = panel
        return frame, panel

    def add_button(parent, row, column, key, label, columnspan=1, sticky="ew", padx=4, pady=4):
        button = tk.Button(
            parent,
            text=label,
            bg=BUTTON_FACE,
            fg=BUTTON_TEXT,
            activebackground=BUTTON_HOVER,
            relief=tk.RAISED,
            bd=1,
            font=("Segoe UI", 9),
        )
        button.grid(row=row, column=column, columnspan=columnspan, sticky=sticky, padx=padx, pady=pady)
        proxy = TkButtonProxy(button)
        button_objects[key] = proxy
        return proxy

    trace_frame, trace_panel = create_tab("trace", "Trace", "Movement, zoom, and click history")
    trace_frame.rowconfigure(2, weight=1)
    trace_text_widget = scrolledtext.ScrolledText(
        trace_frame,
        wrap=tk.WORD,
        bg="#0b1220",
        fg="#cde7ff",
        insertbackground="#cde7ff",
        font=("Consolas", 8),
        relief=tk.FLAT,
        borderwidth=0,
        padx=6,
        pady=6,
        height=16,
    )
    trace_text_widget.grid(row=2, column=0, sticky="nsew")
    trace_text_widget.configure(state=tk.DISABLED)
    trace_text = TkTextProxy(trace_text_widget, history_lines=250)
    trace_panel.children_by_role["trace_text"] = {"artist": trace_text}

    navigation_frame, navigation_panel = create_tab("navigation", "Navigation", "Step-size and motion pad")
    for index in range(3):
        navigation_frame.columnconfigure(index, weight=1)
    nav_radio_var = tk.StringVar(value="5 um")
    nav_radio_frame = tk.LabelFrame(navigation_frame, text="Step Size", bg=PANEL_FACE, fg=TITLE_COLOR, font=("Segoe UI", 9, "bold"))
    nav_radio_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 10))
    for idx, label in enumerate(["1 um", "5 um", "50 um", "200 um"]):
        radio = tk.Radiobutton(
            nav_radio_frame,
            text=label,
            value=label,
            variable=nav_radio_var,
            bg=PANEL_FACE,
            anchor="w",
            font=("Segoe UI", 9),
        )
        radio.grid(row=idx, column=0, sticky="w", padx=8, pady=1)
    radio_step = TkRadioProxy(nav_radio_var, ["1 um", "5 um", "50 um", "200 um"])
    nav_radio_var.trace_add("write", lambda *_args: radio_step._dispatch())
    add_button(navigation_frame, 3, 1, "up", "Up")
    add_button(navigation_frame, 4, 0, "left", "Left")
    add_button(navigation_frame, 4, 1, "down", "Down")
    add_button(navigation_frame, 4, 2, "right", "Right")

    motion_frame, motion_panel = create_tab("motion_view", "Motion", "Scan, PI, zoom, and focus")
    motion_specs = [
        ("pi", "PI Compensation Mode"),
        ("auto", "Auto Scan"),
        ("pause_motion", "Motion: ON"),
        ("stop_move", "Stop Here"),
        ("jump_target", "Go Now"),
        ("focus_reset", "Best Focus"),
        ("zoom_in", "Zoom +"),
        ("zoom_out", "Zoom -"),
        ("z_down", "Z -"),
        ("z_up", "Z +"),
    ]
    for col in range(2):
        motion_frame.columnconfigure(col, weight=1)
    for idx, (key, label) in enumerate(motion_specs):
        add_button(motion_frame, 2 + idx // 2, idx % 2, key, label)

    status_frame, status_panel = create_tab("status_activity", "Status", "Live parameters")
    status_frame.rowconfigure(3, weight=1)
    status_label = tk.Label(status_frame, text="Parameters", bg=PANEL_FACE, fg=TITLE_COLOR, font=("Segoe UI", 9, "bold"), anchor="w")
    status_label.grid(row=2, column=0, sticky="ew", pady=(0, 6))
    status_text_widget = scrolledtext.ScrolledText(
        status_frame,
        wrap=tk.NONE,
        bg="#f6f9fe",
        fg="#24384f",
        insertbackground="#24384f",
        font=("Consolas", 8),
        relief=tk.FLAT,
        borderwidth=0,
        padx=6,
        pady=6,
        height=18,
    )
    status_text_widget.grid(row=3, column=0, sticky="nsew")
    status_text_widget.configure(state=tk.DISABLED)
    status_text = TkTextProxy(status_text_widget)
    status_panel.children_by_role["status_text"] = {"artist": status_text}

    relocation_frame, relocation_panel = create_tab("relocation", "Relocation", "Low-mag recall and high-mag verification")
    for col in range(2):
        relocation_frame.columnconfigure(col, weight=1)
    relocation_specs = [
        ("save_ref", "1. Save Region"),
        ("remove_sample", "2. Remount"),
        ("auto_origin", "3. Pick Origin"),
        ("ml_origin", "4. Find Origin"),
        ("relocate", "5. Recover Site"),
        ("research_patterns", "6. Verify Tip"),
        ("ai_recall", "AI Recall"),
        ("ai_zoom", "AI Zoom"),
        ("smooth_slower", "Slower"),
        ("smooth_faster", "Faster"),
        ("relocation_go_now", "Go Now"),
    ]
    for idx, (key, label) in enumerate(relocation_specs):
        add_button(relocation_frame, 2 + idx // 2, idx % 2, key, label)
    relocation_help_frame = tk.Frame(
        relocation_frame,
        bg="#f6f9fe",
        highlightbackground="#c7d8ec",
        highlightthickness=1,
        bd=0,
        height=64,
    )
    relocation_help_frame.grid(row=8, column=0, columnspan=2, sticky="ew", padx=4, pady=(10, 0))
    relocation_help_frame.grid_propagate(False)
    relocation_help_frame.columnconfigure(0, weight=1)
    relocation_help_widget = tk.Label(
        relocation_help_frame,
        text="Hover over a relocation button to see what it does.",
        bg="#f6f9fe",
        fg=TEXT_COLOR,
        justify=tk.LEFT,
        anchor="nw",
        wraplength=340,
        font=("Segoe UI", 8),
        padx=8,
        pady=6,
    )
    relocation_help_widget.grid(row=0, column=0, sticky="nsew")
    relocation_help = TkLabelProxy(relocation_help_widget)
    relocation_panel.children_by_role["relocation_help"] = {"artist": relocation_help}
    relocation_preview_label = tk.Label(
        relocation_frame,
        text="Stored Relocation Images",
        bg=PANEL_FACE,
        fg=TITLE_COLOR,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
    )
    relocation_preview_label.grid(row=9, column=0, columnspan=2, sticky="ew", padx=4, pady=(10, 4))
    preview_frame = tk.Frame(relocation_frame, bg=PANEL_FACE)
    preview_frame.grid(row=10, column=0, columnspan=2, sticky="nsew", padx=4, pady=(0, 6))
    preview_frame.columnconfigure(0, weight=1)
    preview_frame.columnconfigure(1, weight=1)
    overview_preview = tk.Label(
        preview_frame,
        bg="#f6f9fe",
        fg="#6b7f95",
        relief=tk.SOLID,
        borderwidth=1,
        width=180,
        height=180,
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 8),
        text="No overview",
    )
    overview_preview.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
    reference_preview = tk.Label(
        preview_frame,
        bg="#f6f9fe",
        fg="#6b7f95",
        relief=tk.SOLID,
        borderwidth=1,
        width=180,
        height=180,
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 8),
        text="No reference",
    )
    reference_preview.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
    overview_caption = tk.Label(preview_frame, text="Low-mag overview", bg=PANEL_FACE, fg=SUBTITLE_COLOR, font=("Segoe UI", 8))
    overview_caption.grid(row=1, column=0, sticky="ew", pady=(3, 0))
    reference_caption = tk.Label(preview_frame, text="High-mag reference", bg=PANEL_FACE, fg=SUBTITLE_COLOR, font=("Segoe UI", 8))
    reference_caption.grid(row=1, column=1, sticky="ew", pady=(3, 0))
    relocation_panel.children_by_role["relocation_overview_image"] = {
        "artist": TkImagePreviewProxy(overview_preview, overview_caption, max_size=(180, 180), empty_text="No overview"),
    }
    relocation_panel.children_by_role["relocation_reference_image"] = {
        "artist": TkImagePreviewProxy(reference_preview, reference_caption, max_size=(180, 180), empty_text="No reference"),
    }
    relocation_status_label = tk.Label(
        relocation_frame,
        text="Relocation Storage And Progress",
        bg=PANEL_FACE,
        fg=TITLE_COLOR,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
    )
    relocation_status_label.grid(row=11, column=0, columnspan=2, sticky="ew", padx=4, pady=(10, 4))
    relocation_frame.rowconfigure(12, weight=1)
    relocation_status_widget = scrolledtext.ScrolledText(
        relocation_frame,
        wrap=tk.WORD,
        bg="#f6f9fe",
        fg="#24384f",
        insertbackground="#24384f",
        font=("Consolas", 8),
        relief=tk.FLAT,
        borderwidth=0,
        padx=6,
        pady=6,
        height=14,
    )
    relocation_status_widget.grid(row=12, column=0, columnspan=2, sticky="nsew", padx=4, pady=(0, 0))
    relocation_status_widget.configure(state=tk.DISABLED)
    relocation_status_text = TkTextProxy(relocation_status_widget)
    relocation_panel.children_by_role["relocation_status_text"] = {"artist": relocation_status_text}
    relocation_default_help = relocation_help.get_text()
    relocation_hover_map = {
        "save_ref": "Save Region: store the current low-mag context, high-mag template, zoom context, and landmark memory for later recovery.",
        "remove_sample": "Remount: simulate taking the sample out and putting it back with shift only.",
        "auto_origin": "Pick Origin: choose the strongest distinctive local landmark in the current viewport as the working origin reference.",
        "ml_origin": "Find Origin: search the full sample for the saved origin pattern and move toward the recognized origin if confidence is good.",
        "relocate": "Recover Site: run coarse low-mag recall, estimate rotation and offset, refine with high-mag matching, and propose the recovered site.",
        "research_patterns": "Verify Tip: re-match multiple remembered landmark patterns around the tip to confirm whether the cantilever is at the correct place.",
        "ai_recall": "AI Recall: one-click AI relocation - load site memory, recognize pattern with rotation, move cantilever, verify. Click to correct if needed.",
        "ai_zoom": "AI Zoom: recall saved zoom level, AI-recognize pattern, auto zoom-out search if not found, then move + verify.",
        "smooth_slower": "Slower: reduce the smooth animated cantilever movement speed so relocation motion is easier to watch.",
        "smooth_faster": "Faster: increase the smooth animated cantilever movement speed for quicker relocation travel.",
        "relocation_go_now": "Go Now: immediately jump to the pending AI relocation target, or to the current motion destination if no relocation target is pending.",
    }
    for key, message in relocation_hover_map.items():
        button_objects[key].bind_hover_text(relocation_help, message, relocation_default_help)

    utility_frame, utility_panel = create_tab("utility", "Utility", "Surface image, plots, and data")
    for col in range(2):
        utility_frame.columnconfigure(col, weight=1)
    utility_specs = [
        ("load_default", "Load Default"),
        ("load_image", "Load Image"),
        ("save_default", "Save As Default"),
        ("save_layout", "Save Dock Layout"),
        ("scale_bar", "Scale Bar: 200 um"),
        ("clear", "Clear Path"),
        ("coord", "Tip Position"),
        ("hud", "HUD: ON"),
    ]
    for idx, (key, label) in enumerate(utility_specs):
        add_button(utility_frame, 2 + idx // 2, idx % 2, key, label)

    panels = [
        trace_panel,
        navigation_panel,
        motion_panel,
        status_panel,
        relocation_panel,
        utility_panel,
    ]
    dock_manager = TkDockWindowManager(root, panels, notebook, dock_window, layout_path=layout_path)
    return button_objects, radio_step, status_text, trace_text, dock_manager


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
        button_w = 0.095
        button_h = BUTTON_HEIGHT_WIDE
        gap = 0.008
        start_x = 0.03
        y = 0.10
        roles = [
            "save_ref",
            "remove_sample",
            "auto_origin",
            "ml_origin",
            "relocate",
            "research_patterns",
            "ai_recall",
            "ai_zoom",
            "smooth_slower",
            "smooth_faster",
            "relocation_go_now",
        ]
        for index, role in enumerate(roles):
            panel.set_child_bounds(role, [start_x + index * (button_w + gap), y, button_w, button_h])
    elif width >= height * 1.1:
        panel.set_child_bounds("research_patterns", [0.05, 0.08, 0.43, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("relocate", [0.52, 0.08, 0.43, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("auto_origin", [0.05, 0.23, 0.43, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("ml_origin", [0.52, 0.23, 0.43, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("save_ref", [0.05, 0.38, 0.43, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("remove_sample", [0.52, 0.38, 0.43, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("ai_recall", [0.05, 0.53, 0.43, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("ai_zoom", [0.52, 0.53, 0.43, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("smooth_slower", [0.05, 0.68, 0.27, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("smooth_faster", [0.365, 0.68, 0.27, BUTTON_HEIGHT_WIDE])
        panel.set_child_bounds("relocation_go_now", [0.68, 0.68, 0.27, BUTTON_HEIGHT_WIDE])
    else:
        y_inc = 0.08
        y_start = 0.06
        roles_vert = [
            "save_ref",
            "remove_sample",
            "auto_origin",
            "ml_origin",
            "relocate",
            "research_patterns",
            "ai_recall",
            "ai_zoom",
            "smooth_slower",
            "smooth_faster",
            "relocation_go_now",
        ]
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
    external_dashboard = _create_external_dock_dashboard(fig, layout_path=layout_path)
    if external_dashboard is not None:
        return external_dashboard

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
        relocation_panel.add_button("smooth_slower", "Slower", fontsize=8.8),
        relocation_panel.add_button("smooth_faster", "Faster", fontsize=8.8),
        relocation_panel.add_button("relocation_go_now", "Go Now", fontsize=8.8),
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
