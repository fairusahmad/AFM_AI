from pathlib import Path

import cv2
import numpy as np


def _build_default_sample():
    width_um = 2500
    height_um = 2000

    sample = np.zeros((height_um, width_um), dtype=np.float32)
    grid_size = 8
    spacing = 2

    a_height = 300
    a_width = 200
    start_x = 1000
    start_y = 1500
    middle_y = start_y - a_height // 2

    for y0 in range(start_y - a_height, start_y, grid_size + spacing):
        ratio = (start_y - y0) / a_height
        left_x = int(start_x + ratio * (a_width // 2))
        right_x = int(start_x + a_width - ratio * (a_width // 2))

        for x0 in range(start_x, start_x + a_width, grid_size + spacing):
            if abs(x0 - left_x) < grid_size:
                cv2.rectangle(sample, (x0, y0), (x0 + grid_size, y0 + grid_size), 255, -1)
            if abs(x0 - right_x) < grid_size:
                cv2.rectangle(sample, (x0, y0), (x0 + grid_size, y0 + grid_size), 255, -1)
            if abs(y0 - middle_y) < grid_size:
                cv2.rectangle(sample, (x0, y0), (x0 + grid_size, y0 + grid_size), 255, -1)

    bg_square = 8
    bg_spacing = 2
    for y0 in range(0, height_um, bg_square + bg_spacing):
        for x0 in range(0, width_um, bg_square + bg_spacing):
            if sample[y0, x0] == 0:
                cv2.rectangle(sample, (x0, y0), (x0 + bg_square, y0 + bg_square), 180, -1)

    noise = np.random.normal(0, 10, sample.shape)
    sample += noise
    sample = cv2.normalize(sample, None, 0, 255, cv2.NORM_MINMAX)
    return sample.astype(np.uint8), width_um, height_um


class ArtifactLayer:
    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)
        self.layer = np.zeros((self.height, self.width), dtype=np.uint8)
        self.enabled = True
        self.artefact_list = []

    def reset_canvas(self, width, height):
        self.width = int(width)
        self.height = int(height)
        self.layer = np.zeros((self.height, self.width), dtype=np.uint8)
        self.artefact_list = []
        self.enabled = False

    def add_circle(self, center_x, center_y, radius, intensity=255, label=None, artefact_id=None):
        cv2.circle(self.layer, (center_x, center_y), radius, intensity, -1)
        if label:
            cv2.putText(self.layer, label, (center_x - 15, center_y - radius - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, intensity, 1)
        if artefact_id is not None:
            self.artefact_list.append({"id": artefact_id, "type": "circle", "x": center_x, "y": center_y, "size": radius * 2})

    def add_square(self, center_x, center_y, size, intensity=255, label=None, artefact_id=None):
        half = size // 2
        cv2.rectangle(self.layer, (center_x - half, center_y - half), (center_x + half, center_y + half), intensity, -1)
        if label:
            cv2.putText(self.layer, label, (center_x - 15, center_y - half - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, intensity, 1)
        if artefact_id is not None:
            self.artefact_list.append({"id": artefact_id, "type": "square", "x": center_x, "y": center_y, "size": size})

    def add_cross(self, center_x, center_y, size, intensity=255, label=None, artefact_id=None):
        half = size // 2
        cv2.line(self.layer, (center_x - half, center_y), (center_x + half, center_y), intensity, 3)
        cv2.line(self.layer, (center_x, center_y - half), (center_x, center_y + half), intensity, 3)
        if label:
            cv2.putText(self.layer, label, (center_x - 20, center_y - half - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, intensity, 1)
        if artefact_id is not None:
            self.artefact_list.append({"id": artefact_id, "type": "cross", "x": center_x, "y": center_y, "size": size})

    def add_fiducial_marker(self, center_x, center_y, size=40, intensity=255, artefact_id=None):
        cv2.circle(self.layer, (center_x, center_y), size // 2, intensity, 2)
        cv2.circle(self.layer, (center_x, center_y), size // 4, intensity, -1)
        cv2.line(self.layer, (center_x - size // 2, center_y), (center_x + size // 2, center_y), intensity, 2)
        cv2.line(self.layer, (center_x, center_y - size // 2), (center_x, center_y + size // 2), intensity, 2)
        if artefact_id is not None:
            self.artefact_list.append({"id": artefact_id, "type": "fiducial", "x": center_x, "y": center_y, "size": size})

    def add_scale_bar(self, x, y, length_um, pixel_per_um=1, intensity=255):
        length_px = int(length_um * pixel_per_um)
        cv2.line(self.layer, (x, y), (x + length_px, y), intensity, 3)
        cv2.line(self.layer, (x, y - 5), (x, y + 5), intensity, 2)
        cv2.line(self.layer, (x + length_px, y - 5), (x + length_px, y + 5), intensity, 2)
        cv2.putText(self.layer, f"{length_um} um", (x + length_px // 2 - 20, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, intensity, 1)

    def clear(self):
        self.layer.fill(0)
        self.artefact_list = []

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def get_display(self):
        return self.layer if self.enabled else np.zeros_like(self.layer)


def populate_default_artifacts(artifact_layer):
    artifact_layer.enable()
    artifact_layer.add_cross(100, 100, 60, 220, label="Origin", artefact_id=0)
    artifact_layer.add_fiducial_marker(500, 500, 50, 210, artefact_id=1)
    artifact_layer.add_fiducial_marker(2000, 1500, 50, 210, artefact_id=2)
    artifact_layer.add_circle(800, 800, 25, 200, label="A1", artefact_id=3)
    artifact_layer.add_square(1200, 1200, 40, 200, label="B2", artefact_id=4)
    artifact_layer.add_cross(1800, 600, 55, 210, label="C3", artefact_id=5)
    artifact_layer.add_scale_bar(100, 1850, 200, 1, 200)

    for _ in range(40):
        x = np.random.randint(0, artifact_layer.width)
        y = np.random.randint(0, artifact_layer.height)
        r = np.random.randint(3, 10)
        artifact_layer.add_circle(x, y, r, np.random.randint(150, 230))


def load_real_sample_image(image_path, physical_width_um, physical_height_um):
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to load microscope image: {path}")

    target_width = max(200, int(round(float(physical_width_um))))
    target_height = max(200, int(round(float(physical_height_um))))
    src_height, src_width = image.shape[:2]

    scale = min(target_width / max(src_width, 1), target_height / max(src_height, 1))
    resized_width = max(1, int(round(src_width * scale)))
    resized_height = max(1, int(round(src_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)

    canvas_value = int(np.median(image))
    canvas = np.full((target_height, target_width), canvas_value, dtype=np.uint8)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized

    normalized = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8), float(physical_width_um), float(physical_height_um)


def load_real_sample_image_from_scale(image_path, scale_bar_length_um, scale_bar_length_px):
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to load microscope image: {path}")
    if scale_bar_length_px <= 0 or scale_bar_length_um <= 0:
        raise ValueError("Scale bar length must be positive")

    pixels_per_um = float(scale_bar_length_px) / float(scale_bar_length_um)
    physical_width_um = image.shape[1] / pixels_per_um
    physical_height_um = image.shape[0] / pixels_per_um
    return load_real_sample_image(image_path, physical_width_um, physical_height_um)


sample, width_um, height_um = _build_default_sample()
artifact_layer = ArtifactLayer(width_um, height_um)
populate_default_artifacts(artifact_layer)
