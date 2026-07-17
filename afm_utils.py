import cv2
import numpy as np
from afm_optics_model import apply_defocus_blur, get_defocus_metrics


def get_tip_position(tip, ax):
    vertices = tip.get_xy()
    tip_vertex_axes = vertices[2]
    tip_transform = tip.get_transform()
    tip_vertex_display = tip_transform.transform(tip_vertex_axes)
    tip_vertex_data = ax.transData.inverted().transform(tip_vertex_display)
    return tip_vertex_data[0], tip_vertex_data[1]


def update_title(ax, fig, pi_mode, target_center_x, target_center_y, tip_x, tip_y):
    if pi_mode:
        error_x = target_center_x - tip_x
        error_y = target_center_y - tip_y
        error_total = float(np.hypot(error_x, error_y))
        ax.set_title(
            f"Hysteresis Error: dX={error_x:+.1f} um  dY={error_y:+.1f} um  |e|={error_total:.1f} um",
            fontsize=10,
        )
    else:
        ax.set_title("Linear Mode", fontsize=10)
    fig.canvas.draw_idle()


def get_scale_bar_geometry(x_origin, y_origin, fov_width, fov_height, total_length_um, segments=2):
    total_length_um = float(max(total_length_um, 1.0))
    segments = max(int(segments), 1)
    segment_length_um = total_length_um / segments

    margin_x = fov_width * 0.06
    margin_y = fov_height * 0.08
    x_end = x_origin + fov_width - margin_x
    x_start = x_end - total_length_um
    y_bar = y_origin + fov_height - margin_y

    segment_bounds = []
    for index in range(segments):
        seg_start = x_start + index * segment_length_um
        seg_end = seg_start + segment_length_um
        segment_bounds.append((seg_start, seg_end))

    text_x = (x_start + x_end) / 2.0
    text_y = y_bar - fov_height * 0.035
    return {
        "segments": segment_bounds,
        "text_pos": (text_x, text_y),
        "label": f"{int(round(total_length_um))} um",
        "y": y_bar,
    }


def create_stage_fov(sample, artifact_layer, show_artifact, x, y, fov_width, fov_height, valid_mask=None):
    x0 = int(round(x))
    y0 = int(round(y))
    width = max(int(round(fov_width)), 1)
    height = max(int(round(fov_height)), 1)

    fov = np.zeros((height, width), dtype=sample.dtype)
    outside_mask = np.ones((height, width), dtype=bool)

    sample_h, sample_w = sample.shape[:2]
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(sample_w, x0 + width)
    src_y1 = min(sample_h, y0 + height)

    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        fov[dst_y0:dst_y1, dst_x0:dst_x1] = sample[src_y0:src_y1, src_x0:src_x1]
        if valid_mask is None:
            outside_mask[dst_y0:dst_y1, dst_x0:dst_x1] = False
        else:
            valid_region = valid_mask[src_y0:src_y1, src_x0:src_x1]
            outside_mask[dst_y0:dst_y1, dst_x0:dst_x1] = np.logical_not(valid_region)

        if show_artifact and artifact_layer is not None:
            artifact = artifact_layer.get_display()[src_y0:src_y1, src_x0:src_x1]
            fov[dst_y0:dst_y1, dst_x0:dst_x1] = np.maximum(
                fov[dst_y0:dst_y1, dst_x0:dst_x1],
                artifact,
            )

    return fov, outside_mask, x0, y0


def create_fov_image(sample, artifact_layer, show_artifact, x, y, fov_width, fov_height):
    fov, _, ix, iy = create_stage_fov(sample, artifact_layer, show_artifact, x, y, fov_width, fov_height)
    return fov, ix, iy


def _resize_fov_and_mask(fov, camera_resolution=None, outside_mask=None):
    if camera_resolution is None:
        return fov, outside_mask
    target_w, target_h = camera_resolution
    if target_w <= 0 or target_h <= 0:
        return fov, outside_mask
    if fov.shape[1] == target_w and fov.shape[0] == target_h:
        return fov, outside_mask
    interpolation = cv2.INTER_AREA if (target_w < fov.shape[1] or target_h < fov.shape[0]) else cv2.INTER_CUBIC
    resized = cv2.resize(fov, (int(target_w), int(target_h)), interpolation=interpolation)
    resized_mask = outside_mask
    if outside_mask is not None:
        resized_mask = cv2.resize(
            outside_mask.astype(np.uint8),
            (int(target_w), int(target_h)),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    return resized, resized_mask


def create_probe_occlusion_mask(
    shape,
    *,
    fov_width_um,
    fov_height_um,
    tip_rel_x=0.50,
    tip_rel_y=0.50,
    body_width_um=1600.0,
    tip_width_um=35.0,
    tip_total_length_um=125.0,
    triangular_tip_length_um=15.0,
    visible_body_depth_um=3400.0,
):
    height, width = shape[:2]
    if height <= 0 or width <= 0 or fov_width_um <= 0 or fov_height_um <= 0:
        return np.zeros((max(height, 0), max(width, 0)), dtype=bool)

    px_per_um_x = float(width) / float(fov_width_um)
    px_per_um_y = float(height) / float(fov_height_um)
    tip_x = float(tip_rel_x) * float(width)
    tip_y = float(tip_rel_y) * float(height)

    tip_half_width_px = max(1.0, float(tip_width_um) * px_per_um_x / 2.0)
    body_half_width_px = max(tip_half_width_px, float(body_width_um) * px_per_um_x / 2.0)
    tri_len_px = max(1.0, float(triangular_tip_length_um) * px_per_um_y)
    body_depth_px = max(1.0, float(visible_body_depth_um) * px_per_um_y)

    tri_base_y = min(float(height - 1), tip_y + tri_len_px)
    body_bottom_y = min(float(height - 1), tri_base_y + body_depth_px)

    mask = np.zeros((height, width), dtype=np.uint8)
    triangle = np.array(
        [
            [int(round(tip_x)), int(round(tip_y))],
            [int(round(tip_x - tip_half_width_px)), int(round(tri_base_y))],
            [int(round(tip_x + tip_half_width_px)), int(round(tri_base_y))],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, triangle, 1)
    rect_x0 = max(0, int(round(tip_x - body_half_width_px)))
    rect_x1 = min(width, int(round(tip_x + body_half_width_px)))
    rect_y0 = max(0, int(round(tri_base_y)))
    rect_y1 = min(height, int(round(body_bottom_y)))
    if rect_x1 > rect_x0 and rect_y1 > rect_y0:
        mask[rect_y0:rect_y1, rect_x0:rect_x1] = 1
    return mask > 0


def apply_probe_occlusion(frame, occlusion_mask, fill_value=None):
    if frame is None:
        return None
    occluded = np.array(frame, copy=True)
    if occlusion_mask is None or not np.any(occlusion_mask):
        return occluded
    if fill_value is None:
        fill_value = int(np.median(occluded)) if occluded.ndim == 2 else tuple(int(v) for v in np.median(occluded.reshape(-1, occluded.shape[-1]), axis=0))
    if occluded.ndim == 2:
        occluded[occlusion_mask] = fill_value
    else:
        occluded[occlusion_mask] = np.array(fill_value, dtype=occluded.dtype)
    return occluded


def render_camera_frame(fov, camera_resolution=None, outside_mask=None, outside_color=(155, 24, 24), focus_model=None):
    if fov.size == 0:
        return fov, get_defocus_metrics(focus_model, fov.shape)

    fov, outside_mask = _resize_fov_and_mask(fov, camera_resolution=camera_resolution, outside_mask=outside_mask)

    blurred, focus_metrics = apply_defocus_blur(fov, focus_model)
    return _apply_outside_color(blurred, outside_mask, outside_color), focus_metrics


def render_camera_recognition_frame(
    fov,
    *,
    camera_resolution=None,
    outside_mask=None,
    focus_model=None,
    fov_width_um,
    fov_height_um,
    tip_rel_x=0.50,
    tip_rel_y=0.50,
    body_width_um=1600.0,
    tip_width_um=35.0,
    tip_total_length_um=125.0,
    triangular_tip_length_um=15.0,
    visible_body_depth_um=3400.0,
):
    if fov.size == 0:
        return fov, get_defocus_metrics(focus_model, fov.shape)
    fov, outside_mask = _resize_fov_and_mask(fov, camera_resolution=camera_resolution, outside_mask=outside_mask)
    blurred, focus_metrics = apply_defocus_blur(fov, focus_model)
    occlusion_mask = create_probe_occlusion_mask(
        blurred.shape,
        fov_width_um=fov_width_um,
        fov_height_um=fov_height_um,
        tip_rel_x=tip_rel_x,
        tip_rel_y=tip_rel_y,
        body_width_um=body_width_um,
        tip_width_um=tip_width_um,
        tip_total_length_um=tip_total_length_um,
        triangular_tip_length_um=triangular_tip_length_um,
        visible_body_depth_um=visible_body_depth_um,
    )
    combined_mask = occlusion_mask if outside_mask is None else np.logical_or(outside_mask, occlusion_mask)
    median_fill = int(np.median(blurred)) if blurred.size else 0
    blurred = apply_probe_occlusion(blurred, occlusion_mask, fill_value=median_fill)
    if combined_mask is not None and np.any(combined_mask):
        blurred = np.array(blurred, copy=True)
        blurred[combined_mask] = median_fill
    return blurred, focus_metrics


def rotate_camera_frame(frame, angle_deg, fill_color=(155, 24, 24)):
    angle_deg = float(angle_deg)
    if frame.size == 0 or np.isclose(angle_deg, 0.0):
        return frame

    height, width = frame.shape[:2]
    center = (width / 2.0, height / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # 计算旋转后完整图像的边界框，自动扩展画布避免切角
    rad = np.deg2rad(angle_deg)
    cos_a, sin_a = abs(np.cos(rad)), abs(np.sin(rad))
    new_width = int(round(width * cos_a + height * sin_a))
    new_height = int(round(width * sin_a + height * cos_a))

    # 调整平移使旋转后图像在新画布中居中
    rotation[0, 2] += (new_width - width) / 2.0
    rotation[1, 2] += (new_height - height) / 2.0

    border_value = fill_color if frame.ndim == 3 else fill_color[0]
    return cv2.warpAffine(
        frame,
        rotation,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _apply_outside_color(fov, outside_mask, outside_color):
    if fov.ndim == 2:
        display = np.repeat(fov[..., None], 3, axis=2)
    else:
        display = fov.copy()

    if outside_mask is not None and np.any(outside_mask):
        display[outside_mask] = np.array(outside_color, dtype=display.dtype)
    return display
