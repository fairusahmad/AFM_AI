import cv2
import numpy as np

# AFM optics model.
# Keep optical and blur equations in this file.
# Do not modify these equations unless the user explicitly requests it.
DEFAULT_ZOOM_OUT_LIFT_START = 1.0
DEFAULT_ZOOM_OUT_LIFT_AT_MIN_UM = 550.0
DEFAULT_ZOOM_OUT_LIFT_CURVE_POWER = 2.2


def compute_zoom_out_camera_lift_um(
    zoom_level,
    zoom_levels,
    lift_start_zoom=DEFAULT_ZOOM_OUT_LIFT_START,
    lift_at_min_zoom_um=DEFAULT_ZOOM_OUT_LIFT_AT_MIN_UM,
    curve_power=DEFAULT_ZOOM_OUT_LIFT_CURVE_POWER,
):
    zoom_level = float(zoom_level)
    lift_start_zoom = float(lift_start_zoom)
    if zoom_level >= lift_start_zoom:
        return 0.0

    min_zoom = float(min(zoom_levels))
    zoom_span = max(lift_start_zoom - min_zoom, 1e-6)
    normalized = np.clip((lift_start_zoom - zoom_level) / zoom_span, 0.0, 1.0)
    curved = normalized ** float(max(curve_power, 1e-6))
    return float(lift_at_min_zoom_um * curved)


def get_defocus_metrics(focus_model, image_shape):
    height, width = image_shape[:2]
    if not focus_model:
        return {
            "delta_z_um": 0.0,
            "dof_simple_um": 0.0,
            "dof_camera_um": 0.0,
            "effective_defocus_um": 0.0,
            "blur_diameter_um": 0.0,
            "blur_diameter_px": 0.0,
            "sigma_px": 0.0,
        }

    na = max(float(focus_model.get("numerical_aperture", 0.0)), 1e-6)
    wavelength_um = max(float(focus_model.get("wavelength_um", 0.0)), 1e-6)
    sensor_pixel_size_um = max(float(focus_model.get("sensor_pixel_size_um", 0.0)), 1e-6)
    objective_magnification = max(float(focus_model.get("objective_magnification", 0.0)), 1e-6)
    manual_dof_camera_um = focus_model.get("manual_dof_camera_um")
    z_position_um = float(focus_model.get("z_position_um", 0.0))
    focus_z_um = float(focus_model.get("focus_z_um", 0.0))
    fov_width_um = max(float(focus_model.get("fov_width_um", width)), 1e-6)
    fov_height_um = max(float(focus_model.get("fov_height_um", height)), 1e-6)

    delta_z_um = abs(z_position_um - focus_z_um)
    dof_simple_um = wavelength_um / (na ** 2)
    computed_dof_camera_um = dof_simple_um + (sensor_pixel_size_um / (objective_magnification * na))
    if manual_dof_camera_um is None:
        dof_camera_um = computed_dof_camera_um
    else:
        dof_camera_um = max(float(manual_dof_camera_um), 1e-6)
    effective_defocus_um = max(0.0, delta_z_um - (dof_camera_um / 2.0))
    blur_diameter_um = 2.0 * na * effective_defocus_um

    um_per_px_x = fov_width_um / max(width, 1)
    um_per_px_y = fov_height_um / max(height, 1)
    um_per_px = max((um_per_px_x + um_per_px_y) / 2.0, 1e-6)
    blur_diameter_px = blur_diameter_um / um_per_px
    sigma_px = blur_diameter_px / 2.355

    return {
        "delta_z_um": float(delta_z_um),
        "dof_simple_um": float(dof_simple_um),
        "dof_camera_um": float(dof_camera_um),
        "effective_defocus_um": float(effective_defocus_um),
        "blur_diameter_um": float(blur_diameter_um),
        "blur_diameter_px": float(blur_diameter_px),
        "sigma_px": float(sigma_px),
    }


def apply_defocus_blur(fov, focus_model=None):
    if fov.size == 0:
        return fov, get_defocus_metrics(focus_model, fov.shape)

    metrics = get_defocus_metrics(focus_model, fov.shape)
    sigma_px = metrics["sigma_px"]
    if sigma_px <= 0.12:
        return fov, metrics

    kernel = max(3, int(np.ceil(sigma_px * 6)))
    if kernel % 2 == 0:
        kernel += 1
    kernel = min(kernel, 121)

    blurred = cv2.GaussianBlur(fov, (kernel, kernel), sigmaX=sigma_px, sigmaY=sigma_px)
    return blurred, metrics
