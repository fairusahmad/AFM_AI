import json
import time
from pathlib import Path

import cv2
import numpy as np


def to_grayscale_u8(image):
    if image is None:
        return None
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    if array.dtype != np.uint8:
        array = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return array


def sanitize_token(text, fallback):
    raw = str(text or fallback).strip()
    if not raw:
        raw = fallback
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)
    safe = safe.strip("_")
    return safe or fallback


def build_overview(surface_image, max_dim=512):
    gray = to_grayscale_u8(surface_image)
    if gray is None or gray.size == 0:
        return None

    src_h, src_w = gray.shape[:2]
    if max(src_h, src_w) <= max_dim:
        resized = gray.copy()
    else:
        scale = float(max_dim) / float(max(src_h, src_w))
        resized = cv2.resize(
            gray,
            (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    out_h, out_w = resized.shape[:2]
    return {
        "image": resized,
        "scale_x_um_per_px": float(src_w) / float(max(out_w, 1)),
        "scale_y_um_per_px": float(src_h) / float(max(out_h, 1)),
    }


def _patch_score(patch):
    patch_f = patch.astype(np.float32)
    gy, gx = np.gradient(patch_f)
    return float(np.std(patch_f) + 0.35 * np.mean(np.hypot(gx, gy)))


def extract_landmarks(
    image,
    *,
    base_x_um=0.0,
    base_y_um=0.0,
    scale_x_um_per_px=1.0,
    scale_y_um_per_px=1.0,
    origin_x_um=None,
    origin_y_um=None,
    patch_half=24,
    max_landmarks=8,
    min_distance_px=20,
):
    gray = to_grayscale_u8(image)
    if gray is None or gray.size == 0:
        return []

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max(max_landmarks * 5, 20),
        qualityLevel=0.01,
        minDistance=max(8, int(min_distance_px)),
        blockSize=7,
    )

    candidates = []
    if corners is not None:
        for corner in corners.reshape(-1, 2):
            cx = int(round(float(corner[0])))
            cy = int(round(float(corner[1])))
            x0 = max(0, cx - patch_half)
            x1 = min(gray.shape[1], cx + patch_half)
            y0 = max(0, cy - patch_half)
            y1 = min(gray.shape[0], cy + patch_half)
            patch = gray[y0:y1, x0:x1]
            if patch.shape[0] < 12 or patch.shape[1] < 12:
                continue
            score = _patch_score(patch)
            abs_x_um = float(base_x_um + cx * scale_x_um_per_px)
            abs_y_um = float(base_y_um + cy * scale_y_um_per_px)
            candidates.append(
                {
                    "center_px": (int(cx), int(cy)),
                    "abs_x_um": abs_x_um,
                    "abs_y_um": abs_y_um,
                    "relative_x_um": (
                        None if origin_x_um is None else float(abs_x_um - origin_x_um)
                    ),
                    "relative_y_um": (
                        None if origin_y_um is None else float(abs_y_um - origin_y_um)
                    ),
                    "score": score,
                    "patch": patch.copy(),
                }
            )

    if not candidates:
        cx = gray.shape[1] // 2
        cy = gray.shape[0] // 2
        x0 = max(0, cx - patch_half)
        x1 = min(gray.shape[1], cx + patch_half)
        y0 = max(0, cy - patch_half)
        y1 = min(gray.shape[0], cy + patch_half)
        patch = gray[y0:y1, x0:x1]
        if patch.size:
            candidates.append(
                {
                    "center_px": (int(cx), int(cy)),
                    "abs_x_um": float(base_x_um + cx * scale_x_um_per_px),
                    "abs_y_um": float(base_y_um + cy * scale_y_um_per_px),
                    "relative_x_um": (
                        None if origin_x_um is None else float(base_x_um + cx * scale_x_um_per_px - origin_x_um)
                    ),
                    "relative_y_um": (
                        None if origin_y_um is None else float(base_y_um + cy * scale_y_um_per_px - origin_y_um)
                    ),
                    "score": _patch_score(patch),
                    "patch": patch.copy(),
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    min_dist_um = float(min_distance_px) * max(float(scale_x_um_per_px), float(scale_y_um_per_px))
    for candidate in candidates:
        if any(
            np.hypot(candidate["abs_x_um"] - item["abs_x_um"], candidate["abs_y_um"] - item["abs_y_um"]) < min_dist_um
            for item in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_landmarks:
            break
    return selected


def annotate_landmarks_with_tip_geometry(landmarks, tip_x_um=None, tip_y_um=None):
    if tip_x_um is None or tip_y_um is None:
        return landmarks
    for landmark in landmarks or []:
        dx = float(landmark["abs_x_um"] - tip_x_um)
        dy = float(landmark["abs_y_um"] - tip_y_um)
        landmark["tip_dx_um"] = dx
        landmark["tip_dy_um"] = dy
        landmark["tip_distance_um"] = float(np.hypot(dx, dy))
        landmark["tip_angle_deg"] = float(np.degrees(np.arctan2(dy, dx)))
    return landmarks


def match_template_candidates(search_image, template, top_k=3, suppress_radius_px=None):
    search = to_grayscale_u8(search_image)
    patch = to_grayscale_u8(template)
    if search is None or patch is None:
        return []
    if patch.shape[0] >= search.shape[0] or patch.shape[1] >= search.shape[1]:
        return []

    result = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
    work = result.copy()
    candidates = []
    suppress_radius_px = suppress_radius_px or max(patch.shape[0], patch.shape[1]) // 2
    for _ in range(max(int(top_k), 1)):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if not np.isfinite(max_val):
            break
        candidates.append({"x": int(max_loc[0]), "y": int(max_loc[1]), "score": float(max_val)})
        x0 = max(0, int(max_loc[0] - suppress_radius_px))
        x1 = min(work.shape[1], int(max_loc[0] + suppress_radius_px))
        y0 = max(0, int(max_loc[1] - suppress_radius_px))
        y1 = min(work.shape[0], int(max_loc[1] + suppress_radius_px))
        work[y0:y1, x0:x1] = -np.inf
    return candidates


def estimate_landmark_consensus(
    reference_landmarks,
    search_image,
    *,
    search_origin_x_um=0.0,
    search_origin_y_um=0.0,
    scale_x_um_per_px=1.0,
    scale_y_um_per_px=1.0,
    min_score=0.42,
    min_gap=0.02,
    max_residual_um=75.0,
):
    matches = []
    for landmark in reference_landmarks or []:
        patch = landmark.get("patch")
        if patch is None:
            continue
        candidates = match_template_candidates(search_image, patch, top_k=2)
        if not candidates:
            continue
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        patch_h, patch_w = patch.shape[:2]
        center_x_um = float(search_origin_x_um + (best["x"] + patch_w / 2.0) * scale_x_um_per_px)
        center_y_um = float(search_origin_y_um + (best["y"] + patch_h / 2.0) * scale_y_um_per_px)
        score_gap = float(best["score"] - second["score"]) if second is not None else float(best["score"])
        if best["score"] < min_score or score_gap < min_gap:
            continue
        matches.append(
            {
                "score": float(best["score"]),
                "score_gap": score_gap,
                "abs_x_um": center_x_um,
                "abs_y_um": center_y_um,
                "offset_x_um": float(center_x_um - landmark["abs_x_um"]),
                "offset_y_um": float(center_y_um - landmark["abs_y_um"]),
                "reference_abs_x_um": float(landmark["abs_x_um"]),
                "reference_abs_y_um": float(landmark["abs_y_um"]),
            }
        )

    if not matches:
        return None

    offsets = np.array([[item["offset_x_um"], item["offset_y_um"]] for item in matches], dtype=float)
    median_offset = np.median(offsets, axis=0)
    supporters = []
    for match in matches:
        residual = float(np.hypot(match["offset_x_um"] - median_offset[0], match["offset_y_um"] - median_offset[1]))
        if residual <= max_residual_um:
            match["residual_um"] = residual
            supporters.append(match)

    if not supporters:
        return None

    offset_x_um = float(np.mean([item["offset_x_um"] for item in supporters]))
    offset_y_um = float(np.mean([item["offset_y_um"] for item in supporters]))
    mean_score = float(np.mean([item["score"] for item in supporters]))
    mean_gap = float(np.mean([item["score_gap"] for item in supporters]))
    confidence = float(np.clip(0.75 * mean_score + 0.25 * min(mean_gap * 5.0, 1.0), 0.0, 1.0))
    return {
        "offset_x_um": offset_x_um,
        "offset_y_um": offset_y_um,
        "support_count": len(supporters),
        "mean_score": mean_score,
        "mean_score_gap": mean_gap,
        "confidence": confidence,
        "matches": supporters,
    }


def analyze_landmark_geometry(
    reference_landmarks,
    current_view,
    *,
    view_origin_x_um=0.0,
    view_origin_y_um=0.0,
    tip_x_um=None,
    tip_y_um=None,
    min_score=0.42,
    min_gap=0.02,
):
    def _wrap_angle_deg(angle_deg):
        return ((float(angle_deg) + 180.0) % 360.0) - 180.0

    view = to_grayscale_u8(current_view)
    if view is None or view.size == 0:
        return {"matches": [], "matched_count": 0, "geometry_confidence": 0.0, "distance_confidence": 0.0}

    matches = []
    for index, landmark in enumerate(reference_landmarks or [], start=1):
        patch = landmark.get("patch")
        if patch is None:
            continue
        candidates = match_template_candidates(view, patch, top_k=2)
        if not candidates:
            continue
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        score_gap = float(best["score"] - second["score"]) if second is not None else float(best["score"])
        if best["score"] < min_score or score_gap < min_gap:
            continue
        patch_h, patch_w = patch.shape[:2]
        center_x_um = float(view_origin_x_um + best["x"] + patch_w / 2.0)
        center_y_um = float(view_origin_y_um + best["y"] + patch_h / 2.0)
        dx_tip_um = None
        dy_tip_um = None
        tip_distance_um = None
        tip_angle_deg = None
        distance_error_um = None
        angle_error_deg = None
        if tip_x_um is not None and tip_y_um is not None:
            dx_tip_um = float(center_x_um - tip_x_um)
            dy_tip_um = float(center_y_um - tip_y_um)
            tip_distance_um = float(np.hypot(dx_tip_um, dy_tip_um))
            tip_angle_deg = float(np.degrees(np.arctan2(dy_tip_um, dx_tip_um)))
            if landmark.get("tip_distance_um") is not None:
                distance_error_um = float(abs(tip_distance_um - float(landmark["tip_distance_um"])))
            if landmark.get("tip_angle_deg") is not None:
                angle_error_deg = float(abs(_wrap_angle_deg(tip_angle_deg - float(landmark["tip_angle_deg"]))))
        matches.append(
            {
                "index": index,
                "score": float(best["score"]),
                "score_gap": score_gap,
                "top_left_x_um": float(view_origin_x_um + best["x"]),
                "top_left_y_um": float(view_origin_y_um + best["y"]),
                "center_x_um": center_x_um,
                "center_y_um": center_y_um,
                "width_um": float(patch_w),
                "height_um": float(patch_h),
                "reference_abs_x_um": float(landmark["abs_x_um"]),
                "reference_abs_y_um": float(landmark["abs_y_um"]),
                "reference_tip_distance_um": (
                    None if landmark.get("tip_distance_um") is None else float(landmark["tip_distance_um"])
                ),
                "reference_tip_angle_deg": (
                    None if landmark.get("tip_angle_deg") is None else float(landmark["tip_angle_deg"])
                ),
                "tip_dx_um": dx_tip_um,
                "tip_dy_um": dy_tip_um,
                "tip_distance_um": tip_distance_um,
                "tip_angle_deg": tip_angle_deg,
                "distance_error_um": distance_error_um,
                "angle_error_deg": angle_error_deg,
            }
        )

    pair_errors_um = []
    for i in range(len(matches)):
        for j in range(i + 1, len(matches)):
            match_a = matches[i]
            match_b = matches[j]
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

    distance_errors_um = [item["distance_error_um"] for item in matches if item["distance_error_um"] is not None]
    angle_errors_deg = [item["angle_error_deg"] for item in matches if item["angle_error_deg"] is not None]
    mean_pair_error_um = float(np.mean(pair_errors_um)) if pair_errors_um else None
    mean_distance_error_um = float(np.mean(distance_errors_um)) if distance_errors_um else None
    mean_angle_error_deg = float(np.mean(angle_errors_deg)) if angle_errors_deg else None
    mean_score = float(np.mean([item["score"] for item in matches])) if matches else 0.0
    mean_gap = float(np.mean([item["score_gap"] for item in matches])) if matches else 0.0
    geometry_confidence = 0.0
    distance_confidence = 0.0
    if matches:
        pair_term = 1.0 if mean_pair_error_um is None else float(np.clip(1.0 - mean_pair_error_um / 80.0, 0.0, 1.0))
        distance_term = 1.0 if mean_distance_error_um is None else float(np.clip(1.0 - mean_distance_error_um / 80.0, 0.0, 1.0))
        score_term = float(np.clip(0.75 * mean_score + 0.25 * min(mean_gap * 5.0, 1.0), 0.0, 1.0))
        geometry_confidence = float(np.clip(0.45 * pair_term + 0.35 * distance_term + 0.20 * score_term, 0.0, 1.0))
        distance_confidence = float(np.clip(0.65 * distance_term + 0.35 * score_term, 0.0, 1.0))

    return {
        "matches": matches,
        "matched_count": len(matches),
        "pair_count": len(pair_errors_um),
        "mean_score": mean_score,
        "mean_score_gap": mean_gap,
        "mean_pair_error_um": mean_pair_error_um,
        "mean_distance_error_um": mean_distance_error_um,
        "mean_angle_error_deg": mean_angle_error_deg,
        "geometry_confidence": geometry_confidence,
        "distance_confidence": distance_confidence,
    }


def translate_image(image, shift_x_px, shift_y_px, border_value=None):
    gray = to_grayscale_u8(image)
    if gray is None or gray.size == 0:
        return gray
    if border_value is None:
        border_value = int(np.median(gray))
    matrix = np.array([[1.0, 0.0, float(shift_x_px)], [0.0, 1.0, float(shift_y_px)]], dtype=np.float32)
    return cv2.warpAffine(
        gray,
        matrix,
        (gray.shape[1], gray.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value),
    )


def affine_to_homogeneous(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (2, 3):
        raise ValueError("Affine matrix must be 2x3")
    return np.vstack([matrix, np.array([0.0, 0.0, 1.0], dtype=np.float32)])


def homogeneous_to_affine(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError("Homogeneous matrix must be 3x3")
    return matrix[:2, :]


def invert_affine(matrix):
    homogeneous = affine_to_homogeneous(matrix)
    inverse = np.linalg.inv(homogeneous)
    return homogeneous_to_affine(inverse)


def compose_affine(*matrices):
    result = np.eye(3, dtype=np.float32)
    for matrix in matrices:
        result = affine_to_homogeneous(matrix) @ result
    return homogeneous_to_affine(result)


def transform_point(matrix, x, y):
    point = np.array([float(x), float(y), 1.0], dtype=np.float32)
    transformed = affine_to_homogeneous(matrix) @ point
    return float(transformed[0]), float(transformed[1])


def apply_affine(image, matrix, output_shape=None, border_value=None):
    gray = to_grayscale_u8(image)
    if gray is None or gray.size == 0:
        return gray
    if output_shape is None:
        output_shape = gray.shape[:2]
    out_h, out_w = int(output_shape[0]), int(output_shape[1])
    if border_value is None:
        border_value = int(np.median(gray))
    return cv2.warpAffine(
        gray,
        np.asarray(matrix, dtype=np.float32),
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value),
    )


def rotation_translation_affine(width, height, angle_deg=0.0, shift_x_px=0.0, shift_y_px=0.0, scale=1.0):
    center = (float(width) / 2.0, float(height) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), float(scale))
    matrix[0, 2] += float(shift_x_px)
    matrix[1, 2] += float(shift_y_px)
    return matrix.astype(np.float32)


def expanded_rotation_affine(width, height, angle_deg=0.0, shift_x_px=0.0, shift_y_px=0.0):
    """Compute an expanded output shape and adjusted affine matrix so that
    the entire rotated+translated image fits without clipping.

    Returns (new_width, new_height, adjusted_2x3_matrix).
    """
    center = (float(width) / 2.0, float(height) / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)

    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    rotated = cv2.transform(corners.reshape(1, -1, 2), rot_mat).reshape(-1, 2)
    rotated[:, 0] += float(shift_x_px)
    rotated[:, 1] += float(shift_y_px)

    min_x = float(np.floor(np.min(rotated[:, 0])))
    max_x = float(np.ceil(np.max(rotated[:, 0])))
    min_y = float(np.floor(np.min(rotated[:, 1])))
    max_y = float(np.ceil(np.max(rotated[:, 1])))

    new_w = int(max_x - min_x)
    new_h = int(max_y - min_y)

    matrix = rot_mat.copy()
    matrix[0, 2] += float(shift_x_px) - min_x
    matrix[1, 2] += float(shift_y_px) - min_y

    return new_w, new_h, matrix.astype(np.float32)


def estimate_affine_transform(reference_image, current_image, max_features=600, keep_matches=120):
    ref = to_grayscale_u8(reference_image)
    cur = to_grayscale_u8(current_image)
    if ref is None or cur is None or ref.size == 0 or cur.size == 0:
        return None

    orb = cv2.ORB_create(nfeatures=max(int(max_features), 100))
    ref_keypoints, ref_descriptors = orb.detectAndCompute(ref, None)
    cur_keypoints, cur_descriptors = orb.detectAndCompute(cur, None)
    if ref_descriptors is None or cur_descriptors is None:
        return None
    if len(ref_keypoints) < 4 or len(cur_keypoints) < 4:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(ref_descriptors, cur_descriptors)
    if len(matches) < 4:
        return None
    matches = sorted(matches, key=lambda item: item.distance)[: max(int(keep_matches), 4)]

    src = np.float32([ref_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([cur_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None:
        return None

    matrix = np.asarray(matrix, dtype=np.float32)
    match_count = len(matches)

    # --- Rotation direction verification ---
    # RANSAC may converge to a rotation with the wrong sign (symmetric/repeated patterns).
    # Build a flipped-rotation candidate and compare inlier counts on all matched pairs.
    # Flip rotation sign: [[a,b,tx],[c,d,ty]] → [[a,c,tx],[b,d,ty]]
    # because: cos(-θ)=cos(θ), -sin(-θ)=sin(θ), sin(-θ)=-sin(θ)
    flip_matrix = np.array(
        [
            [matrix[0, 0], matrix[1, 0], matrix[0, 2]],
            [matrix[0, 1], matrix[1, 1], matrix[1, 2]],
        ],
        dtype=np.float32,
    )

    def _count_inliers(m, src_pts, dst_pts, thresh=3.0):
        transformed = cv2.transform(src_pts, m)
        residuals = np.linalg.norm(
            transformed.reshape(-1, 2) - dst_pts.reshape(-1, 2), axis=1
        )
        return int(np.sum(residuals < thresh))

    orig_all = _count_inliers(matrix, src, dst)
    flip_all = _count_inliers(flip_matrix, src, dst)
    margin = max(2, int(match_count * 0.05))

    if flip_all >= orig_all + margin:
        # Refine translation for the flipped rotation using median residual
        transformed = cv2.transform(src, flip_matrix).reshape(-1, 2)
        residuals = dst.reshape(-1, 2) - transformed
        inlier_mask = np.linalg.norm(residuals, axis=1) < 3.0
        if np.sum(inlier_mask) >= 4:
            median_dx = float(np.median(residuals[inlier_mask, 0]))
            median_dy = float(np.median(residuals[inlier_mask, 1]))
            flip_matrix[0, 2] += median_dx
            flip_matrix[1, 2] += median_dy
            flip_all = _count_inliers(flip_matrix, src, dst)

        if flip_all > orig_all:
            matrix = flip_matrix
            inlier_count = flip_all
        else:
            inlier_count = int(np.sum(inliers)) if inliers is not None else 0
    else:
        inlier_count = int(np.sum(inliers)) if inliers is not None else 0

    rotation_rad = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    confidence = 0.0 if match_count == 0 else float(np.clip(inlier_count / max(match_count, 1), 0.0, 1.0))
    return {
        "matrix": matrix,
        "rotation_deg": float(np.degrees(rotation_rad)),
        "scale": scale,
        "translation_px": (float(matrix[0, 2]), float(matrix[1, 2])),
        "match_count": match_count,
        "inlier_count": inlier_count,
        "confidence": confidence,
    }


def estimate_local_affine_reference_match(sample, reference_patch, center_x, center_y, half_range=600):
    sample_gray = to_grayscale_u8(sample)
    reference_gray = to_grayscale_u8(reference_patch)
    if sample_gray is None or reference_gray is None:
        return None

    patch_h, patch_w = reference_gray.shape[:2]
    sample_h, sample_w = sample_gray.shape[:2]
    start_x = int(max(0, center_x - half_range))
    start_y = int(max(0, center_y - half_range))
    end_x = int(min(sample_w, center_x + half_range + patch_w))
    end_y = int(min(sample_h, center_y + half_range + patch_h))
    if end_x - start_x < patch_w or end_y - start_y < patch_h:
        return None

    search = sample_gray[start_y:end_y, start_x:end_x]
    affine = estimate_affine_transform(reference_gray, search, max_features=500, keep_matches=150)
    if affine is None:
        return None

    top_left_x, top_left_y = transform_point(affine["matrix"], 0.0, 0.0)
    return {
        "x": float(start_x + top_left_x),
        "y": float(start_y + top_left_y),
        "rotation_deg": float(affine["rotation_deg"]),
        "scale": float(affine["scale"]),
        "confidence": float(affine["confidence"]),
        "match_count": int(affine["match_count"]),
        "inlier_count": int(affine["inlier_count"]),
        "search_origin": (int(start_x), int(start_y)),
        "matrix": affine["matrix"],
    }


def overview_affine_to_fullres(matrix, reference_overview, current_overview):
    ref_sx = float(reference_overview["scale_x_um_per_px"])
    ref_sy = float(reference_overview["scale_y_um_per_px"])
    cur_sx = float(current_overview["scale_x_um_per_px"])
    cur_sy = float(current_overview["scale_y_um_per_px"])

    to_ref_overview = np.array(
        [[1.0 / ref_sx, 0.0, 0.0], [0.0, 1.0 / ref_sy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    to_cur_full = np.array(
        [[cur_sx, 0.0, 0.0], [0.0, cur_sy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    full = to_cur_full @ affine_to_homogeneous(matrix) @ to_ref_overview
    return homogeneous_to_affine(full)


def build_site_memory(state, stage_history=None):
    overview = build_overview(state.surface_image)
    origin_x = float(state.origin_x) if getattr(state, "origin_defined", False) else None
    origin_y = float(state.origin_y) if getattr(state, "origin_defined", False) else None
    sample_id = sanitize_token(Path(state.sample_path).stem if state.sample_path else state.sample_source, "sample")
    session_id = time.strftime("%Y%m%d_%H%M%S")
    site_label = state.origin_label if getattr(state, "origin_defined", False) else "unlabeled_site"
    site_id = sanitize_token(site_label, f"site_{session_id}")

    reference_source = state.current_camera_view if getattr(state, "current_camera_view", None) is not None else state.current_fov_raw
    reference = to_grayscale_u8(reference_source)
    origin_template = to_grayscale_u8(getattr(state, "origin_template", None))
    lowmag_landmarks = []
    if overview is not None:
        lowmag_landmarks = extract_landmarks(
            overview["image"],
            scale_x_um_per_px=overview["scale_x_um_per_px"],
            scale_y_um_per_px=overview["scale_y_um_per_px"],
            origin_x_um=origin_x,
            origin_y_um=origin_y,
            patch_half=18,
            max_landmarks=8,
            min_distance_px=18,
        )
    highmag_landmarks = extract_landmarks(
        reference,
        base_x_um=float(state.x),
        base_y_um=float(state.y),
        origin_x_um=origin_x,
        origin_y_um=origin_y,
        patch_half=24,
        max_landmarks=6,
        min_distance_px=22,
    )
    annotate_landmarks_with_tip_geometry(
        highmag_landmarks,
        tip_x_um=float(getattr(state, "probe_tip_x", state.x + state.fov_width / 2.0)),
        tip_y_um=float(getattr(state, "probe_tip_y", state.y + state.fov_height / 2.0)),
    )

    target_center_x = float(state.target_x + state.fov_width / 2.0)
    target_center_y = float(state.target_y + state.fov_height / 2.0)
    site_memory = {
        "sample_id": sample_id,
        "session_id": session_id,
        "site_id": site_id,
        "captured_at": session_id,
        "sample_source": str(state.sample_source),
        "sample_path": None if state.sample_path is None else str(state.sample_path),
        "origin": (
            None
            if origin_x is None or origin_y is None
            else {
                "label": str(state.origin_label),
                "x_um": origin_x,
                "y_um": origin_y,
            }
        ),
        "reference_top_left": {"x_um": float(state.x), "y_um": float(state.y)},
        "reference_center": {
            "x_um": float(state.x + state.fov_width / 2.0),
            "y_um": float(state.y + state.fov_height / 2.0),
        },
        "target_center": {"x_um": target_center_x, "y_um": target_center_y},
        "reference_tip": {
            "x_um": float(getattr(state, "probe_tip_x", state.x + state.fov_width / 2.0)),
            "y_um": float(getattr(state, "probe_tip_y", state.y + state.fov_height / 2.0)),
        },
        "fov_size_um": {"width_um": float(state.fov_width), "height_um": float(state.fov_height)},
        "zoom_level": float(state.current_zoom_level),
        "magnification": float(state.get_current_objective_magnification()),
        "tilt_angle_deg": 0.0,
        "focus_state": {
            "probe_gap_um": float(state.get_probe_sample_gap_um()),
            "focus_offset_um": float(state.get_focus_offset_um()),
            "blur_sigma_px": float(getattr(state, "last_blur_sigma_px", 0.0)),
        },
        "motion_history_points": int(0 if stage_history is None else len(stage_history)),
        "overview": overview,
        "reference_template": reference,
        "origin_template": origin_template,
        "lowmag_landmarks": lowmag_landmarks,
        "highmag_landmarks": highmag_landmarks,
    }
    return site_memory


def _write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gray = to_grayscale_u8(image)
    if gray is None or gray.size == 0:
        return None
    cv2.imwrite(str(path), gray)
    return path


def _landmark_metadata(landmark, file_name):
    return {
        "center_px": [int(landmark["center_px"][0]), int(landmark["center_px"][1])],
        "abs_x_um": float(landmark["abs_x_um"]),
        "abs_y_um": float(landmark["abs_y_um"]),
        "relative_x_um": None if landmark["relative_x_um"] is None else float(landmark["relative_x_um"]),
        "relative_y_um": None if landmark["relative_y_um"] is None else float(landmark["relative_y_um"]),
        "score": float(landmark["score"]),
        "tip_dx_um": None if landmark.get("tip_dx_um") is None else float(landmark["tip_dx_um"]),
        "tip_dy_um": None if landmark.get("tip_dy_um") is None else float(landmark["tip_dy_um"]),
        "tip_distance_um": None if landmark.get("tip_distance_um") is None else float(landmark["tip_distance_um"]),
        "tip_angle_deg": None if landmark.get("tip_angle_deg") is None else float(landmark["tip_angle_deg"]),
        "patch_path": str(file_name),
    }


def persist_site_memory(site_memory, base_dir):
    base_path = Path(base_dir)
    output_dir = base_path / site_memory["sample_id"] / f"{site_memory['session_id']}_{site_memory['site_id']}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        key: value
        for key, value in site_memory.items()
        if key not in {"overview", "reference_template", "origin_template", "lowmag_landmarks", "highmag_landmarks"}
    }

    overview = site_memory.get("overview")
    if overview is not None:
        _write_image(output_dir / "lowmag_overview.png", overview["image"])
        metadata["overview"] = {
            "image_path": "lowmag_overview.png",
            "scale_x_um_per_px": float(overview["scale_x_um_per_px"]),
            "scale_y_um_per_px": float(overview["scale_y_um_per_px"]),
        }

    reference_path = _write_image(output_dir / "reference_template.png", site_memory.get("reference_template"))
    metadata["reference_template_path"] = None if reference_path is None else reference_path.name

    origin_template_path = _write_image(output_dir / "origin_template.png", site_memory.get("origin_template"))
    metadata["origin_template_path"] = None if origin_template_path is None else origin_template_path.name

    low_dir = output_dir / "landmarks" / "lowmag"
    high_dir = output_dir / "landmarks" / "highmag"
    metadata["lowmag_landmarks"] = []
    metadata["highmag_landmarks"] = []

    for index, landmark in enumerate(site_memory.get("lowmag_landmarks", []), start=1):
        file_name = f"lowmag_{index:02d}.png"
        _write_image(low_dir / file_name, landmark.get("patch"))
        metadata["lowmag_landmarks"].append(_landmark_metadata(landmark, Path("landmarks") / "lowmag" / file_name))

    for index, landmark in enumerate(site_memory.get("highmag_landmarks", []), start=1):
        file_name = f"highmag_{index:02d}.png"
        _write_image(high_dir / file_name, landmark.get("patch"))
        metadata["highmag_landmarks"].append(_landmark_metadata(landmark, Path("landmarks") / "highmag" / file_name))

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def load_site_memory(site_dir):
    site_dir = Path(site_dir)
    metadata_path = site_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Site-memory metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    overview = metadata.get("overview")
    if overview is not None:
        overview = {
            "image": to_grayscale_u8(cv2.imread(str(site_dir / overview["image_path"]), cv2.IMREAD_GRAYSCALE)),
            "scale_x_um_per_px": float(overview["scale_x_um_per_px"]),
            "scale_y_um_per_px": float(overview["scale_y_um_per_px"]),
        }

    def _load_landmarks(items):
        loaded = []
        for item in items or []:
            patch = to_grayscale_u8(cv2.imread(str(site_dir / item["patch_path"]), cv2.IMREAD_GRAYSCALE))
            loaded.append(
                {
                    "center_px": (int(item["center_px"][0]), int(item["center_px"][1])),
                    "abs_x_um": float(item["abs_x_um"]),
                    "abs_y_um": float(item["abs_y_um"]),
                    "relative_x_um": None if item["relative_x_um"] is None else float(item["relative_x_um"]),
                    "relative_y_um": None if item["relative_y_um"] is None else float(item["relative_y_um"]),
                    "score": float(item["score"]),
                    "tip_dx_um": None if item.get("tip_dx_um") is None else float(item["tip_dx_um"]),
                    "tip_dy_um": None if item.get("tip_dy_um") is None else float(item["tip_dy_um"]),
                    "tip_distance_um": None if item.get("tip_distance_um") is None else float(item["tip_distance_um"]),
                    "tip_angle_deg": None if item.get("tip_angle_deg") is None else float(item["tip_angle_deg"]),
                    "patch": patch,
                }
            )
        return loaded

    site_memory = dict(metadata)
    site_memory["overview"] = overview
    site_memory["reference_template"] = to_grayscale_u8(
        cv2.imread(str(site_dir / metadata["reference_template_path"]), cv2.IMREAD_GRAYSCALE)
    ) if metadata.get("reference_template_path") else None
    site_memory["origin_template"] = to_grayscale_u8(
        cv2.imread(str(site_dir / metadata["origin_template_path"]), cv2.IMREAD_GRAYSCALE)
    ) if metadata.get("origin_template_path") else None
    site_memory["lowmag_landmarks"] = _load_landmarks(metadata.get("lowmag_landmarks"))
    site_memory["highmag_landmarks"] = _load_landmarks(metadata.get("highmag_landmarks"))
    return site_memory


def find_latest_site_memory(base_dir, sample_id):
    base_dir = Path(base_dir)
    sample_id = sanitize_token(sample_id, "sample")
    sample_dir = base_dir / sample_id
    if not sample_dir.exists():
        return None
    candidates = sorted(sample_dir.glob("*/metadata.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    return candidates[0].parent
