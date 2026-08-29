from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from afm_relocation import (
    apply_affine,
    build_overview,
    estimate_affine_transform,
    load_site_memory,
    rotation_translation_affine,
    to_grayscale_u8,
)


PAIR_FEATURE_NAMES = [
    "corrcoef",
    "mean_abs_diff",
    "mse",
    "template_ccorr",
    "template_sqdiff_inv",
    "phase_response",
    "affine_confidence",
    "affine_inlier_ratio",
    "affine_rotation_deg",
    "edge_energy_ratio",
]


def _resize_pair(reference_image, candidate_image, size=(128, 128)):
    ref = to_grayscale_u8(reference_image)
    cand = to_grayscale_u8(candidate_image)
    if ref is None or cand is None:
        return None, None
    width, height = int(size[0]), int(size[1])
    ref_resized = cv2.resize(ref, (width, height), interpolation=cv2.INTER_AREA)
    cand_resized = cv2.resize(cand, (width, height), interpolation=cv2.INTER_AREA)
    return ref_resized, cand_resized


def pair_features(reference_image, candidate_image):
    ref, cand = _resize_pair(reference_image, candidate_image)
    if ref is None or cand is None:
        return np.zeros(len(PAIR_FEATURE_NAMES), dtype=np.float32)

    ref_f = ref.astype(np.float32)
    cand_f = cand.astype(np.float32)
    ref_norm = cv2.normalize(ref_f, None, 0.0, 1.0, cv2.NORM_MINMAX)
    cand_norm = cv2.normalize(cand_f, None, 0.0, 1.0, cv2.NORM_MINMAX)
    diff = ref_norm - cand_norm
    corrcoef = float(np.corrcoef(ref_norm.reshape(-1), cand_norm.reshape(-1))[0, 1])
    if not np.isfinite(corrcoef):
        corrcoef = 0.0
    mean_abs_diff = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))

    ccorr = cv2.matchTemplate(ref_norm, cand_norm, cv2.TM_CCOEFF_NORMED)
    _, ccorr_max, _, _ = cv2.minMaxLoc(ccorr)
    sqdiff = cv2.matchTemplate(ref_norm, cand_norm, cv2.TM_SQDIFF_NORMED)
    sqdiff_min, _, _, _ = cv2.minMaxLoc(sqdiff)
    phase_shift, phase_response = cv2.phaseCorrelate(ref_norm, cand_norm)
    _ = phase_shift

    affine = estimate_affine_transform(ref, cand, max_features=400, keep_matches=100)
    if affine is None:
        affine_confidence = 0.0
        affine_inlier_ratio = 0.0
        affine_rotation_deg = 0.0
    else:
        affine_confidence = float(affine["confidence"])
        affine_inlier_ratio = 0.0 if affine["match_count"] == 0 else float(affine["inlier_count"]) / float(affine["match_count"])
        affine_rotation_deg = float(affine["rotation_deg"])

    ref_edges = cv2.Canny(ref, 80, 160)
    cand_edges = cv2.Canny(cand, 80, 160)
    edge_energy_ratio = float(np.sum(cand_edges > 0) / max(np.sum(ref_edges > 0), 1))

    features = np.array(
        [
            corrcoef,
            mean_abs_diff,
            mse,
            float(ccorr_max),
            float(1.0 - sqdiff_min),
            float(phase_response),
            affine_confidence,
            affine_inlier_ratio,
            affine_rotation_deg,
            edge_energy_ratio,
        ],
        dtype=np.float32,
    )
    features[~np.isfinite(features)] = 0.0
    return features


def score_same_site_probability(model_bundle, reference_image, candidate_image):
    if not model_bundle:
        return None
    model = model_bundle.get("model")
    if model is None:
        return None
    features = pair_features(reference_image, candidate_image).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(features)[0, 1])
    else:
        prediction = float(model.predict(features)[0])
        probability = float(np.clip(prediction, 0.0, 1.0))
    return probability


def predict_remount_transform(model_bundle, reference_overview_image, current_overview_image):
    if not model_bundle:
        return None
    model = model_bundle.get("model")
    if model is None:
        return None
    features = pair_features(reference_overview_image, current_overview_image).reshape(1, -1)
    prediction = np.asarray(model.predict(features), dtype=float).reshape(-1)
    if prediction.size < 3:
        return None
    return {
        "dx_um": float(prediction[0]),
        "dy_um": float(prediction[1]),
        "dtheta_deg": float(prediction[2]),
    }


def _iter_site_memory_dirs(site_memory_root):
    root = Path(site_memory_root)
    if not root.exists():
        return []
    return sorted({path.parent for path in root.glob("**/metadata.json")})


def _load_site_memories(site_memory_root):
    site_memories = []
    for site_dir in _iter_site_memory_dirs(site_memory_root):
        try:
            site_memories.append((site_dir, load_site_memory(site_dir)))
        except Exception:
            continue
    return site_memories


def preferred_training_view(memory, *, strict_camera_only=False):
    live_camera_view = memory.get("live_camera_view")
    if live_camera_view is not None:
        return live_camera_view, "live_camera_view"
    if strict_camera_only:
        return None, None
    reference_template = memory.get("reference_template")
    if reference_template is not None:
        return reference_template, "reference_template_legacy"
    return None, None


def _synthetic_positive_variants(image, count=8):
    gray = to_grayscale_u8(image)
    if gray is None:
        return []
    variants = [gray]
    height, width = gray.shape[:2]
    for _ in range(max(int(count) - 1, 0)):
        shift_x = float(np.random.uniform(-20, 20))
        shift_y = float(np.random.uniform(-20, 20))
        rotation_deg = float(np.random.uniform(-8, 8))
        matrix = rotation_translation_affine(width, height, angle_deg=rotation_deg, shift_x_px=shift_x, shift_y_px=shift_y)
        warped = apply_affine(gray, matrix, output_shape=gray.shape[:2])
        alpha = float(np.random.uniform(0.9, 1.1))
        beta = float(np.random.uniform(-12, 12))
        warped = np.clip(warped.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        variants.append(warped)
    return variants


def _synthetic_negative_variants(image, count=12):
    gray = to_grayscale_u8(image)
    if gray is None:
        return []
    height, width = gray.shape[:2]
    variants = []
    for index in range(max(int(count), 1)):
        if index % 3 == 0:
            shift_x = float(np.random.uniform(-0.35 * width, 0.35 * width))
            shift_y = float(np.random.uniform(-0.35 * height, 0.35 * height))
            rotation_deg = float(np.random.uniform(20, 110)) * float(np.random.choice([-1, 1]))
            scale = float(np.random.uniform(0.65, 1.35))
            matrix = rotation_translation_affine(width, height, angle_deg=rotation_deg, shift_x_px=shift_x, shift_y_px=shift_y, scale=scale)
            variant = apply_affine(gray, matrix, output_shape=gray.shape[:2])
        elif index % 3 == 1:
            crop_scale = float(np.random.uniform(0.45, 0.75))
            crop_w = max(16, int(round(width * crop_scale)))
            crop_h = max(16, int(round(height * crop_scale)))
            x0 = int(np.random.uniform(0, max(width - crop_w, 1)))
            y0 = int(np.random.uniform(0, max(height - crop_h, 1)))
            crop = gray[y0 : y0 + crop_h, x0 : x0 + crop_w]
            variant = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)
        else:
            block_rows = 3
            block_cols = 3
            row_edges = np.linspace(0, height, block_rows + 1, dtype=int)
            col_edges = np.linspace(0, width, block_cols + 1, dtype=int)
            blocks = []
            for row in range(block_rows):
                for col in range(block_cols):
                    blocks.append(gray[row_edges[row] : row_edges[row + 1], col_edges[col] : col_edges[col + 1]].copy())
            np.random.shuffle(blocks)
            variant = np.zeros_like(gray)
            block_index = 0
            for row in range(block_rows):
                for col in range(block_cols):
                    block = cv2.resize(
                        blocks[block_index],
                        (
                            col_edges[col + 1] - col_edges[col],
                            row_edges[row + 1] - row_edges[row],
                        ),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    variant[row_edges[row] : row_edges[row + 1], col_edges[col] : col_edges[col + 1]] = block
                    block_index += 1
        variant = cv2.GaussianBlur(variant, (3, 3), sigmaX=float(np.random.uniform(0.0, 1.2)))
        alpha = float(np.random.uniform(0.85, 1.15))
        beta = float(np.random.uniform(-20, 20))
        variant = np.clip(variant.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        variants.append(variant)
    return variants


def build_same_site_dataset(site_memory_root, positive_augmentations=8, max_negative_pairs=200):
    site_memories = _load_site_memories(site_memory_root)
    if len(site_memories) < 1:
        raise ValueError("Need at least one saved site memory to build a same-site classifier dataset.")

    features = []
    labels = []
    templates = []
    for site_dir, memory in site_memories:
        template, _ = preferred_training_view(memory)
        if template is None:
            continue
        variants = _synthetic_positive_variants(template, count=positive_augmentations)
        templates.append((site_dir, template, variants))
        for variant in variants:
            features.append(pair_features(template, variant))
            labels.append(1)

    negative_pairs = 0
    if len(templates) >= 2:
        for index, (_, template_a, variants_a) in enumerate(templates):
            for jndex, (_, template_b, variants_b) in enumerate(templates):
                if index >= jndex:
                    continue
                candidate_pairs = [(template_a, template_b)]
                candidate_pairs.extend((template_a, variant_b) for variant_b in variants_b[: max(2, positive_augmentations // 2)])
                candidate_pairs.extend((template_b, variant_a) for variant_a in variants_a[: max(2, positive_augmentations // 2)])
                for ref_image, cand_image in candidate_pairs:
                    features.append(pair_features(ref_image, cand_image))
                    labels.append(0)
                    negative_pairs += 1
                    if negative_pairs >= max_negative_pairs:
                        break
            if negative_pairs >= max_negative_pairs:
                break
    else:
        _, memory = site_memories[0]
        template, _ = preferred_training_view(memory)
        if template is None:
            raise ValueError("No valid saved camera view available for same-site dataset generation.")
        hard_negatives = []
        origin_template = memory.get("origin_template")
        if origin_template is not None:
            hard_negatives.append(origin_template)
        for landmark in memory.get("highmag_landmarks", []):
            patch = landmark.get("patch")
            if patch is not None and patch.size:
                hard_negatives.append(patch)
        hard_negatives.extend(_synthetic_negative_variants(template, count=max_negative_pairs))
        for negative_image in hard_negatives[: max_negative_pairs]:
            features.append(pair_features(template, negative_image))
            labels.append(0)

    return np.vstack(features), np.asarray(labels, dtype=np.int32)


def build_remount_transform_dataset(site_memory_root, augmentations_per_site=12):
    site_memories = _load_site_memories(site_memory_root)
    if not site_memories:
        raise ValueError("No saved site memories found for remount-transform dataset generation.")

    features = []
    labels = []
    for _, memory in site_memories:
        overview = memory.get("overview")
        if overview is None or overview.get("image") is None:
            continue
        reference_image = overview["image"]
        scale_x = float(overview["scale_x_um_per_px"])
        scale_y = float(overview["scale_y_um_per_px"])
        height, width = reference_image.shape[:2]
        for _ in range(max(int(augmentations_per_site), 1)):
            shift_x_px = float(np.random.uniform(-60, 60))
            shift_y_px = float(np.random.uniform(-60, 60))
            rotation_deg = float(np.random.uniform(-8, 8))
            matrix = rotation_translation_affine(width, height, angle_deg=rotation_deg, shift_x_px=shift_x_px, shift_y_px=shift_y_px)
            current_image = apply_affine(reference_image, matrix, output_shape=reference_image.shape[:2])
            features.append(pair_features(reference_image, current_image))
            labels.append([shift_x_px * scale_x, shift_y_px * scale_y, rotation_deg])

    return np.vstack(features), np.asarray(labels, dtype=np.float32)


def train_same_site_classifier(site_memory_root, output_path):
    X, y = build_same_site_dataset(site_memory_root)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X, y)
    bundle = {
        "model_type": "same_site_classifier",
        "feature_names": list(PAIR_FEATURE_NAMES),
        "model": model,
    }
    joblib.dump(bundle, output_path)
    return bundle


def train_remount_transform_predictor(site_memory_root, output_path):
    X, y = build_remount_transform_dataset(site_memory_root)
    model = RandomForestRegressor(
        n_estimators=240,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X, y)
    bundle = {
        "model_type": "remount_transform_predictor",
        "feature_names": list(PAIR_FEATURE_NAMES),
        "model": model,
    }
    joblib.dump(bundle, output_path)
    return bundle


def train_lowmag_embedding_index(site_memory_root, output_path, embedding_size=(64, 64)):
    site_memories = _load_site_memories(site_memory_root)
    records = []
    for site_dir, memory in site_memories:
        overview = memory.get("overview")
        if overview is None or overview.get("image") is None:
            continue
        image = to_grayscale_u8(overview["image"])
        resized = cv2.resize(image, embedding_size, interpolation=cv2.INTER_AREA)
        vector = cv2.normalize(resized.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX).reshape(-1)
        records.append(
            {
                "site_dir": str(site_dir),
                "site_id": memory.get("site_id"),
                "sample_id": memory.get("sample_id"),
                "vector": vector,
            }
        )
    bundle = {
        "model_type": "lowmag_embedding_index",
        "embedding_shape": embedding_size,
        "records": records,
    }
    joblib.dump(bundle, output_path)
    return bundle


def retrieve_lowmag_candidates(model_bundle, current_overview_image, top_k=5):
    if not model_bundle or not model_bundle.get("records"):
        return []
    embedding_shape = tuple(model_bundle.get("embedding_shape", (64, 64)))
    image = to_grayscale_u8(current_overview_image)
    if image is None:
        return []
    resized = cv2.resize(image, embedding_shape, interpolation=cv2.INTER_AREA)
    query = cv2.normalize(resized.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX).reshape(-1)
    scored = []
    for record in model_bundle["records"]:
        vector = np.asarray(record["vector"], dtype=np.float32).reshape(-1)
        distance = float(np.linalg.norm(query - vector))
        scored.append(
            {
                "site_dir": record["site_dir"],
                "site_id": record.get("site_id"),
                "sample_id": record.get("sample_id"),
                "distance": distance,
            }
        )
    scored.sort(key=lambda item: item["distance"])
    return scored[: max(int(top_k), 1)]
