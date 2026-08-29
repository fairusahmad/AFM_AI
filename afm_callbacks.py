import csv
import itertools
import random
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np

from afm_phase2_ml import (
    predict_remount_transform,
    retrieve_lowmag_candidates,
    score_same_site_probability,
)
from afm_phase2_ml import pair_features
from afm_ml_recognition import (
    DeepFeatureExtractor,
    MLPatternMatcher,
    MLSameSiteClassifier,
    MLTransformPredictor,
    MLTransformPredictor5W,
    load_deep_same_site_classifier,
    load_deep_remount_predictor,
    load_deep_remount_predictor_5w,
)
from afm_relocation import (
    analyze_landmark_geometry,
    annotate_landmarks_with_tip_geometry,
    apply_affine,
    build_overview,
    build_overview_from_view,
    build_site_memory,
    estimate_landmark_consensus,
    expanded_rotation_affine,
    extract_landmarks,
    find_latest_site_memory,
    invert_affine,
    load_site_memory,
    match_template_candidates,
    merge_landmark_sets,
    persist_site_memory,
    rotation_translation_affine,
    transform_point,
    translate_image,
    to_grayscale_u8,
)
from afm_utils import create_stage_fov, render_camera_frame
from afm_utils import render_camera_matching_frame, render_camera_recognition_frame
from artefact_detector import ArtefactDetector
from image_matching import match_reference_template
from sample_generation import load_real_sample_image, load_real_sample_image_from_scale

BASE_DIR = Path(__file__).resolve().parent


class AFMCallbacks:
    SCALE_BAR_CHOICES_UM = (50.0, 100.0, 200.0, 500.0)
    PROBE_TIP_REL_X = 0.50
    PROBE_TIP_REL_Y = 0.50
    PROBE_BODY_WIDTH_UM = 1600.0
    PROBE_BODY_LENGTH_UM = 3400.0
    PROBE_TIP_WIDTH_UM = 35.0
    PROBE_TIP_TOTAL_LENGTH_UM = 125.0
    PROBE_TRIANGULAR_TIP_LENGTH_UM = 15.0
    PROBE_VISIBLE_BODY_DEPTH_UM = 3400
    VIEWPORT_ARROW_HIT_RADIUS_AX = 0.07
    MATCHING_RESOLUTION = (512, 384)

    def __init__(
        self,
        state,
        stage,
        fig,
        ax,
        tip,
        cantilever,
        rod,
        center_x_ax,
        data,
        update_title_func,
        get_tip_func,
        buttons,
        artifact_layer,
    ):
        self.state = state
        self.stage = stage
        self.fig = fig
        self.ax = ax
        self.tip = tip
        self.cantilever = cantilever
        self.rod = rod
        self.center_x_ax = center_x_ax
        self.data = data
        self.update_title = update_title_func
        self.get_tip = get_tip_func
        self.buttons = buttons
        self.artifact_layer = artifact_layer
        self.img = None
        self.ai_mode = False
        self.ai_compensator = None
        self.same_site_classifier = None
        self.remount_transform_predictor = None
        self.lowmag_embedding_index = None
        self.lowmag_landmark_regressor = None
        self.busy_actions = set()
        self.log_callback = None
        self.status_callback = None
        self.persist_default_callback = None
        self.animation_step_callback = None

        self.artefact_detector = None
        try:
            self.artefact_detector = ArtefactDetector()
            if getattr(self.artefact_detector, "model_loaded", False):
                self.log("Artefact detector ready")
            else:
                self.artefact_detector = None
                self.log("Artefact detector unavailable")
        except Exception as e:
            self.log(f"Failed to load artefact detector: {e}")

        inverse_model_path = self._resolve_project_path("inverse_model.pkl")
        try:
            self.ai_compensator = joblib.load(inverse_model_path)
            self.ai_mode = True
            self.log(f"AI inverse model loaded: {inverse_model_path}")
        except Exception as e:
            self.log(f"AI inverse model not available at {inverse_model_path}: {e}")

        phase2_models_dir = self._resolve_project_path("collected_data/models")
        self.same_site_classifier = self._load_optional_model(
            phase2_models_dir / "same_site_classifier.pkl",
            "same-site classifier",
        )
        self.remount_transform_predictor = self._load_optional_model(
            phase2_models_dir / "remount_transform_predictor.pkl",
            "remount transform predictor",
        )
        self.lowmag_embedding_index = self._load_optional_model(
            phase2_models_dir / "lowmag_embedding_index.pkl",
            "low-mag embedding index",
        )
        self.lowmag_landmark_regressor = self._load_optional_model(
            phase2_models_dir / "lowmag_landmark_regressor.pkl",
            "low-mag landmark regressor",
        )

        # ── Deep ML 模型 (ResNet18 特征 + MLP) ──
        self.deep_feature_extractor = None
        self.ml_pattern_matcher = None
        self.deep_classifier = None
        self.deep_regressor = None
        self.deep_regressor_is_5w = False
        self._logged_5w_overview_skip = False
        self._ml_ready = False
        try:
            self.deep_feature_extractor = DeepFeatureExtractor()
            self.ml_pattern_matcher = MLPatternMatcher(extractor=self.deep_feature_extractor)
            self.deep_classifier = load_deep_same_site_classifier(
                phase2_models_dir / "deep_same_site_classifier.pkl"
            )
            if self.deep_classifier is not None:
                self.deep_classifier.extractor = self.deep_feature_extractor
                self.log("Deep same-site classifier loaded")

            # 优先加载 5w 大样本模型，回退到旧版
            self.deep_regressor = load_deep_remount_predictor_5w(
                phase2_models_dir / "deep_remount_predictor_real.pkl"
            )
            if self.deep_regressor is None:
                self.deep_regressor = load_deep_remount_predictor_5w(
                    phase2_models_dir / "deep_remount_predictor_5w.pkl"
                )
            if self.deep_regressor is None:
                # 回退: _final.pkl (用户手动重命名的副本)
                self.deep_regressor = load_deep_remount_predictor_5w(
                    phase2_models_dir / "deep_remount_predictor_final.pkl"
                )
            if self.deep_regressor is not None:
                self.deep_regressor.extractor = self.deep_feature_extractor
                self.deep_regressor_is_5w = True
                self.log("Deep remount predictor loaded (5w large-sample model)")
            else:
                # 最后回退: 旧版模型
                self.deep_regressor = load_deep_remount_predictor(
                    phase2_models_dir / "deep_remount_predictor.pkl"
                )
                if self.deep_regressor is not None:
                    self.deep_regressor.extractor = self.deep_feature_extractor
                    self.log("Deep remount predictor loaded (legacy model)")

            if self.deep_regressor is not None:
                pass  # 已在上面各自记录

            self._ml_ready = self.deep_classifier is not None or self.deep_regressor is not None
            if self._ml_ready:
                self.log("Deep ML recognition ENABLED")
        except Exception as e:
            self.log(f"Deep ML models not available: {e}")

    def _use_ml(self):
        """检查 ML 模型是否全部就绪"""
        return self._ml_ready and self.deep_classifier is not None

    def set_log_callback(self, callback):
        self.log_callback = callback

    def set_status_callback(self, callback):
        self.status_callback = callback

    def set_persist_default_callback(self, callback):
        self.persist_default_callback = callback

    def set_animation_step_callback(self, callback):
        self.animation_step_callback = callback

    def log(self, message):
        if self.log_callback is not None:
            self.log_callback(message)
        else:
            print(message)
        if self.status_callback is not None:
            self.status_callback()

    def _resolve_project_path(self, path_str):
        path = Path(path_str)
        if path.is_absolute():
            return path
        cwd_candidate = Path.cwd() / path
        if cwd_candidate.exists():
            return cwd_candidate
        return BASE_DIR / path

    def _load_optional_model(self, model_path, label):
        try:
            if Path(model_path).exists():
                model = joblib.load(model_path)
                self.log(f"Loaded {label}: {model_path}")
                return model
        except Exception as e:
            self.log(f"Failed to load {label} from {model_path}: {e}")
        return None

    def _begin_action(self, action_name, message):
        if action_name in self.busy_actions:
            self.log(message)
            return False
        self.busy_actions.add(action_name)
        return True

    def _end_action(self, action_name):
        self.busy_actions.discard(action_name)

    def _camera_frame_only_relocation_enabled(self):
        return bool(getattr(self.state, "relocation_use_camera_frames_only", False))

    def _lowmag_only_relocation_enabled(self):
        return bool(getattr(self.state, "relocation_lowmag_only_mode", False))

    def _site_memory_lowmag_ready(self, site_memory=None):
        site_memory = (self.state.site_memory if site_memory is None else site_memory) or {}
        lowmag_landmarks = site_memory.get("lowmag_landmarks") or []
        min_landmarks = int(site_memory.get("lowmag_ready_min_landmarks", self.state.reference_lowmag_min_landmarks))
        if site_memory.get("lowmag_ready") is not None:
            return bool(site_memory.get("lowmag_ready"))
        return bool(site_memory.get("overview") is not None and len(lowmag_landmarks) >= min_landmarks)

    def _get_current_relocation_view(self, prefer_matching=False):
        if prefer_matching:
            return (
                self.state.current_matching_view
                if self.state.current_matching_view is not None
                else self.state.current_camera_view
        )
        return (
            self.state.current_camera_view
            if self.state.current_camera_view is not None
            else self.state.current_matching_view
        )

    def _predict_lowmag_candidate_target(self, coarse_candidate):
        if coarse_candidate is None:
            return None
        predicted_dx_um = None
        predicted_dy_um = None
        ml_correction = coarse_candidate.get("ml_correction")
        if ml_correction is not None:
            predicted_dx_um = float(ml_correction["predicted_dx_um"])
            predicted_dy_um = float(ml_correction["predicted_dy_um"])
        elif (
            coarse_candidate.get("estimated_tip_dx_um") is not None
            and coarse_candidate.get("estimated_tip_dy_um") is not None
        ):
            predicted_dx_um = float(coarse_candidate["estimated_tip_dx_um"])
            predicted_dy_um = float(coarse_candidate["estimated_tip_dy_um"])
        if predicted_dx_um is None or predicted_dy_um is None:
            return None
        current_tip_x = float(getattr(self.state, "probe_tip_x", self.state.x + self.state.fov_width * 0.5))
        current_tip_y = float(getattr(self.state, "probe_tip_y", self.state.y + self.state.fov_height * 0.5))
        return self._clamp_to_stage_margin(
            current_tip_x + predicted_dx_um - float(self.state.fov_width) * 0.5,
            current_tip_y + predicted_dy_um - float(self.state.fov_height) * 0.5,
        )

    def _relocate_using_current_camera_frame(self, mode_label):
        if self.state.site_memory is None:
            self._try_load_latest_site_memory()
        if self.state.site_memory is not None and self.state.ref_template is None:
            self._activate_site_memory(self.state.site_memory, source_dir=self.state.last_saved_site_dir)
        if self.state.ref_template is None:
            self.log("Please save reference position first")
            return

        site_memory = self.state.site_memory or {}
        self.log(f"{mode_label}: using camera-frame-only relocation (true sample image disabled)")
        coarse_result = None
        fine_match = None
        verification = None
        lowmag_only_mode = self._lowmag_only_relocation_enabled()
        final_zoom = float(site_memory.get("final_zoom_level", site_memory.get("zoom_level", self.state.current_zoom_level)))
        if self._site_memory_lowmag_ready(site_memory):
            coarse_zoom = float(
                (site_memory.get("overview") or {}).get(
                    "zoom_level",
                    site_memory.get("coarse_zoom_level", min(self.state.zoom_levels)),
                )
            )
            if not np.isclose(float(self.state.current_zoom_level), coarse_zoom):
                self._begin_quantized_zoom(coarse_zoom)
                self.log(f"{mode_label}: switching to low magnification {coarse_zoom:.2f}x for landmark search")
                self._wait_for_zoom_complete()
            coarse_result = self._run_lowmag_landmark_search(site_memory=site_memory, zoom_level=coarse_zoom)
        elif site_memory.get("lowmag_landmarks"):
            self.log(
                f"{mode_label}: skipping coarse low-mag relocation because saved site memory is not low-mag ready "
                f"({len(site_memory.get('lowmag_landmarks', []))} landmarks)."
            )
        if lowmag_only_mode:
            self.log(f"{mode_label}: low-magnification-only mode enabled; staying at coarse zoom.")
        if not lowmag_only_mode and not np.isclose(float(self.state.current_zoom_level), final_zoom):
            self._begin_quantized_zoom(final_zoom)
            self.log(f"{mode_label}: returning to saved final zoom {final_zoom:.2f}x")
            self._wait_for_zoom_complete()

        coarse_candidates = []
        if coarse_result is not None:
            coarse_candidates = list(coarse_result.get("coarse_candidates", []))
            if not coarse_candidates:
                coarse_candidates = [coarse_result]
        evaluated_candidates = []
        fine_try_count = max(int(self.state.lowmag_search_fine_try_count), 1)
        if lowmag_only_mode and coarse_candidates:
            best_coarse_candidate = max(
                coarse_candidates,
                key=lambda item: (
                    float(item.get("candidate_rank_score", float("-inf"))),
                    self._score_lowmag_search_candidate(item),
                ),
            )
            predicted_target = self._predict_lowmag_candidate_target(best_coarse_candidate)
            if predicted_target is not None:
                self._jump_view_to_target(predicted_target[0], predicted_target[1])
                coarse_result["selected_candidate_index"] = int(best_coarse_candidate.get("search_index", 1))
                coarse_result["selected_candidate_target_x_um"] = float(predicted_target[0])
                coarse_result["selected_candidate_target_y_um"] = float(predicted_target[1])
                verification = {
                    "verified": False,
                    "reference_score": float(best_coarse_candidate.get("overview_similarity", 0.0)),
                    "reference_score_gap": float(best_coarse_candidate.get("candidate_rank_gap", 0.0)),
                    "mode": "lowmag_only",
                }
                self.log(
                    "Low-mag coarse relocation applied: "
                    f"target X={predicted_target[0]:.1f} um, Y={predicted_target[1]:.1f} um, "
                    f"support={best_coarse_candidate.get('support_count', 0)}, "
                    f"conf={best_coarse_candidate.get('confidence', 0.0):.3f}, "
                    f"geom={best_coarse_candidate.get('geometry_confidence', 0.0):.3f}, "
                    f"overview={best_coarse_candidate.get('overview_similarity', 0.0):.3f}"
                )
                self.log(
                    "Low-mag-only mode stops here. High-magnification refinement and verification are intentionally disabled."
                )
            else:
                self.log("Low-mag-only mode could not derive a usable coarse jump from the saved landmarks.")
        elif coarse_candidates:
            self.log(
                f"{mode_label}: evaluating {min(len(coarse_candidates), fine_try_count)} low-mag candidate(s) at final zoom"
            )
            for index, coarse_candidate in enumerate(coarse_candidates[:fine_try_count], start=1):
                predicted_target = self._predict_lowmag_candidate_target(coarse_candidate)
                if predicted_target is None:
                    continue
                predicted_x, predicted_y = predicted_target
                self._jump_view_to_target(predicted_x, predicted_y)
                self.log(
                    "Low-mag candidate "
                    f"{index}: target X={predicted_x:.1f} um, Y={predicted_y:.1f} um, "
                    f"support={coarse_candidate.get('support_count', 0)}, "
                    f"conf={coarse_candidate.get('confidence', 0.0):.3f}, "
                    f"geom={coarse_candidate.get('geometry_confidence', 0.0):.3f}, "
                    f"overview={coarse_candidate.get('overview_similarity', 0.0):.3f}"
                )
                if index == 1:
                    self.log(
                        "If the viewport is moving too slowly, press the 'Relocation Go Now' button to jump immediately."
                    )
                candidate_fine_match = self._match_camera_template(
                    self.state.ref_template,
                    float(self.state.x),
                    float(self.state.y),
                    half_range_um=float(self.state.relocation_fine_half_range_um),
                )
                candidate_verification = self._verify_relocation(
                    float(self.state.x),
                    float(self.state.y),
                    site_memory=site_memory,
                )
                evaluated_candidates.append(
                    {
                        "coarse_candidate": coarse_candidate,
                        "target_x_um": float(self.state.x),
                        "target_y_um": float(self.state.y),
                        "fine_match": candidate_fine_match,
                        "verification": candidate_verification,
                    }
                )
                self.log(
                    "High-mag re-check "
                    f"{index}: fine score={0.0 if candidate_fine_match is None else candidate_fine_match.get('score', 0.0):.3f}, "
                    f"verify={candidate_verification.get('verified', False)}, "
                    f"reference={candidate_verification.get('reference_score', 0.0):.3f}, "
                    f"gap={candidate_verification.get('reference_score_gap', 0.0):.3f}"
                )
                if candidate_verification.get("verified", False):
                    break

        if evaluated_candidates:
            evaluated_candidates.sort(
                key=lambda item: (
                    1 if item["verification"].get("verified", False) else 0,
                    float(item["verification"].get("reference_score", 0.0)),
                    -float(item["verification"].get("reference_score_gap", 0.0)),
                    -float(0.0 if item["fine_match"] is None else item["fine_match"].get("score", 0.0)),
                    float(item["coarse_candidate"].get("candidate_rank_score", 0.0)),
                ),
                reverse=True,
            )
            best_eval = evaluated_candidates[0]
            self._jump_view_to_target(best_eval["target_x_um"], best_eval["target_y_um"])
            fine_match = best_eval["fine_match"]
            verification = best_eval["verification"]
            if coarse_result is not None:
                coarse_result["selected_candidate_index"] = int(
                    best_eval["coarse_candidate"].get("search_index", 1)
                )
                coarse_result["selected_candidate_target_x_um"] = float(best_eval["target_x_um"])
                coarse_result["selected_candidate_target_y_um"] = float(best_eval["target_y_um"])
                coarse_result["evaluated_candidate_count"] = int(len(evaluated_candidates))
        else:
            fine_match = self._match_camera_template(
                self.state.ref_template,
                float(self.state.x),
                float(self.state.y),
                half_range_um=float(self.state.relocation_fine_half_range_um),
            )
            verification = self._verify_relocation(
                float(self.state.x),
                float(self.state.y),
                site_memory=site_memory,
            )

        self.state.last_relocation_report = {
            "affine": None,
            "coarse": coarse_result,
            "fine_affine": None,
            "fine": fine_match,
            "predicted_current_top_left": {"x_um": float(self.state.x), "y_um": float(self.state.y)},
            "verification": verification,
        }

        if fine_match is not None:
            self.log(
                "Camera-frame reference match: "
                f"score={fine_match['score']:.3f}, gap={fine_match.get('score_gap', 0.0):.3f}"
            )
        if coarse_result is not None:
            self.log(
                "Camera low-mag guidance: "
                f"dX={coarse_result['offset_x_um']:+.1f} um, "
                f"dY={coarse_result['offset_y_um']:+.1f} um, "
                f"support={coarse_result['support_count']}, "
                f"confidence={coarse_result['confidence']:.3f}"
            )

        if lowmag_only_mode:
            if coarse_result is not None:
                self.state.sample_removed = False
            else:
                self.log(
                    "Low-mag-only relocation did not produce a usable coarse result. "
                    "Stay at low magnification and adjust landmarks or region manually."
                )
                self._enter_manual_landmark_guidance(float(self.state.x), float(self.state.y))
            return

        if verification.get("verified", False):
            self.state.sample_removed = False
            self.log(
                "Relocation verified from camera frames only: "
                f"reference score={verification['reference_score']:.3f}, "
                f"gap={verification['reference_score_gap']:.3f}"
            )
            return

        self.log(
            "Camera-frame relocation did not verify the site: "
            f"reference score={verification.get('reference_score', 0.0):.3f}, "
            f"gap={verification.get('reference_score_gap', 0.0):.3f}"
        )
        self._enter_manual_landmark_guidance(float(self.state.x), float(self.state.y))

    def _score_lowmag_search_candidate(self, report):
        if report is None:
            return (-1, -1.0, float("-inf"))
        support = int(report.get("support_count", 0))
        geometry_confidence = float(report.get("geometry_confidence", 0.0))
        confidence = float(report.get("confidence", 0.0))
        overview_similarity = float(report.get("overview_similarity", 0.0))
        distance_confidence = float(report.get("distance_confidence", 0.0))
        manual_constellation_confidence = float(report.get("manual_constellation_confidence", 0.0))
        tip_distance = report.get("estimated_tip_distance_um")
        if tip_distance is None:
            tip_distance = float("inf")
        return (
            support,
            manual_constellation_confidence,
            geometry_confidence,
            confidence,
            distance_confidence,
            overview_similarity,
            -float(tip_distance),
        )

    def _numeric_lowmag_search_score(self, report):
        if report is None:
            return float("-inf")
        support_term = min(float(report.get("support_count", 0)), 5.0) / 5.0
        geometry_term = float(report.get("geometry_confidence", 0.0))
        confidence_term = float(report.get("confidence", 0.0))
        overview_term = max(float(report.get("overview_similarity", 0.0)), 0.0)
        distance_term = float(report.get("distance_confidence", 0.0))
        constellation_term = float(report.get("manual_constellation_confidence", 0.0))
        return float(
            0.20 * support_term
            + 0.22 * geometry_term
            + 0.18 * confidence_term
            + 0.12 * distance_term
            + 0.08 * overview_term
            + 0.20 * constellation_term
        )

    def _save_lowmag_search_trace(self, search_trace, best_report=None, site_memory=None):
        if not search_trace:
            return None
        sample_name = str((site_memory or {}).get("sample_image", "sample")).strip() or "sample"
        site_name = str((site_memory or {}).get("site_name", "site")).strip() or "site"
        safe_sample = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in sample_name)
        safe_site = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in site_name)
        timestamp = __import__("time").strftime("%Y%m%d_%H%M%S")
        output_dir = BASE_DIR / "collected_data" / "relocation_debug"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"lowmag_search_trace_{safe_sample}_{safe_site}_{timestamp}.csv"
        fieldnames = [
            "index",
            "ring",
            "grid_dx",
            "grid_dy",
            "x_um",
            "y_um",
            "support_count",
            "confidence",
            "overview_similarity",
            "estimated_tip_dx_um",
            "estimated_tip_dy_um",
            "estimated_tip_distance_um",
            "ml_predicted_offset_x_um",
            "ml_predicted_offset_y_um",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in search_trace:
                writer.writerow({name: row.get(name) for name in fieldnames})
        if best_report is not None:
            best_report["search_trace_csv"] = str(output_path)
        return output_path

    def _predict_lowmag_landmark_correction(self, report, reference_overview, current_overview):
        bundle = self.lowmag_landmark_regressor
        model = None if bundle is None else bundle.get("model")
        if model is None or report is None or reference_overview is None or current_overview is None:
            return None
        reference_image = to_grayscale_u8(reference_overview.get("image"))
        current_image = to_grayscale_u8(current_overview.get("image"))
        if reference_image is None or current_image is None:
            return None

        geom_features = np.array(
            [
                float(report.get("support_count", 0)),
                float(report.get("confidence", 0.0)),
                float(report.get("offset_x_um", 0.0)),
                float(report.get("offset_y_um", 0.0)),
                float(np.hypot(float(report.get("offset_x_um", 0.0)), float(report.get("offset_y_um", 0.0)))),
            ],
            dtype=np.float32,
        )
        img_features = pair_features(reference_image, current_image).astype(np.float32)
        features = np.concatenate([geom_features, img_features]).reshape(1, -1)
        try:
            prediction = np.asarray(model.predict(features), dtype=np.float32).reshape(-1)
        except Exception as exc:
            self.log(f"Low-mag landmark regressor inference failed: {exc}")
            return None
        if prediction.size < 2:
            return None
        correction_dx_um = float(prediction[0])
        correction_dy_um = float(prediction[1])
        return {
            "correction_dx_um": correction_dx_um,
            "correction_dy_um": correction_dy_um,
            "predicted_dx_um": float(report.get("offset_x_um", 0.0)) + correction_dx_um,
            "predicted_dy_um": float(report.get("offset_y_um", 0.0)) + correction_dy_um,
        }

    def _iter_lowmag_search_positions(self, anchor_x_um, anchor_y_um, step_x_um, step_y_um, max_rings):
        seen = set()
        for ring in range(0, max(int(max_rings), 0) + 1):
            ring_positions = []
            if ring == 0:
                ring_positions.append((0, 0))
            else:
                for dy_index in range(-ring, ring + 1):
                    for dx_index in range(-ring, ring + 1):
                        if max(abs(dx_index), abs(dy_index)) != ring:
                            continue
                        ring_positions.append((dx_index, dy_index))
                ring_positions.sort(key=lambda item: (abs(item[0]) + abs(item[1]), abs(item[1]), abs(item[0])))
            for dx_index, dy_index in ring_positions:
                candidate_x, candidate_y = self._clamp_to_stage_margin(
                    float(anchor_x_um) + float(dx_index) * float(step_x_um),
                    float(anchor_y_um) + float(dy_index) * float(step_y_um),
                )
                key = (round(float(candidate_x), 3), round(float(candidate_y), 3))
                if key in seen:
                    continue
                seen.add(key)
                yield {
                    "ring": int(ring),
                    "grid_dx": int(dx_index),
                    "grid_dy": int(dy_index),
                    "x_um": float(candidate_x),
                    "y_um": float(candidate_y),
                }

    def _run_lowmag_landmark_search(self, *, site_memory=None, zoom_level=None):
        site_memory = self.state.site_memory if site_memory is None else site_memory
        site_memory = site_memory or {}
        lowmag_landmarks = site_memory.get("lowmag_landmarks") or []
        if not lowmag_landmarks:
            self.state.lowmag_guidance_report = None
            return None
        if len(lowmag_landmarks) < 2:
            self.state.lowmag_guidance_report = None
            self.log(
                "Low-mag landmark search skipped: saved site has fewer than 2 low-mag landmarks. "
                "Capture several landmarks before relying on automatic coarse relocation."
            )
            return None
        reference_overview = site_memory.get("overview") or {}

        zoom_level = float(
            self.state.current_zoom_level if zoom_level is None else zoom_level
        )
        if not np.isclose(float(self.state.current_zoom_level), zoom_level):
            self._begin_quantized_zoom(zoom_level)
            self._wait_for_zoom_complete()

        anchor_x_um = float(self.state.x)
        anchor_y_um = float(self.state.y)
        step_x_um = max(20.0, float(self.state.fov_width) * float(self.state.lowmag_search_step_fraction))
        step_y_um = max(20.0, float(self.state.fov_height) * float(self.state.lowmag_search_step_fraction))
        min_support = max(int(self.state.relocation_min_landmark_support), int(self.state.lowmag_search_min_support))
        min_confidence = max(
            float(self.state.relocation_min_match_score) - 0.04,
            float(self.state.lowmag_search_min_confidence),
        )
        min_geometry_confidence = float(self.state.lowmag_search_min_geometry_confidence)
        min_overview_similarity = float(self.state.lowmag_search_min_overview_similarity)
        candidate_limit = max(int(self.state.lowmag_search_candidate_limit), 1)
        fast_fail_enabled = bool(getattr(self.state, "lowmag_search_fast_fail_enabled", True))
        fast_fail_max_frames = max(int(getattr(self.state, "lowmag_search_fast_fail_max_frames", 6)), 1)
        fast_fail_min_support = max(int(getattr(self.state, "lowmag_search_fast_fail_min_support", 2)), 1)
        fast_fail_min_confidence = float(getattr(self.state, "lowmag_search_fast_fail_min_confidence", 0.55))
        fast_fail_min_overview_similarity = float(
            getattr(self.state, "lowmag_search_fast_fail_min_overview_similarity", 0.45)
        )
        manual_authoritative = site_memory.get("lowmag_landmark_source") == "manual_authoritative"
        reference_overview_image = to_grayscale_u8(reference_overview.get("image"))

        best_report = None
        best_position = None
        candidate_reports = []
        search_trace = []
        stop_reason = "not_found"

        for index, candidate in enumerate(
            self._iter_lowmag_search_positions(
                anchor_x_um,
                anchor_y_um,
                step_x_um,
                step_y_um,
                self.state.lowmag_search_max_rings,
            ),
            start=1,
        ):
            self._jump_view_to_target(candidate["x_um"], candidate["y_um"])
            overview = self._build_camera_overview(
                zoom_level=zoom_level,
                center_x_um=float(self.state.probe_tip_x),
                center_y_um=float(self.state.probe_tip_y),
            )
            report = self._build_lowmag_guidance_report(overview, site_memory=site_memory)
            trace_entry = {
                "index": int(index),
                "ring": int(candidate["ring"]),
                "grid_dx": int(candidate["grid_dx"]),
                "grid_dy": int(candidate["grid_dy"]),
                "x_um": float(self.state.x),
                "y_um": float(self.state.y),
                "support_count": 0 if report is None else int(report.get("support_count", 0)),
                "confidence": 0.0 if report is None else float(report.get("confidence", 0.0)),
                "overview_similarity": 0.0,
                "estimated_tip_dx_um": None if report is None else report.get("estimated_tip_dx_um"),
                "estimated_tip_dy_um": None if report is None else report.get("estimated_tip_dy_um"),
                "estimated_tip_distance_um": (
                    None if report is None else report.get("estimated_tip_distance_um")
                ),
                "ml_predicted_offset_x_um": None,
                "ml_predicted_offset_y_um": None,
            }
            if report is not None:
                report = dict(report)
                report["overview"] = overview
                report["overview_similarity"] = self._score_matching_views(
                    reference_overview_image,
                    overview.get("image"),
                )
                trace_entry["overview_similarity"] = float(report["overview_similarity"])
                ml_correction = self._predict_lowmag_landmark_correction(
                    report,
                    reference_overview,
                    overview,
                )
                if ml_correction is not None:
                    report["ml_correction"] = ml_correction
                    report["ml_predicted_offset_x_um"] = float(ml_correction["predicted_dx_um"])
                    report["ml_predicted_offset_y_um"] = float(ml_correction["predicted_dy_um"])
                    trace_entry["ml_predicted_offset_x_um"] = report["ml_predicted_offset_x_um"]
                    trace_entry["ml_predicted_offset_y_um"] = report["ml_predicted_offset_y_um"]
                self.log(
                    "Low-mag search frame "
                    f"{index}: ring={candidate['ring']}, "
                    f"X={self.state.x:.1f} um, Y={self.state.y:.1f} um, "
                    f"support={trace_entry['support_count']}, "
                    f"confidence={trace_entry['confidence']:.3f}, "
                    f"overview={trace_entry['overview_similarity']:.3f}"
                )
                if best_report is None or self._score_lowmag_search_candidate(report) > self._score_lowmag_search_candidate(best_report):
                    best_report = dict(report)
                    best_position = (float(self.state.x), float(self.state.y))
                if (
                    trace_entry["support_count"] >= max(min_support - 1, 2)
                    or report.get("geometry_confidence", 0.0) >= max(min_geometry_confidence - 0.05, 0.20)
                    or trace_entry["confidence"] >= max(min_confidence - 0.06, 0.30)
                ):
                    candidate_copy = dict(report)
                    candidate_copy["search_index"] = int(index)
                    candidate_copy["search_ring"] = int(candidate["ring"])
                    candidate_copy["search_grid_dx"] = int(candidate["grid_dx"])
                    candidate_copy["search_grid_dy"] = int(candidate["grid_dy"])
                    candidate_copy["search_top_left_x_um"] = float(self.state.x)
                    candidate_copy["search_top_left_y_um"] = float(self.state.y)
                    candidate_copy["candidate_rank_score"] = self._numeric_lowmag_search_score(candidate_copy)
                    candidate_reports.append(candidate_copy)
                if (
                    int(candidate["ring"]) >= 1
                    and
                    (
                        (
                            trace_entry["support_count"] >= min_support
                            and trace_entry["confidence"] >= min_confidence
                            and float(report.get("geometry_confidence", 0.0)) >= min_geometry_confidence
                            and float(report.get("overview_similarity", 0.0)) >= min_overview_similarity
                        )
                        or (
                            manual_authoritative
                            and trace_entry["support_count"] >= max(2, min_support - 1)
                            and float(report.get("manual_constellation_confidence", 0.0)) >= max(0.40, min_geometry_confidence)
                        )
                    )
                ):
                    stop_reason = "threshold_reached"
                    search_trace.append(trace_entry)
                    break
                if (
                    fast_fail_enabled
                    and int(index) >= fast_fail_max_frames
                    and (
                        (
                            not manual_authoritative
                            and (
                                trace_entry["support_count"] < fast_fail_min_support
                                or trace_entry["confidence"] < fast_fail_min_confidence
                                or trace_entry["overview_similarity"] < fast_fail_min_overview_similarity
                            )
                        )
                        or (
                            manual_authoritative
                            and trace_entry["support_count"] < max(2, fast_fail_min_support)
                            and float(report.get("manual_constellation_confidence", 0.0)) < 0.40
                        )
                    )
                ):
                    stop_reason = "fast_fail_weak_lowmag"
                    search_trace.append(trace_entry)
                    break
            search_trace.append(trace_entry)

        if best_position is not None:
            self._jump_view_to_target(best_position[0], best_position[1])
        if best_report is None:
            self.state.lowmag_guidance_report = None
            self.log("Low-mag landmark search did not find a usable landmark constellation.")
            return None
        if (
            stop_reason == "fast_fail_weak_lowmag"
            and (
                (
                    not manual_authoritative
                    and (
                        int(best_report.get("support_count", 0)) < fast_fail_min_support
                        or float(best_report.get("confidence", 0.0)) < fast_fail_min_confidence
                        or float(best_report.get("overview_similarity", 0.0)) < fast_fail_min_overview_similarity
                    )
                )
                or (
                    manual_authoritative
                    and int(best_report.get("support_count", 0)) < max(2, fast_fail_min_support)
                    and float(best_report.get("manual_constellation_confidence", 0.0)) < 0.40
                )
            )
        ):
            self.state.lowmag_guidance_report = None
            self.log(
                "Low-mag landmark search fast-failed on weak evidence. "
                "Skipping coarse jump and falling back to fine camera verification."
            )
            return None

        best_report["search_trace"] = search_trace
        best_report["search_frames"] = len(search_trace)
        best_report["search_step_x_um"] = float(step_x_um)
        best_report["search_step_y_um"] = float(step_y_um)
        best_report["search_stop_reason"] = str(stop_reason)
        best_report["search_best_top_left_x_um"] = float(best_position[0])
        best_report["search_best_top_left_y_um"] = float(best_position[1])
        best_report["overview_similarity"] = float(best_report.get("overview_similarity", 0.0))
        if candidate_reports:
            candidate_reports.sort(
                key=lambda item: (
                    float(item.get("candidate_rank_score", float("-inf"))),
                    self._score_lowmag_search_candidate(item),
                ),
                reverse=True,
            )
            best_report["coarse_candidates"] = candidate_reports[:candidate_limit]
            best_report["candidate_rank_score"] = float(best_report.get("candidate_rank_score", self._numeric_lowmag_search_score(best_report)))
            if len(candidate_reports) > 1:
                best_report["candidate_rank_gap"] = float(
                    candidate_reports[0].get("candidate_rank_score", float("-inf"))
                    - candidate_reports[1].get("candidate_rank_score", float("-inf"))
                )
            else:
                best_report["candidate_rank_gap"] = float(candidate_reports[0].get("candidate_rank_score", 0.0))
        else:
            best_report["coarse_candidates"] = [dict(best_report)]
            best_report["candidate_rank_score"] = float(self._numeric_lowmag_search_score(best_report))
            best_report["candidate_rank_gap"] = float(best_report["candidate_rank_score"])
        trace_path = self._save_lowmag_search_trace(search_trace, best_report=best_report, site_memory=site_memory)
        self.state.lowmag_guidance_report = best_report
        self.log(
            "Low-mag landmark search best match: "
            f"frames={len(search_trace)}, support={best_report.get('support_count', 0)}, "
            f"confidence={best_report.get('confidence', 0.0):.3f}, "
            f"geometry={best_report.get('geometry_confidence', 0.0):.3f}, "
            f"overview={best_report.get('overview_similarity', 0.0):.3f}, stop={stop_reason}"
        )
        self.log(
            "Low-mag candidate ranking: "
            f"kept={len(best_report.get('coarse_candidates', []))}, "
            f"gap={best_report.get('candidate_rank_gap', 0.0):.3f}"
        )
        if trace_path is not None:
            self.log(f"Low-mag search trace saved: {trace_path}")
        ml_correction = best_report.get("ml_correction")
        if ml_correction is not None:
            self.log(
                "Low-mag ML correction: "
                f"dX={ml_correction['correction_dx_um']:+.1f} um, "
                f"dY={ml_correction['correction_dy_um']:+.1f} um, "
                f"predicted total dX={ml_correction['predicted_dx_um']:+.1f} um, "
                f"predicted total dY={ml_correction['predicted_dy_um']:+.1f} um"
            )
        return best_report

    def _start_smooth_move(self, target_x, target_y):
        self.state.smooth_move_active = True
        self.state.smooth_move_target_x = target_x
        self.state.smooth_move_target_y = target_y

    def _cancel_active_motion(self):
        self.state.auto_scan_active = False
        self.state.smooth_move_active = False
        self.state.smooth_move_target_x = float(self.state.x)
        self.state.smooth_move_target_y = float(self.state.y)
        self.state.target_x = float(self.state.x)
        self.state.target_y = float(self.state.y)
        if self.state.pi_mode:
            self.stage.reset(self.state.x, self.state.y)
            self.stage.cmd_x = float(self.state.x)
            self.stage.cmd_y = float(self.state.y)

    def _jump_view_to_target(self, target_x=None, target_y=None):
        target_x = float(self.state.target_x if target_x is None else target_x)
        target_y = float(self.state.target_y if target_y is None else target_y)
        target_x, target_y = self._clamp_to_stage_margin(target_x, target_y)
        delta_x = float(target_x - self.state.x)
        delta_y = float(target_y - self.state.y)
        self.state.auto_scan_active = False
        self.state.smooth_move_active = False
        self.state.smooth_move_target_x = target_x
        self.state.smooth_move_target_y = target_y
        self.state.x = target_x
        self.state.y = target_y
        self.state.target_x = target_x
        self.state.target_y = target_y
        self._translate_probe_tip(delta_x, delta_y)
        if self.state.pi_mode:
            self.stage.reset(self.state.x, self.state.y)
            self.stage.cmd_x = float(self.state.x)
            self.stage.cmd_y = float(self.state.y)
        self._refresh_current_view()
        self.update_title()
        return delta_x, delta_y

    def _get_zoom_level_index(self, zoom_level):
        levels = np.array(self.state.zoom_levels, dtype=float)
        return int(np.argmin(np.abs(levels - float(zoom_level))))

    def _wait_for_zoom_complete(self, timeout_sec=2.0, poll_interval=0.05):
        """Block until the current zoom animation finishes, or timeout."""
        import time as _time
        if self.animation_step_callback is not None:
            max_iterations = max(int(self.state.zoom_steps) + 5, 25)
            for _ in range(max_iterations):
                if not self.state.zooming:
                    break
                self.animation_step_callback()
            if self.state.zooming:
                self.log("Zoom wait timed out — forcing zoom completion")
                self.state.current_zoom_level = float(self.state.target_zoom_level)
                self.state.fov_width, self.state.fov_height = self.state.get_fov_for_zoom_level(self.state.current_zoom_level)
                self.state.zooming = False
                self.state.zoom_progress = 0
            self._refresh_current_view()
            return
        start = _time.time()
        while self.state.zooming:
            self.fig.canvas.draw_idle()
            flush_events = getattr(self.fig.canvas, "flush_events", None)
            if callable(flush_events):
                flush_events()
            if _time.time() - start > timeout_sec:
                self.log("Zoom wait timed out — forcing zoom completion")
                self.state.current_zoom_level = float(self.state.target_zoom_level)
                self.state.fov_width, self.state.fov_height = self.state.get_fov_for_zoom_level(self.state.current_zoom_level)
                self.state.zooming = False
                self.state.zoom_progress = 0
                break
            _time.sleep(poll_interval)
        # Force a view refresh
        self._refresh_current_view()

    def _begin_quantized_zoom(self, target_zoom_level):
        levels = list(self.state.zoom_levels)
        target_zoom_level = float(levels[self._get_zoom_level_index(target_zoom_level)])
        if self.state.zooming:
            self.log("Zoom already in progress")
            return
        if np.isclose(target_zoom_level, float(self.state.current_zoom_level)):
            self.log(f"Zoom already at {target_zoom_level:g}x")
            return

        self.state.zooming = True
        self.state.zoom_progress = 0
        self.state.zoom_direction = 1 if target_zoom_level > self.state.current_zoom_level else -1
        self.state.zoom_base_width = self.state.fov_width
        self.state.zoom_base_height = self.state.fov_height
        self.state.zoom_target_width, self.state.zoom_target_height = self.state.get_fov_for_zoom_level(target_zoom_level)
        self.state.target_zoom_level = target_zoom_level
        self.state.zoom_center_x = float(self.state.probe_tip_x)
        self.state.zoom_center_y = float(self.state.probe_tip_y)
        self.log(
            f"Zoom transition: {self.state.current_zoom_level:g}x -> {target_zoom_level:g}x "
            f"around tip X={self.state.zoom_center_x:.1f} um, Y={self.state.zoom_center_y:.1f} um"
        )

    def _set_probe_tip_to_view_center(self):
        self.state.probe_tip_x = float(self.state.x + self.state.fov_width * self.PROBE_TIP_REL_X)
        self.state.probe_tip_y = float(self.state.y + self.state.fov_height * self.PROBE_TIP_REL_Y)

    def _reset_view_to_zoom(self, zoom_level, center_x_um=None, center_y_um=None):
        zoom_level = float(self.state.zoom_levels[self._get_zoom_level_index(zoom_level)])
        center_x_um = float(self.state.probe_tip_x if center_x_um is None else center_x_um)
        center_y_um = float(self.state.probe_tip_y if center_y_um is None else center_y_um)
        self.state.zooming = False
        self.state.zoom_progress = 0
        self.state.zoom_direction = 0
        self.state.current_zoom_level = zoom_level
        self.state.target_zoom_level = zoom_level
        self.state.fov_width, self.state.fov_height = self.state.get_fov_for_zoom_level(zoom_level)
        target_x = float(center_x_um - self.state.fov_width * self.PROBE_TIP_REL_X)
        target_y = float(center_y_um - self.state.fov_height * self.PROBE_TIP_REL_Y)
        target_x, target_y = self._clamp_to_stage_margin(target_x, target_y)
        self.state.x = target_x
        self.state.y = target_y
        self.state.target_x = target_x
        self.state.target_y = target_y
        self._set_probe_tip_to_view_center()

    def _set_random_start_view(self):
        self._reset_view_to_zoom(min(self.state.zoom_levels), center_x_um=self.state.x + self.state.fov_width * 0.5, center_y_um=self.state.y + self.state.fov_height * 0.5)
        max_x = max(float(self.state.width_um) - float(self.state.fov_width), 0.0)
        max_y = max(float(self.state.height_um) - float(self.state.fov_height), 0.0)
        self.state.x = float(random.uniform(0.0, max_x)) if max_x > 0.0 else 0.0
        self.state.y = float(random.uniform(0.0, max_y)) if max_y > 0.0 else 0.0
        self.state.target_x = self.state.x
        self.state.target_y = self.state.y
        self._set_probe_tip_to_view_center()

    def _translate_probe_tip(self, delta_x, delta_y):
        self.state.probe_tip_x = float(self.state.probe_tip_x + delta_x)
        self.state.probe_tip_y = float(self.state.probe_tip_y + delta_y)

    def _pan_viewport_without_moving_probe(self, delta_x, delta_y):
        new_x, new_y = self._clamp_to_stage_margin(self.state.x + delta_x, self.state.y + delta_y)
        actual_dx = float(new_x - self.state.x)
        actual_dy = float(new_y - self.state.y)
        if np.isclose(actual_dx, 0.0) and np.isclose(actual_dy, 0.0):
            self.log("Viewport pan reached the stage limit")
            return
        self.state.auto_scan_active = False
        self.state.smooth_move_active = False
        self.state.x = new_x
        self.state.y = new_y
        self.state.target_x = new_x
        self.state.target_y = new_y
        if self.state.pi_mode:
            self.stage.reset(self.state.x, self.state.y)
            self.stage.cmd_x = self.state.x
            self.stage.cmd_y = self.state.y
        self.update_probe_visuals()
        if self.status_callback is not None:
            self.status_callback()
        self.log(
            f"Viewport panned by dX={actual_dx:+.1f} um, dY={actual_dy:+.1f} um while keeping the cantilever fixed on the sample stage"
        )

    def handle_viewport_arrow_click(self, event):
        if event.inaxes != self.ax or event.x is None or event.y is None:
            return False
        if getattr(event, "button", None) != 1:
            return False
        if self.state.zooming:
            self.log("Wait for zoom to finish before panning the viewport.")
            return True

        x_ax, y_ax = self.ax.transAxes.inverted().transform((event.x, event.y))
        hit = float(self.VIEWPORT_ARROW_HIT_RADIUS_AX)
        step = float(self.state.current_step)

        if abs(x_ax - 0.50) <= hit and abs(y_ax - 0.95) <= hit:
            self._pan_viewport_without_moving_probe(0.0, -step)
            return True
        if abs(x_ax - 0.50) <= hit and abs(y_ax - 0.03) <= hit:
            self._pan_viewport_without_moving_probe(0.0, step)
            return True
        if abs(x_ax - 0.03) <= hit and abs(y_ax - 0.50) <= hit:
            self._pan_viewport_without_moving_probe(-step, 0.0)
            return True
        if abs(x_ax - 0.97) <= hit and abs(y_ax - 0.50) <= hit:
            self._pan_viewport_without_moving_probe(step, 0.0)
            return True
        return False

    def update_probe_visuals(self):
        tip_x = float(self.state.probe_tip_x)
        tip_y = float(self.state.probe_tip_y)
        zoom_level = max(float(self.state.current_zoom_level), 1e-6)
        camera_fixed_scale = 1.0 / zoom_level if zoom_level < 1.0 else 1.0
        geometry_scale = camera_fixed_scale
        tip_half_width_um = (self.PROBE_TIP_WIDTH_UM / 2.0) * geometry_scale
        tip_total_length_um = self.PROBE_TIP_TOTAL_LENGTH_UM * geometry_scale
        triangular_tip_length_um = self.PROBE_TRIANGULAR_TIP_LENGTH_UM * geometry_scale
        visible_body_depth_um = self.PROBE_VISIBLE_BODY_DEPTH_UM * geometry_scale
        body_half_width_um = (self.PROBE_BODY_WIDTH_UM / 2.0) * geometry_scale

        triangle_base_y = tip_y + triangular_tip_length_um
        body_top_y = tip_y + tip_total_length_um
        body_bottom_y = body_top_y + visible_body_depth_um

        self.cantilever.set_transform(self.ax.transData)
        self.rod.set_transform(self.ax.transData)
        self.tip.set_transform(self.ax.transData)

        self.cantilever.set_xy(
            [
                [tip_x - body_half_width_um, body_bottom_y],
                [tip_x + body_half_width_um, body_bottom_y],
                [tip_x + body_half_width_um, body_top_y],
                [tip_x + tip_half_width_um, body_top_y],
                [tip_x + tip_half_width_um, triangle_base_y],
                [tip_x, tip_y],
                [tip_x - tip_half_width_um, triangle_base_y],
                [tip_x - tip_half_width_um, body_top_y],
                [tip_x - body_half_width_um, body_top_y],
            ]
        )

        self.rod.set_xy((tip_x, tip_y))
        self.rod.set_width(0.0)
        self.rod.set_height(0.0)

        self.tip.set_xy(
            [
                [tip_x - tip_half_width_um, triangle_base_y],
                [tip_x + tip_half_width_um, triangle_base_y],
                [tip_x, tip_y],
            ]
        )
        self.fig.canvas.draw_idle()

    def _sync_hud_master_state(self):
        self.state.show_probe_hud = bool(self.state.show_hud_detection or self.state.show_hud_distance)

    def get_hud_button_label(self):
        detection_on = bool(self.state.show_hud_detection)
        distance_on = bool(self.state.show_hud_distance)
        if detection_on and distance_on:
            mode = "ALL"
        elif detection_on:
            mode = "DET"
        elif distance_on:
            mode = "DIST"
        else:
            mode = "OFF"
        return f"HUD: {mode}"

    def toggle_probe_hud(self, event):
        modes = [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ]
        current = (bool(self.state.show_hud_detection), bool(self.state.show_hud_distance))
        try:
            current_index = modes.index(current)
        except ValueError:
            current_index = 0
        next_detection, next_distance = modes[(current_index + 1) % len(modes)]
        self.state.show_hud_detection = next_detection
        self.state.show_hud_distance = next_distance
        self._sync_hud_master_state()
        if "hud" in self.buttons:
            self.buttons["hud"].label.set_text(self.get_hud_button_label())
        if self.status_callback is not None:
            self.status_callback()
        self.fig.canvas.draw_idle()
        self.log(
            "Probe HUD mode: "
            f"detection={'ON' if self.state.show_hud_detection else 'OFF'}, "
            f"distance={'ON' if self.state.show_hud_distance else 'OFF'}"
        )

    def _clamp_z_stage(self, z_position_um):
        half_travel = float(self.state.z_stage_travel_um) / 2.0
        return float(np.clip(z_position_um, -half_travel, half_travel))

    def get_position_center(self):
        return (
            float(self.state.x + self.state.fov_width / 2.0),
            float(self.state.y + self.state.fov_height / 2.0),
        )

    def get_target_center(self):
        return (
            float(self.state.target_x + self.state.fov_width / 2.0),
            float(self.state.target_y + self.state.fov_height / 2.0),
        )

    def set_origin(self, x_um, y_um, label=None):
        self.state.origin_x = float(x_um)
        self.state.origin_y = float(y_um)
        if label:
            self.state.origin_label = str(label)
        elif not self.state.origin_label:
            self.state.origin_label = "Origin"
        self.state.origin_defined = True
        self._capture_origin_template(self.state.origin_x, self.state.origin_y)
        if self.status_callback is not None:
            self.status_callback()
        self.fig.canvas.draw_idle()
        self.log(
            f"Origin set: {self.state.origin_label} at "
            f"X={self.state.origin_x:.1f} um, Y={self.state.origin_y:.1f} um"
        )

    def clear_origin(self):
        self.state.origin_defined = False
        self.state.origin_template = None
        if self.status_callback is not None:
            self.status_callback()
        self.fig.canvas.draw_idle()
        self.log("Origin cleared")

    def _capture_origin_template(self, abs_x, abs_y):
        source_view = self.state.current_matching_view if self.state.current_matching_view is not None else self.state.current_fov_raw
        if source_view is None:
            self.state.origin_template = None
            return
        rel_x = int(round(abs_x - self.state.x))
        rel_y = int(round(abs_y - self.state.y))
        half_size = int(self.state.origin_template_half_size)
        x0 = max(0, rel_x - half_size)
        x1 = min(source_view.shape[1], rel_x + half_size)
        y0 = max(0, rel_y - half_size)
        y1 = min(source_view.shape[0], rel_y + half_size)
        if x1 - x0 < 16 or y1 - y0 < 16:
            self.state.origin_template = None
            self.log("Origin template too small in current view; move origin away from the edge and set it again.")
            return
        self.state.origin_template = source_view[y0:y1, x0:x1].copy()

    def _load_sample_into_state(self, sample, width_um, height_um, sample_source, sample_path=None):
        self.state.surface_image = sample
        self.state.sample = self.state.surface_image
        self.state.surface_valid_mask = np.ones(sample.shape[:2], dtype=bool)
        self.state.width_um = float(width_um)
        self.state.height_um = float(height_um)
        self.state.sample_source = sample_source
        self.state.sample_path = str(sample_path) if sample_path else None
        if sample_path:
            self.state.default_image_path = str(sample_path)

        self._set_random_start_view()
        self.state.stage_margin_um = max(2000.0, float(max(self.state.width_um, self.state.height_um)) * 1.5)
        self.state.sample_removed = False
        self.state.show_artifact = False
        self.state.z_stage_position_um = float(
            self.state.get_effective_camera_stage_position_um() - self.state.focus_z_um
        )
        self.state.simulated_sample_shift_x_um = 0.0
        self.state.simulated_sample_shift_y_um = 0.0
        self.state.simulated_sample_rotation_deg = 0.0
        self.state.ai_desired_history_x = [self.state.x, self.state.x]
        self.state.ai_desired_history_y = [self.state.y, self.state.y]
        self._clear_reference_data()

        if self.artifact_layer is not None:
            self.artifact_layer.reset_canvas(sample.shape[1], sample.shape[0])

        self.stage.reset(self.state.x, self.state.y)
        self._refresh_current_view()
        self._try_load_latest_site_memory()
        self.update_title()

    def load_default_image(self, event=None):
        if not self._begin_action("load_default_image", "Default image loading is already running"):
            return
        try:
            candidate_paths = []
            if self.state.default_image_path:
                candidate_paths.append(Path(self.state.default_image_path))
            candidate_paths.extend(
                [
                    self._resolve_project_path("afm_ideal_scan.png"),
                    self._resolve_project_path("../gambar/figure4_vision.png"),
                ]
            )
            image_path = next((path for path in candidate_paths if path.exists()), None)
            if image_path is None:
                self.log("No default image found. Use Load Image to choose one.")
                return

            raw = plt.imread(image_path)
            if raw.ndim == 3:
                raw = raw[..., 0]
            scale_bar_um = 500.0
            scale_bar_px = 140.0
            self.state.default_scale_um_per_px = scale_bar_um / scale_bar_px
            default_width_um = self.state.default_image_width_um
            default_height_um = self.state.default_image_height_um
            if default_width_um and default_height_um:
                width_um = float(default_width_um)
                height_um = float(default_height_um)
            else:
                width_um = float(raw.shape[1])
                height_um = float(raw.shape[0])
            sample, width_um, height_um = load_real_sample_image(
                str(image_path),
                width_um,
                height_um,
            )
            self._load_sample_into_state(sample, width_um, height_um, "default-image", image_path)
            self.log(f"Loaded default image: {image_path}")
            if default_width_um and default_height_um:
                self.log(
                    f"Loaded saved default calibration: {width_um / 1000.0:.3f} mm x "
                    f"{height_um / 1000.0:.3f} mm"
                )
            else:
                self.log(
                    f"Default image scale estimate: {self.state.default_scale_um_per_px:.3f} um/px "
                    f"from 500 um over ~140 px"
                )
        except Exception as e:
            self.log(f"Failed to load default image: {e}")
        finally:
            self._end_action("load_default_image")

    def _clear_reference_data(self):
        self.state.ref_template = None
        self.state.ref_artefacts = []
        self.state.ref_x = 0.0
        self.state.ref_y = 0.0
        self.state.site_memory = None
        self.state.last_saved_site_dir = None
        self.state.last_relocation_report = None
        self.state.last_affine_transform_report = None

    def _refresh_current_view(self):
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
            tip_rel_x=self.PROBE_TIP_REL_X,
            tip_rel_y=self.PROBE_TIP_REL_Y,
            body_width_um=self.state.probe_body_width_um,
            tip_width_um=self.state.probe_tip_width_um,
            tip_total_length_um=self.state.probe_tip_total_length_um,
            triangular_tip_length_um=self.state.probe_triangular_tip_length_um,
            visible_body_depth_um=self.state.probe_visible_body_depth_um,
        )
        self.state.current_matching_view, _ = render_camera_matching_frame(
            fov,
            camera_resolution=self.MATCHING_RESOLUTION,
            outside_mask=outside_mask,
            focus_model=self.state.get_focus_model(),
            fov_width_um=self.state.fov_width,
            fov_height_um=self.state.fov_height,
            tip_rel_x=self.PROBE_TIP_REL_X,
            tip_rel_y=self.PROBE_TIP_REL_Y,
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
        self.img.set_extent([ix, ix + self.state.fov_width, iy + self.state.fov_height, iy])
        self.ax.set_xlim(ix, ix + self.state.fov_width)
        self.ax.set_ylim(iy + self.state.fov_height, iy)
        self.update_probe_visuals()
        self.fig.canvas.draw_idle()

    def _capture_recognition_view(
        self,
        top_left_x_um,
        top_left_y_um,
        fov_width_um,
        fov_height_um,
        *,
        zoom_level=None,
        camera_resolution=None,
    ):
        fov, outside_mask, _, _ = create_stage_fov(
            self.state.surface_image,
            self.artifact_layer,
            self.state.show_artifact,
            top_left_x_um,
            top_left_y_um,
            fov_width_um,
            fov_height_um,
            valid_mask=self.state.surface_valid_mask,
        )
        return render_camera_recognition_frame(
            fov,
            camera_resolution=(
                self.state.camera_resolution
                if camera_resolution is None
                else camera_resolution
            ),
            outside_mask=outside_mask,
            focus_model=self.state.get_focus_model(
                zoom_level=zoom_level,
                fov_width_um=fov_width_um,
                fov_height_um=fov_height_um,
            ),
            fov_width_um=fov_width_um,
            fov_height_um=fov_height_um,
            tip_rel_x=self.PROBE_TIP_REL_X,
            tip_rel_y=self.PROBE_TIP_REL_Y,
            body_width_um=self.state.probe_body_width_um,
            tip_width_um=self.state.probe_tip_width_um,
            tip_total_length_um=self.state.probe_tip_total_length_um,
            triangular_tip_length_um=self.state.probe_triangular_tip_length_um,
            visible_body_depth_um=self.state.probe_visible_body_depth_um,
        )[0]

    def _capture_matching_template(
        self,
        top_left_x_um,
        top_left_y_um,
        fov_width_um,
        fov_height_um,
        *,
        zoom_level=None,
        disable_blur=False,
    ):
        fov, outside_mask, _, _ = create_stage_fov(
            self.state.surface_image,
            self.artifact_layer,
            self.state.show_artifact,
            top_left_x_um,
            top_left_y_um,
            fov_width_um,
            fov_height_um,
            valid_mask=self.state.surface_valid_mask,
        )
        return render_camera_matching_frame(
            fov,
            camera_resolution=self.MATCHING_RESOLUTION,
            outside_mask=outside_mask,
            focus_model=(
                None
                if disable_blur
                else self.state.get_focus_model(
                    zoom_level=zoom_level,
                    fov_width_um=fov_width_um,
                    fov_height_um=fov_height_um,
                )
            ),
            fov_width_um=fov_width_um,
            fov_height_um=fov_height_um,
            tip_rel_x=self.PROBE_TIP_REL_X,
            tip_rel_y=self.PROBE_TIP_REL_Y,
            body_width_um=self.state.probe_body_width_um,
            tip_width_um=self.state.probe_tip_width_um,
            tip_total_length_um=self.state.probe_tip_total_length_um,
            triangular_tip_length_um=self.state.probe_triangular_tip_length_um,
            visible_body_depth_um=self.state.probe_visible_body_depth_um,
        )[0]

    def _capture_display_panel_snapshot(self):
        canvas = getattr(self.fig, "canvas", None)
        if canvas is None or self.ax is None:
            return None
        try:
            canvas.draw()
            rgba = np.asarray(canvas.buffer_rgba())
        except Exception:
            return None
        if rgba.size == 0:
            return None
        bbox = self.ax.get_window_extent()
        canvas_h, canvas_w = rgba.shape[:2]
        x0 = max(0, int(np.floor(bbox.x0)))
        x1 = min(canvas_w, int(np.ceil(bbox.x1)))
        y0 = max(0, int(np.floor(canvas_h - bbox.y1)))
        y1 = min(canvas_h, int(np.ceil(canvas_h - bbox.y0)))
        if x1 <= x0 or y1 <= y0:
            return None
        crop = rgba[y0:y1, x0:x1, :3]
        if crop.size == 0:
            return None
        return cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    def _capture_live_camera_reference_view(
        self,
        top_left_x_um,
        top_left_y_um,
        fov_width_um,
        fov_height_um,
        *,
        zoom_level=None,
        camera_resolution=None,
    ):
        fov, outside_mask, _, _ = create_stage_fov(
            self.state.surface_image,
            self.artifact_layer,
            self.state.show_artifact,
            top_left_x_um,
            top_left_y_um,
            fov_width_um,
            fov_height_um,
            valid_mask=self.state.surface_valid_mask,
        )
        display_view, _ = render_camera_frame(
            fov,
            camera_resolution=(
                self.state.camera_resolution
                if camera_resolution is None
                else camera_resolution
            ),
            outside_mask=outside_mask,
            focus_model=self.state.get_focus_model(
                zoom_level=zoom_level,
                fov_width_um=fov_width_um,
                fov_height_um=fov_height_um,
            ),
        )
        return to_grayscale_u8(display_view)

    def _score_matching_views(self, reference_view, candidate_view):
        reference = to_grayscale_u8(reference_view)
        candidate = to_grayscale_u8(candidate_view)
        if reference is None or candidate is None or reference.size == 0 or candidate.size == 0:
            return -1.0
        if reference.shape[:2] != candidate.shape[:2]:
            candidate = cv2.resize(candidate, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
        ref_f = reference.astype(np.float32)
        cand_f = candidate.astype(np.float32)
        ref_norm = cv2.normalize(ref_f, None, 0.0, 1.0, cv2.NORM_MINMAX)
        cand_norm = cv2.normalize(cand_f, None, 0.0, 1.0, cv2.NORM_MINMAX)
        ref_std = float(np.std(ref_norm))
        cand_std = float(np.std(cand_norm))
        if ref_std < 1e-6 and cand_std < 1e-6:
            mean_delta = abs(float(np.mean(ref_norm)) - float(np.mean(cand_norm)))
            return float(np.clip(1.0 - 2.0 * mean_delta, -1.0, 1.0))
        if ref_std < 1e-6 or cand_std < 1e-6:
            return -1.0
        score = float(np.corrcoef(ref_norm.reshape(-1), cand_norm.reshape(-1))[0, 1])
        if not np.isfinite(score):
            score = -1.0
        return score

    def _match_camera_template(self, reference_template, center_x_um, center_y_um, half_range_um):
        if reference_template is None:
            return None
        if self._camera_frame_only_relocation_enabled():
            current_view = self._get_current_relocation_view(prefer_matching=False)
            if current_view is None:
                return None
            score = self._score_matching_views(reference_template, current_view)
            return {
                "x": float(self.state.x),
                "y": float(self.state.y),
                "score": float(score),
                "score_gap": float(max(score, 0.0)),
                "candidates": [{"x": float(self.state.x), "y": float(self.state.y), "score": float(score)}],
                "level_candidates": [{"x": float(self.state.x), "y": float(self.state.y), "score": float(score)}],
            }
        start_x, start_y = self._clamp_to_stage_margin(center_x_um, center_y_um)
        current_center_x = float(start_x)
        current_center_y = float(start_y)
        base_step = max(20.0, half_range_um / 2.0)
        step_schedule = [
            float(base_step),
            float(max(base_step / 2.0, 10.0)),
            float(max(base_step / 4.0, 5.0)),
            float(max(base_step / 8.0, 2.0)),
        ]
        best = None
        all_candidates = {}
        final_candidates = []
        for step_um in step_schedule:
            local_offsets = tuple(float(offset) * float(step_um) for offset in (-2.0, -1.0, 0.0, 1.0, 2.0))
            level_candidates = []
            for dy_um in local_offsets:
                for dx_um in local_offsets:
                    x_um = current_center_x + float(dx_um)
                    y_um = current_center_y + float(dy_um)
                    top_left_x, top_left_y = self._clamp_to_stage_margin(float(x_um), float(y_um))
                    candidate_view = self._capture_live_camera_reference_view(
                        top_left_x,
                        top_left_y,
                        float(self.state.fov_width),
                        float(self.state.fov_height),
                        zoom_level=float(self.state.current_zoom_level),
                    )
                    score = self._score_matching_views(reference_template, candidate_view)
                    level_candidates.append(
                        {
                            "x": float(top_left_x),
                            "y": float(top_left_y),
                            "score": score,
                        }
                    )
                    candidate_key = (round(float(top_left_x), 3), round(float(top_left_y), 3))
                    previous = all_candidates.get(candidate_key)
                    if previous is None or score > previous["score"]:
                        all_candidates[candidate_key] = {
                            "x": float(top_left_x),
                            "y": float(top_left_y),
                            "score": score,
                        }
            if not level_candidates:
                continue
            level_candidates.sort(key=lambda item: item["score"], reverse=True)
            best = level_candidates[0]
            current_center_x = float(best["x"])
            current_center_y = float(best["y"])
            final_candidates = level_candidates[:5]

        if best is None:
            return None
        ranked_candidates = sorted(all_candidates.values(), key=lambda item: item["score"], reverse=True)
        best = dict(ranked_candidates[0])
        score_gap = float(best["score"] - ranked_candidates[1]["score"]) if len(ranked_candidates) > 1 else float(best["score"])
        best["score_gap"] = score_gap
        best["candidates"] = ranked_candidates[:10]
        best["level_candidates"] = final_candidates
        return best

    def _select_verified_fine_match(self, fine_match, *, site_memory=None):
        if fine_match is None:
            return None, None
        candidates = [fine_match]
        for candidate in fine_match.get("candidates", []):
            if (
                abs(float(candidate.get("x", 0.0)) - float(fine_match.get("x", 0.0))) < 1e-6
                and abs(float(candidate.get("y", 0.0)) - float(fine_match.get("y", 0.0))) < 1e-6
            ):
                continue
            candidates.append(candidate)

        best_verification = None
        for index, candidate in enumerate(candidates[:5], start=1):
            verification = self._verify_relocation(
                float(candidate["x"]),
                float(candidate["y"]),
                site_memory=site_memory,
            )
            if index > 1:
                self.log(
                    "Fine candidate re-check "
                    f"{index}/5: X={float(candidate['x']):.1f} um, "
                    f"Y={float(candidate['y']):.1f} um, "
                    f"score={float(candidate['score']):.3f}, "
                    f"verified={verification.get('verified', False)}"
                )
            if best_verification is None or verification.get("reference_score", 0.0) > best_verification.get("reference_score", -1.0):
                best_verification = verification
            if verification.get("verified", False):
                selected = dict(candidate)
                selected["verification_rank"] = index
                return selected, verification
        return fine_match, best_verification

    def _estimate_local_match_gap(self, top_left_x_um, top_left_y_um):
        reference_template = self.state.ref_template
        if reference_template is None:
            return 0.0
        if self._camera_frame_only_relocation_enabled():
            current_view = self._get_current_relocation_view(prefer_matching=False)
            return float(max(self._score_matching_views(reference_template, current_view), 0.0))
        center_view = self._capture_live_camera_reference_view(
            float(top_left_x_um),
            float(top_left_y_um),
            float(self.state.fov_width),
            float(self.state.fov_height),
            zoom_level=float(self.state.current_zoom_level),
        )
        center_score = self._score_matching_views(reference_template, center_view)
        offset_step_x = max(5.0, float(self.state.fov_width) * 0.10)
        offset_step_y = max(5.0, float(self.state.fov_height) * 0.10)
        neighbor_scores = []
        for dx_um, dy_um in (
            (-offset_step_x, 0.0),
            (offset_step_x, 0.0),
            (0.0, -offset_step_y),
            (0.0, offset_step_y),
        ):
            nx, ny = self._clamp_to_stage_margin(
                float(top_left_x_um) + float(dx_um),
                float(top_left_y_um) + float(dy_um),
            )
            neighbor_view = self._capture_live_camera_reference_view(
                float(nx),
                float(ny),
                float(self.state.fov_width),
                float(self.state.fov_height),
                zoom_level=float(self.state.current_zoom_level),
            )
            neighbor_scores.append(self._score_matching_views(reference_template, neighbor_view))
        best_neighbor = max(neighbor_scores) if neighbor_scores else -1.0
        return float(max(center_score - best_neighbor, 0.0))

    def _build_camera_overview(
        self,
        *,
        zoom_level,
        center_x_um,
        center_y_um,
        camera_resolution=None,
        max_dim=512,
    ):
        fov_width_um, fov_height_um = self.state.get_fov_for_zoom_level(zoom_level)
        max_x = max(float(self.state.width_um) - float(fov_width_um), 0.0)
        max_y = max(float(self.state.height_um) - float(fov_height_um), 0.0)
        top_left_x_um = float(np.clip(center_x_um - float(fov_width_um) * 0.5, 0.0, max_x))
        top_left_y_um = float(np.clip(center_y_um - float(fov_height_um) * 0.5, 0.0, max_y))
        disable_blur = bool(
            getattr(self.state, "reference_lowmag_disable_blur", False)
            and np.isclose(float(zoom_level), float(min(self.state.zoom_levels)))
        )
        view = self._capture_matching_template(
            top_left_x_um,
            top_left_y_um,
            fov_width_um,
            fov_height_um,
            zoom_level=zoom_level,
            disable_blur=disable_blur,
        )
        if view is None or view.size == 0:
            return None
        scale_x_um_per_px = float(fov_width_um) / float(max(view.shape[1], 1))
        scale_y_um_per_px = float(fov_height_um) / float(max(view.shape[0], 1))
        overview = build_overview_from_view(
            view,
            scale_x_um_per_px=scale_x_um_per_px,
            scale_y_um_per_px=scale_y_um_per_px,
            max_dim=max_dim,
        )
        if overview is None:
            return None
        overview["top_left_x_um"] = top_left_x_um
        overview["top_left_y_um"] = top_left_y_um
        overview["center_x_um"] = float(top_left_x_um + float(fov_width_um) * 0.5)
        overview["center_y_um"] = float(top_left_y_um + float(fov_height_um) * 0.5)
        overview["fov_width_um"] = float(fov_width_um)
        overview["fov_height_um"] = float(fov_height_um)
        overview["zoom_level"] = float(zoom_level)
        overview["render_mode"] = "matching_no_blur" if disable_blur else "matching"
        return overview

    def _capture_reference_lowmag_landmark_map(self):
        coarse_zoom_level = float(min(self.state.zoom_levels))
        tip_x_um = float(getattr(self.state, "probe_tip_x", self.state.x + self.state.fov_width / 2.0))
        tip_y_um = float(getattr(self.state, "probe_tip_y", self.state.y + self.state.fov_height / 2.0))
        coarse_fov_width_um, coarse_fov_height_um = self.state.get_fov_for_zoom_level(coarse_zoom_level)
        step_fraction = float(self.state.reference_lowmag_capture_step_fraction)
        step_x_um = float(coarse_fov_width_um) * step_fraction
        step_y_um = float(coarse_fov_height_um) * step_fraction
        capture_positions = [
            (0.0, 0.0),
            (-step_x_um, 0.0),
            (step_x_um, 0.0),
            (0.0, -step_y_um),
            (0.0, step_y_um),
        ][: max(int(self.state.reference_lowmag_capture_positions), 1)]

        overviews = []
        landmark_sets = []
        for index, (offset_x_um, offset_y_um) in enumerate(capture_positions, start=1):
            overview = self._build_camera_overview(
                zoom_level=coarse_zoom_level,
                center_x_um=tip_x_um + float(offset_x_um),
                center_y_um=tip_y_um + float(offset_y_um),
            )
            if overview is None:
                continue
            landmarks = extract_landmarks(
                overview["image"],
                base_x_um=float(overview.get("top_left_x_um", 0.0)),
                base_y_um=float(overview.get("top_left_y_um", 0.0)),
                scale_x_um_per_px=overview["scale_x_um_per_px"],
                scale_y_um_per_px=overview["scale_y_um_per_px"],
                origin_x_um=float(self.state.origin_x) if getattr(self.state, "origin_defined", False) else None,
                origin_y_um=float(self.state.origin_y) if getattr(self.state, "origin_defined", False) else None,
                patch_half=18,
                max_landmarks=6,
                min_distance_px=18,
            )
            annotate_landmarks_with_tip_geometry(landmarks, tip_x_um=tip_x_um, tip_y_um=tip_y_um)
            for landmark in landmarks:
                landmark["capture_frame_index"] = int(index)
                landmark["capture_offset_x_um"] = float(offset_x_um)
                landmark["capture_offset_y_um"] = float(offset_y_um)
            overviews.append(overview)
            landmark_sets.append(landmarks)

        primary_overview = overviews[0] if overviews else None
        min_distance_um = max(
            80.0,
            float(coarse_fov_width_um) / 6.0,
            float(coarse_fov_height_um) / 6.0,
        )
        merged_landmarks = merge_landmark_sets(
            landmark_sets,
            max_landmarks=int(self.state.reference_lowmag_max_landmarks),
            min_distance_um=min_distance_um,
        )
        capture_report = {
            "frame_count": len(overviews),
            "raw_landmark_count": int(sum(len(items) for items in landmark_sets)),
            "merged_landmark_count": int(len(merged_landmarks)),
            "coarse_zoom_level": coarse_zoom_level,
            "step_x_um": float(step_x_um),
            "step_y_um": float(step_y_um),
        }
        return primary_overview, merged_landmarks, capture_report

    def _capture_reference_fine_bundle(self):
        tip_x_um = float(getattr(self.state, "probe_tip_x", self.state.x + self.state.fov_width / 2.0))
        tip_y_um = float(getattr(self.state, "probe_tip_y", self.state.y + self.state.fov_height / 2.0))
        target_zoom = max(float(self.state.current_zoom_level), float(self.state.reference_auto_fine_zoom_level))
        zoom_levels = np.asarray(self.state.zoom_levels, dtype=float)
        fine_zoom_level = float(zoom_levels[np.argmin(np.abs(zoom_levels - target_zoom))])
        fine_fov_width_um, fine_fov_height_um = self.state.get_fov_for_zoom_level(fine_zoom_level)
        max_x = max(float(self.state.width_um) - float(fine_fov_width_um), 0.0)
        max_y = max(float(self.state.height_um) - float(fine_fov_height_um), 0.0)
        fine_top_left_x = float(np.clip(tip_x_um - float(fine_fov_width_um) * 0.5, 0.0, max_x))
        fine_top_left_y = float(np.clip(tip_y_um - float(fine_fov_height_um) * 0.5, 0.0, max_y))
        fine_reference_template = self._capture_matching_template(
            fine_top_left_x,
            fine_top_left_y,
            fine_fov_width_um,
            fine_fov_height_um,
            zoom_level=fine_zoom_level,
        )
        fine_live_camera_view = self._capture_live_camera_reference_view(
            fine_top_left_x,
            fine_top_left_y,
            fine_fov_width_um,
            fine_fov_height_um,
            zoom_level=fine_zoom_level,
            camera_resolution=self.state.camera_resolution,
        )
        fine_highmag_landmarks = extract_landmarks(
            fine_reference_template,
            base_x_um=fine_top_left_x,
            base_y_um=fine_top_left_y,
            origin_x_um=float(self.state.origin_x) if getattr(self.state, "origin_defined", False) else None,
            origin_y_um=float(self.state.origin_y) if getattr(self.state, "origin_defined", False) else None,
            patch_half=24,
            max_landmarks=6,
            min_distance_px=22,
        )
        annotate_landmarks_with_tip_geometry(fine_highmag_landmarks, tip_x_um=tip_x_um, tip_y_um=tip_y_um)
        return {
            "zoom_level": fine_zoom_level,
            "top_left_x_um": fine_top_left_x,
            "top_left_y_um": fine_top_left_y,
            "reference_template": fine_reference_template,
            "live_camera_view": fine_live_camera_view,
            "highmag_landmarks": fine_highmag_landmarks,
        }

    def _build_lowmag_guidance_report(self, overview, *, site_memory=None):
        site_memory = self.state.site_memory if site_memory is None else site_memory
        site_memory = site_memory or {}
        lowmag_landmarks = site_memory.get("lowmag_landmarks") or []
        if overview is None or not lowmag_landmarks:
            self.state.lowmag_guidance_report = None
            return None
        consensus = estimate_landmark_consensus(
            lowmag_landmarks,
            overview["image"],
            search_origin_x_um=float(overview.get("top_left_x_um", 0.0)),
            search_origin_y_um=float(overview.get("top_left_y_um", 0.0)),
            scale_x_um_per_px=overview["scale_x_um_per_px"],
            scale_y_um_per_px=overview["scale_y_um_per_px"],
            min_score=max(0.30, float(self.state.relocation_min_match_score) - 0.10),
            min_gap=max(0.01, float(self.state.relocation_min_score_gap) - 0.01),
            max_residual_um=120.0,
        )
        if consensus is None:
            self.state.lowmag_guidance_report = None
            return None

        tip_x_um = float(self.state.probe_tip_x)
        tip_y_um = float(self.state.probe_tip_y)
        matches = []
        distance_errors = []
        angle_errors = []
        dx_errors = []
        dy_errors = []
        for match in consensus.get("matches", []):
            guided_tip_x = None
            guided_tip_y = None
            current_tip_dx = None
            current_tip_dy = None
            current_tip_distance = None
            current_tip_angle = None
            distance_error = None
            angle_error = None
            dx_error = None
            dy_error = None
            ref_tip_dx = match.get("reference_tip_dx_um")
            ref_tip_dy = match.get("reference_tip_dy_um")
            ref_tip_distance = match.get("reference_tip_distance_um")
            ref_tip_angle = match.get("reference_tip_angle_deg")
            if ref_tip_dx is not None and ref_tip_dy is not None:
                guided_tip_x = float(match["abs_x_um"] - float(ref_tip_dx))
                guided_tip_y = float(match["abs_y_um"] - float(ref_tip_dy))
                dx_error = float(tip_x_um - guided_tip_x)
                dy_error = float(tip_y_um - guided_tip_y)
                dx_errors.append(dx_error)
                dy_errors.append(dy_error)
                current_tip_dx = float(match["abs_x_um"] - tip_x_um)
                current_tip_dy = float(match["abs_y_um"] - tip_y_um)
                current_tip_distance = float(np.hypot(current_tip_dx, current_tip_dy))
                current_tip_angle = float(np.degrees(np.arctan2(current_tip_dy, current_tip_dx)))
                if ref_tip_distance is not None:
                    distance_error = float(abs(current_tip_distance - float(ref_tip_distance)))
                    distance_errors.append(distance_error)
                if ref_tip_angle is not None:
                    angle_error = float(abs(((current_tip_angle - float(ref_tip_angle) + 180.0) % 360.0) - 180.0))
                    angle_errors.append(angle_error)
            enriched = dict(match)
            enriched["guided_tip_x_um"] = guided_tip_x
            enriched["guided_tip_y_um"] = guided_tip_y
            enriched["current_tip_dx_um"] = current_tip_dx
            enriched["current_tip_dy_um"] = current_tip_dy
            enriched["current_tip_distance_um"] = current_tip_distance
            enriched["current_tip_angle_deg"] = current_tip_angle
            enriched["distance_error_um"] = distance_error
            enriched["angle_error_deg"] = angle_error
            enriched["tip_dx_error_um"] = dx_error
            enriched["tip_dy_error_um"] = dy_error
            matches.append(enriched)

        report = {
            "zoom_level": float(overview.get("zoom_level", self.state.current_zoom_level)),
            "support_count": int(consensus.get("support_count", 0)),
            "confidence": float(consensus.get("confidence", 0.0)),
            "mean_score": float(consensus.get("mean_score", 0.0)),
            "mean_score_gap": float(consensus.get("mean_score_gap", 0.0)),
            "offset_x_um": float(consensus.get("offset_x_um", 0.0)),
            "offset_y_um": float(consensus.get("offset_y_um", 0.0)),
            "tip_x_um": tip_x_um,
            "tip_y_um": tip_y_um,
            "matches": matches,
            "mean_distance_error_um": None if not distance_errors else float(np.mean(distance_errors)),
            "mean_angle_error_deg": None if not angle_errors else float(np.mean(angle_errors)),
            "mean_tip_dx_error_um": None if not dx_errors else float(np.mean(dx_errors)),
            "mean_tip_dy_error_um": None if not dy_errors else float(np.mean(dy_errors)),
        }
        guided_tip_xs = [float(item["guided_tip_x_um"]) for item in matches if item.get("guided_tip_x_um") is not None]
        guided_tip_ys = [float(item["guided_tip_y_um"]) for item in matches if item.get("guided_tip_y_um") is not None]
        if guided_tip_xs and guided_tip_ys:
            estimated_tip_x = float(consensus.get("estimated_tip_x_um", np.mean(guided_tip_xs)))
            estimated_tip_y = float(consensus.get("estimated_tip_y_um", np.mean(guided_tip_ys)))
            report["estimated_tip_x_um"] = estimated_tip_x
            report["estimated_tip_y_um"] = estimated_tip_y
            report["estimated_tip_dx_um"] = float(estimated_tip_x - tip_x_um)
            report["estimated_tip_dy_um"] = float(estimated_tip_y - tip_y_um)
            report["estimated_tip_distance_um"] = float(np.hypot(report["estimated_tip_dx_um"], report["estimated_tip_dy_um"]))
        else:
            report["estimated_tip_x_um"] = None
            report["estimated_tip_y_um"] = None
            report["estimated_tip_dx_um"] = None
            report["estimated_tip_dy_um"] = None
            report["estimated_tip_distance_um"] = None
        geometry_report = analyze_landmark_geometry(
            lowmag_landmarks,
            overview["image"],
            view_origin_x_um=float(overview.get("top_left_x_um", 0.0)),
            view_origin_y_um=float(overview.get("top_left_y_um", 0.0)),
            tip_x_um=tip_x_um,
            tip_y_um=tip_y_um,
            min_score=max(0.30, float(self.state.relocation_min_match_score) - 0.10),
            min_gap=max(0.01, float(self.state.relocation_min_score_gap) - 0.01),
        )
        report["matched_count"] = int(geometry_report.get("matched_count", 0))
        report["geometry_confidence"] = float(geometry_report.get("geometry_confidence", 0.0))
        report["distance_confidence"] = float(geometry_report.get("distance_confidence", 0.0))
        report["mean_pair_error_um"] = geometry_report.get("mean_pair_error_um")
        report["geometry_mean_distance_error_um"] = geometry_report.get("mean_distance_error_um")
        report["geometry_mean_angle_error_deg"] = geometry_report.get("mean_angle_error_deg")
        if site_memory.get("lowmag_landmark_source") == "manual_authoritative":
            geometry_matches = list(geometry_report.get("matches", []))
            if len(geometry_matches) >= 2:
                src_points = np.asarray(
                    [
                        [float(item["reference_tip_dx_um"]), float(item["reference_tip_dy_um"])]
                        for item in geometry_matches
                        if item.get("reference_tip_dx_um") is not None and item.get("reference_tip_dy_um") is not None
                    ],
                    dtype=np.float32,
                )
                dst_points = np.asarray(
                    [
                        [float(item["center_x_um"]), float(item["center_y_um"])]
                        for item in geometry_matches
                        if item.get("reference_tip_dx_um") is not None and item.get("reference_tip_dy_um") is not None
                    ],
                    dtype=np.float32,
                )
                if len(src_points) >= 2 and len(dst_points) >= 2:
                    try:
                        matrix, _ = cv2.estimateAffinePartial2D(
                            src_points.reshape(-1, 1, 2),
                            dst_points.reshape(-1, 1, 2),
                            method=cv2.LMEDS,
                        )
                    except Exception:
                        matrix = None
                    if matrix is not None:
                        predicted_tip_x, predicted_tip_y = transform_point(matrix, 0.0, 0.0)
                        report["manual_constellation_matrix"] = matrix
                        report["manual_constellation_match_count"] = int(len(src_points))
                        report["manual_constellation_confidence"] = float(report["geometry_confidence"])
                        report["estimated_tip_x_um"] = float(predicted_tip_x)
                        report["estimated_tip_y_um"] = float(predicted_tip_y)
                        report["estimated_tip_dx_um"] = float(predicted_tip_x - tip_x_um)
                        report["estimated_tip_dy_um"] = float(predicted_tip_y - tip_y_um)
                        report["estimated_tip_distance_um"] = float(
                            np.hypot(report["estimated_tip_dx_um"], report["estimated_tip_dy_um"])
                        )
        self.state.lowmag_guidance_report = report
        return report

    def _clamp_to_stage_margin(self, x, y):
        min_x = -self.state.stage_margin_um
        min_y = -self.state.stage_margin_um
        max_x = self.state.width_um + self.state.stage_margin_um - self.state.fov_width
        max_y = self.state.height_um + self.state.stage_margin_um - self.state.fov_height
        return float(np.clip(x, min_x, max_x)), float(np.clip(y, min_y, max_y))

    def move_to_clicked_point(self, event):
        if getattr(self.state, "detection_roi_draw_mode", False) or getattr(self.state, "detection_roi_drag_active", False):
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if self.handle_viewport_arrow_click(event):
            return

        if getattr(self.state, "manual_reference_landmark_mode", False):
            if getattr(event, "button", None) == 3:
                self._finish_manual_reference_landmarks()
                return
            if getattr(event, "button", None) != 1:
                return
            self._record_manual_reference_landmark(float(event.xdata), float(event.ydata))
            return

        if getattr(self.state, "manual_landmark_guidance_active", False):
            if getattr(event, "button", None) == 3:
                self._cancel_manual_landmark_guidance("Human-guided landmark correction cancelled.")
                return
            if getattr(event, "button", None) != 1:
                return
            self._record_manual_landmark_click(float(event.xdata), float(event.ydata))
            return

        # Page 2/3: click-to-move correction mode (AI relocation fallback)
        if self.state.ai_relocate_awaiting_click:
            if getattr(event, "button", None) == 3:
                # Right-click: cancel correction mode
                self.state.ai_relocate_awaiting_click = False
                self.state.ai_relocate_pending_target_x = None
                self.state.ai_relocate_pending_target_y = None
                self.log("Click-to-move correction cancelled.")
                return
            if getattr(event, "button", None) != 1:
                return
            clicked_x = float(event.xdata)
            clicked_y = float(event.ydata)
            # Target viewport top-left: center the view on clicked point
            target_tl_x = clicked_x - self.state.fov_width / 2.0
            target_tl_y = clicked_y - self.state.fov_height / 2.0
            target_tl_x, target_tl_y = self._clamp_to_stage_margin(target_tl_x, target_tl_y)
            self.state.ai_relocate_awaiting_click = False
            self.state.ai_relocate_pending_target_x = None
            self.state.ai_relocate_pending_target_y = None
            self._start_smooth_move(target_tl_x, target_tl_y)
            self.state.sample_removed = False
            self.log(
                f"📍 Manual correction: cantilever moved to clicked position "
                f"({clicked_x:.1f}, {clicked_y:.1f}) → viewport top-left=({target_tl_x:.1f}, {target_tl_y:.1f})"
            )
            return

        if getattr(event, "button", None) != 1:
            return
        if self.state.zooming:
            self.log("Wait for zoom to finish before selecting a new scan area.")
            return

        clicked_x = float(event.xdata)
        clicked_y = float(event.ydata)
        delta_x = clicked_x - float(self.state.probe_tip_x)
        delta_y = clicked_y - float(self.state.probe_tip_y)
        target_x = self.state.target_x + delta_x
        target_y = self.state.target_y + delta_y
        target_x, target_y = self._clamp_to_stage_margin(target_x, target_y)
        self.state.auto_scan_active = False

        if self.state.pi_mode:
            self._start_smooth_move(target_x, target_y)
        else:
            self.state.target_x = target_x
            self.state.target_y = target_y

        self.log(
            f"AOI selected at X={clicked_x:.1f} um, Y={clicked_y:.1f} um -> "
            f"moving cantilever tip by dX={delta_x:+.1f} um, dY={delta_y:+.1f} um"
        )

    def _cancel_manual_landmark_guidance(self, message=None):
        self.state.manual_landmark_guidance_active = False
        self.state.manual_landmark_clicked_points = []
        self.state.manual_landmark_estimate = None
        if message:
            self.log(message)

    def begin_manual_reference_landmarks(self, event):
        current_view = self.state.current_matching_view if self.state.current_matching_view is not None else self.state.current_fov_raw
        if current_view is None or current_view.size == 0:
            self.log("No current camera matching view available for manual landmark marking.")
            return
        self.state.manual_reference_landmark_mode = True
        self.state.manual_reference_landmarks = []
        self.log(
            "👉 Pre-remount landmark marking: click 2 to 6 trusted landmarks before saving the site.\n"
            "   These human-selected landmarks will be stored in the site memory.\n"
            "   Right-click to finish or cancel."
        )
        self.fig.canvas.draw_idle()

    def _extract_manual_reference_landmark(self, abs_x_um, abs_y_um):
        source_view = self.state.current_matching_view if self.state.current_matching_view is not None else self.state.current_fov_raw
        if source_view is None or source_view.size == 0:
            return None
        gray = to_grayscale_u8(source_view)
        if gray is None or gray.size == 0:
            return None
        view_h, view_w = gray.shape[:2]
        rel_x = (float(abs_x_um) - float(self.state.x)) / max(float(self.state.fov_width), 1e-6)
        rel_y = (float(abs_y_um) - float(self.state.y)) / max(float(self.state.fov_height), 1e-6)
        px = int(round(np.clip(rel_x * float(view_w), 0, max(view_w - 1, 0))))
        py = int(round(np.clip(rel_y * float(view_h), 0, max(view_h - 1, 0))))
        patch_half = 24
        x0 = max(0, px - patch_half)
        x1 = min(view_w, px + patch_half)
        y0 = max(0, py - patch_half)
        y1 = min(view_h, py + patch_half)
        patch = gray[y0:y1, x0:x1]
        if patch.shape[0] < 12 or patch.shape[1] < 12:
            return None
        tip_x = float(getattr(self.state, "probe_tip_x", self.state.x + self.state.fov_width / 2.0))
        tip_y = float(getattr(self.state, "probe_tip_y", self.state.y + self.state.fov_height / 2.0))
        dx = float(abs_x_um - tip_x)
        dy = float(abs_y_um - tip_y)
        return {
            "center_px": (int(px), int(py)),
            "abs_x_um": float(abs_x_um),
            "abs_y_um": float(abs_y_um),
            "view_local_x_um": float(abs_x_um - float(self.state.x)),
            "view_local_y_um": float(abs_y_um - float(self.state.y)),
            "relative_x_um": None if not self.state.origin_defined else float(abs_x_um - self.state.origin_x),
            "relative_y_um": None if not self.state.origin_defined else float(abs_y_um - self.state.origin_y),
            "score": float(np.std(patch.astype(np.float32))),
            "capture_zoom_level": float(self.state.current_zoom_level),
            "tip_dx_um": dx,
            "tip_dy_um": dy,
            "tip_distance_um": float(np.hypot(dx, dy)),
            "tip_angle_deg": float(np.degrees(np.arctan2(dy, dx))),
            "patch": patch.copy(),
            "manual": True,
        }

    def _record_manual_reference_landmark(self, abs_x_um, abs_y_um):
        landmarks = list(getattr(self.state, "manual_reference_landmarks", []))
        if len(landmarks) >= 6:
            self.log("Already stored 6 manual reference landmarks. Right-click to finish, or Save Region to keep them.")
            self.fig.canvas.draw_idle()
            return
        landmark = self._extract_manual_reference_landmark(abs_x_um, abs_y_um)
        if landmark is None:
            self.log("Selected point is too close to the view edge for a stable landmark patch. Click a more central landmark.")
            self.fig.canvas.draw_idle()
            return
        landmarks.append(landmark)
        self.state.manual_reference_landmarks = landmarks
        self.log(
            f"Stored manual reference landmark {len(landmarks)} at "
            f"viewport-local ({landmark['view_local_x_um']:.1f}, {landmark['view_local_y_um']:.1f}) um "
            f"| tip-relative dX={landmark['tip_dx_um']:+.1f} um, dY={landmark['tip_dy_um']:+.1f} um"
        )
        self.fig.canvas.draw_idle()

    def _finish_manual_reference_landmarks(self):
        count = len(getattr(self.state, "manual_reference_landmarks", []))
        self.state.manual_reference_landmark_mode = False
        if count == 0:
            self.log("Manual landmark marking ended with no saved points.")
        else:
            self.log(
                f"Manual landmark marking finished with {count} saved landmarks. "
                "Now click Save Region to store them into the site memory."
            )
        self.fig.canvas.draw_idle()

    def _enter_manual_landmark_guidance(self, target_x=None, target_y=None):
        site_memory = self.state.site_memory or {}
        if len(site_memory.get("highmag_landmarks") or []) < 2:
            self._enter_click_to_move_correction(target_x, target_y)
            return
        self.state.manual_landmark_guidance_active = True
        self.state.manual_landmark_clicked_points = []
        self.state.manual_landmark_estimate = None
        self.state.ai_relocate_awaiting_click = False
        self.state.ai_relocate_pending_target_x = None
        self.state.ai_relocate_pending_target_y = None
        self.log(
            "👉 Human-guided landmark mode: click 2 or 3 recognizable landmarks in the current camera view.\n"
            "   After two clicks, the system will estimate the saved site pose and guide the viewport.\n"
            "   Right-click to cancel and fall back to click-to-move."
        )

    def _estimate_manual_landmark_pose(self):
        site_memory = self.state.site_memory or {}
        landmarks = site_memory.get("highmag_landmarks") or []
        clicked_points = list(getattr(self.state, "manual_landmark_clicked_points", []))
        if len(landmarks) < 2 or len(clicked_points) < 2:
            return None

        clicked = np.asarray(clicked_points, dtype=np.float32)
        best = None
        for landmark_indexes in itertools.permutations(range(len(landmarks)), len(clicked_points)):
            saved = np.asarray(
                [
                    [float(landmarks[index]["abs_x_um"]), float(landmarks[index]["abs_y_um"])]
                    for index in landmark_indexes
                ],
                dtype=np.float32,
            )
            matrix, _ = cv2.estimateAffinePartial2D(
                saved.reshape(-1, 1, 2),
                clicked.reshape(-1, 1, 2),
                method=cv2.LMEDS,
            )
            if matrix is None:
                continue
            projected = cv2.transform(saved.reshape(1, -1, 2), matrix).reshape(-1, 2)
            residuals = np.linalg.norm(projected - clicked, axis=1)
            mean_residual = float(np.mean(residuals))
            predicted_x, predicted_y = transform_point(matrix, float(self.state.ref_x), float(self.state.ref_y))
            rotation_deg = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
            candidate = {
                "matrix": matrix,
                "rotation_deg": rotation_deg,
                "mean_residual_um": mean_residual,
                "clicked_count": int(len(clicked_points)),
                "matched_indexes": list(landmark_indexes),
                "predicted_x_um": float(predicted_x),
                "predicted_y_um": float(predicted_y),
            }
            if best is None or mean_residual < best["mean_residual_um"]:
                best = candidate
        return best

    def _record_manual_landmark_click(self, clicked_x, clicked_y):
        clicked_points = list(getattr(self.state, "manual_landmark_clicked_points", []))
        clicked_points.append((float(clicked_x), float(clicked_y)))
        self.state.manual_landmark_clicked_points = clicked_points
        self.log(
            f"Human-guided landmark click {len(clicked_points)} recorded at "
            f"({clicked_x:.1f}, {clicked_y:.1f})"
        )
        if len(clicked_points) < 2:
            self.log("Click one more landmark to estimate the saved site pose.")
            return

        estimate = self._estimate_manual_landmark_pose()
        if estimate is None:
            self.log("Human-guided landmark estimate failed. Click another landmark or right-click to cancel.")
            return

        self.state.manual_landmark_estimate = estimate
        predicted_x, predicted_y = self._clamp_to_stage_margin(
            estimate["predicted_x_um"],
            estimate["predicted_y_um"],
        )
        self.log(
            "Human-guided pose estimate: "
            f"predicted top-left=({predicted_x:.1f}, {predicted_y:.1f}), "
            f"rotation={estimate['rotation_deg']:+.2f} deg, "
            f"residual={estimate['mean_residual_um']:.1f} um"
        )

        fine_match = self._match_camera_template(
            self.state.ref_template,
            predicted_x,
            predicted_y,
            half_range_um=max(180.0, float(self.state.relocation_verify_half_range_um) * 2.0),
        )
        if fine_match is not None:
            fine_match, verification = self._select_verified_fine_match(
                fine_match,
                site_memory=self.state.site_memory,
            )
            if verification is None:
                verification = self._verify_relocation(
                    float(fine_match["x"]),
                    float(fine_match["y"]),
                    site_memory=self.state.site_memory,
                )
            if verification.get("verified", False):
                cmd_x, cmd_y = self._clamp_to_stage_margin(float(fine_match["x"]), float(fine_match["y"]))
                self._cancel_manual_landmark_guidance()
                self._start_smooth_move(cmd_x, cmd_y)
                self.state.sample_removed = False
                self.log(
                    "✅ Human-guided landmark recovery accepted: "
                    f"reference score={verification.get('reference_score', 0.0):.3f}, "
                    f"gap={verification.get('reference_score_gap', 0.0):.3f}, "
                    f"moved to ({cmd_x:.1f}, {cmd_y:.1f})"
                )
                return

        self._jump_view_to_target(predicted_x, predicted_y)
        if len(clicked_points) < 3:
            self.log(
                "Viewport guided near the predicted site, but automatic verification is not yet strong enough.\n"
                "Click one more landmark to refine the estimate, or right-click to cancel."
            )
            return

        self._cancel_manual_landmark_guidance()
        self._enter_click_to_move_correction(predicted_x, predicted_y)

    def _append_desired_history(self, axis, value):
        history_attr = f"ai_desired_history_{axis}"
        history = list(getattr(self.state, history_attr))
        history.append(float(value))
        setattr(self.state, history_attr, history[-2:])

    def _predict_compensated_axis(self, desired, axis):
        if not self.ai_mode or not self.ai_compensator:
            return float(desired)

        required_keys = {"model", "scaler_X", "scaler_y"}
        if not required_keys.issubset(self.ai_compensator):
            return float(desired)

        history = getattr(self.state, f"ai_desired_history_{axis}", [float(desired), float(desired)])
        prev_1 = float(history[-1])
        prev_2 = float(history[-2])
        velocity = float(desired - prev_1)
        direction = 0.0 if np.isclose(velocity, 0.0) else float(np.sign(velocity))

        features = np.array([[desired, prev_1, prev_2, direction, velocity]], dtype=float)
        scaler_X = self.ai_compensator["scaler_X"]
        scaler_y = self.ai_compensator["scaler_y"]
        model = self.ai_compensator["model"]

        features_scaled = scaler_X.transform(features)
        prediction_scaled = model.predict(features_scaled).reshape(-1, 1)
        command = float(scaler_y.inverse_transform(prediction_scaled).ravel()[0])

        self._append_desired_history(axis, desired)
        return command

    def load_sample_image(self, event):
        if not self._begin_action("load_sample_image", "Image loading is already running"):
            return
        root = tk.Tk()
        try:
            root.withdraw()
            image_path = filedialog.askopenfilename(
                title="Select Microscope Image",
                filetypes=[
                    ("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),
                    ("All files", "*.*"),
                ],
            )
            if not image_path:
                return

            use_scale_bar = messagebox.askyesno(
                "Scale Calibration",
                "Do you want to calibrate using the scale bar embedded in the image?\n\n"
                "Choose Yes to click the two ends of the original scale bar.\n"
                "Choose No to enter the full image width and height manually.",
            )

            if use_scale_bar:
                raw_image = plt.imread(image_path)
                if raw_image.ndim == 3:
                    raw_image = raw_image[..., 0]

                fig, ax = plt.subplots(figsize=(10, 7))
                ax.imshow(raw_image, cmap="gray")
                ax.set_title("Click the LEFT and RIGHT ends of the original scale bar, then close this window")
                points = plt.ginput(2, timeout=-1)
                plt.close(fig)

                if len(points) != 2:
                    self.log("Scale bar calibration cancelled")
                    return

                (x1, y1), (x2, y2) = points
                scale_length_px = float(np.hypot(x2 - x1, y2 - y1))
                scale_length_um = simpledialog.askfloat(
                    "Scale Bar Length",
                    "Enter the real scale bar length (um):",
                    initialvalue=500.0,
                    minvalue=0.001,
                )
                if scale_length_um is None:
                    return

                sample, width_um, height_um = load_real_sample_image_from_scale(
                    image_path,
                    scale_bar_length_um=scale_length_um,
                    scale_bar_length_px=scale_length_px,
                )
                self.log(
                    f"Scale calibration from image bar: {scale_length_um:.3f} um "
                    f"over {scale_length_px:.2f} px"
                )
            else:
                width_mm = simpledialog.askfloat("Sample Width", "Enter physical image width (mm):", initialvalue=2.0, minvalue=0.01)
                if width_mm is None:
                    return
                height_mm = simpledialog.askfloat("Sample Height", "Enter physical image height (mm):", initialvalue=2.0, minvalue=0.01)
                if height_mm is None:
                    return

                sample, width_um, height_um = load_real_sample_image(image_path, width_mm * 1000.0, height_mm * 1000.0)
                self.log(f"Manual calibration: {width_mm:.3f} mm x {height_mm:.3f} mm")

            self._load_sample_into_state(sample, width_um, height_um, "image", image_path)
            self.log(f"Loaded microscope image: {image_path}")
            self.log(f"Calibrated sample size: {self.state.width_um / 1000.0:.3f} mm x {self.state.height_um / 1000.0:.3f} mm")
            self.log("Internal scale: 1 pixel = 1 um after calibration resize")
        except Exception as e:
            self.log(f"Failed to load microscope image: {e}")
        finally:
            root.destroy()
            self._end_action("load_sample_image")

    def save_current_as_default(self, event=None):
        if not self.state.sample_path:
            self.log("Load an image first before saving it as the default.")
            return
        default_path = Path(self.state.sample_path)
        if not default_path.exists():
            self.log(f"Current image path no longer exists: {default_path}")
            return

        self.state.default_image_path = str(default_path)
        self.state.default_image_width_um = float(self.state.width_um)
        self.state.default_image_height_um = float(self.state.height_um)
        if self.persist_default_callback is not None:
            try:
                self.persist_default_callback(
                    {
                        "path": default_path,
                        "width_um": float(self.state.width_um),
                        "height_um": float(self.state.height_um),
                        "scale_um_per_px": float(self.state.default_scale_um_per_px),
                    }
                )
            except Exception as e:
                self.log(f"Failed to save default image setting: {e}")
                return
        self.log(
            f"Saved default image: {default_path} "
            f"with calibration {self.state.width_um / 1000.0:.3f} mm x {self.state.height_um / 1000.0:.3f} mm"
        )

    def _estimate_relocation_offset(self, reference_detections, current_detections):
        candidates = []
        for ref in reference_detections:
            for curr in current_detections:
                if curr["class_name"] != ref["class_name"]:
                    continue
                offset_x = curr["abs_coord"][0] - ref["abs_coord"][0]
                offset_y = curr["abs_coord"][1] - ref["abs_coord"][1]
                candidates.append(
                    {
                        "ref": ref,
                        "curr": curr,
                        "offset_x": offset_x,
                        "offset_y": offset_y,
                    }
                )

        if not candidates:
            return None

        offsets = np.array([[c["offset_x"], c["offset_y"]] for c in candidates], dtype=float)
        median_offset = np.median(offsets, axis=0)

        supporters = []
        for candidate in candidates:
            residual = float(
                np.hypot(
                    candidate["offset_x"] - median_offset[0],
                    candidate["offset_y"] - median_offset[1],
                )
            )
            if residual <= 75.0:
                candidate["residual"] = residual
                supporters.append(candidate)

        if not supporters:
            supporters = candidates
            for candidate in supporters:
                candidate["residual"] = float(
                    np.hypot(
                        candidate["offset_x"] - median_offset[0],
                        candidate["offset_y"] - median_offset[1],
                    )
                )

        offset_x = float(np.mean([c["offset_x"] for c in supporters]))
        offset_y = float(np.mean([c["offset_y"] for c in supporters]))
        best_match = min(supporters, key=lambda item: item["residual"])

        return {
            "offset_x": offset_x,
            "offset_y": offset_y,
            "best_match": best_match,
            "support_count": len(supporters),
        }

    def _persist_site_memory(self, site_memory):
        output_dir = persist_site_memory(
            site_memory,
            self._resolve_project_path("collected_data/site_memories"),
        )
        self.state.last_saved_site_dir = str(output_dir)
        self.state.saved_site_memories.append(str(output_dir))
        return output_dir

    def _activate_site_memory(self, site_memory, source_dir=None):
        self.state.site_memory = site_memory
        self.state.ref_template = site_memory.get("reference_template")
        if self.state.ref_template is None:
            self.state.ref_template = site_memory.get("live_camera_view")
        if self.state.ref_template is None:
            raise ValueError("Saved live camera frame is missing. Relocation now requires the saved panel snapshot.")
        self.state.ref_artefacts = list(site_memory.get("highmag_landmarks", []))
        reference_top_left = site_memory.get("reference_top_left") or {}
        self.state.ref_x = float(reference_top_left.get("x_um", 0.0))
        self.state.ref_y = float(reference_top_left.get("y_um", 0.0))
        origin_info = site_memory.get("origin")
        if origin_info:
            self.state.origin_label = str(origin_info.get("label", self.state.origin_label or "Origin"))
            self.state.origin_x = float(origin_info.get("x_um", self.state.origin_x))
            self.state.origin_y = float(origin_info.get("y_um", self.state.origin_y))
        self.state.origin_template = site_memory.get("origin_template")
        saved_zoom = site_memory.get("final_zoom_level", site_memory.get("zoom_level"))
        if saved_zoom is not None:
            self._begin_quantized_zoom(float(saved_zoom))
            self.log(f"Restored zoom level to {float(saved_zoom):.2f}x from site memory")
        if source_dir is not None:
            self.state.last_saved_site_dir = str(source_dir)
            if str(source_dir) not in self.state.saved_site_memories:
                self.state.saved_site_memories.append(str(source_dir))
        self.log("Saved viewport anchor restored for internal crop bookkeeping only; relocation uses tip-relative camera geometry.")
        self.log("Fine relocation reference = saved camera matching frame only.")

    def _try_load_latest_site_memory(self):
        sample_id = Path(self.state.sample_path).stem if self.state.sample_path else self.state.sample_source
        latest_dir = find_latest_site_memory(
            self._resolve_project_path("collected_data/site_memories"),
            sample_id,
        )
        if latest_dir is None:
            return False
        try:
            site_memory = load_site_memory(latest_dir)
        except Exception as e:
            self.log(f"Failed to load latest site memory from {latest_dir}: {e}")
            return False
        self._activate_site_memory(site_memory, source_dir=latest_dir)
        self.log(
            f"Loaded saved site memory: {site_memory.get('site_id', 'unknown')} "
            f"from {latest_dir}"
        )
        return True

    def _estimate_coarse_affine_transform(self):
        site_memory = self.state.site_memory or {}
        reference_overview = site_memory.get("overview") or {}
        ref_tpl = reference_overview.get("image")
        if ref_tpl is None:
            self.log("No saved low-mag camera overview available")
            return None

        ref_tl = site_memory.get("reference_top_left") or {}
        ref_x = float(ref_tl.get("x_um", self.state.ref_x))
        ref_y = float(ref_tl.get("y_um", self.state.ref_y))
        coarse_zoom = float(reference_overview.get("zoom_level", site_memory.get("coarse_zoom_level", min(self.state.zoom_levels))))
        current_overview = self._build_camera_overview(
            zoom_level=coarse_zoom,
            center_x_um=float(self.state.probe_tip_x),
            center_y_um=float(self.state.probe_tip_y),
        )
        if current_overview is None or current_overview.get("image") is None:
            self.log("Current low-mag camera overview could not be generated")
            return None
        search_image = current_overview["image"]
        tpl_h, tpl_w = ref_tpl.shape[:2]

        # -- ML 预估旋转, 缩小 NCC 搜索范围 --
        ml_angle = None
        if self.deep_regressor is not None:
            try:
                crop = search_image[:tpl_h, :tpl_w]
                if crop.shape == ref_tpl.shape:
                    result = self.deep_regressor.predict(ref_tpl, crop)
                    if result is not None:
                        ml_angle = float(result["angle_deg"])
                        self.log(f"[ML] rotation estimate: {ml_angle:+.2f} deg")
            except:
                pass
        use_ml_narrow = self.state.force_ml_mode and ml_angle is not None
        if use_ml_narrow:
            sweep = list(range(int(ml_angle)-4, int(ml_angle)+5, 2))
            if 0 not in sweep: sweep.append(0)
            angles_to_try = sorted(set(sweep))
            self.log(f"[ML mode ON] narrow sweep around {ml_angle:+.1f} deg: {angles_to_try}")
        else:
            angles_to_try = list(range(-10, 11, 2))
            if ml_angle is not None:
                self.log(f"[ML mode OFF] full sweep (ML says {ml_angle:+.1f} deg but force_ml_mode=False)")
        ncc_match = None
        best_angle = 0.0
        self.log(f"[NCC] trying angles: {angles_to_try}")

        for try_angle in angles_to_try:
            if abs(try_angle) > 0.5:
                rm = cv2.getRotationMatrix2D((tpl_w / 2, tpl_h / 2), -try_angle, 1.0)
                tpl = cv2.warpAffine(ref_tpl, rm, (tpl_w, tpl_h), borderMode=cv2.BORDER_REPLICATE)
            else:
                tpl = ref_tpl
            candidates = match_template_candidates(search_image, tpl, top_k=2)
            match = None
            if candidates:
                best = candidates[0]
                score_gap = float(best["score"] - candidates[1]["score"]) if len(candidates) > 1 else float(best["score"])
                match = {
                    "x": int(best["x"]),
                    "y": int(best["y"]),
                    "score": float(best["score"]),
                    "score_gap": score_gap,
                }
            if match is not None and (ncc_match is None or match["score"] > ncc_match["score"]):
                ncc_match = match
                best_angle = try_angle
                self.log(f"[NCC] angle={try_angle:+.1f} deg score={match['score']:.3f} at ({match['x']:.0f},{match['y']:.0f})")

        if ncc_match is None:
            self.log("[NCC] no match found in current low-mag camera POV")
            return None

        matched_top_left_x = float(current_overview["top_left_x_um"] + ncc_match["x"] * current_overview["scale_x_um_per_px"])
        matched_top_left_y = float(current_overview["top_left_y_um"] + ncc_match["y"] * current_overview["scale_y_um_per_px"])
        dx_ncc = matched_top_left_x - ref_x
        dy_ncc = matched_top_left_y - ref_y
        final_angle = best_angle
        self.log(f"[NCC] result: dX={dx_ncc:+.0f} dY={dy_ncc:+.0f} um angle={final_angle:+.2f} deg score={ncc_match['score']:.3f}")

        # -- Step 3: ML verification (optional) --
        ml_prediction = None
        if self.deep_regressor is not None:
            try:
                mx2, my2 = int(round(ncc_match["x"])), int(round(ncc_match["y"]))
                sh2, sw2 = search_image.shape[:2]
                cx2, cy2 = min(sw2, mx2 + tpl_w), min(sh2, my2 + tpl_h)
                if cx2 - mx2 >= tpl_w // 2 and cy2 - my2 >= tpl_h // 2:
                    vc = search_image[my2:cy2, mx2:cx2]
                    if vc.shape[0] != tpl_h or vc.shape[1] != tpl_w:
                        fv = int(np.median(search_image))
                        vc = cv2.copyMakeBorder(vc, 0, tpl_h - vc.shape[0], 0, tpl_w - vc.shape[1],
                                                borderType=cv2.BORDER_CONSTANT, value=fv)
                    mr = self.deep_regressor.predict(ref_tpl, vc)
                    if mr is not None:
                        ml_prediction = {"dx_um": mr["dx_px"], "dy_um": mr["dy_px"], "dtheta_deg": mr["angle_deg"]}
            except Exception as e:
                self.log(f"[ML verify] failed: {e}")

        # -- Build report --
        theta_rad = np.radians(final_angle)
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
        full_matrix = np.array([[cos_t, -sin_t, dx_ncc], [sin_t, cos_t, dy_ncc]], dtype=np.float32)
        affine = {"matrix": full_matrix.copy(), "rotation_deg": final_angle, "scale": 1.0,
                  "translation_px": (dx_ncc, dy_ncc), "match_count": 1, "inlier_count": 1,
                  "confidence": float(ncc_match["score"])}

        report = dict(affine)
        report["current_overview"] = current_overview
        report["full_matrix"] = full_matrix
        report["ml_remount_prediction"] = ml_prediction
        cur_img = current_overview.get("image")
        ref_img = reference_overview.get("image")
        report["retrieval_candidates"] = (
            retrieve_lowmag_candidates(self.lowmag_embedding_index, cur_img, top_k=3)
            if cur_img is not None else []
        )
        report["predicted_remount_transform"] = (
            predict_remount_transform(self.remount_transform_predictor, ref_img, cur_img)
            if ref_img is not None and cur_img is not None else None
        )
        self.state.last_affine_transform_report = report
        return report


    def _apply_simulated_sample_translation(self, shift_x_um, shift_y_um):
        self.state.surface_image = translate_image(self.state.surface_image, shift_x_um, shift_y_um)
        self.state.sample = self.state.surface_image
        self.state.surface_valid_mask = translate_image(
            self.state.surface_valid_mask.astype(np.uint8),
            shift_x_um,
            shift_y_um,
            border_value=0,
        ) > 0
        if self.artifact_layer is not None and hasattr(self.artifact_layer, "layer"):
            self.artifact_layer.layer = translate_image(
                self.artifact_layer.layer,
                shift_x_um,
                shift_y_um,
                border_value=0,
            )
        self.state.simulated_sample_shift_x_um += float(shift_x_um)
        self.state.simulated_sample_shift_y_um += float(shift_y_um)

    def _apply_simulated_sample_remount(self, shift_x_um, shift_y_um, rotation_deg):
        h, w = self.state.surface_image.shape[:2]

        # -- compute bounding box of rotated+translated content --
        center = (w / 2.0, h / 2.0)
        rot = cv2.getRotationMatrix2D(center, float(rotation_deg), 1.0)
        corners = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.float32)
        rotated = cv2.transform(corners, rot).reshape(-1, 2)
        rotated[:, 0] += float(shift_x_um)
        rotated[:, 1] += float(shift_y_um)

        min_x = int(np.floor(np.min(rotated[:, 0])))
        max_x = int(np.ceil(np.max(rotated[:, 0])))
        min_y = int(np.floor(np.min(rotated[:, 1])))
        max_y = int(np.ceil(np.max(rotated[:, 1])))

        new_w = max_x - min_x
        new_h = max_y - min_y

        # -- adjust matrix: shift origin by (-min_x, -min_y) --
        matrix = rot.copy()
        matrix[0, 2] += float(shift_x_um) - float(min_x)
        matrix[1, 2] += float(shift_y_um) - float(min_y)

        output_shape = (new_h, new_w)
        median_val = int(np.median(self.state.surface_image))

        self.state.surface_image = apply_affine(
            self.state.surface_image, matrix,
            output_shape=output_shape,
        )
        self.state.sample = self.state.surface_image

        self.state.surface_valid_mask = apply_affine(
            self.state.surface_valid_mask.astype(np.uint8) * 255,
            matrix,
            output_shape=output_shape,
            border_value=0,
        ) > 0

        if self.artifact_layer is not None and hasattr(self.artifact_layer, "layer"):
            self.artifact_layer.layer = apply_affine(
                self.artifact_layer.layer, matrix,
                output_shape=output_shape, border_value=0,
            )

        self.state.width_um = float(new_w)
        self.state.height_um = float(new_h)
        self.state.stage_margin_um = max(2000.0, float(max(new_w, new_h)) * 1.5)

        self.state.simulated_sample_shift_x_um += float(shift_x_um)
        self.state.simulated_sample_shift_y_um += float(shift_y_um)
        self.state.simulated_sample_rotation_deg += float(rotation_deg)

    def _extract_surface_crop(self, top_left_x, top_left_y, margin_um, surface_image=None):
        if self.state.ref_template is None:
            return None, 0, 0
        surface_image = self.state.surface_image if surface_image is None else surface_image
        if surface_image is None:
            return None, 0, 0
        template_h, template_w = self.state.ref_template.shape[:2]
        sample_h, sample_w = surface_image.shape[:2]
        x0 = max(0, int(round(top_left_x - margin_um)))
        y0 = max(0, int(round(top_left_y - margin_um)))
        x1 = min(sample_w, int(round(top_left_x + template_w + margin_um)))
        y1 = min(sample_h, int(round(top_left_y + template_h + margin_um)))
        if x1 <= x0 or y1 <= y0:
            return None, x0, y0
        return surface_image[y0:y1, x0:x1], x0, y0

    def _verify_relocation(self, top_left_x, top_left_y, surface_image=None, site_memory=None):
        site_memory = self.state.site_memory if site_memory is None else site_memory
        candidate_view = self._capture_live_camera_reference_view(
            float(top_left_x),
            float(top_left_y),
            float(self.state.fov_width),
            float(self.state.fov_height),
            zoom_level=float(self.state.current_zoom_level),
        )
        verify_score = self._score_matching_views(self.state.ref_template, candidate_view)
        verify_gap = self._estimate_local_match_gap(float(top_left_x), float(top_left_y))
        verify_match = {
            "score": float(max(verify_score, 0.0)),
            "score_gap": float(max(verify_gap, 0.0)),
            "x": float(top_left_x),
            "y": float(top_left_y),
        }

        landmark_consensus = None
        geometry_check = None
        site_memory = site_memory or {}
        if site_memory.get("highmag_landmarks"):
            crop = candidate_view
            crop_x0 = float(top_left_x)
            crop_y0 = float(top_left_y)
            if crop is not None and crop.size > 0:
                landmark_consensus = estimate_landmark_consensus(
                    site_memory.get("highmag_landmarks", []),
                    crop,
                    search_origin_x_um=float(crop_x0),
                    search_origin_y_um=float(crop_y0),
                    min_score=self.state.relocation_min_match_score,
                    min_gap=self.state.relocation_min_score_gap,
                    max_residual_um=50.0,
                )
                reference_tip_local = site_memory.get("reference_tip_local") or {}
                tip_x_um = None
                tip_y_um = None
                if (
                    reference_tip_local.get("x_um") is not None
                    and reference_tip_local.get("y_um") is not None
                ):
                    tip_x_um = float(crop_x0 + float(reference_tip_local["x_um"]))
                    tip_y_um = float(crop_y0 + float(reference_tip_local["y_um"]))
                geometry_check = analyze_landmark_geometry(
                    site_memory.get("highmag_landmarks", []),
                    crop,
                    view_origin_x_um=float(crop_x0),
                    view_origin_y_um=float(crop_y0),
                    tip_x_um=tip_x_um,
                    tip_y_um=tip_y_um,
                    min_score=self.state.relocation_min_match_score,
                    min_gap=self.state.relocation_min_score_gap,
                )

        match_ok = (
            verify_match["score"] >= self.state.relocation_min_match_score
            and verify_match.get("score_gap", 0.0) >= self.state.relocation_min_score_gap
        )
        landmark_ok = (
            landmark_consensus is None
            or (
                landmark_consensus.get("support_count", 0) >= self.state.relocation_min_landmark_support
                and landmark_consensus["confidence"] >= self.state.relocation_min_match_score
            )
        )
        geometry_ok = (
            geometry_check is None
            or (
                geometry_check.get("matched_count", 0) >= self.state.relocation_min_landmark_support
                and geometry_check.get("geometry_confidence", 0.0) >= self.state.relocation_min_match_score
            )
        )
        same_site_probability = score_same_site_probability(
            self.same_site_classifier,
            self.state.ref_template,
            candidate_view,
        )
        strong_match_override = False
        strong_match_reason = None
        if same_site_probability is not None and same_site_probability < 0.50:
            strong_reference_match = (
                verify_match["score"] >= 0.75
                and verify_match.get("score_gap", 0.0) >= max(self.state.relocation_min_score_gap, 0.05)
            )
            strong_landmark_consensus = (
                landmark_consensus is not None
                and landmark_consensus.get("support_count", 0) >= max(self.state.relocation_min_landmark_support, 4)
                and landmark_consensus.get("confidence", 0.0) >= 0.55
            )
            strong_geometry_check = (
                geometry_check is not None
                and geometry_check.get("matched_count", 0) >= max(self.state.relocation_min_landmark_support, 4)
                and geometry_check.get("geometry_confidence", 0.0) >= 0.55
            )
            if strong_reference_match and (strong_landmark_consensus or strong_geometry_check):
                strong_match_override = True
                strong_match_reason = "strong_template_plus_landmarks"
        same_site_ok = (
            same_site_probability is None
            or same_site_probability >= 0.50
            or strong_match_override
        )
        camera_only_template_override = (
            self._camera_frame_only_relocation_enabled()
            and verify_match["score"] >= 0.95
            and verify_match.get("score_gap", 0.0) >= max(self.state.relocation_min_score_gap, 0.20)
        )
        return {
            "verified": bool(
                camera_only_template_override
                or (match_ok and landmark_ok and geometry_ok and same_site_ok)
            ),
            "reference_score": float(verify_match["score"]),
            "reference_score_gap": float(verify_match.get("score_gap", 0.0)),
            "reference_match_x_um": float(verify_match["x"]),
            "reference_match_y_um": float(verify_match["y"]),
            "affine_verify": None,
            "landmark_consensus": landmark_consensus,
            "geometry_check": geometry_check,
            "same_site_probability": same_site_probability,
            "same_site_override_used": bool(strong_match_override),
            "camera_only_template_override_used": bool(camera_only_template_override),
            "same_site_override_reason": strong_match_reason,
        }

    def set_step(self, step):
        self.state.current_step = step
        self.log(f"Step size set to {step} um")

    def slow_smooth_move(self, event):
        current = float(self.state.smooth_move_step)
        next_step = max(float(self.state.smooth_move_min_step), current / 2.0)
        if np.isclose(next_step, current):
            self.log(f"Smooth move speed already at minimum ({current:.1f} um/frame)")
            return
        self.state.smooth_move_step = next_step
        self.log(f"Smooth move speed set to {next_step:.1f} um/frame")

    def fast_smooth_move(self, event):
        current = float(self.state.smooth_move_step)
        next_step = min(float(self.state.smooth_move_max_step), current * 2.0)
        if np.isclose(next_step, current):
            self.log(f"Smooth move speed already at maximum ({current:.1f} um/frame)")
            return
        self.state.smooth_move_step = next_step
        self.log(f"Smooth move speed set to {next_step:.1f} um/frame")

    def move_up(self, event):
        new_target_x, new_target_y = self._clamp_to_stage_margin(self.state.target_x, self.state.target_y - self.state.current_step)
        if self.state.pi_mode:
            self._start_smooth_move(new_target_x, new_target_y)
        else:
            self.state.target_x = new_target_x
            self.state.target_y = new_target_y
        self.log(f"Move up {self.state.current_step} um -> target center Y={new_target_y + self.state.fov_height / 2.0:.1f}")

    def move_down(self, event):
        new_target_x, new_target_y = self._clamp_to_stage_margin(self.state.target_x, self.state.target_y + self.state.current_step)
        if self.state.pi_mode:
            self._start_smooth_move(new_target_x, new_target_y)
        else:
            self.state.target_x = new_target_x
            self.state.target_y = new_target_y
        self.log(f"Move down {self.state.current_step} um -> target center Y={new_target_y + self.state.fov_height / 2.0:.1f}")

    def move_left(self, event):
        new_target_x, new_target_y = self._clamp_to_stage_margin(self.state.target_x - self.state.current_step, self.state.target_y)
        if self.state.pi_mode:
            self._start_smooth_move(new_target_x, new_target_y)
        else:
            self.state.target_x = new_target_x
            self.state.target_y = new_target_y
        self.log(f"Move left {self.state.current_step} um -> target center X={new_target_x + self.state.fov_width / 2.0:.1f}")

    def move_right(self, event):
        new_target_x, new_target_y = self._clamp_to_stage_margin(self.state.target_x + self.state.current_step, self.state.target_y)
        if self.state.pi_mode:
            self._start_smooth_move(new_target_x, new_target_y)
        else:
            self.state.target_x = new_target_x
            self.state.target_y = new_target_y
        self.log(f"Move right {self.state.current_step} um -> target center X={new_target_x + self.state.fov_width / 2.0:.1f}")

    def random_offset_move(self, event):
        if self.state.zooming:
            self.log("Wait for zoom to finish before applying a random cantilever move.")
            return
        self.state.auto_scan_active = False
        base_x = float(self.state.target_x)
        base_y = float(self.state.target_y)
        requested_dx = float(random.uniform(-200.0, 200.0))
        requested_dy = float(random.uniform(-200.0, 200.0))
        new_target_x, new_target_y = self._clamp_to_stage_margin(base_x + requested_dx, base_y + requested_dy)
        actual_dx = float(new_target_x - base_x)
        actual_dy = float(new_target_y - base_y)
        if self.state.pi_mode:
            self._start_smooth_move(new_target_x, new_target_y)
        else:
            self.state.target_x = new_target_x
            self.state.target_y = new_target_y
        self.log(
            f"Random cantilever move: dX={actual_dx:+.1f} um, dY={actual_dy:+.1f} um "
            "(range +/-200 um)"
        )

    def toggle_pause(self, event):
        self.state.paused = not self.state.paused
        self.buttons["pause_motion"].label.set_text("Motion: OFF" if self.state.paused else "Motion: ON")
        if not self.state.paused:
            self.state.target_x, self.state.target_y = self.state.x, self.state.y
        self.update_title()
        self.log(f"Motion pause: {'ON' if self.state.paused else 'OFF'}")

    def stop_motion(self, event):
        had_motion = (
            self.state.auto_scan_active
            or self.state.smooth_move_active
            or not np.isclose(float(self.state.target_x), float(self.state.x))
            or not np.isclose(float(self.state.target_y), float(self.state.y))
        )
        self._cancel_active_motion()
        self._refresh_current_view()
        self.update_title()
        if had_motion:
            self.log("Active motion cancelled; destination snapped to the current position.")
        else:
            self.log("No active motion to stop.")

    def jump_to_destination(self, event):
        if np.isclose(float(self.state.target_x), float(self.state.x)) and np.isclose(float(self.state.target_y), float(self.state.y)):
            self.log("Already at the current destination.")
            return
        delta_x, delta_y = self._jump_view_to_target()
        self.log(
            f"Jumped immediately to destination by dX={delta_x:+.1f} um, dY={delta_y:+.1f} um"
        )

    def go_now_relocation(self, event):
        pending_x = self.state.ai_relocate_pending_target_x
        pending_y = self.state.ai_relocate_pending_target_y
        if self.state.ai_relocate_awaiting_click and pending_x is not None and pending_y is not None:
            target_x, target_y = self._clamp_to_stage_margin(float(pending_x), float(pending_y))
            self.state.ai_relocate_awaiting_click = False
            self.state.ai_relocate_pending_target_x = None
            self.state.ai_relocate_pending_target_y = None
            delta_x, delta_y = self._jump_view_to_target(target_x, target_y)
            self.state.sample_removed = False
            self.log(
                "AI relocation accepted immediately: "
                f"dX={delta_x:+.1f} um, dY={delta_y:+.1f} um"
            )
            return
        self.jump_to_destination(event)

    def toggle_pi(self, event):
        self.state.pi_mode = not self.state.pi_mode
        # Switching compensation mode should not itself trigger a motion jump.
        # Snap the active target to the current position so any new motion only
        # begins after the next explicit user command.
        self.state.smooth_move_active = False
        self.state.smooth_move_target_x = float(self.state.x)
        self.state.smooth_move_target_y = float(self.state.y)
        self.state.target_x = float(self.state.x)
        self.state.target_y = float(self.state.y)
        self.data.clear()
        self.stage.clear_history()
        if self.state.pi_mode:
            self.stage.reset(self.state.x, self.state.y)
            self.stage.cmd_x = float(self.state.x)
            self.stage.cmd_y = float(self.state.y)
            self.buttons["pi"].label.set_text("PI Compensation: ON")
            self.log("PI mode activated from current position; previous motion history cleared")
        else:
            self.buttons["pi"].label.set_text("PI Compensation Mode")
            self.log("PI mode deactivated; previous motion history cleared")
        self.update_title()

    def zoom_in(self, event):
        current_index = self._get_zoom_level_index(self.state.current_zoom_level)
        levels = list(self.state.zoom_levels)
        self._begin_quantized_zoom(levels[min(current_index + 1, len(levels) - 1)])

    def zoom_out(self, event):
        current_index = self._get_zoom_level_index(self.state.current_zoom_level)
        levels = list(self.state.zoom_levels)
        self._begin_quantized_zoom(levels[max(current_index - 1, 0)])

    def move_z_up(self, event):
        self.state.z_stage_position_um = self._clamp_z_stage(self.state.z_stage_position_um + self.state.z_stage_step_um)
        self._refresh_current_view()
        self.log(
            f"Sample Z stage moved to {self.state.z_stage_position_um:+.1f} um "
            f"(probe gap {self.state.get_probe_sample_gap_um():+.1f} um, focus offset {self.state.get_focus_offset_um():+.1f} um)"
        )

    def move_z_down(self, event):
        self.state.z_stage_position_um = self._clamp_z_stage(self.state.z_stage_position_um - self.state.z_stage_step_um)
        self._refresh_current_view()
        self.log(
            f"Sample Z stage moved to {self.state.z_stage_position_um:+.1f} um "
            f"(probe gap {self.state.get_probe_sample_gap_um():+.1f} um, focus offset {self.state.get_focus_offset_um():+.1f} um)"
        )

    def reset_focus(self, event):
        self.state.z_stage_position_um = float(
            self.state.get_effective_camera_stage_position_um() - self.state.focus_z_um
        )
        self._refresh_current_view()
        self.log(
            f"Sample Z stage returned to best focus at {self.state.z_stage_position_um:.1f} um "
            f"for camera/cantilever stage {self.state.get_effective_camera_stage_position_um():.1f} um"
        )

    def _get_active_dof_value(self):
        if self.state.manual_dof_camera_um is not None:
            return float(self.state.manual_dof_camera_um)
        return float(self.state.last_dof_camera_um)

    def increase_dof(self, event):
        next_dof = max(0.1, self._get_active_dof_value() + float(self.state.dof_step_um))
        self.state.manual_dof_camera_um = next_dof
        self._refresh_current_view()
        self.log(f"Manual DOF set to {next_dof:.2f} um")

    def decrease_dof(self, event):
        next_dof = max(0.1, self._get_active_dof_value() - float(self.state.dof_step_um))
        self.state.manual_dof_camera_um = next_dof
        self._refresh_current_view()
        self.log(f"Manual DOF set to {next_dof:.2f} um")

    def reset_dof_auto(self, event):
        self.state.manual_dof_camera_um = None
        self._refresh_current_view()
        self.log("DOF control returned to automatic optics mode")

    def show_tip_coord(self, event):
        tip_x, tip_y = self.get_tip()
        if self.state.origin_defined:
            self.log(
                f"Tip position: X={tip_x:.2f} um, Y={tip_y:.2f} um "
                f"(relative to {self.state.origin_label}: "
                f"dX={tip_x - self.state.origin_x:+.2f} um, dY={tip_y - self.state.origin_y:+.2f} um)"
            )
        else:
            self.log(f"Tip position: X={tip_x:.2f} um, Y={tip_y:.2f} um")

    def cycle_scale_bar_length(self, event):
        choices = list(self.SCALE_BAR_CHOICES_UM)
        try:
            current_index = choices.index(float(self.state.scale_bar_total_um))
        except ValueError:
            current_index = 0
        next_value = choices[(current_index + 1) % len(choices)]
        self.state.scale_bar_total_um = float(next_value)
        if "scale_bar" in self.buttons:
            self.buttons["scale_bar"].label.set_text(f"Scale Bar: {int(next_value)} um")
        self.fig.canvas.draw_idle()
        self.log(f"Viewport scale bar length set to {int(next_value)} um")

    def auto_origin_unsupervised(self, event):
        current_view = self.state.current_camera_view if self.state.current_camera_view is not None else self.state.current_fov_raw
        if current_view is None or current_view.size == 0:
            self.log("No current viewport image available for unsupervised origin detection.")
            return

        landmarks = extract_landmarks(
            current_view,
            base_x_um=float(self.state.x),
            base_y_um=float(self.state.y),
            patch_half=max(18, int(self.state.origin_template_half_size // 2)),
            max_landmarks=5,
            min_distance_px=20,
        )
        if not landmarks:
            self.log("Unsupervised origin detection could not find a distinctive pattern in the current viewport.")
            return

        best_landmark = max(landmarks, key=lambda item: item["score"])
        abs_x = float(best_landmark["abs_x_um"])
        abs_y = float(best_landmark["abs_y_um"])
        self.set_origin(abs_x, abs_y, label="Auto Origin")
        self.log(
            f"Unsupervised origin selected at X={abs_x:.1f} um, Y={abs_y:.1f} um "
            f"(distinctiveness score {best_landmark['score']:.3f}, "
            f"{len(landmarks)} candidate landmarks stored in the current view)"
        )

    def ml_find_origin(self, event):
        if not self._begin_action("ml_find_origin", "ML origin search is already running"):
            return
        try:
            if self.state.origin_template is None:
                self.log("Set an origin in the viewport first so the supervised ML search has a labeled pattern.")
                return

            template = self.state.origin_template
            surface = self.state.surface_image
            if template.ndim == 3:
                template = template[..., 0]
            if surface.ndim == 3:
                surface = surface[..., 0]

            candidates = match_template_candidates(surface, template, top_k=3)
            if not candidates:
                self.log("ML origin search could not find the labeled pattern.")
                return
            best = candidates[0]
            score_gap = float(best["score"] - candidates[1]["score"]) if len(candidates) > 1 else float(best["score"])
            if best["score"] < self.state.relocation_min_match_score or score_gap < self.state.relocation_min_score_gap:
                self.log(
                    f"ML origin search confidence too low: score={best['score']:.3f}, gap={score_gap:.3f}"
                )
                return

            tpl_h, tpl_w = template.shape[:2]
            origin_x = float(best["x"] + tpl_w / 2.0)
            origin_y = float(best["y"] + tpl_h / 2.0)
            self.state.origin_x = origin_x
            self.state.origin_y = origin_y
            self.state.origin_defined = True
            center_target_x, center_target_y = self._clamp_to_stage_margin(
                origin_x - self.state.fov_width / 2.0,
                origin_y - self.state.fov_height / 2.0,
            )
            if self.state.pi_mode:
                self._start_smooth_move(center_target_x, center_target_y)
            else:
                self.state.target_x = center_target_x
                self.state.target_y = center_target_y
            if self.status_callback is not None:
                self.status_callback()
            self.fig.canvas.draw_idle()
            self.log(
                f"ML origin recognized {self.state.origin_label} at "
                f"X={origin_x:.1f} um, Y={origin_y:.1f} um "
                f"(confidence {best['score']:.3f}, gap {score_gap:.3f})"
            )
        finally:
            self._end_action("ml_find_origin")

    def clear_trails(self, event):
        self.data.clear()
        self.stage.clear_history()
        self.log("Trails cleared")

    def start_auto_scan(self, event):
        if self.state.auto_scan_active:
            self.log("Auto scan is already running")
            return
        self.state.auto_scan_start_x = self.state.target_x
        self.state.auto_scan_end_x = self.state.target_x + 1000
        self.state.auto_scan_step = 0
        self.state.auto_scan_direction = 1
        self.state.auto_scan_active = True
        self.log(f"Auto scan: {self.state.auto_scan_start_x:.0f} -> {self.state.auto_scan_end_x:.0f}")

    def show_hysteresis_curve(self, event):
        if not self._begin_action("show_hysteresis", "Hysteresis plot is already opening"):
            return
        if self.state.pi_mode and len(self.stage.history_cmd) > 0:
            try:
                self.stage.plot_hysteresis("Hysteresis (PI Model)")
            finally:
                self._end_action("show_hysteresis")
        else:
            self._end_action("show_hysteresis")
            self.log("Please enable PI mode and move the tip first")

    def start_data_collection(self, event):
        if not self._begin_action("data_collection", "Data collection is already running"):
            return
        self.log("=== Starting data collection ===")
        try:
            from data_collection import DataCollector

            collector = DataCollector(self.stage, self.state, self.get_tip)
            configs = [
                {"start_x": 200, "end_x": 600, "steps": 80, "speed_factor": 1, "label": "Range_400um_Slow"},
                {"start_x": 200, "end_x": 600, "steps": 80, "speed_factor": 2, "label": "Range_400um_Fast"},
                {"start_x": 200, "end_x": 1000, "steps": 120, "speed_factor": 1, "label": "Range_800um_Slow"},
                {"start_x": 200, "end_x": 1000, "steps": 120, "speed_factor": 2, "label": "Range_800um_Fast"},
                {"start_x": 200, "end_x": 1400, "steps": 160, "speed_factor": 1, "label": "Range_1200um_Slow"},
                {"start_x": 200, "end_x": 1400, "steps": 160, "speed_factor": 2, "label": "Range_1200um_Fast"},
            ]
            collector.collect_multi_configurations(configs, base_wait_time=0.05)
            filename = collector.save_all_to_csv()
            collector.plot_collected_data()
            self.log(f"Data collection completed. File saved: {filename}")
        except Exception as e:
            self.log(f"Data collection failed: {e}")
        finally:
            self._end_action("data_collection")

    def save_reference(self, event):
        """Save the current site memory, reference view, and multi-landmark relocation data."""
        if not self._begin_action("save_reference", "Reference capture is already running"):
            return
        try:
            if self.img is None:
                self.log("Cannot get current image")
                return

            live_camera_view = self._capture_display_panel_snapshot()
            if live_camera_view is None:
                live_camera_view = self.state.current_camera_view if self.state.current_camera_view is not None else self.state.current_fov_raw
            if live_camera_view is None:
                live_camera_view = self.img.get_array()
            reference_template = (
                self.state.current_matching_view
                if self.state.current_matching_view is not None
                else self._capture_matching_template(
                    float(self.state.x),
                    float(self.state.y),
                    float(self.state.fov_width),
                    float(self.state.fov_height),
                    zoom_level=float(self.state.current_zoom_level),
                )
            )
            if live_camera_view is None:
                self.log("Cannot save reference: live panel snapshot is unavailable.")
                return
            fine_bundle = self._capture_reference_fine_bundle()
            effective_reference_template = fine_bundle.get("reference_template")
            if effective_reference_template is None:
                effective_reference_template = reference_template
            effective_live_camera_view = fine_bundle.get("live_camera_view")
            if effective_live_camera_view is None:
                effective_live_camera_view = live_camera_view
            effective_highmag_landmarks = list(fine_bundle.get("highmag_landmarks") or [])
            self.state.ref_template = (
                effective_reference_template.copy()
                if effective_reference_template is not None
                else effective_live_camera_view.copy()
            )
            self.state.ref_artefacts = []
            self.state.ref_x = float(fine_bundle.get("top_left_x_um", self.state.x))
            self.state.ref_y = float(fine_bundle.get("top_left_y_um", self.state.y))
            self.state.ai_desired_history_x = [self.state.x, self.state.x]
            self.state.ai_desired_history_y = [self.state.y, self.state.y]
            overview, captured_lowmag_landmarks, lowmag_capture_report = self._capture_reference_lowmag_landmark_map()
            self.state.site_memory = build_site_memory(
                self.state,
                stage_history=self.stage.history_cmd,
                overview=overview,
                live_camera_view=effective_live_camera_view,
                reference_template=effective_reference_template,
                lowmag_landmarks=captured_lowmag_landmarks,
                highmag_landmarks=effective_highmag_landmarks,
                reference_top_left_override=(
                    float(fine_bundle.get("top_left_x_um", self.state.x)),
                    float(fine_bundle.get("top_left_y_um", self.state.y)),
                ),
                reference_zoom_level_override=float(fine_bundle.get("zoom_level", self.state.current_zoom_level)),
            )
            self.state.site_memory["reference_view_kind"] = "fine_live_camera_frame"
            self.state.site_memory["reference_view_zoom_level"] = float(
                fine_bundle.get("zoom_level", self.state.current_zoom_level)
            )
            manual_landmarks = list(getattr(self.state, "manual_reference_landmarks", []))
            coarse_zoom_level = float(min(self.state.zoom_levels))
            manual_lowmag_landmarks = [
                dict(item)
                for item in manual_landmarks
                if np.isclose(float(item.get("capture_zoom_level", self.state.current_zoom_level)), coarse_zoom_level)
            ]
            manual_highmag_landmarks = [
                dict(item)
                for item in manual_landmarks
                if not np.isclose(float(item.get("capture_zoom_level", self.state.current_zoom_level)), coarse_zoom_level)
            ]
            if manual_lowmag_landmarks:
                authoritative_lowmag = [dict(item) for item in manual_lowmag_landmarks]
                auto_lowmag = list(self.state.site_memory.get("lowmag_landmarks", []))
                min_lowmag_landmarks = int(
                    self.state.site_memory.get("lowmag_ready_min_landmarks", self.state.reference_lowmag_min_landmarks)
                )
                if len(authoritative_lowmag) < min_lowmag_landmarks:
                    merged_lowmag = merge_landmark_sets(
                        [
                            authoritative_lowmag,
                            auto_lowmag,
                        ],
                        max_landmarks=int(self.state.reference_lowmag_max_landmarks),
                        min_distance_um=max(80.0, float(self.state.get_fov_for_zoom_level(coarse_zoom_level)[0]) / 6.0),
                    )
                else:
                    merged_lowmag = authoritative_lowmag
                self.state.site_memory["lowmag_landmarks"] = merged_lowmag
                self.state.site_memory["lowmag_landmark_source"] = "manual_authoritative"
                self.state.site_memory["manual_lowmag_landmark_count"] = int(len(authoritative_lowmag))
                min_lowmag_landmarks = int(self.state.site_memory.get("lowmag_ready_min_landmarks", self.state.reference_lowmag_min_landmarks))
                self.state.site_memory["lowmag_ready"] = bool(
                    self.state.site_memory.get("overview") is not None
                    and len(merged_lowmag) >= min_lowmag_landmarks
                )
                self.log(
                    f"Using {len(manual_lowmag_landmarks)} manually marked low-mag landmarks from {coarse_zoom_level:.2f}x "
                    f"as the authoritative coarse landmark set."
                )
            if manual_highmag_landmarks:
                self.state.site_memory["highmag_landmarks"] = manual_highmag_landmarks
                self.log(
                    f"Using {len(manual_highmag_landmarks)} human-marked high-mag landmarks for this saved site memory."
                )
            self.log("Saved display snapshot from current panel buffer.")
            self.log("Fine relocation reference = saved camera matching frame only.")
            self.log(
                "Auto fine capture: "
                f"{fine_bundle.get('zoom_level', self.state.current_zoom_level):.2f}x at "
                f"X={float(fine_bundle.get('top_left_x_um', self.state.x)):.1f} um, "
                f"Y={float(fine_bundle.get('top_left_y_um', self.state.y)):.1f} um"
            )
            self.log(
                f"Saved low-mag landmark map with "
                f"{len(self.state.site_memory.get('lowmag_landmarks', []))} tip-referenced landmarks."
            )
            if lowmag_capture_report is not None:
                self.log(
                    "Low-mag capture sweep: "
                    f"frames={lowmag_capture_report['frame_count']}, "
                    f"raw landmarks={lowmag_capture_report['raw_landmark_count']}, "
                    f"merged={lowmag_capture_report['merged_landmark_count']}, "
                    f"step=({lowmag_capture_report['step_x_um']:.1f}, {lowmag_capture_report['step_y_um']:.1f}) um"
                )
            lowmag_count = len(self.state.site_memory.get("lowmag_landmarks", []))
            if lowmag_count < int(self.state.reference_lowmag_min_landmarks):
                self.log(
                    "Warning: this saved site memory is not ready for reliable automatic low-mag relocation. "
                    f"Only {lowmag_count} low-mag landmarks were captured; target is at least "
                    f"{int(self.state.reference_lowmag_min_landmarks)}."
                )
                self.log(
                    f"Zoom to {coarse_zoom_level:.2f}x and use 'Mark Reference Landmarks', or move to a richer surrounding area before saving again."
                )
            self.state.ref_artefacts = list(self.state.site_memory.get("highmag_landmarks", []))
            output_dir = self._persist_site_memory(self.state.site_memory)
            self.log(f"Reference position saved: ({self.state.ref_x:.1f}, {self.state.ref_y:.1f})")
            self.log(f"Saved live camera frame size: {self.state.ref_template.shape[1]} x {self.state.ref_template.shape[0]} px")
            self.log(
                "Structured site memory saved with "
                f"{len(self.state.site_memory.get('lowmag_landmarks', []))} low-mag landmarks and "
                f"{len(self.state.site_memory.get('highmag_landmarks', []))} high-mag landmarks"
            )
            if overview is not None:
                self.log(
                    "Saved low-mag reference from camera POV: "
                    f"{overview['image'].shape[1]} x {overview['image'].shape[0]} px at "
                    f"{overview['zoom_level']:.2f}x"
                )
            self.log(f"Site memory folder: {output_dir}")
        finally:
            self._end_action("save_reference")

    def research_patterns(self, event):
        site_memory = self.state.site_memory or {}
        current_view = self.state.current_matching_view if self.state.current_matching_view is not None else self.state.current_fov_raw
        if not site_memory.get("highmag_landmarks"):
            self.log("No saved site memory landmarks available. Save site memory first.")
            return
        if current_view is None or current_view.size == 0:
            self.log("No current camera view available for pattern re-search.")
            return
        tip_x, tip_y = self.get_tip()
        report = analyze_landmark_geometry(
            site_memory.get("highmag_landmarks", []),
            current_view,
            view_origin_x_um=float(self.state.x),
            view_origin_y_um=float(self.state.y),
            tip_x_um=float(tip_x),
            tip_y_um=float(tip_y),
            min_score=max(0.38, float(self.state.relocation_min_match_score) - 0.04),
            min_gap=max(0.01, float(self.state.relocation_min_score_gap) - 0.01),
        )
        self.state.hud_landmark_report = report
        self.state.hud_landmark_matches = list(report.get("matches", []))
        if self.status_callback is not None:
            self.status_callback()
        self.fig.canvas.draw_idle()
        self.log(
            "Pattern re-search: "
            f"matched={report.get('matched_count', 0)}, "
            f"pair_error={0.0 if report.get('mean_pair_error_um') is None else report['mean_pair_error_um']:.1f} um, "
            f"tip_error={0.0 if report.get('mean_distance_error_um') is None else report['mean_distance_error_um']:.1f} um, "
            f"geometry={report.get('geometry_confidence', 0.0):.3f}"
        )

    def remove_sample(self, event):
        """Simulate sample removal by translating the sample relative to the stage."""
        self.state.sample_removed = True
        remount_center_x = float(self.state.probe_tip_x)
        remount_center_y = float(self.state.probe_tip_y)
        dx = random.uniform(-200, 200)
        dy = random.uniform(-200, 200)
        self._apply_simulated_sample_remount(dx, dy, 0.0)
        self._reset_view_to_zoom(min(self.state.zoom_levels), center_x_um=remount_center_x, center_y_um=remount_center_y)
        if self.state.pi_mode:
            self.stage.reset(self.state.x, self.state.y)
            self.stage.cmd_x = self.state.x
            self.stage.cmd_y = self.state.y
        self.state.origin_defined = False
        self._refresh_current_view()
        self.log(
            "Sample removal simulation: sample remounted relative to the stage by "
            f"dX={dx:+.1f} um, dY={dy:+.1f} um at zoom {self.state.current_zoom_level:.2f}x"
        )
        if self.state.origin_template is not None:
            self.log("Stored origin template kept for later re-identification after remount.")

    def relocate(self, event):
        """Relocate using coarse-to-fine matching, landmark consensus, and final verification."""
        if not self._begin_action("relocate", "Relocation is already running"):
            return
        try:
            if self._camera_frame_only_relocation_enabled():
                self._relocate_using_current_camera_frame("Relocation")
                return
            if self.state.site_memory is None:
                self._try_load_latest_site_memory()
            if self.state.site_memory is not None and self.state.ref_template is None:
                self._activate_site_memory(self.state.site_memory, source_dir=self.state.last_saved_site_dir)
            if self.state.ref_template is None:
                self.log("Please save reference position first")
                return

            self.log("Starting coarse-to-fine relocation...")
            site_memory = self.state.site_memory or {}
            coarse_target_x = float(self.state.x)
            coarse_target_y = float(self.state.y)
            coarse_result = None
            affine_report = None
            fine_search_surface = self.state.surface_image
            fine_search_target_x = float(self.state.x)
            fine_search_target_y = float(self.state.y)
            fine_to_current_matrix = None

            if site_memory.get("reference_top_left"):
                coarse_target_x = float(site_memory["reference_top_left"]["x_um"])
                coarse_target_y = float(site_memory["reference_top_left"]["y_um"])
                fine_search_target_x = coarse_target_x
                fine_search_target_y = coarse_target_y

            affine_report = self._estimate_coarse_affine_transform()
            has_rotation = (
                affine_report is not None
                and abs(affine_report.get("rotation_deg", 0.0)) > 1.0
            )
            use_de_rotation = (
                has_rotation
                and affine_report["confidence"] >= max(self.state.relocation_min_affine_confidence, 0.30)
            )

            if use_de_rotation:
                # 有旋转且置信度高 → de-rotate surface 后细搜索
                fine_to_current_matrix = affine_report["full_matrix"]
                inv_matrix = invert_affine(fine_to_current_matrix)
                ref_x = float(self.state.ref_x)
                ref_y = float(self.state.ref_y)
                template_h, template_w = self.state.ref_template.shape[:2]
                search_margin = (
                    self.state.relocation_fine_half_range_um * 4.0
                    + max(float(template_w), float(template_h))
                )

                # Handle negative reference coordinates: the reference may be
                # outside the warped image bounds.  Expand canvas and shift so
                # the reference lands in a searchable positive region.
                offset_x = (-ref_x + search_margin) if ref_x < 0 else 0.0
                offset_y = (-ref_y + search_margin) if ref_y < 0 else 0.0
                surface_h, surface_w = self.state.surface_image.shape[:2]

                if offset_x > 0 or offset_y > 0:
                    new_w = int(surface_w + offset_x)
                    new_h = int(surface_h + offset_y)
                    adjusted_inv = inv_matrix.copy()
                    adjusted_inv[0, 2] = (
                        inv_matrix[0, 2]
                        - (inv_matrix[0, 0] * offset_x + inv_matrix[0, 1] * offset_y)
                    )
                    adjusted_inv[1, 2] = (
                        inv_matrix[1, 2]
                        - (inv_matrix[1, 0] * offset_x + inv_matrix[1, 1] * offset_y)
                    )
                    fine_search_surface = apply_affine(
                        self.state.surface_image, adjusted_inv,
                        output_shape=(new_h, new_w),
                    )
                else:
                    fine_search_surface = apply_affine(
                        self.state.surface_image, inv_matrix,
                        output_shape=(surface_h, surface_w),
                    )
                fine_search_target_x = ref_x + offset_x
                fine_search_target_y = ref_y + offset_y
                self.log(
                    f"Fine search surface: shape={fine_search_surface.shape}, "
                    f"min={np.min(fine_search_surface):.0f}, max={np.max(fine_search_surface):.0f}, "
                    f"mean={np.mean(fine_search_surface):.1f}, "
                    f"offset=({offset_x:.0f},{offset_y:.0f})"
                )
                self.log(
                    f"Invert affine matrix: "
                    f"[{inv_matrix[0,0]:.4f},{inv_matrix[0,1]:.4f},{inv_matrix[0,2]:.1f}; "
                    f"{inv_matrix[1,0]:.4f},{inv_matrix[1,1]:.4f},{inv_matrix[1,2]:.1f}]"
                )
                coarse_target_x, coarse_target_y = transform_point(
                    fine_to_current_matrix,
                    self.state.ref_x,
                    self.state.ref_y,
                )
                self.log(
                    (
                        "[ML mode] Coarse localization: "
                        if affine_report.get("ml_remount_prediction")
                        else "Coarse affine recovery: "
                    )
                    + f"dTheta={affine_report['rotation_deg']:+.2f} deg, "
                    + f"inliers={affine_report['inlier_count']}/{affine_report['match_count']}, "
                    + f"confidence={affine_report['confidence']:.3f}"
                )
                predicted_transform = affine_report.get("predicted_remount_transform")
                if predicted_transform is not None:
                    self.log(
                        "Phase 2 remount predictor: "
                        f"dX={predicted_transform['dx_um']:+.1f} um, "
                        f"dY={predicted_transform['dy_um']:+.1f} um, "
                        f"dTheta={predicted_transform['dtheta_deg']:+.2f} deg"
                    )
                ml_remount = affine_report.get("ml_remount_prediction")
                if ml_remount is not None:
                    self.log(
                        "[ML mode] 5w model prediction: "
                        f"dX={ml_remount['dx_um']:+.1f} um, "
                        f"dY={ml_remount['dy_um']:+.1f} um, "
                        f"dTheta={ml_remount['dtheta_deg']:+.2f} deg"
                    )
                retrieval_candidates = affine_report.get("retrieval_candidates") or []
                if retrieval_candidates:
                    best_candidate = retrieval_candidates[0]
                    self.log(
                        "Phase 2 low-mag retrieval best match: "
                        f"{best_candidate.get('site_id', 'unknown')} "
                        f"(distance {best_candidate['distance']:.3f})"
                    )
            elif affine_report is not None:
                # NCC 粗定位 (无旋转) → 用匹配位置做细搜索起点
                dx_total = float(affine_report["translation_px"][0])
                dy_total = float(affine_report["translation_px"][1])
                fine_search_target_x = float(self.state.ref_x + dx_total)
                fine_search_target_y = float(self.state.ref_y + dy_total)
                self.log(
                    "Coarse NCC localization: "
                    f"dX={dx_total:+.1f} um, dY={dy_total:+.1f} um, "
                    f"confidence={affine_report['confidence']:.3f}"
                )

            if fine_to_current_matrix is None and site_memory.get("lowmag_landmarks"):
                coarse_zoom = float((site_memory.get("overview") or {}).get("zoom_level", site_memory.get("coarse_zoom_level", min(self.state.zoom_levels))))
                overview = self._build_camera_overview(
                    zoom_level=coarse_zoom,
                    center_x_um=float(self.state.probe_tip_x),
                    center_y_um=float(self.state.probe_tip_y),
                )
                if overview is not None:
                    coarse_result = self._build_lowmag_guidance_report(overview, site_memory=site_memory)
                if coarse_result is not None and coarse_result["support_count"] >= self.state.relocation_min_landmark_support:
                    coarse_target_x = float(self.state.ref_x + coarse_result["offset_x_um"])
                    coarse_target_y = float(self.state.ref_y + coarse_result["offset_y_um"])
                    fine_search_target_x = coarse_target_x
                    fine_search_target_y = coarse_target_y
                    self.log(
                        "Coarse low-mag localization: "
                        f"dX={coarse_result['offset_x_um']:+.1f} um, "
                        f"dY={coarse_result['offset_y_um']:+.1f} um, "
                        f"support={coarse_result['support_count']}, "
                        f"confidence={coarse_result['confidence']:.3f}"
                    )
                else:
                    self.state.lowmag_guidance_report = None
                    self.log("Coarse low-mag localization was ambiguous; falling back to the last known reference region.")

            desired_x = float(coarse_target_x)
            desired_y = float(coarse_target_y)
            search_half_range = self.state.relocation_fine_half_range_um
            max_search_half_range = self.state.relocation_fine_half_range_um * 4.0
            self.log(
                f"Fine search starting at X={desired_x:.1f} um, Y={desired_y:.1f} um, "
                f"half_range={search_half_range:.0f} um"
            )
            while True:
                fine_match = self._match_camera_template(
                    self.state.ref_template,
                    desired_x,
                    desired_y,
                    half_range_um=search_half_range,
                )
                if fine_match is not None:
                    desired_x = float(fine_match["x"])
                    desired_y = float(fine_match["y"])
                    self.log(
                        f"Fine relocation pass: "
                        f"X={desired_x:.1f} um, Y={desired_y:.1f} um, "
                        f"score={fine_match['score']:.3f}, gap={fine_match.get('score_gap', 0.0):.3f}"
                    )
                    if fine_match["score"] >= self.state.relocation_min_match_score:
                        break
                    self.log("Fine match score still weak; expanding range")
                    fine_match = None
                search_half_range *= 2.0
                if search_half_range > max_search_half_range:
                    self.log(
                        f"Reference template could not be matched "
                        f"(tried up to {max_search_half_range:.0f} um half-range)"
                    )
                    return
                self.log(
                    f"No match at current range, expanding to half_range={search_half_range:.0f} um"
                )

            if fine_match is None:
                self.log("Fine relocation did not produce a usable reference match.")
                return

            if fine_to_current_matrix is not None:
                desired_current_x, desired_current_y = transform_point(
                    fine_to_current_matrix,
                    desired_x,
                    desired_y,
                )
            else:
                desired_current_x = desired_x
                desired_current_y = desired_y

            fine_match, verification = self._select_verified_fine_match(
                fine_match,
                site_memory=site_memory,
            )
            desired_x = float(fine_match["x"])
            desired_y = float(fine_match["y"])
            if verification is None:
                verification = self._verify_relocation(desired_x, desired_y, site_memory=site_memory)
            self.state.last_relocation_report = {
                "affine": affine_report,
                "coarse": coarse_result,
                "fine_affine": None,
                "fine": fine_match,
                "predicted_current_top_left": {"x_um": float(desired_current_x), "y_um": float(desired_current_y)},
                "verification": verification,
            }
            if not verification.get("verified", False):
                self.log(
                    "Relocation verification failed: "
                    f"reference score={verification.get('reference_score', 0.0):.3f}, "
                    f"gap={verification.get('reference_score_gap', 0.0):.3f}"
                )
                landmark_consensus = verification.get("landmark_consensus")
                if landmark_consensus is not None:
                    self.log(
                        "High-mag landmark verification: "
                        f"support={landmark_consensus['support_count']}, "
                        f"confidence={landmark_consensus['confidence']:.3f}"
                    )
                geometry_check = verification.get("geometry_check")
                if geometry_check is not None:
                    self.log(
                        "HUD landmark geometry check: "
                        f"matched={geometry_check.get('matched_count', 0)}, "
                        f"pair_error={0.0 if geometry_check.get('mean_pair_error_um') is None else geometry_check['mean_pair_error_um']:.1f} um, "
                        f"tip_error={0.0 if geometry_check.get('mean_distance_error_um') is None else geometry_check['mean_distance_error_um']:.1f} um, "
                        f"confidence={geometry_check.get('geometry_confidence', 0.0):.3f}"
                    )
                affine_verify = verification.get("affine_verify")
                if affine_verify is not None:
                    self.log(
                        "Affine verification: "
                        f"dTheta={affine_verify['rotation_deg']:+.2f} deg, "
                        f"inliers={affine_verify['inlier_count']}/{affine_verify['match_count']}, "
                        f"confidence={affine_verify['confidence']:.3f}"
                    )
                if verification.get("same_site_probability") is not None:
                    self.log(
                        f"Phase 2 same-site probability: {verification['same_site_probability']:.3f}"
                    )
                self._enter_manual_landmark_guidance(desired_current_x, desired_current_y)
                return

            if self.ai_compensator is not None and self.ai_mode:
                cmd_x = self._predict_compensated_axis(desired_current_x, "x")
                cmd_y = self._predict_compensated_axis(desired_current_y, "y")
                self.log("AI compensation applied to relocation command")
            else:
                cmd_x = float(desired_current_x)
                cmd_y = float(desired_current_y)

            cmd_x, cmd_y = self._clamp_to_stage_margin(cmd_x, cmd_y)

            self._start_smooth_move(cmd_x, cmd_y)
            self.state.sample_removed = False
            self.log(
                "Relocation verified and accepted: "
                f"reference score={verification['reference_score']:.3f}, "
                f"gap={verification['reference_score_gap']:.3f}"
            )
            landmark_consensus = verification.get("landmark_consensus")
            if landmark_consensus is not None:
                self.log(
                    "High-mag landmark verification: "
                    f"support={landmark_consensus['support_count']}, "
                    f"confidence={landmark_consensus['confidence']:.3f}"
                )
            geometry_check = verification.get("geometry_check")
            if geometry_check is not None:
                self.log(
                    "HUD landmark geometry check: "
                    f"matched={geometry_check.get('matched_count', 0)}, "
                    f"pair_error={0.0 if geometry_check.get('mean_pair_error_um') is None else geometry_check['mean_pair_error_um']:.1f} um, "
                    f"tip_error={0.0 if geometry_check.get('mean_distance_error_um') is None else geometry_check['mean_distance_error_um']:.1f} um, "
                    f"confidence={geometry_check.get('geometry_confidence', 0.0):.3f}"
                )
            affine_verify = verification.get("affine_verify")
            if affine_verify is not None:
                self.log(
                    "Affine verification: "
                    f"dTheta={affine_verify['rotation_deg']:+.2f} deg, "
                    f"inliers={affine_verify['inlier_count']}/{affine_verify['match_count']}, "
                    f"confidence={affine_verify['confidence']:.3f}"
                )
            if verification.get("same_site_probability") is not None:
                self.log(
                    f"Phase 2 same-site probability: {verification['same_site_probability']:.3f}"
                )
            self.log(f"Relocation complete, moved to ({cmd_x:.1f}, {cmd_y:.1f})")
        finally:
            self._end_action("relocate")

    # ------------------------------------------------------------------
    # ML Recognition Helpers
    # ------------------------------------------------------------------
    def _ml_recognize_pattern(self, search_image, ref_template, center_x=None, center_y=None, half_range_um=None):
        """用 ML 特征匹配在 search_image 中找到 ref_template 的最佳匹配位置。

        Returns:
            (x: int, y: int, score: float) 或 None
        """
        if not self._use_ml() or self.ml_pattern_matcher is None:
            return None
        try:
            # 如果提供了搜索范围，裁剪搜索图像以加速
            if center_x is not None and center_y is not None and half_range_um is not None:
                tpl_h, tpl_w = ref_template.shape[:2]
                sch_h, sch_w = search_image.shape[:2]
                x0 = int(max(0, center_x - half_range_um))
                y0 = int(max(0, center_y - half_range_um))
                x1 = int(min(sch_w, center_x + half_range_um + tpl_w))
                y1 = int(min(sch_h, center_y + half_range_um + tpl_h))
                if x1 <= x0 or y1 <= y0:
                    return None
                cropped = search_image[y0:y1, x0:x1]
                matches = self.ml_pattern_matcher.match(
                    ref_template, cropped, top_k=1, stride_frac=0.25, min_score=0.35
                )
                if matches:
                    best = matches[0]
                    return (best["x"] + x0, best["y"] + y0, best["score"])
                return None

            matches = self.ml_pattern_matcher.match(
                ref_template, search_image, top_k=1, stride_frac=0.30, min_score=0.35
            )
            if matches:
                best = matches[0]
                return (best["x"], best["y"], best["score"])
        except Exception as e:
            self.log(f"ML pattern recognition failed: {e}")
        return None

    def _ml_predict_remount(self, ref_overview_image, cur_overview_image,
                            scale_x_um_per_px=1.0, scale_y_um_per_px=1.0):
        """用 ML 回归器预测 remount 变换 (dx, dy, dtheta)。

        Returns:
            dict: {"dx_um": float, "dy_um": float, "dtheta_deg": float} 或 None
        """
        if self.deep_regressor is None:
            return None
        if self.deep_regressor_is_5w:
            # The 5w regressor is trained on reference_template patch pairs in pixel
            # space, not low-magnification overview images. Using it here causes
            # predictions that look valid offline but fail inside the program.
            if not self._logged_5w_overview_skip:
                self.log(
                    "Skipping overview ML remount prediction: the 5w model is "
                    "template-trained and is kept only for template-based rotation hints."
                )
                self._logged_5w_overview_skip = True
            return None
        try:
            result = self.deep_regressor.predict(ref_overview_image, cur_overview_image)
            return result
        except Exception as e:
            self.log(f"ML remount prediction failed: {e}")
        return None

    def _ml_verify_same_site(self, ref_image, candidate_image):
        """用 ML 分类器验证两张图是否同一 site。

        Returns:
            (is_same: bool, probability: float) 或 (False, None)
        """
        if not self._use_ml() or self.deep_classifier is None:
            return False, None
        try:
            return self.deep_classifier.classify(ref_image, candidate_image, threshold=0.5)
        except Exception as e:
            self.log(f"ML site verification failed: {e}")
        return False, None

    # ------------------------------------------------------------------
    # Page 2 – AI Recall & Recover (PPT slide 2)
    # ------------------------------------------------------------------
    def ai_recall_and_recover(self, event):
        """Two-stage AI relocation: low-mag coarse recall first, then high-mag refinement."""
        if not self._begin_action("ai_recall", "AI recall is already running"):
            return
        try:
            if self._camera_frame_only_relocation_enabled():
                self._relocate_using_current_camera_frame("AI Recall & Recover")
                return
            self.log("===== AI Recall & Recover =====")
            if self.state.site_memory is None:
                self._try_load_latest_site_memory()
            if self.state.site_memory is not None and self.state.ref_template is None:
                self._activate_site_memory(self.state.site_memory, source_dir=self.state.last_saved_site_dir)
            if self.state.ref_template is None:
                self.log("No saved site memory found. Save a reference region first (1. Save Region).")
                return

            self._wait_for_zoom_complete()
            site_memory = self.state.site_memory or {}
            final_zoom = float(site_memory.get("final_zoom_level", site_memory.get("zoom_level", self.state.current_zoom_level)))
            coarse_zoom = float(site_memory.get("coarse_zoom_level", min(self.state.zoom_levels)))
            coarse_reference_top_left = site_memory.get("coarse_reference_top_left") or site_memory.get("reference_top_left") or {}
            final_reference_top_left = site_memory.get("reference_top_left") or {}

            if not np.isclose(float(self.state.current_zoom_level), coarse_zoom):
                self._begin_quantized_zoom(coarse_zoom)
                self.log(f"Stage 1/3: returning to low magnification at {coarse_zoom:.2f}x")
                self._wait_for_zoom_complete()
            else:
                self.log(f"Stage 1/3: already at low magnification {coarse_zoom:.2f}x")

            self.log("Running low-magnification coarse recall...")

            affine_report = self._estimate_coarse_affine_transform()
            if affine_report is not None and affine_report["confidence"] >= self.state.relocation_min_affine_confidence:
                self.log(
                    "Low-mag pattern recognition: "
                    f"rotation={affine_report['rotation_deg']:+.2f} deg, "
                    f"inliers={affine_report['inlier_count']}/{affine_report['match_count']}, "
                    f"confidence={affine_report['confidence']:.3f}"
                )
            else:
                affine_confidence = "N/A" if affine_report is None else f"{affine_report['confidence']:.3f}"
                self.log(
                    "Low-mag pattern recognition confidence low "
                    f"({affine_confidence}), "
                    "proceeding with landmark-only fallback."
                )

            coarse_target_x = float(self.state.x)
            coarse_target_y = float(self.state.y)
            coarse_result = None
            fine_search_surface = self.state.surface_image
            fine_search_target_x = float(self.state.x)
            fine_search_target_y = float(self.state.y)
            fine_to_current_matrix = None
            coarse_strategy = "saved coarse reference"

            if coarse_reference_top_left:
                coarse_target_x = float(coarse_reference_top_left.get("x_um", coarse_target_x))
                coarse_target_y = float(coarse_reference_top_left.get("y_um", coarse_target_y))
            if final_reference_top_left:
                fine_search_target_x = float(final_reference_top_left.get("x_um", fine_search_target_x))
                fine_search_target_y = float(final_reference_top_left.get("y_um", fine_search_target_y))

            if affine_report is not None and affine_report["confidence"] >= self.state.relocation_min_affine_confidence:
                fine_to_current_matrix = affine_report["full_matrix"]
                coarse_strategy = "low-mag camera NCC"
                fine_search_surface = apply_affine(
                    self.state.surface_image,
                    invert_affine(fine_to_current_matrix),
                    output_shape=self.state.surface_image.shape[:2],
                )
                fine_search_target_x = float(self.state.ref_x)
                fine_search_target_y = float(self.state.ref_y)
                coarse_target_x, coarse_target_y = transform_point(
                    fine_to_current_matrix, coarse_target_x, coarse_target_y,
                )

            if fine_to_current_matrix is None and site_memory.get("lowmag_landmarks"):
                overview = self._build_camera_overview(
                    zoom_level=coarse_zoom,
                    center_x_um=float(self.state.probe_tip_x),
                    center_y_um=float(self.state.probe_tip_y),
                )
                if overview is not None:
                    coarse_result = self._build_lowmag_guidance_report(overview, site_memory=site_memory)
                    if coarse_result is not None:
                        coarse_strategy = "low-mag landmark consensus"
                        coarse_target_x = float(coarse_result["offset_x_um"] + coarse_target_x)
                        coarse_target_y = float(coarse_result["offset_y_um"] + coarse_target_y)
                        fine_search_target_x = float(coarse_result["offset_x_um"] + self.state.ref_x)
                        fine_search_target_y = float(coarse_result["offset_y_um"] + self.state.ref_y)
                        self.log(
                            "Low-mag landmark fallback: "
                            f"dX={coarse_result['offset_x_um']:+.1f} um, "
                            f"dY={coarse_result['offset_y_um']:+.1f} um, "
                            f"support={coarse_result['support_count']}, "
                            f"confidence={coarse_result['confidence']:.3f}"
                        )
                    else:
                        self.state.lowmag_guidance_report = None

            ml_remount = None
            if fine_to_current_matrix is None and coarse_result is None:
                cur_overview = self._build_camera_overview(
                    zoom_level=coarse_zoom,
                    center_x_um=float(self.state.probe_tip_x),
                    center_y_um=float(self.state.probe_tip_y),
                )
                if cur_overview is not None and site_memory.get("overview", {}).get("image") is not None:
                    ml_remount = self._ml_predict_remount(
                        site_memory.get("overview", {}).get("image"),
                        cur_overview.get("image"),
                        scale_x_um_per_px=cur_overview.get("scale_x_um_per_px", 1.0),
                        scale_y_um_per_px=cur_overview.get("scale_y_um_per_px", 1.0),
                    )
                    if ml_remount is not None:
                        coarse_strategy = "ML remount prediction"
                        coarse_target_x = float(coarse_target_x + ml_remount["dx_um"])
                        coarse_target_y = float(coarse_target_y + ml_remount["dy_um"])
                        fine_search_target_x = float(self.state.ref_x + ml_remount["dx_um"])
                        fine_search_target_y = float(self.state.ref_y + ml_remount["dy_um"])
                        self.log(
                            f"🤖 ML remount guides coarse: "
                            f"dX={ml_remount['dx_um']:+.1f} um, "
                            f"dY={ml_remount['dy_um']:+.1f} um, "
                            f"dTheta={ml_remount['dtheta_deg']:+.2f} deg"
                        )
                else:
                    ml_remount = None

            if fine_to_current_matrix is None and coarse_result is None and ml_remount is None:
                self.log(
                    "No coarse low-mag match was confirmed. "
                    "Falling back to the saved coarse region and widening high-mag refinement there."
                )

            coarse_cmd_x, coarse_cmd_y = self._clamp_to_stage_margin(coarse_target_x, coarse_target_y)
            self._jump_view_to_target(coarse_cmd_x, coarse_cmd_y)
            if fine_to_current_matrix is None and coarse_result is None and ml_remount is None:
                fine_search_target_x = float(coarse_cmd_x)
                fine_search_target_y = float(coarse_cmd_y)
            if site_memory.get("lowmag_landmarks"):
                refreshed_overview = self._build_camera_overview(
                    zoom_level=coarse_zoom,
                    center_x_um=float(self.state.probe_tip_x),
                    center_y_um=float(self.state.probe_tip_y),
                )
                self._build_lowmag_guidance_report(refreshed_overview, site_memory=site_memory)
            self.log(
                f"Stage 2/3: low-mag recall positioned viewport near the saved region at "
                f"({coarse_cmd_x:.1f}, {coarse_cmd_y:.1f}) using {coarse_strategy}"
            )

            if not np.isclose(float(self.state.current_zoom_level), final_zoom):
                self._begin_quantized_zoom(final_zoom)
                self.log(f"Stage 3/3: returning to saved final zoom {final_zoom:.2f}x for fine relocation")
                self._wait_for_zoom_complete()
            else:
                self.log(f"Stage 3/3: already at saved final zoom {final_zoom:.2f}x")

            self.log("Running high-magnification refinement...")
            if fine_to_current_matrix is None and coarse_result is None and ml_remount is None:
                fine_search_target_x = float(self.state.x)
                fine_search_target_y = float(self.state.y)

            ml_match_result = None
            if ml_remount is None:
                cur_overview = self._build_camera_overview(
                    zoom_level=coarse_zoom,
                    center_x_um=float(self.state.probe_tip_x),
                    center_y_um=float(self.state.probe_tip_y),
                )
                if cur_overview is not None and site_memory.get("overview", {}).get("image") is not None:
                    ml_remount = self._ml_predict_remount(
                        site_memory.get("overview", {}).get("image"),
                        cur_overview.get("image"),
                        scale_x_um_per_px=cur_overview.get("scale_x_um_per_px", 1.0),
                        scale_y_um_per_px=cur_overview.get("scale_y_um_per_px", 1.0),
                    )

            fine_match = None
            ml_used = False

            if ml_match_result is not None:
                ml_x, ml_y, ml_score = ml_match_result
                fine_match = {"x": float(ml_x), "y": float(ml_y), "score": ml_score, "score_gap": ml_score}
                ml_used = True
                self.log(
                    f"🤖 ML pattern matched: position=({ml_x}, {ml_y}), "
                    f"score={ml_score:.3f}"
                )
                if ml_remount is not None:
                    self.log(
                        f"🤖 ML remount prediction: "
                        f"dX={ml_remount['dx_um']:+.1f} um, "
                        f"dY={ml_remount['dy_um']:+.1f} um, "
                        f"dTheta={ml_remount['dtheta_deg']:+.2f} deg"
                    )
            else:
                search_half_range = float(self.state.relocation_fine_half_range_um)
                max_search_half_range = float(self.state.relocation_fine_half_range_um) * 4.0
                search_center_x = float(coarse_cmd_x)
                search_center_y = float(coarse_cmd_y)
                while search_half_range <= max_search_half_range:
                    fine_match = self._match_camera_template(
                        self.state.ref_template,
                        search_center_x,
                        search_center_y,
                        half_range_um=search_half_range,
                    )
                    if fine_match is not None:
                        if search_half_range > float(self.state.relocation_fine_half_range_um):
                            self.log(
                                "Expanded high-mag search recovered a candidate: "
                                f"half_range={search_half_range:.0f} um, "
                                f"score={fine_match['score']:.3f}, "
                                f"gap={fine_match.get('score_gap', 0.0):.3f}"
                            )
                        break
                    search_half_range *= 2.0
                    if search_half_range <= max_search_half_range:
                        self.log(
                            "High-mag template not found yet; expanding search window to "
                            f"{search_half_range:.0f} um half-range"
                        )

            if fine_match is None and fine_to_current_matrix is not None:
                self.log("De-rotated surface search failed, trying on original surface ...")
                raw_search_x, raw_search_y = transform_point(
                    fine_to_current_matrix, self.state.ref_x, self.state.ref_y,
                )
                fine_match = match_reference_template(
                    self.state.surface_image, self.state.ref_template,
                    raw_search_x, raw_search_y,
                    half_range=self.state.relocation_fine_half_range_um * 2,
                )
                if fine_match is not None:
                    self.log("Original surface search succeeded after de-rotated surface failed")
                    fine_search_surface = self.state.surface_image
                    fine_search_target_x = raw_search_x
                    fine_search_target_y = raw_search_y
                    if fine_match is not None:
                        self.log(
                            f"Template matched on original surface: "
                            f"score={fine_match['score']:.3f}, gap={fine_match.get('score_gap', 0):.3f}"
                        )

            if fine_match is None:
                self.log("AI relocation could not produce a usable reference match.")
                self._enter_manual_landmark_guidance()
                return

            desired_x = float(fine_match["x"])
            desired_y = float(fine_match["y"])

            for iteration in range(int(self.state.relocation_max_iterations)):
                fine_match = self._match_camera_template(
                    self.state.ref_template,
                    desired_x,
                    desired_y,
                    half_range_um=self.state.relocation_fine_half_range_um,
                )
                if fine_match is not None:
                    desired_x = float(fine_match["x"])
                    desired_y = float(fine_match["y"])
                    self.log(
                        f"ML/CV refinement pass {iteration + 1}: "
                        f"X={desired_x:.1f} um, Y={desired_y:.1f} um, "
                        f"score={fine_match['score']:.3f}"
                    )
                else:
                    break

            if fine_to_current_matrix is not None:
                desired_current_x, desired_current_y = transform_point(
                    fine_to_current_matrix, desired_x, desired_y,
                )
            else:
                desired_current_x = desired_x
                desired_current_y = desired_y

            fine_match, verification = self._select_verified_fine_match(
                fine_match,
                site_memory=site_memory,
            )
            desired_x = float(fine_match["x"])
            desired_y = float(fine_match["y"])
            if verification is None:
                verification = self._verify_relocation(desired_x, desired_y, site_memory=site_memory)
            ml_verified, ml_prob = self._ml_verify_same_site(
                self.state.ref_template,
                self._capture_matching_template(
                    desired_x,
                    desired_y,
                    float(self.state.fov_width),
                    float(self.state.fov_height),
                    zoom_level=float(self.state.current_zoom_level),
                ),
            )
            if ml_prob is not None:
                verification["ml_same_site_probability"] = ml_prob
                verification["ml_verified"] = ml_verified
                self.log(f"🤖 ML site verification: probability={ml_prob:.3f}, same_site={'YES' if ml_verified else 'NO'}")

            final_verified = verification.get("verified", False)
            if ml_verified and not final_verified:
                self.log("Traditional verification failed but ML classifier confirms same-site — accepting.")
                final_verified = True
                verification["verified"] = True

            self.state.last_relocation_report = {
                "affine": affine_report,
                "coarse": coarse_result,
                "fine_affine": None,
                "fine": fine_match,
                "ml_used": ml_used,
                "ml_remount": ml_remount,
                "predicted_current_top_left": {"x_um": float(desired_current_x), "y_um": float(desired_current_y)},
                "verification": verification,
            }

            if final_verified:
                cmd_x, cmd_y = self._clamp_to_stage_margin(desired_current_x, desired_current_y)
                self._start_smooth_move(cmd_x, cmd_y)
                self.state.sample_removed = False
                self.log(
                    "✅ AI Recall SUCCESS: "
                    f"reference score={verification['reference_score']:.3f}, "
                    f"gap={verification['reference_score_gap']:.3f}, "
                    f"{'ML' if ml_used else 'CV'} recognition, "
                    f"moved to ({cmd_x:.1f}, {cmd_y:.1f})"
                )
            else:
                self.log(
                    "⚠️ AI Recall verification FAILED: "
                    f"reference score={verification.get('reference_score', 0.0):.3f}, "
                    f"gap={verification.get('reference_score_gap', 0.0):.3f}"
                )
                geometry_check = verification.get("geometry_check")
                if geometry_check is not None:
                    self.log(
                        "Landmark geometry: "
                        f"matched={geometry_check.get('matched_count', 0)}, "
                        f"confidence={geometry_check.get('geometry_confidence', 0.0):.3f}"
                    )
                self._enter_manual_landmark_guidance(desired_current_x, desired_current_y)
        finally:
            self._end_action("ai_recall")

    def _enter_click_to_move_correction(self, target_x=None, target_y=None):
        """Enter click-to-move mode: user clicks the correct position on the viewport
        to manually guide the cantilever."""
        self.state.ai_relocate_awaiting_click = True
        self.state.ai_relocate_pending_target_x = target_x
        self.state.ai_relocate_pending_target_y = target_y
        if target_x is not None and target_y is not None:
            self.log(
                "👉 Click-to-move correction: click the CORRECT position on the viewport.\n"
                f"   AI best guess was ({target_x:.1f}, {target_y:.1f}).\n"
                "   Right-click to cancel correction mode."
            )
        else:
            self.log(
                "👉 Click-to-move: click the target position on the viewport.\n"
                "   Right-click to cancel."
            )

    # ------------------------------------------------------------------
    # Page 3 – AI Zoom & Recover (PPT slide 3)
    # ------------------------------------------------------------------
    def ai_zoom_recover(self, event):
        """AI Zoom: recall last zoom value, recognize surrounding pattern.
        If not recognized, zoom out and in to search. Then move + verify."""
        if not self._begin_action("ai_zoom", "AI zoom recovery is already running"):
            return
        try:
            if self._camera_frame_only_relocation_enabled():
                self._relocate_using_current_camera_frame("AI Zoom & Recover")
                return
            self.log("===== AI Zoom & Recover =====")
            # 1. Machine recall the last zoom value
            if self.state.site_memory is None:
                self._try_load_latest_site_memory()
            if self.state.site_memory is not None and self.state.ref_template is None:
                self._activate_site_memory(self.state.site_memory, source_dir=self.state.last_saved_site_dir)

            site_memory = self.state.site_memory or {}
            saved_zoom = site_memory.get("zoom_level")
            if saved_zoom is not None:
                self.state.ai_zoom_recalled_level = float(saved_zoom)
                self._begin_quantized_zoom(float(saved_zoom))
                self.log(f"Recalled zoom: {float(saved_zoom):.2f}x. AI analyzing pattern...")
            else:
                self.log("No saved zoom level in site memory. Using current zoom for recognition.")

            # 2. AI recognize the current surrounding pattern at recalled zoom
            #    Wait for zoom animation to complete + view refresh
            self._wait_for_zoom_complete()

            recognized = self._attempt_pattern_recognition_at_current_zoom()
            if recognized is not None:
                rx, ry, rconf = recognized
                self.log(
                    f"AI recognized pattern at zoom {self.state.current_zoom_level:.2f}x: "
                    f"position=({rx:.1f}, {ry:.1f}), confidence={rconf:.3f}"
                )
                # 4. Move to recognized position
                self._finish_ai_zoom_move(rx, ry, rconf)
            else:
                # 3. Not recognized → zoom out search, then zoom in
                self.log("Pattern NOT recognized at current zoom. Starting zoom-out search...")
                self.state.ai_zoom_search_active = True
                self.state.ai_zoom_search_stage = "zooming_out"
                self._start_zoom_out_search()
        finally:
            self._end_action("ai_zoom")

    def _attempt_pattern_recognition_at_current_zoom(self):
        """Try to recognize the saved high-mag landmarks in the current viewport.
        Uses ML first, falls back to CV.
        Returns (target_x_um, target_y_um, confidence) or None."""
        site_memory = self.state.site_memory or {}
        current_view = self.state.current_matching_view if self.state.current_matching_view is not None else self.state.current_fov_raw
        if current_view is None or current_view.size == 0:
            return None

        # ── ML path: ResNet + sliding window ──
        if self._use_ml() and self.state.ref_template is not None:
            ml_result = self._ml_recognize_pattern(current_view, self.state.ref_template)
            if ml_result is not None:
                ml_x, ml_y, ml_score = ml_result
                tpl_h, tpl_w = self.state.ref_template.shape[:2]
                target_x = float(self.state.x + ml_x + tpl_w / 2.0)
                target_y = float(self.state.y + ml_y + tpl_h / 2.0)
                self.log(f"🤖 ML recognized pattern at zoom {self.state.current_zoom_level:.2f}x: score={ml_score:.3f}")
                return (target_x, target_y, float(ml_score))

        # ── CV fallback: landmark geometry + template matching ──
        highmag_landmarks = site_memory.get("highmag_landmarks", [])
        if not highmag_landmarks:
            self.log("No saved high-mag landmarks to recognize.")
            return None

        report = analyze_landmark_geometry(
            highmag_landmarks,
            current_view,
            view_origin_x_um=float(self.state.x),
            view_origin_y_um=float(self.state.y),
            tip_x_um=float(self.state.probe_tip_x),
            tip_y_um=float(self.state.probe_tip_y),
            min_score=max(0.35, float(self.state.relocation_min_match_score) - 0.07),
            min_gap=max(0.01, float(self.state.relocation_min_score_gap) - 0.01),
        )
        matched_count = report.get("matched_count", 0)
        geometry_conf = report.get("geometry_confidence", 0.0)
        if matched_count >= self.state.relocation_min_landmark_support and geometry_conf >= 0.30:
            # Use the matched landmarks to compute target position
            matches = report.get("matches", [])
            if matches:
                ref_tl = site_memory.get("reference_top_left") or {}
                ref_tip = site_memory.get("reference_tip") or {}
                avg_dx = float(np.mean([m.get("tip_dx_um", 0.0) or 0.0 for m in matches]))
                avg_dy = float(np.mean([m.get("tip_dy_um", 0.0) or 0.0 for m in matches]))
                target_x = float(self.state.probe_tip_x + avg_dx)
                target_y = float(self.state.probe_tip_y + avg_dy)
                return (target_x, target_y, float(geometry_conf))

        # Even with fewer matches, try template matching on current view
        if self.state.ref_template is not None:
            candidates = match_template_candidates(current_view, self.state.ref_template, top_k=1)
            if candidates and candidates[0]["score"] >= 0.40:
                best = candidates[0]
                tpl_h, tpl_w = self.state.ref_template.shape[:2]
                target_x = float(self.state.x + best["x"] + tpl_w / 2.0)
                target_y = float(self.state.y + best["y"] + tpl_h / 2.0)
                return (target_x, target_y, float(best["score"]))

        return None

    def _start_zoom_out_search(self):
        """Zoom out step by step, searching for landmarks at each level."""
        levels = list(self.state.zoom_levels)
        current_idx = self._get_zoom_level_index(self.state.current_zoom_level)
        # Find the next wider zoom level
        search_idx = max(0, current_idx - 1)
        if search_idx == current_idx:
            self.log("Already at widest zoom. Cannot zoom out further.")
            self.state.ai_zoom_search_active = False
            self.state.ai_zoom_search_stage = "idle"
            return
        self.state.ai_zoom_search_stage = "zooming_out"
        self.state._ai_zoom_search_idx = search_idx
        self._begin_quantized_zoom(levels[search_idx])
        self.log(f"Zooming out to {levels[search_idx]:.2f}x for search...")
        self._wait_for_zoom_complete()
        self._continue_zoom_search()

    def _continue_zoom_search(self):
        """Continue the zoom search: test recognition at current zoom level."""
        if not self.state.ai_zoom_search_active:
            return

        self.state.ai_zoom_search_stage = "searching"
        recognized = self._attempt_pattern_recognition_at_current_zoom()
        if recognized is not None:
            rx, ry, rconf = recognized
            self.log(
                f"Pattern FOUND during zoom search at {self.state.current_zoom_level:.2f}x: "
                f"confidence={rconf:.3f}"
            )
            self.state.ai_zoom_search_stage = "zooming_in"
            recalled_zoom = self.state.ai_zoom_recalled_level
            if recalled_zoom is not None and recalled_zoom != self.state.current_zoom_level:
                self._begin_quantized_zoom(float(recalled_zoom))
                self.log(f"Zooming back in to recalled level {float(recalled_zoom):.2f}x...")
                self._wait_for_zoom_complete()
            self.state.ai_zoom_search_active = False
            self.state.ai_zoom_search_stage = "done"
            self._finish_ai_zoom_move(rx, ry, rconf)
        else:
            # Try zooming out one more step
            levels = list(self.state.zoom_levels)
            current_idx = self._get_zoom_level_index(self.state.current_zoom_level)
            search_idx = getattr(self.state, "_ai_zoom_search_idx", current_idx - 1)
            next_idx = max(0, current_idx - 1)
            if next_idx < current_idx:
                self.state._ai_zoom_search_idx = next_idx
                self._begin_quantized_zoom(levels[next_idx])
                self.log(f"Still not found. Zooming further out to {levels[next_idx]:.2f}x...")
                self._wait_for_zoom_complete()
                self._continue_zoom_search()
            else:
                self.log("Zoom search exhausted. Pattern not found at any zoom level.")
                self.state.ai_zoom_search_active = False
                self.state.ai_zoom_search_stage = "idle"
                self._enter_click_to_move_correction()

    def _finish_ai_zoom_move(self, target_x, target_y, confidence):
        """Move cantilever to the AI-recognized position and verify."""
        # Convert from absolute coordinates to viewport top-left
        target_tl_x = target_x - self.state.fov_width / 2.0
        target_tl_y = target_y - self.state.fov_height / 2.0
        target_tl_x, target_tl_y = self._clamp_to_stage_margin(target_tl_x, target_tl_y)

        site_memory = self.state.site_memory or {}
        verification = self._verify_relocation(
            target_tl_x, target_tl_y,
            surface_image=self.state.surface_image,
            site_memory=site_memory,
        )
        if verification.get("verified", False):
            self._start_smooth_move(target_tl_x, target_tl_y)
            self.state.sample_removed = False
            self.log(
                "✅ AI Zoom SUCCESS: "
                f"confidence={confidence:.3f}, "
                f"verify_score={verification['reference_score']:.3f}, "
                f"moved to ({target_tl_x:.1f}, {target_tl_y:.1f})"
            )
        else:
            self.log(
                "⚠️ AI Zoom verification FAILED: "
                f"recognition_confidence={confidence:.3f}, "
                f"verify_score={verification.get('reference_score', 0.0):.3f}"
            )
            self._enter_click_to_move_correction(target_tl_x, target_tl_y)

    def toggle_ai_mode(self, event):
        """Toggle AI compensation mode."""
        self.ai_mode = not self.ai_mode
        self.log(f"AI compensation mode: {'ON' if self.ai_mode else 'OFF'}")
