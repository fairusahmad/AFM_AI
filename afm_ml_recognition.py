"""
afm_ml_recognition.py — Machine Learning 驱动的 AFM 图案识别与重定位

用预训练 ResNet18 提取深度特征，替代传统 CV（ORB + matchTemplate）：
  - DeepFeatureExtractor: ResNet18 特征提取 (512维)
  - MLPatternMatcher: 滑动窗口 ML 特征匹配 + 余弦相似度
  - MLTransformPredictor: MLP 回归器预测 dx, dy, dtheta
  - MLSameSiteClassifier: MLP 分类器判断是否同一 site

训练数据由 afm_phase2_ml.py 的合成变体生成器提供。
"""

import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

from afm_relocation import to_grayscale_u8

# ── 特征维度 ──────────────────────────────────────────────
FEATURE_DIM = 512  # ResNet18 去掉 fc 后的输出

# ── 图像预处理 ────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_preprocess = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def _preprocess_image(image):
    """灰度/彩色图像 → (1, 3, 224, 224) 标准化 tensor"""
    gray = to_grayscale_u8(image)
    if gray is None:
        return None
    # 转 3 通道
    if gray.ndim == 2:
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        rgb = gray[..., :3]
    return _preprocess(rgb).unsqueeze(0)


def _ensure_rgb_tensor(image):
    """numpy 灰度/彩色 → (1, 3, 224, 224) tensor（已在 GPU 上则保持）"""
    t = _preprocess_image(image)
    if t is None:
        return None
    return t


# ════════════════════════════════════════════════════════════
# 特征提取器
# ════════════════════════════════════════════════════════════
class DeepFeatureExtractor:
    """用预训练 ResNet18 提取 512 维图像特征"""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()  # 去掉分类头
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract(self, image):
        """图像 → (512,) numpy 特征向量。image 为 numpy 灰度或彩色。"""
        t = _ensure_rgb_tensor(image)
        if t is None:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        t = t.to(self.device)
        features = self.model(t)
        vec = features.cpu().numpy().reshape(-1).astype(np.float32)
        # L2 归一化
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        return vec

    @torch.no_grad()
    def extract_batch(self, images):
        """批量提取特征。images: list of numpy arrays"""
        tensors = []
        for img in images:
            t = _ensure_rgb_tensor(img)
            if t is not None:
                tensors.append(t)
        if not tensors:
            return np.zeros((0, FEATURE_DIM), dtype=np.float32)
        batch = torch.cat(tensors, dim=0).to(self.device)
        features = self.model(batch)
        vecs = features.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        vecs = vecs / norms
        return vecs


# ════════════════════════════════════════════════════════════
# ML 图案匹配器（滑动窗口 + 深度特征）
# ════════════════════════════════════════════════════════════
class MLPatternMatcher:
    """
    用深度特征做滑动窗口匹配，替代 cv2.matchTemplate。

    工作流程：
      1. 在 search_image 上以 stride 滑动窗口
      2. 每个窗口提取 ResNet 特征
      3. 与 reference 特征计算余弦相似度
      4. 返回 top-k 匹配位置
    """

    def __init__(self, extractor=None, device=None):
        self.extractor = extractor or DeepFeatureExtractor(device=device)

    def match(
        self,
        reference_image,
        search_image,
        top_k=3,
        stride_frac=0.25,
        min_score=0.30,
        suppress_radius_frac=0.5,
    ):
        """
        在 search_image 中匹配 reference_image。

        Returns:
            list of dict: [{"x": int, "y": int, "score": float}, ...]
        """
        ref_gray = to_grayscale_u8(reference_image)
        search_gray = to_grayscale_u8(search_image)
        if ref_gray is None or search_gray is None:
            return []

        ref_h, ref_w = ref_gray.shape[:2]
        sch_h, sch_w = search_gray.shape[:2]
        if ref_h > sch_h or ref_w > sch_w:
            return []

        ref_feat = self.extractor.extract(ref_gray)

        stride_px = max(8, int(min(ref_h, ref_w) * stride_frac))
        candidates = []

        for y in range(0, sch_h - ref_h + 1, stride_px):
            for x in range(0, sch_w - ref_w + 1, stride_px):
                patch = search_gray[y : y + ref_h, x : x + ref_w]
                patch_feat = self.extractor.extract(patch)
                score = float(np.dot(ref_feat, patch_feat))
                candidates.append({"x": int(x), "y": int(y), "score": score})

        if not candidates:
            return []

        # 排序并做非极大值抑制
        candidates.sort(key=lambda c: c["score"], reverse=True)
        selected = []
        suppress_radius = int(max(ref_h, ref_w) * suppress_radius_frac)

        for cand in candidates:
            if cand["score"] < min_score:
                continue
            if any(
                abs(cand["x"] - s["x"]) < suppress_radius
                and abs(cand["y"] - s["y"]) < suppress_radius
                for s in selected
            ):
                continue
            selected.append(cand)
            if len(selected) >= top_k:
                break

        return selected

    def score_pair(self, image_a, image_b):
        """计算两幅图像的相似度分数（余弦相似度）"""
        feat_a = self.extractor.extract(image_a)
        feat_b = self.extractor.extract(image_b)
        return float(np.dot(feat_a, feat_b))


# ════════════════════════════════════════════════════════════
# 深度特征拼接 (用于训练 MLP 分类器/回归器)
# ════════════════════════════════════════════════════════════
def deep_pair_features(extractor, reference_image, candidate_image):
    """
    提取两幅图像的深度特征，返回拼接特征向量。
    维度: ref_feat(512) + cand_feat(512) + diff_feat(512) + stat(5) = 1541
    与 train_remount_5w.py 的训练特征完全一致。
    """
    ref_feat = extractor.extract(reference_image)
    cand_feat = extractor.extract(candidate_image)

    # 深度特征差异
    diff_feat = ref_feat - cand_feat

    # 额外统计特征
    ref_norm = float(np.linalg.norm(ref_feat))
    cand_norm = float(np.linalg.norm(cand_feat))
    cosine_sim = float(np.dot(ref_feat, cand_feat))
    l1_dist = float(np.sum(np.abs(diff_feat)))
    l2_dist = float(np.linalg.norm(diff_feat))

    stat_features = np.array(
        [ref_norm, cand_norm, cosine_sim, l1_dist, l2_dist],
        dtype=np.float32,
    )

    return np.concatenate([ref_feat, cand_feat, diff_feat, stat_features])


# ════════════════════════════════════════════════════════════
# MLP 分类器：判断是否同一 site
# ════════════════════════════════════════════════════════════
class MLSameSiteClassifier:
    """基于深度特征的同一site分类器"""

    def __init__(self, model_bundle=None, device=None):
        self.extractor = DeepFeatureExtractor(device=device)
        self.model = None
        if model_bundle is not None:
            self.model = model_bundle.get("model")

    def predict_proba(self, reference_image, candidate_image):
        """返回是同一site的概率 [0, 1]"""
        if self.model is None:
            return None
        features = deep_pair_features(self.extractor, reference_image, candidate_image)
        features = features.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(features)[0, 1])
        else:
            prob = float(np.clip(self.model.predict(features)[0], 0.0, 1.0))
        return prob

    def classify(self, reference_image, candidate_image, threshold=0.5):
        prob = self.predict_proba(reference_image, candidate_image)
        if prob is None:
            return False, None
        return prob >= threshold, prob


# ════════════════════════════════════════════════════════════
# MLP 回归器：预测 remount 变换 (dx, dy, dtheta)
# ════════════════════════════════════════════════════════════
class MLTransformPredictor:
    """基于深度特征预测位移和旋转"""

    def __init__(self, model_bundle=None, device=None):
        self.extractor = DeepFeatureExtractor(device=device)
        self.model = None
        self.scale_factors = None  # 训练时的归一化因子
        if model_bundle is not None:
            self.model = model_bundle.get("model")
            self.scale_factors = model_bundle.get("scale_factors")

    def predict(self, reference_image, candidate_image):
        """
        预测位移和旋转。

        Returns:
            dict: {"dx_um": float, "dy_um": float, "dtheta_deg": float} 或 None
        """
        if self.model is None:
            return None
        features = deep_pair_features(self.extractor, reference_image, candidate_image)
        features = features.reshape(1, -1)
        prediction = np.asarray(self.model.predict(features), dtype=float).reshape(-1)
        if prediction.size < 3:
            return None
        dx, dy, dtheta = float(prediction[0]), float(prediction[1]), float(prediction[2])
        # 反归一化
        if self.scale_factors is not None:
            dx *= float(self.scale_factors.get("dx_scale", 1.0))
            dy *= float(self.scale_factors.get("dy_scale", 1.0))
            dtheta *= float(self.scale_factors.get("dtheta_scale", 1.0))
        return {"dx_um": dx, "dy_um": dy, "dtheta_deg": dtheta}


# ════════════════════════════════════════════════════════════
# 训练函数
# ════════════════════════════════════════════════════════════
def build_deep_same_site_dataset(
    site_memory_root,
    extractor=None,
    positive_augmentations=8,
    max_negative_pairs=200,
):
    """构建深度特征 same-site 分类数据集"""
    from afm_phase2_ml import (
        _load_site_memories,
        _synthetic_negative_variants,
        _synthetic_positive_variants,
    )

    if extractor is None:
        extractor = DeepFeatureExtractor()

    site_memories = _load_site_memories(site_memory_root)
    if len(site_memories) < 1:
        raise ValueError("Need at least one saved site memory.")

    features = []
    labels = []
    templates = []

    for site_dir, memory in site_memories:
        template = memory.get("reference_template")
        if template is None:
            continue
        variants = _synthetic_positive_variants(template, count=positive_augmentations)
        templates.append((site_dir, template, variants))
        for variant in variants:
            features.append(deep_pair_features(extractor, template, variant))
            labels.append(1)

    negative_pairs = 0
    if len(templates) >= 2:
        for index, (_, template_a, variants_a) in enumerate(templates):
            for jndex, (_, template_b, variants_b) in enumerate(templates):
                if index >= jndex:
                    continue
                pairs = [(template_a, template_b)]
                pairs.extend(
                    (template_a, vb) for vb in variants_b[: max(2, positive_augmentations // 2)]
                )
                pairs.extend(
                    (template_b, va) for va in variants_a[: max(2, positive_augmentations // 2)]
                )
                for ref_img, cand_img in pairs:
                    features.append(deep_pair_features(extractor, ref_img, cand_img))
                    labels.append(0)
                    negative_pairs += 1
                    if negative_pairs >= max_negative_pairs:
                        break
            if negative_pairs >= max_negative_pairs:
                break
    else:
        _, memory = site_memories[0]
        template = memory.get("reference_template")
        hard_negatives = []
        origin_template = memory.get("origin_template")
        if origin_template is not None:
            hard_negatives.append(origin_template)
        for landmark in memory.get("highmag_landmarks", []):
            patch = landmark.get("patch")
            if patch is not None and patch.size:
                hard_negatives.append(patch)
        hard_negatives.extend(
            _synthetic_negative_variants(template, count=max_negative_pairs)
        )
        for neg in hard_negatives[:max_negative_pairs]:
            features.append(deep_pair_features(extractor, template, neg))
            labels.append(0)

    return np.vstack(features), np.asarray(labels, dtype=np.int32)


def build_deep_remount_dataset(
    site_memory_root,
    extractor=None,
    augmentations_per_site=12,
):
    """构建深度特征 remount 变换回归数据集"""
    from afm_phase2_ml import _load_site_memories
    from afm_relocation import apply_affine, rotation_translation_affine

    if extractor is None:
        extractor = DeepFeatureExtractor()

    site_memories = _load_site_memories(site_memory_root)
    if not site_memories:
        raise ValueError("No saved site memories.")

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
            matrix = rotation_translation_affine(
                width, height,
                angle_deg=rotation_deg,
                shift_x_px=shift_x_px,
                shift_y_px=shift_y_px,
            )
            current_image = apply_affine(
                reference_image, matrix, output_shape=reference_image.shape[:2]
            )
            features.append(deep_pair_features(extractor, reference_image, current_image))
            labels.append([shift_x_px * scale_x, shift_y_px * scale_y, rotation_deg])

    labels = np.asarray(labels, dtype=np.float32)
    # 归一化
    scale_factors = {
        "dx_scale": float(max(np.std(labels[:, 0]) * 3.0, 1.0)),
        "dy_scale": float(max(np.std(labels[:, 1]) * 3.0, 1.0)),
        "dtheta_scale": float(max(np.std(labels[:, 2]) * 3.0, 1.0)),
    }
    labels[:, 0] /= scale_factors["dx_scale"]
    labels[:, 1] /= scale_factors["dy_scale"]
    labels[:, 2] /= scale_factors["dtheta_scale"]

    return np.vstack(features), labels, scale_factors


def train_deep_same_site_classifier(site_memory_root, output_path, device=None):
    """训练基于深度特征的 same-site MLP 分类器"""
    extractor = DeepFeatureExtractor(device=device)
    print("Building deep same-site dataset ...")
    X, y = build_deep_same_site_dataset(site_memory_root, extractor=extractor)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Positive: {int(np.sum(y > 0))}, Negative: {int(np.sum(y == 0))}")

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        batch_size=32,
        learning_rate="adaptive",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    print("Training MLP classifier ...")
    model.fit(X, y)
    print(f"  Training accuracy: {model.score(X, y):.3f}")

    bundle = {
        "model_type": "deep_same_site_classifier",
        "feature_dim": FEATURE_DIM,
        "model": model,
    }
    joblib.dump(bundle, output_path)
    print(f"Saved to {output_path}")
    return bundle


def train_deep_remount_predictor(site_memory_root, output_path, device=None):
    """训练基于深度特征的 remount 变换回归器"""
    extractor = DeepFeatureExtractor(device=device)
    print("Building deep remount dataset ...")
    X, y, scale_factors = build_deep_remount_dataset(site_memory_root, extractor=extractor)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Scale factors: {scale_factors}")

    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        batch_size=32,
        learning_rate="adaptive",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    print("Training MLP regressor ...")
    model.fit(X, y)
    train_score = model.score(X, y)
    print(f"  Training R^2: {train_score:.3f}")

    bundle = {
        "model_type": "deep_remount_predictor",
        "feature_dim": FEATURE_DIM,
        "model": model,
        "scale_factors": scale_factors,
    }
    joblib.dump(bundle, output_path)
    print(f"Saved to {output_path}")
    return bundle


# ════════════════════════════════════════════════════════════
# 便捷加载函数
# ════════════════════════════════════════════════════════════
def load_deep_same_site_classifier(model_path, device=None):
    """加载训练好的 same-site 分类器，如果文件不存在返回 None"""
    path = Path(model_path)
    if not path.exists():
        return None
    bundle = joblib.load(path)
    return MLSameSiteClassifier(model_bundle=bundle, device=device)


def load_deep_remount_predictor(model_path, device=None):
    """加载训练好的 remount 变换预测器，如果文件不存在返回 None"""
    path = Path(model_path)
    if not path.exists():
        return None
    bundle = joblib.load(path)
    return MLTransformPredictor(model_bundle=bundle, device=device)


# ════════════════════════════════════════════════════════════
# 5w 大样本 Remount 预测器 (像素空间)
# ════════════════════════════════════════════════════════════
class MLTransformPredictor5W:
    """基于 5w 大样本训练的 Remount 变换预测器。

    与旧版 MLTransformPredictor 的区别：
      - 旧版: 输出 dx_um/dy_um (微米空间), 用 std*3 归一化
      - 新版: 输出 dx_px/dy_px (像素空间), 用 mean/std 归一化
             调用方需自行乘以 scale_um_per_px 转为微米
    """

    def __init__(self, model_bundle=None, device=None):
        self.extractor = DeepFeatureExtractor(device=device)
        self.model = None
        self.scale_factors = None
        if model_bundle is not None:
            self.model = model_bundle.get("model")
            self.scale_factors = model_bundle.get("scale_factors", {})

    def predict(self, reference_image, candidate_image):
        """预测像素空间的相对偏移。

        Returns:
            dict: {"dx_px": float, "dy_px": float, "angle_deg": float} 或 None
        """
        if self.model is None:
            return None
        features = deep_pair_features(self.extractor, reference_image, candidate_image)
        features = features.reshape(1, -1)
        prediction = np.asarray(self.model.predict(features), dtype=float).reshape(-1)
        if prediction.size < 3:
            return None

        dx_norm, dy_norm, angle_norm = (
            float(prediction[0]),
            float(prediction[1]),
            float(prediction[2]),
        )

        if self.scale_factors:
            # 反归一化: x_orig = x_norm * std + mean
            dx_px = dx_norm * float(self.scale_factors.get("dx_std", 1.0)) + float(
                self.scale_factors.get("dx_mean", 0.0)
            )
            dy_px = dy_norm * float(self.scale_factors.get("dy_std", 1.0)) + float(
                self.scale_factors.get("dy_mean", 0.0)
            )
            angle_deg = angle_norm * float(self.scale_factors.get("angle_std", 1.0)) + float(
                self.scale_factors.get("angle_mean", 0.0)
            )
        else:
            dx_px, dy_px, angle_deg = dx_norm, dy_norm, angle_norm

        return {"dx_px": dx_px, "dy_px": dy_px, "angle_deg": angle_deg}

    def predict_um(self, reference_image, candidate_image, scale_x_um_per_px=1.0, scale_y_um_per_px=1.0):
        """预测微米空间的相对偏移（兼容旧接口）。

        Returns:
            dict: {"dx_um": float, "dy_um": float, "dtheta_deg": float} 或 None
        """
        result = self.predict(reference_image, candidate_image)
        if result is None:
            print("[DEBUG 5w] predict() returned None")
            return None
        dx_um = result["dx_px"] * scale_x_um_per_px
        dy_um = result["dy_px"] * scale_y_um_per_px
        print(f"[DEBUG 5w] ref_img={reference_image.shape if hasattr(reference_image,'shape') else '?'}, "
              f"cand_img={candidate_image.shape if hasattr(candidate_image,'shape') else '?'}")
        print(f"[DEBUG 5w] RAW px: dx_px={result['dx_px']:+.1f}, dy_px={result['dy_px']:+.1f}, angle_deg={result['angle_deg']:+.2f}")
        print(f"[DEBUG 5w] scale: x={scale_x_um_per_px:.4f} um/px, y={scale_y_um_per_px:.4f} um/px")
        print(f"[DEBUG 5w] FINAL um: dx_um={dx_um:+.1f}, dy_um={dy_um:+.1f}, dtheta_deg={result['angle_deg']:+.2f}")
        return {
            "dx_um": dx_um,
            "dy_um": dy_um,
            "dtheta_deg": result["angle_deg"],
        }


def load_deep_remount_predictor_5w(model_path, device=None):
    """加载 5w 大样本 remount 预测器，如果文件不存在返回 None"""
    path = Path(model_path)
    if not path.exists():
        return None
    bundle = joblib.load(path)
    if bundle.get("model_type") != "deep_remount_predictor_5w":
        return None
    return MLTransformPredictor5W(model_bundle=bundle, device=device)
