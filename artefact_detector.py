"""
artefact_detector.py - AI object detector (AI only, no traditional fallback)
"""

from pathlib import Path

import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


BASE_DIR = Path(__file__).resolve().parent

class ArtefactDetector:
    def __init__(self, model_path="artefact_detector/exp1/weights/best.pt", conf_threshold=0.3):
        """
        Initialize detector
        conf_threshold: confidence threshold for AI detection (default 0.3, can be lowered to increase detection rate)
        """
        self.conf_threshold = conf_threshold
        self.model_path = self._resolve_model_path(model_path)
        self.model = None
        if YOLO is None:
            print("Ultralytics is not installed; artefact detector disabled.")
            self.model_loaded = False
            self.class_names = {0: 'cross', 1: 'fiducial', 2: 'circle', 3: 'square'}
            return
        try:
            self.model = YOLO(str(self.model_path))
            self.model_loaded = True
            print(f"Artefact detector model loaded: {self.model_path}")
            print(f"Recognizable classes: {self.model.names}")
            print(f"Detection confidence threshold: {self.conf_threshold}")
        except Exception as e:
            print(f"Failed to load artefact detector model from {self.model_path}: {e}")
            self.model_loaded = False
        
        self.class_names = {0: 'cross', 1: 'fiducial', 2: 'circle', 3: 'square'}

    def _resolve_model_path(self, model_path):
        path = Path(model_path)
        if path.is_absolute():
            return path
        cwd_candidate = Path.cwd() / path
        if cwd_candidate.exists():
            return cwd_candidate
        return BASE_DIR / path

    def _merge_detection(self, detected, candidate, distance_threshold=40):
        """Merge duplicate scan detections from overlapping fields of view."""
        cand_x, cand_y = candidate["abs_coord"]
        for existing in detected:
            if existing["class_name"] != candidate["class_name"]:
                continue
            ex_x, ex_y = existing["abs_coord"]
            if np.hypot(cand_x - ex_x, cand_y - ex_y) <= distance_threshold:
                if candidate["confidence"] > existing["confidence"]:
                    existing.update(candidate)
                return
        detected.append(candidate)
    
    def detect(self, image, conf_threshold=None):
        """AI only detection, no traditional method"""
        if not self.model_loaded:
            return []
        
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        results = self.model(image, verbose=False, conf=conf_threshold)
        detections = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    center_x = (xyxy[0] + xyxy[2]) / 2
                    center_y = (xyxy[1] + xyxy[3]) / 2
                    
                    detections.append({
                        'method': 'ai',
                        'class_id': cls,
                        'class_name': self.class_names.get(cls, 'unknown'),
                        'center': (center_x, center_y),
                        'bbox': xyxy,
                        'confidence': conf
                    })
        return detections
    
    def detect_in_fov(self, fov_image, fov_x, fov_y, conf_threshold=None):
        detections = self.detect(fov_image, conf_threshold=conf_threshold)
        results = []
        for det in detections:
            cx, cy = det['center']
            results.append({
                **det,
                'abs_coord': (fov_x + cx, fov_y + cy)
            })
        return results
    
    def detect_in_scan(self, state, artifact_layer, show_artifact, 
                       half_range=500, step=100, conf_threshold=None):
        from afm_utils import create_fov_image
        
        start_x = max(0, state.x - half_range)
        end_x = min(state.width_um - state.fov_width, state.x + half_range)
        start_y = max(0, state.y - half_range)
        end_y = min(state.height_um - state.fov_height, state.y + half_range)
        
        detected = []
        
        print(f"Scan range: X=[{start_x:.0f}, {end_x:.0f}], Y=[{start_y:.0f}, {end_y:.0f}]")
        
        for cx in np.arange(start_x, end_x, step):
            for cy in np.arange(start_y, end_y, step):
                fov, _, _ = create_fov_image(
                    state.sample, artifact_layer, show_artifact,
                    cx, cy, state.fov_width, state.fov_height
                )
                detections = self.detect_in_fov(fov, cx, cy, conf_threshold=conf_threshold)
                
                for det in detections:
                    candidate = {
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "abs_coord": det["abs_coord"],
                        "scan_origin": (cx, cy),
                    }
                    self._merge_detection(detected, candidate)

        for det in detected:
            print(
                f"  Detected: {det['class_name']} "
                f"(confidence: {det['confidence']:.2f}) at "
                f"({det['abs_coord'][0]:.1f}, {det['abs_coord'][1]:.1f})"
            )

        return detected
