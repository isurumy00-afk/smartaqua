"""ONNX Runtime Disease Detection module for Side View (Camera 1) frames.

Classifies fish diseases using an ONNX model — optimised for Raspberry Pi 4B
with single-session, multi-threaded CPU execution via onnxruntime.

Model input:  [1, H, W, 3] float32 — pixel values in [0, 1]
Model output: [1, num_classes] float32 — class probability logits / softmax
"""

import json
from typing import Dict, Any, Optional
import numpy as np
from config import DISEASE_CLASSES_PATH, DISEASE_MODEL_ONNX_PATH
from utils.logger import get_logger

LOG = get_logger(__name__)


class DiseaseDetector:
    """ONNX Runtime session manager for side-view fish disease classification."""

    def __init__(self):
        self.session = None
        self.classes = None
        self.input_name: str = ""
        self.input_shape: tuple = (224, 224)  # (H, W) — updated on load

    def _load(self) -> bool:
        """Lazy-load ONNX model and class label mappings."""
        if self.session is not None:
            return True

        if not DISEASE_MODEL_ONNX_PATH.exists():
            LOG.warning("Disease ONNX model not found: %s", DISEASE_MODEL_ONNX_PATH)
            return False
        if not DISEASE_CLASSES_PATH.exists():
            LOG.warning("Disease class_names.json not found: %s", DISEASE_CLASSES_PATH)
            return False

        try:
            import onnxruntime as ort

            self.classes = json.loads(DISEASE_CLASSES_PATH.read_text(encoding="utf-8"))

            # Use 4 inter-op threads to match Pi 4B quad-core CPU
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                str(DISEASE_MODEL_ONNX_PATH),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

            meta = self.session.get_inputs()[0]
            self.input_name = meta.name
            # shape: [batch, H, W, C] or [batch, C, H, W]
            shape = meta.shape
            if len(shape) == 4:
                if shape[1] in (1, 3):   # NCHW
                    self.input_shape = (int(shape[2]), int(shape[3]))
                else:                     # NHWC
                    self.input_shape = (int(shape[1]), int(shape[2]))
            LOG.info(
                "Disease ONNX model loaded: %s  input=%s  classes=%d",
                DISEASE_MODEL_ONNX_PATH.name,
                self.input_shape,
                len(self.classes) if self.classes else 0,
            )
            return True
        except Exception as exc:
            LOG.error("Failed to initialise disease ONNX session: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame, fish_id: Optional[int] = None, tracks: Optional[list] = None) -> Dict[str, Any]:
        """Detect disease class and confidence on a frame or per-fish crops.

        When YOLOv8 tracking bounding boxes are provided via *tracks*, each
        cropped fish image is passed through the ONNX disease model individually.
        """
        if frame is None:
            return {"fish_id": fish_id, "disease_class": "Healthy", "confidence": 1.0, "per_fish_diseases": []}

        if tracks:
            return self.detect_from_tracks(frame, tracks)

        if not self._load():
            return {
                "fish_id": fish_id,
                "disease_class": "Healthy",
                "confidence": 1.0,
                "note": "ONNX model not available",
                "per_fish_diseases": [],
            }

        try:
            import cv2

            h, w = self.input_shape
            resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (w, h))
            input_data = (np.expand_dims(resized, axis=0).astype(np.float32) / 255.0)

            outputs = self.session.run(None, {self.input_name: input_data})
            probabilities = np.squeeze(outputs[0])  # shape: [num_classes]

            predicted_index = int(np.argmax(probabilities))
            disease = self.classes[predicted_index] if self.classes else f"Class_{predicted_index}"
            confidence = round(float(probabilities[predicted_index]), 4)

            return {
                "fish_id": fish_id,
                "disease_class": disease,
                "confidence": confidence,
                "per_fish_diseases": [],
            }
        except Exception as exc:
            LOG.error("Disease ONNX inference failed: %s", exc)
            return {
                "fish_id": fish_id,
                "disease_class": "Unknown",
                "confidence": 0.0,
                "error": str(exc),
                "per_fish_diseases": [],
            }

    def detect_from_tracks(self, frame, tracks: list) -> Dict[str, Any]:
        """Crop each YOLOv8 fish bounding box from frame and run disease inference."""
        if frame is None or not tracks:
            return self.detect(frame)

        h, w = frame.shape[:2]
        per_fish_results = []

        for fish in tracks:
            bbox = fish.get("bbox")
            fid = fish.get("fish_id")
            if not bbox or len(bbox) < 4:
                continue

            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))

            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                res = self.detect(crop, fish_id=fid)
                res["bbox"] = [x1, y1, x2, y2]
                per_fish_results.append(res)

        if not per_fish_results:
            return self.detect(frame)

        # Prioritise highest-confidence non-Healthy result
        diseased = [r for r in per_fish_results if "healthy" not in r.get("disease_class", "").lower()]
        primary = max(diseased or per_fish_results, key=lambda r: r.get("confidence", 0.0))

        return {
            "disease_class": primary.get("disease_class", "Healthy"),
            "confidence": primary.get("confidence", 1.0),
            "fish_id": primary.get("fish_id"),
            "per_fish_diseases": per_fish_results,
        }
