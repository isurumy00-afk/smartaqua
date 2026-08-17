"""Top View (Camera 2) YOLOv8 Fish Hunger Detection module.

Detects hungry fish hovering in the feeding zone of Camera 2 using the
fine-tuned YOLOv8 ONNX model at models/feeding/model.onnx.
Features 4-thread CPU execution optimized for Raspberry Pi 4B and
automatic contour-based computer vision fallback.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from config import FEEDING_MODEL_ONNX_PATH, FISH_CONFIDENCE
from utils.logger import get_logger

LOG = get_logger(__name__)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """Non-maximum suppression (numpy-only, no torch dependency)."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    kept = []
    while order.size > 0:
        i = order[0]
        kept.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou < iou_threshold]

    return kept


class HungerDetector:
    """YOLOv8 ONNX detector for top-camera feeding zone fish detection."""

    def __init__(self, model_path=FEEDING_MODEL_ONNX_PATH, confidence_threshold: float = FISH_CONFIDENCE):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.session = None
        self.input_name: str = ""
        self.input_shape: tuple = (640, 640)

    def _load_model(self) -> bool:
        """Lazy-load the feeding ONNX session once."""
        if self.session is not None:
            return True

        if not self.model_path.exists():
            LOG.warning("Top Camera YOLOv8 feeding model not found: %s", self.model_path)
            return False

        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            meta = self.session.get_inputs()[0]
            self.input_name = meta.name
            shape = meta.shape
            if len(shape) == 4:
                if shape[1] in (1, 3):  # NCHW
                    self.input_shape = (int(shape[2]), int(shape[3]))
                else:                    # NHWC
                    self.input_shape = (int(shape[1]), int(shape[2]))

            LOG.info(
                "Top Camera YOLOv8 feeding detector loaded: %s input=%s",
                self.model_path.name,
                self.input_shape,
            )
            return True
        except Exception as exc:
            LOG.warning("Failed to load Top Camera YOLOv8 feeding model: %s", exc)
            self.session = None
            return False

    def detect(self, frame) -> Dict[str, Any]:
        """Detect hungry fish using YOLOv8 ONNX model with contour fallback."""
        if frame is None:
            return {
                "hungry_fish_ids": [],
                "hungry_count": 0,
                "confidence": 0.0,
                "hunger_level": "Normal",
                "detections": [],
            }

        # 1. Attempt YOLOv8 ONNX inference
        if self._load_model():
            try:
                import cv2

                orig_h, orig_w = frame.shape[:2]
                target_h, target_w = self.input_shape

                # Pre-process: resize -> RGB -> CHW float32 / 255.0
                resized = cv2.resize(frame, (target_w, target_h))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                blob = np.ascontiguousarray(
                    rgb.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32) / 255.0
                )

                outputs = self.session.run(None, {self.input_name: blob})
                raw = outputs[0][0].T  # -> (num_anchors, 4 + num_classes)

                boxes_xywh = raw[:, :4]
                class_probs = raw[:, 4:]
                scores = class_probs.max(axis=1) if class_probs.shape[1] > 1 else class_probs[:, 0]

                # Filter by confidence threshold
                mask = scores >= self.confidence_threshold
                if mask.any():
                    boxes_xywh = boxes_xywh[mask]
                    scores = scores[mask]

                    # Convert cx,cy,w,h -> x1,y1,x2,y2
                    cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2
                    xyxy = np.stack([x1, y1, x2, y2], axis=1)

                    # NMS
                    keep = _nms(xyxy, scores, iou_threshold=0.45)
                    xyxy = xyxy[keep]
                    scores = scores[keep]

                    # Scale back to original frame dimensions
                    sx = orig_w / target_w
                    sy = orig_h / target_h
                    xyxy[:, [0, 2]] *= sx
                    xyxy[:, [1, 3]] *= sy

                    detections = []
                    for idx, box in enumerate(xyxy):
                        detections.append({
                            "bbox": [round(float(v), 2) for v in box],
                            "confidence": round(float(scores[idx]), 3),
                        })

                    hungry_count = min(len(detections), 4)
                    hungry_ids = [i + 1 for i in range(hungry_count)]
                    mean_conf = round(float(np.mean(scores[:hungry_count])), 3) if hungry_count > 0 else 0.0

                    hunger_level = (
                        "Normal" if hungry_count == 0
                        else ("Low" if hungry_count == 1
                              else ("Moderate" if hungry_count == 2
                                    else "High"))
                    )

                    return {
                        "hungry_fish_ids": hungry_ids,
                        "hungry_count": hungry_count,
                        "confidence": mean_conf,
                        "hunger_level": hunger_level,
                        "detections": detections[:4],
                        "source": "yolov8_onnx",
                    }
                else:
                    return {
                        "hungry_fish_ids": [],
                        "hungry_count": 0,
                        "confidence": 0.0,
                        "hunger_level": "Normal",
                        "detections": [],
                        "source": "yolov8_onnx",
                    }

            except Exception as exc:
                LOG.warning("YOLOv8 feeding detection failed: %s. Falling back to contour analysis.", exc)

        # 2. Fallback: Classical contour analysis of top feeding zone
        return self._contour_fallback(frame)

    def _contour_fallback(self, frame) -> Dict[str, Any]:
        """Classical CV contour analysis fallback when ONNX model is unavailable."""
        try:
            import cv2

            h, w = frame.shape[:2]
            # Define central top feeding area ROI (middle 50% width, top 60% height)
            roi = frame[0:int(h * 0.6), int(w * 0.25):int(w * 0.75)]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            fish_contours = [c for c in contours if 300 < cv2.contourArea(c) < 15000]
            hungry_count = min(len(fish_contours), 4)
            hungry_ids = [i + 1 for i in range(hungry_count)]
            confidence = round(float(np.clip(0.70 + (0.05 * hungry_count), 0.0, 0.95)), 2) if hungry_count > 0 else 0.0

            hunger_level = (
                "Normal" if hungry_count == 0
                else ("Low" if hungry_count == 1
                      else ("Moderate" if hungry_count == 2
                            else "High"))
            )

            return {
                "hungry_fish_ids": hungry_ids,
                "hungry_count": hungry_count,
                "confidence": confidence,
                "hunger_level": hunger_level,
                "detections": [],
                "source": "contour_fallback",
            }
        except Exception as exc:
            LOG.warning("Hunger contour fallback evaluation failed: %s", exc)
            return {
                "hungry_fish_ids": [],
                "hungry_count": 0,
                "confidence": 0.0,
                "hunger_level": "Normal",
                "detections": [],
                "source": "error",
                "error": str(exc),
            }


# Module-level singleton instance for direct function calls
_DETECTOR = HungerDetector()


def detect(frame) -> Dict[str, Any]:
    """Detect hungry fish from Top Camera 2 frame using YOLOv8 ONNX model."""
    return _DETECTOR.detect(frame)
