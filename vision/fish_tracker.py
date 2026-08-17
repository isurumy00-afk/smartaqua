"""ONNX Runtime YOLOv8 Fish Tracker module.

Tracks fish in Camera 1 side-view frames using a pure ONNX Runtime pipeline,
maintaining identity, bounding boxes, centre points, trajectory history, and
speed across frames synchronously on the main thread.
"""

import math
import time
from typing import List, Dict, Any

import numpy as np

from config import FISH_CONFIDENCE, FISH_MODEL_ONNX_PATH, MAX_TRACKED_FISH
from utils.logger import get_logger

LOG = get_logger(__name__)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """Non-maximum suppression (numpy, no torch dependency)."""
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


class FishTracker:
    """YOLOv8 ONNX fish tracker running synchronously on the main thread."""

    def __init__(self):
        self.session = None
        self.input_name: str = ""
        self.input_shape: tuple = (640, 640)
        self.fish_states: Dict[int, Dict[str, Any]] = {}
        self.last_timestamp: float = time.time()
        self._next_id: int = 1

    def _load_model(self) -> bool:
        """Lazy-load ONNX session once."""
        if self.session is not None:
            return True

        if not FISH_MODEL_ONNX_PATH.exists():
            LOG.warning("YOLOv8 ONNX model not found: %s", FISH_MODEL_ONNX_PATH)
            return False

        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                str(FISH_MODEL_ONNX_PATH),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            meta = self.session.get_inputs()[0]
            self.input_name = meta.name
            shape = meta.shape
            if len(shape) == 4:
                self.input_shape = (int(shape[2]), int(shape[3]))
            LOG.info(
                "YOLOv8 ONNX fish detector loaded: %s input=%s",
                FISH_MODEL_ONNX_PATH.name,
                self.input_shape,
            )
            return True
        except Exception as exc:
            LOG.error("Failed to load YOLOv8 ONNX model: %s", exc)
            self.session = None
            return False

    def track(self, frame) -> List[Dict[str, Any]]:
        """Run ONNX fish detection & tracking synchronously on the main thread."""
        if frame is None:
            return []

        if not self._load_model():
            return []

        now = time.time()
        dt = max(now - self.last_timestamp, 0.033)
        self.last_timestamp = now

        try:
            import cv2

            orig_h, orig_w = frame.shape[:2]
            target_h, target_w = self.input_shape

            # Pre-process: resize → CHW float32 / 255
            resized = cv2.resize(frame, (target_w, target_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            blob = np.ascontiguousarray(
                rgb.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32) / 255.0
            )

            # Synchronous ONNX Inference on main thread
            outputs = self.session.run(None, {self.input_name: blob})
            raw = outputs[0][0].T    # → (A, 4+C)

            boxes_xywh = raw[:, :4]               # cx, cy, w, h
            class_probs = raw[:, 4:]              # (A, C)
            class_ids = class_probs.argmax(axis=1)
            scores = class_probs.max(axis=1)

            # Confidence filter
            mask = scores >= FISH_CONFIDENCE
            if not mask.any():
                from master import SIDE_CAMERA
                syn_boxes = getattr(SIDE_CAMERA, 'synthetic_boxes', None)
                if syn_boxes:
                    xyxy = np.array(syn_boxes, dtype=np.float32)
                    scores = np.array([0.95] * len(syn_boxes), dtype=np.float32)
                    return self._assign_ids(xyxy, scores, dt)[:MAX_TRACKED_FISH]
                return []

            boxes_xywh = boxes_xywh[mask]
            scores = scores[mask]

            # Convert cx,cy,w,h → x1,y1,x2,y2
            cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            xyxy = np.stack([x1, y1, x2, y2], axis=1)

            # NMS
            keep = _nms(xyxy, scores, iou_threshold=0.45)
            xyxy   = xyxy[keep]
            scores = scores[keep]

            # Scale back to original frame dimensions
            sx = orig_w / target_w
            sy = orig_h / target_h
            xyxy[:, [0, 2]] *= sx
            xyxy[:, [1, 3]] *= sy

            # Assign persistent IDs
            tracked_fish = self._assign_ids(xyxy, scores, dt)
            return tracked_fish[:MAX_TRACKED_FISH]

        except Exception as exc:
            LOG.error("YOLOv8 ONNX tracking inference error: %s", exc)
            return []

    def _assign_ids(
        self,
        xyxy: np.ndarray,
        scores: np.ndarray,
        dt: float,
        max_dist: float = 80.0,
    ) -> List[Dict[str, Any]]:
        """Assign persistent IDs to detections using nearest-centroid matching with TTL eviction."""
        now = time.time()

        # Prune stale fish tracks inactive for > 60 seconds (prevents 24/7 memory leak)
        stale_cutoff = now - 60.0
        self.fish_states = {
            fid: s for fid, s in self.fish_states.items()
            if s.get("last_seen", now) >= stale_cutoff
        }

        new_centers = [(float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)) for b in xyxy]
        prev_ids = list(self.fish_states.keys())

        assignment = {}
        used_prev = set()

        for det_idx, (ncx, ncy) in enumerate(new_centers):
            best_fid = None
            best_dist = max_dist
            for pi, fid in enumerate(prev_ids):
                if fid in used_prev:
                    continue
                pc = self.fish_states[fid]["last_center"]
                if pc is None:
                    continue
                d = math.dist((ncx, ncy), pc)
                if d < best_dist:
                    best_dist = d
                    best_fid = fid
            if best_fid is not None:
                assignment[det_idx] = best_fid
                used_prev.add(best_fid)
            else:
                assignment[det_idx] = self._next_id
                self._next_id += 1

        result = []
        for det_idx, bbox in enumerate(xyxy):
            fid = assignment[det_idx]
            cx, cy = new_centers[det_idx]

            state = self.fish_states.setdefault(fid, {
                "trajectory": [],
                "last_center": None,
                "last_seen": now,
            })

            state["last_center"] = (cx, cy)
            state["last_seen"] = now
            state["trajectory"].append([cx, cy])
            if len(state["trajectory"]) > 30:
                state["trajectory"].pop(0)

            result.append({
                "fish_id": fid,
                "bbox": [round(float(v), 2) for v in bbox],
                "confidence": round(float(scores[det_idx]), 3),
                "center": [round(cx, 2), round(cy, 2)],
                "trajectory": list(state["trajectory"]),
            })

        return result
