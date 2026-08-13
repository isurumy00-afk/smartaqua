"""YOLO fish-behaviour monitor with selectable ByteTrack or BoT-SORT + ReID."""

import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "model.pt"
VIDEO_PATH = "side1.mp4"

# Choose ``"bytetrack"`` for the tuned tracker requested below, or ``"botsort"``
# when fish frequently overlap and retaining identity is more important than speed.
TRACKER_MODE = "bytetrack"
TRACKER_CONFIGS = {
    "bytetrack": "bytetrack_fish.yaml",
    "botsort": "botsort_fish_reid.yaml",
}
TRACKER_CONFIDENCE = 0.20
TRACKER_IOU = 0.50

TOP_REGION_PERCENT = 0.30
BOTTOM_REGION_PERCENT = 0.25
FREEZE_SPEED_THRESHOLD = 5.0
# Classify high-speed swimming only after it persists continuously long enough
# to be behaviour rather than a one-frame tracking jump.
ABNORMAL_SPEED_THRESHOLD = 100.0       # pixels/second
ABNORMAL_SPEED_DURATION = 3.0          # seconds at/above the threshold
SHOW_WINDOW = True


def classify_stress(top_time, bottom_time, freeze_time, mean_speed, crossings,
                    longest_bottom, surface_visits, total_time,
                    current_region=None, current_speed=None,
                    high_speed_duration=0.0):
    if total_time <= 0:
        return 0.0, "Healthy", (0, 255, 0), "Normal"

    bottom_ratio, top_ratio = bottom_time / total_time, top_time / total_time
    # For the live overlay, bottom dwelling and speed are current behaviours,
    # not lifetime labels.  The optional arguments keep the final CSV summary
    # based on the complete track.
    bottom_score = min(bottom_ratio / 0.70, 1.0)
    if current_region is not None and current_region != "bottom":
        bottom_score = 0.0
    speed_for_status = mean_speed if current_speed is None else current_speed
    speed_score = min(abs(speed_for_status - 40.0) / 40.0, 1.0)
    high_speed = high_speed_duration >= ABNORMAL_SPEED_DURATION
    components = {
        "Bottom Dwelling": 0.22 * bottom_score,
        "Freezing": 0.22 * min(freeze_time / 20.0, 1.0),
        "Abnormal Speed": 0.16 * speed_score,
        "Erratic Swimming": 0.10 * min(crossings / 15.0, 1.0),
        "Low Surface Activity": 0.10 * (1 - min(top_ratio / 0.30, 1.0)),
        "Prolonged Bottom Stay": 0.12 * min(longest_bottom / 30.0, 1.0),
        "Frequent Surfacing": 0.08 * min(surface_visits / 20.0, 1.0),
    }
    score = max(0.0, min(sum(components.values()), 1.0))
    reason = max(components, key=components.get)
    # A sustained fast measurement must not be hidden by the weighted score.
    if high_speed:
        return max(score, 0.60), "High Stress", (0, 0, 255), "Abnormal Speed"
    if score < 0.30:
        return score, "Healthy", (0, 255, 0), "Normal"
    if score < 0.60:
        return score, "Mild Stress", (0, 255, 255), reason
    return score, "High Stress", (0, 0, 255), reason


def make_state():
    return {"last": None, "dist": 0.0, "cross": 0, "top": 0.0,
            "bottom": 0.0, "freeze": 0.0, "tracked": 0.0,
            "last_region": "middle", "current_bottom": 0.0,
            "longest_bottom": 0.0, "surface_visits": 0,
            "high_speed_duration": 0.0}


def classify_tank_stress(scores):
    """Return a live whole-tank score from the fish visible in this frame."""
    if not scores:
        return 0.0, "No fish detected", (180, 180, 180)
    score = float(np.mean(scores))
    if score < 0.30:
        return score, "Healthy", (0, 255, 0)
    if score < 0.60:
        return score, "Mild Stress", (0, 255, 255)
    return score, "High Stress", (0, 0, 255)


def main():
    if TRACKER_MODE not in TRACKER_CONFIGS:
        raise ValueError(f"TRACKER_MODE must be one of {tuple(TRACKER_CONFIGS)}")
    tracker = Path(__file__).with_name(TRACKER_CONFIGS[TRACKER_MODE])
    if not tracker.is_file():
        raise FileNotFoundError(f"Tracker configuration not found: {tracker}")

    model = YOLO(MODEL_PATH)
    fps, frame_no, state = 30.0, 0, {}
    top_line = bottom_line = width = None

    # Requested tracking arguments. ``persist=True`` keeps IDs between frames.
    results = model.track(
        source=VIDEO_PATH,
        stream=True,
        persist=True,
        tracker=str(tracker),
        conf=TRACKER_CONFIDENCE,
        iou=TRACKER_IOU,
        verbose=False,
    )
    for result in results:
        frame, frame_no = result.orig_img, frame_no + 1
        if top_line is None:
            height, width = frame.shape[:2]
            top_line = int(height * TOP_REGION_PERCENT)
            bottom_line = int(height * (1 - BOTTOM_REGION_PERCENT))
        fish_scores = []

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid in zip(boxes, ids):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                s = state.setdefault(tid, make_state())
                speed = math.dist(s["last"], (cx, cy)) * fps if s["last"] else 0.0
                s["dist"] += speed / fps
                s["tracked"] += 1 / fps
                if speed < FREEZE_SPEED_THRESHOLD:
                    s["freeze"] += 1 / fps
                if speed >= ABNORMAL_SPEED_THRESHOLD:
                    s["high_speed_duration"] += 1 / fps
                else:
                    s["high_speed_duration"] = 0.0

                if cy < top_line:
                    region, s["top"], s["current_bottom"] = "top", s["top"] + 1 / fps, 0.0
                elif cy > bottom_line:
                    region = "bottom"
                    s["bottom"] += 1 / fps
                    s["current_bottom"] += 1 / fps
                    s["longest_bottom"] = max(s["longest_bottom"], s["current_bottom"])
                else:
                    region, s["current_bottom"] = "middle", 0.0
                previous = s["last_region"]
                if region != previous and region != "middle" and previous != "middle":
                    s["cross"] += 1
                if previous != "top" and region == "top":
                    s["surface_visits"] += 1
                s["last_region"], s["last"] = region, (cx, cy)

                mean_speed = s["dist"] / max(s["tracked"], 1e-6)
                score, label, color, reason = classify_stress(
                    s["top"], s["bottom"], s["freeze"], mean_speed, s["cross"],
                    s["longest_bottom"], s["surface_visits"], s["tracked"],
                    current_region=region, current_speed=speed,
                    high_speed_duration=s["high_speed_duration"])
                fish_scores.append(score)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                for text, y in ((f"ID {tid} | {speed:.1f} px/s", y1 - 65),
                                (f"{label} ({score:.2f})", y1 - 45),
                                (reason, y1 - 25)):
                    cv2.putText(frame, text, (int(x1), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.50, color, 2)

        tank_score, tank_label, tank_color = classify_tank_stress(fish_scores)
        cv2.line(frame, (0, top_line), (width, top_line), (255, 0, 0), 2)
        cv2.line(frame, (0, bottom_line), (width, bottom_line), (0, 0, 255), 2)
        cv2.putText(frame, f"Tank Stress: {tank_label} ({tank_score:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, tank_color, 2)
        if SHOW_WINDOW:
            cv2.imshow("Fish Behaviour", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
