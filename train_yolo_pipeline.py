#!/usr/bin/env python3
"""All-in-One YOLOv8 Dataset Collection, Auto-Labeling, Training, and ONNX Conversion Pipeline.

Designed for Windows PC.

Pipeline Stages:
  1. Image Collection : Captures 100 photos from USB Camera 0 into dataset/images/
  2. Auto-Labeling   : Generates YOLO .txt label files in dataset/labels/ & dataset.yaml
  3. YOLOv8 Training : Fine-tunes yolov8n.pt for custom fish detection (50 epochs)
  4. ONNX Conversion : Exports trained model to ONNX format & updates models/vision/fish_detector_yolov8.onnx
"""

import os
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

# System Paths
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
PREVIEW_DIR = DATASET_DIR / "annotated_preview"
YAML_PATH = DATASET_DIR / "dataset.yaml"
OUTPUT_ONNX_PATH = BASE_DIR / "models" / "vision" / "fish_detector_yolov8.onnx"

# Ensure runtime directories exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

CAMERA_INDEX = 0
TARGET_IMAGE_COUNT = 100
AUTO_CAPTURE_INTERVAL = 5.0  # seconds between auto snapshots


# ---------------------------------------------------------------------------
# STAGE 1: Image Collection from USB Camera 0
# ---------------------------------------------------------------------------

def collect_images() -> bool:
    print("\n" + "=" * 70)
    print(" STAGE 1: USB Camera Image Collection (100 Images)")
    print("=" * 70)
    print(" Controls:")
    print("   [SPACE] : Snap single image manually")
    print("   [A]     : Toggle Auto-Capture mode (1 photo / 5 sec)")
    print("   [Q/ESC] : Skip to training with current images")
    print("=" * 70)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap or not cap.isOpened():
        if os.name == "nt":
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap or not cap.isOpened():
        print(f"[ERROR] Could not open USB Camera at index {CAMERA_INDEX}.")
        print("Falling back to existing images in dataset/images/...")
        existing_imgs = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
        return len(existing_imgs) > 0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    existing = list(IMAGES_DIR.glob("img_*.jpg"))
    saved_count = len(existing)
    print(f"[INFO] Currently {saved_count} images in {IMAGES_DIR}")

    auto_mode = False
    last_auto_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        now = time.time()
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Auto-capture logic
        if auto_mode and (now - last_auto_time >= AUTO_CAPTURE_INTERVAL):
            saved_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            filename = f"img_{saved_count:03d}_{timestamp}.jpg"
            filepath = IMAGES_DIR / filename
            cv2.imwrite(str(filepath), frame)
            print(f" [{saved_count}/{TARGET_IMAGE_COUNT}] Saved (Auto): {filepath.name}")
            last_auto_time = now

        # Overlay Banner
        cv2.rectangle(vis, (0, 0), (w, 50), (30, 30, 30), -1)
        progress_text = f"Collected: {saved_count} / {TARGET_IMAGE_COUNT} images"
        progress_color = (0, 255, 0) if saved_count >= TARGET_IMAGE_COUNT else (0, 255, 255)
        cv2.putText(vis, progress_text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, progress_color, 2)

        mode_text = "AUTO (1/5s)" if auto_mode else "MANUAL"
        mode_color = (0, 255, 0) if auto_mode else (200, 200, 200)
        cv2.putText(vis, mode_text, (w - 160, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

        cv2.rectangle(vis, (0, h - 35), (w, h), (20, 20, 20), -1)
        cv2.putText(vis, "SPACE: Snap | A: Toggle Auto | Q: Proceed to Labeling",
                    (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        cv2.imshow("Stage 1: Image Collection", vis)

        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):
            print("[INFO] Collection finished by user.")
            break
        elif key == ord(' '):
            saved_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            filename = f"img_{saved_count:03d}_{timestamp}.jpg"
            filepath = IMAGES_DIR / filename
            cv2.imwrite(str(filepath), frame)
            print(f" [{saved_count}/{TARGET_IMAGE_COUNT}] Saved (Manual): {filepath.name}")

            flash = vis.copy()
            cv2.rectangle(flash, (0, 0), (w, h), (0, 255, 0), 10)
            cv2.imshow("Stage 1: Image Collection", flash)
            cv2.waitKey(50)

        elif key == ord('a') or key == ord('A'):
            auto_mode = not auto_mode
            status = "ENABLED (1 photo / 5 sec)" if auto_mode else "DISABLED"
            print(f"[MODE] Auto-Capture {status}")
            last_auto_time = time.time()

        if saved_count >= TARGET_IMAGE_COUNT:
            print(f"[SUCCESS] Target reached! {saved_count} images collected.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved_count > 0


# ---------------------------------------------------------------------------
# STAGE 2: Automated Dataset Labeling
# ---------------------------------------------------------------------------

def convert_to_yolo_bbox(bbox, img_width, img_height, class_id=0):
    """Convert bounding box [x1, y1, x2, y2] to YOLO normalized string format."""
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_width, x2), min(img_height, y2)

    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None

    cx = (x1 + box_w / 2.0) / img_width
    cy = (y1 + box_h / 2.0) / img_height
    norm_w = box_w / img_width
    norm_h = box_h / img_height

    return f"{class_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}"


def fallback_contour_detect(frame, min_area=300):
    """Adaptive contour detection fallback for unlabelled fish targets."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = frame.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < (w * h * 0.4):
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(bh)
            if 0.3 < aspect < 3.5:
                boxes.append([x, y, x + bw, y + bh])

    return boxes


def autolabel_dataset() -> bool:
    print("\n" + "=" * 70)
    print(" STAGE 2: Automated Dataset Labeling & YAML Generation")
    print("=" * 70)

    image_files = sorted(list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png")))
    if not image_files:
        print("[ERROR] No dataset images found.")
        return False

    try:
        from vision.fish_tracker import FishTracker
        tracker = FishTracker()
    except Exception:
        tracker = None

    total_boxes = 0

    for idx, img_path in enumerate(image_files, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        vis = frame.copy()

        # Run detector / tracker
        boxes = []
        if tracker:
            tracks = tracker.track(frame)
            boxes = [t["bbox"] for t in tracks if "bbox" in t]

        if not boxes:
            boxes = fallback_contour_detect(frame)

        yolo_lines = []
        for box in boxes:
            yolo_str = convert_to_yolo_bbox(box, w, h, class_id=0)
            if yolo_str:
                yolo_lines.append(yolo_str)
                total_boxes += 1
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, "fish", (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save label file (.txt)
        label_path = LABELS_DIR / (img_path.stem + ".txt")
        label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        # Save preview image
        cv2.imwrite(str(PREVIEW_DIR / img_path.name), vis)

    # Generate dataset.yaml (using relative paths for Windows/Linux compatibility)
    yaml_content = f"""# Custom Aquarium Fish Dataset for YOLOv8
path: {DATASET_DIR.resolve().as_posix()}
train: images
val: images

names:
  0: fish
"""
    YAML_PATH.write_text(yaml_content, encoding="utf-8")
    print(f"[OK] Auto-labeled {len(image_files)} images ({total_boxes} total fish bboxes).")
    print(f"[OK] Generated dataset YAML: {YAML_PATH}")
    return True


# ---------------------------------------------------------------------------
# STAGE 3 & 4: YOLOv8 Training & ONNX Export
# ---------------------------------------------------------------------------

def train_and_export_onnx(epochs: int = 50, imgsz: int = 640) -> bool:
    print("\n" + "=" * 70)
    print(f" STAGE 3 & 4: Fine-Tuning YOLOv8 (yolov8n.pt) & Exporting to ONNX")
    print("=" * 70)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics package is not installed.")
        print("Please run: pip install ultralytics onnx onnxruntime")
        return False

    print(f"[INFO] Loading base model: yolov8n.pt")
    model = YOLO("yolov8n.pt")

    print(f"[INFO] Starting YOLOv8 fine-tuning ({epochs} epochs, imgsz={imgsz})...")
    train_results = model.train(
        data=str(YAML_PATH),
        epochs=epochs,
        imgsz=imgsz,
        project=str(BASE_DIR / "runs" / "train"),
        name="fish_yolov8_custom",
        exist_ok=True,
        verbose=True,
    )

    print(f"\n[INFO] Exporting fine-tuned model to ONNX format (opset 17)...")
    exported_onnx_path = model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True)
    exported_onnx_path = Path(exported_onnx_path)

    if exported_onnx_path and exported_onnx_path.exists():
        shutil.copy2(exported_onnx_path, OUTPUT_ONNX_PATH)
        print(f"\n" + "=" * 70)
        print(f" SUCCESS: Fine-tuned YOLOv8 ONNX model ready!")
        print(f" Saved at: {OUTPUT_ONNX_PATH}")
        print(f" Size    : {OUTPUT_ONNX_PATH.stat().st_size / (1024*1024):.2f} MB")
        print("=" * 70)
        return True
    else:
        print("[ERROR] ONNX export failed.")
        return False


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print(" Smart Aquarium — Complete YOLOv8 Windows Training Pipeline")
    print("=" * 70)

    # Stage 1: Image Collection
    #if not collect_images():
        #print("[ERROR] Stage 1 failed. Aborting pipeline.")
        #return

    # Stage 2: Auto-Labeling
    if not autolabel_dataset():
        print("[ERROR] Stage 2 failed. Aborting pipeline.")
        return

    # Stage 3 & 4: Training & ONNX Export
    train_and_export_onnx(epochs=50, imgsz=640)


if __name__ == "__main__":
    main()
