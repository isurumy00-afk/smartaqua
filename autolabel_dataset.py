#!/usr/bin/env python3
"""Auto-Labeling Script for YOLOv8 Custom Dataset.

Reads captured images in dataset/images/, detects fish targets using the pre-trained
ONNX model (or contour motion detection), and generates YOLO-formatted label files (.txt)
in dataset/labels/ along with a dataset.yaml file ready for YOLOv8 retraining.

YOLO Label Format per line:
    <class_id> <center_x> <center_y> <width> <height>  (all normalized 0.0 - 1.0)
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Paths
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

IMAGES_DIR = BASE_DIR / "dataset" / "images"
LABELS_DIR = BASE_DIR / "dataset" / "labels"
PREVIEW_DIR = BASE_DIR / "dataset" / "annotated_preview"
YAML_PATH = BASE_DIR / "dataset" / "dataset.yaml"

LABELS_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

# Import system vision components
from vision.fish_tracker import FishTracker
FISH_TRACKER = FishTracker()


def convert_to_yolo_bbox(bbox, img_width, img_height, class_id=0):
    """Convert [x1, y1, x2, y2] to YOLO normalized format: class_id cx cy w h."""
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
    """Contour-based fish detection fallback when ONNX model threshold is low."""
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
            # Filter unrealistic aspect ratios
            aspect = bw / float(bh)
            if 0.3 < aspect < 3.5:
                boxes.append([x, y, x + bw, y + bh])

    return boxes


def create_dataset_yaml():
    """Create dataset.yaml required for YOLOv8 training."""
    yaml_content = f"""# Custom Aquarium Fish Dataset Configuration for YOLOv8
path: {BASE_DIR / 'dataset'}
train: images
val: images

names:
  0: fish
"""
    YAML_PATH.write_text(yaml_content, encoding="utf-8")
    print(f"[OK] Generated dataset configuration: {YAML_PATH}")


def main():
    print("=" * 60)
    print(" Smart Aquarium — YOLOv8 Automated Dataset Labeler")
    print("=" * 60)
    print(f" Source Images : {IMAGES_DIR}")
    print(f" Output Labels : {LABELS_DIR}")
    print(f" Preview Folder: {PREVIEW_DIR}")
    print("=" * 60)

    image_files = sorted(list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png")))
    if not image_files:
        print(f"[ERROR] No image files found in {IMAGES_DIR}.")
        print("Please run collect_images.py first to capture dataset images.")
        return

    print(f"[INFO] Found {len(image_files)} images to auto-label...")
    labeled_count = 0
    total_boxes = 0

    for idx, img_path in enumerate(image_files, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        vis = frame.copy()

        # 1. Run ONNX Model tracking
        tracks = FISH_TRACKER.track(frame)
        boxes = [t["bbox"] for t in tracks if "bbox" in t]

        # 2. If ONNX model found no detections, use contour detection fallback
        if not boxes:
            boxes = fallback_contour_detect(frame)

        yolo_lines = []
        for box in boxes:
            yolo_str = convert_to_yolo_bbox(box, w, h, class_id=0)
            if yolo_str:
                yolo_lines.append(yolo_str)
                total_boxes += 1

                # Draw green bounding box on preview frame
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, "fish", (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write YOLO format .txt file
        label_filename = img_path.stem + ".txt"
        label_path = LABELS_DIR / label_filename
        label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        # Write annotated preview image
        preview_path = PREVIEW_DIR / img_path.name
        cv2.imwrite(str(preview_path), vis)

        labeled_count += 1
        print(f" [{idx}/{len(image_files)}] Auto-labeled {img_path.name} -> {len(yolo_lines)} fish boxes")

    create_dataset_yaml()

    print("=" * 60)
    print(" AUTO-LABELING COMPLETED SUCCESSFULLY!")
    print(f" Images Processed : {labeled_count}")
    print(f" Total Fish BBoxes: {total_boxes}")
    print(f" Labels Folder    : {LABELS_DIR}")
    print(f" Verification Folder: {PREVIEW_DIR}")
    print("=" * 60)
    print("\nNext Step for Retraining YOLOv8:")
    print(" Run YOLOv8 fine-tuning with your dataset:")
    print(f"   yolo detect train data={YAML_PATH} model=yolov8n.pt epochs=50 imgsz=640")
    print("=" * 60)


if __name__ == "__main__":
    main()
