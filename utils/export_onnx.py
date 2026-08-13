#!/usr/bin/env python3
"""ONNX Export Helper — one-time model conversion script.

Run this script on a development machine (NOT on the Pi) to convert:
  1. YOLOv8 fish detector  : models/vision/fish_detector_yolov8.pt   → models/vision/fish_detector_yolov8.onnx
  2. TFLite disease model  : models/disease/fish_disease_model.tflite → models/disease/fish_disease_model.onnx

Requirements (dev machine only, NOT needed on Pi):
    pip install ultralytics onnx onnxruntime tf2onnx tensorflow

Usage:
    python utils/export_onnx.py
    python utils/export_onnx.py --yolo-only
    python utils/export_onnx.py --disease-only
    python utils/export_onnx.py --imgsz 640
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VISION_MODEL_DIR = BASE_DIR / "models" / "vision"
DISEASE_MODEL_DIR = BASE_DIR / "models" / "disease"

PT_PATH    = VISION_MODEL_DIR / "fish_detector_yolov8.pt"
YOLO_ONNX  = VISION_MODEL_DIR / "fish_detector_yolov8.onnx"

TFLITE_PATH    = DISEASE_MODEL_DIR / "fish_disease_model.tflite"
DISEASE_ONNX   = DISEASE_MODEL_DIR / "fish_disease_model.onnx"


def export_yolo(imgsz: int = 640) -> bool:
    """Export YOLOv8 .pt → .onnx using ultralytics."""
    if not PT_PATH.exists():
        print(f"[ERROR] YOLOv8 source model not found: {PT_PATH}")
        return False

    print(f"[INFO] Exporting YOLOv8 model: {PT_PATH} → {YOLO_ONNX}")
    try:
        from ultralytics import YOLO
        model = YOLO(str(PT_PATH))
        # Export to ONNX — opset 17 is widely supported by onnxruntime ≥ 1.16
        model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True)
        # ultralytics saves alongside the .pt file — move if needed
        candidate = PT_PATH.with_suffix(".onnx")
        if candidate.exists() and candidate != YOLO_ONNX:
            candidate.rename(YOLO_ONNX)
        if YOLO_ONNX.exists():
            print(f"[OK] YOLOv8 ONNX model saved: {YOLO_ONNX}")
            return True
        print("[WARN] ONNX export finished but output file not found at expected path.")
        return False
    except Exception as exc:
        print(f"[ERROR] YOLOv8 ONNX export failed: {exc}")
        return False


def export_disease_tflite(imgsz: int = 224) -> bool:
    """Convert TFLite disease model → ONNX using tf2onnx."""
    if not TFLITE_PATH.exists():
        print(f"[ERROR] TFLite disease model not found: {TFLITE_PATH}")
        return False

    print(f"[INFO] Converting TFLite model: {TFLITE_PATH} → {DISEASE_ONNX}")
    try:
        import tf2onnx
        import tensorflow as tf

        # Load TFLite model to inspect input shape
        interp = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        shape = inp["shape"].tolist()  # e.g. [1, 224, 224, 3]
        h, w = shape[1], shape[2]
        print(f"[INFO] Detected input shape: {shape}")

        # Convert via tf2onnx CLI wrapper
        import subprocess
        cmd = [
            sys.executable, "-m", "tf2onnx.convert",
            "--tflite", str(TFLITE_PATH),
            "--output", str(DISEASE_ONNX),
            "--opset", "17",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] tf2onnx conversion failed:\n{result.stderr}")
            return False

        print(f"[OK] Disease ONNX model saved: {DISEASE_ONNX}")
        return True
    except ImportError as exc:
        print(f"[ERROR] Missing dependency for TFLite→ONNX conversion: {exc}")
        print("       Install with: pip install tf2onnx tensorflow")
        return False
    except Exception as exc:
        print(f"[ERROR] TFLite→ONNX conversion failed: {exc}")
        return False


def verify_onnx(path: Path) -> bool:
    """Quick sanity check that the exported ONNX file is valid."""
    try:
        import onnx
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        print(f"[OK] ONNX model verified: {path.name}")
        return True
    except ImportError:
        print("[WARN] onnx package not installed — skipping model verification.")
        return True
    except Exception as exc:
        print(f"[ERROR] ONNX verification failed for {path}: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export ML models to ONNX format for Pi 4B deployment.")
    parser.add_argument("--yolo-only",    action="store_true", help="Export only the YOLOv8 fish detector")
    parser.add_argument("--disease-only", action="store_true", help="Export only the disease classifier")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for YOLOv8 export (default: 640)")
    args = parser.parse_args()

    export_yolo_flag    = not args.disease_only
    export_disease_flag = not args.yolo_only

    print("=" * 60)
    print(" Smart Aquarium — ONNX Model Export Utility")
    print("=" * 60)

    results = {}

    if export_yolo_flag:
        ok = export_yolo(imgsz=args.imgsz)
        if ok:
            verify_onnx(YOLO_ONNX)
        results["YOLOv8 fish detector"] = ok

    if export_disease_flag:
        ok = export_disease_tflite()
        if ok:
            verify_onnx(DISEASE_ONNX)
        results["Disease classifier"] = ok

    print("\n" + "=" * 60)
    print(" Export Summary")
    print("=" * 60)
    all_ok = True
    for name, status in results.items():
        icon = "[OK]  " if status else "[FAIL]"
        print(f"  {icon} {name}")
        if not status:
            all_ok = False

    if all_ok:
        print("\n  All models exported successfully.")
        print("  Copy the .onnx files to the Raspberry Pi under models/vision/ and models/disease/")
    else:
        print("\n  Some exports failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
