"""Classify one image on a Raspberry Pi 4B using TensorFlow Lite.

Put these files beside this script:
  fish_disease_model.tflite
  class_names.json
  test_image.jpg

Then run: python3 local_test_model.py
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

MODEL_FILE = Path("fish_disease_model.tflite")
CLASS_NAMES_FILE = Path("class_names.json")
TEST_IMAGE_FILE = Path("test_image.jpg")

if not MODEL_FILE.exists() or not CLASS_NAMES_FILE.exists() or not TEST_IMAGE_FILE.exists():
    raise FileNotFoundError(
        "Keep fish_disease_model.tflite, class_names.json, and test_image.jpg beside this script."
    )

with open(CLASS_NAMES_FILE, encoding="utf-8") as file:
    class_names = json.load(file)

# Pi 4B has four CPU cores; use all four for the convolution layers.
interpreter = tflite.Interpreter(model_path=str(MODEL_FILE), num_threads=4)
interpreter.allocate_tensors()
input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]
height, width = input_info["shape"][1:3]

image = Image.open(TEST_IMAGE_FILE).convert("RGB").resize((width, height))
input_data = np.expand_dims(np.asarray(image), axis=0)

# The exported dynamic-range model accepts float32. This also supports a fully
# quantized model if one is exported later.
if input_info["dtype"] == np.float32:
    input_data = input_data.astype(np.float32)
else:
    scale, zero_point = input_info["quantization"]
    input_data = np.round(input_data / scale + zero_point).astype(input_info["dtype"])

interpreter.set_tensor(input_info["index"], input_data)
interpreter.invoke()
probabilities = interpreter.get_tensor(output_info["index"])[0]

if output_info["dtype"] != np.float32:
    scale, zero_point = output_info["quantization"]
    probabilities = (probabilities.astype(np.float32) - zero_point) * scale

predicted_index = int(np.argmax(probabilities))
print(f"Predicted class: {class_names[predicted_index]}")
print(f"Confidence: {probabilities[predicted_index] * 100:.2f}%")
