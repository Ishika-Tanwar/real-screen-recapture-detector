"""
app.py — Flask web app for live camera-based screen-recapture detection.

Usage:
    python app.py
Then open http://localhost:5000 in your browser.
"""

import io
import time
import base64
import numpy as np
import onnxruntime as ort
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = "model.onnx"
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name


def _preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def predict_image(img: Image.Image) -> float:
    input_tensor = _preprocess(img)
    logits = _session.run(None, {_input_name: input_tensor})[0]
    return float(_sigmoid(logits[0][0]))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    image_data = data["image"].split(",")[1]  # strip "data:image/jpeg;base64,"
    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes))

    start = time.perf_counter()
    score = predict_image(img)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return jsonify({
        "score": round(score, 4),
        "latency_ms": round(elapsed_ms, 1),
        "is_screen": score >= 0.5,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)