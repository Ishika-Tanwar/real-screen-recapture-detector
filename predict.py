# """
# predict.py — Detect whether an image is a real photo or a photo of a screen.

# Usage:
#     python predict.py some_image.jpg

# Prints ONE number from 0 to 1:
#     0 = real photo
#     1 = photo of a screen (recapture / fraud)
# """

# import sys
# import time
# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# from PIL import Image

# # ── Config ────────────────────────────────────────────────────────────────────
# MODEL_PATH = "model.pth"   # produced by train.py
# IMG_SIZE   = 224

# # ── Device ────────────────────────────────────────────────────────────────────
# device = (
#     "cuda" if torch.cuda.is_available()
#     else "mps" if torch.backends.mps.is_available()
#     else "cpu"
# )

# # ── Load model (once, at import time) ─────────────────────────────────────────
# def _load_model():
#     model = models.mobilenet_v2(weights=None)
#     in_features = model.classifier[1].in_features
#     model.classifier = nn.Sequential(
#         nn.Dropout(0.2),
#         nn.Linear(in_features, 1),
#     )
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.eval()
#     model.to(device)
#     return model

# _model = _load_model()

# # ── Transform ─────────────────────────────────────────────────────────────────
# _transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225]),
# ])

# # ── Predict ───────────────────────────────────────────────────────────────────
# def predict(image_path: str) -> float:
#     img    = Image.open(image_path).convert("RGB")
#     tensor = _transform(img).unsqueeze(0).to(device)

#     with torch.no_grad():
#         logit = _model(tensor)
#         score = torch.sigmoid(logit).item()

#     return score


# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python predict.py <image_path>")
#         sys.exit(1)

#     image_path = sys.argv[1]

#     start = time.perf_counter()
#     score = predict(image_path)
#     elapsed_ms = (time.perf_counter() - start) * 1000

#     print(f"{score:.4f}")
#     print(f"# latency: {elapsed_ms:.1f} ms on {device}", file=sys.stderr)

"""
predict.py — Detect whether an image is a real photo or a photo of a screen.
Uses ONNX Runtime for fast CPU inference.

Usage:
    python predict.py some_image.jpg
"""

import sys
import time
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = "model.onnx"
IMG_SIZE   = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name


def _preprocess(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def predict(image_path: str) -> float:
    input_tensor = _preprocess(image_path)
    logits = _session.run(None, {_input_name: input_tensor})[0]
    return float(_sigmoid(logits[0][0]))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    start = time.perf_counter()
    score = predict(image_path)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"{score:.4f}")
    print(f"# latency: {elapsed_ms:.1f} ms on cpu (ONNX Runtime)", file=sys.stderr)