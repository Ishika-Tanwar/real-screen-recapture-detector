"""
export_onnx.py — Convert the trained PyTorch model to ONNX for faster CPU inference.

Usage:
    python export_onnx.py

Outputs:
    model.onnx
"""

import torch
import torch.nn as nn
from torchvision import models

MODEL_PATH = "model.pth"
ONNX_PATH  = "model.onnx"
IMG_SIZE   = 224

model = models.mobilenet_v2(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(in_features, 1),
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17,
)

print(f"Exported to {ONNX_PATH}")