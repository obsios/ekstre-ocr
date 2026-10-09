"""Download and convert OCR models into the image during docker build."""

import os

import numpy as np

os.environ["OCR_DEVICE"] = "cpu"
os.environ["ALLOW_CPU_FALLBACK"] = "1"

from ocr_engine import PaddleOcrService  # noqa: E402


service = PaddleOcrService(device="cpu")
# Constructor initialization downloads the models. A tiny inference also proves
# that the cached ONNX artifacts are complete before the image is published.
service.recognize(np.full((64, 256, 3), 255, dtype=np.uint8), 1)
print(f"PaddleOCR models cached in {service.load_seconds}s")

