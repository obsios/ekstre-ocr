from __future__ import annotations

import os
import threading
import time
from typing import Any

import numpy as np
import onnxruntime as ort
from paddleocr import PaddleOCR


DETECTION_MODEL = os.getenv("OCR_DETECTION_MODEL", "PP-OCRv6_small_det")
RECOGNITION_MODEL = os.getenv("OCR_RECOGNITION_MODEL", "PP-OCRv6_small_rec")
RECOGNITION_BATCH_SIZE = int(os.getenv("OCR_RECOGNITION_BATCH_SIZE", "8"))
OCR_DEVICE = os.getenv("OCR_DEVICE", "gpu:0")
ALLOW_CPU_FALLBACK = os.getenv("ALLOW_CPU_FALLBACK", "0") == "1"


def _json_value(prediction: Any) -> dict[str, Any]:
    value = prediction.json
    if callable(value):
        value = value()
    if not isinstance(value, dict):
        raise RuntimeError("PaddleOCR returned an unexpected result type.")
    result = value.get("res", value)
    if not isinstance(result, dict):
        raise RuntimeError("PaddleOCR result does not contain an object payload.")
    return result


def _polygon(points: Any) -> list[list[float]]:
    array = np.asarray(points, dtype=float)
    if array.ndim == 1 and array.size == 4:
        x0, y0, x1, y1 = array.tolist()
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return array.reshape(-1, 2).tolist()


class PaddleOcrService:
    def __init__(self, *, device: str = OCR_DEVICE) -> None:
        started = time.perf_counter()
        providers = ort.get_available_providers()
        wants_gpu = device.startswith("gpu")
        if wants_gpu and "CUDAExecutionProvider" not in providers:
            if not ALLOW_CPU_FALLBACK:
                raise RuntimeError(
                    "CUDAExecutionProvider is unavailable; refusing a slow CPU fallback."
                )
            device = "cpu"

        selected_providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("gpu")
            else ["CPUExecutionProvider"]
        )
        self.device = device
        self.providers = selected_providers
        self._lock = threading.Lock()
        self.engine = PaddleOCR(
            text_detection_model_name=DETECTION_MODEL,
            text_recognition_model_name=RECOGNITION_MODEL,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_recognition_batch_size=RECOGNITION_BATCH_SIZE,
            engine="onnxruntime",
            engine_config={"providers": selected_providers},
            device=device,
        )
        self.load_seconds = round(time.perf_counter() - started, 3)

    def recognize(self, image: np.ndarray, page_number: int) -> dict[str, Any]:
        started = time.perf_counter()
        with self._lock:
            predictions = list(self.engine.predict(image))
        if len(predictions) != 1:
            raise RuntimeError(
                f"Expected one OCR result for page {page_number}, got {len(predictions)}."
            )

        result = _json_value(predictions[0])
        texts = list(result.get("rec_texts") or [])
        scores = list(result.get("rec_scores") or [])
        polygons = result.get("rec_polys")
        if polygons is None:
            polygons = result.get("rec_boxes") or []
        polygons = list(polygons)
        if not (len(texts) == len(scores) == len(polygons)):
            raise RuntimeError("PaddleOCR text, confidence, and polygon counts differ.")

        words: list[dict[str, Any]] = []
        for text, score, points in zip(texts, scores, polygons):
            value = str(text).strip()
            if not value:
                continue
            polygon = _polygon(points)
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            words.append(
                {
                    "text": value,
                    "confidence": round(float(score), 6),
                    "polygon": polygon,
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                }
            )

        height, width = image.shape[:2]
        return {
            "page": page_number,
            "width": int(width),
            "height": int(height),
            "text": "\n".join(word["text"] for word in words),
            "words": words,
            "word_count": len(words),
            "ocr_seconds": round(time.perf_counter() - started, 3),
        }

