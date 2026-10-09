from __future__ import annotations

import os
import time
from typing import Any

import runpod

from input_loader import _positive_int, load_document, render_pages
from ocr_engine import (
    DETECTION_MODEL,
    RECOGNITION_BATCH_SIZE,
    RECOGNITION_MODEL,
    PaddleOcrService,
)


MAXIMUM_BYTES = int(os.getenv("MAX_DOCUMENT_MB", "35")) * 1024 * 1024
MAXIMUM_PAGES = int(os.getenv("MAX_PAGES", "100"))
DEFAULT_DPI = int(os.getenv("OCR_DPI", "300"))

# Module-level initialization is intentional: Runpod reuses this model for every
# job handled by the warm worker instead of loading it once per request.
OCR = PaddleOcrService()


def handler(event: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    job_input = event.get("input")
    if not isinstance(job_input, dict):
        raise ValueError("The request body must contain an input object.")

    dpi = _positive_int(job_input.get("dpi"), "dpi", DEFAULT_DPI, 400)
    document = load_document(job_input, MAXIMUM_BYTES)

    render_started = time.perf_counter()
    images = render_pages(
        document.content,
        dpi=dpi,
        maximum_pages=MAXIMUM_PAGES,
    )
    render_seconds = time.perf_counter() - render_started

    ocr_started = time.perf_counter()
    pages = [OCR.recognize(image, index) for index, image in enumerate(images, 1)]
    ocr_seconds = time.perf_counter() - ocr_started

    return {
        "engine": {
            "name": "PaddleOCR",
            "detection_model": DETECTION_MODEL,
            "recognition_model": RECOGNITION_MODEL,
            "recognition_batch_size": RECOGNITION_BATCH_SIZE,
            "device": OCR.device,
            "providers": OCR.providers,
            "worker_model_load_seconds": OCR.load_seconds,
        },
        "document": {
            "filename": document.filename,
            "source": document.source,
            "sha256": document.sha256,
            "bytes": len(document.content),
            "page_count": len(pages),
            "dpi": dpi,
        },
        "pages": pages,
        "text": "\n\f\n".join(page["text"] for page in pages),
        "timing": {
            "render_seconds": round(render_seconds, 3),
            "ocr_seconds": round(ocr_seconds, 3),
            "total_seconds": round(time.perf_counter() - started, 3),
        },
    }


runpod.serverless.start({"handler": handler})

