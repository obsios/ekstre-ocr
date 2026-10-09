import threading
import unittest

import numpy as np

import ocr_engine
from ocr_engine import PaddleOcrService


class _Prediction:
    def __init__(self, text: str) -> None:
        self.json = {
            "res": {
                "rec_texts": [text],
                "rec_scores": [0.99],
                "rec_polys": [[[1, 2], [3, 2], [3, 4], [1, 4]]],
            }
        }


class _Engine:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def predict(self, images):
        self.calls.append(len(images))
        return [_Prediction(f"page-{int(image[0, 0, 0])}") for image in images]


class OcrEngineTests(unittest.TestCase):
    def test_recognize_many_preserves_page_order_and_batches(self):
        service = PaddleOcrService.__new__(PaddleOcrService)
        service.engine = _Engine()
        service._lock = threading.Lock()
        images = [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(10)]

        original_batch_size = ocr_engine.PAGE_BATCH_SIZE
        ocr_engine.PAGE_BATCH_SIZE = 4
        try:
            pages = service.recognize_many(images)
        finally:
            ocr_engine.PAGE_BATCH_SIZE = original_batch_size

        self.assertEqual(service.engine.calls, [4, 4, 2])
        self.assertEqual([page["page"] for page in pages], list(range(1, 11)))
        self.assertEqual([page["text"] for page in pages], [f"page-{i}" for i in range(10)])
        self.assertTrue(all(page["ocr_timing_scope"] == "batch_average" for page in pages))
        self.assertTrue(all(page["width"] == 3 and page["height"] == 2 for page in pages))


if __name__ == "__main__":
    unittest.main()
