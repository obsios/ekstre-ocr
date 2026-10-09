import base64
import io
import unittest

from PIL import Image

from input_loader import load_document, render_pages


class InputLoaderTests(unittest.TestCase):
    def test_base64_image_round_trip(self):
        buffer = io.BytesIO()
        Image.new("RGB", (40, 20), "white").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        document = load_document(
            {"base64": encoded, "filename": "test.png"},
            maximum_bytes=1024 * 1024,
        )
        pages = render_pages(document.content, dpi=300, maximum_pages=2)

        self.assertEqual(document.filename, "test.png")
        self.assertEqual(document.source, "base64")
        self.assertEqual(pages[0].shape, (20, 40, 3))

    def test_requires_exactly_one_source(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            load_document({}, maximum_bytes=100)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            load_document({"url": "https://x", "base64": "eA=="}, 100)


if __name__ == "__main__":
    unittest.main()

