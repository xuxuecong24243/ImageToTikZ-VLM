"""Small synthetic images test rendering utilities, not AI reconstruction."""

import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

import image_compare
from image_compare import ImageOutputError, make_comparison, render_pdf


class ComparisonTests(OfflineTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_equal_panels_preserve_aspect_ratio(self):
        left = self.root / "original.png"
        right = self.root / "rendered.png"
        Image.new("RGB", (100, 50), "red").save(left)
        Image.new("RGB", (100, 100), "blue").save(right)
        result = make_comparison(left, right, self.root / "comparison.png")
        with Image.open(result) as image:
            y = image_compare.MARGIN + image_compare.HEADER_HEIGHT
            x = image_compare.MARGIN
            self.assertEqual(image.size, (260, 184))
            self.assertEqual(image.getpixel((x + 50, y + 10)), (255, 255, 255))
            self.assertEqual(image.getpixel((x + 50, y + 30)), (255, 0, 0))
            self.assertEqual(image.getpixel((x + 50, y + 90)), (255, 255, 255))
            self.assertEqual(image.getpixel((x + 120 + 50, y + 10)), (0, 0, 255))

    def test_alpha_is_composited_onto_white(self):
        left = self.root / "alpha.png"
        right = self.root / "white.png"
        Image.new("RGBA", (80, 80), (0, 0, 0, 0)).save(left)
        Image.new("RGB", (80, 80), "white").save(right)
        result = make_comparison(left, right, self.root / "comparison.png")
        with Image.open(result) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.getpixel((60, 104)), (255, 255, 255))

    def test_exif_orientation_is_applied(self):
        source = self.root / "rotated.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (80, 40), "red").save(source, exif=exif)
        self.assertEqual(image_compare._read_rgb(source).size, (40, 80))

    def test_inputs_cannot_be_overwritten(self):
        source = self.root / "source.png"
        Image.new("RGB", (40, 40), "red").save(source)
        with self.assertRaisesRegex(ImageOutputError, "overwrite"):
            make_comparison(source, source, source)

    def test_invalid_image_and_pixel_limit(self):
        source = self.root / "bad.png"
        source.write_bytes(b"not an image")
        with self.assertRaises(ImageOutputError):
            image_compare._read_rgb(source)
        Image.new("RGB", (20, 20), "white").save(source)
        with patch.object(image_compare, "MAX_IMAGE_PIXELS", 100):
            with self.assertRaisesRegex(ImageOutputError, "pixels"):
                image_compare._read_rgb(source)


class PDFRenderTests(OfflineTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.pdf = self.root / "mock.pdf"
        self.pdf.write_bytes(b"%PDF-1.7\nMOCK: not a real PDF")
        self.png = self.root / "preview.png"

    def fake_renderer(self, pages=1, width=100, height=100):
        module = MagicMock()
        document = module.open.return_value.__enter__.return_value
        document.is_pdf = True
        document.needs_pass = False
        document.page_count = pages
        page = document.load_page.return_value
        page.rect = SimpleNamespace(width=width, height=height)
        page.get_pixmap.return_value.save.side_effect = (
            lambda path: Image.new("RGB", (20, 20), "white").save(path)
        )
        return module, document, page

    def test_render_command_uses_rgb_and_requested_scale(self):
        module, _, page = self.fake_renderer()
        with patch.object(image_compare, "_load_pymupdf", return_value=module):
            self.assertEqual(render_pdf(self.pdf, self.png, dpi=144), self.png.resolve())
        module.Matrix.assert_called_once_with(2.0, 2.0)
        self.assertFalse(page.get_pixmap.call_args.kwargs["alpha"])
        self.assertEqual(page.get_pixmap.call_args.kwargs["colorspace"], module.csRGB)

    def test_multi_page_output_is_not_silently_truncated(self):
        module, _, page = self.fake_renderer(pages=2)
        with patch.object(image_compare, "_load_pymupdf", return_value=module):
            with self.assertRaisesRegex(ImageOutputError, "2 pages"):
                render_pdf(self.pdf, self.png)
        page.get_pixmap.assert_not_called()

    def test_oversized_render_is_rejected_before_allocation(self):
        module, _, page = self.fake_renderer(width=100000, height=100000)
        with patch.object(image_compare, "_load_pymupdf", return_value=module):
            with self.assertRaisesRegex(ImageOutputError, "pixels"):
                render_pdf(self.pdf, self.png)
        page.get_pixmap.assert_not_called()

    def test_invalid_dpi_or_missing_pdf(self):
        for dpi in (0, 301, float("nan"), True):
            with self.subTest(dpi=dpi), self.assertRaises(ImageOutputError):
                render_pdf(self.pdf, self.png, dpi=dpi)
        with self.assertRaises(ImageOutputError):
            render_pdf(self.root / "missing.pdf", self.png)

    def test_missing_pymupdf_has_install_instruction(self):
        with patch.object(image_compare.importlib, "import_module", side_effect=ImportError):
            with self.assertRaisesRegex(ImageOutputError, "pip install PyMuPDF"):
                image_compare._load_pymupdf()


if __name__ == "__main__":
    unittest.main()
