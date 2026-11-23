"""Render a single-page PDF and create an aspect-preserving visual comparison.

These helpers do not claim that matching appearance proves Chinese text, formulae
or diagram connections are correct. They never call an AI service.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]
MAX_IMAGE_PIXELS = 25_000_000
MAX_RENDER_PIXELS = 25_000_000
MAX_PANEL_SIZE = 1800
MARGIN = 20
HEADER_HEIGHT = 44


class ImageOutputError(RuntimeError):
    """A PDF or comparison image could not be produced safely."""


def _load_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except ImportError as exc:
        raise ImageOutputError(
            "Pillow is missing. Install it with: python -m pip install Pillow"
        ) from exc
    return Image, ImageDraw, ImageFont, ImageOps


def _load_pymupdf():
    for name in ("pymupdf", "fitz"):
        try:
            module = importlib.import_module(name)
            if hasattr(module, "open") and hasattr(module, "Matrix"):
                return module
        except ImportError:
            pass
    raise ImageOutputError(
        "PyMuPDF is missing. Install it with: python -m pip install PyMuPDF"
    )


def _output_path(path: PathLike, *sources: PathLike) -> Path:
    output = Path(path).expanduser().resolve()
    if output.suffix.lower() != ".png":
        raise ImageOutputError("The output filename must end in .png")
    if output in {Path(source).expanduser().resolve() for source in sources}:
        raise ImageOutputError("An output image must not overwrite an input file")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def render_pdf(pdf_path: PathLike, png_path: PathLike, *, dpi: int = 150) -> Path:
    """Render exactly one PDF page as RGB PNG, with explicit size limits.

    Rejecting multi-page documents avoids silently comparing only part of a
    generated figure. A successful result does not imply semantic correctness.
    """
    source = Path(pdf_path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise ImageOutputError("PDF does not exist or is empty: {}".format(source))
    if isinstance(dpi, bool) or not isinstance(dpi, (int, float)):
        raise ImageOutputError("DPI must be a number between 1 and 300")
    if not math.isfinite(dpi) or not 1 <= dpi <= 300:
        raise ImageOutputError("DPI must be between 1 and 300")
    output = _output_path(png_path, source)
    fitz = _load_pymupdf()
    try:
        with fitz.open(str(source)) as document:
            if not document.is_pdf or document.needs_pass:
                raise ImageOutputError("Expected an unencrypted PDF document")
            if document.page_count != 1:
                raise ImageOutputError(
                    "Expected one figure page; PDF contains {} pages".format(
                        document.page_count
                    )
                )
            page = document.load_page(0)
            scale = dpi / 72.0
            width, height = page.rect.width * scale, page.rect.height * scale
            if not all(math.isfinite(value) and value > 0 for value in (width, height)):
                raise ImageOutputError("PDF page has invalid dimensions")
            if math.ceil(width) * math.ceil(height) > MAX_RENDER_PIXELS:
                raise ImageOutputError(
                    "Rendered image would exceed {:,} pixels; lower DPI or figure size".format(
                        MAX_RENDER_PIXELS
                    )
                )
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False
            )
            pixmap.save(str(output))
    except ImageOutputError:
        raise
    except Exception as exc:
        raise ImageOutputError("PDF rendering failed: {}".format(exc)) from exc
    if not output.is_file() or output.stat().st_size == 0:
        raise ImageOutputError("PDF renderer did not produce a PNG")
    return output


def _read_rgb(path: PathLike):
    Image, _, _, ImageOps = _load_pillow()
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ImageOutputError("Image does not exist: {}".format(source))
    try:
        with Image.open(source) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ImageOutputError(
                    "Image exceeds {:,} pixels: {}".format(MAX_IMAGE_PIXELS, source)
                )
            upright = ImageOps.exif_transpose(image)
            rgba = upright.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(background, rgba).convert("RGB")
    except ImageOutputError:
        raise
    except Exception as exc:
        raise ImageOutputError("Cannot read image {}: {}".format(source, exc)) from exc


def make_comparison(
    original_path: PathLike, rendered_path: PathLike, output_path: PathLike
) -> Path:
    """Write original on the left and compiled preview on the right.

    Both images fit inside equal white panels without stretching. EXIF rotation
    and alpha transparency are handled before resizing. English panel labels
    use Pillow's bundled font, without depending on a system Chinese font.
    """
    Image, ImageDraw, ImageFont, ImageOps = _load_pillow()
    original, rendered = _read_rgb(original_path), _read_rgb(rendered_path)
    output = _output_path(output_path, original_path, rendered_path)
    panel_width = min(MAX_PANEL_SIZE, max(original.width, rendered.width))
    panel_height = min(MAX_PANEL_SIZE, max(original.height, rendered.height))
    canvas = Image.new(
        "RGB",
        (2 * panel_width + 3 * MARGIN, panel_height + 2 * MARGIN + HEADER_HEIGHT),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=24)
    except TypeError:  # Compatibility with older Pillow installations.
        font = ImageFont.load_default()
    for index, (label, image) in enumerate(
        (("Original", original), ("XeLaTeX", rendered))
    ):
        x = MARGIN + index * (panel_width + MARGIN)
        y = MARGIN + HEADER_HEIGHT
        fitted = ImageOps.contain(
            image, (panel_width, panel_height), method=Image.Resampling.LANCZOS
        )
        canvas.paste(
            fitted,
            (x + (panel_width - fitted.width) // 2, y + (panel_height - fitted.height) // 2),
        )
        draw.rectangle(
            (x - 1, y - 1, x + panel_width, y + panel_height), outline="#cccccc"
        )
        bounds = draw.textbbox((0, 0), label, font=font)
        label_width = bounds[2] - bounds[0]
        draw.text((x + (panel_width - label_width) // 2, MARGIN), label, font=font, fill="#222222")
    try:
        canvas.save(output, format="PNG")
    except OSError as exc:
        raise ImageOutputError("Cannot save comparison PNG: {}".format(exc)) from exc
    return output
