"""Optional text-box adapter and a small, explicitly non-OCR fallback.

No model, executable, network request or dependency installation is started here.
Inject a local detector returning dictionaries with bbox_px=[x,y,width,height],
source='ocr' and confidence (or None). Coordinates must use the stored pixels.
"""
from __future__ import annotations

import math


def normalize_text_regions(regions, width, height):
    """Validate an entire detector result before using any of its boxes."""
    result = []
    for index, region in enumerate(regions, 1):
        box = region.get("bbox_px")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("Text bbox_px must be [x,y,width,height].")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            raise ValueError("Text coordinates must be finite numbers.")
        x, y, w, h = box
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            raise ValueError("Text box is outside the stored image.")
        source = region.get("source", "external_boxes")
        if source not in {"ocr", "external_boxes", "heuristic_connected_components"}:
            raise ValueError("Unknown text-box source; do not label heuristics as OCR.")
        confidence = region.get("confidence")
        if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                                       or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("Text confidence must be None or a number in [0,1].")
        if source == "heuristic_connected_components":
            confidence = None  # No calibrated OCR probability exists for this method.
        result.append({"id": "text_%03d" % index, "bbox_px": list(box),
                       "bbox": dict(zip(("x", "y", "width", "height"),
                                        (round(x / width, 6), round(y / height, 6),
                                         round(w / width, 6), round(h / height, 6)))),
                       "source": source, "confidence": confidence})
    return result


def _heuristic_boxes(pixels):
    """Group compact dark components into text-like rows, without recognizing text.

    Thin strokes/large connected grids are rejected before grouping. Symbols,
    arrowheads and short geometry can still be mistaken for glyphs.
    """
    import cv2
    import numpy as np

    height, width = pixels.shape[:2]
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    ink = (gray < 150).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    selected = []
    max_h, max_w = max(32, height * .075), max(20, width * .05)
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if (3 <= h <= max_h and 1 <= w <= max_w and area >= 4
                and .12 <= w / h <= 5 and .07 <= area / (w * h) <= .94):
            selected.append(index)
    if not selected:
        return []
    lookup = np.zeros(count, dtype=np.uint8)
    lookup[selected] = 255
    glyphs = lookup[labels]
    # Small gaps join characters in a row, without merging separate label rows.
    gap = max(3, min(13, int(np.median(stats[selected, cv2.CC_STAT_HEIGHT]) * .55)))
    joined = cv2.dilate(glyphs, np.ones((3, gap), np.uint8))
    groups, group_labels, group_stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    members = {}
    for index in selected:
        x, y, w, h, _ = (int(v) for v in stats[index])
        local_y, local_x = np.nonzero(labels[y:y+h, x:x+w] == index)
        group = int(group_labels[y + local_y[0], x + local_x[0]])
        members.setdefault(group, []).append(index)
    result = []
    for group, indices in members.items():
        if group == 0 or len(indices) < 2:
            continue
        boxes = stats[indices]
        x, y = boxes[:, 0].min(), boxes[:, 1].min()
        right = (boxes[:, 0] + boxes[:, 2]).max()
        bottom = (boxes[:, 1] + boxes[:, 3]).max()
        w, h = int(right - x), int(bottom - y)
        if h > max_h * 1.4 or w < 6 or w / max(1, h) < .4:
            continue
        x, y = max(0, int(x) - 2), max(0, int(y) - 2)
        right, bottom = min(width, int(right) + 2), min(height, int(bottom) + 2)
        result.append({"bbox_px": [x, y, right - x, bottom - y],
                       "source": "heuristic_connected_components", "confidence": None})
    return sorted(result, key=lambda r: (r["bbox_px"][1], r["bbox_px"][0]))


def detect_text_regions(pixels, *, supplied_regions=None, detector=None):
    """Return boxes plus diagnostics; missing/broken OCR never blocks geometry.

    supplied_regions=[] explicitly disables automatic detection. A detector is
    called once, synchronously; callers must supply an offline local adapter.
    """
    height, width = pixels.shape[:2]
    method = "supplied_boxes" if supplied_regions is not None else "injected_detector" if detector else "heuristic_connected_components"
    try:
        if supplied_regions is not None and detector is not None:
            raise ValueError("Supply boxes or a detector, not both.")
        raw = supplied_regions if supplied_regions is not None else detector(pixels) if detector else _heuristic_boxes(pixels)
        regions = normalize_text_regions(raw, width, height)
        return regions, {"status": "ok", "method": method,
                         "ocr_regions": sum(r["source"] == "ocr" for r in regions),
                         "heuristic_regions": sum(r["source"] == "heuristic_connected_components" for r in regions),
                         "warnings": ["Heuristic boxes are text-like components, not OCR; isolated characters and formulas may be missed."]
                                     if method == "heuristic_connected_components" else []}
    except Exception as exc:
        # Do not log arbitrary adapter exception text (it can contain credentials).
        return [], {"status": "unavailable_or_failed", "method": method,
                    "error_type": type(exc).__name__, "ocr_regions": 0, "heuristic_regions": 0,
                    "warnings": ["Text detector unavailable, failed or returned invalid boxes; original geometry was retained."]}
