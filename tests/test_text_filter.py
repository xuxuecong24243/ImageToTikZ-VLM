"""Offline text-overlap tests; mock boxes are not real OCR measurements."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from PIL import Image, ImageDraw

from geometry_analyzer import analyze_image, classify_text_overlap
from text_region_detector import normalize_text_regions


def line(a=(20, 20), b=(50, 20), *, protected=False):
    import math
    return {"id": "line_fixture", "pixel": dict(zip(("x1", "y1", "x2", "y2"), (*a, *b))),
            "length_px": math.dist(a, b), "score": .8,
            "orientation": "horizontal" if a[1] == b[1] else "vertical",
            "evidence": {"possible_role": "border_or_grid" if protected else "unclassified_line",
                         "contained_in": ["rect_fixture"] if protected else []}}


def boxes(*values):
    return normalize_text_regions([{"bbox_px": list(value), "source": "ocr", "confidence": .9}
                                   for value in values], 1000, 500)


class TextOverlapTests(OfflineTestCase):
    def classify(self, candidate, regions):
        return classify_text_overlap([candidate], regions, 1000, 500)[0]

    def test_short_line_entirely_in_text_is_downweighted_not_erased(self):
        original = line()
        result = self.classify(original, boxes((10, 10, 60, 30)))
        self.assertEqual(result["classification"], "text_stroke_candidate")
        self.assertEqual(result["text_overlap_ratio"], 1)
        self.assertEqual(result["filter_reason"], "short_line_inside_text_region")
        self.assertEqual(result["score"], original["score"])
        self.assertLess(result["context_priority"], result["score"])
        self.assertEqual(result["pixel"], original["pixel"])
        self.assertNotIn("classification", original)

    def test_partial_intersection_and_center_inside_do_not_remove(self):
        result = self.classify(line(), boxes((30, 10, 10, 30)))
        self.assertEqual(result["classification"], "uncertain")
        self.assertAlmostEqual(result["text_overlap_ratio"], 1 / 3, places=5)

    def test_long_connection_crossing_text_remains_uncertain(self):
        result = self.classify(line(b=(400, 20)), boxes((10, 10, 410, 30)))
        self.assertEqual(result["classification"], "uncertain")
        self.assertEqual(result["filter_reason"], "long_line_with_text_overlap")

    def test_grid_even_inside_a_text_box_is_protected(self):
        result = self.classify(line(protected=True), boxes((10, 10, 60, 30)))
        self.assertEqual(result["classification"], "geometry")
        self.assertEqual(result["filter_reason"], "protected_border_or_grid")
        self.assertEqual(result["context_priority"], result["score"])

    def test_dimension_or_formula_rule_is_uncertain(self):
        result = self.classify(line(b=(55, 20)), boxes((10, 16, 60, 8)))
        self.assertEqual(result["classification"], "uncertain")
        self.assertEqual(result["filter_reason"], "possible_dimension_or_formula_rule")

    def test_no_text_boxes_preserves_priority_and_line(self):
        result = self.classify(line(), [])
        self.assertEqual(result["classification"], "geometry")
        self.assertEqual(result["text_overlap_ratio"], 0)
        self.assertEqual(result["context_priority"], result["score"])

    def test_overlapping_text_boxes_do_not_double_count_length(self):
        result = self.classify(line(), boxes((20, 10, 20, 30), (30, 10, 20, 30)))
        self.assertEqual(result["text_overlap_ratio"], 1)
        self.assertEqual(len(result["text_region_ids"]), 2)

    def test_vertical_segment_uses_length_not_center_only(self):
        result = self.classify(line((20, 20), (20, 50)), boxes((10, 30, 30, 10)))
        self.assertAlmostEqual(result["text_overlap_ratio"], 1 / 3, places=5)
        self.assertEqual(result["classification"], "uncertain")

    def test_serialization_records_reason_and_real_source(self):
        regions = boxes((10, 10, 60, 30))
        result = self.classify(line(), regions)
        restored = json.loads(json.dumps({"text_regions": regions, "lines": [result]}, allow_nan=False))
        self.assertEqual(restored["text_regions"][0]["source"], "ocr")
        self.assertEqual(restored["lines"][0]["filter_reason"], "short_line_inside_text_region")
        self.assertEqual(regions[0]["bbox"], {"x": .01, "y": .02, "width": .06, "height": .06})


@unittest.skipUnless(importlib.util.find_spec("cv2"), "Project geometry dependencies unavailable")
class TextDetectorTests(OfflineTestCase):
    def test_heuristic_boxes_are_not_ocr(self):
        import cv2
        import numpy as np
        from text_region_detector import detect_text_regions
        image = np.full((200, 400, 3), 255, dtype=np.uint8)
        cv2.putText(image, "TEXT LABEL", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        regions, diagnostic = detect_text_regions(image)
        self.assertTrue(regions)
        self.assertEqual(diagnostic["ocr_regions"], 0)
        self.assertTrue(all(r["source"] == "heuristic_connected_components" and r["confidence"] is None for r in regions))

    def test_ocr_unavailable_keeps_original_geometric_candidates(self):
        def unavailable(_):
            raise ImportError("not installed")
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "outputs") as directory:
            path = Path(directory) / "synthetic.png"
            image = Image.new("RGB", (300, 200), "white")
            ImageDraw.Draw(image).line((20, 40, 250, 40), fill="black", width=2)
            image.save(path)
            result = analyze_image(path, text_detector=unavailable)
            self.assertTrue(result["lines"])
            self.assertEqual(result["text_detection"]["status"], "unavailable_or_failed")
            self.assertEqual(result["text_regions"], [])
            self.assertEqual(result["counts"]["geometry"], len(result["lines"]))

    def test_bad_detector_boxes_are_rejected_without_partial_results(self):
        import numpy as np
        from text_region_detector import detect_text_regions
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        regions, diagnostic = detect_text_regions(image, supplied_regions=[
            {"bbox_px": [10, 10, 20, 20]}, {"bbox_px": [0, 0, 200, 100]}])
        self.assertEqual(regions, [])
        self.assertEqual(diagnostic["status"], "unavailable_or_failed")


if __name__ == "__main__":
    unittest.main()
