"""Offline geometry tests use synthetic pictures, not claims of real reconstruction."""
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

from geometry_analyzer import (
    GeometryError, analyze_image, deduplicate_rectangles, merge_line_segments,
    normalized_bbox, save_geometry_artifacts,
)


HAS_OPENCV = importlib.util.find_spec("cv2") is not None


class GeometryMathTests(OfflineTestCase):
    def test_bbox_normalization_preserves_aspect_and_coordinates(self):
        self.assertEqual(normalized_bbox((20, 10, 80, 50), 200, 100),
                         {"x": .1, "y": .1, "width": .4, "height": .5})

    def test_duplicate_edges_and_contained_grid_cells_are_suppressed(self):
        candidates = [
            {"box": (20, 20, 100, 70), "methods": ["outer"], "score": .8},
            {"box": (21, 21, 98, 68), "methods": ["duplicate"], "score": .85},
            {"box": (22, 22, 30, 30), "methods": ["cell"], "score": .7},
            {"box": (160, 30, 50, 40), "methods": ["separate"], "score": .8},
        ]
        result = deduplicate_rectangles(candidates, 500 * 300)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["methods"], ["duplicate", "outer"])
        self.assertEqual(result[0]["suppressed_inner_candidates"], 1)

    def test_large_panels_do_not_erase_all_inner_objects(self):
        candidates = [{"box": (0, 0, 250, 190), "methods": ["outer"], "score": .8},
                      {"box": (20, 20, 80, 40), "methods": ["inner"], "score": .8}]
        self.assertEqual(len(deduplicate_rectangles(candidates, 300 * 200)), 2)

    def test_collinear_segments_merge_but_separate_parallel_lines_do_not(self):
        merged = merge_line_segments([(10, 20, 60, 20), (62, 21, 110, 21), (10, 50, 60, 50)])
        self.assertEqual(len(merged), 2)
        first = next(s for s in merged if s[2] == 2)
        self.assertLessEqual(first[0][0], 10)
        self.assertGreaterEqual(first[1][0], 109)

    def test_vertical_and_diagonal_segments_merge(self):
        vertical = merge_line_segments([(20, 10, 20, 50), (21, 52, 21, 90)])
        self.assertEqual(len(vertical), 1)
        diagonal = merge_line_segments([(10, 10, 50, 50), (52, 52, 90, 90)])
        self.assertEqual(len(diagonal), 1)
        self.assertEqual(diagonal[0][2], 2)


@unittest.skipUnless(HAS_OPENCV, "Install geometry dependencies in the project .venv to run detection tests")
class GeometryDetectionTests(OfflineTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "outputs")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def save(self, image, name="测试图.png"):
        path = self.root / name
        image.save(path)
        return path

    def test_real_dimensions_channels_and_empty_white_image(self):
        result = analyze_image(self.save(Image.new("RGB", (320, 160), "white")))
        self.assertEqual(result["image"]["width"], 320)
        self.assertEqual(result["image"]["height"], 160)
        self.assertEqual(result["image"]["aspect_ratio"], 2)
        self.assertEqual(result["image"]["channels"], 3)
        self.assertEqual(result["coordinate_system"], "top-left-normalized")
        self.assertEqual(result["rectangles"], [])
        self.assertEqual(result["lines"], [])
        self.assertEqual(result["regions"], [])
        self.assertTrue(any("No rectangle" in warning for warning in result["warnings"]))

    def test_invalid_or_missing_image_reports_a_clear_error(self):
        path = self.root / "invalid.png"
        path.write_bytes(b"not-an-image")
        for source in (path, self.root / "missing.png"):
            with self.subTest(source=source), self.assertRaises(GeometryError):
                analyze_image(source)

    def test_rectangle_outline_and_grid_deduplication(self):
        image = Image.new("RGB", (420, 250), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 40, 200, 150), outline="black", width=2)
        for x in (80, 120, 160):
            draw.line((x, 40, x, 150), fill="black", width=2)
        draw.line((40, 95, 200, 95), fill="black", width=2)
        result = analyze_image(self.save(image))
        self.assertEqual(len(result["rectangles"]), 1)
        box = result["rectangles"][0]["bbox_px"]
        self.assertLessEqual(abs(box["x"] - 40), 3)
        self.assertLessEqual(abs(box["y"] - 40), 3)
        self.assertLessEqual(abs(box["width"] - 160), 5)
        self.assertLessEqual(abs(box["height"] - 110), 5)
        self.assertGreater(result["counts"]["rectangle_candidates_removed_or_capped"], 0)

    def test_three_line_orientations_and_unknown_directions(self):
        image = Image.new("RGB", (420, 280), "white")
        draw = ImageDraw.Draw(image)
        draw.line((25, 30, 185, 30), fill="black", width=2)
        draw.line((230, 30, 230, 175), fill="black", width=2)
        draw.line((40, 130, 160, 245), fill="black", width=2)
        result = analyze_image(self.save(image))
        self.assertEqual({line["orientation"] for line in result["lines"]},
                         {"horizontal", "vertical", "diagonal"})
        self.assertTrue(all(line["direction"] == "unknown" for line in result["lines"]))
        for line in result["lines"]:
            self.assertEqual(len(line["endpoint_candidates"]), 2)
            for key in ("x1", "y1", "x2", "y2"):
                self.assertTrue(0 <= line[key] <= 1)

    def test_chromatic_shape_is_explicitly_only_a_component_bbox(self):
        image = Image.new("RGB", (400, 240), "white")
        ImageDraw.Draw(image).polygon(((60, 80), (90, 50), (240, 50), (240, 150), (210, 180), (60, 180)),
                                     fill=(190, 220, 245))
        result = analyze_image(self.save(image))
        self.assertTrue(any("chromatic_component_bounding_box" in item["evidence"]["methods"]
                            for item in result["rectangles"]))
        self.assertTrue(all(item["evidence"]["meaning"] == "geometric_candidate_not_semantic_object"
                            for item in result["rectangles"]))

    def test_alpha_compositing_does_not_turn_empty_transparency_into_black_shapes(self):
        result = analyze_image(self.save(Image.new("RGBA", (100, 80), (0, 0, 0, 0))))
        self.assertEqual(result["image"]["channels"], 4)
        self.assertEqual(result["rectangles"], [])
        self.assertEqual(result["lines"], [])

    def test_json_artifacts_unique_ids_and_original_unchanged(self):
        image = Image.new("RGB", (400, 240), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((25, 40, 125, 120), fill=(190, 220, 245), outline="black", width=2)
        draw.rectangle((150, 40, 250, 120), fill=(190, 220, 245), outline="black", width=2)
        path = self.save(image)
        original = path.read_bytes()
        result = analyze_image(path)
        identifiers = [item["id"] for group in ("rectangles", "lines", "regions") for item in result[group]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        artifacts = save_geometry_artifacts(path, self.root / "artifacts", result)
        self.assertEqual(json.loads(artifacts["geometry"].read_text(encoding="utf-8")), result)
        with Image.open(artifacts["debug"]) as debug:
            self.assertEqual(debug.size, (400, 310))
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaisesRegex(GeometryError, "already exists"):
            save_geometry_artifacts(path, self.root / "artifacts", result)

    def test_wrong_image_geometry_cannot_create_a_misleading_overlay(self):
        first = self.save(Image.new("RGB", (100, 100), "white"))
        geometry = analyze_image(first)
        second = self.save(Image.new("RGB", (100, 100), "black"), "second.png")
        with self.assertRaisesRegex(GeometryError, "different image"):
            save_geometry_artifacts(second, self.root / "bad", geometry)
        self.assertFalse((self.root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
