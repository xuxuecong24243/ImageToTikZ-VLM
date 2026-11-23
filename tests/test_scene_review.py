"""Offline review tests use explicit synthetic declarations, not real approvals."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from unittest.mock import patch

from PIL import Image

from scene_review import (
    HEADER_HEIGHT, _edge_points, export_verified_scene, save_scene_review,
    validate_review_status,
)
from structured_tikz.validator import SceneValidationError, validate_scene


def fixture():
    return {"canvas": {"width": 400, "height": 200, "coordinate_system": "top-left"},
            "styles": {}, "nodes": [
                {"id": "rect_001", "type": "rectangle", "bbox": [20, 60, 80, 40],
                 "metadata": {"source": "opencv", "source_id": "rect_001", "review_status": "unverified"}},
                {"id": "rect_002", "type": "rectangle", "bbox": [220, 80, 70, 40],
                 "metadata": {"review_status": "unverified"}},
            ], "edges": [], "labels": [],
            "metadata": {"review_status": "unverified", "source_image": {"width": 400, "height": 200}}}


def declared_scene():
    scene = fixture()
    scene["metadata"]["review_status"] = "verified"
    for node in scene["nodes"]:
        node["metadata"]["review_status"] = "verified"
    return scene


class SceneReviewTests(OfflineTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "outputs")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "image.png"
        Image.new("RGB", (400, 200), "white").save(self.image)

    def test_preview_keeps_source_and_shows_only_declared_edges(self):
        scene = fixture()
        scene["edges"] = [
            {"id": "edge_1", "points": [[100, 80], [220, 80]], "arrow": "end",
             "metadata": {"review_status": "verified"}},
            {"id": "edge_2", "points": [[100, 100], [220, 100]], "metadata": {"review_status": "unverified"}},
        ]
        original_scene, original_bytes = copy.deepcopy(scene), self.image.read_bytes()
        with patch("scene_review._draw_edge") as draw_edge:
            output = save_scene_review(scene, self.image, self.root / "review.png")
        self.assertEqual(draw_edge.call_count, 1)
        self.assertEqual(draw_edge.call_args.args[2]["id"], "edge_1")
        with Image.open(output) as image:
            self.assertEqual(image.size, (400, 200 + HEADER_HEIGHT))
        self.assertEqual(self.image.read_bytes(), original_bytes)
        self.assertEqual(scene, original_scene)
        with self.assertRaises(FileExistsError):
            save_scene_review(scene, self.image, output)

    def test_dimensions_and_source_hash_are_checked(self):
        scene = fixture()
        scene["canvas"]["width"] = 401
        with self.assertRaises(SceneValidationError):
            save_scene_review(scene, self.image, self.root / "wrong-size.png")
        scene = fixture()
        scene["metadata"]["source_image"]["sha256"] = "0" * 64
        with self.assertRaises(SceneValidationError):
            save_scene_review(scene, self.image, self.root / "wrong-image.png")
        scene["metadata"]["source_image"]["sha256"] = hashlib.sha256(self.image.read_bytes()).hexdigest()
        self.assertTrue(save_scene_review(scene, self.image, self.root / "matched.png").exists())

    def test_node_anchor_preview_route_matches_declared_axis(self):
        scene = fixture()
        nodes = {node["id"]: node for node in scene["nodes"]}
        edge = {"from": "rect_001", "to": "rect_002", "from_anchor": "south", "to_anchor": "north",
                "routing": "orthogonal", "first_axis": "y"}
        self.assertEqual(_edge_points(edge, nodes), [(60, 100), (60, 80), (255, 80)])
        edge["first_axis"] = "x"
        self.assertEqual(_edge_points(edge, nodes), [(60, 100), (255, 100), (255, 80)])

    def test_schema_valid_draft_is_not_automatically_verified(self):
        scene = fixture()
        validate_scene(scene)
        with self.assertRaises(SceneValidationError):
            export_verified_scene(scene, self.root / "verified.json", confirm_reviewed=True, reviewer="test-only")
        scene["metadata"]["review_status"] = "verified"
        with self.assertRaisesRegex(SceneValidationError, "still need review"):
            export_verified_scene(scene, self.root / "verified.json", confirm_reviewed=True, reviewer="test-only")

    def test_export_requires_explicit_confirmation_and_reviewer(self):
        scene = declared_scene()
        for kwargs in ({}, {"reviewer": "test-only"}, {"confirm_reviewed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(SceneValidationError):
                export_verified_scene(scene, self.root / "verified.json", **kwargs)

    def test_rejections_preserved_as_audit_but_not_drawable_objects(self):
        scene = declared_scene()
        scene["nodes"][1]["metadata"]["review_status"] = "rejected"
        before = copy.deepcopy(scene)
        output = export_verified_scene(scene, self.root / "verified.json", confirm_reviewed=True, reviewer="test-only")
        saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual([node["id"] for node in saved["nodes"]], ["rect_001"])
        self.assertEqual(saved["metadata"]["review_declaration"]["rejected_objects"]["nodes"][0]["id"], "rect_002")
        self.assertEqual(scene, before)
        validate_scene(saved)
        with self.assertRaises(FileExistsError):
            export_verified_scene(scene, output, confirm_reviewed=True, reviewer="test-only")

    def test_edges_to_rejected_nodes_and_invalid_math_block_export(self):
        scene = declared_scene()
        scene["nodes"][1]["metadata"]["review_status"] = "rejected"
        scene["edges"] = [{"id": "edge", "from": "rect_001", "to": "rect_002",
                           "metadata": {"review_status": "verified"}}]
        with self.assertRaises(SceneValidationError):
            export_verified_scene(scene, self.root / "verified.json", confirm_reviewed=True, reviewer="test-only")
        scene = declared_scene()
        scene["labels"] = [{"id": "formula", "x": 50, "y": 20, "text": r"\input{bad}", "mode": "math",
                            "metadata": {"review_status": "verified"}}]
        with self.assertRaises(SceneValidationError):
            export_verified_scene(scene, self.root / "verified.json", confirm_reviewed=True, reviewer="test-only")

    def test_invalid_status_is_not_silently_treated_as_confirmation(self):
        scene = fixture()
        scene["nodes"][0]["metadata"]["review_status"] = "probably_correct"
        with self.assertRaises(SceneValidationError):
            validate_review_status(scene)


if __name__ == "__main__":
    unittest.main()
