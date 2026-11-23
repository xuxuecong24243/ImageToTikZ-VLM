"""Offline candidate-selection contracts; no API, compiler or real-image search."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from unittest.mock import patch

from auto_scene_validator import overlap, whole_box
from structured_tikz.validator import validate_scene
from visual_patch_optimizer import optimize_visual_patches, scene_fingerprint
from visual_patch_validator import apply_visual_patch, parse_visual_feedback


ROOT = Path(__file__).resolve().parents[1]


def text_node(identifier="caption", box=None, text="256", mode="math"):
    return {"id": identifier, "type": "text", "bbox": box or [40, 40, 10, 10],
            "text": text, "mode": mode}


def scene_fixture():
    return {"canvas": {"width": 300, "height": 200, "coordinate_system": "top-left"},
            "styles": {}, "nodes": [text_node()], "labels": [],
            "edges": [{"id": "wire", "points": [[45, 20], [45, 80]], "arrow": "end"}]}


def evidence_for(scene):
    records = []
    for node in scene["nodes"]:
        if node["type"] != "text":
            continue
        x, y, width, height = node["bbox"]
        records.append({"id": node["id"], "collection": "nodes", "status": "measured",
                        "bbox_scene": list(node["bbox"]),
                        "line_boxes_scene": [list(node["bbox"])], "line_count": 1,
                        "expected_bbox": list(node["bbox"]),
                        "original_position": {"x": x + width / 2, "y": y + height / 2,
                                              "anchor": "center"}})
    return {"scene_fingerprint": scene_fingerprint(scene),
            "bounds_report": {"mapping": {"status": "verified", "scale": 1, "tx": 0, "ty": 0},
                              "labels": records}}


def move(identifier="caption", dx=40, dy=0):
    return {"op": "move_label", "id": identifier, "dx": dx, "dy": dy}


def feedback_for(*operations, issue="edge_text_overlap"):
    return {"status": "patch_proposed", "observations": [
        {"object_id": operation["id"], "issue": issue, "confidence": .8,
         "reason": "Synthetic offline geometry fixture; not a real model judgment."}
        for operation in operations], "operations": list(operations)}


def tensor_fixture(box):
    scene = scene_fixture()
    scene["edges"] = []
    scene["nodes"] = [
        {"id": "tensor", "type": "tensor3d", "bbox": [100, 100, 50, 50],
         "depth": {"dx": 30, "dy": -20}, "grid": {"rows": 2, "cols": 2}},
        text_node(box=box, text="d = 256")]
    return scene


def joint_fixture():
    scene = scene_fixture()
    scene["nodes"].append(text_node("other", [68, 40, 10, 10], "L = 4"))
    scene["edges"].append({"id": "other_wire", "points": [[73, 20], [73, 80]], "arrow": "end"})
    return scene


class VisualPatchOptimizerTests(OfflineTestCase):
    def optimize(self, scene=None, feedback=None, evidence=None):
        scene = scene if scene is not None else scene_fixture()
        return optimize_visual_patches(scene, feedback or feedback_for(move()),
                                       evidence if evidence is not None else evidence_for(scene))

    def test_oversized_model_move_becomes_a_legal_candidate(self):
        scene = scene_fixture()
        result, report = self.optimize(scene)
        self.assertEqual(result["operations"], [move(dx=10)])
        self.assertEqual(report["items"][0]["reason"], "verified_collision_removed")
        after, checked = apply_visual_patch(scene, result, evidence_for(scene)["bounds_report"])
        self.assertTrue(checked["accepted"])
        self.assertEqual(after["nodes"][0]["bbox"], [50, 40, 10, 10])

    def test_direction_search_is_not_clamping_the_model_number(self):
        scene = scene_fixture()
        selections = []
        for magnitude in (25, 40, 200):
            result, _ = self.optimize(scene, feedback_for(move(dx=magnitude)))
            selections.append(result["operations"])
        self.assertEqual(selections, [[move(dx=10)]] * 3)
        self.assertNotEqual(selections[0][0]["dx"], 20)

    def test_minimum_effective_fixed_step_is_selected(self):
        scene = scene_fixture()
        scene["nodes"][0]["bbox"] = [41, 40, 10, 10]
        result, _ = self.optimize(scene)
        # At dx=6 the 3 px safety envelope still intersects x=45;
        # at dx=8 it starts at x=46. No larger move is necessary.
        self.assertEqual(result["operations"], [move(dx=8)])

    def test_euclidean_limit_holds_for_diagonal_candidates(self):
        scene = scene_fixture()
        result, _ = self.optimize(scene, feedback_for(move(dx=40, dy=40)))
        self.assertTrue(result["operations"])
        for operation in result["operations"]:
            distance = math.hypot(operation["dx"], operation["dy"])
            self.assertGreater(distance, 0)
            self.assertLessEqual(distance, 20)

    def test_repeat_from_same_baseline_is_deterministic_without_drift(self):
        scene = scene_fixture()
        evidence = evidence_for(scene)
        first, _ = self.optimize(scene, evidence=evidence)
        second, _ = self.optimize(scene, evidence=evidence)
        self.assertEqual(first, second)
        after, _ = apply_visual_patch(scene, first, evidence["bounds_report"])
        stale, report = self.optimize(after, evidence=evidence)
        self.assertEqual(stale["operations"], [])
        self.assertEqual(report["items"][0]["reason"], "stale_or_unverified_evidence")

    def test_no_safe_candidate_preserves_a_wide_label(self):
        scene = scene_fixture()
        scene["nodes"][0]["bbox"] = [20, 40, 60, 10]
        scene["edges"][0]["points"] = [[50, 20], [50, 80]]
        result, report = self.optimize(scene)
        self.assertEqual(result["operations"], [])
        self.assertEqual(report["items"][0]["reason"], "no_safe_candidate")

    def test_tensor_suggestion_without_measured_collision_is_rejected(self):
        scene = tensor_fixture([220, 100, 10, 10])
        result, report = self.optimize(scene, feedback_for(move(), issue="label_on_tensor"))
        self.assertEqual(result["operations"], [])
        self.assertEqual(report["items"][0]["reason"], "insufficient_geometric_evidence")

    def test_tensor_polygon_not_whole_bounding_rectangle_controls_evidence(self):
        scene = tensor_fixture([174, 138, 4, 6])
        # This box lies within the overall tensor bounding rectangle but beyond
        # the real diagonal bottom edge of the right face.
        self.assertGreater(overlap(scene["nodes"][1]["bbox"], whole_box(scene["nodes"][0])), 0)
        result, report = self.optimize(scene, feedback_for(move(), issue="label_on_tensor"))
        self.assertEqual(result["operations"], [])
        self.assertEqual(report["items"][0]["reason"], "insufficient_geometric_evidence")

    def test_verified_tensor_face_overlap_can_be_resolved_locally(self):
        scene = tensor_fixture([172, 100, 8, 8])
        result, _ = self.optimize(scene, feedback_for(move(dx=25), issue="label_on_tensor"))
        self.assertEqual(result["operations"], [move(dx=12)])

    def test_wrapped_label_cannot_be_fixed_by_translation(self):
        scene = scene_fixture()
        result, report = self.optimize(scene, feedback_for(move(dx=5, dy=-5), issue="wrapped_label"))
        self.assertEqual(result["operations"], [])
        self.assertEqual(report["items"][0]["reason"], "wrong_operation_type")

    def test_measured_multiline_text_also_rejects_move_even_with_another_issue_code(self):
        scene = scene_fixture()
        evidence = evidence_for(scene)
        record = evidence["bounds_report"]["labels"][0]
        record.update(line_count=2, bbox_scene=[40, 35, 10, 20],
                      line_boxes_scene=[[40, 35, 10, 8], [40, 47, 10, 8]])
        result, report = self.optimize(scene, evidence=evidence)
        self.assertEqual(result["operations"], [])
        self.assertEqual(report["items"][0]["reason"], "wrong_operation_type")

    def test_individually_safe_candidates_are_not_assumed_jointly_safe(self):
        scene = joint_fixture()
        right, _ = self.optimize(scene, feedback_for(move()))
        left, _ = self.optimize(scene, feedback_for(move("other", -40)))
        self.assertEqual(right["operations"], [move(dx=10)])
        self.assertEqual(left["operations"], [move("other", -10)])
        combined, report = self.optimize(scene, feedback_for(move(), move("other", -40)))
        self.assertEqual(combined["operations"], [])
        self.assertTrue(all(item["reason"] == "joint_validation_failed" for item in report["items"]))

    def test_joint_failure_is_atomic_and_leaves_no_selected_operation(self):
        scene = joint_fixture()
        snapshot = copy.deepcopy(scene)
        result, report = self.optimize(scene, feedback_for(move(), move("other", -40)))
        self.assertEqual(scene, snapshot)
        self.assertEqual(result["status"], "no_change")
        self.assertEqual(result["operations"], [])
        self.assertFalse(any(item["status"] == "accepted" for item in report["items"]))

    def test_inputs_feedback_and_measurements_are_not_mutated(self):
        scene = scene_fixture()
        feedback = feedback_for(move())
        evidence = evidence_for(scene)
        snapshots = copy.deepcopy((scene, feedback, evidence))
        self.optimize(scene, feedback, evidence)
        self.assertEqual((scene, feedback, evidence), snapshots)

    def test_text_math_ids_topology_and_geometry_are_preserved(self):
        scene = scene_fixture()
        scene["nodes"][0].update(text=r"W_E\in\mathbb{R}^{50257\times256}", mode="math")
        scene["nodes"].extend([
            text_node("chinese", [180, 140, 80, 20], "中文输入嵌入", "text"),
            {"id": "matrix", "type": "matrix", "bbox": [200, 20, 40, 40],
             "grid": {"rows": 2, "cols": 4}}])
        optimized, _ = self.optimize(scene)
        after, _ = apply_visual_patch(scene, optimized, evidence_for(scene)["bounds_report"])
        validate_scene(after)
        self.assertEqual(after["edges"], scene["edges"])
        self.assertEqual(after["nodes"][2], scene["nodes"][2])
        for old, new in zip(scene["nodes"], after["nodes"]):
            self.assertEqual(old["id"], new["id"])
            self.assertEqual(old.get("text"), new.get("text"))
            self.assertEqual(old.get("mode"), new.get("mode"))
        self.assertEqual(scene["nodes"][0]["bbox"], [40, 40, 10, 10])

    def test_mismatched_measurement_position_is_not_used(self):
        scene = scene_fixture()
        evidence = evidence_for(scene)
        evidence["bounds_report"]["labels"][0]["original_position"]["x"] += 1
        result, report = self.optimize(scene, evidence=evidence)
        self.assertEqual(result["operations"], [])
        self.assertEqual(report["items"][0]["reason"], "stale_or_unverified_evidence")

    def test_unverified_coordinate_mapping_is_not_used(self):
        scene = scene_fixture()
        evidence = evidence_for(scene)
        evidence["bounds_report"]["mapping"]["status"] = "unresolved"
        result, report = self.optimize(scene, evidence=evidence)
        self.assertEqual(result["operations"], [])
        self.assertEqual(report["items"][0]["reason"], "stale_or_unverified_evidence")

    def test_saved_deepseek_response_parses_without_transport(self):
        # Format compatibility only: this does not perform the real optimization
        # experiment or interpret synthetic test success as visual success.
        source = ROOT / "tests/fixtures/historical/visual_feedback_raw.txt"
        feedback = parse_visual_feedback(source.read_text(encoding="utf-8"))
        self.assertEqual(len(feedback["operations"]), 6)
        self.assertEqual([operation["dx"] for operation in feedback["operations"]],
                         [40, -45, 25, 25, 25, 5])
        self.assertTrue(all(operation["op"] == "move_label" for operation in feedback["operations"]))

    def test_no_legal_candidate_entry_never_compiles(self):
        from offline_visual_patch_optimization import build_parser, run_experiment

        scene = scene_fixture()
        feedback = {"status": "no_change", "observations": [], "operations": []}
        evaluation = {"items": [], "counts": {}, "joint_validation": {"accepted": True, "applied": False}}
        loaded = (scene, json.dumps(scene).encode("utf-8"), feedback,
                  evidence_for(scene)["bounds_report"], {}, {})
        with tempfile.TemporaryDirectory() as temporary:
            args = build_parser().parse_args(["--output-root", temporary])
            with patch("offline_visual_patch_optimization._load_verified_inputs", return_value=loaded), \
                    patch("offline_visual_patch_optimization.optimize_visual_patches", return_value=(feedback, evaluation)), \
                    patch("offline_visual_patch_optimization.compile_bounded") as compiler:
                report = run_experiment(args)
            compiler.assert_not_called()
            self.assertEqual(report["status"], "no_change")
            self.assertEqual(report["compile_count"], 0)
            self.assertFalse((Path(report["output_dir"]) / "scene_after.json").exists())


if __name__ == "__main__":
    unittest.main()
