"""Offline orchestration fixtures; mocked geometry/compilation is not image-quality evidence."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from unittest.mock import Mock, patch

from PIL import Image

import main
from geometry_analyzer import GeometryError
from prompts import build_generation_prompt


VALID = r"\begin{tikzpicture}\node (sample) at (0,0) {离线测试};\end{tikzpicture}"


class GeometryModeTests(OfflineTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=main.PROJECT_DIR / "outputs")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "fixture.png"
        Image.new("RGB", (120, 60), "white").save(self.image)
        self.image_bytes = self.image.read_bytes()
        self.source_hash = hashlib.sha256(self.image_bytes).hexdigest()
        self.geometry = {
            "schema_version": "1.0",
            "image": {"width": 120, "height": 60, "aspect_ratio": 2.0, "sha256": self.source_hash},
            "coordinate_system": "top-left-normalized",
            "rectangles": [{"id": "rect_001", "bbox": {"x": .1, "y": .1, "width": .2, "height": .3},
                            "score": .8, "evidence": {"methods": ["offline_fixture"]}}],
            "lines": [{"id": "line_001", "x1": .4, "y1": .6, "x2": .8, "y2": .6,
                       "orientation": "horizontal", "direction": "unknown", "score": .7}],
            "regions": [], "warnings": ["UNIT TEST FIXTURE: not an actual detector result"],
        }
        self.client = Mock()
        self.client.config.model = "mock-model"
        self.client.generate.return_value = VALID
        self.client.repair.return_value = VALID
        self.client.last_diagnostics = {}
        self.env = {"API_BASE_URL": "https://api.deepseek.com", "MODEL_NAME": "deepseek-flash",
                    "API_KEY": "unit-test-key-only"}

    def fake_artifacts(self, image_path, directory, geometry=None):
        """Only support integration tests; no claims about detection or overlays."""
        directory = Path(directory)
        geometry_path = directory / "geometry.json"
        debug_path = directory / "geometry_debug.png"
        geometry_path.write_text(json.dumps(geometry, ensure_ascii=False), encoding="utf-8")
        debug_path.write_bytes(Path(image_path).read_bytes())
        return {"geometry": geometry_path, "debug": debug_path}

    @staticmethod
    def compile_result(tex_path, success=True, **kwargs):
        tex_path = Path(tex_path)
        return SimpleNamespace(success=success, returncode=0 if success else 1, timed_out=False,
                               pdf_path=tex_path.with_suffix(".pdf") if success else None,
                               log_path=tex_path.with_suffix(".log"),
                               console_path=tex_path.with_suffix(".console.log"),
                               errors="" if success else "! Mock compile error",
                               command=["xelatex", "-no-shell-escape", str(tex_path)])

    def run_experiment(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return main.run_experiment(self.image, self.root / "runs", self.client, **kwargs)

    def test_off_preserves_original_generation_call_and_does_not_load_cv(self):
        self.assertEqual(main.build_parser().parse_args(["--image", str(self.image)]).geometry_mode, "off")
        with patch.dict(sys.modules, {"cv2": None}), patch("main.prepare_geometry") as geometry, \
             contextlib.ExitStack() as stack:
            stack.enter_context(patch("main.compile_tex", side_effect=self.compile_result))
            stack.enter_context(patch("main.render_pdf", side_effect=lambda pdf, png: png))
            stack.enter_context(patch("main.make_comparison", side_effect=lambda original, png, out: out))
            report = self.run_experiment()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["geometry_mode"], "off")
        geometry.assert_not_called()
        self.client.generate.assert_called_once_with(self.image.resolve())
        self.assertNotIn("geometry", report)
        self.assertFalse((Path(report["output_dir"]) / "geometry.json").exists())
        self.assertEqual(Path(report["original_snapshot"]).read_bytes(), self.image_bytes)
        self.assertFalse(report["quality_verified"])

    def test_analyze_cli_needs_no_key_and_ignores_invalid_model_configuration(self):
        environment = {"API_MAX_OUTPUT_TOKENS": "invalid", "API_TOKEN_LIMIT_PARAMETER": "invalid",
                       "API_THINKING_MODE": "invalid", "API_REASONING_EFFORT": "invalid"}
        with patch.dict(os.environ, environment, clear=True), \
             patch("geometry_analyzer.analyze_image", return_value=self.geometry) as analyze, \
             patch("geometry_analyzer.save_geometry_artifacts", side_effect=self.fake_artifacts), \
             patch("main.APIConfig.from_env") as api_config, patch("main.VLMClient") as client, \
             patch("main.OutputConfig.from_env") as output_config, patch("main.compile_tex") as compiler, \
             patch("vlm_client.urllib.request.build_opener") as network, \
             contextlib.redirect_stdout(io.StringIO()):
            status = main.main(["--image", str(self.image), "--geometry-mode", "analyze",
                                "--output-dir", str(self.root / "analysis")])
        self.assertEqual(status, 0)
        analyze.assert_called_once_with(self.image.resolve())
        api_config.assert_not_called()
        output_config.assert_not_called()
        client.assert_not_called()
        compiler.assert_not_called()
        network.assert_not_called()
        reports = list((self.root / "analysis").glob("*/run.json"))
        self.assertEqual(len(reports), 1)
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "analyzed")
        self.assertFalse(report["api_called"])
        self.assertEqual(report["image_sha256"], self.source_hash)
        self.assertEqual(Path(report["original_snapshot"]).read_bytes(), self.image_bytes)
        context = json.loads(Path(report["geometry"]["context_json"]).read_text(encoding="utf-8"))
        self.assertEqual(context["usage"], "preview_only_no_api")

    def test_assist_requires_authorization_before_analysis_or_client_creation(self):
        with patch.dict(os.environ, self.env, clear=True), patch("main.prepare_geometry") as geometry, \
             patch("main.run_experiment") as experiment, patch("main.VLMClient") as client, \
             patch("vlm_client.urllib.request.build_opener") as network, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = main.main(["--image", str(self.image), "--geometry-mode", "assist",
                                "--output-dir", str(self.root / "blocked")])
        self.assertEqual(status, 2)
        geometry.assert_not_called()
        experiment.assert_not_called()
        client.assert_not_called()
        network.assert_not_called()
        self.assertFalse((self.root / "blocked").exists())

    def test_assist_saves_exact_context_prompt_snapshot_and_keeps_repair_contract(self):
        results = iter([False, True])

        def compile_once_then_succeed(tex_path, **kwargs):
            return self.compile_result(tex_path, success=next(results))

        with patch("geometry_analyzer.analyze_image", return_value=self.geometry) as analyze, \
             patch("geometry_analyzer.save_geometry_artifacts", side_effect=self.fake_artifacts), \
             patch("main.compile_tex", side_effect=compile_once_then_succeed) as compiler, \
             patch("main.render_pdf", side_effect=lambda pdf, png: png), \
             patch("main.make_comparison", side_effect=lambda original, png, out: out):
            report = self.run_experiment(geometry_mode="assist", geometry_context_max_chars=3000)
        self.assertEqual(report["status"], "completed")
        self.assertFalse(report["quality_verified"])
        self.assertEqual(analyze.call_count, 1)
        self.assertEqual(compiler.call_count, 2)
        paths = report["geometry"]
        context = json.loads(Path(paths["context_json"]).read_text(encoding="utf-8"))
        text = Path(paths["context_text"]).read_text(encoding="utf-8")
        self.assertEqual(text, context["text"])
        self.assertLessEqual(len(text), 3000)
        self.assertEqual(len(text), paths["context_characters"])
        self.assertEqual(context["selected_ids"], paths["selected_ids"])
        compact = json.loads(text.split("```json\n", 1)[1].rsplit("\n```", 1)[0])
        for kind in ("rectangles", "lines", "regions"):
            self.assertEqual([item["id"] for item in compact[kind]], context["selected_ids"][kind])
        self.client.generate.assert_called_once_with(self.image.resolve(), geometry_context=text)
        self.client.repair.assert_called_once_with(self.image.resolve(), VALID, "! Mock compile error")
        first_prompt = Path(report["output_dir"]) / "attempt_01" / "generation_prompt_01.txt"
        self.assertEqual(first_prompt.read_text(encoding="utf-8"), build_generation_prompt(text))
        self.assertFalse((Path(report["output_dir"]) / "attempt_02" / "generation_prompt_02.txt").exists())
        self.assertEqual(Path(report["original_snapshot"]).read_bytes(), self.image_bytes)
        self.assertEqual(self.image.read_bytes(), self.image_bytes)
        self.assertEqual(json.loads(Path(paths["json"]).read_text(encoding="utf-8")), self.geometry)
        self.assertTrue(Path(paths["debug"]).is_file())

    def test_geometry_failure_does_not_silently_fall_back_or_generate(self):
        with patch("main.prepare_geometry", side_effect=GeometryError("mock OpenCV failure")), \
             patch("main.compile_tex") as compiler:
            report = self.run_experiment(geometry_mode="assist")
        self.assertEqual(report["status"], "geometry_failed")
        self.assertIn("mock OpenCV failure", report["error"])
        self.assertEqual(report["attempts"], [])
        self.client.generate.assert_not_called()
        self.client.repair.assert_not_called()
        compiler.assert_not_called()
        saved = json.loads((Path(report["output_dir"]) / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "geometry_failed")

    def test_analyze_failure_is_recorded_without_model_or_compiler(self):
        with patch("main.prepare_geometry", side_effect=GeometryError("mock decode failure")), \
             patch("main.VLMClient") as client, patch("main.compile_tex") as compiler, \
             contextlib.redirect_stdout(io.StringIO()):
            report = main.run_geometry_analysis(self.image, self.root / "analysis-failure")
        self.assertEqual(report["status"], "geometry_failed")
        self.assertFalse(report["api_called"])
        client.assert_not_called()
        compiler.assert_not_called()

    def test_layout_advisories_do_not_replace_compilation_or_repair_rules(self):
        advisory = {"schema_version": "1.0", "advisory_only": True,
                    "summary": {"errors": 1, "warnings": 1},
                    "issues": [{"severity": "warning", "code": "fixture", "line": 1,
                                "message": "mock warning: not an actual overlap finding"}]}
        with patch("main.validate_layout", return_value=advisory), \
             patch("main.compile_tex", side_effect=self.compile_result) as compiler, \
             patch("main.render_pdf", side_effect=lambda pdf, png: png), \
             patch("main.make_comparison", side_effect=lambda original, png, out: out):
            report = self.run_experiment()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(compiler.call_count, 1)
        self.client.repair.assert_not_called()
        layout_path = Path(report["attempts"][0]["layout_report"])
        self.assertEqual(json.loads(layout_path.read_text(encoding="utf-8")), advisory)
        self.assertEqual(report["attempts"][0]["layout_summary"], advisory["summary"])

    def test_modes_requiring_image_do_not_accept_check_or_smoke(self):
        for flag in ("--check", "--smoke-test"):
            with self.subTest(flag=flag), patch("main.VLMClient") as client, \
                 patch("main.run_geometry_analysis") as geometry, \
                 contextlib.redirect_stderr(io.StringIO()):
                status = main.main([flag, "--geometry-mode", "analyze"])
                self.assertEqual(status, 2)
                client.assert_not_called()
                geometry.assert_not_called()


if __name__ == "__main__":
    unittest.main()
