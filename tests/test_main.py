"""Orchestration tests use mocked API/compiler/renderers, not real conversions."""
import contextlib
import io
import json
import os
import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

import main
from vlm_client import APIResponseError, OutputConfig, ThinkingConfig


VALID = r"\begin{tikzpicture}\node{中文 $x_1$};\end{tikzpicture}"


class WorkflowTests(OfflineTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=main.PROJECT_DIR / "outputs")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "fixture.png"
        Image.new("RGB", (24, 12), "white").save(self.image)
        self.client = Mock()
        self.client.config.model = "mock-vision-model"
        self.client.generate.return_value = VALID
        self.client.repair.return_value = VALID

    def result(self, tex, success=False, **kwargs):
        tex = Path(tex)
        return SimpleNamespace(success=success, returncode=0 if success else 1,
                               timed_out=False, pdf_path=tex.with_suffix(".pdf") if success else None,
                               log_path=tex.with_suffix(".log"),
                               console_path=tex.with_suffix(".console.txt"),
                               errors="" if success else "! Undefined control sequence. l.20",
                               command=["xelatex", "-no-shell-escape", str(tex)])

    def run_quiet(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return main.run_experiment(self.image, self.root / "results", self.client, **kwargs)

    def test_two_repairs_is_a_hard_upper_limit(self):
        with patch("main.compile_tex", side_effect=self.result) as compile_mock:
            report = self.run_quiet()
        self.assertEqual(self.client.generate.call_count, 1)
        self.assertEqual(self.client.repair.call_count, 2)
        self.assertEqual(compile_mock.call_count, 3)
        self.assertEqual(report["status"], "compile_failed")
        for index, attempt in enumerate(report["attempts"], 1):
            self.assertEqual(Path(attempt["tex"]).name, "generation_%02d.tex" % index)
            self.assertFalse(attempt["compiled"])
            self.assertTrue(Path(attempt["tex"]).exists())
        self.assertTrue((Path(report["output_dir"]) / "run.json").exists())
        first_repair = self.client.repair.call_args_list[0].args
        self.assertEqual(first_repair[0], self.image.resolve())
        self.assertEqual(first_repair[1], VALID)
        self.assertIn("Undefined", first_repair[2])

    def test_successful_second_attempt_stops_repairs(self):
        results = iter([False, True])
        def compile_side_effect(tex, **kwargs):
            return self.result(tex, success=next(results))
        with patch("main.compile_tex", side_effect=compile_side_effect), \
             patch("main.render_pdf", side_effect=lambda pdf, png: png), \
             patch("main.make_comparison", side_effect=lambda a, b, out: out):
            report = self.run_quiet()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(self.client.repair.call_count, 1)
        self.assertFalse(report["quality_verified"])
        self.assertEqual([a["compiled"] for a in report["attempts"]], [False, True])

    def test_zero_repairs_and_invalid_limit(self):
        with patch("main.compile_tex", side_effect=self.result):
            report = self.run_quiet(max_repairs=0)
        self.assertEqual(len(report["attempts"]), 1)
        self.client.repair.assert_not_called()
        with self.assertRaises(ValueError):
            self.run_quiet(max_repairs=3)

    def test_api_failure_preserves_report_without_fabricated_attempt(self):
        self.client.generate.side_effect = RuntimeError("mock API failure")
        with patch("main.compile_tex") as compiler:
            report = self.run_quiet()
        self.assertEqual(report["status"], "generation_failed")
        self.assertEqual(report["attempts"], [])
        compiler.assert_not_called()

    def test_unsafe_output_never_reaches_compiler_or_repair(self):
        self.client.generate.return_value = VALID.replace(r"\node", r"\input{secret}\node")
        with patch("main.compile_tex") as compiler:
            report = self.run_quiet()
        self.assertEqual(report["status"], "validation_failed")
        compiler.assert_not_called()
        self.client.repair.assert_not_called()

    def test_render_failure_keeps_compile_success_and_artifacts(self):
        with patch("main.compile_tex", side_effect=lambda tex, **kw: self.result(tex, True)), \
             patch("main.render_pdf", side_effect=RuntimeError("mock missing renderer")):
            report = self.run_quiet()
        self.assertEqual(report["status"], "rendering_failed")
        self.assertTrue(report["attempts"][0]["compiled"])
        self.client.repair.assert_not_called()

    def test_no_key_blocks_cli_before_client_creation(self):
        with patch.dict(os.environ, {}, clear=True), patch("main.VLMClient") as client, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = main.main(["--image", str(self.image), "--allow-api"])
        self.assertEqual(status, 2)
        client.assert_not_called()

    def test_configured_key_is_not_authorization(self):
        env = {"API_BASE_URL": "https://example.invalid/v1", "API_KEY": "unit-test-key", "MODEL_NAME": "test"}
        with patch.dict(os.environ, env, clear=True), patch("main.VLMClient") as client, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = main.main(["--image", str(self.image)])
        self.assertEqual(status, 2)
        client.assert_not_called()

    def test_repair_api_failure_keeps_first_compile_record(self):
        self.client.repair.side_effect = RuntimeError("mock repair failure")
        with patch("main.compile_tex", side_effect=self.result):
            report = self.run_quiet()
        self.assertEqual(report["status"], "repair_api_failed")
        saved = json.loads((Path(report["output_dir"]) / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(len(saved["attempts"]), 1)

    def test_run_directories_do_not_overwrite_each_other(self):
        first = main.new_run_directory(self.root)
        second = main.new_run_directory(self.root)
        self.assertNotEqual(first, second)

    def test_nonfinite_timeout_rejected_before_any_generation(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.run_quiet(compile_timeout=value)
        self.client.generate.assert_not_called()

    def test_truncation_saves_diagnostics_without_compilation_or_repair(self):
        self.client.output_config = OutputConfig(24000, "max_completion_tokens", 32000)
        self.client.config.base_url = "https://example.test/v1/chat/completions"
        self.client.config.api_key = "private-test-key"
        self.client.save_truncated_response = True
        self.client.last_diagnostics = {
            "finish_reason": "length", "usage": {"completion_tokens": 24000},
            "request": {"max_completion_tokens": 24000},
        }
        self.client.last_truncated_text = r"\begin{tikzpicture}\node{unfinished"
        self.client.generate.side_effect = APIResponseError("Model output was truncated (finish_reason=length)")
        with patch("main.compile_tex") as compiler:
            report = self.run_quiet()
        compiler.assert_not_called()
        self.client.repair.assert_not_called()
        self.assertEqual(report["status"], "generation_failed")
        self.assertEqual(report["attempts"], [])
        self.assertFalse(report["quality_verified"])
        self.assertEqual(report["output_config"]["max_tokens"], 24000)
        self.assertEqual(report["api_host"], "example.test")
        record = report["api_requests"][0]
        diagnostics = json.loads(Path(record["diagnostics"]).read_text(encoding="utf-8"))
        self.assertEqual(diagnostics["usage"]["completion_tokens"], 24000)
        partial = Path(record["partial_response"])
        self.assertEqual(partial.name, "response_01.partial.txt")
        self.assertEqual(partial.read_text(encoding="utf-8"), self.client.last_truncated_text)
        self.assertFalse(partial.with_name("generation_01.tex").exists())
        self.assertFalse(partial.with_name("response_01.txt").exists())
        saved = (Path(report["output_dir"]) / "run.json").read_text(encoding="utf-8")
        self.assertNotIn("private-test-key", saved)
        self.assertNotIn("https://example.test/v1", saved)

    def test_partial_text_is_not_saved_without_opt_in(self):
        self.client.save_truncated_response = False
        self.client.last_diagnostics = {"finish_reason": "length", "usage": {"completion_tokens": 12000}}
        # Defense in depth: even a misbehaving client cannot make main persist it without opt-in.
        self.client.last_truncated_text = "not authorized for persistence"
        self.client.generate.side_effect = APIResponseError("finish_reason=length")
        report = self.run_quiet()
        record = report["api_requests"][0]
        self.assertTrue(Path(record["diagnostics"]).exists())
        self.assertNotIn("partial_response", record)
        self.assertEqual(list(Path(report["output_dir"]).rglob("*.partial.txt")), [])

    def test_truncated_repair_preserves_previous_compile_and_both_diagnostics(self):
        self.client.save_truncated_response = True
        def generate(image):
            self.client.last_diagnostics = {"finish_reason": "stop", "usage": {"completion_tokens": 30}}
            self.client.last_truncated_text = None
            return VALID
        def repair(*args):
            self.client.last_diagnostics = {"finish_reason": "length", "usage": {"completion_tokens": 12000}}
            self.client.last_truncated_text = "incomplete repair"
            raise APIResponseError("finish_reason=length")
        self.client.generate.side_effect = generate
        self.client.repair.side_effect = repair
        with patch("main.compile_tex", side_effect=self.result) as compiler:
            report = self.run_quiet()
        self.assertEqual(compiler.call_count, 1)
        self.assertEqual(self.client.repair.call_count, 1)
        self.assertEqual(report["status"], "repair_api_failed")
        self.assertEqual(len(report["attempts"]), 1)
        self.assertEqual(len(report["api_requests"]), 2)
        first, second = report["api_requests"]
        self.assertNotIn("partial_response", first)
        self.assertEqual(Path(second["partial_response"]).name, "response_02.partial.txt")


class CommandLineOutputTests(OfflineTestCase):
    def setUp(self):
        self.env = {"API_BASE_URL": "https://example.test/v1", "API_KEY": "secret-test-key",
                    "MODEL_NAME": "chosen-vision-model"}

    def check(self, args=(), env=None):
        stream = io.StringIO()
        with patch.dict(os.environ, self.env if env is None else env, clear=True), \
             patch("main.VLMClient") as client, \
             contextlib.redirect_stdout(stream), contextlib.redirect_stderr(io.StringIO()):
            status = main.main(["--check", *args])
        client.assert_not_called()
        return status, stream.getvalue()

    def test_check_resolves_defaults_without_requesting_or_exposing_secrets(self):
        status, output = self.check()
        self.assertEqual(status, 0)
        data = json.loads(output)
        self.assertEqual(data["output_config"], {
            "parameter": "max_tokens", "max_tokens": 12000, "model_output_limit": None,
        })
        self.assertEqual(data["api_host"], "example.test")
        self.assertEqual(data["model"], "chosen-vision-model")
        self.assertNotIn("secret-test-key", output)
        self.assertNotIn("https://example.test/v1", output)

    def test_environment_then_cli_precedence_and_alias(self):
        env = dict(self.env, API_MAX_OUTPUT_TOKENS="20000", API_TOKEN_LIMIT_PARAMETER="max_completion_tokens",
                   API_MODEL_OUTPUT_LIMIT="32000")
        status, output = self.check(env=env)
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output)["output_config"], {
            "parameter": "max_completion_tokens", "max_tokens": 20000, "model_output_limit": 32000,
        })
        status, output = self.check(("--max-output-tokens", "24000", "--token-limit-parameter", "max_tokens",
                                     "--model-output-limit", "30000", "--save-api-diagnostics"), env)
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output)["output_config"], {
            "parameter": "max_tokens", "max_tokens": 24000, "model_output_limit": 30000,
        })
        self.assertTrue(json.loads(output)["save_truncated_response"])

    def test_check_still_shows_budget_without_api_credentials(self):
        status, output = self.check(env={"API_MAX_OUTPUT_TOKENS": "16000"})
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output)["output_config"]["max_tokens"], 16000)
        self.assertIn("Missing API configuration", json.loads(output)["api_config"])

    def test_official_flash_profile_is_reported_and_capped_before_network(self):
        env = dict(self.env, API_BASE_URL="https://api.deepseek.com", MODEL_NAME="deepseek-flash")
        status, output = self.check(env=env)
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output)["output_config"], {
            "parameter": "max_tokens", "max_tokens": 65536, "model_output_limit": 393216,
        })
        status, _ = self.check(("--max-tokens", "393217"), env)
        self.assertEqual(status, 2)

    def test_invalid_budget_stops_before_client_creation(self):
        for args in (("--max-tokens", "0"), ("--max-tokens", "32000", "--model-output-limit", "16000")):
            with self.subTest(args=args):
                status, _ = self.check(args)
                self.assertEqual(status, 2)

    def test_resolved_values_reach_client_with_existing_authorization(self):
        report = {"status": "completed", "output_dir": "mock-output"}
        with patch.dict(os.environ, self.env, clear=True), \
             patch("main.validate_image_path", return_value=Path("fixture.png")), \
             patch("main.require_local_tools"), patch("main.run_experiment", return_value=report), \
             patch("main.VLMClient") as client, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = main.main(["--image", "fixture.png", "--allow-api", "--max-tokens", "24000",
                                "--token-limit-parameter", "max_completion_tokens", "--model-output-limit",
                                "32000", "--save-api-diagnostics"])
        self.assertEqual(status, 0)
        self.assertEqual(client.call_count, 1)
        self.assertEqual(client.call_args.kwargs, {
            "allow_api": True, "timeout": 180.0, "max_tokens": 24000,
            "token_limit_parameter": "max_completion_tokens", "model_output_limit": 32000,
            "save_truncated_response": True,
            "thinking_mode": None, "reasoning_effort": None,
        })


class ThinkingCommandLineTests(OfflineTestCase):
    def setUp(self):
        self.env = {"API_BASE_URL": "https://api.deepseek.com", "API_KEY": "secret-test-key",
                    "MODEL_NAME": "deepseek-flash"}

    def check(self, args=(), env=None):
        stream = io.StringIO()
        with patch.dict(os.environ, self.env if env is None else env, clear=True), \
             patch("main.VLMClient") as client, contextlib.redirect_stdout(stream), \
             contextlib.redirect_stderr(io.StringIO()):
            status = main.main(["--check", *args])
        client.assert_not_called()
        self.assertNotIn("secret-test-key", stream.getvalue())
        return status, json.loads(stream.getvalue()) if status == 0 else {}

    def test_official_default_and_generic_unspecified(self):
        status, report = self.check()
        self.assertEqual(status, 0)
        self.assertEqual(report["thinking_config"], {
            "thinking_mode": "disabled", "reasoning_effort": None, "thinking_enabled": False,
        })
        _, report = self.check(env=dict(self.env, API_BASE_URL="https://other.test"))
        self.assertEqual(report["thinking_config"], {
            "thinking_mode": None, "reasoning_effort": None, "thinking_enabled": None,
        })

    def test_cli_over_environment_over_defaults(self):
        env = dict(self.env, API_THINKING_MODE="enabled", API_REASONING_EFFORT="medium")
        _, report = self.check(env=env)
        self.assertEqual(report["thinking_config"], {
            "thinking_mode": "enabled", "reasoning_effort": "medium", "thinking_enabled": True,
        })
        _, report = self.check(("--thinking-mode", "disabled", "--reasoning-effort", "low"), env)
        self.assertEqual(report["thinking_config"], {
            "thinking_mode": "disabled", "reasoning_effort": "low", "thinking_enabled": False,
        })

    def test_explicit_enabled_defaults_to_low(self):
        _, report = self.check(("--thinking-mode", "enabled"))
        self.assertEqual(report["thinking_config"]["reasoning_effort"], "low")

    def test_invalid_environment_stops_before_client_creation(self):
        for name in ("API_THINKING_MODE", "API_REASONING_EFFORT"):
            with self.subTest(name=name):
                status, _ = self.check(env=dict(self.env, **{name: "invalid"}))
                self.assertEqual(status, 2)
        status, report = self.check(("--thinking-mode", "enabled", "--reasoning-effort", "low"),
                                   dict(self.env, API_THINKING_MODE="invalid", API_REASONING_EFFORT="invalid"))
        self.assertEqual(status, 0)
        self.assertEqual(report["thinking_config"]["reasoning_effort"], "low")

    def test_cli_values_reach_the_authorized_client(self):
        with patch.dict(os.environ, dict(self.env, API_THINKING_MODE="disabled",
                                        API_REASONING_EFFORT="high"), clear=True), \
             patch("main.validate_image_path", return_value=Path("fixture.png")), \
             patch("main.require_local_tools"), patch("main.run_experiment", return_value={
                 "status": "generation_failed", "output_dir": "mock-only"}), \
             patch("main.VLMClient") as client, contextlib.redirect_stdout(io.StringIO()):
            status = main.main(["--image", "fixture.png", "--allow-api",
                                "--thinking-mode", "enabled", "--reasoning-effort", "low"])
        self.assertEqual(status, 1)
        self.assertEqual(client.call_args.kwargs["thinking_mode"], "enabled")
        self.assertEqual(client.call_args.kwargs["reasoning_effort"], "low")

    def test_truncation_reports_mode_and_usage_without_compiling(self):
        with tempfile.TemporaryDirectory(dir=main.PROJECT_DIR / "outputs") as folder:
            root = Path(folder)
            image = root / "fixture.png"
            Image.new("RGB", (8, 8), "white").save(image)
            client = Mock()
            client.config.model = "mock-model"
            client.thinking_config = ThinkingConfig("enabled", "high")
            client.output_config = OutputConfig(32768)
            client.generate.side_effect = APIResponseError("finish_reason=length; fixture only")
            client.last_diagnostics = {
                "requested_output_parameter": "max_tokens", "requested_output_tokens": 32768,
                "finish_reason": "length", "requested_thinking_mode": "enabled",
                "requested_reasoning_effort": "high", "thinking_enabled": True,
                "usage": {"completion_tokens": 32768, "completion_tokens_details": {"reasoning_tokens": 32768}},
            }
            stream = io.StringIO()
            with patch("main.compile_tex") as compiler, contextlib.redirect_stdout(stream):
                report = main.run_experiment(image, root / "results", client)
            self.assertEqual(report["status"], "generation_failed")
            self.assertEqual(report["thinking_config"]["thinking_mode"], "enabled")
            self.assertTrue(report["thinking_config"]["thinking_enabled"])
            self.assertIn('"reasoning_tokens": 32768', stream.getvalue())
            self.assertIn('"thinking_mode": "enabled"', stream.getvalue())
            self.assertIn('"reasoning_effort": "high"', stream.getvalue())
            self.assertEqual(report["attempts"], [])
            compiler.assert_not_called()
            client.repair.assert_not_called()
            saved = json.loads(Path(report["api_requests"][0]["diagnostics"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["thinking_enabled"], True)
            self.assertFalse(list(Path(report["output_dir"]).rglob("*.tex")))


if __name__ == "__main__":
    unittest.main()
