"""Thinking configuration and payload tests; HTTP is always mocked."""
import json
import os
from pathlib import Path
import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from unittest.mock import Mock, patch

from PIL import Image

from vlm_client import (
    APIConfig, APIResponseError, ApiNotAuthorized, ConfigError, OutputConfig,
    ThinkingConfig, VLMClient, response_diagnostics,
)


class ThinkingConfigurationTests(OfflineTestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.official = APIConfig("https://api.deepseek.com", "test-private-key", "deepseek-flash")

    def test_official_default_disables_thinking_without_effort(self):
        settings = ThinkingConfig.from_env(api_config=self.official)
        self.assertEqual(settings.thinking_mode, "disabled")
        self.assertIsNone(settings.reasoning_effort)
        self.assertEqual(settings.request_fields(), {"thinking": {"type": "disabled"}})

    def test_enabled_official_mode_uses_low_effort_by_default(self):
        settings = ThinkingConfig.from_env(api_config=self.official, thinking_mode="enabled")
        self.assertEqual(settings.request_fields(), {
            "thinking": {"type": "enabled"}, "reasoning_effort": "low",
        })

    def test_cli_values_override_environment_and_environment_overrides_default(self):
        env = {"API_THINKING_MODE": "enabled", "API_REASONING_EFFORT": "high"}
        self.assertEqual(ThinkingConfig.from_env(api_config=self.official, environ=env),
                         ThinkingConfig("enabled", "high"))
        self.assertEqual(ThinkingConfig.from_env(api_config=self.official, environ=env,
                                                 thinking_mode="disabled", reasoning_effort="medium"),
                         ThinkingConfig("disabled", "medium"))
        # CLI overrides are evaluated before invalid shadowed environment values.
        self.assertEqual(ThinkingConfig.from_env(environ={"API_THINKING_MODE": "invalid",
                                                          "API_REASONING_EFFORT": "invalid"},
                                                 thinking_mode="enabled", reasoning_effort="low"),
                         ThinkingConfig("enabled", "low"))

    def test_process_environment_is_supported_and_empty_values_use_defaults(self):
        env = {"API_BASE_URL": "https://api.deepseek.com/v1", "MODEL_NAME": "deepseek-flash",
               "API_THINKING_MODE": " ", "API_REASONING_EFFORT": " "}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(ThinkingConfig.from_env(), ThinkingConfig("disabled", None))
        env.update(API_THINKING_MODE="enabled", API_REASONING_EFFORT="medium")
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(ThinkingConfig.from_env(), ThinkingConfig("enabled", "medium"))

    def test_other_services_have_no_implicit_thinking_fields(self):
        for base, model in (("https://proxy.test", "deepseek-flash"),
                            ("https://api.deepseek.com.evil.test", "deepseek-flash"),
                            ("https://api.deepseek.com:8443", "deepseek-flash"),
                            ("https://api.deepseek.com", "different-model")):
            with self.subTest(base=base, model=model):
                config = APIConfig(base, "test-key", model)
                self.assertEqual(ThinkingConfig.from_env(api_config=config).request_fields(), {})

    def test_explicit_fields_are_preserved_independently(self):
        self.assertEqual(ThinkingConfig.from_env(thinking_mode="disabled").request_fields(),
                         {"thinking": {"type": "disabled"}})
        self.assertEqual(ThinkingConfig.from_env(reasoning_effort="low").request_fields(),
                         {"reasoning_effort": "low"})
        # Explicit effort does not switch official disabled mode back on.
        settings = ThinkingConfig.from_env(api_config=self.official, reasoning_effort="high")
        self.assertEqual(settings.request_fields(), {
            "thinking": {"type": "disabled"}, "reasoning_effort": "high",
        })

    def test_invalid_settings_fail_without_echoing_the_value_or_network(self):
        with patch("vlm_client.urllib.request.build_opener") as network:
            for field, bad in (("thinking_mode", "test-private-key"), ("thinking_mode", True),
                               ("thinking_mode", "auto"), ("reasoning_effort", "max"),
                               ("reasoning_effort", 3), ("reasoning_effort", ["low"])):
                with self.subTest(field=field, bad=bad), self.assertRaises(ConfigError) as caught:
                    VLMClient(self.official, allow_api=True, **{field: bad})
                self.assertNotIn("test-private-key", str(caught.exception))
            for name in ("API_THINKING_MODE", "API_REASONING_EFFORT"):
                with self.subTest(name=name), self.assertRaises(ConfigError):
                    ThinkingConfig.from_env(environ={name: "unsupported"})
            network.assert_not_called()


class ThinkingPayloadTests(OfflineTestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "outputs")
        self.addCleanup(directory.cleanup)
        self.image = Path(directory.name) / "fixture.png"
        Image.new("RGB", (8, 8), "white").save(self.image)
        self.key = "test-private-api-key"
        self.official = APIConfig("https://api.deepseek.com", self.key, "deepseek-flash")
        self.generic = APIConfig("https://example.test/v1", self.key, "configured-model")

    @staticmethod
    def response(reason="stop", content=r"\begin{tikzpicture}\node{测试};\end{tikzpicture}"):
        return {"choices": [{"finish_reason": reason, "message": {"content": content}}],
                "usage": {"completion_tokens": 20,
                          "completion_tokens_details": {"reasoning_tokens": 0}}}

    def opener(self, body):
        reply = Mock()
        reply.__enter__ = Mock(return_value=reply)
        reply.__exit__ = Mock(return_value=False)
        reply.read.return_value = json.dumps(body).encode("utf-8")
        opener = Mock()
        opener.open.return_value = reply
        return opener

    def test_official_default_payload_and_diagnostics(self):
        opener = self.opener(self.response())
        client = VLMClient(self.official, allow_api=True)
        with patch("vlm_client.urllib.request.build_opener", return_value=opener):
            client.generate(self.image)
        payload = json.loads(opener.open.call_args.args[0].data)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(payload["max_tokens"], 65536)
        self.assertEqual(client.last_diagnostics["requested_thinking_mode"], "disabled")
        self.assertIsNone(client.last_diagnostics["requested_reasoning_effort"])
        self.assertIs(client.last_diagnostics["thinking_enabled"], False)

    def test_generation_and_repair_use_same_explicit_fields(self):
        for effort in ("low", "medium", "high"):
            with self.subTest(effort=effort):
                opener = self.opener(self.response())
                client = VLMClient(self.official, allow_api=True, thinking_mode="enabled",
                                   reasoning_effort=effort)
                with patch("vlm_client.urllib.request.build_opener", return_value=opener):
                    client.generate(self.image)
                    client.repair(self.image, "old TikZ", "compile error")
                self.assertEqual(opener.open.call_count, 2)
                for call in opener.open.call_args_list:
                    payload = json.loads(call.args[0].data)
                    self.assertEqual(payload["thinking"], {"type": "enabled"})
                    self.assertEqual(payload["reasoning_effort"], effort)
                self.assertTrue(client.last_diagnostics["thinking_enabled"])
                self.assertEqual(client.last_diagnostics["requested_reasoning_effort"], effort)

    def test_generic_payload_omits_unconfigured_fields(self):
        for kwargs, expected in (({}, {}), ({"thinking_mode": "enabled"},
                                             {"thinking": {"type": "enabled"}}),
                                  ({"reasoning_effort": "medium"}, {"reasoning_effort": "medium"})):
            with self.subTest(kwargs=kwargs):
                opener = self.opener(self.response())
                client = VLMClient(self.generic, allow_api=True, **kwargs)
                with patch("vlm_client.urllib.request.build_opener", return_value=opener):
                    client.generate(self.image)
                payload = json.loads(opener.open.call_args.args[0].data)
                self.assertEqual({key: payload[key] for key in ("thinking", "reasoning_effort") if key in payload},
                                 expected)
                if "thinking_mode" not in kwargs:
                    self.assertIsNone(client.last_diagnostics["thinking_enabled"])

    def test_thinking_configuration_does_not_authorize_network(self):
        with patch("vlm_client.urllib.request.build_opener") as network:
            client = VLMClient(self.official, thinking_mode="enabled", reasoning_effort="low")
            with self.assertRaises(ApiNotAuthorized):
                client.generate(self.image)
            network.assert_not_called()

    def test_length_with_all_reasoning_remains_failure_and_records_requested_mode(self):
        body = self.response("length", "")
        body["usage"] = {"completion_tokens": 32768,
                         "completion_tokens_details": {"reasoning_tokens": 32768}}
        body["choices"][0]["message"]["reasoning_content"] = "private reasoning " + self.key
        opener = self.opener(body)
        client = VLMClient(self.official, allow_api=True, max_tokens=32768, thinking_mode="disabled",
                           save_truncated_response=True)
        with patch("vlm_client.urllib.request.build_opener", return_value=opener), \
             self.assertRaises(APIResponseError) as caught:
            client.generate(self.image)
        self.assertEqual(opener.open.call_count, 1)
        message = str(caught.exception)
        for text in ("finish_reason=length", "max_tokens=32768", "completion_tokens=32768",
                     "reasoning_tokens=32768", "thinking=disabled", "reasoning_effort=omitted"):
            self.assertIn(text, message)
        self.assertEqual(client.last_diagnostics["finish_reason"], "length")
        self.assertIs(client.last_diagnostics["thinking_enabled"], False)
        self.assertTrue(client.last_diagnostics["reasoning_content_present"])
        self.assertEqual(client.last_diagnostics["visible_content_characters"], 0)
        saved = json.dumps(client.last_diagnostics) + message + (client.last_truncated_text or "")
        self.assertNotIn(self.key, saved)
        self.assertNotIn("private reasoning", saved)

    def test_length_never_accepts_even_complete_looking_tikz(self):
        opener = self.opener(self.response("length"))
        client = VLMClient(self.official, allow_api=True, thinking_mode="enabled")
        with patch("vlm_client.urllib.request.build_opener", return_value=opener), \
             self.assertRaises(APIResponseError) as caught:
            client.generate(self.image)
        self.assertIn("thinking=enabled", str(caught.exception))
        self.assertIn("reasoning_effort=low", str(caught.exception))

    def test_metadata_without_requested_fields_keeps_state_unknown(self):
        diagnostics = response_diagnostics(self.response(), self.generic, OutputConfig())
        self.assertIsNone(diagnostics["requested_thinking_mode"])
        self.assertIsNone(diagnostics["requested_reasoning_effort"])
        self.assertIsNone(diagnostics["thinking_enabled"])


if __name__ == "__main__":
    unittest.main()
