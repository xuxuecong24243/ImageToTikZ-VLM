"""Budget and diagnostic regressions; all HTTP responses are offline fixtures."""
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
import urllib.error

from PIL import Image

from vlm_client import APIConfig, APIError, APIResponseError, ConfigError, OutputConfig, VLMClient


class OutputConfigurationTests(OfflineTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.official = APIConfig("https://api.deepseek.com", "test-only-key", "deepseek-flash")

    def test_verified_official_default_and_generic_fallback(self):
        output = OutputConfig.from_env(api_config=self.official)
        self.assertEqual((output.max_tokens, output.parameter, output.model_output_limit),
                         (65536, "max_tokens", 393216))
        self.assertEqual(OutputConfig.from_env().max_tokens, 12000)
        self.assertIsNone(OutputConfig.from_env().model_output_limit)

    def test_policy_does_not_apply_to_other_hosts_models_or_ports(self):
        for base, model in (("https://api.deepseek.com.evil.test", "deepseek-flash"),
                            ("https://proxy.test", "deepseek-flash"),
                            ("https://api.deepseek.com:8443", "deepseek-flash"),
                            ("https://api.deepseek.com", "another-model")):
            with self.subTest(base=base, model=model):
                output = OutputConfig.from_env(api_config=APIConfig(base, "test", model))
                self.assertEqual(output.max_tokens, 12000)
                self.assertIsNone(output.model_output_limit)
        output = OutputConfig.from_env(environ={"API_BASE_URL": "https://api.deepseek.com/v1",
                                                "MODEL_NAME": "deepseek-flash"})
        self.assertEqual(output.max_tokens, 65536)

    def test_cli_over_environment_over_provider_default(self):
        env = {"API_MAX_OUTPUT_TOKENS": "32768", "API_TOKEN_LIMIT_PARAMETER": "max_tokens",
               "API_MODEL_OUTPUT_LIMIT": "100000"}
        self.assertEqual(OutputConfig.from_env(environ=env, api_config=self.official).max_tokens, 32768)
        output = OutputConfig.from_env(environ=env, api_config=self.official, max_tokens=40000)
        self.assertEqual(output.max_tokens, 40000)
        self.assertEqual(output.model_output_limit, 100000)
        output = OutputConfig.from_env(environ={"API_MAX_OUTPUT_TOKENS": "bad"}, max_tokens=1000)
        self.assertEqual(output.max_tokens, 1000)

    def test_official_cap_cannot_be_bypassed_by_configuration(self):
        for values in ({"max_tokens": 393217},
                       {"max_tokens": 400000, "model_output_limit": 500000},
                       {"model_output_limit": 500000},
                       {"max_tokens": 65536, "model_output_limit": 32768}):
            with self.subTest(values=values), self.assertRaises(ConfigError):
                OutputConfig.from_env(api_config=self.official, **values)
        self.assertEqual(OutputConfig.from_env(api_config=self.official, max_tokens=393216).max_tokens, 393216)

    def test_bad_numeric_and_parameter_settings_fail_before_network(self):
        for number in (0, -1, True, 1.5, "30000"):
            with self.subTest(number=number), self.assertRaises(ConfigError):
                OutputConfig.from_env(max_tokens=number)
        for value in ("0", "-1", "NaN", "1.5", "foo"):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                OutputConfig.from_env(environ={"API_MAX_OUTPUT_TOKENS": value})
        with self.assertRaises(ConfigError):
            OutputConfig.from_env(parameter="max_output_tokens")
        with self.assertRaises(ConfigError):
            OutputConfig.from_env(api_config=self.official, parameter="max_completion_tokens")


class ResponseDiagnosticsTests(OfflineTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "outputs")
        self.addCleanup(self.temp.cleanup)
        self.image = Path(self.temp.name) / "fixture.png"
        Image.new("RGB", (8, 8), "white").save(self.image)
        self.key = "test-private-api-key"
        self.config = APIConfig("https://example.test/v1", self.key, "test-model")

    def body(self, reason="length"):
        return {
            "model": "test-model",
            "choices": [{"finish_reason": reason, "message": {
                "content": "\\begin{tikzpicture}\\node {中文};",
                "reasoning_content": "hidden test reasoning",
            }}],
            "usage": {"prompt_tokens": 2000, "completion_tokens": 12000, "total_tokens": 14000,
                      "completion_tokens_details": {"reasoning_tokens": 11000}},
        }

    def invoke(self, client, body):
        reply = Mock()
        reply.__enter__ = Mock(return_value=reply)
        reply.__exit__ = Mock(return_value=False)
        reply.read.return_value = json.dumps(body, ensure_ascii=False).encode("utf-8")
        opener = Mock()
        opener.open.return_value = reply
        with patch("vlm_client.urllib.request.build_opener", return_value=opener):
            try:
                result = client.generate(self.image)
            except APIResponseError as exc:
                result = exc
        return result, opener

    def test_length_keeps_usage_but_is_always_rejected_without_retry(self):
        client = VLMClient(self.config, allow_api=True, save_truncated_response=True)
        result, opener = self.invoke(client, self.body())
        self.assertIsInstance(result, APIResponseError)
        self.assertIn("finish_reason=length", str(result))
        self.assertIn("reasoning_tokens=11000", str(result))
        self.assertEqual(opener.open.call_count, 1)
        diagnostic = client.last_diagnostics
        self.assertEqual(diagnostic["requested_output_tokens"], 12000)
        self.assertEqual(diagnostic["finish_reason"], "length")
        self.assertEqual(diagnostic["usage"]["completion_tokens"], 12000)
        self.assertTrue(diagnostic["reported_completion_tokens_reached_requested_limit"])
        self.assertIn("中文", client.last_truncated_text)
        self.assertTrue(diagnostic["reasoning_content_present"])
        self.assertNotIn("hidden test reasoning", json.dumps(diagnostic))

    def test_partial_content_requires_explicit_opt_in(self):
        client = VLMClient(self.config, allow_api=True)
        result, _ = self.invoke(client, self.body())
        self.assertIsInstance(result, APIResponseError)
        self.assertIsNone(client.last_truncated_text)
        self.assertEqual(client.last_diagnostics["usage"]["total_tokens"], 14000)

    def test_diagnostics_redact_key_and_image_echoes_and_allowlist_usage(self):
        body = self.body()
        body["model"] = self.key
        body["headers"] = {"Authorization": "Bearer " + self.key}
        body["usage"]["arbitrary"] = self.key
        body["usage"]["completion_tokens_details"]["secret"] = self.key
        body["choices"][0]["message"]["content"] += self.key + " Bearer other-secret data:image/png;base64,AAAA"
        client = VLMClient(self.config, allow_api=True, save_truncated_response=True)
        result, _ = self.invoke(client, body)
        saved = json.dumps(client.last_diagnostics) + client.last_truncated_text + str(result)
        for secret in (self.key, "other-secret", "base64,AAAA", "Authorization", "arbitrary", "hidden test reasoning"):
            self.assertNotIn(secret, saved)

    def test_no_usage_does_not_invent_token_counts(self):
        body = self.body()
        body.pop("usage")
        body["choices"][0]["message"]["content"] = None
        client = VLMClient(self.config, allow_api=True, save_truncated_response=True)
        result, _ = self.invoke(client, body)
        self.assertIsInstance(result, APIResponseError)
        self.assertEqual(client.last_diagnostics["usage"], {})
        self.assertNotIn("reported_completion_tokens_reached_requested_limit", client.last_diagnostics)
        self.assertIsNone(client.last_truncated_text)
        self.assertIn("not reported", str(result))

    def test_success_and_other_reasons_never_create_partial(self):
        for reason in ("stop", "content_filter", "tool_calls"):
            with self.subTest(reason=reason):
                client = VLMClient(self.config, allow_api=True, save_truncated_response=True)
                result, _ = self.invoke(client, self.body(reason))
                self.assertIsNone(client.last_truncated_text)
                self.assertEqual(client.last_diagnostics["finish_reason"], reason)
                self.assertEqual(isinstance(result, str), reason == "stop")

    def test_exactly_one_budget_parameter_in_each_request(self):
        for parameter in ("max_tokens", "max_completion_tokens"):
            with self.subTest(parameter=parameter):
                client = VLMClient(self.config, allow_api=True, max_tokens=32768,
                                   token_limit_parameter=parameter)
                _, opener = self.invoke(client, self.body("stop"))
                payload = json.loads(opener.open.call_args.args[0].data)
                self.assertEqual(payload[parameter], 32768)
                alternate = "max_completion_tokens" if parameter == "max_tokens" else "max_tokens"
                self.assertNotIn(alternate, payload)

    def test_official_request_uses_65536_and_max_tokens(self):
        official = APIConfig("https://api.deepseek.com", self.key, "deepseek-flash")
        client = VLMClient(official, allow_api=True)
        _, opener = self.invoke(client, self.body("stop"))
        payload = json.loads(opener.open.call_args.args[0].data)
        self.assertEqual(payload["max_tokens"], 65536)
        self.assertNotIn("max_completion_tokens", payload)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", payload)

    def test_failed_next_request_cannot_reuse_previous_diagnostics(self):
        client = VLMClient(self.config, allow_api=True, save_truncated_response=True)
        self.invoke(client, self.body())
        self.assertTrue(client.last_diagnostics)
        opener = Mock()
        opener.open.side_effect = urllib.error.URLError("test only")
        with patch("vlm_client.urllib.request.build_opener", return_value=opener), self.assertRaises(APIError):
            client.repair(self.image, "previous", "error")
        self.assertEqual(client.last_diagnostics['error_code'], 'network_error')
        self.assertIsNone(client.last_diagnostics['http_status'])
        self.assertNotIn('usage', client.last_diagnostics)
        self.assertNotIn('finish_reason', client.last_diagnostics)
        self.assertIsNone(client.last_truncated_text)

    def test_non_numeric_usage_is_not_persisted(self):
        body = self.body()
        body["usage"] = {"prompt_tokens": self.key, "completion_tokens": True,
                         "total_tokens": -1, "completion_tokens_details": {"reasoning_tokens": self.key}}
        client = VLMClient(self.config, allow_api=True)
        self.invoke(client, body)
        self.assertEqual(client.last_diagnostics["usage"], {"completion_tokens_details": {}})


if __name__ == "__main__":
    unittest.main()
