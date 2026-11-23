"""Offline tests only. Tiny images are interface fixtures, not conversion results."""

import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from unittest.mock import Mock, patch
import urllib.error
import urllib.request

from PIL import Image

from prompts import GENERATION_PROMPT, build_repair_prompt
from vlm_client import (
    APIConfig, APIError, APIResponseError, ApiNotAuthorized, ConfigError,
    VLMClient, _NoRedirect, parse_response, validate_image_path,
)


def response(text="\\begin{tikzpicture}\\node {测试};\\end{tikzpicture}", reason="stop"):
    return {"choices": [{"finish_reason": reason, "message": {"content": text}}]}


class ConfigurationTests(OfflineTestCase):
    def test_reads_explicit_environment_and_does_not_print_key(self):
        config = APIConfig.from_env({"API_BASE_URL": "https://example.test/v1/",
                                     "API_KEY": "private-test-key", "MODEL_NAME": "chosen-vision-model"})
        self.assertEqual(config.chat_url, "https://example.test/v1/chat/completions")
        self.assertEqual(config.model, "chosen-vision-model")
        self.assertNotIn("private-test-key", repr(config))

    def test_no_defaults_or_missing_key(self):
        with self.assertRaisesRegex(ConfigError, "API_BASE_URL, API_KEY, MODEL_NAME"):
            APIConfig.from_env({})
        with self.assertRaisesRegex(ConfigError, "API_KEY"):
            APIConfig.from_env({"API_BASE_URL": "https://example.test/v1", "MODEL_NAME": "user-model"})

    def test_default_environ_source(self):
        with patch.dict("os.environ", {"API_BASE_URL": "https://example.test/v1", "API_KEY": "test",
                                       "MODEL_NAME": "user-model"}, clear=True):
            self.assertEqual(APIConfig.from_env().model, "user-model")

    def test_full_endpoint_and_localhost_are_supported(self):
        for url in ("https://example.test/v1/chat/completions", "http://localhost:8080/v1",
                    "http://127.0.0.1:8080/v1", "http://[::1]:8080/v1"):
            with self.subTest(url=url):
                config = APIConfig(url, "test", "user-model")
                self.assertTrue(config.chat_url.endswith("/chat/completions"))

    def test_unsafe_or_secret_bearing_endpoint_rejected(self):
        for url in ("http://remote.test/v1", "https://name:secret@example.test/v1",
                    "https://example.test/v1?key=secret", "file:///tmp/test", "https://example.test/#frag",
                    "http://localhost.remote.test/v1", "https://example.test:bad/v1"):
            with self.subTest(url=url), self.assertRaises(ConfigError):
                APIConfig(url, "private-test-key", "model")


class ResponseTests(OfflineTestCase):
    def test_plain_and_structured_text(self):
        self.assertEqual(parse_response(response("  some TikZ  ")), "some TikZ")
        body = response([{"type": "text", "text": "one"}, {"type": "text", "text": "two"}])
        self.assertEqual(parse_response(body), "one\ntwo")

    def test_truncated_filtered_refused_or_incomplete_output_rejected(self):
        for reason in ("length", "content_filter", "tool_calls", None):
            with self.subTest(reason=reason), self.assertRaises(APIResponseError):
                parse_response(response(reason=reason))
        body = response()
        body["choices"][0]["message"]["refusal"] = "refused"
        with self.assertRaises(APIResponseError):
            parse_response(body)

    def test_malformed_empty_and_error_responses_rejected(self):
        for body in ([], {}, {"error": {"message": "secret echoed"}}, {"choices": []},
                     {"choices": [None]}, response(""), response(None), response([{"type": "image"}])):
            with self.subTest(body=body), self.assertRaises(APIResponseError):
                parse_response(body)


class ImageAndRequestTests(OfflineTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.temp.name)
        self.image = self.root / "中文测试.png"
        Image.new("RGB", (11, 7), "white").save(self.image)
        self.config = APIConfig("https://example.test/v1", "private-test-key", "user-vision-model")

    def tearDown(self):
        self.temp.cleanup()

    def test_image_path_and_actual_decoding(self):
        self.assertEqual(validate_image_path(self.image), self.image.resolve())
        bad = self.root / "invalid.png"
        bad.write_bytes(b"not an image")
        for path in ("", self.root, self.root / "missing.png", bad):
            with self.subTest(path=str(path)), self.assertRaises(ValueError):
                validate_image_path(path)

    def test_disallow_animated_images_and_unsupported_format(self):
        gif = self.root / "image.gif"
        Image.new("RGB", (3, 3)).save(gif)
        with self.assertRaises(ValueError):
            validate_image_path(gif)
        apng = self.root / "animation.png"
        Image.new("RGB", (3, 3), "red").save(
            apng, save_all=True, append_images=[Image.new("RGB", (3, 3), "blue")], duration=20)
        with self.assertRaisesRegex(ValueError, "single-frame"):
            validate_image_path(apng)

    def test_no_network_without_explicit_authorization(self):
        with patch("vlm_client.urllib.request.build_opener") as build:
            client = VLMClient(self.config)
            with self.assertRaises(ApiNotAuthorized):
                client.generate(self.image)
            with self.assertRaises(ApiNotAuthorized):
                client.repair(self.image, "previous", "diagnostic")
            build.assert_not_called()

    def test_generation_and_repair_send_same_original_image_and_model(self):
        reply = Mock()
        reply.__enter__ = Mock(return_value=reply)
        reply.__exit__ = Mock(return_value=False)
        reply.read.return_value = json.dumps(response()).encode("utf-8")
        opener = Mock()
        opener.open.return_value = reply
        with patch("vlm_client.urllib.request.build_opener", return_value=opener):
            client = VLMClient(self.config, allow_api=True)
            client.generate(self.image)
            client.repair(self.image, "old fragment", "Undefined control sequence")
        self.assertEqual(opener.open.call_count, 2)
        for call in opener.open.call_args_list:
            request = call.args[0]
            payload = json.loads(request.data.decode("utf-8"))
            self.assertEqual(payload["model"], "user-vision-model")
            self.assertFalse(payload["stream"])
            self.assertEqual(payload["max_tokens"], 12000)
            image_part = payload["messages"][0]["content"][1]
            uri = image_part["image_url"]["url"]
            self.assertEqual(base64.b64decode(uri.split(",", 1)[1]), self.image.read_bytes())
            self.assertEqual(request.get_header("Authorization"), "Bearer private-test-key")
            self.assertEqual(call.kwargs["timeout"], 180.0)
        repair_payload = json.loads(opener.open.call_args_list[1].args[0].data.decode("utf-8"))
        prompt = repair_payload["messages"][0]["content"][0]["text"]
        self.assertIn("old fragment", prompt)
        self.assertIn("Undefined control sequence", prompt)

    def test_http_errors_do_not_echo_api_key_or_retry(self):
        opener = Mock()
        opener.open.side_effect = urllib.error.HTTPError(
            "https://example.test/v1", 429, "private-test-key", {}, io.BytesIO(b"private-test-key"))
        with patch("vlm_client.urllib.request.build_opener", return_value=opener):
            with self.assertRaises(APIError) as context:
                VLMClient(self.config, allow_api=True).generate(self.image)
        self.assertIn("429", str(context.exception))
        self.assertNotIn("private-test-key", str(context.exception))
        self.assertEqual(opener.open.call_count, 1)

    def test_redirects_do_not_forward_authorization(self):
        handler = _NoRedirect()
        request = urllib.request.Request("https://example.test/v1", headers={"Authorization": "Bearer test"})
        for code in (301, 302, 303, 307, 308):
            self.assertIsNone(handler.redirect_request(request, None, code, "redirect", {}, "https://other.test"))

    def test_provider_error_content_not_used_as_latex(self):
        reply = Mock()
        reply.__enter__ = Mock(return_value=reply)
        reply.__exit__ = Mock(return_value=False)
        reply.read.return_value = b"not-json"
        opener = Mock()
        opener.open.return_value = reply
        with patch("vlm_client.urllib.request.build_opener", return_value=opener), self.assertRaises(APIResponseError):
            VLMClient(self.config, allow_api=True).generate(self.image)


class PromptTests(OfflineTestCase):
    def test_contract_and_reference_numbers(self):
        for text in ("tikzpicture", "documentclass", "includegraphics", "50257", "8×4×256"):
            self.assertIn(text, GENERATION_PROMPT)
        self.assertIn("其他图片没有这些数字时", GENERATION_PROMPT)

    def test_repair_quotes_data_and_limits_diagnostics(self):
        prompt = build_repair_prompt("\\node {中文};", "X" * 17000)
        self.assertIn("只修复", prompt)
        self.assertIn("previous_tikz", prompt)
        self.assertIn("中文", prompt)
        self.assertNotIn("X" * 16001, prompt)


if __name__ == "__main__":
    unittest.main()
