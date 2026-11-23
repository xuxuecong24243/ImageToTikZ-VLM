"""Offline response diagnostics: every HTTP attempt is replaced by a mock."""

import hashlib
import io
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

from vlm_client import (
    APIConfig, APIError, APIResponseError, ApiNotAuthorized, VLMClient,
    parse_response,
)


def completion_bytes(text="visible answer"):
    return json.dumps({
        "model": "fixture-model",
        "choices": [{"finish_reason": "stop", "message": {"content": text}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }, ensure_ascii=False).encode("utf-8")


class ResponseFailureDiagnosticsTests(OfflineTestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "fixture.png"
        Image.new("RGB", (3, 3), "white").save(self.image)
        self.capture_dir = self.root / "responses"
        self.key = "fake-diagnostics-api-key-never-real"
        self.config = APIConfig("https://example.test/v1", self.key, "fixture-model")
        self.opener = Mock()
        builder = patch("vlm_client.urllib.request.build_opener", return_value=self.opener)
        self.build_opener = builder.start()
        self.addCleanup(builder.stop)

    def client(self, **kwargs):
        kwargs.setdefault("response_directory", self.capture_dir)
        kwargs.setdefault("allow_api", True)
        return VLMClient(self.config, **kwargs)

    def reply(self, data, *, status=200, content_type="application/json", headers=None):
        reply = Mock()
        reply.status = status
        reply.headers = {"Content-Type": content_type, **(headers or {})}
        reply.read.return_value = data
        reply.__enter__ = Mock(return_value=reply)
        reply.__exit__ = Mock(return_value=False)
        self.opener.open.return_value = reply
        return reply

    def call(self, client):
        return client.complete_images([self.image], "Offline test prompt")

    def assert_failure(self, client, code, stage, *, status=200, error_type=None):
        diagnostic = client.last_diagnostics
        self.assertEqual(diagnostic["error_code"], code)
        self.assertEqual(diagnostic["error_stage"], stage)
        self.assertEqual(diagnostic["http_status"], status)
        if error_type is not None:
            self.assertEqual(diagnostic["error_type"], error_type)
        self.assertIn("content_type", diagnostic)
        self.assertIn("response_byte_length", diagnostic)
        self.assertIn("response_sha256", diagnostic)
        self.assertIn("read_state", diagnostic)
        self.assertEqual(self.opener.open.call_count, 1)
        return diagnostic

    def persisted_text(self):
        return "\n".join(
            path.read_bytes().decode("utf-8", errors="replace")
            for path in self.capture_dir.rglob("*") if path.is_file()
        )

    def test_success_is_saved_exactly_and_can_be_parsed_offline(self):
        content = "visible \u6d4b\u8bd5 answer"
        raw = completion_bytes(content)
        reply = self.reply(raw)
        client = self.client()
        self.assertEqual(self.call(client), content)
        diagnostic = client.last_diagnostics
        self.assertEqual(diagnostic["http_status"], 200)
        self.assertEqual(diagnostic["content_type"], "application/json")
        self.assertEqual(diagnostic["response_byte_length"], len(raw))
        self.assertEqual(diagnostic["response_sha256"], hashlib.sha256(raw).hexdigest())
        capture = diagnostic["capture"]
        self.assertTrue(capture["saved"])
        self.assertTrue(capture["complete_saved"])
        self.assertTrue(capture["recoverable_exact"])
        saved = Path(capture["path"])
        self.assertEqual(saved.read_bytes(), raw)
        self.assertEqual(parse_response(json.loads(saved.read_text(encoding="utf-8"))), content)
        self.opener.open.assert_called_once()
        reply.close.assert_called_once()

    def test_empty_body_is_not_reported_as_json_or_network_error(self):
        self.reply(b"")
        client = self.client()
        with self.assertRaises(APIResponseError):
            self.call(client)
        diagnostic = self.assert_failure(client, "empty_response", "response_empty")
        self.assertEqual(diagnostic["response_byte_length"], 0)
        self.assertEqual(diagnostic["response_sha256"], hashlib.sha256(b"").hexdigest())

    def test_invalid_utf8_keeps_status_and_encoding_exception_type(self):
        raw = b"\xff\xfeinvalid utf8"
        self.reply(raw)
        client = self.client()
        with self.assertRaises(APIResponseError):
            self.call(client)
        diagnostic = self.assert_failure(
            client, "response_encoding_error", "response_encoding", error_type="UnicodeDecodeError")
        self.assertEqual(diagnostic["response_byte_length"], len(raw))
        self.assertEqual(diagnostic["response_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertFalse(diagnostic["capture"]["complete_saved"])
        self.assertFalse(diagnostic["capture"]["recoverable_exact"])

    def test_html_response_is_json_failure_and_not_exact_recovery(self):
        raw = b"<html><script>alert('fixture')</script>Bad gateway</html>"
        self.reply(raw, content_type="text/html; charset=utf-8")
        client = self.client()
        with self.assertRaises(APIResponseError):
            self.call(client)
        diagnostic = self.assert_failure(
            client, "response_json_error", "response_json", error_type="JSONDecodeError")
        self.assertIn("text/html", diagnostic["content_type"])
        self.assertEqual(diagnostic["response_byte_length"], len(raw))
        self.assertFalse(diagnostic["capture"]["complete_saved"])
        self.assertFalse(diagnostic["capture"]["recoverable_exact"])

    def test_truncated_json_keeps_original_json_exception_type(self):
        raw = b'{"choices": [{"message": '
        self.reply(raw)
        client = self.client()
        with self.assertRaises(APIResponseError):
            self.call(client)
        diagnostic = self.assert_failure(
            client, "response_json_error", "response_json", error_type="JSONDecodeError")
        self.assertEqual(diagnostic["response_byte_length"], len(raw))
        self.assertEqual(diagnostic["response_sha256"], hashlib.sha256(raw).hexdigest())

    def test_http_400_and_503_keep_http_status_and_read_error_body_once(self):
        for status in (400, 503):
            with self.subTest(status=status):
                self.opener.reset_mock()
                raw = b'{"error": {"message": "fixture provider failure"}}'
                stream = io.BytesIO(raw)
                error = urllib.error.HTTPError(
                    self.config.chat_url, status, "fixture reason",
                    {"Content-Type": "application/json"}, stream)
                self.opener.open.side_effect = error
                client = self.client()
                with self.assertRaises(APIError):
                    self.call(client)
                diagnostic = self.assert_failure(
                    client, "http_error", "response_http", status=status, error_type="HTTPError")
                self.assertEqual(diagnostic["content_type"], "application/json")
                self.assertEqual(diagnostic["response_byte_length"], len(raw))
                self.assertEqual(diagnostic["response_sha256"], hashlib.sha256(raw).hexdigest())
                self.assertTrue(stream.closed)

    def test_body_read_failure_is_distinct_from_open_failure(self):
        reply = self.reply(b"")
        reply.read.side_effect = OSError("private response read detail " + self.key)
        client = self.client()
        with self.assertRaises(APIError) as caught:
            self.call(client)
        diagnostic = self.assert_failure(
            client, "response_read_error", "response_read", error_type="OSError")
        self.assertNotIn(self.key, str(caught.exception) + json.dumps(diagnostic))
        self.assertNotEqual(diagnostic["read_state"], "complete")
        reply.close.assert_called_once()

    def test_capture_write_failure_cannot_replace_json_failure(self):
        self.reply(b'{"choices":')
        client = self.client()
        with patch("vlm_client.capture_response", side_effect=OSError("write fixture " + self.key)):
            with self.assertRaises(APIResponseError) as caught:
                self.call(client)
        diagnostic = self.assert_failure(
            client, "response_json_error", "response_json", error_type="JSONDecodeError")
        self.assertFalse(diagnostic["capture"]["saved"])
        self.assertNotIn(self.key, str(caught.exception) + json.dumps(diagnostic))

    def test_sensitive_body_headers_and_exception_messages_are_not_persisted(self):
        hidden = "hidden-fixture-reasoning-do-not-save"
        image_data = "data:image/png;base64,VEVTVF9JTUFHRV9EQVRB"
        raw = json.dumps({
            "error": {"message": self.key + " Bearer alternate-secret " + image_data},
            "headers": {"Authorization": "Bearer " + self.key},
            "reasoning_content": hidden,
        }).encode("utf-8")
        self.opener.open.side_effect = urllib.error.HTTPError(
            self.config.chat_url, 400, "exception detail " + self.key,
            {"Content-Type": "application/json", "Authorization": "Bearer " + self.key,
             "Set-Cookie": "session=private-cookie-value"}, io.BytesIO(raw))
        client = self.client()
        with self.assertRaises(APIError) as caught:
            self.call(client)
        diagnostic = self.assert_failure(
            client, "http_error", "response_http", status=400, error_type="HTTPError")
        saved = str(caught.exception) + json.dumps(diagnostic) + self.persisted_text()
        for secret in (self.key, "alternate-secret", image_data, hidden,
                       "private-cookie-value", "Authorization", "Set-Cookie"):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, saved)
        self.assertFalse(diagnostic["capture"]["complete_saved"])
        self.assertFalse(diagnostic["capture"]["recoverable_exact"])

    def test_network_open_failure_has_no_invented_http_status(self):
        self.opener.open.side_effect = urllib.error.URLError("proxy detail " + self.key)
        client = self.client()
        with self.assertRaises(APIError) as caught:
            self.call(client)
        diagnostic = self.assert_failure(
            client, "network_error", "request_open", status=None, error_type="URLError")
        self.assertNotIn(self.key, str(caught.exception) + json.dumps(diagnostic) + self.persisted_text())
        self.assertNotEqual(diagnostic["read_state"], "complete")

    def test_unauthorized_call_never_constructs_http_opener(self):
        client = self.client(allow_api=False)
        with self.assertRaises(ApiNotAuthorized):
            self.call(client)
        self.build_opener.assert_not_called()
        self.opener.open.assert_not_called()

    def test_invalid_api_envelope_is_content_failure_after_json_parse(self):
        for body in ({}, [], {"choices": []}, {"error": {"message": "fixture"}}):
            with self.subTest(body=body):
                self.opener.reset_mock()
                raw = json.dumps(body).encode("utf-8")
                self.reply(raw)
                client = self.client()
                with self.assertRaises(APIResponseError):
                    self.call(client)
                diagnostic = self.assert_failure(
                    client, "invalid_api_response", "response_content", error_type="APIResponseError")
                self.assertEqual(diagnostic["response_byte_length"], len(raw))

    def test_without_capture_directory_failure_still_has_memory_diagnostics(self):
        raw = b'{"choices":'
        self.reply(raw)
        client = self.client(response_directory=None)
        with self.assertRaises(APIResponseError):
            self.call(client)
        diagnostic = self.assert_failure(
            client, "response_json_error", "response_json", error_type="JSONDecodeError")
        self.assertEqual(diagnostic["response_byte_length"], len(raw))
        self.assertFalse(diagnostic["capture"]["saved"])
        self.assertFalse(self.capture_dir.exists())

    def test_capture_failure_does_not_turn_success_into_failure(self):
        self.reply(completion_bytes("still available"))
        client = self.client()
        with patch("vlm_client.capture_response", side_effect=OSError("fixture capture failure")):
            self.assertEqual(self.call(client), "still available")
        diagnostic = client.last_diagnostics
        self.assertEqual(diagnostic["http_status"], 200)
        self.assertFalse(diagnostic["capture"]["saved"])
        self.assertFalse(diagnostic.get("error_code"))
        self.opener.open.assert_called_once()

    def test_two_calls_keep_distinct_captures_without_overwriting(self):
        client = self.client()
        first = completion_bytes("first response")
        second = completion_bytes("second response")
        self.reply(first)
        self.assertEqual(self.call(client), "first response")
        first_path = Path(client.last_diagnostics["capture"]["path"])
        self.reply(second)
        self.assertEqual(self.call(client), "second response")
        second_path = Path(client.last_diagnostics["capture"]["path"])
        self.assertNotEqual(first_path, second_path)
        self.assertEqual(first_path.read_bytes(), first)
        self.assertEqual(second_path.read_bytes(), second)
        self.assertEqual(self.opener.open.call_count, 2)

    def test_cli_prints_actionable_failure_without_retrying(self):
        from region_vlm_middle import main

        report = {
            "status": "failed_no_retry", "output_dir": str(self.root),
            "api_request_count": 1, "compile_count": 0,
            "error_stage": "response_json", "error_type": "JSONDecodeError",
            "http_status": 200, "error": "API returned invalid JSON.",
        }
        stdout = io.StringIO()
        with patch("region_vlm_middle.run_experiment", return_value=report) as run:
            with patch("sys.stdout", stdout):
                self.assertEqual(main([]), 1)
        printed = stdout.getvalue()
        for expected in ("response_json", "JSONDecodeError", "200", report["error"],
                         "Real API attempts: 1", "Compile attempts: 0"):
            self.assertIn(expected, printed)
        run.assert_called_once()
        self.build_opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
