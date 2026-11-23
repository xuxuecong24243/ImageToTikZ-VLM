"""Remaining response acceptance checks; HTTP is mocked, sockets/processes forbidden."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import ctypes
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from unittest.mock import Mock, patch

from PIL import Image
import response_capture as capture_module
import region_vlm_middle as middle
import vlm_client
from vlm_client import APIConfig, APIResponseError, VLMClient


ARTIFACT_ROOT = None  # The acceptance runner supplies a new, persistent evidence directory.
ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def completion():
    return b'{"choices":[{"finish_reason":"stop","message":{"content":"ok"}}]}'


class BoundedReply:
    def __init__(self, data, headers=None):
        self.status = 200
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.stream = io.BytesIO(data)
        self.read_sizes = []
        self.bytes_read = 0
        self.closed = False

    def read(self, size):
        self.read_sizes.append(size)
        result = self.stream.read(size)
        self.bytes_read += len(result)
        return result

    def close(self):
        self.closed = True
        self.stream.close()


def windows_dacl_sddl(path):
    """Read the actual stored DACL; do not infer permissions from chmod/stat."""
    from ctypes import wintypes
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    get_security = advapi.GetFileSecurityW
    get_security.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p,
                            wintypes.DWORD, ctypes.POINTER(wintypes.DWORD))
    get_security.restype = wintypes.BOOL
    convert = advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW
    convert.argtypes = (ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                       ctypes.POINTER(wintypes.LPWSTR), ctypes.POINTER(wintypes.DWORD))
    convert.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = (ctypes.c_void_p,)
    kernel.LocalFree.restype = ctypes.c_void_p
    required = wintypes.DWORD()
    get_security(str(path), 4, None, 0, ctypes.byref(required))
    if not required.value:
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor = ctypes.create_string_buffer(required.value)
    if not get_security(str(path), 4, descriptor, required.value, ctypes.byref(required)):
        raise ctypes.WinError(ctypes.get_last_error())
    sddl = wintypes.LPWSTR()
    if not convert(descriptor, 1, 4, ctypes.byref(sddl), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return sddl.value
    finally:
        kernel.LocalFree(ctypes.cast(sddl, ctypes.c_void_p))


class ResponseAcceptanceTests(OfflineTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))
        for name in ("socket.socket.connect", "socket.socket.connect_ex",
                     "socket.create_connection", "socket.getaddrinfo", "subprocess.Popen"):
            self.stack.enter_context(patch(name, side_effect=AssertionError("External I/O forbidden")))
        if ARTIFACT_ROOT is None:
            temp = tempfile.TemporaryDirectory(dir=ROOT)
            self.addCleanup(temp.cleanup)
            self.root = Path(temp.name)
        else:
            self.root = Path(ARTIFACT_ROOT) / self._testMethodName
            self.root.mkdir(parents=True, exist_ok=False)
        self.key = "fake-acceptance-key-never-real"
        self.image = self.root / "fixture.png"
        Image.new("RGB", (3, 3), "white").save(self.image)
        self.capture_dir = self.root / "responses"
        self.opener = Mock()
        self.stack.enter_context(patch("vlm_client.urllib.request.build_opener", return_value=self.opener))

    def client(self):
        return VLMClient(APIConfig("https://example.test/v1", self.key, "fixture"),
                         allow_api=True, response_directory=self.capture_dir)

    def reply(self, raw, headers=None):
        self.opener.reset_mock()
        self.opener.open.side_effect = None
        reply = BoundedReply(raw, headers)
        self.opener.open.return_value = reply
        return reply

    def call(self, client):
        return client.complete_images([self.image], "Offline acceptance fixture")

    def failed(self, client, stage="response_json", error_type="JSONDecodeError"):
        with self.assertRaises(APIResponseError):
            self.call(client)
        diagnostic = client.last_diagnostics
        self.assertEqual(diagnostic["http_status"], 200)
        self.assertEqual(diagnostic["error_stage"], stage)
        self.assertEqual(diagnostic["error_type"], error_type)
        self.opener.open.assert_called_once()
        return diagnostic

    def sidecar(self, diagnostic):
        path = Path(diagnostic["capture"]["diagnostics_path"])
        self.assertLessEqual(path.stat().st_size, capture_module.MAX_DIAGNOSTICS_BYTES)
        return json.loads(path.read_text(encoding="utf-8"))

    def assert_no_secret_files(self):
        for path in self.root.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".txt"}:
                self.assertNotIn(self.key.encode(), path.read_bytes(), str(path))

    def test_response_size_boundaries(self):
        save_limit = capture_module.MAX_RESPONSE_BYTES
        read_limit = vlm_client.MAX_RESPONSE_BYTES
        records = []
        for size in (128, save_limit - 1, save_limit, save_limit + 1,
                     read_limit - 1, read_limit, read_limit + 128):
            with self.subTest(size=size):
                raw = completion() + b" " * (size - len(completion()))
                reply = self.reply(raw, {"Content-Length": str(size)})
                client = self.client()
                if size > read_limit:
                    diagnostic = self.failed(client, "response_read", "ValueError")
                    self.assertEqual(diagnostic["read_state"], "size_limit")
                else:
                    self.assertEqual(self.call(client), "ok")
                    diagnostic = client.last_diagnostics
                    self.assertEqual(diagnostic["read_state"], "complete")
                read_bytes = raw[:read_limit + 1]
                self.assertEqual(reply.read_sizes, [read_limit + 1])
                self.assertEqual(reply.bytes_read, min(size, read_limit + 1))
                self.assertTrue(reply.closed)
                self.opener.open.assert_called_once()
                self.assertEqual(diagnostic["response_byte_length"], len(read_bytes))
                self.assertEqual(diagnostic["response_sha256"], digest(read_bytes))
                capture = diagnostic["capture"]
                self.assertTrue(capture["saved"])
                self.assertEqual(capture["recoverable_exact"], size <= save_limit)
                self.assertEqual(capture["complete_saved"], size <= save_limit)
                saved = Path(capture["path"])
                self.assertLessEqual(saved.stat().st_size, save_limit)
                if size <= save_limit:
                    self.assertEqual(saved.read_bytes(), raw)
                else:
                    self.assertLessEqual(saved.stat().st_size, capture_module.MAX_PREVIEW_BYTES)
                    self.assertEqual(capture["capture_reason"], "byte_limit")
                disk = self.sidecar(diagnostic)
                self.assertEqual(disk["response_byte_length"], len(read_bytes))
                self.assertEqual(disk["response_capture"]["recoverable_exact"], size <= save_limit)
                records.append({"body_bytes": size, "read_bytes": reply.bytes_read,
                                "saved_bytes": saved.stat().st_size,
                                "recoverable_exact": capture["recoverable_exact"]})
        (self.root / "boundary_results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    def test_diagnostics_size_limit_preserves_original_error(self):
        # A real 8 KiB preview expands beyond 32 KiB when JSON escapes controls.
        self.reply(b"\x01" * capture_module.MAX_PREVIEW_BYTES)
        client = self.client()
        diagnostic = self.failed(client)
        capture = diagnostic["capture"]
        self.assertTrue(capture["saved"])
        self.assertIsNone(capture["diagnostics_path"])
        self.assertEqual(capture["diagnostic_write_error_type"], "ValueError")
        self.assertEqual(list(self.capture_dir.rglob("diagnostics_*.json")), [])
        self.assert_no_secret_files()

    def test_response_file_write_failures_are_safe(self):
        real_open = capture_module.os.open
        for failure in ("open", "fsync"):
            with self.subTest(failure=failure):
                self.capture_dir = self.root / failure
                raw = b'{"authorization":"Bearer ' + self.key.encode()  # Invalid and sensitive.
                self.reply(raw)
                client = self.client()
                def failing_open(path, *args, **kwargs):
                    if Path(path).name.startswith("response"):
                        raise PermissionError("fixture " + self.key)
                    return real_open(path, *args, **kwargs)
                injection = (patch("response_capture.os.open", side_effect=failing_open)
                             if failure == "open" else
                             patch("response_capture.os.fsync", side_effect=OSError("fixture " + self.key)))
                with injection:
                    diagnostic = self.failed(client)
                capture = diagnostic["capture"]
                self.assertFalse(capture["saved"])
                self.assertFalse(capture["recoverable_exact"])
                self.assertEqual(capture["persistence_error"], "response_capture_failed")
                for file in self.capture_dir.rglob("response*.txt"):
                    self.assertLessEqual(file.stat().st_size, capture_module.MAX_PREVIEW_BYTES)
                    self.assertIn(b"Response body withheld", file.read_bytes())
                self.assert_no_secret_files()

    def test_diagnostic_file_write_failure_preserves_failure_and_success(self):
        real_open = capture_module.os.open
        def failing_open(path, *args, **kwargs):
            if Path(path).name.startswith("diagnostics_"):
                raise PermissionError("fixture " + self.key)
            return real_open(path, *args, **kwargs)
        for successful in (False, True):
            with self.subTest(successful=successful):
                self.reply(completion() if successful else b'{"choices":')
                client = self.client()
                with patch("response_capture.os.open", side_effect=failing_open):
                    if successful:
                        self.assertEqual(self.call(client), "ok")
                        diagnostic = client.last_diagnostics
                        self.assertIsNone(diagnostic["error_code"])
                    else:
                        diagnostic = self.failed(client)
                capture = diagnostic["capture"]
                self.assertTrue(capture["saved"])
                self.assertTrue(Path(capture["path"]).is_file())
                self.assertIsNone(capture["diagnostics_path"])
                self.assertEqual(capture["diagnostic_write_error_type"], "PermissionError")
                self.assert_no_secret_files()

    def test_permission_failures_prevent_body_writes(self):
        real_restrict = capture_module._restrict_windows
        for fail_directory in (True, False):
            with self.subTest(directory=fail_directory):
                self.capture_dir = self.root / ("directory" if fail_directory else "file")
                self.reply(b'{"authorization":"' + self.key.encode())
                client = self.client()
                if os.name == "nt":
                    def restrict(path, *, directory):
                        if directory == fail_directory:
                            raise PermissionError("fixture " + self.key)
                        return real_restrict(path, directory=directory)
                    injection = patch("response_capture._restrict_windows", side_effect=restrict)
                else:
                    injection = patch("pathlib.Path.chmod" if fail_directory else "os.fchmod",
                                      side_effect=PermissionError("fixture " + self.key))
                with injection:
                    diagnostic = self.failed(client)
                capture = diagnostic["capture"]
                self.assertFalse(capture["saved"])
                self.assertFalse(capture["recoverable_exact"])
                self.assertEqual(capture["persistence_error_type"], "PermissionError")
                self.assertTrue(all(p.stat().st_size == 0 for p in self.capture_dir.rglob("*") if p.is_file()))
                self.assert_no_secret_files()

    def test_incomplete_read_preserves_partial_metadata(self):
        raw = b'{"choices":' + b" " * 9000
        reply = self.reply(b"")
        reply.read = Mock(side_effect=http.client.IncompleteRead(raw, 100))
        client = self.client()
        diagnostic = self.failed(client, "response_read", "IncompleteRead")
        self.assertEqual(diagnostic["read_state"], "partial")
        self.assertFalse(diagnostic["response_length_complete"])
        self.assertEqual(diagnostic["response_byte_length"], len(raw))
        self.assertEqual(diagnostic["response_sha256"], digest(raw))
        self.assertFalse(diagnostic["capture"]["recoverable_exact"])
        self.assertLessEqual(Path(diagnostic["capture"]["path"]).stat().st_size, capture_module.MAX_PREVIEW_BYTES)
        disk = self.sidecar(diagnostic)
        self.assertEqual(disk["error_type"], "IncompleteRead")
        self.assertFalse(disk["response_length_complete"])
        reply.read.assert_called_once_with(vlm_client.MAX_RESPONSE_BYTES + 1)
        self.assertTrue(reply.closed)

    def test_content_length_mismatch_is_never_exact_recovery(self):
        raw = completion()
        for declared in (len(raw) + 1, len(raw) - 1):
            with self.subTest(declared=declared):
                self.reply(raw, {"Content-Length": str(declared)})
                client = self.client()
                diagnostic = self.failed(client, "response_read",
                                         "IncompleteRead" if declared > len(raw) else "ValueError")
                self.assertFalse(diagnostic["response_length_complete"])
                self.assertFalse(diagnostic["capture"]["recoverable_exact"])
                self.assertFalse(self.sidecar(diagnostic)["response_recoverable"])

    def test_truncated_chunked_http_response_is_not_recoverable(self):
        # Exercise stdlib HTTP framing with an in-memory socket, never a real socket.
        socket_fixture = Mock()
        socket_fixture.makefile.return_value = io.BytesIO(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Type: application/json\r\n\r\n"
            b"20\r\n{\"choices\":")
        reply = http.client.HTTPResponse(socket_fixture)
        reply.begin()
        self.opener.open.return_value = reply
        client = self.client()
        diagnostic = self.failed(client, "response_read", "IncompleteRead")
        self.assertFalse(diagnostic["response_length_complete"])
        self.assertFalse(diagnostic["capture"]["recoverable_exact"])
        self.assertFalse(self.sidecar(diagnostic)["response_recoverable"])
        self.assertTrue(reply.isclosed())

    @unittest.skipUnless(os.name == "nt", "Windows DACL requires a native Windows run")
    def test_actual_windows_dacl(self):
        self.reply(completion())
        client = self.client()
        self.assertEqual(self.call(client), "ok")
        capture = client.last_diagnostics["capture"]
        records = []
        for field in ("directory", "path", "diagnostics_path"):
            path = Path(capture[field])
            sddl = windows_dacl_sddl(path)
            self.assertIn("P", sddl.split("(", 1)[0])
            entries = [entry.split(";") for entry in re.findall(r"\(([^()]*)\)", sddl)]
            self.assertEqual(len(entries), 2)
            self.assertEqual({entry[5] for entry in entries}, {"OW", "SY"})
            for entry in entries:
                self.assertEqual(entry[0], "A")
                self.assertIn(entry[2], {"FA", "0x1f01ff"})
                self.assertNotIn("ID", entry[1])
                if field == "directory":
                    self.assertIn("OI", entry[1])
                    self.assertIn("CI", entry[1])
            records.append({"kind": field, "path": str(path), "sddl": sddl})
        (self.root / "permissions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    @unittest.skipUnless(os.name == "posix", "POSIX modes not verified on Windows")
    def test_actual_posix_permissions(self):
        self.reply(completion())
        client = self.client()
        self.assertEqual(self.call(client), "ok")
        capture = client.last_diagnostics["capture"]
        for field, expected in (("directory", 0o700), ("path", 0o600), ("diagnostics_path", 0o600)):
            self.assertEqual(stat.S_IMODE(Path(capture[field]).stat().st_mode), expected)

    def middle_fixture(self):
        original = self.root / "original.png"
        crop = self.root / "region_002.png"
        image = Image.new("RGB", (1536, 1024), "white")
        image.putpixel((500, 100), (15, 40, 80))
        image.save(original)
        image.crop((444, 0, 1050, 1024)).save(crop)
        manifest = self.root / "region_manifest.json"
        manifest.write_text(json.dumps({
            "image": {"path": str(original), "width": 1536, "height": 1024,
                      "sha256": digest(original.read_bytes())},
            "regions": [{"region_id": "region_002", "bbox": [456, 0, 582, 1024],
                         "crop_bbox": [444, 0, 606, 1024], "offset": [444, 0], "crop_path": crop.name}],
        }), encoding="utf-8")
        return original, crop, manifest

    def run_middle_chain(self, *, fail_final_hash=False):
        original, crop, manifest = self.middle_fixture()
        raw = b'{"choices":'
        reply = self.reply(raw)
        output_root = self.root / "runs"
        stdout = io.StringIO()
        real_sha = middle._sha
        protected = {path.resolve() for path in (original, crop, manifest)}
        def checked_sha(path):
            if fail_final_hash and self.opener.open.call_count and Path(path).resolve() in protected:
                raise PermissionError("source fixture " + self.key)
            return real_sha(path)
        environment = {"API_BASE_URL": "https://api.deepseek.com", "API_KEY": self.key,
                       "MODEL_NAME": "deepseek-flash"}
        with patch.dict(os.environ, environment, clear=True), redirect_stdout(stdout):
            with patch("region_vlm_middle.compile_bounded", side_effect=AssertionError("No compilation")) as compiler:
                with patch("region_vlm_middle._sha", side_effect=checked_sha):
                    exit_code = middle.main(["--manifest", str(manifest), "--output-root", str(output_root),
                                             "--baseline-scene", str(self.root / "absent_baseline.json"), "--allow-api"])
                compiler.assert_not_called()
        self.assertEqual(exit_code, 1)
        self.opener.open.assert_called_once()
        self.assertEqual(reply.read_sizes, [vlm_client.MAX_RESPONSE_BYTES + 1])
        reports = list(output_root.glob("*/run_report.json"))
        self.assertEqual(len(reports), 1)
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed_no_retry")
        self.assertEqual(report["api_request_count"], 1)
        self.assertEqual(report["compile_count"], 0)
        self.assertEqual(report["http_status"], 200)
        self.assertEqual(report["error_type"], "JSONDecodeError")
        self.assertEqual(report["error_stage"], "response_json")
        self.assertEqual(report["error_code"], "response_json_error")
        self.assertEqual(report["error"], "HTTP response body is not valid JSON.")
        self.assertEqual(report["response_byte_length"], len(raw))
        self.assertEqual(report["response_sha256"], digest(raw))
        self.assertTrue(report["response_saved"])
        self.assertTrue(report["capture"]["complete_saved"])
        self.assertEqual(Path(report["capture"]["path"]).read_bytes(), raw)
        disk = self.sidecar(report)
        for field in ("http_status", "error_stage", "error_type", "response_sha256"):
            self.assertEqual(disk[field], report[field])
        ledger = json.loads((reports[0].parent / "request_01" / "request.json").read_text(encoding="utf-8"))
        self.assertEqual(ledger["status"], "request_failed")
        self.assertEqual(ledger["diagnostics"]["error_type"], "JSONDecodeError")
        for expected in ("Error stage: response_json", "Error type: JSONDecodeError", "HTTP status: 200",
                         "Error: " + report["error"], "Real API attempts: 1", "Compile attempts: 0"):
            self.assertIn(expected, stdout.getvalue())
        (self.root / "console.txt").write_text(stdout.getvalue(), encoding="utf-8")
        (self.root / "mock_execution.json").write_text(json.dumps({
            "http_mock_attempts": self.opener.open.call_count, "real_api_requests": 0,
            "compile_calls": compiler.call_count, "complete_images_mocked": False,
            "run_experiment_mocked": False, "run_report": str(reports[0]),
        }, indent=2), encoding="utf-8")
        self.assert_no_secret_files()
        return report

    def test_middle_http200_invalid_json_full_chain(self):
        report = self.run_middle_chain()
        self.assertTrue(report["source_files_unchanged"])

    def test_middle_final_hash_failure_preserves_api_error_and_report(self):
        report = self.run_middle_chain(fail_final_hash=True)
        self.assertIsNone(report["source_files_unchanged"])
        self.assertEqual(report["source_verification_error_type"], "PermissionError")


if __name__ == "__main__":
    unittest.main()
