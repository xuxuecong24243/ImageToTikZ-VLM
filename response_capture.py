"""Bounded, private response capture before parsing a provider's JSON.

Only response bytes enter this module: request objects and HTTP headers must
never be passed here. Uncertain sanitization produces a fixed explanatory
preview, rather than a potentially sensitive excerpt. No response JSON is
parsed, and capture failures never replace the original API outcome.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid


MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_PREVIEW_BYTES = 8 * 1024
MAX_DIAGNOSTICS_BYTES = 32 * 1024

_ESCAPE = re.compile(r'\\u[0-9a-fA-F]{4}|\\["\\/bfnrt]')
_ESCAPE_VALUES = {'"': '"', '\\': '\\', '/': '/', 'b': '\b',
                  'f': '\f', 'n': '\n', 'r': '\r', 't': '\t'}
_SENSITIVE_FIELD = re.compile(
    r'(?i)(?:["\']|\b)(?:api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|'
    r'id[-_ ]?token|token|auth|credentials|secret|client[-_ ]?secret|password|passwd|'
    r'authorization|proxy[-_ ]?authorization|'
    r'cookie|set[-_ ]?cookie|headers?|request[-_ ]?headers?|request|messages|'
    r'url|uri|image[-_ ]?url|images?|audio|reasoning(?:[-_ ]?(?:content|details))?|'
    r'thinking|analysis|tool|function|tool[-_ ]?(?:calls?|results?)|function[-_ ]?call|'
    r'arguments)["\']?\s*[:=]'
)
_HIDDEN_CONTENT_TYPE = re.compile(
    r'(?i)["\']?(?:type|role|channel)["\']?\s*:\s*["\']'
    r'(?:reasoning|analysis|thinking|tool(?:_use|_result|_call)?|function)["\']'
)
_HTML = re.compile(r'(?is)^\s*<|<\s*(?:!doctype|html|head|body|script|iframe|'
                   r'meta|form|input)\b')
_BEARER = re.compile(r'(?i)\bBearer\s+[^\s"\'<>]+')
_IMAGE_DATA = re.compile(r'(?i)data:image/')
_URL_CREDENTIALS = re.compile(r'(?i)\b[a-z][a-z0-9+.-]*://[^\s/"\'<>]*@')
_LONG_BASE64 = re.compile(r'(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{256,}={0,2}'
                          r'(?![A-Za-z0-9+/=])')


def _inspection_text(text: str) -> str:
    """Normalize string escapes for detection only, without parsing JSON."""
    def replace(match):
        token = match.group(0)
        return chr(int(token[2:], 16)) if token[1] == 'u' else _ESCAPE_VALUES[token[1]]

    for _ in range(3):
        changed = _ESCAPE.sub(replace, text)
        if changed == text:
            break
        text = changed
    # Combine escaped UTF-16 surrogate pairs for non-ASCII API keys.
    return text.encode('utf-16', errors='surrogatepass').decode('utf-16')


def _payload(data: bytes, api_key: str) -> tuple[bytes, str, bool, bool]:
    """Return safe bytes, reason, redacted, and whether all input is represented."""
    def withheld(reason):
        return (('Response body withheld: ' + reason + '. Full recovery is unavailable.\n')
                .encode('ascii'), reason, True, False)

    if len(data) > MAX_RESPONSE_BYTES:
        return withheld('byte_limit')
    try:
        text = data.decode('utf-8', errors='strict')
    except UnicodeError:
        return withheld('invalid_encoding')

    safe = text
    if api_key:
        # Handle the usual JSON encodings before conservative escape inspection.
        for key in (api_key, json.dumps(api_key, ensure_ascii=False)[1:-1],
                    json.dumps(api_key, ensure_ascii=True)[1:-1]):
            safe = safe.replace(key, '[REDACTED_API_KEY]')
    safe = _BEARER.sub('Bearer [REDACTED]', safe)
    try:
        inspected = _inspection_text(safe)
    except UnicodeError:
        return withheld('uncertain_escapes')
    if api_key and api_key in inspected:
        return withheld('escaped_credential')
    if re.search(r'\\u[0-9a-fA-F]{4}', inspected):
        return withheld('uncertain_escapes')
    if _HTML.search(inspected):
        return withheld('html_response')
    # Providers commonly include this key even when reasoning is disabled.
    # Null and empty text carry no hidden content; retain the original bytes.
    inspected = re.sub(r'(?i)["\']reasoning_content["\']\s*:\s*'
                       r'(?:null|""|\'\')(?=\s*[,}])', '', inspected)
    if (_SENSITIVE_FIELD.search(inspected) or _HIDDEN_CONTENT_TYPE.search(inspected)
            or _IMAGE_DATA.search(inspected)
            or _URL_CREDENTIALS.search(inspected) or _LONG_BASE64.search(inspected)):
        return withheld('sensitive_content')
    # Escaped Bearer syntax cannot be safely replaced in the original text.
    if any(match.group(0) != 'Bearer [REDACTED]' for match in _BEARER.finditer(inspected)):
        return withheld('escaped_credential')
    encoded = safe.encode('utf-8')
    if len(encoded) > MAX_RESPONSE_BYTES:
        return withheld('redacted_byte_limit')
    redacted = encoded != data
    return encoded, 'redacted' if redacted else 'full', redacted, True


def _restrict_windows(path: Path, *, directory: bool) -> None:
    """Apply a protected DACL granting only the owner and Local System access."""
    import ctypes
    from ctypes import wintypes

    advapi = ctypes.WinDLL('advapi32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    convert = advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = (wintypes.LPCWSTR, wintypes.DWORD,
                        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.DWORD))
    convert.restype = wintypes.BOOL
    set_security = advapi.SetFileSecurityW
    set_security.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p)
    set_security.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = (ctypes.c_void_p,)
    kernel.LocalFree.restype = ctypes.c_void_p
    inheritance = 'OICI' if directory else ''
    sddl = 'D:P(A;' + inheritance + ';FA;;;OW)(A;' + inheritance + ';FA;;;SY)'
    descriptor = ctypes.c_void_p()
    if not convert(sddl, 1, ctypes.byref(descriptor), None):
        raise OSError('private_acl_unavailable')
    try:
        # DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION.
        if not set_security(str(path), 0x80000004, descriptor):
            raise OSError('private_acl_unavailable')
    finally:
        kernel.LocalFree(descriptor)


def _private_directory(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / ('response_' + uuid.uuid4().hex)
    target.mkdir(mode=0o700, exist_ok=False)
    if os.name == 'nt':
        _restrict_windows(target, directory=True)
    else:
        target.chmod(0o700)
        if stat.S_IMODE(target.stat().st_mode) != 0o700:
            raise OSError('private_directory_unavailable')
    return target


def _private_write(path: Path, content: bytes) -> None:
    # Exclusivity prevents replacing an existing artifact. Nothing is written
    # until file permissions have been applied successfully.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, 'O_BINARY', 0), 0o600)
    try:
        if os.name == 'nt':
            _restrict_windows(path, directory=False)
        else:
            os.fchmod(descriptor, 0o600)
            if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
                raise OSError('private_file_unavailable')
        with os.fdopen(descriptor, 'wb') as stream:
            descriptor = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor is not None:
            os.close(descriptor)


def capture_response(data: bytes, directory: Path | None, api_key: str,
                     *, complete: bool = True) -> dict:
    """Capture bounded sanitized bytes; metadata refers to original bytes too.

    ``complete_saved`` describes a complete sanitized HTTP body. Only
    ``recoverable_exact`` additionally guarantees original bytes were saved.
    Invalid encoding, sensitive/uncertain content and over-limit bodies receive
    fixed previews with both recovery flags false. No directory means no I/O.
    """
    capture = {
        'format_version': 1,
        'sha256': hashlib.sha256(data).hexdigest(),
        'byte_length': len(data),
        'read_state': 'complete' if complete else 'partial',
        'saved': False, 'path': None, 'directory': None, 'diagnostics_path': None,
        'saved_sha256': None, 'saved_byte_length': 0,
        'preview': '', 'preview_truncated': False, 'preview_complete': False,
        'complete_saved': False, 'redacted': False, 'recoverable_exact': False,
        'capture_reason': None, 'persistence_error': None, 'persistence_error_type': None,
        'diagnostic_write_error': None, 'diagnostic_write_error_type': None,
    }
    try:
        content, reason, redacted, represented = _payload(data, api_key)
        capture.update({
            'capture_reason': reason, 'redacted': redacted,
            'preview': content[:MAX_PREVIEW_BYTES].decode('utf-8', errors='ignore'),
            'preview_truncated': len(content) > MAX_PREVIEW_BYTES,
            'preview_complete': bool(complete and represented and len(content) <= MAX_PREVIEW_BYTES),
        })
        if directory is None:
            return capture
        if not complete:
            # A partial transport read is explicitly only a bounded preview.
            content = capture['preview'].encode('utf-8')
            if represented:
                capture['capture_reason'] = 'partial_read'
            represented = False
        target = _private_directory(Path(directory).expanduser().absolute())
        capture['directory'] = str(target)
        filename = 'response.txt' if represented and complete else 'response.preview.txt'
        path = target / filename
        _private_write(path, content)
        capture.update({
            'saved': True, 'path': str(path),
            'saved_sha256': hashlib.sha256(content).hexdigest(),
            'saved_byte_length': len(content),
            'complete_saved': bool(complete and represented),
            'recoverable_exact': bool(complete and represented and content == data),
        })
    except Exception as exc:
        # Never persist exception messages: paths and platform errors can
        # themselves contain credentials or other sensitive information.
        capture['persistence_error'] = 'response_capture_failed'
        capture['persistence_error_type'] = type(exc).__name__[:64]
    return capture


_NUMERIC_DIAGNOSTICS = {
    'requested_output_tokens', 'model_output_limit', 'http_status',
    'reasoning_content_characters', 'visible_content_characters',
    'response_byte_length', 'declared_content_length',
}
_BOOLEAN_DIAGNOSTICS = {
    'thinking_enabled', 'reasoning_content_present',
    'reported_completion_tokens_reached_requested_limit',
    'response_length_complete', 'response_saved', 'response_recoverable',
}
_ENUM_DIAGNOSTICS = {
    'requested_output_parameter': {'max_tokens', 'max_completion_tokens'},
    'requested_thinking_mode': {'enabled', 'disabled'},
    'requested_reasoning_effort': {'low', 'medium', 'high'},
    'finish_reason': {'stop', 'length', 'content_filter', 'tool_calls', 'function_call'},
    'error_code': {'network_error', 'http_error', 'response_read_error',
                   'response_encoding_error', 'response_json_error',
                   'invalid_api_response', 'empty_response'},
    'error_stage': {'request_open', 'request_encoding', 'response_read',
                   'response_encoding', 'response_json', 'response_content',
                   'response_http', 'response_empty'},
    'error_type': {'URLError', 'HTTPError', 'TimeoutError', 'OSError', 'ConnectionError',
                  'ConnectionResetError', 'UnicodeEncodeError', 'UnicodeDecodeError',
                  'JSONDecodeError', 'RecursionError', 'IncompleteRead',
                  'RemoteDisconnected', 'BadStatusLine', 'TypeError', 'ValueError',
                  'APIResponseError'},
    'read_state': {'not_started', 'reading', 'failed', 'partial', 'size_limit', 'complete'},
}
_CAPTURE_FIELDS = {
    'format_version', 'sha256', 'byte_length', 'read_state', 'saved', 'path', 'directory',
    'diagnostics_path', 'saved_sha256', 'saved_byte_length', 'preview',
    'preview_truncated', 'preview_complete', 'complete_saved', 'redacted',
    'recoverable_exact', 'capture_reason', 'persistence_error', 'persistence_error_type',
    'diagnostic_write_error', 'diagnostic_write_error_type',
}
_USAGE_FIELDS = {
    'prompt_tokens', 'completion_tokens', 'total_tokens', 'prompt_cache_hit_tokens',
    'prompt_cache_miss_tokens', 'reasoning_tokens', 'cached_tokens', 'audio_tokens',
    'accepted_prediction_tokens', 'rejected_prediction_tokens',
}


def _safe_diagnostics(diagnostics: dict) -> dict:
    """Allowlist metadata; arbitrary strings may echo secrets from a provider."""
    selected = {}
    for name in _NUMERIC_DIAGNOSTICS:
        value = diagnostics.get(name)
        if value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0):
            selected[name] = value
    for name in _BOOLEAN_DIAGNOSTICS:
        value = diagnostics.get(name)
        if value is None or isinstance(value, bool):
            selected[name] = value
    for name, allowed in _ENUM_DIAGNOSTICS.items():
        value = diagnostics.get(name)
        if value is None or (isinstance(value, str) and value in allowed):
            selected[name] = value
        elif name == 'error_type':
            selected[name] = 'unknown'
    mime = diagnostics.get('content_type')
    if (isinstance(mime, str) and len(mime) <= 128
            and re.fullmatch(r'[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+', mime)):
        selected['content_type'] = mime
    digest = diagnostics.get('response_sha256')
    if isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest):
        selected['response_sha256'] = digest
    for field, positions in (('encoding_error', ('start', 'end')),
                             ('json_error', ('line', 'column', 'position'))):
        source = diagnostics.get(field)
        if isinstance(source, dict):
            selected[field] = {name: source[name] for name in positions
                               if isinstance(source.get(name), int)
                               and not isinstance(source[name], bool) and source[name] >= 0}
    if diagnostics.get('requested_response_format') == {'type': 'json_object'}:
        selected['requested_response_format'] = {'type': 'json_object'}
    usage = diagnostics.get('usage')
    if isinstance(usage, dict):
        def numbers(source):
            return {name: value for name, value in source.items()
                    if name in _USAGE_FIELDS and isinstance(value, int)
                    and not isinstance(value, bool) and value >= 0}
        selected['usage'] = numbers(usage)
        for name in ('completion_tokens_details', 'prompt_tokens_details'):
            if isinstance(usage.get(name), dict):
                selected['usage'][name] = numbers(usage[name])
    return selected


def save_diagnostics(capture: dict, diagnostics: dict) -> None:
    """Write an exclusive sidecar using only capture and allowlisted metadata."""
    directory = capture.get('directory')
    if not directory:
        return
    try:
        target = Path(directory)
        # Reassert permissions before every sidecar write; failures are closed.
        if os.name == 'nt':
            _restrict_windows(target, directory=True)
        else:
            target.chmod(0o700)
        path = target / ('diagnostics_' + uuid.uuid4().hex + '.json')
        record = _safe_diagnostics(diagnostics)
        capture['diagnostics_path'] = str(path)
        record['response_capture'] = {name: capture[name] for name in _CAPTURE_FIELDS
                                      if name in capture}
        encoded = (json.dumps(record, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        if len(encoded) > MAX_DIAGNOSTICS_BYTES:
            raise ValueError('diagnostics_byte_limit')
        _private_write(path, encoded)
    except Exception as exc:
        capture['diagnostics_path'] = None
        capture['diagnostic_write_error'] = 'diagnostics_capture_failed'
        capture['diagnostic_write_error_type'] = type(exc).__name__[:64]
