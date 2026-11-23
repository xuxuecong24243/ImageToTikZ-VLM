"""Small, explicitly authorized, image-bearing Chat Completions client.

Only standard-library HTTP is used. There is no network retry and no implicit
model selection. API keys and image payloads are never logged by this module.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import ipaddress
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union
import urllib.error
import urllib.parse
import urllib.request

from prompts import build_generation_prompt, build_repair_prompt
from response_capture import capture_response, save_diagnostics

MAX_RESPONSE_BYTES = 4 * 1024 * 1024

# Verified 2026-11-12, scoped to the official host and exact model only:
# https://api-docs.deepseek.com/api/create-chat-completion/
# https://api-docs.deepseek.com/quick_start/pricing/
DEEPSEEK_FLASH_DEFAULT_OUTPUT = 65536
DEEPSEEK_FLASH_MAX_OUTPUT = 393216


class ConfigError(ValueError):
    """Missing or invalid API configuration; no request has been sent."""


PROJECT_ENV_FILE = Path(__file__).resolve().parent / ".env"
_CONFIG_NAMES = frozenset({
    "API_BASE_URL", "API_BASE", "API_KEY", "MODEL_NAME",
    "API_MAX_OUTPUT_TOKENS", "API_TOKEN_LIMIT_PARAMETER", "API_MODEL_OUTPUT_LIMIT",
    "API_THINKING_MODE", "API_REASONING_EFFORT",
})


def _canonical_config(values: Mapping[str, str]) -> dict:
    result = {key: value for key, value in values.items() if key in _CONFIG_NAMES}
    # An explicitly set canonical name wins, including an empty value. This
    # lets callers clear a file setting without silently selecting another host.
    if "API_BASE_URL" not in result and "API_BASE" in result:
        result["API_BASE_URL"] = result["API_BASE"]
    return result


def _project_config_values() -> dict:
    """Read a small literal NAME=value file; never execute or interpolate it."""
    try:
        with PROJECT_ENV_FILE.open("rb") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ConfigError("Project .env exceeds the 64 KiB limit.")
        text = raw.decode("utf-8-sig")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError):
        # Do not include exception byte excerpts, lines or values: they may be keys.
        raise ConfigError("Cannot read project .env as UTF-8.") from None
    values = {}
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^export\s+", "", line, count=1)
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ConfigError("Invalid project .env assignment at line {}.".format(number))
        if key not in _CONFIG_NAMES:
            continue
        if key in values:
            raise ConfigError("Duplicate project .env setting at line {}.".format(number))
        value = value.strip()
        if value.startswith(("'", '"')):
            match = re.fullmatch(r'''(['"])(.*?)\1\s*(?:#.*)?''', value)
            if match is None:
                raise ConfigError("Invalid project .env quoting at line {}.".format(number))
            value = match.group(2)
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        values[key] = value
    return _canonical_config(values)


def read_config_environment(environ: Optional[Mapping[str, str]] = None) -> dict:
    """Explicit mapping is isolated; otherwise process values override root .env.

    No cwd search, .env.example loading, OS environment mutation or API access.
    Loading credentials never grants the separate --allow-api authorization.
    """
    if environ is not None:
        return _canonical_config(environ)
    values = _project_config_values()
    values.update(_canonical_config(os.environ))
    return values


class APIError(RuntimeError):
    """An authorized request could not produce a usable response."""


class ApiNotAuthorized(APIError):
    """Real API access was not explicitly enabled by the caller."""


class APIResponseError(APIError):
    """Refused, truncated, malformed or empty model response."""


def _validate_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value or any(ch.isspace() for ch in value):
        raise ConfigError("API_BASE_URL must be a nonempty HTTP(S) endpoint without whitespace.")
    try:
        parts = urllib.parse.urlsplit(value)
        # Accessing .port also validates malformed ports.
        _ = parts.port
    except ValueError as exc:
        raise ConfigError("API_BASE_URL is not a valid URL.") from exc
    if not parts.hostname or parts.username is not None or parts.password is not None:
        raise ConfigError("API_BASE_URL must have a hostname and must not contain credentials.")
    if parts.query or parts.fragment:
        raise ConfigError("API_BASE_URL must not contain query parameters or a fragment.")
    local = parts.hostname.lower() == "localhost"
    try:
        local = local or ipaddress.ip_address(parts.hostname).is_loopback
    except ValueError:
        pass
    if parts.scheme != "https" and not (parts.scheme == "http" and local):
        raise ConfigError("Use HTTPS for API_BASE_URL; HTTP is allowed only for loopback servers.")
    return value


@dataclass(frozen=True)
class APIConfig:
    base_url: str
    api_key: str = field(repr=False)
    model: str

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ConfigError("API_KEY is required; no API request will be sent.")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ConfigError("MODEL_NAME is required; this project does not choose a model.")
        if not isinstance(self.base_url, str):
            raise ConfigError("API_BASE_URL is required.")
        if "\r" in self.api_key or "\n" in self.api_key:
            raise ConfigError("API_KEY must not contain line breaks.")
        object.__setattr__(self, "base_url", _validate_base_url(self.base_url))
        object.__setattr__(self, "api_key", self.api_key.strip())
        object.__setattr__(self, "model", self.model.strip())

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "APIConfig":
        values = read_config_environment(environ)
        names = ("API_BASE_URL", "API_KEY", "MODEL_NAME")
        missing = [name for name in names if not str(values.get(name, "")).strip()]
        if missing:
            raise ConfigError("Missing API configuration: " + ", ".join(missing))
        return cls(values["API_BASE_URL"], values["API_KEY"], values["MODEL_NAME"])

    @property
    def chat_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"


def _is_official_flash(api_config: Optional[APIConfig], values: Mapping[str, str]) -> bool:
    """Scope provider-specific defaults to the verified host and exact model."""
    base_url = api_config.base_url if api_config is not None else values.get("API_BASE_URL", "")
    model = api_config.model if api_config is not None else values.get("MODEL_NAME", "").strip()
    try:
        parsed = urllib.parse.urlsplit(base_url)
        return (parsed.scheme == "https" and parsed.hostname == "api.deepseek.com"
                and parsed.port in {None, 443} and model == "deepseek-flash")
    except ValueError:
        return False


@dataclass(frozen=True)
class ThinkingConfig:
    """Optional request fields; None means the field is not sent to the service."""

    thinking_mode: Optional[str] = None
    reasoning_effort: Optional[str] = None

    def __post_init__(self) -> None:
        if self.thinking_mode not in (None, "enabled", "disabled"):
            raise ConfigError("Thinking mode must be enabled or disabled.")
        if self.reasoning_effort not in (None, "low", "medium", "high"):
            raise ConfigError("Reasoning effort must be low, medium or high.")

    @classmethod
    def from_env(cls, *, thinking_mode=None, reasoning_effort=None,
                 environ: Optional[Mapping[str, str]] = None,
                 api_config: Optional[APIConfig] = None) -> "ThinkingConfig":
        values = read_config_environment(environ)
        official_flash = _is_official_flash(api_config, values)
        mode = (thinking_mode if thinking_mode is not None else
                values.get("API_THINKING_MODE", "").strip() or
                ("disabled" if official_flash else None))
        effort = (reasoning_effort if reasoning_effort is not None else
                  values.get("API_REASONING_EFFORT", "").strip() or
                  ("low" if official_flash and mode == "enabled" else None))
        return cls(mode, effort)

    def request_fields(self) -> dict:
        fields = {}
        if self.thinking_mode is not None:
            fields["thinking"] = {"type": self.thinking_mode}
        if self.reasoning_effort is not None:
            fields["reasoning_effort"] = self.reasoning_effort
        return fields


@dataclass(frozen=True)
class OutputConfig:
    """An output budget, distinct from a provider's total context window."""

    max_tokens: int = 12000
    parameter: str = "max_tokens"
    model_output_limit: Optional[int] = None

    def __post_init__(self) -> None:
        for name, value in (("max_tokens", self.max_tokens),
                            ("model_output_limit", self.model_output_limit)):
            if value is None and name == "model_output_limit":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(name + " must be a positive integer.")
        if self.parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ConfigError("Token limit parameter must be max_tokens or max_completion_tokens.")
        if self.model_output_limit is not None and self.max_tokens > self.model_output_limit:
            raise ConfigError("Requested output tokens exceed the configured model output limit.")

    @classmethod
    def from_env(cls, *, max_tokens=None, parameter=None, model_output_limit=None,
                 environ: Optional[Mapping[str, str]] = None,
                 api_config: Optional[APIConfig] = None) -> "OutputConfig":
        values = read_config_environment(environ)
        official_flash = _is_official_flash(api_config, values)

        def number(explicit, variable, fallback):
            if explicit is not None:
                return explicit
            raw = values.get(variable, "").strip()
            if not raw:
                return fallback
            if not re.fullmatch(r"[0-9]+", raw):
                raise ConfigError(variable + " must be a positive integer.")
            return int(raw)

        output = cls(
            max_tokens=number(max_tokens, "API_MAX_OUTPUT_TOKENS",
                              DEEPSEEK_FLASH_DEFAULT_OUTPUT if official_flash else 12000),
            parameter=(parameter if parameter is not None else
                       values.get("API_TOKEN_LIMIT_PARAMETER", "").strip() or "max_tokens"),
            model_output_limit=number(model_output_limit, "API_MODEL_OUTPUT_LIMIT",
                                      DEEPSEEK_FLASH_MAX_OUTPUT if official_flash else None),
        )
        if official_flash:
            if output.model_output_limit > DEEPSEEK_FLASH_MAX_OUTPUT:
                raise ConfigError("Configured limit exceeds the verified deepseek-flash maximum of 393216.")
            if output.parameter != "max_tokens":
                raise ConfigError("The official deepseek-flash API documents max_tokens, not max_completion_tokens.")
        return output


def _redact(text: str, api_key: str) -> str:
    """Scrub known credentials and accidental image echoes before persistence."""
    if api_key:
        text = text.replace(api_key, "[REDACTED_API_KEY]")
    text = re.sub(r"(?i)Bearer\s+[^\s\"'<>]+", "Bearer [REDACTED]", text)
    return re.sub(r"(?i)data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+",
                  "[REDACTED_IMAGE_DATA]", text)


def _response_text(body) -> Optional[str]:
    """Extract visible text only; never store hidden reasoning/tool content."""
    try:
        content = body["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(item["text"] for item in content if isinstance(item, dict)
                         and item.get("type") in {"text", "output_text"}
                         and isinstance(item.get("text"), str))
    return None


def response_diagnostics(body, config: APIConfig, output: OutputConfig,
                         thinking: Optional[ThinkingConfig] = None) -> dict:
    """Allowlist metadata; never persist an entire provider response or request."""
    # No inference about a generic provider's default when request fields are absent.
    thinking = thinking or ThinkingConfig()
    result = {
        "requested_model": _redact(config.model, config.api_key),
        "api_host": urllib.parse.urlsplit(config.base_url).hostname,
        "requested_output_parameter": output.parameter,
        "requested_output_tokens": output.max_tokens,
        "model_output_limit": output.model_output_limit,
        "requested_thinking_mode": thinking.thinking_mode,
        "requested_reasoning_effort": thinking.reasoning_effort,
        "thinking_enabled": (None if thinking.thinking_mode is None else
                             thinking.thinking_mode == "enabled"),
        "finish_reason": None, "usage": {},
    }
    if not isinstance(body, dict):
        return result
    if isinstance(body.get("model"), str):
        result["response_model"] = _redact(body["model"], config.api_key)[:256]
    try:
        reason = body["choices"][0].get("finish_reason")
        if isinstance(reason, str):
            result["finish_reason"] = _redact(reason, config.api_key)[:128]
        message = body["choices"][0].get("message")
        if isinstance(message, dict):
            reasoning = message.get("reasoning_content")
            result["reasoning_content_present"] = isinstance(reasoning, str) and bool(reasoning)
            if isinstance(reasoning, str):
                # Character count is NOT a token estimate. Do not save the reasoning itself.
                result["reasoning_content_characters"] = len(reasoning)
        visible = _response_text(body)
        if visible is not None:
            result["visible_content_characters"] = len(visible)
    except (KeyError, IndexError, TypeError, AttributeError):
        pass
    usage = body.get("usage")
    if isinstance(usage, dict):
        scalar_names = ("prompt_tokens", "completion_tokens", "total_tokens",
                        "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
        detail_names = ("reasoning_tokens", "cached_tokens", "audio_tokens",
                        "accepted_prediction_tokens", "rejected_prediction_tokens")

        def numbers(source, names):
            return {name: source[name] for name in names
                    if isinstance(source.get(name), int) and not isinstance(source[name], bool)
                    and source[name] >= 0}

        result["usage"] = numbers(usage, scalar_names)
        for name in ("completion_tokens_details", "prompt_tokens_details"):
            if isinstance(usage.get(name), dict):
                result["usage"][name] = numbers(usage[name], detail_names)
    count = result["usage"].get("completion_tokens")
    if count is not None:
        result["reported_completion_tokens_reached_requested_limit"] = count >= output.max_tokens
    return result


def validate_image_path(path: Union[str, Path]) -> Path:
    """Validate the actual original image without replacing or resizing it."""
    if path is None or not str(path).strip():
        raise ValueError("An explicit image path is required.")
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise ValueError("Image file does not exist: " + str(image_path))
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise ValueError("Pillow is missing. Install the project's requirements.txt first.") from exc
    try:
        with Image.open(image_path) as image:
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Supported image formats are PNG, JPEG and WebP.")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Use a single-frame image, not an animation.")
            image.verify()
        with Image.open(image_path) as image:
            image.load()
            if image.width < 1 or image.height < 1:
                raise ValueError("Image has no pixels.")
    except (OSError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("Image could not be decoded: " + str(image_path)) from exc
    return image_path


def parse_response(body: Mapping[str, Any]) -> str:
    """Reject incomplete/refused output so it never becomes a success artifact."""
    if not isinstance(body, dict):
        raise APIResponseError("API response is not a JSON object.")
    if "error" in body:
        raise APIResponseError("API returned an error object instead of a completion.")
    try:
        choice = body["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise APIResponseError("API response is missing choices[0].message.") from exc
    if not isinstance(choice, dict) or not isinstance(message, dict):
        raise APIResponseError("API completion and message must be JSON objects.")
    if message.get("refusal"):
        raise APIResponseError("The model refused this request.")
    reason = choice.get("finish_reason")
    if reason == "length":
        raise APIResponseError(
            "Model output was truncated (finish_reason=length); partial output was rejected. "
            "Check usage and the provider's output/context limits before changing the budget."
        )
    if reason == "content_filter":
        raise APIResponseError("Model output was blocked (finish_reason=content_filter).")
    if reason != "stop":
        raise APIResponseError("API did not report a complete text answer (expected finish_reason=stop).")
    content = message.get("content")
    if isinstance(content, list):
        if any(not isinstance(item, dict) or item.get("type") not in {"text", "output_text"}
               or not isinstance(item.get("text"), str) for item in content):
            raise APIResponseError("API returned unsupported non-text content.")
        content = "\n".join(item["text"] for item in content)
    if not isinstance(content, str) or not content.strip():
        raise APIResponseError("API returned empty text content.")
    return content.strip()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not forward authorization or original images to a redirected host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class VLMClient:
    def __init__(
        self,
        config: APIConfig,
        *,
        allow_api: bool = False,
        timeout: float = 180.0,
        max_tokens: Optional[int] = None,
        token_limit_parameter: Optional[str] = None,
        model_output_limit: Optional[int] = None,
        save_truncated_response: bool = False,
        thinking_mode: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        response_directory: Optional[Union[str, Path]] = None,
    ) -> None:
        if not isinstance(config, APIConfig):
            raise ConfigError("Provide a validated APIConfig.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("API timeout must be positive and finite.")
        self.output_config = OutputConfig.from_env(
            max_tokens=max_tokens, parameter=token_limit_parameter,
            model_output_limit=model_output_limit, api_config=config,
        )
        self.thinking_config = ThinkingConfig.from_env(
            thinking_mode=thinking_mode, reasoning_effort=reasoning_effort,
            api_config=config,
        )
        self.config = config
        self.allow_api = allow_api is True
        self.timeout = timeout
        self.max_tokens = self.output_config.max_tokens
        self.save_truncated_response = save_truncated_response is True
        self.last_diagnostics = {}
        self.last_truncated_text = None
        self.response_directory = response_directory

    def _flush_diagnostics(self):
        """Diagnostic writes must never replace a request/response error."""
        capture = self.last_diagnostics.get('capture', {})
        try:
            save_diagnostics(capture, self.last_diagnostics)
        except Exception as exc:
            self.last_diagnostics['diagnostic_write_error'] = type(exc).__name__

    def _fail(self, code, stage, exc, message, *, response_error=True):
        self.last_diagnostics.update(error_code=code, error_stage=stage,
                                     error_type=type(exc).__name__, error_message=message)
        self._flush_diagnostics()
        error = APIResponseError if response_error else APIError
        # Preserve the actual type and safe details in diagnostics, not an unsafe
        # traceback containing URLs, headers, provider text, or credentials.
        raise error(message) from None

    def _headers(self, response):
        status = getattr(response, 'status', None)
        if not isinstance(status, int) or isinstance(status, bool):
            status = getattr(response, 'code', None)
        self.last_diagnostics['http_status'] = status if isinstance(status, int) and not isinstance(status, bool) else None
        try:
            headers = getattr(response, 'headers', {})
            content_type = headers.get('Content-Type')
            if isinstance(content_type, str):
                mime = _redact(content_type.split(';', 1)[0].strip(), self.config.api_key)
                if re.fullmatch(r'[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+', mime):
                    self.last_diagnostics['content_type'] = mime[:128]
            length = headers.get('Content-Length')
            if isinstance(length, str) and re.fullmatch(r'[0-9]{1,16}', length.strip()):
                self.last_diagnostics['declared_content_length'] = int(length)
        except Exception as exc:
            self.last_diagnostics['header_diagnostic_error'] = type(exc).__name__

    def _capture(self, data, complete):
        self.last_diagnostics.update(response_byte_length=len(data),
                                     response_sha256=hashlib.sha256(data).hexdigest(),
                                     response_length_complete=complete)
        try:
            capture = capture_response(data, self.response_directory, self.config.api_key, complete=complete)
        except Exception as exc:
            capture = {'saved': False, 'complete_saved': False, 'recoverable_exact': False,
                       'persistence_error': type(exc).__name__}
        self.last_diagnostics['capture'] = capture
        self.last_diagnostics['response_saved'] = capture.get('saved', False)
        self.last_diagnostics['response_recoverable'] = capture.get('recoverable_exact', False)

    def _read_response(self, response):
        """Return available bytes and a read error separately from HTTP errors."""
        self._headers(response)  # Recorded before read/decode/JSON can fail.
        self.last_diagnostics['read_state'] = 'reading'
        try:
            data = response.read(MAX_RESPONSE_BYTES + 1)
            if not isinstance(data, bytes):
                raise TypeError('Response reader must return bytes.')
        except (OSError, TimeoutError, http.client.HTTPException, ValueError, TypeError) as exc:
            partial = getattr(exc, 'partial', b'')
            data = partial if isinstance(partial, bytes) else b''
            data = data[:MAX_RESPONSE_BYTES + 1]
            self.last_diagnostics['read_state'] = 'partial' if data else 'failed'
            self._capture(data, False)
            return data, exc
        expected = self.last_diagnostics.get('declared_content_length')
        if len(data) > MAX_RESPONSE_BYTES:
            self.last_diagnostics['read_state'] = 'size_limit'
            self._capture(data, False)
            return data, ValueError('Response exceeds the bounded read limit.')
        if expected is not None and len(data) != expected:
            self.last_diagnostics['read_state'] = 'partial'
            self._capture(data, False)
            if len(data) < expected:
                return data, http.client.IncompleteRead(data, expected - len(data))
            return data, ValueError('Response byte length exceeds declared Content-Length.')
        self.last_diagnostics['read_state'] = 'complete'
        self._capture(data, True)  # Persist safely BEFORE any outer JSON parsing.
        return data, None

    def _close_response(self, response):
        try:
            response.close()
        except Exception as exc:
            self.last_diagnostics['response_close_error'] = type(exc).__name__

    def generate(self, image_path: Union[str, Path], *, geometry_context: Optional[str] = None) -> str:
        return self._complete(image_path, build_generation_prompt(geometry_context))

    def repair(self, image_path: Union[str, Path], fragment: str, error_log: str) -> str:
        return self._complete(image_path, build_repair_prompt(fragment, error_log))

    def _complete(self, image_path: Union[str, Path], prompt: str) -> str:
        return self.complete_images([image_path], prompt)

    def complete_images(self, image_paths, prompt: str, *, response_format=None) -> str:
        """Send one original image, optionally followed by an ID annotation image.

        This is the same authorized completion path used by generate/repair.
        It returns visible text; callers validate their own output schema.
        """
        self.last_diagnostics = {'http_status': None, 'content_type': None,
                                 'response_byte_length': None, 'response_sha256': None,
                                 'read_state': 'not_started', 'response_saved': False,
                                 'response_recoverable': False,
                                 'error_code': None, 'error_stage': None, 'error_type': None}
        self.last_truncated_text = None
        if not self.allow_api:
            raise ApiNotAuthorized(
                "Real API access is disabled. Explicitly authorize paid/network calls with --allow-api."
            )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Completion prompt must be nonempty text.")
        if response_format is not None and response_format != {"type": "json_object"}:
            raise ValueError("Only the explicitly supported json_object response format is allowed.")
        if not isinstance(image_paths, (list, tuple)) or not 1 <= len(image_paths) <= 2:
            raise ValueError("Provide one or two image paths in a list or tuple.")
        from PIL import Image

        content = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            original = validate_image_path(image_path)
            with Image.open(original) as image:
                mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[image.format]
            # Preserve each image's original bytes and the caller's explicit order.
            data_uri = "data:" + mime + ";base64," + base64.b64encode(original.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": data_uri}})
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
            self.output_config.parameter: self.max_tokens,
            "stream": False,
        }
        payload.update(self.thinking_config.request_fields())
        if response_format is not None:
            payload["response_format"] = dict(response_format)
        try:
            request = urllib.request.Request(
                self.config.chat_url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json", "Accept": "application/json",
                         "Authorization": "Bearer " + self.config.api_key},
                method="POST",
            )
        except UnicodeError as exc:
            self._fail('network_error', 'request_encoding', exc,
                       'Request encoding failed before a response was obtained.', response_error=False)
        opener = urllib.request.build_opener(_NoRedirect())
        try:
            # Exactly one HTTP attempt: retries need a separate explicit invocation.
            response = opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            try:
                _, read_error = self._read_response(exc)
                if read_error is not None:
                    self.last_diagnostics['http_body_read_error'] = type(read_error).__name__
            finally:
                self._close_response(exc)
            self._fail('http_error', 'response_http', exc,
                       'API returned HTTP {}; no automatic retry.'.format(exc.code), response_error=False)
        except UnicodeError as exc:
            self._fail('network_error', 'request_encoding', exc,
                       'Request encoding failed before a response was obtained.', response_error=False)
        except (urllib.error.URLError, OSError, TimeoutError, http.client.HTTPException) as exc:
            self._fail('network_error', 'request_open', exc,
                       'API connection failed or timed out; no automatic retry.', response_error=False)
        try:
            raw, read_error = self._read_response(response)
        finally:
            self._close_response(response)
        status = self.last_diagnostics['http_status']
        if status is not None and status >= 400:
            if read_error is not None:
                self.last_diagnostics['http_body_read_error'] = type(read_error).__name__
            self._fail('http_error', 'response_http', urllib.error.HTTPError('', status, '', {}, None),
                       'API returned HTTP {}; no automatic retry.'.format(status), response_error=False)
        if read_error is not None:
            self._fail('response_read_error', 'response_read', read_error,
                       'Response bytes could not be read completely within the size limit.')
        if not raw:
            self._fail('empty_response', 'response_empty', APIResponseError('Empty response'),
                       'API returned an empty HTTP response body.')
        try:
            decoded = raw.decode('utf-8', errors='strict')
        except UnicodeDecodeError as exc:
            self.last_diagnostics['encoding_error'] = {'start': exc.start, 'end': exc.end, 'reason': exc.reason[:160]}
            self._fail('response_encoding_error', 'response_encoding', exc,
                       'HTTP response body is not valid UTF-8.')
        if not decoded.strip():
            self._fail('empty_response', 'response_empty', APIResponseError('Whitespace response'),
                       'API returned a whitespace-only HTTP response body.')
        try:
            body = json.loads(decoded)
        except json.JSONDecodeError as exc:
            self.last_diagnostics['json_error'] = {'line': exc.lineno, 'column': exc.colno,
                                                  'position': exc.pos, 'message': exc.msg[:160]}
            self._fail('response_json_error', 'response_json', exc,
                       'HTTP response body is not valid JSON.')
        except RecursionError as exc:
            self._fail('response_json_error', 'response_json', exc,
                       'HTTP response JSON exceeds the parser nesting limit.')
        self.last_diagnostics.update(response_diagnostics(body, self.config, self.output_config, self.thinking_config))
        self.last_diagnostics['requested_response_format'] = payload.get('response_format')
        if self.last_diagnostics.get("finish_reason") == "length" and self.save_truncated_response:
            visible_text = _response_text(body)
            if visible_text is not None:
                self.last_truncated_text = _redact(visible_text, self.config.api_key)
        try:
            result = parse_response(body)
        except APIResponseError as exc:
            message = str(exc)
            if self.last_diagnostics.get("finish_reason") == "length":
                usage = self.last_diagnostics["usage"]
                details = usage.get("completion_tokens_details", {})
                message = (
                    "{} Requested {}={}; completion_tokens={}; reasoning_tokens={}; "
                    "thinking={}; reasoning_effort={}.".format(
                        str(exc), self.output_config.parameter, self.max_tokens,
                        usage.get("completion_tokens", "not reported"),
                        details.get("reasoning_tokens", "not reported"),
                        self.thinking_config.thinking_mode or "omitted",
                        self.thinking_config.reasoning_effort or "omitted",
                    )
                )
            self._fail('invalid_api_response', 'response_content', exc, message)
        self._flush_diagnostics()
        return result
