"""Run local XeLaTeX with explicit arguments and retained diagnostic files.

No shell is used. Disabled shell escape and TeX file restrictions are defense
in depth, NOT a security sandbox. Never expose this directly as a public service.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import signal
import subprocess
from typing import List, Optional


@dataclass
class CompileResult:
    success: bool
    returncode: Optional[int]
    timed_out: bool
    pdf_path: Optional[Path]
    log_path: Path
    console_path: Path
    errors: str
    command: List[str]


def build_compile_command(tex_path: Path, xelatex: str = "xelatex") -> List[str]:
    tex_path = Path(tex_path).resolve()
    return [
        str(xelatex), "-no-shell-escape", "-interaction=nonstopmode",
        "-halt-on-error", "-file-line-error", "-recorder",
        "-output-directory", str(tex_path.parent), "./" + tex_path.name,
    ]


def extract_errors(log: str, limit: int = 12000) -> str:
    """Return error lines with nearby context, while full logs stay on disk."""
    lines = log.splitlines()
    selected = set()
    pattern = re.compile(r"^!|^l\.\d+|:\d+:|Emergency stop|Fatal error|Undefined control sequence|Missing character", re.I)
    for index, line in enumerate(lines):
        if pattern.search(line):
            selected.update(range(max(0, index - 1), min(len(lines), index + 4)))
    if selected:
        summary = "\n".join(lines[index] for index in sorted(selected))
    else:
        summary = "\n".join(lines[-60:])
    return summary[:limit]


def _text(value) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Best-effort timeout cleanup for the process group/tree."""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5,
                shell=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if process.poll() is None:
        process.kill()


def compile_tex(tex_path: Path, *, xelatex: str = "xelatex", timeout: float = 60.0) -> CompileResult:
    """Compile in the source's own working directory and preserve every log.

    The caller creates a dedicated per-attempt directory. Only stale products
    sharing this source's stem are removed; the .tex source is never modified.
    """
    tex_path = Path(tex_path).resolve()
    if not tex_path.is_file():
        raise FileNotFoundError("LaTeX source does not exist: " + str(tex_path))
    if tex_path.suffix.lower() != ".tex":
        raise ValueError("The compiler expects a .tex source file.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Compile timeout must be a positive finite number.")
    pdf_path = tex_path.with_suffix(".pdf")
    log_path = tex_path.with_suffix(".log")
    console_path = tex_path.with_suffix(".console.log")
    # A previous PDF must never turn a failed/empty invocation into success.
    for suffix in (".pdf", ".log", ".aux", ".out", ".fls", ".xdv"):
        product = tex_path.with_suffix(suffix)
        if product.is_file():
            product.unlink()
    command = build_compile_command(tex_path, xelatex)
    environment = os.environ.copy()
    # The compiler has no need for the model service's credentials.
    for name in ("API_KEY", "API_BASE_URL", "MODEL_NAME"):
        environment.pop(name, None)
    environment.update({"openin_any": "p", "openout_any": "p", "shell_escape": "f"})
    kwargs = dict(
        cwd=str(tex_path.parent), env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, shell=False,
    )
    if os.name == "nt":
        kwargs["creationflags"] = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                                    | getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        kwargs["start_new_session"] = True
    stdout, stderr, startup_error = b"", b"", ""
    returncode, timed_out = None, False
    try:
        process = subprocess.Popen(command, **kwargs)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            _terminate_process_tree(process)
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired as cleanup_exc:
                stdout = cleanup_exc.output or exc.output or b""
                stderr = cleanup_exc.stderr or exc.stderr or b""
                startup_error = "Process cleanup did not finish within 5 seconds."
        returncode = process.returncode
    except OSError as exc:
        startup_error = "XeLaTeX could not start: {}".format(exc)
    console = (
        "COMMAND (argument list): {!r}\nRETURN CODE: {}\nTIMED OUT: {}\n".format(command, returncode, timed_out)
        + startup_error + "\n\n=== STDOUT ===\n" + _text(stdout)
        + "\n\n=== STDERR ===\n" + _text(stderr)
    )
    console_path.write_text(console, encoding="utf-8")
    if log_path.is_file():
        # Keep the engine's original bytes untouched, even if decoding is imperfect.
        full_log = log_path.read_bytes().decode("utf-8", errors="replace")
    else:
        full_log = "XeLaTeX did not create a .log file.\n" + console
        log_path.write_text(full_log, encoding="utf-8")
    valid_pdf = False
    if pdf_path.is_file():
        with pdf_path.open("rb") as handle:
            valid_pdf = handle.read(5) == b"%PDF-"
    success = returncode == 0 and not timed_out and not startup_error and valid_pdf
    errors = ""
    if not success:
        reasons = []
        if timed_out:
            reasons.append("XeLaTeX exceeded the {:.1f}-second timeout.".format(timeout))
        if startup_error:
            reasons.append(startup_error)
        if returncode is not None and returncode != 0:
            reasons.append("XeLaTeX exit code: {}.".format(returncode))
        if not valid_pdf:
            reasons.append("No newly generated PDF with a valid PDF header was found.")
        reasons.append(extract_errors(full_log + "\n" + _text(stdout) + "\n" + _text(stderr)))
        errors = "\n".join(reason for reason in reasons if reason)
    return CompileResult(success, returncode, timed_out, pdf_path if valid_pdf else None,
                         log_path, console_path, errors, command)
