"""Small, auditable image -> VLM -> TikZ -> XeLaTeX experiment.

No network request is made without explicit --allow-api authorization.
The offline smoke test checks plumbing only, never image reconstruction quality.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from image_compare import make_comparison, render_pdf
from latex_compiler import compile_tex
from latex_template import extract_tikz, wrap_document
from layout_validator import validate_layout
from prompts import build_generation_prompt, build_geometry_context
from vlm_client import APIConfig, OutputConfig, ThinkingConfig, VLMClient, validate_image_path


PROJECT_DIR = Path(__file__).resolve().parent
MAX_REPAIRS = 2

# Deliberately tiny, human-written fixture: NOT an AI reconstruction of the user image.
SMOKE_TIKZ = r"""\begin{tikzpicture}[>=Latex, every node/.style={font=\small}]
\node[draw=blue!65!black,fill=blue!6,rounded corners,align=center] (a) at (0,0)
  {本地编译自检（非 AI 转换）\\中文、English 与数学公式};
\node[draw,rounded corners,align=center,below=8mm of a] (b)
  {$W_E\in\mathbb{R}^{50257\times256}$\\
   $W_P\in\mathbb{R}^{4\times256}$\\
   $E_{\mathrm{input}}\in\mathbb{R}^{8\times4\times256}$};
\draw[->,thick] (a) -- (b);
\node[right=8mm of b] {$\begin{pmatrix}1&2\\3&4\end{pmatrix}$};
\end{tikzpicture}
"""


def new_run_directory(output_root: Path, prefix: str = "run") -> Path:
    """Unique directory: never overwrite a previous experiment."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(output_root).resolve() / (prefix + "_" + stamp + "_" + uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def attempt_paths(run_dir: Path, number: int) -> dict:
    if not 1 <= number <= MAX_REPAIRS + 1:
        raise ValueError("Attempt number must be 1, 2 or 3")
    directory = run_dir / ("attempt_%02d" % number)
    directory.mkdir(parents=True, exist_ok=True)
    stem = "generation_%02d" % number
    return {
        "directory": directory,
        "response": directory / ("response_%02d.txt" % number),
        "partial": directory / ("response_%02d.partial.txt" % number),
        "api_diagnostics": directory / ("api_diagnostics_%02d.json" % number),
        "layout_report": directory / ("layout_report_%02d.json" % number),
        "generation_prompt": directory / ("generation_prompt_%02d.txt" % number),
        "fragment": directory / (stem + ".tikz"),
        "tex": directory / (stem + ".tex"),
        "png": directory / (stem + ".png"),
    }


def write_report(run_dir: Path, report: dict) -> None:
    (run_dir / "run.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare_geometry(image_path: Path, run_dir: Path, *, context_max_chars: int = 6000,
                     preview_only: bool = False):
    """Analyze locally, retain evidence, and prepare a bounded optional context."""
    # Lazy import keeps off mode usable without the optional OpenCV dependency.
    from geometry_analyzer import analyze_image, save_geometry_artifacts

    if not 800 <= context_max_chars <= 12000:
        raise ValueError("Geometry context budget must be between 800 and 12000 characters")
    geometry = analyze_image(image_path)
    context = build_geometry_context(geometry, max_chars=context_max_chars)
    artifacts = save_geometry_artifacts(image_path, run_dir, geometry=geometry)
    context_path = run_dir / "geometry_context.json"
    context_text_path = run_dir / "geometry_context.txt"
    context.update(usage="preview_only_no_api" if preview_only else "initial_generation_context",
                   note="Selection is geometry evidence, not verified semantics or proof of API success.")
    context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    context_text_path.write_text(context["text"], encoding="utf-8")
    metadata = {
        "json": str(artifacts["geometry"]), "debug": str(artifacts["debug"]),
        "context_json": str(context_path), "context_text": str(context_text_path),
        "context_characters": context["char_count"],
        "selected_ids": context["selected_ids"],
        "counts": {name: len(geometry[name]) for name in ("rectangles", "lines", "regions")},
        "warnings": geometry.get("warnings", []),
    }
    print("  Geometry: " + json.dumps(metadata["counts"], ensure_ascii=False)
          + "; context characters=" + str(context["char_count"]))
    return metadata, context["text"]


def snapshot_image(image_path: Path, run_dir: Path) -> Path:
    target = run_dir / ("original" + image_path.suffix.lower())
    # Each run directory is new, but still reject an accidental overwrite.
    with target.open("xb") as destination, image_path.open("rb") as source:
        shutil.copyfileobj(source, destination)
    return target


def run_geometry_analysis(image_path: Path, output_root: Path, *, context_max_chars: int = 6000) -> dict:
    """Geometry only: no credentials, API client, compiler or repair loop."""
    image_path = validate_image_path(image_path)
    run_dir = new_run_directory(output_root, "geometry")
    report = {"kind": "opencv_geometry_analysis", "geometry_mode": "analyze",
              "status": "started", "output_dir": str(run_dir), "image": str(image_path),
              "api_called": False, "quality_verified": False,
              "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()}
    try:
        report["original_snapshot"] = str(snapshot_image(image_path, run_dir))
        report["geometry"], _ = prepare_geometry(image_path, run_dir,
                                                context_max_chars=context_max_chars, preview_only=True)
        report["status"] = "analyzed"
    except Exception as exc:
        report.update(status="geometry_failed", error=str(exc))
    write_report(run_dir, report)
    return report


def save_layout_report(fragment: str, path: Path) -> dict:
    """Advisory only: compilation and the original repair mechanism remain authoritative."""
    try:
        report = validate_layout(fragment)
    except Exception as exc:
        report = {"schema_version": "1.0", "advisory_only": True, "analysis_error": str(exc),
                  "summary": {"errors": 0, "warnings": 1},
                  "limitations": ["Static analysis failed; no layout conclusion is available."]}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  Layout advisory: " + json.dumps(report.get("summary", {}), ensure_ascii=False))
    return report


def _output_metadata(settings: OutputConfig) -> dict:
    return {
        "parameter": settings.parameter,
        "max_tokens": settings.max_tokens,
        "model_output_limit": settings.model_output_limit,
    }


def _thinking_metadata(settings: ThinkingConfig) -> dict:
    return {
        "thinking_mode": settings.thinking_mode,
        "reasoning_effort": settings.reasoning_effort,
        # None means unspecified, not proof that a provider disables reasoning.
        "thinking_enabled": (None if settings.thinking_mode is None else
                             settings.thinking_mode == "enabled"),
    }


def _api_host(config) -> str:
    """Record the provider host only, never its full URL or credentials."""
    base_url = getattr(config, "base_url", None)
    if not isinstance(base_url, str):
        return ""
    host = urlsplit(base_url).hostname or ""
    key = getattr(config, "api_key", None)
    if isinstance(key, str) and key:
        host = host.replace(key, "[REDACTED]")
    return host


def _save_api_diagnostics(client, paths: dict, number: int, report: dict) -> None:
    """Persist only the client's sanitized metadata and explicitly opted-in text."""
    diagnostics = getattr(client, "last_diagnostics", None)
    if not isinstance(diagnostics, dict) or not diagnostics:
        return
    paths["api_diagnostics"].write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    record = {"attempt": number, "diagnostics": str(paths["api_diagnostics"])}
    partial = getattr(client, "last_truncated_text", None)
    if (getattr(client, "save_truncated_response", False) is True
            and isinstance(partial, str) and partial):
        # A .partial.txt is never passed to extraction, wrapping or compilation.
        paths["partial"].write_text(partial, encoding="utf-8")
        record["partial_response"] = str(paths["partial"])
    report["api_requests"].append(record)
    usage = diagnostics.get("usage", {})
    details = usage.get("completion_tokens_details", {})
    summary = {
        "requested_output_parameter": diagnostics.get("requested_output_parameter"),
        "requested_output_tokens": diagnostics.get("requested_output_tokens"),
        "finish_reason": diagnostics.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "thinking_mode": diagnostics.get("requested_thinking_mode"),
        "thinking_enabled": diagnostics.get("thinking_enabled"),
        "reasoning_effort": diagnostics.get("requested_reasoning_effort"),
    }
    print("  API: " + json.dumps(summary, ensure_ascii=False))


def _compile_record(number: int, paths: dict, result) -> dict:
    return {
        "attempt": number,
        "tex": str(paths["tex"]),
        "fragment": str(paths["fragment"]),
        "compiled": result.success,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "pdf": str(result.pdf_path) if result.pdf_path else None,
        "log": str(result.log_path),
        "console": str(result.console_path),
        "command": result.command,
        "errors": result.errors,
    }


def run_experiment(image_path: Path, output_root: Path, client, *, max_repairs: int = 2,
                   xelatex: str = "xelatex", compile_timeout: float = 60.0,
                   geometry_mode: str = "off", geometry_context_max_chars: int = 6000) -> dict:
    """Run one initial generation and at most two compile-error repairs.

    Dependency boundaries are intentionally ordinary functions so tests can mock
    HTTP/compilation without claiming a successful real conversion.
    """
    if not 0 <= max_repairs <= MAX_REPAIRS:
        raise ValueError("max_repairs must be between 0 and 2")
    if not math.isfinite(compile_timeout) or compile_timeout <= 0:
        raise ValueError("Compile timeout must be positive and finite")
    if geometry_mode not in ("off", "assist"):
        raise ValueError("run_experiment accepts off/assist; analyze uses the offline entry point")
    image_path = validate_image_path(image_path)
    run_dir = new_run_directory(output_root)
    report = {
        "kind": "vlm_image_to_tikz", "status": "started",
        "output_dir": str(run_dir), "image": str(image_path),
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "model": client.config.model,
        "geometry_mode": geometry_mode,
        "api_host": _api_host(client.config),
        "max_repairs": max_repairs, "attempts": [], "api_requests": [],
        "quality_verified": False,
        "quality_note": "Compilation is not proof of Chinese, formula, number or topology fidelity.",
    }
    settings = getattr(client, "output_config", None)
    if isinstance(settings, OutputConfig):
        report["output_config"] = _output_metadata(settings)
    thinking = getattr(client, "thinking_config", None)
    if isinstance(thinking, ThinkingConfig):
        report["thinking_config"] = _thinking_metadata(thinking)
    write_report(run_dir, report)
    previous_fragment, previous_errors = "", ""
    phase = "input_snapshot"
    try:
        report["original_snapshot"] = str(snapshot_image(image_path, run_dir))
        geometry_context = None
        if geometry_mode == "assist":
            phase = "geometry"
            report["geometry"], geometry_context = prepare_geometry(
                image_path, run_dir, context_max_chars=geometry_context_max_chars)
            write_report(run_dir, report)
        for number in range(1, max_repairs + 2):
            paths = attempt_paths(run_dir, number)
            phase = "generation" if number == 1 else "repair_api"
            print("[%d/%d] %s" % (number, max_repairs + 1,
                  "正在请求图像生成（真实 API）" if number == 1 else "正在请求编译错误修复（真实 API）"))
            try:
                if number == 1:
                    if geometry_mode == "assist":
                        paths["generation_prompt"].write_text(
                            build_generation_prompt(geometry_context), encoding="utf-8")
                        response = client.generate(image_path, geometry_context=geometry_context)
                    else:
                        response = client.generate(image_path)
                else:
                    # Keep the original compile-repair contract and request behavior.
                    response = client.repair(image_path, previous_fragment, previous_errors)
            finally:
                # The client rejects length before returning; retain diagnostics even then.
                _save_api_diagnostics(client, paths, number, report)
                write_report(run_dir, report)
            paths["response"].write_text(response, encoding="utf-8")
            phase = "validation"
            fragment = extract_tikz(response)
            # The raw response above survives validation failures for diagnosis.
            paths["fragment"].write_text(fragment, encoding="utf-8")
            layout = save_layout_report(fragment, paths["layout_report"])
            document = wrap_document(fragment)
            paths["tex"].write_text(document, encoding="utf-8")
            phase = "compilation"
            result = compile_tex(paths["tex"], xelatex=xelatex, timeout=compile_timeout)
            record = _compile_record(number, paths, result)
            record["layout_report"] = str(paths["layout_report"])
            record["layout_summary"] = layout.get("summary", {})
            report["attempts"].append(record)
            write_report(run_dir, report)
            print("  XeLaTeX: success=%s, exit=%s, timeout=%s" %
                  (result.success, result.returncode, result.timed_out))
            if result.success:
                phase = "rendering"
                png = render_pdf(result.pdf_path, paths["png"])
                phase = "comparison"
                comparison = make_comparison(image_path, png, run_dir / "comparison.png")
                report.update(status="completed", tex=str(paths["tex"]),
                              pdf=str(result.pdf_path), png=str(png), comparison=str(comparison))
                record["png"] = str(png)
                break
            previous_fragment, previous_errors = fragment, result.errors
            if number == max_repairs + 1:
                report["status"] = "compile_failed"
                report["error"] = "Compilation failed after the configured repair limit. See per-attempt logs."
        write_report(run_dir, report)
    except Exception as exc:
        # Keep prior attempts and compiler artifacts even if repair/rendering fails.
        report.update(status=phase + "_failed", error=str(exc))
        write_report(run_dir, report)
    return report


def run_smoke_test(output_root: Path, *, xelatex: str = "xelatex",
                   compile_timeout: float = 60.0) -> dict:
    """Real offline XeLaTeX + PNG test, with a clearly marked handwritten fixture."""
    run_dir = new_run_directory(output_root, "smoke")
    paths = attempt_paths(run_dir, 1)
    paths["fragment"].write_text(SMOKE_TIKZ, encoding="utf-8")
    paths["tex"].write_text(wrap_document(SMOKE_TIKZ), encoding="utf-8")
    report = {"kind": "offline_handwritten_smoke_test", "status": "started",
              "output_dir": str(run_dir), "api_called": False,
              "quality_verified": False, "attempts": [],
              "note": "Only template/compiler/rendering validation; not target image reconstruction."}
    try:
        result = compile_tex(paths["tex"], xelatex=xelatex, timeout=compile_timeout)
        report["attempts"].append(_compile_record(1, paths, result))
        if result.success:
            png = render_pdf(result.pdf_path, paths["png"])
            report.update(status="completed", tex=str(paths["tex"]),
                          pdf=str(result.pdf_path), png=str(png))
        else:
            report.update(status="compile_failed", error=result.errors)
    except Exception as exc:
        report.update(status="smoke_failed", error=str(exc))
    write_report(run_dir, report)
    return report


def preflight(xelatex: str) -> dict:
    return {
        "python": sys.version.split()[0],
        "xelatex": shutil.which(xelatex),
        "Pillow": importlib.util.find_spec("PIL") is not None,
        "PyMuPDF": (importlib.util.find_spec("pymupdf") is not None
                    or importlib.util.find_spec("fitz") is not None),
    }


def require_local_tools(xelatex: str) -> None:
    checks = preflight(xelatex)
    if not checks["xelatex"]:
        raise ValueError("XeLaTeX 未找到。请把已有 TeX Live/MiKTeX 的 bin 加入 PATH，或传 --xelatex 路径。不会自动安装。")
    if not checks["Pillow"] or not checks["PyMuPDF"]:
        raise ValueError("缺少图片依赖。请自行运行 python -m pip install -r requirements.txt；不会自动安装。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="中文图片直接输入 VLM，生成可编辑 TikZ 并用 XeLaTeX 验证。")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=Path, help="原始 PNG/JPEG/WebP 的实际本地路径")
    mode.add_argument("--check", action="store_true", help="只检查环境，不请求 API、不创建结果")
    mode.add_argument("--smoke-test", action="store_true", help="离线手写中文图编译自检，不代表 AI 图片转换")
    parser.add_argument("--allow-api", action="store_true", help="明确允许本次真实 API 调用及最多两次付费修复；默认禁止")
    parser.add_argument("--max-repairs", type=int, choices=range(3), default=2)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--xelatex", default="xelatex")
    parser.add_argument("--compile-timeout", type=float, default=60.0)
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", "--max-output-tokens", dest="max_tokens", type=int,
                        help="输出预算；CLI > API_MAX_OUTPUT_TOKENS > 服务商默认值")
    parser.add_argument("--token-limit-parameter", choices=("max_tokens", "max_completion_tokens"),
                        help="服务商接受的输出字段；覆盖 API_TOKEN_LIMIT_PARAMETER")
    parser.add_argument("--model-output-limit", type=int,
                        help="已由服务商文档核实的输出上限；覆盖 API_MODEL_OUTPUT_LIMIT")
    parser.add_argument("--thinking-mode", choices=("enabled", "disabled"),
                        help="思考开关；覆盖 API_THINKING_MODE；官方 deepseek-flash 默认 disabled")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"),
                        help="思考强度；覆盖 API_REASONING_EFFORT；开启思考时才有意义")
    parser.add_argument("--save-api-diagnostics", action="store_true",
                        help="确认服务商允许本地留存后，保存脱敏截断文本；元数据始终保存")
    parser.add_argument("--geometry-mode", choices=("off", "analyze", "assist"), default="off",
                        help="off=原流程；analyze=只离线分析；assist=附加几何候选，仍需API授权")
    parser.add_argument("--geometry-context-max-chars", type=int, default=6000,
                        help="几何附加上下文字符上限（800至12000，默认6000，不是Token数）")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.geometry_mode != "off" and args.image is None:
            raise ValueError("Geometry analyze/assist requires --image")
        if args.geometry_mode == "analyze":
            # Intentionally before API/thinking/budget parsing: this mode is fully offline.
            report = run_geometry_analysis(args.image, args.output_dir,
                                           context_max_chars=args.geometry_context_max_chars)
            print("状态：%s" % report["status"])
            print("结果目录：%s" % report["output_dir"])
            if report.get("error"):
                print("错误：%s" % report["error"])
            return 0 if report["status"] == "analyzed" else 1
        if (not math.isfinite(args.compile_timeout) or not math.isfinite(args.api_timeout)
                or args.compile_timeout <= 0 or args.api_timeout <= 0):
            raise ValueError("Timeouts must be positive and finite")
        output_config = OutputConfig.from_env(
            max_tokens=args.max_tokens, parameter=args.token_limit_parameter,
            model_output_limit=args.model_output_limit,
        )
        thinking_config = ThinkingConfig.from_env(
            thinking_mode=args.thinking_mode, reasoning_effort=args.reasoning_effort,
        )
        if args.check:
            checks = preflight(args.xelatex)
            checks["output_config"] = _output_metadata(output_config)
            checks["thinking_config"] = _thinking_metadata(thinking_config)
            checks["save_truncated_response"] = args.save_api_diagnostics
            try:
                config = APIConfig.from_env()
                checks["api_config"] = "configured (values hidden); API requests still disabled by default"
                checks["api_host"] = _api_host(config)
                checks["model"] = config.model
            except ValueError as exc:
                checks["api_config"] = str(exc)
            print(json.dumps(checks, ensure_ascii=False, indent=2))
            return 0
        if args.smoke_test:
            require_local_tools(args.xelatex)
            report = run_smoke_test(args.output_dir, xelatex=args.xelatex,
                                    compile_timeout=args.compile_timeout)
        else:
            image_path = validate_image_path(args.image)
            print("[1] 原图检查通过：%s" % image_path)
            config = APIConfig.from_env()
            if not args.allow_api:
                raise ValueError("未授权真实 API 请求。确认费用后，显式添加 --allow-api；最多 1 次生成 + 2 次修复。")
            require_local_tools(args.xelatex)
            client = VLMClient(config, allow_api=True, timeout=args.api_timeout,
                               max_tokens=output_config.max_tokens,
                               token_limit_parameter=output_config.parameter,
                               model_output_limit=output_config.model_output_limit,
                               thinking_mode=thinking_config.thinking_mode,
                               reasoning_effort=thinking_config.reasoning_effort,
                               save_truncated_response=args.save_api_diagnostics)
            report = run_experiment(image_path, args.output_dir, client,
                                    max_repairs=args.max_repairs, xelatex=args.xelatex,
                                    compile_timeout=args.compile_timeout,
                                    geometry_mode=args.geometry_mode,
                                    geometry_context_max_chars=args.geometry_context_max_chars)
        print("状态：%s" % report["status"])
        print("结果目录：%s" % report["output_dir"])
        for key in ("tex", "pdf", "png", "comparison", "error"):
            if report.get(key):
                print("%s: %s" % (key, report[key]))
        print("编译/Mock 自检不证明模型已正确识别中文、公式或连接关系。")
        return 0 if report["status"] == "completed" else 1
    except (ValueError, RuntimeError, OSError) as exc:
        print("停止：%s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
