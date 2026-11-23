"""Opt-in image -> IR -> Scene pipeline; default auto_pipeline stays unchanged.

Offline inputs are supplied data, never claimed as VLM recognition. Online stage
one is explicit and bounded by the existing request ledger; stage two is local.
Partial Scene output is preserved but is never compiled as a complete result.
"""
from copy import copy
from dataclasses import asdict
import json
from pathlib import Path

from auto_compile import compile_bounded
from image_compare import make_comparison, render_pdf
from latex_template import wrap_document
from main import new_run_directory, snapshot_image
from structured_tikz.generator import generate_tikz
from structured_ir.schema import ir_schema
from structured_ir.validator import inspect_ir
from structured_ir.converter import convert_ir
from structured_ir.prompts import build_ir_prompt, parse_ir_response
from structured_ir.coverage import evaluate_ir_coverage
from structured_ir.regions import plan_regions, save_crops
from vlm_client import APIConfig, VLMClient, validate_image_path, _redact


def run_two_stage(args, *, client=None):
    from auto_pipeline import save_json, RequestLedger, _validate_options
    if args.offline_scene:
        raise ValueError('--offline-scene belongs to single-stage mode; use --offline-ir.')
    if args.allow_completion or args.allow_visual_refinement or args.stage_strategy != 'single':
        raise ValueError('Two-stage prototype makes at most one IR request and no model repairs.')
    if args.offline_ir and args.allow_api:
        raise ValueError('Offline IR cannot be mixed with API authorization.')
    checked = copy(args)
    # Reuse numeric bounds but let absent authorization produce an auditable
    # awaiting_ir result after preparing geometry/prompt, with zero API calls.
    checked.dry_run = True
    _validate_options(checked)
    original_path = validate_image_path(args.image)
    from PIL import Image
    with Image.open(original_path) as image:
        size = image.size
        if image.getexif().get(274, 1) != 1:
            raise ValueError('Normalize EXIF rotation before coordinate-based parsing.')
    if max(size) > 20000 or size[0]*size[1] > 25_000_000:
        raise ValueError('Image exceeds existing supported canvas limits.')
    output = new_run_directory(args.output_root, prefix='ir')
    image_path = snapshot_image(original_path, output)
    report = {'status': 'preparing', 'output_dir': str(output), 'pipeline': 'two-stage',
              'recognition': 'not_performed', 'api_called': False, 'api_call_count': 0,
              'compile_attempt_count': 0, 'automatic_repair_count': 0,
              'quality': 'not_verified', 'stage': 'regions'}
    ledger = None
    key = ''
    try:
        geometry = None
        if args.geometry_cache:
            geometry = json.loads(Path(args.geometry_cache).read_text(encoding='utf-8-sig'))
            report['geometry_source'] = 'supplied_cache_hash_checked_by_region_planner'
        elif args.geometry_mode == 'assist':
            from geometry_analyzer import analyze_image
            geometry = analyze_image(image_path)
            report['geometry_source'] = 'existing_analyzer'
        else:
            report['geometry_source'] = 'disabled'
        if geometry is not None:
            save_json(output/'geometry.json', geometry)
            plan = plan_regions(image_path, geometry)
            save_crops(image_path, plan, output/'crops')
        else:
            plan = {'canvas': {'width': size[0], 'height': size[1]}, 'candidates': [],
                    'diagnostics': ['Region analysis disabled; no semantic partition claimed.']}
        save_json(output/'region_analysis.json', plan)
        save_json(output/'ir_schema.json', ir_schema())
        prompt = build_ir_prompt(size, region_plan=plan)
        (output/'ir_prompt.txt').write_text(prompt, encoding='utf-8')
        report['stage'] = 'ir_acquisition'
        if args.dry_run:
            report['status'] = 'prepared_offline'
            return report
        if not args.offline_ir and not args.allow_api:
            report.update(status='awaiting_ir', error='Provide --offline-ir or explicitly authorize --allow-api. No reference IR was fabricated.')
            return report
        if args.offline_ir:
            raw_bytes = Path(args.offline_ir).read_bytes()
            (output/'supplied_ir.json').write_bytes(raw_bytes)
            raw = raw_bytes.decode('utf-8-sig')
            report['recognition'] = 'not_performed_offline_supplied_ir'
        else:
            if client is None:
                config = APIConfig.from_env()
                client = VLMClient(config, allow_api=True, max_tokens=args.max_tokens,
                                   timeout=args.api_timeout, thinking_mode=args.thinking_mode,
                                   reasoning_effort=args.reasoning_effort)
            key = getattr(getattr(client, 'config', None), 'api_key', '')
            ledger = RequestLedger(client, args, output)
            raw = ledger.request([image_path], prompt, 'generate')
            report['recognition'] = 'model_ir_response_unverified'
        (output/'ir_response.txt').write_text(_redact(raw, key), encoding='utf-8')
        report['stage'] = 'ir_parse'
        ir = parse_ir_response(raw)
        save_json(output/'ir_raw.json', ir)
        report['stage'] = 'ir_validation'
        validation = inspect_ir(ir)
        if validation['valid'] and (ir['canvas']['width'], ir['canvas']['height']) != size:
            validation['valid'] = False
            validation['errors'].append({'path': '/canvas', 'code': 'source_dimensions_mismatch',
                                         'message': 'IR canvas must match the supplied image dimensions.'})
        save_json(output/'ir_validation_report.json', validation)
        if not validation['valid']:
            report['status'] = 'ir_invalid'
            save_json(output/'coverage_report.json', evaluate_ir_coverage(ir, geometry=geometry))
            if ledger:
                ledger.mark('ir_validation_failed')
            return report
        if ledger:
            ledger.mark('ir_validated')
        report['stage'] = 'ir_to_scene'
        scene, conversion = convert_ir(ir)
        save_json(output/'ir_to_scene_report.json', conversion)
        coverage = evaluate_ir_coverage(ir, geometry=geometry, conversion_report=conversion)
        save_json(output/'coverage_report.json', coverage)
        if scene is not None:
            save_json(output/'scene.json', scene)
        if not conversion['complete']:
            report.update(status='conversion_partial' if scene is not None else 'conversion_failed',
                          error='Unsupported or unplaced elements remain; see ir_to_scene_report.json.')
            return report
        report['stage'] = 'tikz'
        fragment = generate_tikz(scene)
        (output/'generated.tikz').write_text(fragment, encoding='utf-8')
        tex = output/'generated.tex'
        tex.write_text(wrap_document(fragment), encoding='utf-8')
        if args.no_compile:
            report['status'] = 'tex_generated_unverified'
            return report
        report['stage'] = 'compile'
        report['compile_attempt_count'] += 1
        result = compile_bounded(tex, xelatex=args.xelatex, timeout=args.compile_timeout, memory_mb=args.memory_limit_mb)
        save_json(output/'compile.json', asdict(result))
        if not result.success:
            report.update(status='compile_failed', error=result.errors)
            return report
        report['stage'] = 'render'
        png = render_pdf(result.pdf_path, output/'generated.png', dpi=150)
        make_comparison(image_path, png, output/'comparison.png')
        report.update(status='rendered_unverified', tex=str(tex), pdf=str(result.pdf_path), png=str(png),
                      comparison=str(output/'comparison.png'))
        return report
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=_redact(str(exc), key))
        if hasattr(exc, 'diagnostic'):
            save_json(output/'parse_error.json', exc.diagnostic)
        return report
    finally:
        if ledger:
            report['api'] = ledger.summary()
            report['api_call_count'] = len(ledger.entries)
            report['api_called'] = bool(ledger.entries)
        save_json(output/'run.json', report)
