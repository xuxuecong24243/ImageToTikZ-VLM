"""ImageToTikZ: technical images to editable LaTeX and PDF/PNG previews.

Use --backend local for the verified v0.5 OCR/OpenCV pipeline, without an API
key. Optional VLM recognition requires --allow-api. Hybrid remains experimental
and is not available through this CLI. Partial results are reported explicitly.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from auto_pipeline import build_parser as pipeline_parser, save_json
from two_stage_pipeline import run_two_stage
from auto_compile import compile_bounded
from image_compare import render_pdf, make_comparison
from latex_template import wrap_document
from structured_ir.consistency import inspect_consistency
from structured_ir.geometry_repair import repair_geometry
from structured_tikz.generator import generate_tikz
from structured_tikz.validator import validate_scene
from vlm_client import validate_image_path, _redact


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Input PNG/JPG image (paths with spaces must be quoted).')
    parser.add_argument('--output', required=True, type=Path, help='New or empty output directory; existing results are never overwritten.')
    parser.add_argument('--backend', choices=('vlm', 'local', 'hybrid'), default='vlm',
                        help='vlm preserves the existing default; local runs offline OCR/OpenCV; hybrid is not implemented.')
    parser.add_argument('--ocr-python', type=Path, help='Local OCR Python executable; defaults to the separate .venv-ocr environment.')
    parser.add_argument('--ocr-timeout', type=float, default=300, help='Bounded local OCR worker timeout, 1..900 seconds.')
    parser.add_argument('--allow-api', action='store_true', help='Explicit permission to send the image to the configured API, at most once.')
    parser.add_argument('--max-api-requests', type=int, choices=(0, 1), default=1, help='Hard request cap, including failed attempts (default: 1).')
    parser.add_argument('--offline-ir', type=Path, help='Replay saved IR without API/config reads; not new image recognition.')
    parser.add_argument('--xelatex', default='xelatex', help='XeLaTeX executable or full path.')
    parser.add_argument('--no-compile', action='store_true', help='Keep editable TeX only; previews reported as skipped.')
    parser.add_argument('--max-tokens', type=int, help='Optional model output-token limit; existing model defaults otherwise.')
    parser.add_argument('--api-timeout', type=float, default=180)
    parser.add_argument('--compile-timeout', type=float, default=60)
    return parser


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def run_conversion(args, *, client=None):
    """Return a persisted report. Client injection remains authorization gated.

Output setup errors raise without writing into a prior run. All subsequent
failures are recorded, including invalid input and missing API configuration.
"""
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Output directory is not empty; choose a new --output to preserve previous results.')
    report_path = output/'report.json'
    with report_path.open('x', encoding='utf-8') as stream:
        stream.write('{}\n')
    report = {
        'version': '0.1', 'status': 'failed', 'stage': 'input',
        'input': {'path': str(args.input.expanduser().resolve())},
        'mode': 'offline_replay' if args.offline_ir else 'image_to_ir',
        'recognition': 'not_performed', 'model': None,
        'api_request_count': 0, 'max_api_requests': args.max_api_requests,
        'token_usage': {'known': True, 'total_tokens': 0, 'requests': []},
        'ir_validation': {'status': 'not_run', 'valid': None, 'errors': []},
        'scene_validation': {'status': 'not_run', 'valid': None, 'errors': []},
        'conversion_status': 'not_run', 'semantic_status': 'unverified',
        'visual_status': 'unverified',
        'latex': {'status': 'not_run'}, 'pdf': {'status': 'not_run'}, 'png': {'status': 'not_run'},
        'compile_attempts': 0, 'errors': [], 'warnings': [],
        'paths': {'report': str(report_path), 'log': str(output/'pipeline.log')},
        'limits': {'api_retries': 0, 'compile_retries': 0},
    }
    secret = ''
    backend = getattr(args, 'backend', 'vlm')
    report['backend'] = backend
    if backend == 'local':
        report.update(version='0.3', mode='local_ocr_geometry')

    def fail(code, message):
        report['errors'].append({'stage': report['stage'], 'code': code, 'message': message})

    try:
        with (output/'pipeline.log').open('x', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
            image = validate_image_path(args.input)
            from PIL import Image
            with Image.open(image) as source:
                if source.format not in {'PNG', 'JPEG'}:
                    raise ValueError('v0.1 accepts PNG/JPG images.')
                if source.getexif().get(274, 1) != 1:
                    raise ValueError('Normalize EXIF rotation before coordinate-based conversion.')
                size = source.size
            if max(size) > 20000 or size[0]*size[1] > 25_000_000:
                raise ValueError('Image exceeds supported canvas/pixel limits.')
            source_sha = hashlib.sha256(image.read_bytes()).hexdigest()
            report['input'].update(width=size[0], height=size[1], sha256=source_sha)
            report['stage'] = 'authorization'
            if backend == 'hybrid':
                raise ValueError('Hybrid backend is not implemented; use --backend local or vlm.')
            if backend == 'local' and (args.allow_api or args.offline_ir or client is not None):
                raise ValueError('Local backend accepts images only; do not combine it with API authorization, injected clients or offline IR.')
            if type(args.max_api_requests) is not int or args.max_api_requests not in (0, 1):
                raise ValueError('--max-api-requests must be 0 or 1; v0.1 never retries.')
            if args.offline_ir and args.allow_api:
                raise ValueError('Do not combine --offline-ir and --allow-api.')
            if backend == 'vlm' and not args.offline_ir and not args.allow_api:
                raise ValueError('Image recognition requires explicit --allow-api; no API configuration was read and no request was made.')
            if backend == 'vlm' and not args.offline_ir and args.max_api_requests == 0:
                raise ValueError('API request budget is zero; no request was made.')

            # Reuse acquisition, response persistence, parsing, validation and
            # deterministic conversion. Keep legacy CLI policies unchanged.
            upstream_args = pipeline_parser().parse_args([
                '--image', str(image), '--pipeline-mode', 'two-stage',
                '--output-root', str(output/'intermediates'), '--geometry-mode', 'off', '--no-compile'])
            upstream_args.allow_api = args.allow_api
            upstream_args.offline_ir = args.offline_ir
            upstream_args.max_api_requests = max(1, args.max_api_requests)  # offline makes zero calls
            upstream_args.max_tokens = args.max_tokens
            upstream_args.api_timeout = args.api_timeout
            upstream_args.compile_timeout = args.compile_timeout
            upstream_args.max_output_budget = max(65536, args.max_tokens or 0)
            report['stage'] = 'pipeline'
            if backend == 'local':
                from local_pipeline import prepare_local
                upstream = prepare_local(image, output, ocr_python=args.ocr_python, ocr_timeout=args.ocr_timeout,
                                         xelatex=args.xelatex, compile_timeout=args.compile_timeout,
                                         text_layout=not args.no_compile)
                report['local_recognition'] = upstream
                report['recognition_status'] = upstream['recognition_status']
                if upstream.get('typography_version'):
                    report['version'] = upstream['typography_version']
                    report['typography'] = upstream['typography']
                if upstream.get('local_features_version'):
                    report['local_features_version'] = upstream['local_features_version']
                    if not args.no_compile:
                        report['version'] = upstream['local_features_version']
            else:
                upstream = run_two_stage(upstream_args, client=client)
            if client is not None:
                secret = getattr(getattr(client, 'config', None), 'api_key', '')
            work = Path(upstream['output_dir'])
            report['paths']['intermediates'] = str(work)
            report['recognition'] = upstream['recognition']
            report['api_request_count'] = upstream['api_call_count']
            api = upstream.get('api', {})
            entries = api.get('requests', [])
            report['api'] = api
            usage = [entry.get('diagnostics', {}).get('usage', {}) for entry in entries]
            report['token_usage'] = {
                'known': api.get('all_usage_reported', not entries),
                'total_tokens': api.get('reported_total_usage_tokens', 0) if api.get('all_usage_reported', True) else None,
                'requests': usage, 'reserved_output_tokens': api.get('reserved_output_tokens', 0)}
            report['model'] = upstream.get('model') or next((e.get('diagnostics', {}).get('response_model')
                                    or e.get('diagnostics', {}).get('requested_model')
                                    or e.get('diagnostics', {}).get('model')
                                    for e in entries if any(e.get('diagnostics', {}).get(k)
                                    for k in ('response_model', 'requested_model', 'model'))), None)
            if client is not None and entries and report['model'] is None:
                report['model'] = getattr(getattr(client, 'config', None), 'model', None)
            for name in ('ir_raw.json', 'ir_response.txt', 'ir_validation_report.json', 'scene.json',
                         'ir_to_scene_report.json', 'coverage_report.json', 'parse_error.json',
                         'ocr_raw.json', 'geometry_raw.json', 'fusion_result.json', 'ir_generated.json',
                         'ir_validation.json', 'scene_generated.json', 'style_map.json',
                         'style_extraction_report.json', 'layer_analysis.json',
                         'text_source_analysis.json','text_measurement_report.json','text_collision_report.json',
                         'font_mapping_report.json','text_layout_selection.json',
                         'small_shape_detection.json','arrow_detection.json','connection_mapping.json'):
                if (work/name).exists():
                    report['paths'][name] = str(work/name)
            validation_path = work/('ir_validation.json' if backend == 'local' else 'ir_validation_report.json')
            ir_path = work/('ir_generated.json' if backend == 'local' else 'ir_raw.json')
            scene_path = work/('scene_generated.json' if backend == 'local' else 'scene.json')
            if validation_path.exists():
                validation = _read(validation_path)
                report['ir_validation'] = dict(validation, status='valid' if validation['valid'] else 'invalid')
            if ir_path.exists():
                ir = _read(ir_path)
                # Existing adapter can only repair geometry with independent
                # reviewed evidence. No such evidence is fabricated by this CLI.
                _, recovery = repair_geometry(ir, image_sha256=source_sha)
                save_json(work/'recovery_report.json', recovery)
                report['recovery'] = {'status': recovery['status'], 'changes': recovery['changes'],
                                      'report': str(work/'recovery_report.json')}
            if (work/'ir_to_scene_report.json').exists():
                conversion = _read(work/'ir_to_scene_report.json')
                report['conversion_status'] = conversion['status']
                report['conversion'] = conversion
            if upstream['status'] not in ('tex_generated_unverified', 'conversion_partial', 'local_scene_ready'):
                report['stage'] = upstream['stage']
                message = upstream.get('error')
                if not message and report['ir_validation']['errors']:
                    first = report['ir_validation']['errors'][0]
                    message = '{}: {} ({} validation errors; see report.json)'.format(
                        first['path'], first['message'], len(report['ir_validation']['errors']))
                fail(upstream['status'], message or 'Pipeline stopped; inspect validation and intermediate reports.')
                return report
            report['stage'] = 'scene_validation'
            scene = _read(scene_path)
            try:
                validate_scene(scene)
            except ValueError as exc:
                report['scene_validation'] = {'status': 'invalid', 'valid': False, 'errors': [str(exc)]}
                raise
            report['scene_validation'] = {'status': 'valid', 'valid': True, 'errors': []}
            if not scene['nodes'] and not scene['edges']:
                fail('no_drawable_content', 'No safely drawable content; an empty Scene is not a conversion result.')
                return report
            semantic = inspect_consistency(ir, scene, conversion)
            save_json(work/'semantic_report.json', semantic)
            report['semantic_status'] = semantic['semantic_status']
            report['paths']['semantic_report'] = str(work/'semantic_report.json')
            if not conversion['complete']:
                report['warnings'].append('Partial conversion: unsupported/unplaced content remains in the saved IR; see conversion diagnostics.')
            report['warnings'].append('Editable output does not certify text recognition, visual fidelity or full semantic correctness.')
            if backend == 'local':
                report['warnings'].append(upstream.get('warning', 'Local recognition is partial and unverified.'))
            report['stage'] = 'latex'
            fragment = generate_tikz(scene)
            document = wrap_document(fragment)
            (output/'generated.tikz').write_text(fragment, encoding='utf-8')
            (output/'generated.tex').write_text(document, encoding='utf-8')
            report['latex'] = {'status': 'generated', 'editable': True}
            report['paths'].update(tex=str(output/'generated.tex'), tikz=str(output/'generated.tikz'))
            report['status'] = 'partial'  # TeX retained even if previews fail.
            if args.no_compile:
                report['pdf'] = {'status': 'skipped', 'reason': '--no-compile'}
                report['png'] = {'status': 'skipped', 'reason': 'PDF compilation skipped'}
                report['warnings'].append('Preview generation was explicitly skipped.')
                report['stage'] = 'done'
                return report
            report['stage'] = 'compile'
            report['compile_attempts'] = 1
            result = compile_bounded(output/'generated.tex', xelatex=args.xelatex,
                                     timeout=args.compile_timeout, memory_mb=512)
            save_json(output/'compile_report.json', asdict(result))
            report['pdf'] = {'status': 'compiled' if result.success else 'failed',
                             'returncode': result.returncode, 'timed_out': result.timed_out,
                             'log': str(result.log_path), 'reason': result.errors or None}
            if not result.success:
                report['png'] = {'status': 'skipped', 'reason': 'PDF compilation failed'}
                fail('compile_failed', result.errors)
                return report
            pdf = output/'generated.pdf'
            if not pdf.is_file() or not pdf.stat().st_size:
                raise ValueError('Compiler reported success but generated.pdf is missing or empty.')
            report['paths']['pdf'] = str(pdf)
            if backend == 'local' and scene['canvas'].get('measure_text'):
                from local_text_layout import verify_final_text_layout
                report['stage'] = 'text_layout_verification'
                report['text_layout_verification'] = verify_final_text_layout(ir,scene,output)
            report['stage'] = 'render'
            png = render_pdf(pdf, output/'generated.png', dpi=150)
            report['png'] = {'status': 'rendered'}
            report['paths']['png'] = str(png)
            if backend == 'local':
                try:
                    make_comparison(image, png, output/'comparison.png')
                    report['paths']['comparison'] = str(output/'comparison.png')
                except Exception as exc:
                    report['warnings'].append('Comparison unavailable: '+str(exc))
            report['status'] = ('success' if conversion['complete'] and upstream.get('recognition_status') != 'partial' else 'partial')
            report['stage'] = 'done'
            return report
    except Exception as exc:
        message = _redact(str(exc), secret)
        fail(type(exc).__name__, message)
        if report['stage'] == 'compile':
            report['pdf'] = {'status': 'failed', 'reason': message}
            report['png'] = {'status': 'skipped', 'reason': 'PDF compilation failed'}
        elif report['stage'] == 'render':
            report['png'] = {'status': 'failed', 'reason': message}
        elif report['stage'] == 'latex':
            report['latex'] = {'status': 'failed', 'reason': message}
        return report
    finally:
        save_json(report_path, report)


def main(argv=None):
    args = build_parser().parse_args(argv)
    print('ImageToTikZ v0.5' if args.backend == 'local' else 'ImageToTikZ v0.1')
    print('Input:', args.input)
    try:
        report = run_conversion(args)
    except (ValueError, OSError) as exc:
        print('[FAIL] Output:', str(exc).splitlines()[0][:300])
        return 1
    if 'width' in report['input']:
        print('[OK] Image loaded')
    if report.get('local_recognition', {}).get('ocr_count') is not None:
        print('[OK] Local OCR: {} text regions'.format(report['local_recognition']['ocr_count']))
    for state, message in [(report['ir_validation']['status'] == 'valid', 'IR replayed' if args.offline_ir else 'IR generated'),
                           (report['scene_validation']['status'] == 'valid', 'Scene validated'),
                           (report['latex']['status'] == 'generated', 'LaTeX generated'),
                           (report['pdf']['status'] == 'compiled', 'PDF compiled'),
                           (report['png']['status'] == 'rendered', 'PNG rendered')]:
        if state:
            print('[OK]', message)
    print('Status:', report['status'])
    for error in report['errors']:
        print('[FAIL] {}: {}'.format(error['stage'], error['message'].splitlines()[0][:300]))
    if report['status'] == 'partial':
        print('[WARN] Partial result; inspect report.json for content loss or unavailable previews.')
    print('Output:', args.output)
    for key in ('tex', 'pdf', 'png', 'report'):
        if key in report['paths']:
            print(' ', Path(report['paths'][key]).name)
    return {'success': 0, 'partial': 2, 'failed': 1}[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())
