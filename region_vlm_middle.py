"""One middle-column experiment. Default: offline preparation, no API or compiler.

--allow-api separately authorizes one paid request. No retry, completion, visual
feedback, OCR, whole-image recognition, or layout optimization is performed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from types import SimpleNamespace
from uuid import uuid4

from PIL import Image
from auto_compile import compile_bounded
from auto_pipeline import RequestLedger
from auto_scene_generator import _strict_json, build_scene_prompt, normalize_scene_provenance
from auto_scene_validator import diagnose_scene
from image_compare import make_comparison, render_pdf
from latex_template import wrap_document
from region_scene import make_region_scene, validate_manifest
from region_scene_merger import merge_region_scenes
from structured_tikz.generator import generate_tikz
from structured_tikz.validator import validate_scene
from vlm_client import APIConfig, VLMClient, ThinkingConfig, _is_official_flash, _redact

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / 'outputs/region_reconstruction_20261114T133100Z_06de95e9/region_manifest.json'
DEFAULT_BASELINE = ROOT / 'outputs/offline_gpt2_scene_repair_20261114T093235Z_6394846a/scene_after.json'
MAX_TOKENS = 32768


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False, default=str) + '\n', encoding='utf-8')


def load_middle_input(manifest_path):
    """Bind this experiment to saved crop pixels; no new crop is written."""
    path = Path(manifest_path).resolve()
    manifest = _strict_json(path.read_text(encoding='utf-8-sig'))
    validate_manifest(manifest)
    if [manifest['image']['width'], manifest['image']['height']] != [1536, 1024]:
        raise ValueError('Expected original dimensions 1536 x 1024.')
    matches = [r for r in manifest['regions'] if r['region_id'] == 'region_002']
    if len(matches) != 1:
        raise ValueError('Exactly one region_002 is required.')
    region = matches[0]
    if region['crop_bbox'] != [444, 0, 606, 1024] or region['offset'] != [444, 0]:
        raise ValueError('Expected crop [444,0,606,1024] and offset [444,0], not core x=456.')
    crop = (path.parent / region['crop_path']).resolve()
    crop.relative_to(path.parent)
    source = Path(manifest['image']['path'])
    source = source.resolve() if source.is_absolute() else (path.parent / source).resolve()
    if _sha(source) != manifest['image'].get('sha256'):
        raise ValueError('Original image SHA-256 differs from the manifest.')
    with Image.open(source) as original, Image.open(crop) as local:
        if original.size != (1536, 1024) or local.size != (606, 1024) or local.format != 'PNG':
            raise ValueError('Unexpected source/crop dimensions or crop format.')
        expected = original.crop((444, 0, 1050, 1024))
        if local.mode != expected.mode or local.tobytes() != expected.tobytes():
            raise ValueError('Saved crop is not pixel-identical to the original region.')
    single = deepcopy(manifest)
    single['regions'] = [deepcopy(region)]
    info = {'path': str(crop), 'sha256': _sha(crop), 'width': 606, 'height': 1024,
            'crop_bbox': region['crop_bbox'], 'offset': region['offset'],
            'original_path': str(source), 'original_sha256': _sha(source),
            'manifest_path': str(path), 'manifest_sha256': _sha(path),
            'pixel_exact': True, 'rescaled': False, 'image_count': 1}
    return single, crop, info


def build_middle_prompt():
    prompt = build_scene_prompt((606, 1024))
    prompt = prompt.replace('"text"|"math"', '"text"|"math"|"rich"')
    prompt = prompt.replace('fill additionally permits "none".', 'fill and stroke additionally permit "none".')
    return prompt + r'''
SINGLE-REGION TASK: Observe ONLY the attached middle-panel crop. Its canvas is
606 x 1024 pixels with LOCAL origin (0,0). Do not add any global coordinate offset.
Do not reconstruct neighboring panels. Preserve ALL visible passages: panel title,
step numbers, titles, every explanatory bullet, function/variable names, formulas,
subscripts, numerical values and dimensions. Do not summarize separate passages.
Count visible main steps and nested substeps separately; assume no preset count.
Keep the observed hierarchy, cards, title strips, badges and ONLY visible lines.
Do not invent arrows from adjacency or text from general GPT-2 knowledge.
Do not complete text cut off at crop margins. Record uncertain/unreadable fragments
and their local bboxes in metadata; never guess merely to make the Scene complete.
RICH TEXT: mode="rich" uses segments:[{type:"text"|"math",content:"..."}], in
original order (1..128 segments). Chinese/English prose and code-like names go in
text segments; formulas go in math segments WITHOUT dollar delimiters. Optional
text preserves source wording; segments alone are valid. Never render raw LaTeX
as literal prose. Preserve meaningful newlines. Never duplicate a passage as both
a text node and a separate label. Prefer complete transcription over decoration.
Only existing primitive types/colors are legal; do not add circle/ellipse types.
If a primitive cannot represent the exact shape, record that limitation in metadata.
Assignment arrows printed inside text are content, not connecting edges. Preserve
an operator unsupported by the math allowlist as visible Unicode in a text segment.
Do not add root type="json_object". Return ONE complete strict JSON, no Markdown.
'''


def compatible_scene(scene):
    result, changes = deepcopy(scene), []
    if result.get('type') == 'json_object':
        del result['type']
        changes.append({'path': '$.type', 'before': 'json_object', 'action': 'remove_transport_marker'})
    # stroke=none is already supported. Do not rewrite text, math, IDs or geometry.
    return result, changes


def _text(item):
    if item.get('mode') == 'rich':
        return ''.join(s.get('content', '') for s in item.get('segments', []) if isinstance(s, dict))
    return item.get('text', '')


def text_inventory(scene):
    return [{'id': item['id'], 'collection': kind, 'content': _text(item),
             'mode': item.get('mode', 'text'), 'segments': deepcopy(item.get('segments')),
             'bbox': item.get('bbox', item.get('metadata', {}).get('bbox')),
             'position': [item.get('x'), item.get('y')]}
            for kind in ('nodes', 'labels') for item in scene.get(kind, []) if _text(item)]


def validate_local_scene(scene, region):
    try:
        make_region_scene(region, scene)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return {'schema_valid': False, 'errors': [{'code': 'invalid_local_scene', 'message': str(exc)}], 'risks': []}
    # Old diagnostics expect text even for rich labels. Adapt only a throwaway
    # diagnostic copy; do not mutate or render the source text twice.
    diagnostic = deepcopy(scene)
    for item in diagnostic['nodes'] + diagnostic['labels']:
        if item.get('mode') == 'rich':
            item['text'] = _text(item)
    report = diagnose_scene(diagnostic, image_size=(606, 1024))
    texts, seen = text_inventory(scene), {}
    for item in texts:
        key = item['content'].strip()
        if key in seen:
            report['risks'].append({'code': 'repeated_text_requires_review', 'ids': [seen[key], item['id']],
                                    'message': 'Repeated wording can be legitimate; no automatic deletion.'})
        seen[key] = item['id']
    if len(texts) < 10:
        report['risks'].append({'code': 'few_text_objects_requires_review', 'count': len(texts),
                                'message': 'A warning only; not a content completeness score.'})
    report['content_completeness_verified'] = False
    report['rich_diagnostic_text_only_on_copy'] = True
    return report


def map_to_global(scene, single_manifest):
    if (len(single_manifest.get('regions', [])) != 1
            or single_manifest['regions'][0].get('region_id') != 'region_002'
            or single_manifest['regions'][0].get('offset') != [444, 0]
            or single_manifest['regions'][0].get('crop_bbox') != [444, 0, 606, 1024]):
        raise ValueError('Map exactly one middle region using crop offset [444,0].')
    region = single_manifest['regions'][0]
    bundle = make_region_scene(region, scene, metadata={'source': scene.get('metadata', {}).get('source', 'unverified_region_scene')})
    result, report = merge_region_scenes(single_manifest, [bundle])
    result['metadata']['coverage'] = 'middle_region_only_not_full_image_reconstruction'
    result['metadata']['crop_origin'] = list(region['offset'])
    return result, report


def mock_scene():
    return {'canvas': {'width': 606, 'height': 1024, 'coordinate_system': 'top-left', 'scale': .4},
            'styles': {'plain': {'stroke': 'none', 'fill': 'none'}},
            'nodes': [{'id': 'tensor', 'type': 'tensor3d', 'bbox': [40, 150, 100, 60],
                       'grid': {'rows': 2, 'cols': 3}, 'depth': {'dx': 12, 'dy': -10}},
                      {'id': 'matrix', 'type': 'matrix', 'bbox': [220, 150, 120, 60], 'grid': {'rows': 2, 'cols': 4}}],
            'labels': [{'id': 'caption', 'mode': 'rich', 'x': 30, 'y': 40, 'anchor': 'north west',
                        'segments': [{'type': 'text', 'content': '离线 Mock：中文与公式 '},
                                     {'type': 'math', 'content': r'W\in\mathbb{R}^{4\times8}'}],
                        'style': 'plain', 'metadata': {'node_id': 'matrix', 'bbox': [30, 40, 450, 35]}}],
            'edges': [{'id': 'path', 'points': [[140, 180], [220, 180]], 'arrow': 'end',
                       'metadata': {'source_id': 'tensor', 'target_id': 'matrix'}}],
            'metadata': {'source': 'synthetic_mock_not_recognition'}}


def _configuration():
    report = {'environment_present': {name: bool(os.environ.get(name, '').strip())
                                      for name in ('API_BASE_URL', 'API_KEY', 'MODEL_NAME')},
              'ready': False, 'network_checked': False}
    key = os.environ.get('API_KEY', '')
    try:
        config = APIConfig.from_env()
        client = VLMClient(config, allow_api=False, max_tokens=MAX_TOKENS,
                           token_limit_parameter='max_tokens', thinking_mode='disabled',
                           save_truncated_response=True)
        client.thinking_config = ThinkingConfig(thinking_mode='disabled')
        supported = _is_official_flash(config, {})
        report.update(ready=supported, model=_redact(config.model, key),
                      json_output_supported_by_existing_adapter=supported,
                      thinking={'type': 'disabled'}, max_tokens=MAX_TOKENS)
        if not supported:
            report['error'] = 'Existing JSON adapter is verified only for official DeepSeek flash; no request sent.'
        return client, report
    except ValueError as exc:
        report['error'] = _redact(str(exc), key)
        return None, report


def _baseline(path):
    if path is None or not Path(path).is_file():
        return {'available': False, 'reason': 'No available baseline supplied.'}
    path = Path(path)
    scene = _strict_json(path.read_text(encoding='utf-8-sig'))
    validate_scene(scene)
    nodes = {item['id']: item for item in scene['nodes']}
    objects = {item['id']: item for kind in ('nodes', 'labels') for item in scene[kind]}
    items = []
    for item in text_inventory(scene):
        box = item['bbox']
        x = box[0] + box[2] / 2 if box else item['position'][0]
        if isinstance(x, (int, float)) and 456 <= x < 1038:
            source = objects[item['id']]
            owner = nodes.get(source.get('metadata', {}).get('node_id'), {})
            extent = box or owner.get('bbox')
            item['scope'] = ('cross_region_title_or_content' if extent and
                             (extent[0] < 444 or extent[0] + extent[2] > 1050)
                             else 'middle_panel')
            items.append(item)
    return {'available': True, 'path': str(path.resolve()), 'sha256': _sha(path),
            'selection': 'text anchor or bbox centre in middle core x=[456,1038)',
            'text_items': items, 'semantic_completeness': 'requires_original_image_review'}


def run_experiment(args, *, client=None):
    if args.max_api_requests != 1 or args.max_tokens != MAX_TOKENS:
        raise ValueError('Exactly one request with max_tokens=32768 is permitted.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = Path(args.output_root).resolve() / ('region_vlm_middle_' + stamp + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'preparing', 'output_dir': str(output), 'allow_api': args.allow_api is True,
              'api_called': False, 'api_request_count': 0, 'compile_count': 0,
              'http_status': None, 'finish_reason': None, 'usage': None,
              'recognition_performed': False, 'content_completeness_verified': False}
    ledger, info, key = None, None, os.environ.get('API_KEY', '')
    try:
        manifest, crop, info = load_middle_input(args.manifest)
        report['input'] = info
        saved_manifest = deepcopy(manifest)
        saved_manifest['image']['path'] = info['original_path']
        saved_manifest['regions'][0]['crop_path'] = 'region_input.png'
        _save(output / 'region_manifest.json', saved_manifest)
        shutil.copy2(crop, output / 'region_input.png')
        configured_client, configuration = _configuration()
        report.update(api_configuration=configuration, api_config_ready=configuration['ready'])
        prompt = build_middle_prompt()
        _save(output / 'request.json', {'authorized': report['allow_api'], 'sent': False,
              'configuration': configuration, 'image': info, 'prompt': prompt,
              'max_api_requests': 1, 'max_tokens': MAX_TOKENS, 'thinking': {'type': 'disabled'},
              'response_format': {'type': 'json_object'}, 'stream': False, 'automatic_retries': 0,
              'response_raw_scope': 'complete_visible_message_content_redacted_not_HTTP_envelope'})
        baseline = _baseline(args.baseline_scene)
        _save(output / 'baseline_middle_inventory.json', baseline)
        completeness = {'status': 'not_evaluated_no_real_response', 'recognition_performed': False,
                        'visible_main_steps': None, 'visible_nested_substeps': None,
                        'restored': None, 'complete': None, 'partial': None, 'missing': None,
                        'text_categories': {k: None for k in ['Chinese', 'functions', 'variables', 'formulas', 'numbers']},
                        'improved_over_full_image': None, 'baseline': baseline,
                        'required_review': 'Compare every visible passage, main/substep, card, badge and actual line against the real response and rendered PNG; counts and Schema are not completeness evidence.'}
        _save(output / 'content_completeness.json', completeness)
        # Offline configuration and Mock validation precede even an authorized request.
        fixture = mock_scene()
        validation = validate_local_scene(fixture, manifest['regions'][0])
        if not validation['schema_valid'] or validation['errors']:
            raise ValueError('Offline Mock local validation failed.')
        global_fixture, mapping = map_to_global(fixture, manifest)
        _save(output / 'mock_local_scene.json', fixture)
        _save(output / 'mock_global_scene.json', global_fixture)
        _save(output / 'mock_merge_report.json', mapping)
        _save(output / 'mock_validation.json', {'scope': 'synthetic_mock_only', 'local': validation, 'global_schema_valid': True})
        (output / 'mock_scene.tex').write_text(wrap_document(generate_tikz(fixture)), encoding='utf-8')
        report.update(mock_schema_valid=True, mock_global_mapping_valid=True,
                      mock_counts={k: len(fixture[k]) for k in ('nodes', 'labels', 'edges')})
        if not report['allow_api']:
            _save(output / 'validation.json', {'scope': 'synthetic_mock_only', 'local': validation, 'global_schema_valid': True})
            report['status'] = 'offline_ready_api_not_authorized'
            return report
        if not configuration['ready']:
            raise ValueError('API configuration preflight failed; inspect sanitized request.json.')
        client = client or configured_client
        key = client.config.api_key
        if client.max_tokens != MAX_TOKENS or not _is_official_flash(client.config, {}):
            raise ValueError('Client differs from the validated one-request configuration.')
        client.allow_api = True
        client.thinking_config = ThinkingConfig(thinking_mode='disabled')
        budget = SimpleNamespace(allow_api=True, max_api_requests=1, max_output_budget=MAX_TOKENS,
                                 max_total_usage_tokens=MAX_TOKENS, allow_completion=False,
                                 allow_visual_refinement=False, stage_strategy='fixed')
        ledger = RequestLedger(client, budget, output)
        report['status'] = 'request_attempting'
        _save(output / 'run_report.json', report)
        report['api_called'] = True
        raw = _redact(ledger.request([output / 'region_input.png'], prompt, 'generate'), key)
        report['response_received'] = True
        (output / 'response_raw.txt').write_text(raw, encoding='utf-8')
        parsed = _strict_json(raw)
        _save(output / 'response_parsed.json', parsed)
        scene, changes = compatible_scene(parsed)
        scene = normalize_scene_provenance(scene)
        _save(output / 'compatibility_repairs.json', changes)
        _save(output / 'local_scene.json', scene)
        report['recognition_performed'] = True
        report['local_counts'] = {k: len(scene[k]) if isinstance(scene.get(k), list) else None
                                  for k in ('nodes', 'labels', 'edges')}
        validation = validate_local_scene(scene, manifest['regions'][0])
        _save(output / 'validation.json', validation)
        report['schema_valid'] = validation['schema_valid']
        completeness.update(status='requires_manual_content_review', recognition_performed=True,
                            restored_text_inventory=text_inventory(scene) if validation['schema_valid'] else None)
        _save(output / 'content_completeness.json', completeness)
        if not validation['schema_valid'] or validation['errors']:
            report['status'] = 'scene_invalid_no_retry'
            return report
        global_scene, mapping = map_to_global(scene, manifest)
        _save(output / 'global_scene.json', global_scene)
        _save(output / 'merge_report.json', mapping)
        (output / 'scene.tex').write_text(wrap_document(generate_tikz(scene)), encoding='utf-8')
        report.update(compile_count=1, status='compiling_local_scene')
        _save(output / 'run_report.json', report)
        compiled = compile_bounded(output / 'scene.tex', xelatex=args.xelatex, timeout=60, memory_mb=512)
        report['compile'] = asdict(compiled)
        shutil.copy2(compiled.log_path, output / 'compile.log')
        if not compiled.success:
            report['status'] = 'compile_failed_no_retry'
            return report
        png = render_pdf(compiled.pdf_path, output / 'scene.png', dpi=150)
        comparison = make_comparison(output / 'region_input.png', png, output / 'comparison.png')
        report.update(status='compiled_requires_content_review', pdf=str(compiled.pdf_path),
                      png=str(png), comparison=str(comparison), global_mapping_valid=True)
    except Exception as exc:
        report.update(status='failed_no_retry', error=_redact(str(exc), key))
        report['error_type'] = type(exc).__name__
        known_http_error = re.match(r'API returned HTTP ([1-5][0-9]{2})\.', str(exc))
        if known_http_error:
            report['http_status'] = int(known_http_error.group(1))
        try:
            if not (output / 'validation.json').exists():
                _save(output / 'validation.json', {'scope': 'real_response_not_validated',
                      'schema_valid': None, 'errors': [report['error']]})
            if client is not None and getattr(client, 'last_truncated_text', None):
                (output / 'response_raw.txt').write_text(_redact(client.last_truncated_text, key), encoding='utf-8')
        except Exception as logging_error:
            report['failure_log_write_error'] = type(logging_error).__name__
    finally:
        if ledger is not None:
            report['request_ledger'] = ledger.summary()
            report['api_request_count'] = len(ledger.entries)
            diagnostic = getattr(client, 'last_diagnostics', {})
            report.update(http_status=diagnostic.get('http_status') or report['http_status'], finish_reason=diagnostic.get('finish_reason'),
                          usage=diagnostic.get('usage'))
            for field in ('error_stage', 'error_type', 'error_code', 'content_type',
                          'response_byte_length', 'response_sha256', 'read_state',
                          'response_saved', 'response_recoverable', 'capture'):
                if diagnostic.get(field) is not None:
                    report[field] = diagnostic[field]
            try:
                request = _strict_json((output / 'request.json').read_text(encoding='utf-8'))
                request['sent'] = bool(ledger.entries)
                _save(output / 'request.json', request)
            except Exception as logging_error:
                report['request_log_write_error'] = type(logging_error).__name__
        if info is not None:
            try:
                report['source_files_unchanged'] = all(Path(info[p]).is_file() and _sha(info[p]) == info[h] for p, h in
                    [('path', 'sha256'), ('original_path', 'original_sha256'), ('manifest_path', 'manifest_sha256')])
            except OSError as verification_error:
                # Unknown is distinct from a confirmed change; preserve the API error.
                report['source_files_unchanged'] = None
                report['source_verification_error_type'] = type(verification_error).__name__
        try:
            _save(output / 'run_report.json', json.loads(_redact(json.dumps(report, ensure_ascii=False), key)))
        except Exception as logging_error:
            report['run_report_write_error'] = type(logging_error).__name__
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--baseline-scene', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--allow-api', action='store_true', help='Explicitly authorize one paid DeepSeek request.')
    parser.add_argument('--max-api-requests', type=int, choices=(1,), default=1)
    parser.add_argument('--max-tokens', type=int, choices=(MAX_TOKENS,), default=MAX_TOKENS)
    parser.add_argument('--xelatex', default='xelatex')
    return parser


def main(argv=None):
    report = run_experiment(build_parser().parse_args(argv))
    print('Status:', report['status'])
    if report.get('error'):
        print('Error stage:', report.get('error_stage', 'local_processing'))
        print('Error type:', report.get('error_type', 'unknown'))
        print('HTTP status:', report.get('http_status'))
        print('Error:', report['error'])
    print('Output:', report['output_dir'])
    print('Real API attempts:', report['api_request_count'])
    print('Compile attempts:', report['compile_count'])
    return 0 if report['status'] in {'offline_ready_api_not_authorized', 'compiled_requires_content_review'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
