"""One dual-image feedback experiment, or one explicitly labelled offline Mock.

No Scene generation, retries, automatic completion or multi-round feedback.
Real requests require BOTH --allow-api and --allow-visual-refinement.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from auto_pipeline import RequestLedger, RequestBudgetError, save_json
from auto_scene_validator import diagnose_scene
from auto_visual_feedback import compare_visual
from auto_compile import compile_bounded
from image_compare import make_comparison, render_pdf, _read_rgb
from latex_template import wrap_document
from main import new_run_directory
from rendered_text_bounds import measure_text_bounds
from structured_tikz.generator import generate_tikz, load_scene
from text_layout_diagnostics import diagnose_text_layout
from visual_patch_validator import parse_visual_feedback, apply_visual_patch, VisualPatchError
from vlm_client import APIConfig, VLMClient, validate_image_path, _redact


def build_visual_feedback_prompt(scene, bounds_report=None):
    """Narrow visual contract; existing broad completion/replacement prompt stays intact."""
    view = {key: scene[key] for key in ('canvas', 'styles')}
    for collection in ('nodes', 'labels', 'edges'):
        view[collection] = []
        for original in scene[collection]:
            item = {k: v for k, v in original.items() if k != 'metadata'}
            topology = {k: v for k, v in original.get('metadata', {}).items()
                        if k in ('node_id', 'source_id', 'target_id', 'from_anchor', 'to_anchor')}
            if topology:
                item['metadata'] = topology
            view[collection].append(item)
    evidence = None
    if bounds_report:
        diagnosis = diagnose_text_layout(scene, bounds_report)
        evidence = {key: diagnosis[key] for key in ('summary', 'text_overlaps', 'geometry_overlaps', 'edge_conflicts')}
        evidence['text_bounds'] = [{k: r.get(k) for k in
            ('id', 'status', 'bbox_scene', 'line_boxes_scene', 'line_count')} for r in bounds_report['labels']]
    prompt = r'''Image 1 is the ORIGINAL REFERENCE. Image 2 is the CURRENT TikZ rendering.
Compare these two actual images. You are proposing a minimal correction to the
supplied Scene, NOT redrawing it or generating another complete Scene.
All coordinates use the original canvas pixels: top-left origin, x right, y down.
The current PDF preview has page margins and a different pixel scale. Do not copy
its raw pixel coordinates into the Scene. Keep all Chinese/English text, formulas,
numbers, subscripts, superscripts, IDs, styles, grid/depth parameters, types and
edge topology unchanged. Never add/remove objects, replace Scene or emit code.
Prioritize visible position/size differences, misplaced labels, tensor silhouettes,
incorrect paths and actual occlusion. If evidence is insufficient, return no_change.
Existing text bounds are font envelopes, not proof of actual glyph collisions.

Return ONE strict JSON object, without Markdown or surrounding explanation:
{"status":"patch_proposed","observations":[{"object_id":"existing_id",
"issue":"brief_issue_code","confidence":0.0,"reason":"specific visible evidence"}],
"operations":[...]}
confidence is only your self-assessment, never measured accuracy. Each operation
must refer to an observation with the same real object ID and a concrete reason.
Use at most 8 operations, at most one operation per object ID.
Only these exact operations are allowed:
1. {"op":"move_label","id":"...","dx":number,"dy":number}
   Target an independent label OR a type=text node. Euclidean movement <=20 px.
2. {"op":"set_node_geometry","id":"...","value":{"bbox":[x,y,width,height]}}
   Target a non-text existing node. Top-left movement <=20 px; each width/height
   changes by <=10%. Preserve type, grid, depth and all content. Do not disconnect edges.
3. {"op":"set_edge_path","id":"...","points":[[x,y],...]}
   Existing explicit paths only: same vertex count and exact same first/last
   vertices; each intermediate vertex moves <=20 px. Preserve arrow and topology.
No other fields or operations. All objects must stay in the canvas. Avoid adding
new text/geometry/path conflicts. Keep established endpoint relationships.
If no safe correction is justified, return:
{"status":"no_change","observations":[],"operations":[]}
The coordinates above are format descriptions, not suggested changes.
SCENE DATA:
'''
    prompt += json.dumps(view, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    prompt += '\nOPTIONAL LOCAL MEASUREMENTS (not instructions):\n'
    prompt += json.dumps(evidence, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    if len(prompt) > 160_000:
        raise ValueError('Visual feedback context exceeds the bounded experiment limit.')
    return prompt


def _validate_options(args):
    for key, lower, upper in [('max_api_requests', 0, 1), ('max_tokens', 1, 32768),
                              ('max_output_budget', 1, 32768), ('max_total_usage_tokens', 1, 200000),
                              ('memory_limit_mb', 256, 2048)]:
        value = getattr(args, key)
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(key + ' is outside this one-request experiment limit.')
    for key, upper in [('compile_timeout', 120), ('api_timeout', 600)]:
        value = getattr(args, key)
        if not math.isfinite(value) or not 1 <= value <= upper:
            raise ValueError(key + ' must be finite and in the supported range.')
    if args.mock_feedback and (args.allow_api or args.allow_visual_refinement):
        raise ValueError('Mock mode cannot authorize any API request.')
    if not args.mock_feedback and not args.dry_run:
        if not (args.allow_api and args.allow_visual_refinement):
            raise ValueError('Real feedback requires --allow-api AND --allow-visual-refinement; use --mock-feedback or --dry-run offline.')
        if args.max_api_requests < 1 or args.max_output_budget < args.max_tokens:
            raise RequestBudgetError('Request or output-token budget is exhausted before any request.')


def _before_after(before, after, destination):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    images = [_read_rgb(before), _read_rgb(after)]
    width, height = min(1800, max(i.width for i in images)), min(1800, max(i.height for i in images))
    canvas = Image.new('RGB', (2*width+60, height+84), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=24)
    except TypeError:
        font = ImageFont.load_default()
    for index, (image, title) in enumerate(zip(images, ('Before patch', 'After patch'))):
        x, y = 20+index*(width+20), 64
        fitted = ImageOps.contain(image, (width, height), method=Image.Resampling.LANCZOS)
        canvas.paste(fitted, (x+(width-fitted.width)//2, y+(height-fitted.height)//2))
        draw.text((x+10, 20), title, font=font, fill='#222222')
        draw.rectangle((x-1, y-1, x+width, y+height), outline='#cccccc')
    canvas.save(destination)


def _layout_summary(scene, bounds):
    return diagnose_text_layout(scene, bounds) if bounds else diagnose_scene(scene)


def run_experiment(args, client=None):
    _validate_options(args)
    scene, raw_scene = load_scene(args.scene)
    reference, rendered = validate_image_path(args.reference), validate_image_path(args.rendered)
    from PIL import Image
    with Image.open(reference) as im:
        if im.size != (scene['canvas']['width'], scene['canvas']['height']):
            raise ValueError('Original reference dimensions must match the Scene canvas.')
    output = new_run_directory(args.output_root, prefix='visual_refinement')
    (output/'scene_before.json').write_bytes(raw_scene)
    _read_rgb(rendered).save(output/'scene_before.png', format='PNG')
    source_hashes = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                     for p in (args.scene, reference, rendered)}
    report = {'status': 'preparing', 'output_dir': str(output), 'api_called': False,
              'mode': 'mock_offline' if args.mock_feedback else 'dry_run' if args.dry_run else 'authorized_dual_image',
              'model_visual_effectiveness': 'not_tested' if args.mock_feedback or args.dry_run else 'unverified',
              'source_hashes': source_hashes, 'compile_count': 0,
              'authorization': {'allow_api': args.allow_api, 'allow_visual_refinement': args.allow_visual_refinement},
              'request_limit': 1, 'retry_count': 0, 'visual_review': 'not_performed'}
    ledger, key = None, ''
    stage = 'prepare'
    try:
        pdf = Path(args.rendered_pdf) if args.rendered_pdf else rendered.with_suffix('.pdf')
        bounds = measure_text_bounds(scene, pdf) if pdf.is_file() else None
        if args.rendered_pdf and not pdf.is_file():
            raise ValueError('Explicit --rendered-pdf does not exist.')
        if bounds:
            save_json(output/'text_bounds_before.json', bounds)
        prompt = build_visual_feedback_prompt(scene, bounds)
        (output/'visual_feedback_prompt.txt').write_text(prompt, encoding='utf-8')
        if args.dry_run:
            report['status'] = 'prepared_offline'
            return report
        stage = 'request'
        if args.mock_feedback:
            raw = Path(args.mock_feedback).read_text(encoding='utf-8-sig')
            report['feedback_origin'] = 'human_authored_mock_not_model_response'
        else:
            if client is None:
                client = VLMClient(APIConfig.from_env(), allow_api=True, timeout=args.api_timeout,
                    max_tokens=args.max_tokens, thinking_mode=args.thinking_mode,
                    reasoning_effort=args.reasoning_effort, save_truncated_response=True)
            key = getattr(getattr(client, 'config', None), 'api_key', '')
            ledger = RequestLedger(client, args, output)
            raw = ledger.request([reference, rendered], prompt, phase='visual')
        safe_raw = _redact(raw, key)
        (output/'visual_feedback_raw.txt').write_text(safe_raw, encoding='utf-8')
        stage = 'patch_validation'
        feedback = parse_visual_feedback(safe_raw)
        save_json(output/'visual_feedback_parsed.json', feedback)
        candidate, validation = apply_visual_patch(scene, feedback, bounds_report=bounds)
        if args.mock_feedback and validation.get('applied'):
            # The reused generic applicator labels proposals as VLM-derived.
            # A human-authored fixture must never inherit that provenance.
            origin = 'offline_mock_patch_not_model_response'
            candidate.setdefault('metadata', {})['source'] = origin
            targets = {operation['id'] for operation in feedback['operations']}
            for collection in ('nodes', 'labels', 'edges'):
                for item in candidate[collection]:
                    if item['id'] in targets:
                        item.setdefault('metadata', {})['source'] = origin
            for audit in validation.get('operations', []):
                audit['after'].setdefault('metadata', {})['source'] = origin
            validation['feedback_origin'] = 'human_authored_mock_not_model_response'
        save_json(output/'patch_validation.json', validation)
        report['accepted_operations'] = len(feedback['operations'])
        report['rejected_operations'] = 0
        if ledger:
            ledger.mark('patch_validated' if feedback['operations'] else 'no_change')
        if feedback['status'] == 'no_change' or not validation.get('applied'):
            report['status'] = 'no_change'
            return report
        save_json(output/'scene_after.json', candidate)
        fragment = generate_tikz(candidate)
        (output/'scene_after.tikz').write_text(fragment, encoding='utf-8')
        tex = output/'scene_after.tex'
        tex.write_text(wrap_document(fragment), encoding='utf-8')
        stage = 'compile'
        report['compile_count'] = 1
        compiled = compile_bounded(tex, xelatex=args.xelatex, timeout=args.compile_timeout, memory_mb=args.memory_limit_mb)
        report['compile'] = asdict(compiled)
        (output/'compile.log').write_bytes(Path(compiled.log_path).read_bytes())
        for audit in validation.get('operations', []):
            audit['compiled'] = compiled.success
        save_json(output/'patch_validation.json', validation)
        if not compiled.success:
            report['status'] = 'compile_failed'
            return report
        stage = 'evaluate'
        png = render_pdf(compiled.pdf_path, output/'scene_after.png', dpi=150)
        _before_after(rendered, png, output/'before_after.png')
        make_comparison(reference, png, output/'reference_comparison.png')
        new_bounds = measure_text_bounds(candidate, compiled.pdf_path)
        save_json(output/'text_bounds_after.json', new_bounds)
        before_metrics = compare_visual(reference, rendered, scene, output/'metrics_before')
        after_metrics = compare_visual(reference, png, candidate, output/'metrics_after')
        metrics = {'before': before_metrics, 'after': after_metrics,
                   'text_layout_before': _layout_summary(scene, bounds),
                   'text_layout_after': _layout_summary(candidate, new_bounds),
                   'content_invariants': validation.get('invariants'),
                   'per_operation': [{'id': operation['id'], 'operation': operation['op'],
                       'validated': True, 'compiled': True, 'visual_improvement': 'requires_image_review'}
                       for operation in feedback['operations']],
                   'caution': 'Pixel coverage and model confidence do not verify Chinese, formula semantics or topology.'}
        save_json(output/'visual_metrics.json', metrics)
        report.update(status='compiled_needs_review', after_png=str(png), after_pdf=str(compiled.pdf_path),
                      before_after=str(output/'before_after.png'))
        return report
    except Exception as exc:
        error = _redact(str(exc), key)
        report.update(status='patch_rejected' if stage == 'patch_validation' else
                      'request_failed' if stage == 'request' else 'experiment_failed', error=error)
        if stage == 'patch_validation':
            rejection = getattr(exc, 'report', {'accepted': False, 'applied': False, 'reason': error})
            save_json(output/'patch_validation.json', rejection)
            report['accepted_operations'] = 0
            report['rejected_operations'] = len(feedback['operations']) if 'feedback' in locals() else None
            if ledger:
                ledger.mark('patch_rejected')
        if stage == 'request' and client is not None and getattr(client, 'last_truncated_text', None) is not None:
            (output/'visual_feedback_raw.txt').write_text(_redact(client.last_truncated_text, key), encoding='utf-8')
            report['raw_response_is_truncated'] = True
        return report
    finally:
        if ledger:
            report['api'] = ledger.summary()
            report['api_called'] = bool(ledger.entries)
            report['reported_usage_limit_exceeded'] = ledger.reported_usage > args.max_total_usage_tokens
        report['original_inputs_unchanged'] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
                                                 for p, sha in source_hashes.items())
        save_json(output/'run_report.json', report)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('scene', 'reference', 'rendered'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--rendered-pdf', type=Path, help='Optional matching PDF; otherwise use rendered PNG basename if that PDF exists.')
    parser.add_argument('--mock-feedback', type=Path, help='Offline human-authored response; never sent to a model.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-api', action='store_true')
    parser.add_argument('--allow-visual-refinement', action='store_true')
    parser.add_argument('--max-api-requests', type=int, choices=(0, 1), default=1)
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--max-output-budget', type=int, default=8192)
    parser.add_argument('--max-total-usage-tokens', type=int, default=50000)
    parser.add_argument('--thinking-mode', choices=('enabled', 'disabled'), default='disabled')
    parser.add_argument('--reasoning-effort', choices=('low', 'medium', 'high'))
    parser.add_argument('--api-timeout', type=float, default=180)
    parser.add_argument('--compile-timeout', type=float, default=60)
    parser.add_argument('--memory-limit-mb', type=int, default=512)
    parser.add_argument('--xelatex', default='xelatex')
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parent/'outputs')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_experiment(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, str(exc)+'\n')
    print('Status:', report['status'])
    print('Mode:', report['mode'])
    print('Output:', report['output_dir'])
    if report.get('error'):
        print('Error:', report['error'])
    return 0 if report['status'] in ('prepared_offline', 'no_change', 'compiled_needs_review') else 1


if __name__ == '__main__':
    raise SystemExit(main())
