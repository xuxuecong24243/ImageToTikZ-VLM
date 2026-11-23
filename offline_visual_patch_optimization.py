"""One offline optimization of saved real feedback, with at most one compilation.

There is no model request path, retry loop or authorization flag in this entry.
Saved evidence must match the original feedback run before it can be used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

from auto_compile import compile_bounded
from image_compare import make_comparison, render_pdf, _read_rgb
from latex_template import wrap_document
from rendered_text_bounds import measure_text_bounds
from structured_tikz.generator import generate_tikz, load_scene, _unique_keys
from structured_tikz.validator import validate_scene
from text_layout_diagnostics import diagnose_text_layout
from visual_patch_optimizer import optimize_visual_patches, scene_fingerprint, measure_layout
from visual_patch_validator import apply_visual_patch, parse_visual_feedback
from visual_refinement_experiment import _before_after

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / 'outputs/layout_refinement_20261113T053553Z_94a01765'
FEEDBACK_RUN = ROOT / 'outputs/visual_refinement_20261113T125903Z_c1b611f6'


def _save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=str) + '\n', encoding='utf-8')


def _bad_constant(value):
    raise ValueError('Non-finite JSON constant: ' + value)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      object_pairs_hook=_unique_keys, parse_constant=_bad_constant)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_verified_inputs(args):
    """Bind bounds and feedback to the historical Scene/reference/rendered pair."""
    scene, raw_scene = load_scene(args.scene)
    feedback_path, bounds_path = Path(args.feedback).resolve(), Path(args.bounds).resolve()
    history = feedback_path.parent
    run_path = history / 'run_report.json'
    saved_scene_path = history / 'scene_before.json'
    saved_bounds_path = history / 'text_bounds_before.json'
    saved_response = history / 'request_01/response.txt'
    parsed_path = history / 'visual_feedback_parsed.json'
    historical_rejection = history / 'patch_validation.json'
    run, bounds = _read_json(run_path), _read_json(bounds_path)
    feedback = parse_visual_feedback(feedback_path.read_text(encoding='utf-8-sig'))
    if not run.get('api_called') or run.get('mode') != 'authorized_dual_image':
        raise ValueError('Input is not a saved real dual-image feedback run.')
    requests = run.get('api', {}).get('requests', [])
    if len(requests) != 1:
        raise ValueError('Expected exactly one saved feedback request.')
    diagnostics = requests[0].get('diagnostics', {})
    if diagnostics.get('http_status') != 200 or diagnostics.get('finish_reason') != 'stop':
        raise ValueError('Saved response did not finish successfully; refusing partial feedback.')
    if feedback != _read_json(parsed_path) or feedback != parse_visual_feedback(
            saved_response.read_text(encoding='utf-8-sig')):
        raise ValueError('Feedback differs from the historical raw/parsed response.')
    saved_scene, _ = load_scene(saved_scene_path)
    if scene != saved_scene:
        raise ValueError('Scene version differs from the input used by the saved feedback run.')
    saved_bounds = _read_json(saved_bounds_path)
    bounds_selection = 'saved_measurement_report'
    if isinstance(bounds, dict) and 'after' in bounds:
        if (bounds.get('after') != diagnose_text_layout(scene, saved_bounds)
                or bounds.get('after_pdf') != saved_bounds.get('pdf')):
            raise ValueError('Combined bounds report does not match the saved baseline measurements.')
        bounds = saved_bounds
        bounds_selection = 'after_diagnosis_verified_against_saved_measurements'
    elif bounds != saved_bounds:
        raise ValueError('Bounds must be the matching saved text_bounds_before report.')
    expected = {str(Path(p).resolve()): digest for p, digest in run.get('source_hashes', {}).items()}
    for path in (args.scene, args.reference, args.rendered):
        path = Path(path).resolve()
        if expected.get(str(path)) != _sha(path):
            raise ValueError('Source file/version does not match the feedback run: ' + str(path))
    if _read_rgb(args.reference).size != (scene['canvas']['width'], scene['canvas']['height']):
        raise ValueError('Reference dimensions differ from the Scene canvas.')
    if bounds.get('mapping', {}).get('status') != 'verified':
        raise ValueError('Saved PDF-to-Scene mapping is not verified.')
    if Path(bounds.get('pdf', '')).resolve() != Path(args.rendered).resolve().with_suffix('.pdf'):
        raise ValueError('Measured PDF does not correspond to the supplied rendering.')
    rejection = _read_json(historical_rejection)
    paths = (args.scene, args.reference, args.rendered, feedback_path, bounds_path,
             run_path, saved_scene_path, saved_bounds_path, saved_response, parsed_path, historical_rejection)
    hashes = {str(Path(p).resolve()): _sha(p) for p in paths}
    source_summary = {'feedback_run': str(history), 'historical_diagnostics': diagnostics,
                      'historical_patch_rejection': rejection,
                      'scene_and_images_match_saved_hashes': True,
                      'bounds_match_saved_scene_version': True,
                      'bounds_input': str(bounds_path),
                      'bounds_selection': bounds_selection,
                      'bounds_are_font_envelopes_not_glyph_masks': True}
    return scene, raw_scene, feedback, bounds, hashes, source_summary


def _mark_local_origin(candidate, validation, optimized):
    origin = 'offline_geometry_optimization_of_saved_real_visual_feedback'
    candidate.setdefault('metadata', {})['source'] = origin
    targets = {operation['id'] for operation in optimized['operations']}
    for collection in ('nodes', 'labels'):
        for item in candidate[collection]:
            if item['id'] in targets:
                item.setdefault('metadata', {})['source'] = origin
    for operation in validation.get('operations', []):
        operation['after'].setdefault('metadata', {})['source'] = origin
    validation['selected_displacement_source'] = origin
    validate_scene(candidate)


def run_experiment(args):
    if not math.isfinite(args.compile_timeout) or not 1 <= args.compile_timeout <= 120:
        raise ValueError('Compile timeout must be in [1,120] seconds.')
    if isinstance(args.memory_limit_mb, bool) or not 256 <= args.memory_limit_mb <= 2048:
        raise ValueError('Compiler memory limit must be in [256,2048] MB.')
    scene, raw_scene, feedback, bounds, hashes, provenance = _load_verified_inputs(args)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = Path(args.output_root).resolve() / ('offline_patch_optimization_' + stamp + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'preparing', 'mode': 'offline_saved_real_feedback', 'api_called': False,
              'compile_count': 0, 'output_dir': str(output), 'source_hashes': hashes,
              'provenance': provenance, 'visual_review': 'requires_png_inspection'}
    stage = 'optimize'
    try:
        (output / 'scene_before.json').write_bytes(raw_scene)
        _save(output / 'model_feedback_original.json', feedback)
        _save(output / 'text_bounds_before.json', bounds)
        optimized, evaluation = optimize_visual_patches(scene, feedback, {
            'bounds_report': bounds, 'scene_fingerprint': scene_fingerprint(scene)})
        _save(output / 'candidate_evaluation.json', evaluation)
        _save(output / 'optimized_patches.json', optimized)
        report['patch_counts'] = evaluation.get('counts', {})
        stage = 'joint_validation'
        candidate, validation = apply_visual_patch(scene, optimized, bounds_report=bounds)
        _save(output / 'patch_validation.json', validation)
        report['accepted_operations'] = len(optimized['operations'])
        if not optimized['operations'] or not validation.get('applied'):
            report['status'] = 'no_change'
            return report
        _mark_local_origin(candidate, validation, optimized)
        _save(output / 'patch_validation.json', validation)
        _save(output / 'scene_after.json', candidate)
        report['content_invariants'] = dict(validation.get('invariants', {}),
            major_geometry_unchanged=[n for n in scene['nodes'] if n['type'] != 'text'] ==
                                     [n for n in candidate['nodes'] if n['type'] != 'text'],
            edges_exactly_unchanged=scene['edges'] == candidate['edges'])
        report['layout_before'] = measure_layout(scene, bounds)
        fragment = generate_tikz(candidate)
        (output / 'scene_after.tikz').write_text(fragment, encoding='utf-8')
        tex = output / 'scene_after.tex'
        tex.write_text(wrap_document(fragment), encoding='utf-8')
        stage = 'compile'
        report['compile_count'] = 1
        compiled = compile_bounded(tex, xelatex=args.xelatex, timeout=args.compile_timeout,
                                   memory_mb=args.memory_limit_mb)
        report['compile'] = asdict(compiled)
        (output / 'compile.log').write_bytes(Path(compiled.log_path).read_bytes())
        for operation in validation.get('operations', []):
            operation['compiled'] = compiled.success
        _save(output / 'patch_validation.json', validation)
        if not compiled.success:
            report['status'] = 'compile_failed'
            return report
        stage = 'evaluate'
        png = render_pdf(compiled.pdf_path, output / 'scene_after.png', dpi=150)
        _before_after(args.rendered, png, output / 'before_after.png')
        make_comparison(args.reference, png, output / 'reference_comparison.png')
        after_bounds = measure_text_bounds(candidate, compiled.pdf_path)
        _save(output / 'text_bounds_after.json', after_bounds)
        report['layout_after'] = measure_layout(candidate, after_bounds)
        report.update(status='compiled_needs_review', after_png=str(png),
                      after_pdf=str(compiled.pdf_path), before_after=str(output / 'before_after.png'),
                      reference_comparison=str(output / 'reference_comparison.png'))
        return report
    except Exception as exc:
        report.update(status='patch_rejected' if stage == 'joint_validation' else 'experiment_failed',
                      stage=stage, error=str(exc))
        if stage == 'joint_validation':
            _save(output / 'patch_validation.json', getattr(exc, 'report', {
                'accepted': False, 'applied': False, 'atomic_rollback': True, 'reason': str(exc)}))
        return report
    finally:
        report['original_inputs_unchanged'] = all(Path(p).is_file() and _sha(p) == digest
                                                for p, digest in hashes.items())
        _save(output / 'optimization_report.json', report)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (('scene', BASELINE / 'scene_after.json'),
                          ('feedback', FEEDBACK_RUN / 'visual_feedback_raw.txt'),
                          ('bounds', FEEDBACK_RUN / 'text_bounds_before.json'),
                          ('reference', ROOT / 'inputs/test.png'),
                          ('rendered', BASELINE / 'scene_after.png'), ('output-root', ROOT / 'outputs')):
        parser.add_argument('--' + name, type=Path, default=default)
    parser.add_argument('--compile-timeout', type=float, default=60)
    parser.add_argument('--memory-limit-mb', type=int, default=512)
    parser.add_argument('--xelatex', default='xelatex')
    return parser


def main(argv=None):
    parser = build_parser()
    try:
        report = run_experiment(parser.parse_args(argv))
    except (OSError, ValueError) as exc:
        parser.exit(2, str(exc) + '\n')
    print('Status:', report['status'])
    print('Output:', report['output_dir'])
    print('API called:', report['api_called'])
    print('Compile attempts:', report['compile_count'])
    if report.get('error'):
        print('Error:', report['error'])
    return 0 if report['status'] in ('no_change', 'compiled_needs_review') else 1


if __name__ == '__main__':
    raise SystemExit(main())
