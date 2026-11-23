"""Image -> model Scene -> deterministic TikZ -> bounded compile -> feedback.

Default: one authorized request, <=2 local refinement rounds, no model retries.
--dry-run prepares geometry/prompt only. --offline-scene checks supplied data only;
it is never reported as automatic image recognition. Existing outputs stay intact.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from auto_compile import compile_bounded
from image_compare import make_comparison, render_pdf
from latex_template import wrap_document
from main import new_run_directory, prepare_geometry, snapshot_image
from structured_tikz.generator import generate_tikz
from vlm_client import APIConfig, VLMClient, validate_image_path, _redact


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')


class RequestBudgetError(ValueError):
    pass


class RequestLedger:
    """Reserve output budget BEFORE each attempt; no refund/retry after failure.

    Reported total usage includes input, but is only known after a provider reply.
    Therefore this is an output-reservation cap plus a stop-on-usage cap, NOT a
    guaranteed dollar-cost cap. Missing usage prevents any additional requests.
    """
    def __init__(self, client, args, directory):
        self.client, self.args, self.directory = client, args, Path(directory)
        self.entries = []
        self.reserved_output = 0
        self.reported_usage = 0
        self.usage_known = True

    def request(self, images, prompt, phase):
        if not self.args.allow_api:
            raise RequestBudgetError('API is not authorized.')
        if phase == 'complete' and not (self.args.allow_completion and self.args.stage_strategy == 'adaptive'):
            raise RequestBudgetError('Additional completion is not separately authorized.')
        if phase == 'visual' and not self.args.allow_visual_refinement:
            raise RequestBudgetError('Visual feedback is not separately authorized.')
        if phase not in {'generate', 'complete', 'visual'}:
            raise RequestBudgetError('Unknown request phase.')
        if phase == 'generate' and self.entries:
            raise RequestBudgetError('Only one initial Scene request is permitted.')
        if len(self.entries) >= self.args.max_api_requests:
            raise RequestBudgetError('Maximum API request count reached.')
        tokens = self.client.max_tokens
        if self.reserved_output + tokens > self.args.max_output_budget:
            raise RequestBudgetError('Cumulative requested output-token budget would be exceeded.')
        if self.entries and (not self.usage_known or self.reported_usage >= self.args.max_total_usage_tokens):
            raise RequestBudgetError('Reported usage cap reached or prior usage unavailable; no further requests.')
        self.reserved_output += tokens
        directory = self.directory/('request_%02d' % (len(self.entries)+1))
        directory.mkdir(exist_ok=False)
        key = getattr(getattr(self.client, 'config', None), 'api_key', '')
        (directory/'prompt.txt').write_text(_redact(prompt, key), encoding='utf-8')
        entry = {'phase': phase, 'status': 'attempted', 'reserved_output_tokens': tokens,
                 'image_count': len(images), 'directory': str(directory)}
        self.entries.append(entry)
        if isinstance(self.client, VLMClient):
            self.client.response_directory = directory
        try:
            from auto_scene_generator import complete_scene_images
            raw = complete_scene_images(self.client, images, prompt)
            (directory/'response.txt').write_text(_redact(raw, key), encoding='utf-8')
            entry['status'] = 'response_received_not_yet_validated'
            return raw
        except Exception as exc:
            entry.update(status='request_failed', error=_redact(str(exc), key))
            raise
        finally:
            diagnostics = getattr(self.client, 'last_diagnostics', {})
            diagnostics = json.loads(_redact(json.dumps(diagnostics, default=str), key))
            entry['diagnostics'] = diagnostics
            usage = diagnostics.get('usage', {})
            total = usage.get('total_tokens')
            if total is None and all(isinstance(usage.get(k), int) for k in ('prompt_tokens', 'completion_tokens')):
                total = usage['prompt_tokens']+usage['completion_tokens']
            if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
                self.reported_usage += total
            else:
                self.usage_known = False
            try:
                save_json(directory/'request.json', entry)
            except Exception as exc:
                # Logging failures must not mask an active API/response exception.
                entry['diagnostic_write_error'] = type(exc).__name__

    def mark(self, status):
        self.entries[-1]['status'] = status
        save_json(Path(self.entries[-1]['directory'])/'request.json', self.entries[-1])

    def summary(self):
        return {'requests': self.entries, 'request_count': len(self.entries),
                'reserved_output_tokens': self.reserved_output,
                'reported_total_usage_tokens': self.reported_usage,
                'all_usage_reported': self.usage_known,
                'cost_in_currency': None,
                'limitation': 'Input/image tokens are provider-dependent. Usage cap is checked after replies; no currency estimate.'}


def _validate_options(args):
    for key, low, high in [('max_api_requests', 1, 3), ('max_local_rounds', 0, 2),
                           ('max_iterations', 1, 4), ('memory_limit_mb', 256, 2048),
                           ('max_output_budget', 1, 1_000_000), ('max_total_usage_tokens', 1, 2_000_000)]:
        value = getattr(args, key)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(key+' is outside the supported bounded range.')
    for key, upper in [('compile_timeout', 120), ('api_timeout', 600)]:
        value = getattr(args, key)
        if not math.isfinite(value) or not 1 <= value <= upper:
            raise ValueError(key+' must be finite and between 1 and '+str(upper)+'.')
    if args.offline_scene and (args.allow_api or args.allow_completion or args.allow_visual_refinement):
        raise ValueError('Supplied offline Scene mode cannot make any API calls.')
    if not args.dry_run and not args.offline_scene and not args.allow_api:
        raise ValueError('Use --allow-api for generation, or --dry-run for zero-request preparation.')
    if args.allow_completion and args.stage_strategy != 'adaptive':
        raise ValueError('--allow-completion requires --stage-strategy adaptive.')


def _reference_evaluation(reference_path, scene, reports, output):
    """Separate, optional evaluation AFTER iterations; never sent to a model."""
    from structured_tikz.generator import load_scene
    from semantic_enrichment_evaluation import evaluate_enrichment
    reference, _ = load_scene(reference_path)
    evaluation = evaluate_enrichment(scene, scene, {}, {}, reference)
    evaluation.update(reference_used_for_generation=False,
                      note='Model object coordinates are estimates, not OpenCV measurements. Before/after are the same final Scene.',
                      recognition_accuracy={'chinese': 'unverified', 'mathematical_equivalence': 'unverified',
                                            'connection_topology': 'unverified'},
                      compilation_success_count=sum(bool(r.get('compile', {}).get('success')) for r in reports),
                      compilation_attempt_count=sum('compile' in r for r in reports),
                      human_intervention_required='quality_unverified')
    save_json(output/'reference_evaluation.json', evaluation)


def run_pipeline(args, *, client=None):
    """Dependency injection is for offline tests; it does not bypass authorization."""
    if getattr(args, 'pipeline_mode', 'single-stage') == 'two-stage':
        from two_stage_pipeline import run_two_stage
        return run_two_stage(args, client=client)
    if getattr(args, 'offline_ir', None) or getattr(args, 'geometry_cache', None) or getattr(args, 'no_compile', False):
        raise ValueError('IR-specific options require --pipeline-mode two-stage.')
    _validate_options(args)
    image = validate_image_path(args.image)
    from PIL import Image
    with Image.open(image) as source:
        image_size = source.size
        # Raw bytes/coordinates must agree; do not silently rotate the VLM image.
        if source.getexif().get(274, 1) != 1:
            raise ValueError('Normalize EXIF rotation before running this coordinate-based experiment.')
        if image_size[0]*image_size[1] > 25_000_000 or max(image_size) > 20000:
            raise ValueError('Image exceeds the supported canvas/pixel limits.')
    output = new_run_directory(args.output_root, prefix='auto')
    original = snapshot_image(image, output)
    report = {'status': 'preparing', 'output_dir': str(output), 'api_called': False,
              'mode': 'dry_run' if args.dry_run else 'offline_supplied_scene' if args.offline_scene else 'image_to_scene',
              'source_image_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
              'iterations': [], 'quality': 'not_verified', 'reference_used_for_generation': False,
              'limits': {key: getattr(args, key) for key in ('max_api_requests', 'max_iterations',
                          'max_local_rounds', 'max_output_budget', 'max_total_usage_tokens',
                          'compile_timeout', 'memory_limit_mb')},
              'limitations': ['Compiled files do not establish content or layout accuracy.',
                              'Static TeX checks and resource caps are not a security sandbox.',
                              'Public compilation requires OS/container isolation.',
                              'Image metrics do not verify Chinese, math or topology.']}
    ledger = None
    scene = None
    try:
        from auto_scene_generator import (build_scene_prompt, build_patch_prompt, _strict_json,
                                          parse_scene_response, parse_patch_response, apply_scene_patch)
        from auto_scene_validator import diagnose_scene
        from auto_layout_refiner import refine_scene
        from auto_visual_feedback import compare_visual
        context = None
        if args.geometry_mode == 'assist':
            report['geometry'], context = prepare_geometry(original, output, preview_only=args.dry_run)
        prompt = build_scene_prompt(image_size, context)
        from structured_tikz.schema import scene_contract
        save_json(output/'schema_contract.json', scene_contract(image_size))
        (output/'scene_prompt.txt').write_text(prompt, encoding='utf-8')
        if args.dry_run:
            report.update(status='prepared_offline', scene_generated=False)
            return report
        if args.offline_scene:
            raw = Path(args.offline_scene).read_text(encoding='utf-8-sig')
            # Saved human data is not a model response. Preserve its provenance
            # and review decisions; the offline mode cannot authorize API calls.
            scene = _strict_json(raw)
            (output/'supplied_scene.json').write_text(raw, encoding='utf-8')
            report['recognition'] = 'not_performed_offline_supplied_scene'
        else:
            if client is None:
                config = APIConfig.from_env()
                client = VLMClient(config, allow_api=True, max_tokens=args.max_tokens,
                                   timeout=args.api_timeout, thinking_mode=args.thinking_mode,
                                   reasoning_effort=args.reasoning_effort)
            ledger = RequestLedger(client, args, output)
            raw = ledger.request([original], prompt, 'generate')
            # Preserve the model object before provenance normalization/recovery.
            save_json(output/'scene_raw.json', _strict_json(raw))
            scene = parse_scene_response(raw)
            ledger.mark('scene_parsed')
        save_json(output/'initial_scene.json', scene)
        if not args.offline_scene:
            from scene_structure import recover_scene_structure
            response_format = getattr(client, 'last_diagnostics', {}).get('requested_response_format')
            scene, recovery = recover_scene_structure(
                scene, image_size, coordinate_system='top-left', response_format=response_format)
            save_json(output/'structural_recovery.json', recovery)
            save_json(output/'scene_after_recovery.json', scene)
            report['structural_recovery'] = recovery
            root_blockers = [e for e in recovery['after']['errors'] if e['code'] in {
                'unknown_root_field', 'missing_root_field', 'invalid_collection_type',
                'root_configuration_invalid', 'canvas_image_mismatch'}]
            if root_blockers:
                report['completion_blocked_reason'] = {
                    'reason':'Existing targeted patches cannot repair these root contract errors; stopped before another API call.',
                    'errors':root_blockers}
            # Recovery never invents content or claims full image reconstruction.
            report['quality'] = 'not_verified'
        used_local = 0
        completion_used = visual_used = False
        pending_changes = []
        seen = set()
        for index in range(args.max_iterations):
            iteration = output/('iteration_%02d' % (index+1))
            iteration.mkdir(exist_ok=False)
            digest = hashlib.sha256(json.dumps(scene, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if digest in seen:
                report['status'] = 'stopped_repeated_scene'
                break
            seen.add(digest)
            save_json(iteration/'scene.json', scene)
            save_json(iteration/'changes.json', pending_changes)
            diagnosis = diagnose_scene(scene, image_size=image_size)
            save_json(iteration/'validation.json', diagnosis)
            from scene_coverage import evaluate_coverage
            coverage = evaluate_coverage(scene)
            save_json(iteration/'coverage.json', coverage)
            entry = {'iteration': index+1, 'directory': str(iteration), 'diagnosis': diagnosis,
                     'changes': pending_changes, 'status': 'diagnosed', 'coverage': coverage}
            report['iterations'].append(entry)
            remaining = index+1 < args.max_iterations
            # Deterministic routing/local corrections never consume API requests.
            if used_local < args.max_local_rounds and remaining:
                candidate, changes = refine_scene(scene)
                if changes:
                    save_json(iteration/'proposed_local_scene.json', candidate)
                    save_json(iteration/'proposed_local_changes.json', changes)
                    scene, pending_changes = candidate, changes
                    used_local += 1
                    entry['status'] = 'local_refinement_applied'
                    continue
            if not diagnosis['schema_valid'] or diagnosis['errors']:
                entry['status'] = report['status'] = 'scene_invalid'
                if report.get('completion_blocked_reason'):
                    break
                if (remaining and ledger and not completion_used and args.allow_completion
                        and args.stage_strategy == 'adaptive'):
                    phase = 'complete'
                else:
                    break
            else:
                try:
                    fragment = generate_tikz(scene)
                    document = wrap_document(fragment)
                    (iteration/'scene.tikz').write_text(fragment, encoding='utf-8')
                    tex = iteration/'scene.tex'
                    tex.write_text(document, encoding='utf-8')
                    result = compile_bounded(tex, xelatex=args.xelatex,
                                             timeout=args.compile_timeout, memory_mb=args.memory_limit_mb)
                    entry['compile'] = asdict(result)
                    save_json(iteration/'compile.json', entry['compile'])
                    if not result.success:
                        # Local evidence is retained; do not hallucinate a generic TeX fix.
                        entry['status'] = report['status'] = 'compile_failed'
                        save_json(iteration/'compile_diagnosis.json', {
                            'errors': result.errors, 'scene_validation': diagnosis,
                            'automatic_fix': 'not_available_for_arbitrary_engine_errors',
                            'action': 'Inspect the first engine error; no unchanged recompilation.'})
                        break
                    png = render_pdf(result.pdf_path, iteration/'scene.png', dpi=150)
                    make_comparison(original, png, iteration/'comparison.png')
                    visual = compare_visual(original, png, scene, iteration)
                    save_json(iteration/'visual_report.json', visual)
                    entry['visual_report'] = str(iteration/'visual_report.json')
                    entry['status'] = 'rendered_unverified'
                    report.update(status='rendered_unverified', final_scene=str(iteration/'scene.json'),
                                  tex=str(tex), pdf=str(result.pdf_path), png=str(png),
                                  comparison=str(iteration/'comparison.png'))
                    if diagnosis['risks'] or scene.get('metadata', {}).get('pending_edges') or coverage['quality_warning']:
                        report['status'] = 'rendered_needs_review'
                    # Optional dual-image pass. Exactly one, separately authorized.
                    if not (remaining and ledger and args.allow_visual_refinement and not visual_used):
                        break
                    phase = 'visual'
                except (ValueError, OSError, RuntimeError) as exc:
                    entry['status'] = report['status'] = 'local_output_failed'
                    entry['error'] = str(exc)
                    break
            if phase == 'complete':
                images, feedback = [original], None
                completion_used = True
            else:
                images, feedback = [original, png], visual
                visual_used = True
            patch_prompt = build_patch_prompt(scene, diagnosis, phase=phase, feedback=feedback)
            try:
                raw_patch = ledger.request(images, patch_prompt, phase)
            except RequestBudgetError as exc:
                report['feedback_stopped_reason'] = str(exc)
                break
            patch = parse_patch_response(raw_patch)
            save_json(iteration/'model_patch.json', patch)
            try:
                candidate, changes = apply_scene_patch(scene, patch)
            except (ValueError, TypeError, KeyError) as exc:
                save_json(iteration/'patch_rejection.json', {'reason': str(exc), 'applied': False})
                ledger.mark('patch_rejected')
                raise
            save_json(iteration/'model_changes.json', changes)
            ledger.mark('patch_validated')
            if not changes:
                report['feedback_stopped_reason'] = 'No patch changes returned.'
                break
            scene, pending_changes = candidate, changes
        else:
            report['iteration_limit_reached'] = True
        report['local_refinement_rounds'] = used_local
        # Evaluation can never supply data to an earlier generation/feedback stage.
        if args.reference and scene is not None:
            try:
                _reference_evaluation(args.reference, scene, report['iterations'], output)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                report['evaluation_unavailable'] = str(exc)
        return report
    except Exception as exc:
        key = getattr(getattr(client, 'config', None), 'api_key', '') if client else ''
        report.update(status='pipeline_failed', error=_redact(str(exc), key))
        diagnostic = getattr(exc, 'diagnostic', None)
        if isinstance(diagnostic, dict):
            report['parse_diagnostic'] = json.loads(_redact(json.dumps(diagnostic), key))
            save_json(output/'parse_diagnostic.json', report['parse_diagnostic'])
        return report
    finally:
        if ledger:
            report['api'] = ledger.summary()
            report['api_called'] = bool(ledger.entries)
        save_json(output/'run.json', report)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--pipeline-mode', choices=('single-stage', 'two-stage'), default='single-stage')
    parser.add_argument('--offline-ir', type=Path, help='Supplied IR; never reported as automatic image recognition.')
    parser.add_argument('--geometry-cache', type=Path, help='Two-stage source-hash-bound geometry cache.')
    parser.add_argument('--no-compile', action='store_true', help='Two-stage validation/TeX only; skip XeLaTeX.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--offline-scene', type=Path, help='Supplied Scene downstream test, NEVER AI recognition.')
    parser.add_argument('--allow-api', action='store_true')
    parser.add_argument('--allow-completion', action='store_true', help='Separate permission for one missing-field patch request.')
    parser.add_argument('--allow-visual-refinement', action='store_true', help='Separate permission for one dual-image patch request.')
    parser.add_argument('--stage-strategy', choices=('single', 'adaptive'), default='single')
    parser.add_argument('--max-api-requests', type=int, default=1)
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--max-output-budget', type=int, default=65536, help='Sum of requested output tokens, reserved before calls.')
    parser.add_argument('--max-total-usage-tokens', type=int, default=200000, help='Stop further calls once reported total usage reaches this limit.')
    parser.add_argument('--thinking-mode', choices=('enabled', 'disabled'))
    parser.add_argument('--reasoning-effort', choices=('low', 'medium', 'high'))
    parser.add_argument('--max-local-rounds', type=int, default=2)
    parser.add_argument('--max-iterations', type=int, default=3)
    parser.add_argument('--geometry-mode', choices=('off', 'assist'), default='assist')
    parser.add_argument('--api-timeout', type=float, default=180)
    parser.add_argument('--compile-timeout', type=float, default=60)
    parser.add_argument('--memory-limit-mb', type=int, default=512)
    parser.add_argument('--xelatex', default='xelatex')
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parent/'outputs')
    parser.add_argument('--reference', type=Path, help='Optional post-run evaluation only; never sent to the model.')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_pipeline(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc)+'\n')
    print('Status:', report['status'])
    print('Output:', report['output_dir'])
    if report.get('error'):
        print('Error:', report['error'])
    return 0 if report['status'] in {'prepared_offline', 'rendered_unverified', 'rendered_needs_review', 'tex_generated_unverified'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
