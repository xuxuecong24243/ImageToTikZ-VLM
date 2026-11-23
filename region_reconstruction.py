"""One offline region/crop -> explicitly synthetic Mock -> global Scene run.

No API client, prompt editing, learned model, or retry path. Existing geometry
may be supplied only when its source-image digest and dimensions match.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

from auto_compile import compile_bounded
from image_compare import make_comparison, render_pdf
from latex_template import wrap_document
from region_mock_data import build_mock_region_scenes
from region_scene import validate_manifest, validate_region_scene
from region_scene_merger import merge_region_scenes
from region_segmenter import segment_regions, save_region_crops
from structured_tikz.generator import generate_tikz, _unique_keys
from structured_tikz.validator import validate_scene


def _read(path):
    def nonfinite(value):
        raise ValueError('Non-finite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=_unique_keys,
                      parse_constant=nonfinite)


def _save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False, default=str) + '\n', encoding='utf-8')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_experiment(args):
    image = Path(args.image).resolve()
    source_hash = _sha(image)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = Path(args.output_root).resolve() / ('region_reconstruction_' + stamp + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'preparing', 'mode': 'offline_mock_region_reconstruction',
              'output_dir': str(output), 'api_called': False, 'real_region_recognition_performed': False,
              'mock_merge_count': 0, 'compile_count': 0, 'source_image': str(image),
              'source_image_sha256': source_hash, 'visual_review': 'not_performed'}
    stage = 'segmentation'
    try:
        geometry = _read(args.geometry) if args.geometry else None
        config = _read(args.region_config) if args.region_config else None
        if args.geometry:
            report['geometry_source'] = {'path': str(Path(args.geometry).resolve()), 'sha256': _sha(args.geometry),
                                         'kind': 'existing_image_bound_opencv_evidence'}
        regions = segment_regions(image, geometry=geometry, region_config=config, overlap_px=args.overlap_px)
        manifest = save_region_crops(image, regions, output)
        validate_manifest(manifest)
        report.update(region_count=len(regions), segmentation_automatic=manifest.get('automatic'),
                      segmentation_status=manifest.get('segmentation_status'))
        shutil.copy2(image, output / ('original' + image.suffix.lower()))
        stage = 'mock_local_scenes'
        bundles = build_mock_region_scenes(manifest)
        local = output / 'local_scenes'
        local.mkdir(exist_ok=False)
        by_region = {r['region_id']: r for r in manifest['regions']}
        for bundle in bundles:
            validate_region_scene(bundle, by_region[bundle['region_id']])
            _save(local / (bundle['region_id'] + '.json'), bundle)
        stage = 'merge'
        report['mock_merge_count'] = 1
        merged, merge_report = merge_region_scenes(manifest, bundles)
        _save(output / 'merged_scene.json', merged)
        _save(output / 'merge_report.json', merge_report)
        validate_scene(merged)
        _save(output / 'validation.json', {'schema_valid': True, 'errors': [],
                                         'semantic_recognition_verified': False})
        report['merged_counts'] = {k: len(merged[k]) for k in ('nodes', 'labels', 'edges')}
        report['schema_valid'] = True
        stage = 'generate'
        fragment = generate_tikz(merged)
        (output / 'scene.tikz').write_text(fragment, encoding='utf-8')
        tex = output / 'scene.tex'
        tex.write_text(wrap_document(fragment), encoding='utf-8')
        stage = 'compile'
        report['compile_count'] = 1
        _save(output / 'run.json', report)
        compiled = compile_bounded(tex, xelatex=args.xelatex, timeout=args.compile_timeout,
                                   memory_mb=args.memory_limit_mb)
        report['compile'] = asdict(compiled)
        shutil.copy2(compiled.log_path, output / 'compile.log')
        if not compiled.success:
            report['status'] = 'compile_failed'
            return report
        stage = 'render'
        png = render_pdf(compiled.pdf_path, output / 'scene.png', dpi=150)
        comparison = make_comparison(image, png, output / 'mock_comparison.png')
        report.update(status='mock_compiled_needs_review', pdf=str(compiled.pdf_path),
                      png=str(png), comparison=str(comparison),
                      limitations=['Mock data verifies mechanics, never VLM recognition or original-image fidelity.',
                                   'Uncertain cross-region references remain pending, not inferred from proximity.',
                                   'Panel-based heuristic segmentation is not guaranteed for arbitrary layouts.'])
    except Exception as exc:
        report.update(status='failed', stage=stage, error=str(exc))
    finally:
        report['source_image_unchanged'] = image.is_file() and _sha(image) == source_hash
        _save(output / 'run.json', report)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--geometry', type=Path, help='Optional saved geometry with matching image SHA-256.')
    parser.add_argument('--region-config', type=Path, help='Optional explicit manual region list; never labelled automatic.')
    parser.add_argument('--overlap-px', type=int, default=12)
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    parser.add_argument('--xelatex', default='xelatex')
    parser.add_argument('--compile-timeout', type=float, default=60)
    parser.add_argument('--memory-limit-mb', type=int, default=512)
    return parser


def main(argv=None):
    report = run_experiment(build_parser().parse_args(argv))
    print('Status:', report['status'])
    print('Output:', report['output_dir'])
    print('API called:', report['api_called'])
    print('Compile attempts:', report['compile_count'])
    if report.get('error'):
        print('Error:', report['error'])
    return 0 if report['status'] == 'mock_compiled_needs_review' else 1


if __name__ == '__main__':
    raise SystemExit(main())
