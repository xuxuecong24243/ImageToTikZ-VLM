"""Offline, one-pass semantic/geometry enrichment. No API, CV or TeX execution."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from semantic_schema import parse_semantic_response
from semantic_scene_merger import enrich_semantic_scene
from structured_tikz.generator import load_scene


def _json(path, value):
    # Exclusive writes: historical files cannot be overwritten, even by accident.
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _read_context(path):
    raw = Path(path).read_bytes()
    if len(raw) > 1_000_000:
        raise ValueError('Context exceeds 1 MB.')

    def unique(pairs):
        data = {}
        for key, value in pairs:
            if key in data:
                raise ValueError('Duplicate context JSON key.')
            data[key] = value
        return data

    def nonfinite(_):
        raise ValueError('Non-finite context values are not allowed.')

    return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=unique, parse_constant=nonfinite)


def _dash(draw, a, b, color, width=2):
    length = math.hypot(b[0]-a[0], b[1]-a[1])
    if length == 0:
        return
    for start in range(0, int(length), 14):
        end = min(start+7, length)
        draw.line([(a[0]+(b[0]-a[0])*t/length, a[1]+(b[1]-a[1])*t/length)
                   for t in (start, end)], fill=color, width=width)


def render_review(image_path, scene, reports, destination):
    """Overlay evidence, not a generated diagram; no recognition takes place."""
    from PIL import Image, ImageDraw, ImageFont
    with Image.open(image_path) as image:
        original = image.convert('RGB')
    header, sidebar = 75, 350
    canvas = Image.new('RGB', (original.width+sidebar, original.height+header), 'white')
    canvas.paste(original, (0, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=12)
    draw.text((12, 8), 'OFFLINE ENRICHMENT REVIEW - ALL proposals remain UNVERIFIED', fill='black', font=font)
    draw.text((12, 30), 'Orange solid: measured bbox | cyan dashed: derived front | green dashed: text association', fill='black', font=font)
    draw.text((12, 51), 'Blue solid: measured route proposal | gray dashed: semantic relation ONLY (not a recovered path)', fill='black', font=font)

    def box(bbox, color, dashed=False):
        x, y, w, h = bbox
        y += header
        points = [(x, y), (x+w, y), (x+w, y+h), (x, y+h), (x, y)]
        for a, b in zip(points, points[1:]):
            if dashed:
                _dash(draw, a, b, color)
            else:
                draw.line([a, b], fill=color, width=2)

    nodes = {node['id']: node for node in scene['nodes']}
    # Draw unlocated semantic suggestions first, as faint dashed lines behind boxes.
    for entry in reports['edges']['entries']:
        if entry['status'] == 'recovered':
            continue
        endpoints = [nodes.get(entry[key]) for key in ('source_id', 'target_id')]
        if all(endpoints):
            points = [(n['bbox'][0]+n['bbox'][2]/2, n['bbox'][1]+n['bbox'][3]/2+header) for n in endpoints]
            _dash(draw, *points, '#aaaaaa', width=1)
    for node in scene['nodes']:
        original_box = node.get('metadata', {}).get('geometry_enrichment', {}).get('original_bbox', node['bbox'])
        box(original_box, '#ef8a00')
        if node['bbox'] != original_box:
            box(node['bbox'], '#009cbe', dashed=True)
        draw.text((original_box[0], original_box[1]+header-15), node['id'], fill='#995500', font=small)
    for entry in reports['text']['entries']:
        if entry['status'] == 'aligned' and entry['bbox']:
            box(entry['bbox'], '#15963c', dashed=True)
            x, y, _, _ = entry['bbox']
            draw.text((x, y+header-13), entry['source_text_id'], fill='#147931', font=small)
    for entry in reports['edges']['entries']:
        if entry['status'] != 'recovered':
            continue
        points = [(p[0], p[1]+header) for p in entry['points']]
        draw.line(points, fill='#1258d6', width=3)
        for x, y in (points[0], points[-1]):
            draw.ellipse((x-4, y-4, x+4, y+4), fill='#1258d6')
    x, y = original.width+14, header+8
    lines = ['NEEDS HUMAN REVIEW', 'Texts not automatically located:']
    lines += [e['source_text_id']+' : '+e['status'] for e in reports['text']['entries']
              if e['status'] in ('unresolved', 'ambiguous')]
    lines += ['', 'Shape proposals / missing parameters:']
    lines += [e['node_id']+' : '+e['status'] for e in reports['shapes']['entries']]
    lines += ['', 'Relations (source -> target):']
    for e in reports['edges']['entries']:
        lines.extend([str(e['source_id'])+' -> '+str(e['target_id']), '  '+e['status']])
    lines += ['', 'Arrow direction: VLM suggestion.', 'No OCR verification. No TikZ rendered.', 'See JSON reports for exact evidence.']
    for line in lines:
        draw.text((x, y), line, fill='#333333', font=small)
        y += 17
    with Path(destination).open('xb') as stream:
        canvas.save(stream, format='PNG')


def run(args):
    scene, _ = load_scene(args.scene)
    semantic = parse_semantic_response(Path(args.semantic).read_text(encoding='utf-8-sig'), scene)
    context = _read_context(args.context)
    from PIL import Image
    with Image.open(args.image) as image:
        if image.size != (scene['canvas']['width'], scene['canvas']['height']):
            raise ValueError('Image size differs from measured Scene canvas.')
    paths = {key: Path(getattr(args, key)) for key in ('scene', 'semantic', 'context', 'image')}
    provenance = {key: {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                  for key, path in paths.items()}
    expected = scene.get('metadata', {}).get('source_image', {}).get('sha256')
    if expected and expected != provenance['image']['sha256']:
        raise ValueError('Source image hash does not match the existing geometry.')
    enriched, reports = enrich_semantic_scene(scene, semantic, context)
    enriched['metadata']['offline_enrichment']['inputs'] = provenance
    output = Path(args.output_root)/('enriched_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    _json(output/'scene_enriched_draft.json', enriched)
    # Validate saved bytes through the existing bounded file loader too.
    load_scene(output/'scene_enriched_draft.json')
    for name, key in [('text_alignment_report', 'text'), ('shape_enrichment_report', 'shapes'),
                      ('edge_routing_report', 'edges')]:
        _json(output/(name+'.json'), reports[key])
    render_review(args.image, enriched, reports, output/'scene_enriched_review.png')
    _json(output/'enrichment_report.json', {
        'status': 'unverified_draft_created', 'inputs': provenance, 'summary': reports['summary'],
        'counts': {key: reports[key]['counts'] for key in ('text', 'shapes', 'edges')},
        'execution': {'api_called': False, 'cv_rerun': False, 'tikz_generated': False,
                      'tex_compiled': False, 'reference_used_for_enrichment': False}})
    # This is intentionally AFTER every automatic enrichment artifact is saved.
    # Reference data can only produce a separate evaluation, never feed proposals.
    if args.reference:
        from semantic_enrichment_evaluation import evaluate_enrichment
        reference, _ = load_scene(args.reference)
        evaluation = evaluate_enrichment(scene, enriched, semantic, reports, reference)
        evaluation['reference_path'] = str(Path(args.reference).resolve())
        evaluation['reference_loaded_after_enrichment_artifacts'] = True
        _json(output/'reference_evaluation.json', evaluation)
    print(json.dumps({'output': str(output), 'summary': reports['summary'],
                      'counts': {key: reports[key]['counts'] for key in ('text', 'shapes', 'edges')}},
                     ensure_ascii=False, indent=2))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('semantic', 'scene', 'context', 'image'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('outputs'))
    parser.add_argument('--reference', type=Path, help='Evaluation only; read AFTER all enrichment artifacts are saved.')
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, 'Offline enrichment failed: '+str(error)+'\n')


if __name__ == '__main__':
    main()
