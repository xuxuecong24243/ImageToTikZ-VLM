"""Offline scene JSON -> TikZ -> existing XeLaTeX and PNG helpers.

Run: python -m structured_tikz.generator --input examples/gpt_embedding_scene.json
     --image inputs/test.png
Generation has no random values, timestamps, image recognition or network calls.
Only the separate output-directory name uses the current time to prevent overwrite.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from .schema import COLORS
from .validator import SceneValidationError, validate_scene, resolve_style
from . import primitives


def _layered_commands(scene):
    """Explicit compositing without changing supplied geometry or edge topology.

    z_index is computed by the caller's spatial analysis. Separate painting passes
    keep child fills from erasing parent borders, with edges/text above all fills.
    Naming the fill anchors separately ensures edge anchors are defined exactly
    once by the border pass, at the same location as the legacy primitive.
    """
    canvas = scene['canvas']
    nodes = sorted(scene['nodes'], key=lambda n: n.get('z_index', 0))
    result = []
    for phase in ('fill', 'stroke'):
        for original in nodes:
            if original['type'] == 'text' and original.get('background_shape') != 'circle':
                if phase == 'stroke':
                    result.append(primitives.anchor_node(original, canvas))
                continue
            style = resolve_style(scene, original)
            style['stroke' if phase == 'fill' else 'fill'] = 'none'
            style.pop('stroke_color' if phase == 'fill' else 'fill_color', None)
            node = dict(original, style=style)
            if phase == 'fill':
                # Hyphens cannot occur in source IDs, so this cannot collide
                # with a supplied node named e.g. fill_A.
                node['id'] = 'paint-fill-' + original['id']
            result.append('% ' + phase + ': ' + original['id'])
            if node.get('background_shape') == 'circle':
                x,y,w,h = node['bbox']
                item = dict(node, x=x+w/2, y=y+h/2, anchor='center', align='center')
                # Keep native circle dimensions including text growth while
                # deferring its visible glyphs until after all backgrounds.
                result.append(primitives.label(item, scene, name_prefix='n-',
                    circle_diameter=w, _text_opacity=0))
            else:
                result.extend(primitives.shape(node, scene))
    for item in sorted(scene['edges'], key=lambda n: n.get('z_index', 0)):
        result.extend(['% edge: ' + item['id'], primitives.edge(item, scene)])
    result.append('% Text above all backgrounds, borders and connectors.')
    for node in nodes:
        if not node.get('text') and node.get('mode') != 'rich':
            continue
        result.append('% text: ' + node['id'])
        if node['type'] == 'diamond':
            result.append(primitives.diamond_text(node, scene))
            if canvas.get('measure_text',False):
                result.extend(primitives.text_measurement_commands('t-'+node['id'],node['id']))
            continue
        x,y,w,h = node['bbox']
        item = {k:node[k] for k in ('id','text','segments','style','mode','font_size','align') if k in node}
        item.update(x=x+w/2,y=y+h/2,anchor='center')
        if node.get('background_shape') == 'circle':
            item['align'] = 'center'
            item['style'] = dict(resolve_style(scene,node),fill='none',stroke='none')
            item['style'].pop('fill_color',None)
            item['style'].pop('stroke_color',None)
            result.append(primitives.label(item,scene,name_prefix='t-',circle_diameter=w))
        else:
            result.append(primitives.label(item,scene,name_prefix='t-',
                text_width=w if node['type']=='text' and node.get('wrap',True) else None))
        if canvas.get('measure_text',False):
            result.extend(primitives.text_measurement_commands('t-'+node['id'],node['id']))
    for item in sorted(scene['labels'], key=lambda n: n.get('z_index', 0)):
        result.append(primitives.label(item, scene))
        if canvas.get('measure_text',False):
            result.extend(primitives.text_measurement_commands('l-'+item['id'],item['id']))
    return result


def generate_tikz(scene: dict) -> str:
    """Draw shapes, then edges, then all labels, preserving each array's order."""
    validate_scene(scene)
    canvas = scene['canvas']
    output = ['% Deterministic scene renderer; semantic objects and coordinates are supplied by the caller.']
    for name, value in sorted(COLORS.items()):
        output.append(r'\definecolor{%s}{HTML}{%s}' % (primitives.color(name), value))
    # Safe literal RGB values only: names are derived from validated hex digits.
    exact_colors = set()
    for collection in ('nodes', 'edges', 'labels'):
        for item in scene[collection]:
            style = resolve_style(scene,item)
            exact_colors.update(style[key].upper() for key in ('fill_color','stroke_color','text_color_rgb') if key in style)
    for value in sorted(exact_colors):
        # Definitions live in the document body. Suppress their newline glue
        # so many measured colors cannot shift the following TikZ hbox right.
        # Leave the six legacy palette lines byte-identical.
        output.append((r'\definecolor{%s}{HTML}{%s}' % (primitives.color(value), value[1:])) + '%')
    output.extend([r'\begin{tikzpicture}[>=Latex]', r'\useasboundingbox %s rectangle %s;' % (
        primitives.point(0, canvas['height'], canvas), primitives.point(canvas['width'], 0, canvas))])
    if canvas.get('layered', False):
        output.extend(_layered_commands(scene))
        output.append(r'\end{tikzpicture}')
        fragment = '\n'.join(output) + '\n'
        from latex_template import validate_tikz
        validate_tikz(fragment)
        return fragment
    node_labels = []
    for node in scene['nodes']:
        output.append('% node: ' + node['id'] + ' (' + node['type'] + ')')
        output.extend(primitives.shape(node, scene))
        if canvas.get('measure_text',False) and node.get('background_shape')=='circle' and (node.get('text') or node.get('mode')=='rich'):
            output.extend(primitives.text_measurement_commands('n-'+node['id'],node['id']))
        if (node.get('text') or node.get('mode') == 'rich') and node.get('background_shape') != 'circle':
            if node['type'] == 'diamond':
                node_labels.append(primitives.diamond_text(node, scene))
                if canvas.get('measure_text',False):
                    node_labels.extend(primitives.text_measurement_commands('t-'+node['id'],node['id']))
                continue
            x, y, w, h = node['bbox']
            item = {k: node[k] for k in ('id', 'text', 'segments', 'style', 'mode', 'font_size', 'align') if k in node}
            item.update(x=x + w / 2, y=y + h / 2, anchor='center')
            node_labels.append(primitives.label(item, scene, name_prefix='t-',
                                                text_width=w if node['type'] == 'text' and node.get('wrap', True) else None))
            if canvas.get('measure_text',False):
                node_labels.extend(primitives.text_measurement_commands('t-'+node['id'],node['id']))
    for item in scene['edges']:
        output.extend(['% edge: ' + item['id'], primitives.edge(item, scene)])
    output.append('% Labels are independent editable nodes, drawn after all filled shapes and edges.')
    output.extend(node_labels)
    for item in scene['labels']:
        output.append(primitives.label(item, scene))
        if canvas.get('measure_text',False):
            output.extend(primitives.text_measurement_commands('l-'+item['id'],item['id']))
    output.append(r'\end{tikzpicture}')
    fragment = '\n'.join(output) + '\n'
    from latex_template import validate_tikz
    validate_tikz(fragment)
    return fragment


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SceneValidationError('Duplicate JSON field: ' + key)
        result[key] = value
    return result


def load_scene(path):
    raw = Path(path).read_bytes()
    if len(raw) > 1_000_000:
        raise SceneValidationError('Scene JSON exceeds the 1 MB experiment limit.')
    scene = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_unique_keys)
    validate_scene(scene)
    return scene, raw


def run_scene(input_path, output_root, *, image_path=None, xelatex='xelatex', timeout=60):
    from latex_template import wrap_document
    from latex_compiler import compile_tex
    from image_compare import render_pdf, make_comparison
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Timeout must be finite and positive.')
    scene, raw = load_scene(input_path)
    image = Path(image_path).resolve() if image_path else None
    if image is not None and not image.is_file():
        raise FileNotFoundError('Comparison image does not exist: ' + str(image))
    fragment = generate_tikz(scene)
    document = wrap_document(fragment)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = Path(output_root).resolve() / ('structured_' + stamp)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'scene.json').write_bytes(raw)
    (output / 'scene.tikz').write_text(fragment, encoding='utf-8')
    tex = output / 'scene.tex'
    tex.write_text(document, encoding='utf-8')
    report = {'kind': 'deterministic_structured_tikz', 'api_called': False,
              'status': 'generated', 'output_dir': str(output),
              'input_sha256': hashlib.sha256(raw).hexdigest(),
              'tikz_sha256': hashlib.sha256(fragment.encode('utf-8')).hexdigest(),
              'annotation_metadata': scene.get('metadata', {}),
              'draw_order': ['nodes', 'edges', 'node_labels', 'labels'],
              'counts': {key: len(scene[key]) for key in ('nodes', 'edges', 'labels')},
              'limitations': ['No automatic image recognition or overlap avoidance.',
                              'Math command allowlist and existing TeX checks are not a security sandbox.',
                              'Font metrics and supplied coordinates determine text spacing.']}
    try:
        result = compile_tex(tex, xelatex=xelatex, timeout=timeout)
        report['compile'] = asdict(result)
        report['status'] = 'compile_failed'
        if result.success:
            report['status'] = 'compiled'
            png = render_pdf(result.pdf_path, output / 'scene.png', dpi=200)
            report['png'] = str(png)
            if image is not None:
                original = output / ('original' + image.suffix.lower())
                original.write_bytes(image.read_bytes())
                report['original_sha256'] = hashlib.sha256(original.read_bytes()).hexdigest()
                report['comparison'] = str(make_comparison(original, png, output / 'comparison.png'))
            report['status'] = 'success'
    except Exception as exc:
        report.update(status='output_failed', error=str(exc))
    (output / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Manually supplied scene JSON')
    parser.add_argument('--image', type=Path, help='Optional original image for visual comparison')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'outputs')
    parser.add_argument('--xelatex', default='xelatex')
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args(argv)
    try:
        report = run_scene(args.input, args.output_dir, image_path=args.image,
                           xelatex=args.xelatex, timeout=args.timeout)
    except (OSError, ValueError) as exc:
        parser.exit(2, 'Scene error: ' + str(exc) + '\n')
    print('Status:', report['status'])
    print('Output:', report['output_dir'])
    if report.get('compile'):
        print('XeLaTeX exit:', report['compile']['returncode'])
        if report['compile']['errors']:
            print(report['compile']['errors'])
    if report.get('error'):
        print(report['error'])
    return 0 if report['status'] == 'success' else 1


if __name__ == '__main__':
    raise SystemExit(main())
