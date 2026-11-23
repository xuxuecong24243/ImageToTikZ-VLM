"""Measure existing PDF text, with verified coordinates and conservative ID matching.

PDF bounds are font-metric boxes, not proof of visible ink occlusion. We require
multiple known vector rectangles to calibrate the standalone page translation.
Label attribution requires exact supported text, position AND renderer order.
Unsupported formula encodings / ambiguous associations remain unresolved.
"""
from __future__ import annotations

import math
import re
from statistics import median

from structured_tikz.validator import resolve_style, validate_scene

TEX_PT_TO_PDF_PT = 72 / 72.27


def _valid_box(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
            and box[2] > 0 and box[3] > 0)


def infer_pdf_mapping(scene, drawings):
    """Calibrate x_pdf=scale*x_scene+tx, y_pdf=scale*y_scene+ty.

Only drawn axis-aligned rectangle paths with the expected dimensions count.
Repeated dimensions require a single common translation across >=3 nodes.
"""
    scale = scene['canvas'].get('scale', .4) * TEX_PT_TO_PDF_PT
    rectangles = []
    for drawing in drawings:
        for item in drawing.get('items', []):
            if item[0] == 're':
                rect = list(item[1])
                if len(rect) == 4:
                    rectangles.append(rect)
    pairs = []
    for node in scene['nodes']:
        if node['type'] not in ('rectangle', 'matrix'):
            continue
        x, y, w, h = node['bbox']
        for index, rect in enumerate(rectangles):
            if abs(rect[2]-rect[0]-w*scale) <= .04 and abs(rect[3]-rect[1]-h*scale) <= .04:
                pairs.append({'id': node['id'], 'rectangle_index': index,
                    'tx': rect[0]-x*scale, 'ty': rect[1]-y*scale,
                    'scene_bbox': node['bbox'], 'pdf_bbox': rect})
    groups = []
    for seed in pairs:
        group = [p for p in pairs if abs(p['tx']-seed['tx']) < .08 and abs(p['ty']-seed['ty']) < .08]
        if len({p['id'] for p in group}) >= 3 and len({p['rectangle_index'] for p in group}) >= 3:
            groups.append(group)
    failure = {'status': 'unresolved', 'reason': 'Need one consistent translation supported by >=3 vector rectangles.',
               'nominal_scale': scale, 'matched_rectangles': []}
    if not groups:
        return failure
    best = max(groups, key=lambda g: len({p['id'] for p in g}))
    tx, ty = median(p['tx'] for p in best), median(p['ty'] for p in best)
    if any(len({p['id'] for p in g}) == len({p['id'] for p in best})
           and (abs(median(p['tx'] for p in g)-tx) > .08 or abs(median(p['ty'] for p in g)-ty) > .08)
           for g in groups):
        return dict(failure, reason='Ambiguous vector-to-Scene coordinate calibration.')
    return {'status': 'verified', 'scale': scale, 'tx': tx, 'ty': ty,
            'formula': 'PDF top-left points = Scene top-left pixels * scale + (tx, ty)',
            'scale_basis': 'canvas.scale TeX pt/pixel times 72/72.27 PDF pt/TeX pt',
            'translation_basis': 'consensus of existing vector rectangle corners, not assumed page margins',
            'matched_rectangles': best,
            'max_corner_residual_pdf_pt': max(abs(p[k]-v) for p in best for k, v in [('tx', tx), ('ty', ty)])}


def pdf_bbox_to_scene(bbox, mapping):
    if mapping.get('status') != 'verified':
        raise ValueError('A verified PDF-to-Scene mapping is required.')
    if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox) or bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        raise ValueError('PDF bbox must be finite [x0,y0,x1,y1].')
    s = mapping['scale']
    if not math.isfinite(s) or s <= 0:
        raise ValueError('Mapping scale must be positive and finite.')
    return [(bbox[0]-mapping['tx'])/s, (bbox[1]-mapping['ty'])/s,
            (bbox[2]-bbox[0])/s, (bbox[3]-bbox[1])/s]


def _compact_text(text):
    return re.sub(r'\s+', '', text)


_SYMBOLS = {'in': '\u2208', 'times': '\u00d7', 'cdot': '\u00b7',
            'alpha': '\u03b1', 'beta': '\u03b2', 'gamma': '\u03b3', 'delta': '\u03b4',
            'theta': '\u03b8', 'sigma': '\u03c3'}
_STYLES = {'mathbb', 'mathrm', 'mathbf', 'mathit', 'text', 'operatorname', 'left', 'right'}


def _math_text(text):
    """Limited matching key, NEVER an expression rewrite or correctness check."""
    unsupported = []
    def command(match):
        name = match.group(1)
        if name in _STYLES or name in (' ', ',', ';', ':', '!'):
            return ''
        if name in _SYMBOLS:
            return _SYMBOLS[name]
        unsupported.append(name)
        return ''
    value = re.sub(r'\\([A-Za-z]+|.)', command, text)
    if unsupported:
        return None
    return _compact_text(re.sub(r'[{}_^]', '', value))


def _content(item):
    mode = item.get('mode', 'text')
    if mode == 'math':
        return _math_text(item['text'])
    if mode == 'rich':
        parts = [_math_text(p['content']) if p['type'] == 'math' else _compact_text(p['content'])
                 for p in item['segments']]
        return None if None in parts else ''.join(parts)
    return _compact_text(item['text'])


def _entries(scene):
    entries = []
    for collection in ('nodes', 'labels'):
        for item in scene[collection]:
            if not item.get('text') and item.get('mode') != 'rich':
                continue
            if collection == 'nodes':
                x, y, w, h = item['bbox']
                position = {'x': x+w/2, 'y': y+h/2, 'anchor': 'center'}
                expected = list(item['bbox'])
            else:
                position = {k: item.get(k, 'center') for k in ('x', 'y', 'anchor')}
                expected = item.get('metadata', {}).get('bbox')
            entries.append((item, {'id': item['id'], 'collection': collection,
                'status': 'unresolved', 'bbox_scene': None, 'line_boxes_scene': [], 'line_count': 0,
                'original_position': position, 'expected_bbox': expected if _valid_box(expected) else None,
                'method': 'pymupdf_font_metric_line_bounds', 'needs_manual_review': True}))
    return entries


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _position_matches(item, record, box, scene):
    x, y, w, h = box
    anchor = record['original_position']
    px = x if 'west' in anchor['anchor'] else x+w if 'east' in anchor['anchor'] else x+w/2
    py = y if 'north' in anchor['anchor'] else y+h if 'south' in anchor['anchor'] else y+h/2
    expected_x, expected_y = anchor['x'], anchor['y']
    if record['collection'] == 'nodes' and item['type'] == 'text' and item.get('wrap', True):
        # Text-width paragraphs align glyphs within a fixed-width, centered node.
        if item.get('align', 'center') == 'left':
            px, expected_x = x, item['bbox'][0]
        elif item.get('align', 'center') == 'right':
            px, expected_x = x+w, item['bbox'][0]+item['bbox'][2]
    tolerance = max(4, resolve_style(scene, item)['font_size']*.6)
    return abs(px-expected_x) <= tolerance and abs(py-expected_y) <= tolerance


def associate_text_lines(scene, lines, mapping):
    """Match contiguous PDF lines using supported contents + position + draw order."""
    entries = _entries(scene)
    if mapping.get('status') != 'verified':
        for _, record in entries:
            record['reason'] = 'unverified_coordinate_mapping'
        return [r for _, r in entries]
    used, previous_end = set(), -1
    for item, record in entries:
        expected = _content(item)
        if not expected:
            record['reason'] = 'unsupported_or_empty_text_matching_key'
            continue
        candidates = []
        for start in range(previous_end+1, len(lines)):
            combined = ''
            for end in range(start, min(start+16, len(lines))):
                combined += _compact_text(lines[end]['text'])
                if not expected.startswith(combined):
                    break
                if combined == expected:
                    pdf_box = _union([line['bbox'] for line in lines[start:end+1]])
                    box = pdf_bbox_to_scene(pdf_box, mapping)
                    if not used.intersection(range(start, end+1)) and _position_matches(item, record, box, scene):
                        candidates.append((start, end, pdf_box, box))
                    break
        if len(candidates) != 1:
            record['reason'] = 'ambiguous_label_association' if candidates else 'no_content_position_order_match'
            continue
        start, end, pdf_box, box = candidates[0]
        used.update(range(start, end+1))
        previous_end = end
        record.update(status='measured', bbox_scene=box, bbox_pdf=pdf_box,
            line_boxes_scene=[pdf_bbox_to_scene(line['bbox'], mapping) for line in lines[start:end+1]],
            line_count=end-start+1, pdf_line_indices=list(range(start, end+1)),
            pdf_text=[line['text'] for line in lines[start:end+1]], needs_manual_review=False,
            association='unique_exact_supported_content_plus_position_plus_renderer_order',
            mapping_basis='verified vector rectangle consensus')
    return [r for _, r in entries]


def measure_text_bounds(scene, pdf_path):
    """Read only; no compilation, image recognition, API call or Scene mutation."""
    validate_scene(scene)
    import fitz
    with fitz.open(pdf_path) as doc:
        if doc.page_count != 1 or doc.needs_pass:
            raise ValueError('Measurement requires one unencrypted PDF page.')
        page = doc[0]
        mapping = infer_pdf_mapping(scene, page.get_drawings())
        lines = []
        for block in page.get_text('dict', sort=False)['blocks']:
            for line in block.get('lines', []):
                if tuple(line['dir']) != (1.0, 0.0):
                    continue
                lines.append({'text': ''.join(span['text'] for span in line['spans']),
                              'bbox': list(line['bbox']), 'span_count': len(line['spans'])})
        labels = associate_text_lines(scene, lines, mapping)
        return {'pdf': str(pdf_path), 'page_bbox_pdf': list(page.rect), 'mapping': mapping,
                'labels': labels, 'pdf_line_count': len(lines),
                'counts': {'measured': sum(r['status'] == 'measured' for r in labels),
                           'unresolved': sum(r['status'] != 'measured' for r in labels)},
                'limitations': ['PDF font-metric boxes include whitespace, not only visible ink.',
                    'Only supported exact text encodings are associated; unsupported/ambiguous labels stay unresolved.',
                    'Measured bounds do not prove mathematical or semantic correctness.']}
