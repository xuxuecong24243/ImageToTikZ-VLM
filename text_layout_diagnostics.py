"""Diagnostics from mapped PDF text bounds, without claiming glyph collisions.

Measured rectangles are conservative font bounds. Their intersections are layout
risks, not proof that two visible ink strokes overlap. Unresolved labels are not
automatically repaired. This module never changes the Scene.
"""
from __future__ import annotations

from copy import deepcopy
import re

from auto_scene_validator import box_valid, edge_points, metadata, overlap, whole_box, within
from semantic_edge_router import _inside_length
from structured_tikz.validator import resolve_style, validate_scene


def padded(box, amount=3.0):
    x, y, width, height = box
    return [x-amount, y-amount, width+2*amount, height+2*amount]


def text_boxes(record):
    """Prefer measured line envelopes; their union includes interline whitespace."""
    if record.get('status') != 'measured' or not box_valid(record.get('bbox_scene')):
        return []
    lines = record.get('line_boxes_scene')
    if isinstance(lines, list) and lines and all(box_valid(box) for box in lines):
        return [list(box) for box in lines]
    return [list(record['bbox_scene'])]


def path_intersects(points, boxes, minimum=.1):
    return bool(points and any(_inside_length(a, b, box) > minimum
                              for a, b in zip(points, points[1:]) for box in boxes))


def dimension_text(item):
    """Only a known single-line scalar dimension; do not guess math semantics."""
    text = item.get('text', '')
    return (item.get('mode') == 'math' and '\n' not in text
            and re.fullmatch(r'\s*(?:B|L|d)\s*=\s*\d+(?:\.\d+)?\s*', text) is not None)


def diagnose_text_layout(scene, bounds_report, *, padding=3.0):
    validate_scene(scene)
    width, height = scene['canvas']['width'], scene['canvas']['height']
    nodes = {item['id']: item for item in scene['nodes']}
    items = {item['id']: item for key in ('nodes', 'labels') for item in scene[key]}
    measured = []
    report = {'labels': [], 'text_overlaps': [], 'geometry_overlaps': [],
              'edge_conflicts': [], 'mapping': deepcopy(bounds_report.get('mapping')),
              'padding_px': padding,
              'interpretation': 'Measured font boxes; intersection is a layout risk, not a verified ink collision.'}
    for raw in bounds_report.get('labels', []):
        record = deepcopy(raw)
        record.update(conflicts=[], needs_manual_review=False)
        item = items.get(record.get('id'))
        boxes = text_boxes(record)
        if not item or not boxes:
            record.update(needs_manual_review=True, outside_canvas=None,
                          exceeds_expected_bbox=None, unexpected_wrap=None)
            report['labels'].append(record)
            continue
        box, expected = record['bbox_scene'], record.get('expected_bbox')
        # Font ascender boxes can extend slightly past declared extents: report
        # width/height independently rather than calling any offset an error.
        record['exceeds_expected_bbox'] = (box[2] > expected[2]+1 or box[3] > expected[3]+1) if box_valid(expected) else None
        record['outside_canvas'] = not within(box, width, height)
        record['unexpected_wrap'] = bool(dimension_text(item) and record.get('line_count', 0) > 1)
        record['measurement_class'] = 'measured_font_envelope'
        owner = metadata(item).get('node_id')
        for node_id, node in nodes.items():
            if node['type'] == 'text' or node_id == owner:
                continue
            obstacle = whole_box(node)
            if not obstacle:
                continue
            area = sum(overlap(line, obstacle) for line in boxes)
            if area <= .1:
                continue
            # A text-bearing shape and a fully contained label may deliberately
            # draw text inside the shape; do not auto-label this as an error.
            contained = overlap(box, obstacle) >= box[2]*box[3]-.1
            if node_id == record['id'] or contained:
                record.setdefault('intentional_or_unconfirmed_containment', []).append(node_id)
                continue
            issue = {'label_id': record['id'], 'node_id': node_id,
                     'overlap_area_px2': round(area, 3), 'severity': 'layout_risk',
                     'basis': 'measured_line_bounds_vs_geometry_projection_bbox'}
            report['geometry_overlaps'].append(issue)
            record['conflicts'].append(issue)
        measured.append((record, boxes))
        report['labels'].append(record)
    for index, (record, boxes) in enumerate(measured):
        for other, other_boxes in measured[index+1:]:
            area = sum(overlap(first, second) for first in boxes for second in other_boxes)
            if area > .1:
                issue = {'label_ids': [record['id'], other['id']], 'overlap_area_px2': round(area, 3),
                         'severity': 'layout_risk', 'basis': 'measured_line_bounds'}
                report['text_overlaps'].append(issue)
                record['conflicts'].append(issue)
                other['conflicts'].append(issue)
    for edge in scene['edges']:
        points = edge_points(edge, nodes)
        for record, boxes in measured:
            crossing = path_intersects(points, boxes)
            near = path_intersects(points, [padded(box, padding) for box in boxes])
            if crossing or near:
                issue = {'edge_id': edge['id'], 'label_id': record['id'],
                         'kind': 'intersects_text_bounds' if crossing else 'within_text_padding',
                         'severity': 'layout_risk', 'basis': 'measured_line_bounds',
                         'padding_px': padding}
                report['edge_conflicts'].append(issue)
                record['conflicts'].append(issue)
    for record in report['labels']:
        record['needs_manual_review'] = bool(record['needs_manual_review'] or record['conflicts']
                                            or record.get('outside_canvas') or record.get('unexpected_wrap'))
    report['summary'] = {
        'measured': len(measured), 'unresolved': len(report['labels'])-len(measured),
        'unexpected_wraps': sum(bool(item.get('unexpected_wrap')) for item in report['labels']),
        'text_overlap_pairs': len(report['text_overlaps']),
        'geometry_overlap_pairs': len(report['geometry_overlaps']),
        'edge_text_pairs': len(report['edge_conflicts']),
        'edges_with_text_conflicts': len({item['edge_id'] for item in report['edge_conflicts']}),
        'outside_canvas': sum(bool(item.get('outside_canvas')) for item in report['labels']),
    }
    return report


def fix_dimension_wrapping(scene, bounds_report):
    """Remove only confirmed automatic wrapping of short math dimension nodes.

No coordinate, bbox, font, contents or edge changes. A stale/unresolved
measurement cannot trigger this correction. Formal re-rendering must verify it.
"""
    validate_scene(scene)
    result, report = deepcopy(scene), {'changes': [], 'unresolved': []}
    records = {item['id']: item for item in bounds_report.get('labels', [])}
    for before, after in zip(scene['nodes'], result['nodes']):
        if before.get('type') != 'text' or not dimension_text(before) or before.get('wrap', True) is False:
            continue
        record = records.get(before['id'])
        if not record or not text_boxes(record):
            report['unresolved'].append({'id': before['id'], 'reason': 'dimension_bounds_unresolved'})
            continue
        if record.get('line_count', 0) <= 1:
            continue
        if metadata(before).get('review_status') in ('verified', 'rejected'):
            report['unresolved'].append({'id': before['id'], 'reason': 'protected_dimension_label'})
            continue
        # This is a conservative preflight ESTIMATE, never a measured new bbox.
        # Reflow can widen B=8 into a neighbouring formula even at the same anchor.
        lines = record.get('line_boxes_scene')
        font = resolve_style(scene, before)['font_size']
        if isinstance(lines, list) and len(lines) == record['line_count'] and all(box_valid(b) for b in lines):
            estimated_width = sum(b[2] for b in lines) + font/2
            estimated_height = max(b[3] for b in lines)
        else:
            estimated_width = record['bbox_scene'][2]*record['line_count'] + font/2
            estimated_height = record['bbox_scene'][3]/record['line_count']
        x, y, w, h = before['bbox']
        predicted = [x+w/2-estimated_width/2, y+h/2-estimated_height/2, estimated_width, estimated_height]
        blockers = [other['id'] for other in records.values() if other['id'] != before['id']
                    and any(overlap(predicted, padded(b, 2)) > .1 for b in text_boxes(other))]
        blockers += [node['id'] for node in scene['nodes'] if node['type'] != 'text'
                     and whole_box(node) and overlap(predicted, padded(whole_box(node), 2)) > .1]
        prediction = {'bbox_scene': predicted, 'kind': 'conservative_estimate_not_measurement',
                      'basis': 'measured_line_width_sum_plus_half_font_size', 'clearance_px': 2}
        if blockers or not within(predicted, scene['canvas']['width'], scene['canvas']['height']):
            report['unresolved'].append({'id': before['id'], 'reason': 'nowrap_prediction_not_safe',
                                         'blocker_ids': blockers, 'prediction': prediction})
            continue
        after['wrap'] = False
        report['changes'].append({'id': before['id'], 'before': deepcopy(before), 'after': deepcopy(after),
            'reason': 'short_math_dimension_visibly_wrapped_in_mapped_pdf',
            'measured_bbox': deepcopy(record['bbox_scene']), 'measured_line_count': record['line_count'],
            'measurement_method': record.get('method'), 'prediction': prediction, 'visual_review_required': True})
    validate_scene(result)
    return result, report
