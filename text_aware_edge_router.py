"""Finite local orthogonal candidates avoiding mapped text, never a global solver.

Only explicit routes with measured text conflicts are eligible. Endpoints,
semantic IDs, metadata and arrows are immutable. Font-box avoidance is only a
candidate layout improvement; the caller must inspect the final rendered PDF.
"""
from __future__ import annotations

from copy import deepcopy
import math

from auto_layout_refiner import _compact, _safe
from auto_scene_validator import box_valid, metadata, whole_box
from structured_tikz.validator import validate_scene
from text_layout_diagnostics import padded, path_intersects, text_boxes


def _point_inside(point, box):
    return box[0] <= point[0] <= box[0]+box[2] and box[1] <= point[1] <= box[1]+box[3]


def _orthogonal(points):
    return len(points) >= 2 and all(a != b and (a[0] == b[0] or a[1] == b[1])
                                  for a, b in zip(points, points[1:]))


def _step(a, b, distance):
    length = math.dist(a, b)
    return [a[i] + (b[i]-a[i])*distance/length for i in (0, 1)]


def _candidates(points, boxes):
    """Try fixed endpoint approaches and, preferably, existing first/last bends."""
    first, last = points[0], points[-1]
    a = _step(first, points[1], min(8, math.dist(first, points[1])/2))
    b = _step(last, points[-2], min(8, math.dist(last, points[-2])/2))
    prefixes, suffixes = [[first, a]], [[b, last]]
    if len(points) > 2:
        prefixes.insert(0, points[:2])
        suffixes.insert(0, points[-2:])
    middle = [(first[0]+last[0])/2, (first[1]+last[1])/2]
    nearest = sorted(boxes, key=lambda box: abs(box[0]+box[2]/2-middle[0])
                     +abs(box[1]+box[3]/2-middle[1]))[:24]
    x_values = sorted(set([point[0] for point in points] +
                         [x for box in nearest for x in (box[0]-1, box[0]+box[2]+1)]))
    y_values = sorted(set([point[1] for point in points] +
                         [y for box in nearest for y in (box[1]-1, box[1]+box[3]+1)]))
    seen = set()
    for prefix in prefixes:
        for suffix in suffixes:
            a, b = prefix[-1], suffix[0]
            paths = [[a, [b[0], a[1]], b], [a, [a[0], b[1]], b]]
            paths += [[a, [x, a[1]], [x, b[1]], b] for x in x_values]
            paths += [[a, [a[0], y], [b[0], y], b] for y in y_values]
            for middle_path in paths:
                candidate = _compact(prefix + middle_path + suffix)
                key = tuple(tuple(point) for point in candidate)
                if key not in seen:
                    seen.add(key)
                    yield candidate


def _direction(a, b):
    return tuple(0 if b[i] == a[i] else 1 if b[i] > a[i] else -1 for i in (0, 1))


def route_text_aware_edges(scene, bounds_report, *, padding=3.0):
    validate_scene(scene)
    if not 0 <= padding <= 10:
        raise ValueError('Text safety padding must be in [0,10] pixels.')
    result = deepcopy(scene)
    report = {'changes': [], 'unresolved': [], 'unchanged': [], 'padding_px': padding,
              'algorithm': 'finite_orthogonal_candidates_with_fixed_endpoint_approaches',
              'visual_success_verified': False}
    measured, uncertain = [], []
    for record in bounds_report.get('labels', []):
        boxes = text_boxes(record)
        if boxes:
            measured += [(record['id'], padded(box, padding)) for box in boxes]
        elif box_valid(record.get('expected_bbox')):
            uncertain.append((record['id'], padded(record['expected_bbox'], padding)))
    nodes = {item['id']: item for item in scene['nodes'] if item['type'] != 'text'}
    width, height = scene['canvas']['width'], scene['canvas']['height']
    for original, edge in zip(scene['edges'], result['edges']):
        points = edge.get('points')
        if not points:
            report['unchanged'].append({'id': edge['id'], 'reason': 'not_an_explicit_route'})
            continue
        conflicts = sorted({identifier for identifier, box in measured if path_intersects(points, [box])})
        if not conflicts:
            report['unchanged'].append({'id': edge['id'], 'reason': 'no_measured_text_conflict'})
            continue
        if not _orthogonal(points):
            report['unresolved'].append({'id': edge['id'], 'reason': 'non_orthogonal_route_preserved', 'text_ids': conflicts})
            continue
        if metadata(edge).get('review_status') in ('verified', 'rejected'):
            report['unresolved'].append({'id': edge['id'], 'reason': 'protected_route', 'text_ids': conflicts})
            continue
        endpoints = (metadata(edge).get('source_id'), metadata(edge).get('target_id'))
        if not all(identifier in nodes for identifier in endpoints):
            report['unresolved'].append({'id': edge['id'], 'reason': 'semantic_endpoints_unresolved', 'text_ids': conflicts})
            continue
        boxes = [box for _, box in measured + uncertain]
        for identifier, node in nodes.items():
            if identifier not in endpoints:
                box = whole_box(node)
                if box:
                    boxes.append(padded(box, .5))
        if any(_point_inside(point, box) for point in (points[0], points[-1]) for box in boxes):
            report['unresolved'].append({'id': edge['id'], 'reason': 'fixed_endpoint_inside_obstacle', 'text_ids': conflicts})
            continue
        candidates = []
        for candidate in _candidates(points, boxes):
            if (_orthogonal(candidate) and _safe(candidate, boxes, width, height)
                    and _direction(candidate[0], candidate[1]) == _direction(points[0], points[1])
                    and _direction(candidate[-2], candidate[-1]) == _direction(points[-2], points[-1])):
                candidates.append(candidate)
        if not candidates:
            report['unresolved'].append({'id': edge['id'], 'reason': 'no_safe_finite_candidate', 'text_ids': conflicts})
            continue
        # Prefer short paths, few bends, and retaining the established bend coordinates.
        def cost(candidate):
            return (sum(math.dist(a, b) for a, b in zip(candidate, candidate[1:]))
                    + 8*len(candidate) + 2*sum(point not in points for point in candidate), candidate)
        new_points = min(candidates, key=cost)
        edge['points'] = new_points
        assert edge['points'][0] == original['points'][0] and edge['points'][-1] == original['points'][-1]
        assert {k:v for k,v in edge.items() if k != 'points'} == {k:v for k,v in original.items() if k != 'points'}
        report['changes'].append({'id': edge['id'], 'before': deepcopy(points), 'after': deepcopy(new_points),
            'reason': 'avoid_measured_text_bounds_with_fixed_endpoints', 'text_ids': conflicts,
            'measurement_basis': 'mapped_pdf_line_bounds',
            'unresolved_text_boxes_used_conservatively': [identifier for identifier, _ in uncertain],
            'source_id': endpoints[0], 'target_id': endpoints[1], 'arrow': edge.get('arrow', 'none'),
            'visual_review_required': True})
    validate_scene(result)
    return result, report
