"""Conservative semantic relations + measured line segments -> edge proposals.

No image recognition, shortest-path search, invented gap filling, or arrowhead
detection occurs here. A path uses one segment or at most three approximately
orthogonal measured segments. Intersections may extend an endpoint by <= 3 px
(Hough rounding tolerance); node associations use 15% of the smaller bbox side,
clamped to [12, 25] px. These fixed thresholds are not fitted to the test image.
Every proposal remains unverified; arrow directions come only from the VLM.
"""
from __future__ import annotations

import copy
import math

from semantic_schema import validate_semantic_result


INTERSECTION_TOLERANCE_PX = 3.0
MAX_LINES = 256
MAX_PATH_CANDIDATES = 256


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _bbox(node):
    """Use original measurements even when a tensor's front face was proposed."""
    raw = node.get('metadata', {}).get('source_candidate', {}).get('bbox_px')
    if isinstance(raw, dict) and all(k in raw for k in ('x', 'y', 'width', 'height')):
        return [raw[k] for k in ('x', 'y', 'width', 'height')]
    if isinstance(raw, list) and len(raw) == 4:
        return raw
    return node['bbox']


def _association(point, box):
    x, y, w, h = box
    px, py = point
    if x + 2 < px < x + w - 2 and y + 2 < py < y + h - 2:
        return None
    cx, cy = min(max(px, x), x + w), min(max(py, y), y + h)
    projections = [('north', [cx, y]), ('south', [cx, y + h]),
                   ('west', [x, cy]), ('east', [x + w, cy])]
    name, projected = min(projections, key=lambda pair: _distance(point, pair[1]))
    distance = _distance(point, projected)
    tolerance = min(25.0, max(12.0, .15 * min(w, h)))
    if distance > tolerance:
        return None
    return {'anchor': name, 'boundary_point': projected, 'measured_point': list(point),
            'distance_px': round(distance, 3), 'tolerance_px': round(tolerance, 3),
            'source': 'opencv_bbox_boundary_projection',
            'meaning': 'side_projection_not_an_exact_TikZ_cardinal_anchor'}


def _exits(point, next_point, anchor):
    normals = {'north': (0, -1), 'south': (0, 1), 'west': (-1, 0), 'east': (1, 0)}
    nx, ny = normals[anchor['anchor']]
    return (next_point[0] - point[0]) * nx + (next_point[1] - point[1]) * ny > .1


def _intersection(first, second):
    a, b = first['points']; c, d = second['points']
    u = (b[0] - a[0], b[1] - a[1]); v = (d[0] - c[0], d[1] - c[1])
    length_u, length_v = math.hypot(*u), math.hypot(*v)
    if abs(u[0] * v[0] + u[1] * v[1]) > .08 * length_u * length_v:
        return None
    denominator = u[0] * v[1] - u[1] * v[0]
    if abs(denominator) < 1e-8:
        return None
    delta = (c[0] - a[0], c[1] - a[1])
    t = (delta[0] * v[1] - delta[1] * v[0]) / denominator
    s = (delta[0] * u[1] - delta[1] * u[0]) / denominator
    tolerance = INTERSECTION_TOLERANCE_PX
    if not (-tolerance / length_u <= t <= 1 + tolerance / length_u and
            -tolerance / length_v <= s <= 1 + tolerance / length_v):
        return None
    return [a[0] + t * u[0], a[1] + t * u[1]]


def _inside_length(a, b, box):
    """Liang-Barsky clipped length, including boundary overlaps."""
    x, y, w, h = box
    dx, dy = b[0] - a[0], b[1] - a[1]
    low, high = 0.0, 1.0
    for p, q in ((-dx, a[0] - x), (dx, x + w - a[0]),
                 (-dy, a[1] - y), (dy, y + h - a[1])):
        if abs(p) < 1e-9:
            if q < 0:
                return 0.0
        elif p < 0:
            low = max(low, q / p)
        else:
            high = min(high, q / p)
        if low >= high:
            return 0.0
    return (high - low) * math.hypot(dx, dy)


def _conflicts(points, nodes, source_id, target_id, text_boxes):
    reasons = []
    for node_id, node in nodes.items():
        x, y, w, h = _bbox(node)
        # One-pixel boundary rounding is allowed; paths through interiors are not.
        inset = min(1.0, w / 4, h / 4)
        box = [x + inset, y + inset, w - 2 * inset, h - 2 * inset]
        if any(_inside_length(a, b, box) > 2 for a, b in zip(points, points[1:])):
            kind = 'endpoint_node_interior' if node_id in (source_id, target_id) else 'unrelated_node'
            reasons.append(kind + ':' + node_id)
    for index, box in enumerate(text_boxes):
        if any(_inside_length(a, b, box) > 1 for a, b in zip(points, points[1:])):
            reasons.append('known_text_box:' + str(index))
    return reasons


def _line_candidates(scene):
    lines, excluded = [], []
    for item in scene.get('metadata', {}).get('review_candidates', {}).get('lines', []):
        original = item.get('source_candidate', {})
        evidence = original.get('evidence', {})
        reason = None
        if evidence.get('contained_in') or evidence.get('possible_role') == 'border_or_grid':
            reason = 'border_or_grid_evidence'
        elif item.get('classification', original.get('classification')) == 'text_stroke_candidate':
            reason = 'text_stroke_candidate'
        points = item.get('points', [])
        valid = (isinstance(points, list) and len(points) == 2 and
                 all(isinstance(p, (list, tuple)) and len(p) == 2 and
                     all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)
                         for v in p) for p in points))
        if not valid or valid and _distance(*points) < 4:
            reason = reason or 'invalid_or_short_measurement'
        if reason:
            excluded.append({'id': item.get('id'), 'reason': reason})
        else:
            lines.append({'id': item['id'], 'points': copy.deepcopy(points),
                          'classification': item.get('classification', 'uncertain')})
    if len(lines) > MAX_LINES:
        raise ValueError('Too many connector candidates; bounded edge router accepts at most 256.')
    return lines, excluded


def _edge_pair(edge):
    metadata = edge.get('metadata', {})
    relation = metadata.get('relation', {})
    return (edge.get('from', relation.get('source_id', metadata.get('source_id'))),
            edge.get('to', relation.get('target_id', metadata.get('target_id'))))


def _status(item):
    return item.get('metadata', {}).get('review_status', 'unverified')


def _same_points(a, b):
    return len(a) == len(b) and (all(_distance(x, y) <= 2 for x, y in zip(a, b)) or
                                all(_distance(x, y) <= 2 for x, y in zip(a, reversed(b))))


def _paths(lines, adjacency, source_box, target_box):
    """Explicitly enumerate 1/2/3-segment chains, never perform global routing."""
    found, partial = [], []
    for first_index, first in enumerate(lines):
        for start in first['points']:
            source_anchor = _association(start, source_box)
            if source_anchor is None:
                continue
            other = first['points'][1] if start == first['points'][0] else first['points'][0]
            if not _exits(start, other, source_anchor):
                continue
            partial.append({'points': [list(start), list(other)], 'matched_line_ids': [first['id']],
                            'anchors': {'source': source_anchor, 'target': None}})
            chains = [([first_index], [list(start)])]
            chains += [([first_index, j], [list(start), intersection])
                       for j, intersection in adjacency[first_index]]
            chains += [([first_index, j, k], [list(start), intersection, second_intersection])
                       for j, intersection in adjacency[first_index]
                       for k, second_intersection in adjacency[j] if k != first_index]
            if len(chains) > MAX_PATH_CANDIDATES:
                return [], partial, True
            for indices, prefix in chains:
                last = lines[indices[-1]]
                for end in last['points']:
                    target_anchor = _association(end, target_box)
                    points = prefix + [list(end)]
                    if target_anchor is None or any(_distance(a, b) < 2 for a, b in zip(points, points[1:])):
                        continue
                    if not _exits(points[0], points[1], source_anchor) or not _exits(points[-1], points[-2], target_anchor):
                        continue
                    item = {'points': [[round(v, 4) for v in p] for p in points],
                            'matched_line_ids': [lines[i]['id'] for i in indices],
                            'anchors': {'source': source_anchor, 'target': target_anchor}}
                    if not any(_same_points(item['points'], old['points']) for old in found):
                        found.append(item)
                    if len(found) > MAX_PATH_CANDIDATES:
                        return [], partial, True
    return found, partial, False


def route_edges(scene, semantic_result, *, text_boxes=None):
    """Return schema-compatible unverified edges and a report for every relation.

    Inputs are never mutated. Text boxes are pixel [x,y,w,h] arrays. If omitted,
    measured scene text-region boxes are used. Invalid relation IDs are rejected
    by the existing semantic validator instead of being guessed or repaired.
    """
    validate_semantic_result(semantic_result, scene)
    nodes = {node['id']: node for node in scene['nodes']}
    lines, excluded = _line_candidates(scene)
    if text_boxes is None:
        text_boxes = [item['bbox'] for item in scene.get('metadata', {}).get('review_candidates', {}).get('text_regions', [])
                      if item.get('bbox') is not None]
    for box in text_boxes:
        if (not isinstance(box, (list, tuple)) or len(box) != 4 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box) or
                box[2] <= 0 or box[3] <= 0):
            raise ValueError('Text boxes must be finite pixel [x,y,width,height] arrays.')
    adjacency = {i: [] for i in range(len(lines))}
    for i, first in enumerate(lines):
        for j in range(i + 1, len(lines)):
            point = _intersection(first, lines[j])
            if point is not None:
                adjacency[i].append((j, point)); adjacency[j].append((i, point))
    proposals, entries = [], []
    identifiers = {item['id'] for kind in ('nodes', 'edges', 'labels') for item in scene[kind]}
    for index, relation in enumerate(semantic_result['relations'], 1):
        source_id, target_id = relation['source_id'], relation['target_id']
        entry = {'index': index, 'source_id': source_id, 'target_id': target_id,
                 'relation': copy.deepcopy(relation), 'status': 'unresolved',
                 'points': None, 'matched_line_ids': [], 'anchors': None, 'reasons': [],
                 'review_status': 'unverified',
                 'sources': {'relation': 'vlm_proposal', 'arrow_direction': 'vlm_proposal',
                             'path': 'opencv_lines_and_bounded_intersections', 'anchors': 'bbox_boundary_projection'}}
        entries.append(entry)
        protected = [edge for edge in scene['edges'] if _edge_pair(edge) in
                     ((source_id, target_id), (target_id, source_id))]
        if (_status(scene) in {'verified', 'rejected'} or protected or
                any(_status(nodes[node_id]) == 'rejected' for node_id in (source_id, target_id) if node_id in nodes)):
            entry.update(status='protected', reasons=['existing_scene_edge_or_rejected_node_is_protected'],
                         protected_edge_ids=[edge['id'] for edge in protected])
            continue
        if relation['relation'] not in {'arrow', 'direct_line'} or source_id is None or target_id is None:
            entry['reasons'] = ['semantic_relation_does_not_establish_a_connector']
            continue
        if relation['relation'] == 'arrow' and relation['direction'] in {'none', 'unknown'}:
            entry['reasons'] = ['arrow_direction_unknown']
            continue
        candidates, partial, limited = _paths(lines, adjacency, _bbox(nodes[source_id]), _bbox(nodes[target_id]))
        entry['candidate_count'] = len(candidates)
        if limited:
            entry['reasons'] = ['bounded_candidate_limit_reached_no_route_selected']
            continue
        if not candidates:
            entry['reasons'] = ['no_complete_measured_path_with_both_boundary_associations']
            if partial:
                # Only display measured partial evidence; do not bridge its gap.
                best = min(partial, key=lambda item: _distance(item['points'][-1],
                           [_bbox(nodes[target_id])[0] + _bbox(nodes[target_id])[2] / 2,
                            _bbox(nodes[target_id])[1] + _bbox(nodes[target_id])[3] / 2]))
                entry.update(copy.deepcopy(best))
                entry['partial_evidence_only'] = True
            continue
        safe = []
        for candidate in candidates:
            candidate['conflicts'] = _conflicts(candidate['points'], nodes, source_id, target_id, text_boxes)
            if not candidate['conflicts']:
                safe.append(candidate)
        if not safe:
            entry.update(copy.deepcopy(candidates[0]))
            entry.update(status='conflict', reasons=['all_measured_routes_cross_known_boxes'])
            entry['candidate_paths'] = candidates[:8]
            continue
        if len(safe) != 1:
            entry.update(status='unresolved', reasons=['multiple_measured_routes_require_review'],
                         candidate_paths=safe[:8], matched_line_ids=sorted({line for c in safe for line in c['matched_line_ids']}))
            continue
        candidate = safe[0]
        existing = [edge for edge in scene['edges'] if 'points' in edge and _same_points(edge['points'], candidate['points'])]
        if existing:
            entry.update(copy.deepcopy(candidate))
            entry.update(status='protected', reasons=['existing_absolute_path_preserved'], protected_edge_ids=[e['id'] for e in existing])
            continue
        edge_id = 'enrich_edge_{:03d}'.format(index)
        suffix = 1
        while edge_id in identifiers:
            edge_id = 'enrich_edge_{:03d}_{}'.format(index, suffix); suffix += 1
        identifiers.add(edge_id)
        arrow = 'none' if relation['relation'] == 'direct_line' else {'forward': 'end', 'reverse': 'start', 'bidirectional': 'both'}[relation['direction']]
        metadata = {'review_status': 'unverified', 'source': 'rule_derived_geometry_with_vlm_relation',
                    'relation': copy.deepcopy(relation), 'geometry_line_ids': candidate['matched_line_ids'],
                    'anchors': candidate['anchors'], 'arrow_source': 'vlm_proposal_not_measured',
                    'parameter_sources': {'points': 'opencv_lines_and_bounded_intersections',
                                          'arrow': 'vlm_proposal', 'node_association': 'rule_derived_bbox_boundary'},
                    'human_review_required': True}
        proposals.append({'id': edge_id, 'points': candidate['points'], 'arrow': arrow, 'metadata': metadata})
        entry.update(copy.deepcopy(candidate))
        entry.update(status='recovered', edge_id=edge_id, reasons=['measured_path_is_a_reviewable_proposal_not_verified_topology'])
    counts = {'input_relations': len(entries), 'recovered_paths': len(proposals),
              'semantic_only': sum(e['status'] in {'unresolved', 'conflict'} for e in entries),
              'matched_relations': sum(e.get('candidate_count', 0) > 0 for e in entries),
              'path_conflicts': sum(e['status'] == 'conflict' for e in entries)}
    return proposals, {'entries': entries, 'counts': counts, 'excluded_line_candidates': excluded,
                       'thresholds': {'intersection_tolerance_px': INTERSECTION_TOLERANCE_PX,
                                      'endpoint_tolerance': '15% of smaller side, clamped to 12..25 px',
                                      'max_segments': 3},
                       'review_status': 'unverified'}
