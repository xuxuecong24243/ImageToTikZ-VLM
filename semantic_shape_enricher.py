"""Derive drawing parameters from retained measured lines, never math labels.

This deliberately narrow detector requires complete boundary evidence. Missing
Hough segments stay unresolved; a coloured bounding box is not a tensor front.
No image/model/file IO occurs here. All proposals still need human review.
"""
from __future__ import annotations

import copy
import math

from semantic_schema import validate_semantic_result
from structured_tikz.validator import validate_scene


POSITION_TOLERANCE = 3.0
BOUNDARY_TOLERANCE = 4.0
MIN_COVERAGE = .7
SPACING_RELATIVE_TOLERANCE = .2


def _retained_lines(scene):
    result = []
    records = scene.get('metadata', {}).get('review_candidates', {}).get('lines', [])
    for item in records:
        if item.get('classification') == 'text_stroke_candidate':
            continue
        if item.get('review_status') == 'rejected':
            continue
        points = item.get('points')
        if not isinstance(points, (list, tuple)) or len(points) != 2:
            continue
        if any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in points):
            continue
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for p in points for v in p):
            continue
        a, b = points
        dx, dy = abs(a[0]-b[0]), abs(a[1]-b[1])
        if max(dx, dy) <= POSITION_TOLERANCE:
            continue
        orientation = ('horizontal' if dy <= POSITION_TOLERANCE else
                       'vertical' if dx <= POSITION_TOLERANCE else 'diagonal')
        result.append({'id': item['id'], 'a': list(a), 'b': list(b), 'orientation': orientation})
    return result


def _inside(line, box):
    x, y, w, h = box
    t = BOUNDARY_TOLERANCE
    return all(x-t <= p[0] <= x+w+t and y-t <= p[1] <= y+h+t
               for p in (line['a'], line['b']))


def _clusters(values):
    groups = []
    for value, identifier in sorted(values):
        # Compare to the first member, not a drifting running average.
        if groups and value-groups[-1][0][0] <= POSITION_TOLERANCE:
            groups[-1].append((value, identifier))
        else:
            groups.append([(value, identifier)])
    return [{'position': sum(v for v, _ in group)/len(group),
             'line_ids': sorted({identifier for _, identifier in group})}
            for group in groups]


def _axis_grid(lines, box, orientation):
    x, y, w, h = box
    horizontal = orientation == 'horizontal'
    span, start, end = (w, y, y+h) if horizontal else (h, x, x+w)
    values = []
    for line in lines:
        if line['orientation'] != orientation or not _inside(line, box):
            continue
        a, b = line['a'], line['b']
        length = abs(b[0]-a[0]) if horizontal else abs(b[1]-a[1])
        if length >= MIN_COVERAGE*span:
            coordinate = (a[1]+b[1])/2 if horizontal else (a[0]+b[0])/2
            values.append((coordinate, line['id']))
    groups = _clusters(values)
    missing = []
    if not groups or abs(groups[0]['position']-start) > BOUNDARY_TOLERANCE:
        missing.append('start_boundary')
    if not groups or abs(groups[-1]['position']-end) > BOUNDARY_TOLERANCE:
        missing.append('end_boundary')
    if len(groups) < 2:
        missing.append('at_least_two_distinct_boundaries')
    gaps = [b['position']-a['position'] for a, b in zip(groups, groups[1:])]
    if gaps:
        mean = sum(gaps)/len(gaps)
        if any(abs(gap-mean) > max(POSITION_TOLERANCE, SPACING_RELATIVE_TOLERANCE*mean)
               for gap in gaps):
            missing.append('regular_spacing')
    count = len(groups)-1 if not missing else None
    if count is not None and count > 50:
        missing.append('supported_grid_count')
        count = None
    return {'count': count, 'positions_px': [g['position'] for g in groups],
            'supporting_line_ids': sorted({i for g in groups for i in g['line_ids']}),
            'missing': missing}


def _grid(lines, box):
    rows = _axis_grid(lines, box, 'horizontal')
    cols = _axis_grid(lines, box, 'vertical')
    missing = ['grid.rows.'+reason for reason in rows['missing']]
    missing += ['grid.cols.'+reason for reason in cols['missing']]
    parameters = {name: axis['count'] for name, axis in (('rows', rows), ('cols', cols))
                  if axis['count'] is not None}
    return {'parameters': parameters, 'missing': missing,
            'supporting_line_ids': sorted(set(rows['supporting_line_ids']+cols['supporting_line_ids'])),
            'axis_evidence': {'rows': rows, 'cols': cols}}


def _boundary_support(lines, start, end, orientation):
    axis = 0 if orientation == 'horizontal' else 1
    perpendicular = 1-axis
    lo, hi = sorted((start[axis], end[axis]))
    ids = []
    for line in lines:
        if line['orientation'] != orientation:
            continue
        a, b = line['a'], line['b']
        if any(abs(p[perpendicular]-start[perpendicular]) > POSITION_TOLERANCE for p in (a, b)):
            continue
        left, right = sorted((a[axis], b[axis]))
        if left >= lo-BOUNDARY_TOLERANCE and right <= hi+BOUNDARY_TOLERANCE:
            if min(right, hi)-max(left, lo) >= MIN_COVERAGE*(hi-lo):
                ids.append(line['id'])
    return sorted(ids)


def _tensor(lines, box):
    diagonals = []
    for line in lines:
        if line['orientation'] != 'diagonal' or not _inside(line, box):
            continue
        front, back = sorted((line['a'], line['b']), key=lambda p: p[0])
        dx, dy = back[0]-front[0], back[1]-front[1]
        if dx > POSITION_TOLERANCE and dy < -POSITION_TOLERANCE:
            diagonals.append({'id': line['id'], 'front': front, 'back': back, 'dx': dx, 'dy': dy})
    base = {'parameters': {}, 'missing': [], 'supporting_line_ids': [d['id'] for d in diagonals],
            'diagonal_candidate_count': len(diagonals)}
    if not 3 <= len(diagonals) <= 64:
        base['missing'] = ['unique_front_face', 'three_consistent_depth_diagonals', 'grid', 'depth']
        return base
    candidates = {}
    for left in diagonals:
        for right in diagonals:
            x, y = left['front']
            rx, ry = right['front']
            if rx-x <= 2*BOUNDARY_TOLERANCE or abs(ry-y) > POSITION_TOLERANCE:
                continue
            for bottom in diagonals:
                bx, by = bottom['front']
                if abs(bx-rx) > POSITION_TOLERANCE or by-y <= 2*BOUNDARY_TOLERANCE:
                    continue
                triple = (left, right, bottom)
                if any(abs(d['dx']-left['dx']) > POSITION_TOLERANCE or
                       abs(d['dy']-left['dy']) > POSITION_TOLERANCE for d in triple):
                    continue
                dx = sum(d['dx'] for d in triple)/3
                dy = sum(d['dy'] for d in triple)/3
                front = [x, y, rx-x, by-y]
                outer = [x, y+dy, front[2]+dx, front[3]-dy]
                if any(abs(a-b) > BOUNDARY_TOLERANCE for a, b in zip(outer, box)):
                    continue
                grid = _grid(lines, front)
                if grid['missing']:
                    continue
                top_ids = _boundary_support(lines, [x+dx, y+dy], [rx+dx, y+dy], 'horizontal')
                side_ids = _boundary_support(lines, [rx+dx, y+dy], [rx+dx, by+dy], 'vertical')
                if not top_ids or not side_ids:
                    continue
                ids = sorted(set([d['id'] for d in triple]+grid['supporting_line_ids']+top_ids+side_ids))
                key = tuple(round(v, 3) for v in front+[dx, dy])
                candidates[key] = {'parameters': {'bbox': front, 'grid': grid['parameters'],
                                                   'depth': {'dx': dx, 'dy': dy}},
                                   'missing': [], 'supporting_line_ids': ids,
                                   'axis_evidence': grid['axis_evidence']}
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    base['missing'] = ['unique_front_face', 'front_and_back_boundary_evidence', 'grid', 'depth']
    base['valid_front_candidate_count'] = len(candidates)
    return base


def enrich_shapes(scene: dict, semantic_result: dict):
    """Return complete node proposals and evidence report; inputs stay untouched.

    Incomplete estimates appear only in the report. Existing measured nodes are
    not upgraded until their entire supported drawing contract can be derived.
    Mathematical dimensions and reference scenes are never used as grid counts.
    """
    validate_semantic_result(semantic_result, scene)
    lines = _retained_lines(scene)
    nodes = {node['id']: node for node in scene['nodes']}
    proposals, entries = [], []
    for semantic in semantic_result['objects']:
        node = nodes[semantic['candidate_id']]
        kind = semantic['semantic_type']
        entry = {'node_id': node['id'], 'semantic_type': kind, 'status': 'unresolved',
                 'parameters': {}, 'missing': [], 'supporting_line_ids': []}
        entries.append(entry)
        if (scene.get('metadata', {}).get('review_status') in {'verified', 'rejected'} or
                node.get('metadata', {}).get('review_status') in {'verified', 'rejected'}):
            entry.update(status='protected', reason='existing_review_decision_preserved')
            continue
        if kind not in {'matrix', 'tensor3d'}:
            entry.update(status='completed' if node['type'] == kind else 'unresolved',
                         reason='no_grid_or_depth_enrichment_applicable')
            continue
        if kind == 'matrix':
            evidence = _grid(lines, node['bbox'])
            evidence['parameters'] = {'grid': evidence['parameters']}
        else:
            evidence = _tensor(lines, node['bbox'])
        entry.update(copy.deepcopy(evidence))
        if evidence['missing']:
            continue
        proposal = copy.deepcopy(node)
        proposal.update(copy.deepcopy(evidence['parameters']), type=kind)
        enrichment = {
            'original_bbox': copy.deepcopy(node['bbox']), 'original_type': node['type'],
            'drawing_bbox_source': 'derived_from_geometry' if kind == 'tensor3d' else 'original_measurement',
            'parameter_sources': {'type': 'existing_vlm_semantic_proposal',
                                  'grid': 'derived_from_geometry',
                                  'bbox': 'derived_from_geometry' if kind == 'tensor3d' else 'original_measurement'},
            'supporting_line_ids': copy.deepcopy(evidence['supporting_line_ids']),
            'review_status': 'unverified',
            'grid_meaning': 'visible_regular_drawing_cells_not_mathematical_dimensions',
            'limitations': 'Retained line evidence may omit grid divisions; manual review is still required.',
        }
        if kind == 'tensor3d':
            enrichment['parameter_sources']['depth'] = 'derived_from_geometry'
        proposal.setdefault('metadata', {}).update(geometry_enrichment=enrichment, review_status='unverified')
        miniature = {'canvas': scene['canvas'], 'styles': scene['styles'], 'nodes': [proposal],
                     'edges': [], 'labels': []}
        try:
            validate_scene(miniature)
        except ValueError as error:
            entry.update(missing=['scene_validation'], validation_error=str(error))
            continue
        entry['status'] = 'completed'
        proposals.append(proposal)
    counts = {status: sum(e['status'] == status for e in entries)
              for status in ('completed', 'unresolved', 'protected')}
    counts.update(proposed_nodes=len(proposals), measured_lines_considered=len(lines))
    return proposals, {'entries': entries, 'counts': counts,
                       'rules': {'position_tolerance_px': POSITION_TOLERANCE,
                                 'boundary_tolerance_px': BOUNDARY_TOLERANCE,
                                 'minimum_span_coverage': MIN_COVERAGE,
                                 'spacing_relative_tolerance': SPACING_RELATIVE_TOLERANCE},
                       'review_status': 'unverified'}
