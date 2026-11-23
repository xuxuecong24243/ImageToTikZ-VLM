"""Finite, offline proposals from measured text and existing Scene geometry.

No Scene is mutated here. The existing atomic visual validator applies every
trial on a copy and remains the final authority. Font envelopes are not ink
masks: tiny tensor intersections are deliberately insufficient evidence.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

from auto_scene_validator import box_valid, edge_points, finite, metadata
from semantic_edge_router import _inside_length
from structured_tikz.validator import validate_scene
from text_layout_diagnostics import padded
from visual_patch_validator import (MAX_MOVE_PX, VisualPatchError,
    _item_position, _risk_metrics, _text_extents, apply_visual_patch,
    parse_visual_feedback)

STEP_PX = 2
CLEARANCE_PX = 3.0
UNCERTAINTY_INSET_PX = 2.0


def scene_fingerprint(scene):
    return hashlib.sha256(json.dumps(scene, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def tensor_faces(node):
    """Same vertices as structured_tikz.primitives.tensor3d; no AABB substitute."""
    if node.get('type') != 'tensor3d' or not box_valid(node.get('bbox')):
        return {}
    depth = node.get('depth', {})
    dx, dy = depth.get('dx'), depth.get('dy')
    if not (finite(dx) and finite(dy) and dx > 0 and dy < 0):
        return {}
    x, y, w, h = node['bbox']
    return {
        'front': [[x,y], [x+w,y], [x+w,y+h], [x,y+h]],
        'top': [[x,y], [x+dx,y+dy], [x+w+dx,y+dy], [x+w,y]],
        'right': [[x+w,y], [x+w+dx,y+dy], [x+w+dx,y+h+dy], [x+w,y+h]],
    }


def polygon_box_area(polygon, box):
    """Clip a convex face to an axis-aligned font box (Sutherland-Hodgman)."""
    if not box_valid(box):
        return 0.0
    x, y, w, h = box
    vertices = [list(p) for p in polygon]
    for axis, bound, sign in ((0,x,1), (0,x+w,-1), (1,y,1), (1,y+h,-1)):
        result = []
        for a, b in zip(vertices, vertices[1:]+vertices[:1]):
            ia, ib = sign*(a[axis]-bound) >= 0, sign*(b[axis]-bound) >= 0
            if ia != ib:
                t = (bound-a[axis])/(b[axis]-a[axis])
                result.append([a[i]+t*(b[i]-a[i]) for i in (0,1)])
            if ib:
                result.append(b)
        vertices = result
        if not vertices:
            return 0.0
    return abs(sum(a[0]*b[1]-b[0]*a[1]
                   for a,b in zip(vertices, vertices[1:]+vertices[:1])))/2


def _face_areas(faces, boxes, margin=0):
    return {name: sum(polygon_box_area(vertices, padded(b, margin)) for b in boxes)
            for name, vertices in faces.items()}


def _point_segment_distance(p, a, b):
    d = [b[i]-a[i] for i in (0,1)]
    denominator = sum(v*v for v in d)
    t = min(1, max(0, sum((p[i]-a[i])*d[i] for i in (0,1))/denominator)) if denominator else 0
    return math.dist(p, [a[i]+t*d[i] for i in (0,1)])


def _segment_box_distance(a, b, box):
    if _inside_length(a, b, box) > 0:
        return 0.0
    x, y, w, h = box
    corners = [[x,y], [x+w,y], [x+w,y+h], [x,y+h]]
    distances = [_point_segment_distance(p,a,b) for p in corners]
    for p in (a,b):
        distances.append(math.hypot(max(x-p[0],0,p[0]-x-w), max(y-p[1],0,p[1]-y-h)))
    return min(distances)


def _line_metric(points, boxes):
    segments = list(zip(points, points[1:]))
    return {
        'intersection_length_px': sum(_inside_length(a,b,box) for a,b in segments for box in boxes),
        'padded_intersection_length_px': sum(_inside_length(a,b,padded(box,CLEARANCE_PX))
                                             for a,b in segments for box in boxes),
        'distance_px': min((_segment_box_distance(a,b,box) for a,b in segments for box in boxes), default=None),
    }


def measure_layout(scene, bounds_report):
    """Read-only metrics; tensor entries use faces, not overall bounding boxes."""
    validate_scene(scene)
    extents = _text_extents(scene, bounds_report)
    nodes = {n['id']:n for n in scene['nodes']}
    risks, outside = _risk_metrics(scene, extents)
    lines, tensors = [], []
    for identifier, record in extents.items():
        if not record['measured_input']:
            continue
        for edge in scene['edges']:
            points = edge_points(edge,nodes)
            if points:
                metric = _line_metric(points,record['boxes'])
                if metric['padded_intersection_length_px'] > .1:
                    lines.append(dict(metric, text_id=identifier, edge_id=edge['id']))
        for node in scene['nodes']:
            faces = tensor_faces(node)
            if not faces or node['id'] in (identifier,record['owner']):
                continue
            areas = _face_areas(faces,record['boxes'])
            core = _face_areas(faces,record['boxes'],-UNCERTAINTY_INSET_PX)
            if sum(areas.values()) > .1:
                tensors.append({'text_id':identifier, 'tensor_id':node['id'],
                    'face_overlap_area_px2':areas, 'interior_overlap_area_px2':core,
                    'substantial_envelope_intersection':sum(core.values()) > 1})
    return {'summary': {
        'edge_text_pairs':sum(v['intersection_length_px'] > .1 for v in lines),
        'edge_text_padded_pairs':len(lines), 'tensor_text_pairs':len(tensors),
        'tensor_text_substantial_pairs':sum(v['substantial_envelope_intersection'] for v in tensors),
        'text_overlap_pairs':sum(k[0]=='text_text' for k in risks), 'outside_canvas':len(outside),
        'unmeasured_text_objects':sum(not v['measured_input'] for v in extents.values())},
        'edge_text':lines, 'tensor_text':tensors, 'outside_canvas':outside,
        'envelope_risks':[{'kind':k[0], 'ids':list(k[1:]), 'value':v} for k,v in sorted(risks.items())],
        'interpretation':'Mapped font envelopes versus Scene segments and exact tensor faces; not measured ink occlusion.'}


def _evidence_error(scene, evidence):
    if evidence.get('scene_fingerprint') != scene_fingerprint(scene):
        return 'Scene fingerprint differs from the immutable measurement baseline.'
    bounds = evidence.get('bounds_report', {})
    if bounds.get('mapping', {}).get('status') != 'verified':
        return 'PDF-to-Scene coordinate mapping is not verified.'
    items = {i['id']:(c,i) for c in ('nodes','labels') for i in scene[c]}
    seen = set()
    for record in bounds.get('labels', []):
        identifier = record.get('id')
        if identifier in seen or identifier not in items:
            return 'Unknown or duplicate measured object.'
        seen.add(identifier)
        if record.get('status') != 'measured':
            continue
        collection,item = items[identifier]
        position = record.get('original_position', {})
        expected = item['bbox'] if collection=='nodes' else metadata(item).get('bbox')
        if (record.get('collection',collection) != collection
                or any(not finite(position.get(k)) or abs(position[k]-v) > .01
                       for k,v in zip(('x','y'),_item_position(item,collection)))
                or record.get('expected_bbox') != expected):
            return 'Measured position/expected bbox differs from current Scene.'
    return None


def _envelope(feedback, operations):
    ids = {op['id'] for op in operations}
    return {'status':'patch_proposed' if operations else 'no_change',
            'operations':deepcopy(operations),
            'observations':[deepcopy(o) for o in feedback['observations'] if o['object_id'] in ids]}


def _target_evidence(scene, identifier, boxes, issue):
    nodes = {n['id']:n for n in scene['nodes']}
    lines, tensors = [], []
    if issue == 'edge_text_overlap':
        for edge in scene['edges']:
            points = edge_points(edge,nodes)
            if points and _line_metric(points,boxes)['intersection_length_px'] > .1:
                lines.append({'edge_id':edge['id'], 'points':points, **_line_metric(points,boxes)})
    elif issue == 'label_on_tensor':
        for node in scene['nodes']:
            faces = tensor_faces(node)
            if not faces or node['id'] == identifier:
                continue
            areas = _face_areas(faces,boxes)
            if sum(areas.values()) > .1:
                core = _face_areas(faces,boxes,-UNCERTAINTY_INSET_PX)
                tensors.append({'tensor_id':node['id'], 'faces':faces,
                    'face_overlap_area_px2':areas, 'interior_overlap_area_px2':core,
                    'sufficient_evidence':sum(core.values()) > 1})
    return {'text_boxes':deepcopy(boxes), 'lines':lines, 'tensors':tensors,
            'sufficient_evidence':bool(lines or any(v['sufficient_evidence'] for v in tensors)),
            'basis':'measured_font_bounds_and_scene_geometry_not_model_confidence'}


def _evaluate(evidence, boxes):
    lines = [{'edge_id':v['edge_id'], **_line_metric(v['points'],boxes)} for v in evidence['lines']]
    tensors = [{'tensor_id':v['tensor_id'], 'face_overlap_area_px2':_face_areas(v['faces'],boxes),
                'padded_overlap_area_px2':_face_areas(v['faces'],boxes,CLEARANCE_PX)}
               for v in evidence['tensors'] if v['sufficient_evidence']]
    clear = (all(v['padded_intersection_length_px'] <= .1 for v in lines)
             and all(sum(v['padded_overlap_area_px2'].values()) <= .1 for v in tensors))
    return {'text_boxes':boxes, 'lines':lines, 'tensors':tensors, 'target_conflict_removed':clear}


def optimize_visual_patches(scene, model_feedback, geometry_evidence):
    """Return strict optimized feedback plus audit; all moves remain baseline-relative.

    This intentionally requires COMPLETE target clearance. Partial gains remain
    diagnostics only. A failed joint check rejects the entire selected group.
    """
    validate_scene(scene)
    feedback = parse_visual_feedback(json.dumps(model_feedback,ensure_ascii=False,allow_nan=False))
    error = _evidence_error(scene,geometry_evidence)
    bounds = geometry_evidence.get('bounds_report',{})
    extents = _text_extents(scene,bounds)
    measurements = {r['id']:r for r in bounds.get('labels',[])}
    observations = {o['object_id']:o for o in feedback['observations']}
    records, selected = [], []
    for op in feedback['operations']:
        identifier = op['id']
        row = {'object_id':identifier, 'model_dx':op.get('dx'), 'model_dy':op.get('dy'),
               'selected_dx':0, 'selected_dy':0, 'status':'rejected', 'reason':None,
               'evidence':{}, 'candidates':[]}
        records.append(row)
        record = extents.get(identifier)
        issue = observations[identifier]['issue']
        if error:
            row.update(reason='stale_or_unverified_evidence',detail=error); continue
        if (op['op'] != 'move_label' or issue == 'wrapped_label'
                or measurements.get(identifier,{}).get('line_count',0) > 1):
            row.update(reason='wrong_operation_type',detail='Translation cannot repair wrapping or text width.'); continue
        if not record or not record['measured_input']:
            row.update(reason='insufficient_geometric_evidence',detail='No matching measured text bounds.'); continue
        dx, dy = op['dx'], op['dy']
        if not (finite(dx) and finite(dy) and 0 < math.hypot(dx,dy) < math.inf):
            row.update(reason='invalid_direction'); continue
        evidence = _target_evidence(scene,identifier,record['boxes'],issue)
        row['evidence'] = evidence
        if not evidence['sufficient_evidence']:
            row.update(reason='insufficient_geometric_evidence'); continue
        length = math.hypot(dx,dy)
        row['candidates'].append({'dx':0,'dy':0,'movement_px':0,
            'status':'baseline', **_evaluate(evidence,record['boxes'])})
        for step in range(STEP_PX,int(MAX_MOVE_PX)+1,STEP_PX):
            sx, sy = dx/length*step, dy/length*step
            if math.hypot(sx,sy) > MAX_MOVE_PX:
                factor = (MAX_MOVE_PX-1e-9)/math.hypot(sx,sy)
                sx, sy = sx*factor, sy*factor
            boxes = [[b[0]+sx,b[1]+sy,b[2],b[3]] for b in record['boxes']]
            trial = {'dx':sx,'dy':sy,'movement_px':math.hypot(sx,sy), **_evaluate(evidence,boxes)}
            row['candidates'].append(trial)
            if not trial['target_conflict_removed']:
                trial.update(status='rejected',reason='target_conflict_remains'); continue
            proposal = {'op':'move_label','id':identifier,'dx':sx,'dy':sy}
            try:
                _, validation = apply_visual_patch(scene,_envelope(feedback,[proposal]),bounds)
            except VisualPatchError as exc:
                trial.update(status='rejected',reason='individual_validation_failed',validation=exc.report)
                continue
            trial.update(status='selected',validation=validation)
            row.update(status='accepted',reason='verified_collision_removed',selected_dx=sx,selected_dy=sy)
            selected.append(proposal)
            break
        if row['status'] != 'accepted':
            row['reason'] = 'no_safe_candidate'
    optimized = _envelope(feedback,selected)
    try:
        _, joint = apply_visual_patch(scene,optimized,bounds)
    except VisualPatchError as exc:
        joint = exc.report
        for row in records:
            if row['status'] == 'accepted':
                row.update(status='rejected', reason='joint_validation_failed',selected_dx=0,selected_dy=0)
        optimized = _envelope(feedback,[])
    counts = {'original_suggestions':len(records),
        'geometric_evidence':sum(bool(r['evidence'].get('sufficient_evidence')) for r in records),
        'individually_selected':len(selected),
        'accepted':len(optimized['operations']),
        'insufficient_evidence':sum(r['reason']=='insufficient_geometric_evidence' for r in records),
        'wrong_operation_type':sum(r['reason']=='wrong_operation_type' for r in records),
        'no_safe_candidate':sum(r['reason']=='no_safe_candidate' for r in records),
        'joint_rejected':sum(r['reason']=='joint_validation_failed' for r in records)}
    return optimized, {'items':records,'counts':counts,'joint_validation':joint,
        'baseline_fingerprint':scene_fingerprint(scene),'movement_limit_px':MAX_MOVE_PX,
        'candidate_step_px':STEP_PX,'clearance_px':CLEARANCE_PX,
        'tensor_uncertainty_inset_px':UNCERTAINTY_INSET_PX,
        'policy':'Smallest fully clearing safe candidate on a 2px grid; no partial repair or sequential accumulation.',
        'limitations':['Finite direction-only candidates can miss a better route.',
            'Font envelopes are not visible ink; tensor core overlap is a conservative heuristic.',
            'Projected translations require one final PDF measurement and human image review.']}
