"""Narrow, atomic visual patches over the existing Scene patch applicator.

Model confidence is self-reported. Passing these local checks does not establish
visual improvement: projected text extents still need a new PDF measurement.
No transport, retries, code execution or compilation occurs in this module.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import re

from auto_scene_generator import _strict_json, apply_scene_patch
from auto_scene_validator import (anchor, box_valid, edge_points, finite, label_box,
                                  metadata, overlap, whole_box, within)
from semantic_edge_router import _inside_length
from structured_tikz.validator import validate_scene
from text_layout_diagnostics import padded

MAX_OPERATIONS = 8
MAX_MOVE_PX = 20.0
MAX_RESIZE_FRACTION = .10
_ID = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,63}')
_PROVENANCE = {'source', 'review_status', 'human_review_required'}
_FIELDS = {
    'move_label': {'op', 'id', 'dx', 'dy'},
    'set_node_geometry': {'op', 'id', 'value'},
    'set_edge_path': {'op', 'id', 'points'},
}


class VisualPatchError(ValueError):
    def __init__(self, reason, *, report=None):
        super().__init__(reason)
        self.report = {'accepted': False, 'applied': False, 'reason': reason,
                       'atomic_rollback': True, **(report or {})}


def _require(condition, message):
    if not condition:
        raise VisualPatchError(message)


def _identifier(value):
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def _explanation(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 500


def parse_visual_feedback(raw: str) -> dict:
    """Require one strict JSON envelope; Markdown and duplicate keys are rejected."""
    _require(isinstance(raw, str) and not raw.lstrip().startswith('```'),
             'Visual feedback must be strict JSON without Markdown.')
    try:
        data = _strict_json(raw)
        _require(set(data) == {'status', 'observations', 'operations'}, 'Unexpected feedback fields.')
        _require(data['status'] in ('no_change', 'patch_proposed'), 'Unsupported feedback status.')
        observations, operations = data['observations'], data['operations']
        _require(isinstance(observations, list) and len(observations) <= 16, 'Observations must be a bounded array.')
        _require(isinstance(operations, list) and len(operations) <= MAX_OPERATIONS, 'At most 8 operations are allowed.')
        observed = set()
        for observation in observations:
            _require(isinstance(observation, dict) and set(observation) == {'object_id', 'issue', 'confidence', 'reason'},
                     'Each observation needs object_id, issue, confidence and reason only.')
            identifier = observation['object_id']
            _require(_identifier(identifier) and identifier not in observed, 'Observation IDs must be simple and unique.')
            observed.add(identifier)
            _require(_explanation(observation['issue']) and _explanation(observation['reason']), 'Observation needs bounded visual evidence.')
            value = observation['confidence']
            _require(finite(value) and 0 <= value <= 1, 'Confidence must be a finite self-reported value in [0,1].')
        if data['status'] == 'no_change':
            _require(not operations, 'no_change cannot contain operations.')
        else:
            _require(bool(operations), 'patch_proposed requires at least one operation.')
        targeted = set()
        for operation in operations:
            _require(isinstance(operation, dict), 'Each operation must be an object.')
            name, identifier = operation.get('op'), operation.get('id')
            _require(isinstance(name, str) and name in _FIELDS, 'Unsupported visual patch operation.')
            _require(set(operation) == _FIELDS[name], 'Operation contains missing or unauthorized fields.')
            _require(_identifier(identifier) and identifier in observed, 'Each operation needs evidence for its existing object ID.')
            _require(identifier not in targeted, 'Only one operation per target prevents cumulative limit bypass.')
            targeted.add(identifier)
        return data
    except VisualPatchError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        raise VisualPatchError('Invalid visual feedback JSON: '+str(exc)) from exc


def _item_position(item, collection):
    if collection == 'nodes':
        x, y, w, h = item['bbox']
        return x+w/2, y+h/2
    return item['x'], item['y']


def _text_extents(scene, bounds_report):
    supplied = {r.get('id'): r for r in (bounds_report or {}).get('labels', []) if isinstance(r, dict)}
    mapping_ok = (bounds_report or {}).get('mapping', {}).get('status') == 'verified'
    result = {}
    for collection in ('nodes', 'labels'):
        for item in scene[collection]:
            if not item.get('text') and item.get('mode') != 'rich':
                continue
            record = supplied.get(item['id'], {})
            measured = mapping_ok and record.get('status') == 'measured' and box_valid(record.get('bbox_scene'))
            if measured and isinstance(record.get('original_position'), dict):
                position = record['original_position']
                actual = _item_position(item, collection)
                measured = all(finite(position.get(k)) and abs(position[k]-v) < .01 for k,v in zip(('x','y'),actual))
            if measured:
                lines = record.get('line_boxes_scene')
                boxes = lines if isinstance(lines, list) and lines and all(box_valid(b) for b in lines) else [record['bbox_scene']]
                source = 'input_pdf_font_bounds'
            elif collection == 'nodes':
                boxes, source = [item['bbox']], 'scene_node_bbox_estimate'
            else:
                box, source = label_box(item)
                boxes = [box] if box_valid(box) else []
            result[item['id']] = {'boxes': deepcopy(boxes), 'source': source,
                'measured_input': measured, 'projected': False, 'collection': collection,
                'owner': metadata(item).get('node_id'), 'position': list(_item_position(item, collection))}
    return result


def _project_extents(before, after, extents):
    result = deepcopy(extents)
    items = {item['id']:(collection,item) for collection in ('nodes','labels') for item in after[collection]}
    for identifier, record in result.items():
        collection, item = items[identifier]
        position = _item_position(item, collection)
        dx, dy = [a-b for a,b in zip(position, record['position'])]
        if dx or dy:
            record['boxes'] = [[b[0]+dx,b[1]+dy,b[2],b[3]] for b in record['boxes']]
            record.update(projected=True, position=list(position), source='translation_of_'+record['source'])
    return result


def _risk_metrics(scene, extents):
    """Comparable envelope risks; never label projected output bounds as measured."""
    metrics, outside = {}, []
    width, height = scene['canvas']['width'], scene['canvas']['height']
    nodes = {n['id']:n for n in scene['nodes']}
    geometry = {key:whole_box(n) for key,n in nodes.items() if n['type'] != 'text'}
    def put(key, value):
        if value > .1:
            metrics[key] = value
    for identifier, record in extents.items():
        if not record['boxes']:
            continue
        if any(not within(b,width,height) for b in record['boxes']):
            outside.append(identifier)
        for nid, box in geometry.items():
            if nid in (identifier, record['owner']) or not box:
                continue
            # Unowned containment remains a risk. A new shape swallowing a
            # label must not disappear from the before/after risk comparison.
            put(('text_geometry',identifier,nid),sum(overlap(b,box) for b in record['boxes']))
    for index,(identifier,record) in enumerate(extents.items()):
        for oid,other in list(extents.items())[index+1:]:
            put(('text_text',identifier,oid),sum(overlap(a,b) for a in record['boxes'] for b in other['boxes']))
    geometry_items = list(geometry.items())
    for index,(identifier,box) in enumerate(geometry_items):
        for oid,other in geometry_items[index+1:]:
            put(('geometry_geometry',identifier,oid),overlap(box,other))
    for edge in scene['edges']:
        points = edge_points(edge,nodes)
        if not points:
            continue
        for identifier,record in extents.items():
            put(('edge_text',edge['id'],identifier),sum(_inside_length(a,b,padded(box,3))
                for a,b in zip(points,points[1:]) for box in record['boxes']))
        endpoints = {edge.get('from'),edge.get('to'),metadata(edge).get('source_id'),metadata(edge).get('target_id')}
        for identifier,box in geometry.items():
            if identifier not in endpoints:
                put(('edge_geometry',edge['id'],identifier),sum(_inside_length(a,b,box) for a,b in zip(points,points[1:])))
    return metrics, outside


def _preserve_attachments(before, after):
    old_nodes, new_nodes = ({n['id']:n for n in value['nodes']} for value in (before,after))
    for edge in before['edges']:
        if 'points' not in edge:
            continue  # Reference edges follow their nodes by definition.
        meta = metadata(edge)
        for identifier, anchor_name, endpoint in ((meta.get('source_id'),meta.get('from_anchor'),edge['points'][0]),
                                                 (meta.get('target_id'),meta.get('to_anchor'),edge['points'][-1])):
            if identifier not in old_nodes or old_nodes[identifier]['bbox'] == new_nodes[identifier]['bbox']:
                continue
            first, second = anchor(old_nodes[identifier],anchor_name),anchor(new_nodes[identifier],anchor_name)
            _require(first is not None and second is not None, 'Changed endpoint node lacks a known attachment anchor.')
            _require(math.dist(second,endpoint) <= math.dist(first,endpoint)+1,
                     'Node change increases its gap to a fixed edge endpoint.')


def apply_visual_patch(scene, feedback, bounds_report=None):
    """Validate the entire group on a copy; reject atomically with an audit report."""
    report = {'accepted':False,'applied':False,'operations':[],
              'confidence_interpretation':'model_self_report_not_accuracy',
              'limits':{'max_operations':MAX_OPERATIONS,'movement_px':MAX_MOVE_PX,'resize_fraction':MAX_RESIZE_FRACTION}}
    try:
        validate_scene(scene)
        feedback = parse_visual_feedback(json.dumps(feedback,ensure_ascii=False,allow_nan=False))
        report['proposed_operations'] = deepcopy(feedback['operations'])
        items = {item['id']:(collection,item) for collection in ('nodes','labels','edges') for item in scene[collection]}
        observations = {o['object_id']:o for o in feedback['observations']}
        _require(all(identifier in items for identifier in observations), 'Observation references a missing object.')
        if feedback['status'] == 'no_change':
            report.update(accepted=True,applied=False,status='no_change')
            return deepcopy(scene),report
        translated = []
        for operation in feedback['operations']:
            name, identifier = operation['op'],operation['id']
            _require(identifier in items, 'Patch target is missing.')
            collection,old = items[identifier]
            reason = observations[identifier]['reason']
            if name == 'move_label':
                _require(collection == 'labels' or collection == 'nodes' and old['type'] == 'text',
                         'move_label requires a label or pure text node.')
                dx,dy = operation['dx'],operation['dy']
                _require(finite(dx) and finite(dy) and 0 < math.hypot(dx,dy) <= MAX_MOVE_PX, 'Label move must be nonzero and at most 20 px in total.')
                if collection == 'labels':
                    translated.append(dict(operation,reason=reason))
                else:
                    x,y,w,h = old['bbox']
                    translated.append({'op':'set_node_geometry','id':identifier,'reason':reason,'value':{'bbox':[x+dx,y+dy,w,h]}})
            elif name == 'set_node_geometry':
                _require(collection == 'nodes' and old['type'] != 'text', 'Geometry operation requires a non-text node.')
                value = operation['value']
                _require(isinstance(value,dict) and set(value) == {'bbox'} and box_valid(value['bbox']), 'Only a finite positive bbox may change.')
                box,previous = value['bbox'],old['bbox']
                _require(box != previous and math.dist(box[:2],previous[:2]) <= MAX_MOVE_PX, 'Node movement exceeds 20 px or is unchanged.')
                _require(all(abs(box[i]/previous[i]-1) <= MAX_RESIZE_FRACTION+1e-12 for i in (2,3)), 'Node width/height changes exceed 10%.')
                translated.append(dict(operation,reason=reason))
            else:
                _require(collection == 'edges' and 'points' in old, 'Path operation requires an existing explicit edge.')
                points = operation['points']
                _require(isinstance(points,list) and len(points) == len(old['points']) and len(points) >= 3,
                         'Keep the number of existing path vertices, including at least one intermediate point.')
                _require(all(isinstance(p,list) and len(p)==2 and all(finite(v) for v in p) for p in points), 'Path coordinates must be finite pairs.')
                _require(points != old['points'] and points[0] == old['points'][0] and points[-1] == old['points'][-1], 'Path endpoints are immutable; a no-op is not a proposal.')
                _require(all(math.dist(a,b) <= MAX_MOVE_PX for a,b in zip(points[1:-1],old['points'][1:-1])), 'Intermediate path movements exceed 20 px.')
                if all(a[0]==b[0] or a[1]==b[1] for a,b in zip(old['points'],old['points'][1:])):
                    _require(all(a[0]==b[0] or a[1]==b[1] for a,b in zip(points,points[1:])), 'Preserve an existing orthogonal route.')
                value = deepcopy(old);value['points'] = deepcopy(points)
                translated.append({'op':'replace_edge','id':identifier,'reason':reason,'value':value})
        candidate,changes = apply_scene_patch(scene,{'operations':translated})
        _preserve_attachments(scene,candidate)
        for collection,mutable in (('nodes',{'bbox','metadata'}),('labels',{'x','y','metadata'}),('edges',{'points','metadata'})):
            _require(len(scene[collection])==len(candidate[collection]), 'Object counts must not change.')
            for old,new in zip(scene[collection],candidate[collection]):
                _require({k:v for k,v in old.items() if k not in mutable} == {k:v for k,v in new.items() if k not in mutable},
                         'Content, styles, IDs, types, grids, depths or topology changed.')
                expected_metadata = deepcopy({k:v for k,v in metadata(old).items() if k not in _PROVENANCE})
                box = expected_metadata.get('bbox')
                if (collection == 'labels' and isinstance(box,list) and len(box)==4
                        and all(finite(v) for v in box)):
                    # Existing move_label translates this supplied layout hint;
                    # allow exactly that translation, never owner/topology edits.
                    box[0] += new['x']-old['x']
                    box[1] += new['y']-old['y']
                _require(expected_metadata == {k:v for k,v in metadata(new).items() if k not in _PROVENANCE},
                         'Semantic metadata changed.')
        _require(scene['canvas']==candidate['canvas'] and scene['styles']==candidate['styles'], 'Canvas and styles are immutable.')
        _require(metadata(scene).get('pending_edges',[])==metadata(candidate).get('pending_edges',[]), 'Pending topology changed.')
        extents = _text_extents(scene,bounds_report)
        projected = _project_extents(scene,candidate,extents)
        old_risks,old_outside = _risk_metrics(scene,extents)
        new_risks,new_outside = _risk_metrics(candidate,projected)
        increased = [{'kind':key[0],'ids':list(key[1:]),'before':old_risks.get(key,0),'after':value}
                     for key,value in new_risks.items() if key not in old_risks or value > old_risks[key]+1]
        new_overflow = sorted(set(new_outside)-set(old_outside))
        report['layout_guard'] = {'increased_risks':increased,'new_text_overflow':new_overflow,
            'risk_count_before':len(old_risks),'risk_count_candidate':len(new_risks),
            'candidate_text_extents':projected,
            'interpretation':'Conservative envelope preflight. Translations are projected estimates, not measurements of the new PDF.'}
        _require(not increased and not new_overflow, 'Patch would introduce or increase a known layout conflict.')
        _require(all(projected[op['id']]['boxes'] for op in feedback['operations'] if op['op']=='move_label'),
                 'Moved label lacks usable layout bounds.')
        validate_scene(candidate)
        report['operations'] = [dict(operation=deepcopy(op),observation=deepcopy(observations[op['id']]),
                                      before=change['before'],after=change['after'],passed=True,
                                      compiled=False,visual_improvement='unverified') for op,change in zip(feedback['operations'],changes)]
        report.update(accepted=True,applied=True,status='patch_validated',invariants={
            'content_and_formula_unchanged':True,'ids_counts_types_unchanged':True,
            'topology_and_arrows_unchanged':True,'source_scene_unmodified':True,'schema_valid':True})
        return candidate,report
    except (ValueError,TypeError,KeyError,OverflowError,RecursionError) as exc:
        if isinstance(exc,VisualPatchError):
            report.update(exc.report)
        raise VisualPatchError(str(exc),report=report) from exc
