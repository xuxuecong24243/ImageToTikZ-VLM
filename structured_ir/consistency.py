"""Optional evidence-scoped semantic audit; never redefine converter.complete.

Reference data is external reviewed annotation, not inferred from names or
proximity. Bind it to image_sha256, ir_sha256, confirmed=True and a review basis.
connections contains exact expected fields and optional source_type/target_type
plus allowed_type_pairs (types are supplied by the reference, never guessed).
pairs contains shape_id, text_id, text, shape_bbox, text_bbox in sequence order.
overlaps contains ids=[a,b], classification=legal_overlap|duplicate_detection.
Overall semantic status stays unverified without full-image adjudication; local
checks can be verified, conflict, unverified or not_applicable independently.
"""
from copy import deepcopy
import re
from .regions import _box
from .validator import inspect_ir, _contains
from .geometry_repair import ir_fingerprint, repair_geometry, _intersection
from .schema import TEXT_TYPES
from structured_tikz.geometry import node_anchor, edge_route_points
from structured_tikz.validator import validate_scene

SHAPE_TYPES = frozenset({'rectangle', 'rounded_rectangle', 'circle', 'diamond', 'title_band'})

def _reference(ir, reference, image_sha256):
    if reference is None:
        return {}
    if (not isinstance(reference, dict) or reference.get('confirmed') is not True
            or not isinstance(reference.get('basis'), str) or not reference['basis'].strip()
            or not isinstance(image_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', image_sha256)
            or reference.get('image_sha256') != image_sha256
            or reference.get('ir_sha256') != ir_fingerprint(ir)
            or reference.get('coordinate_system') != ir['canvas']['coordinate_system']
            or reference.get('canvas_size') != [ir['canvas']['width'],ir['canvas']['height']]):
        raise ValueError('Reference requires reviewed, image/IR-bound evidence.')
    for key in ('connections', 'pairs', 'overlaps'):
        if not isinstance(reference.get(key, []), list) or len(reference.get(key, [])) > 1000:
            raise ValueError('Invalid reference collection: ' + key)
    if reference.get('sequence_axis',0) not in (0,1): raise ValueError('Sequence axis must be x=0 or y=1.')
    connection_ids = set()
    for c in reference.get('connections', []):
        if (not isinstance(c, dict) or not isinstance(c.get('id'), str) or not c['id']
                or c['id'] in connection_ids or c.get('status', 'verified') not in ('verified', 'unverified')):
            raise ValueError('Connection references require unique IDs and an explicit supported status.')
        connection_ids.add(c['id'])
        for key in ('source_id', 'target_id', 'source_type', 'target_type'):
            if key in c and (not isinstance(c[key], str) or not c[key]):
                raise ValueError('Invalid connection reference field: ' + key)
        if 'allowed_type_pairs' in c and (not isinstance(c['allowed_type_pairs'], list)
                or any(not isinstance(p, list) or len(p) != 2
                       or any(not isinstance(v, str) or not v for v in p) for p in c['allowed_type_pairs'])):
            raise ValueError('Allowed type pairs must be pairs of nonempty type names.')
    for p in reference.get('pairs',[]):
        if not isinstance(p,dict) or not {'shape_id','text_id','text','shape_bbox','text_bbox'} <= set(p):
            raise ValueError('Incomplete pair reference.')
        if any(not isinstance(p[k], str) or not p[k] for k in ('shape_id', 'text_id', 'text')):
            raise ValueError('Pair references require nonempty IDs and literal text.')
        _box(p['shape_bbox']); _box(p['text_bbox'])
    overlap_ids = set()
    for o in reference.get('overlaps',[]):
        if (not isinstance(o, dict) or not isinstance(o.get('ids'), list) or len(o['ids']) != 2
                or any(not isinstance(v, str) or not v for v in o['ids'])
                or o['ids'][0] == o['ids'][1] or frozenset(o['ids']) in overlap_ids
                or o.get('classification') not in ('legal_overlap','duplicate_detection')):
            raise ValueError('Invalid overlap reference.')
        overlap_ids.add(frozenset(o['ids']))
    return reference


def inspect_consistency(ir, scene=None, conversion_report=None, reference=None, *, image_sha256=None):
    """Read-only audit. No absence-of-errors or conversion-success certification."""
    validation = inspect_ir(ir)
    report = {'conversion_status': (conversion_report or {}).get('status', 'not_evaluated'),
              'semantic_status': 'unverified', 'visual_status': 'unverified',
              'scope': 'Only explicit reference relationships and geometry; not full-image certification.',
              'ir_validation': validation, 'checks': [], 'connections': [], 'collisions': [], 'pairings': [], 'api_calls': 0}
    def check(code, status, ids, evidence, **details):
        row = dict(code=code, status=status, ids=ids, evidence=evidence, **details)
        report['checks'].append(row)
        return row
    if not validation['valid']:
        for error in validation['errors']:
            check(error['code'], 'conflict', [], error['message'], path=error['path'])
        report['semantic_status'] = 'partial'
        return report
    ref = _reference(ir, reference, image_sha256)
    if scene is not None:
        try: validate_scene(scene)
        except ValueError as exc:
            check('invalid_scene','conflict',[],str(exc))
            report['semantic_status']='partial'
            return report
    elements = {e['id']: e for e in ir['elements']}
    nodes = {n['id']: n for n in (scene or {}).get('nodes', [])}
    edges = {e['id']: e for e in (scene or {}).get('edges', [])}
    expected_edges = {c['id']: c for c in ref.get('connections', [])}
    for c in ir['connections']:
        row = {'id': c['id'], 'ir': deepcopy(c), 'scene_mapping': 'unverified',
               'semantic_status': 'unverified', 'evidence': 'No reviewed relation for this edge.'}
        expected = expected_edges.get(c['id'])
        if expected:
            exact = {k: expected[k] for k in ('source_id','target_id','direction','kind','branch_label',
                                            'source_anchor','target_anchor','points') if k in expected}
            if expected.get('status') == 'unverified':
                state = 'unverified'
            else:
                sufficient = all(k in exact for k in ('source_id','target_id','direction'))
                state = 'conflict' if any(c.get(k) != v for k,v in exact.items()) else 'verified' if sufficient else 'unverified'
            row.update(semantic_status=state, evidence=expected.get('basis', ref['basis']))
            check('reference_relation', state, [c['id']], row['evidence'], expected=exact)
            if 'allowed_type_pairs' in expected:
                pair = [expected.get('source_type'), expected.get('target_type')]
                dtype_state = 'unverified' if None in pair or state == 'unverified' else 'conflict' if state == 'conflict' or pair not in expected['allowed_type_pairs'] else 'verified'
                check('dataflow_types', dtype_state, [c['id']], 'Explicit reviewed endpoint types and allowed transitions.', types=pair)
                if dtype_state != 'verified' and row['semantic_status'] != 'conflict':
                    row['semantic_status'] = dtype_state
        if scene is not None:
            e = edges.get(c['id'])
            errors = []
            if e is None:
                errors.append('missing_scene_edge')
            else:
                meta = e.get('metadata', {})
                for source, direct in [('source_id','from'),('target_id','to')]:
                    actual = e.get(direct, meta.get(source))
                    if actual != c[source]: errors.append(source+'_mapping_mismatch')
                if e.get('arrow','none') != {'forward':'end','reverse':'start','both':'both','none':'none'}[c['direction']]:
                    errors.append('direction_mapping_mismatch')
                try:
                    points = edge_route_points(e, nodes)
                    for actual, key, anchor in [(points[0],'source_id','source_anchor'),(points[-1],'target_id','target_anchor')]:
                        target = node_anchor(elements[c[key]], c[anchor])
                        if any(abs(a-b)>1e-6 for a,b in zip(actual,target)): errors.append(anchor+'_mismatch')
                    if c.get('points') and e.get('points') != c['points']: errors.append('path_points_changed')
                except (KeyError, ValueError, IndexError):
                    errors.append('unresolvable_scene_anchor')
                if 'branch_label' in c:
                    named = nodes.get(meta.get('branch_label_node_id'))
                    labels = [n for n in (scene or {}).get('labels', []) if n.get('metadata',{}).get('ir_connection') == c['id']]
                    if named: labels.append(named)
                    if len(labels)!=1 or labels[0].get('text') != c['branch_label']:
                        errors.append('branch_label_mapping_unverified')
                    elif 'label_bbox' in c:
                        label = labels[0]; x,y,w,h = c['label_bbox']
                        if named:
                            placed = label.get('bbox') == c['label_bbox']
                        else:
                            placed = (label.get('metadata',{}).get('bbox') == c['label_bbox']
                                      and label.get('x') == x+w/2 and label.get('y') == y+h/2
                                      and label.get('anchor','center') == 'center')
                        if not placed: errors.append('branch_label_position_mismatch')
            row.update(scene_mapping='conflict' if errors else 'verified', mapping_errors=errors)
            check('connection_mapping', row['scene_mapping'], [c['id']], 'ID, directed endpoints, anchors, explicit path and label mapping only.', errors=errors)
        report['connections'].append(row)
    for ident in sorted(set(expected_edges)-{c['id'] for c in ir['connections']}):
        check('missing_reference_connection','conflict',[ident],'Explicit expected relation absent.')
    overlaps = {frozenset(o['ids']): o for o in ref.get('overlaps', [])}
    distinct_shapes = {p['shape_id'] for p in ref.get('pairs', [])}
    values = list(elements.values())
    shapes = SHAPE_TYPES
    for i,a in enumerate(values):
        for b in values[i+1:]:
            area = _intersection(a['bbox'], b['bbox'])
            if not area: continue
            ids = [a['id'],b['id']]; proof = overlaps.get(frozenset(ids))
            duplicate = a['bbox']==b['bbox']
            containment = ((a['type'] in shapes and _contains(a['bbox'],b['bbox']))
                           or (b['type'] in shapes and _contains(b['bbox'],a['bbox'])))
            ratio = area/min(a['bbox'][2]*a['bbox'][3], b['bbox'][2]*b['bbox'][3])
            if proof and proof['classification']=='legal_overlap':
                row = check('legal_overlap','verified',ids,proof.get('basis',ref['basis']))
            elif proof and proof['classification']=='duplicate_detection':
                row = check('duplicate_detection','conflict',ids,proof.get('basis',ref['basis']))
            elif duplicate:
                row = check('bbox_collision','conflict' if set(ids)<=distinct_shapes else 'unverified', ids,
                            'Equal boxes; distinct reviewed sequence members.' if set(ids)<=distinct_shapes else 'Equal boxes alone do not prove duplicate objects or illegal overlap.')
            elif containment:
                continue  # containment is normal; do not infer semantic ownership
            elif ratio>=.65:
                row = check('large_overlap','unverified',ids,'Large overlap is a review candidate, not proof of an error.', ratio=ratio)
            else: continue
            report['collisions'].append(row)
    used_shapes, used_texts, positions, orders = set(), set(), [], []
    for pair in ref.get('pairs', []):
        s,t = elements.get(pair['shape_id']), elements.get(pair['text_id'])
        errors = []
        if s is None or t is None: errors.append('unknown_pair_reference')
        else:
            if s['type'] not in shapes or t['type'] not in TEXT_TYPES:
                errors.append('invalid_pair_element_type')
            if s['id'] in used_shapes or t['id'] in used_texts: errors.append('non_bijective_pairing')
            used_shapes.add(s['id']); used_texts.add(t['id'])
            if t.get('raw_text')!=pair['text']: errors.append('text_pair_mismatch')
            if s['bbox']!=pair['shape_bbox']: errors.append('shape_measurement_mismatch')
            if t['bbox']!=pair['text_bbox']: errors.append('text_measurement_mismatch')
            if not _contains(s['bbox'],t['bbox']): errors.append('text_outside_paired_container')
            positions.append(s['bbox'][ref.get('sequence_axis',0)])
            orders.append(t.get('reading_order',-1))
            if scene is not None:
                for element in (s,t):
                    n=nodes.get(element['id'])
                    if n is None or n['bbox']!=element['bbox'] or ('raw_text' in element and n.get('text')!=element['raw_text']):
                        errors.append('pair_scene_mapping_mismatch')
        row=check('element_pairing','conflict' if errors else 'verified',[pair['shape_id'],pair['text_id']],ref['basis'],errors=errors)
        report['pairings'].append(row)
    if ref.get('pairs'):
        good=(len(positions) == len(ref['pairs']) and all(row['status']=='verified' for row in report['pairings'])
              and all(a<b for a,b in zip(positions,positions[1:])) and all(a<b for a,b in zip(orders,orders[1:])))
        report['pairings'].append(check('sequence_order','verified' if good else 'conflict',[p['shape_id'] for p in ref['pairs']],'Reference sequence order, spatial order and recorded reading order; never numeric token magnitude.'))
    else:
        report['pairings'].append(check('element_pairing','not_applicable',[],'No explicit sequence or pairing reference supplied.'))
    conflicts=any(c['status']=='conflict' for c in report['checks'])
    report['semantic_status']='partial' if conflicts else 'unverified'
    report['visual_status']='partial' if any(c['status']!='verified' for c in report['collisions']) else 'unverified'
    return report


def repair_reviewed_pairings(ir, reference, *, image_sha256):
    """Atomic group bbox repair using the existing geometry adapter. No graph edits."""
    if reference is None:
        return deepcopy(ir), {'status':'unverified','reason':'No reviewed evidence.','changes':[]}
    ref = _reference(ir, reference, image_sha256)
    before = inspect_consistency(ir, reference=ref, image_sha256=image_sha256)
    result = deepcopy(ir)
    if not before['ir_validation']['valid'] or not ref.get('pairs'):
        return result, {'status':'unverified','reason':'Valid input and explicit reviewed pairings required.','changes':[]}
    elements={e['id']:e for e in ir['elements']}; observations=[]; used=set()
    try:
        for pair in ref['pairs']:
            s,t = elements[pair['shape_id']],elements[pair['text_id']]
            if (s['id'] == t['id'] or s['type'] not in SHAPE_TYPES or t['type'] not in TEXT_TYPES
                    or s['id'] in used or t['id'] in used or t.get('raw_text')!=pair['text']
                    or not _contains(pair['shape_bbox'],pair['text_bbox'])):
                raise ValueError('Pairing must be one-to-one and preserve reviewed text inside its container.')
            used.update((s['id'],t['id']))
            for item,box in ((s,pair['shape_bbox']),(t,pair['text_bbox'])):
                observations.append({'element_id':item['id'],'expected_bbox':item['bbox'],'expected_region_id':item['region_id'],
                     'action':'measure_bbox','measured_bbox':box,'confirmed':True,'basis':ref['basis']})
        evidence={'image_sha256':image_sha256,'ir_sha256':ir_fingerprint(ir),'coordinate_system':ir['canvas']['coordinate_system'],
                  'canvas_size':[ir['canvas']['width'],ir['canvas']['height']],'observations':observations}
        repaired, report=repair_geometry(ir,evidence,image_sha256=image_sha256)
        if report['status'] not in ('repaired','unchanged'): raise ValueError('Geometry adapter rejected part of the group; all changes rolled back.')
        rebound=dict(ref,ir_sha256=ir_fingerprint(repaired))
        after=inspect_consistency(repaired,reference=rebound,image_sha256=image_sha256)
        if any(row['status']=='conflict' for row in after['pairings']): raise ValueError('Paired sequence still conflicts; atomic rollback.')
        old={(c['code'],tuple(c['ids'])) for c in before['collisions']}
        if any((c['code'],tuple(c['ids'])) not in old and c['status']!='verified' for c in after['collisions']):
            raise ValueError('New geometric collision; atomic rollback.')
        return repaired, report
    except (KeyError,ValueError,TypeError) as exc:
        return result, {'status':'unverified','reason':str(exc),'changes':[]}
