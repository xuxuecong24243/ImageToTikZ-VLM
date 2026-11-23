"""Bounded structure-only experiment. No API, OCR, schema or renderer changes.

Model output supplies hypotheses, never pixels or drawing coordinates. Existing
IR objects remain byte-equivalent data; new arrows require independent measured
head and shaft evidence in the original image. Unsupported candidates are records.
"""
from copy import deepcopy
import hashlib
import json
import math
import re
from pathlib import Path

from structured_ir.validator import inspect_ir, validate_ir, _walk

ANCHORS = {'east', 'west', 'north', 'south'}
SHAPES = {'rectangle', 'rounded_rectangle', 'circle', 'diamond', 'title_band'}
TEXTS = {'text', 'multiline_text', 'code', 'math'}


def structure_schema():
    text = {'type': 'string', 'minLength': 1, 'maxLength': 2000}
    ident = {'type': 'string', 'pattern': r'^[A-Za-z][A-Za-z0-9_]{0,63}$'}
    ids = {'type': 'array', 'items': ident, 'maxItems': 200}
    confidence = {'type': 'number', 'minimum': 0, 'maximum': 1}
    def obj(fields):
        return {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}
    def array(item):
        return {'type': 'array', 'items': item, 'maxItems': 200}
    return obj({
        'groups': array(obj({'group_id': ident, 'element_ids': ids,
            'container_id': {'type': ['string', 'null'], 'pattern': ident['pattern']},
            'relation': {'enum': ['contains']}, 'evidence': text, 'confidence': confidence})),
        'connections': array(obj({'source_id': ident, 'target_id': ident,
            'source_anchor': {'enum': sorted(ANCHORS)}, 'target_anchor': {'enum': sorted(ANCHORS)},
            'relation': {'enum': ['data_flow', 'residual']}, 'line_ids': ids,
            'evidence': text, 'confidence': confidence})),
        'candidates': array(obj({'type': {'enum': ['dashed_enclosure', 'circle_plus', 'curve', 'missing_shape', 'other']},
            'related_element_ids': ids, 'evidence': text, 'uncertainty': text})),
        'unresolved': array(text)})


def _check_structure(value):
    errors = []
    from structured_ir.validator import _json_safety
    _json_safety(value, '', errors)
    if not errors:
        _walk(value, structure_schema(), '', {}, errors)
    if errors:
        first = errors[0]
        raise ValueError('{}: {} ({})'.format(first['path'], first['message'], first['code']))


def parse_structure(text):
    """Strict JSON or one complete JSON fence; no repair, substring or coercion."""
    if not isinstance(text, str) or len(text) > 2_000_000:
        raise ValueError('Structure response must be text within the 2 MB limit.')
    value = text.strip()
    fence = re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```', value)
    if fence:
        value = fence.group(1)
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = item
        return result
    result = json.loads(value, object_pairs_hook=pairs,
                        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON number.')))
    _check_structure(result)
    return result


def _contains(outer, inner):
    return (inner[0] >= outer[0] and inner[1] >= outer[1] and
            inner[0]+inner[2] <= outer[0]+outer[2] and inner[1]+inner[3] <= outer[1]+outer[3])


def _face(item, anchor):
    x,y,w,h = item['bbox']
    return {'east': (0,1,x+w,y,y+h), 'west': (0,-1,x,y,y+h),
            'south': (1,1,y+h,x,x+w), 'north': (1,-1,y,x,x+w)}[anchor]


def _corridor(source, target, proposal, width, height):
    axis, sign, start, slo, shi = _face(source, proposal['source_anchor'])
    ta, ts, end, tlo, thi = _face(target, proposal['target_anchor'])
    if axis != ta or sign != -ts or (end-start)*sign <= 3:
        return None, 'nonfacing_or_overlapping_endpoints'
    lo, hi = max(slo,tlo)+1, min(shi,thi)-1
    if hi-lo < 6:
        return None, 'insufficient_boundary_projection_overlap'
    amin, amax = min(start,end)+1, max(start,end)-1
    # Do not scan broad page interiors or create a route through a module.
    if amax-amin > max(width,height)*.4:
        return None, 'endpoint_gap_exceeds_experiment_scope'
    box = [amin,lo,amax,hi] if axis == 0 else [lo,amin,hi,amax]
    box = [max(0,math.floor(box[0])),max(0,math.floor(box[1])),
           min(width,math.ceil(box[2])),min(height,math.ceil(box[3]))]
    if (box[2]-box[0])*(box[3]-box[1]) > 1_000_000:
        return None,'corridor_resource_limit'
    return (box,axis,sign,start,end), None


def _crosses_text(points, elements):
    for e in elements:
        if e['type'] not in TEXTS:
            continue
        x,y,w,h = e['bbox']
        for a,b in zip(points,points[1:]):
            count=max(2,math.ceil(math.dist(a,b)))
            for step in range(count+1):
                px=a[0]+(b[0]-a[0])*step/count;py=a[1]+(b[1]-a[1])*step/count
                if x < px < x+w and y < py < y+h:
                    return e['id']
    return None


def _local_arrows(pixels, corridor):
    """Reuse v0.5 head/shaft algorithms on the proposed facing-boundary gap."""
    from geometry_analyzer import _dependencies
    from local_arrow_detection import _heads, _shaft_path
    cv2,np = _dependencies()
    box,axis,sign,start,end = corridor
    x0,y0,x1,y1 = box
    crop=pixels[y0:y1,x0:x1]
    if not crop.size:
        return []
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    masks=[('chromatic',cv2.inRange(hsv,np.array([0,40,25]),np.array([179,255,255]))),
           ('neutral',((hsv[:,:,1]<40)&(hsv[:,:,2]<155)).astype('uint8')*255)]
    found=[]
    for method,mask in masks:
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        for label in range(1,n):
            x,y,w,h,area=map(int,stats[label])
            if max(w,h)<10 or min(w,h)<5 or area<20:
                continue
            component=(labels[y:y+h,x:x+w]==label).astype('uint8')
            heads=_heads(component,cv2,np)
            if len(heads)!=1:
                continue
            head=heads[0]
            if head['axis']!=axis or head['sign']!=sign:
                continue
            route,reason=_shaft_path(component,head,cv2,np)
            if route is None:
                continue
            points=[[float(px+x+x0),float(py+y+y0)] for px,py in route]
            allowed_gap=max(8,head['width_px']*1.5)
            if abs(points[0][axis]-start)>allowed_gap or abs(points[-1][axis]-end)>allowed_gap:
                continue
            rgb=np.median(crop[y:y+h,x:x+w][component>0][:,::-1],axis=0).round().astype(int).tolist()
            found.append({'points':points, 'style':{'stroke_color':'#'+''.join('%02X'%v for v in rgb),
                'line_width':max(.1,min(10,float(head['shaft_width_px'])))},
                'evidence':{'method':'vlm_endpoint_guided_original_pixel_'+method,
                    'corridor_xyxy':box, 'head_fill_support':float(head['fill_support']),
                    'head_width_px':float(head['width_px']), 'shaft_width_px':float(head['shaft_width_px']),
                    'source_gap_px':abs(points[0][axis]-start),'target_gap_px':abs(points[-1][axis]-end),
                    'direction_measured':True}})
    unique=[]
    for a in found:
        if not any(math.dist(a['points'][0],b['points'][0])<3 and math.dist(a['points'][-1],b['points'][-1])<3 for b in unique):
            unique.append(a)
    return unique


def _distance_to_segment(point,a,b):
    d=[b[0]-a[0],b[1]-a[1]];length=d[0]**2+d[1]**2
    if not length:return math.dist(point,a)
    t=max(0,min(1,((point[0]-a[0])*d[0]+(point[1]-a[1])*d[1])/length))
    return math.dist(point,[a[0]+t*d[0],a[1]+t*d[1]])


def _overlapping_line(points,elements):
    for e in elements:
        if e['type'] not in {'connector','arrow'}:
            continue
        pts=e['points']
        samples=[]
        for a,b in zip(points,points[1:]):
            n=max(1,math.ceil(math.dist(a,b)))
            samples += [[a[0]+(b[0]-a[0])*i/n,a[1]+(b[1]-a[1])*i/n] for i in range(n)]
        if samples and sum(min(_distance_to_segment(p,a,b) for a,b in zip(pts,pts[1:]))<2 for p in samples)/len(samples)>.7:
            return e['id']
    return None


def fuse_structure(ir, supplement, *, image_path, arrow_report):
    """Return a new validated IR and explicit per-proposal evidence/diagnostics.

    No semantic proposal is permission to invent geometry. Groups annotate the
    existing objects only; new graph edges need an original-pixel filled head and
    continuous shaft. Complicated routes and missing shapes stay unverified.
    """
    validate_ir(ir)
    _check_structure(supplement)
    from geometry_analyzer import _load_image
    pixels,image=_load_image(image_path)
    if (image['width'],image['height']) != (ir['canvas']['width'],ir['canvas']['height']):
        raise ValueError('Source image and IR canvas dimensions differ.')
    expected=ir.get('provenance',{}).get('image_sha256')
    if expected and image['sha256']!=expected:
        raise ValueError('Source image hash differs from local IR provenance.')
    reported=(arrow_report or {}).get('image',{}).get('sha256')
    if reported and reported != image['sha256']:
        raise ValueError('Arrow evidence comes from a different image.')
    merged=deepcopy(ir);elements={e['id']:e for e in ir['elements']}
    report={'status':'partial','api_calls':0,'groups':[],'connections':[],'candidates':[],
            'unresolved':deepcopy(supplement['unresolved']), 'image':image,
            'policy':'Existing objects preserved; model candidates are hypotheses; groups metadata only; new paths measured locally.',
            'limits':['contains_verified verifies geometric containment only, not model semantics.',
                'line_ids are audited existing geometry, not reconstructed into complex residual routes.',
                'Residual is an unverified semantic label; a measured straight arrow does not prove a full residual branch.'],
            'diagnostics':[]}
    valid_groups=[];seen_groups=set()
    for n,g in enumerate(supplement['groups']):
        entry={'proposal_index':n,'proposal':deepcopy(g),'status':'unverified'}
        ids=g['element_ids'];container=g['container_id']
        if g['group_id'] in seen_groups:
            entry.update(status='rejected',reason='duplicate_group_id')
        elif not ids or len(ids)!=len(set(ids)) or any(i not in elements for i in ids) or (container is not None and container not in elements):
            entry.update(status='rejected',reason='unknown_empty_or_duplicate_element_reference')
        elif g['confidence']<.8:
            entry['reason']='low_model_confidence'
        elif container is not None and (container in ids or elements[container]['type'] not in SHAPES or not all(_contains(elements[container]['bbox'],elements[i]['bbox']) for i in ids)):
            entry.update(status='rejected',reason='containment_conflict')
        else:
            entry.update(status='contains_verified' if container else 'semantic_group_unverified',
                         reason='metadata_only_no_drawable_geometry_added')
            valid_groups.append(deepcopy(g))
        seen_groups.add(g['group_id']);report['groups'].append(entry)
    # Reject cycles in the proposed element-containment graph without changing old regions.
    parents={}
    for g in valid_groups:
        if g['container_id']:
            parents.setdefault(g['container_id'],set()).update(g['element_ids'])
    def cyclic(start,current,seen):
        return any(child==start or (child not in seen and cyclic(start,child,seen|{child}))
                   for child in parents.get(current,()))
    bad={p for p in parents if cyclic(p,p,{p})}
    if bad:
        valid_groups=[g for g in valid_groups if g['container_id'] not in bad]
        for entry in report['groups']:
            if entry['proposal']['container_id'] in bad:
                entry.update(status='rejected',reason='containment_cycle')
    semantic=[]
    for n,p in enumerate(supplement['connections']):
        entry={'proposal_index':n,'proposal':deepcopy(p),'status':'unverified'}
        report['connections'].append(entry)
        source=elements.get(p['source_id']);target=elements.get(p['target_id'])
        if source is None or target is None or any(i not in elements or elements[i]['type'] not in {'connector','arrow'} for i in p['line_ids']):
            entry.update(status='rejected',reason='unknown_or_invalid_element_reference');continue
        if p['source_id']==p['target_id'] or source['type'] not in SHAPES or target['type'] not in SHAPES:
            entry.update(status='rejected',reason='unsupported_endpoint_type_or_self_connection');continue
        if len(set(p['line_ids']))!=len(p['line_ids']):
            entry.update(status='rejected',reason='duplicate_line_reference');continue
        existing=[c for c in merged['connections'] if c['source_id']==p['source_id'] and c['target_id']==p['target_id']]
        reverse=[c for c in merged['connections'] if c['source_id']==p['target_id'] and c['target_id']==p['source_id']]
        if existing:
            entry.update(status='already_present',connection_ids=[c['id'] for c in existing]);continue
        if reverse:
            entry.update(status='rejected',reason='conflicts_with_existing_direction');continue
        if p['confidence']<.8:
            entry['reason']='low_model_confidence';continue
        corridor,reason=_corridor(source,target,p,image['width'],image['height'])
        if corridor is None:
            entry['reason']=reason;continue
        # Line IDs are audited evidence, not permission to repaint old lines or
        # trust model-supplied coordinates. Complex routes remain explicit partial.
        if p['line_ids']:
            entry['referenced_measured_lines']=[deepcopy(elements[i]) for i in p['line_ids']]
        measured=_local_arrows(pixels,corridor)
        entry['measured_arrow_candidates']=measured
        if len(measured)!=1:
            entry['reason']='no_supported_pixel_arrow' if not measured else 'ambiguous_pixel_arrows';continue
        a=measured[0];crossing=_crosses_text(a['points'],ir['elements'])
        if crossing:
            entry.update(reason='path_crosses_existing_text',element_id=crossing);continue
        overlaps=_overlapping_line(a['points'],ir['elements'])
        if overlaps:
            entry.update(status='semantic_only',reason='existing_geometry_not_duplicated',element_id=overlaps)
            semantic.append(deepcopy(p));continue
        if any(math.dist(a['points'][0],c.get('points',[[float('inf'),0]])[0])<3 and
               math.dist(a['points'][-1],c.get('points',[[float('inf'),0]])[-1])<3 for c in merged['connections']):
            entry.update(status='already_present',reason='same_measured_route');continue
        ident='hybrid_flow_'+hashlib.sha256(json.dumps([p['source_id'],p['target_id'],a['points']],sort_keys=True).encode()).hexdigest()[:16]
        used={o['id'] for k in ('elements','regions','connections') for o in merged[k]}|{merged['canvas']['id']}
        if ident in used:
            entry.update(status='rejected',reason='generated_identifier_collision');continue
        connection={'id':ident,'source_id':p['source_id'],'target_id':p['target_id'],
            'source_anchor':p['source_anchor'],'target_anchor':p['target_anchor'], 'direction':'forward',
            'kind':'flow','points':deepcopy(a['points']),'visual_style':deepcopy(a['style']),
            'confidence':min(.85,p['confidence']), 'uncertainty':'Model semantic interpretation; original-pixel head and shaft verified locally.'}
        merged['connections'].append(connection)
        entry.update(status='fused',connection_id=ident,relation=p['relation'],reason='original_pixel_head_shaft_and_facing_boundaries')
    for n,c in enumerate(supplement['candidates']):
        valid=all(i in elements for i in c['related_element_ids'])
        report['candidates'].append({'proposal_index':n,'proposal':deepcopy(c),
            'status':'unverified' if valid else 'rejected',
            'reason':'No validated new geometry is created from model descriptions.' if valid else 'unknown_element_reference',
            'representation':{'curve':'No native IR/Scene curve primitive.',
                'dashed_enclosure':'Scene edges support dashed; IR has no dashed field; no measured enclosure to convert.',
                'circle_plus':'Existing circle and text can render; missing glyph/shape needs measured local evidence.'}.get(c['type'],'Not independently verified.')})
    merged.setdefault('provenance',{})['hybrid_structure_experiment']={
        'groups':valid_groups,'semantic_only_connections':semantic,
        'candidates':deepcopy(report['candidates']),'unresolved':deepcopy(supplement['unresolved']),
        'status':'unverified_semantics_with_locally_measured_new_edges'}
    report['preservation']={'elements_equal':merged['elements']==ir['elements'],
        'regions_equal':merged['regions']==ir['regions'],
        'canvas_equal':merged['canvas']==ir['canvas'],
        'existing_connections_equal':merged['connections'][:len(ir['connections'])]==ir['connections']}
    report['ir_validation']=inspect_ir(merged)
    report['counts']={'groups_recorded':len(valid_groups),'connections_added':len(merged['connections'])-len(ir['connections']),
        'candidates_unverified':sum(c['status']=='unverified' for c in report['candidates']),
        'connections_unverified':sum(c['status']=='unverified' for c in report['connections']),
        'rejected_proposals':sum(c['status']=='rejected' for k in ('groups','connections','candidates') for c in report[k])}
    if not report['ir_validation']['valid'] or not all(report['preservation'].values()):
        report['status']='failed';return None,report
    return merged,report
