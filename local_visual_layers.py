"""Apply measured appearance without changing recognized text or graph topology.

Visual patches are independent editable IR elements, not new semantic nodes.
Containment determines fill order; the opt-in Scene renderer paints borders,
connectors and text separately so later child fills cannot erase them.
"""
from copy import deepcopy
import math
import re

from structured_ir.validator import validate_ir


SHAPES = {'rectangle', 'rounded_rectangle', 'circle', 'diamond', 'title_band'}
TEXT = {'text', 'multiline_text', 'code', 'math'}


def _area(box):
    return box[2] * box[3]


def _contains(outer, inner):
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[0]+outer[2] >= inner[0]+inner[2]
            and outer[1]+outer[3] >= inner[1]+inner[3])


def _intersection(a, b):
    return (max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
            * max(0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1])))


def _box(value, width, height):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)):
        raise ValueError('Measured visual bbox must contain four finite pixel coordinates.')
    x,y,w,h=value
    if x<0 or y<0 or w<=0 or h<=0 or x+w>width or y+h>height:
        raise ValueError('Measured visual bbox lies outside source canvas.')
    return list(value)


def _unique_owner(box, regions):
    parents=[r for r in regions if _contains(r['bbox'],box)]
    deepest=[p for p in parents if all(p['id']==q['id'] or
             (_area(q['bbox'])>_area(p['bbox']) and _contains(q['bbox'],p['bbox'])) for q in parents)]
    return (deepest[0]['id'] if len(deepest)==1 else None),[p['id'] for p in parents]


def _same_boundary(a,b):
    overlap=_intersection(a,b)
    if not overlap: return False
    edges=lambda v:(v[0],v[1],v[0]+v[2],v[1]+v[3])
    return (overlap/(_area(a)+_area(b)-overlap)>=.96
            and max(abs(x-y) for x,y in zip(edges(a),edges(b)))<=2)


def apply_visual_layers(ir, fusion, style_map, style_report):
    """Return independent (styled IR, unchanged role map, layer diagnostics).

The input must already be valid. Measured styles override legacy role defaults;
unknown/missing measurements retain defaults. Unknown colors are not guessed.
Existing element IDs, boxes, ownership, text and connections remain intact.
"""
    validate_ir(ir)
    if not all(isinstance(v,dict) for v in (fusion,style_map,style_report)):
        raise ValueError('Fusion, style map and visual report must be objects.')
    result, styles = deepcopy(ir),deepcopy(style_map)
    width,height=result['canvas']['width'],result['canvas']['height']
    image=style_report.get('image',{})
    if image:
        if image.get('width')!=width or image.get('height')!=height:
            raise ValueError('Visual measurement dimensions differ from the IR canvas.')
        expected=ir.get('provenance',{}).get('image_sha256')
        if expected and image.get('sha256')!=expected:
            raise ValueError('Visual measurement source hash differs from the IR.')
    report={'status':'partial','coordinate_system':'top-left','bbox_format':'xywh',
        'style_updates':[],'added_elements':[],'matched_visual_regions':[],
        'containment':[],'overlaps':[],'diagnostics':[],
        'policy':'Measured fill containment order; separate border pass; edges and text above fills.',
        'semantic_connections_modified':False,'api_calls':0}
    by_id={e['id']:e for e in result['elements']}
    used={result['canvas']['id']} | {v['id'] for k in ('regions','elements','connections') for v in result[k]}
    roles={}

    def update(element, measurement, source):
        raw=measurement.get('visual_style',{})
        if not isinstance(raw,dict):
            raise ValueError('Measured visual_style must be an object.')
        if raw:
            element.setdefault('visual_style',{}).update(deepcopy(raw))
            report['style_updates'].append({'element_id':element['id'],'source_id':source,
                'fields':sorted(raw),'evidence':deepcopy(measurement.get('evidence',{}))})
        radius=raw.get('rounded_radius')
        if (element['type']=='rectangle' and measurement.get('type')=='rounded_rectangle'
                and type(radius) in (int,float) and radius>0):
            element['type']='rounded_rectangle'
            report['diagnostics'].append({'code':'measured_rounding','element_id':element['id'],
                'message':'Kept original bbox; positive measured radius enables existing rounded primitive.'})
        roles[element['id']]=measurement.get('role_candidate','container_background')

    recovered_shapes=[]
    region_sources={m['source_id']:m['region_id'] for m in fusion.get('region_matches',[])}
    for measured in style_report.get('shapes',[]):
        source=measured['source_id']; region=region_sources.get(source)
        candidates=[e for e in result['elements'] if e['type'] in SHAPES and e['region_id']==region] if region else []
        candidates=[e for e in candidates if e['bbox']==measured['bbox']]
        if len(candidates)==1:
            update(candidates[0],measured,source)
        elif (not candidates and measured.get('type') in {'rectangle','rounded_rectangle','title_band'}
                and 'fill_color' in measured.get('visual_style',{})):
            # A formerly unknown contour can acquire independent visual evidence
            # without renaming any prior element or altering semantic topology.
            recovered_shapes.append(dict(deepcopy(measured),id='visual_shape_'+source))
        else:
            report['diagnostics'].append({'code':'unmapped_shape_style','source_id':source,
                'message':'No unique existing drawable element; measurement retained without invented replacement.'})
    source_text={m['source_id']:m['element_id'] for m in fusion.get('text_matches',[])}
    for source,measured in style_report.get('text_styles',{}).items():
        ident=source_text.get(source)
        if ident in by_id and by_id[ident]['type'] in TEXT:
            # The visual text sampler may change only ink appearance.
            raw=measured.get('visual_style',{})
            if set(raw)-{'text_color_rgb'}:
                raise ValueError('OCR visual styles may only set measured text_color_rgb.')
            update(by_id[ident],measured,source)
            roles[ident]='text'
        else:
            report['diagnostics'].append({'code':'unmapped_text_style','source_id':source})

    def add_patch(measured, page=False):
        box=_box(measured['bbox'],width,height)
        raw=measured.get('visual_style',{})
        if not isinstance(raw,dict): raise ValueError('Measured visual_style must be an object.')
        if 'fill_color' not in raw and raw.get('fill','none')=='none':
            report['diagnostics'].append({'code':'uncertain_fill_not_added',
                'source_id':measured.get('id','page_background'),'evidence':deepcopy(measured.get('evidence',{}))})
            return
        source='page_background' if page else measured['id']
        kind=measured.get('type','rectangle')
        if kind not in {'rectangle','rounded_rectangle','title_band'}:
            raise ValueError('Visual patches must use measured editable rectangular primitives.')
        # Reuse an already measured shape boundary instead of repainting duplicate
        # borders. The historical element ID and bbox remain authoritative.
        matches=[e for e in result['elements'] if e['type'] in {'rectangle','rounded_rectangle','title_band'}
                 and _same_boundary(e['bbox'],box)] if not page else []
        if len(matches)==1:
            update(matches[0],measured,source)
            report['matched_visual_regions'].append({'source_id':source,'element_id':matches[0]['id'],
                'measured_bbox':box,'retained_bbox':deepcopy(matches[0]['bbox'])})
            return
        if not isinstance(source,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,48}',source):
            raise ValueError('Visual source IDs must be short ASCII identifiers.')
        ident='local_visual_'+source
        if ident in used:
            raise ValueError('Duplicate visual element ID: '+ident)
        used.add(ident)
        owner,candidates=(None,[]) if page else _unique_owner(box,result['regions'])
        item={'id':ident,'type':kind,'bbox':box,'region_id':owner,
              'visual_style':deepcopy(raw),
              'uncertainty':'Measured visual background candidate; not a semantic node or endpoint.'}
        result['elements'].append(item);by_id[ident]=item
        roles[ident]='page_background' if page else measured.get('role_candidate','background_fill')
        report['added_elements'].append({'element_id':ident,'source_id':source,'bbox':box,
            'region_id':owner,'role_candidate':roles[ident],'containing_regions':candidates})
        if owner is None and candidates:
            report['diagnostics'].append({'code':'ambiguous_patch_owner','element_id':ident,
                'candidates':candidates,'message':'No unique innermost container; unassigned ownership retained.'})

    page=style_report.get('page_background')
    if page:
        if page['bbox']!=[0,0,width,height]:
            raise ValueError('Page background must describe the original full canvas.')
        add_patch(page,True)
    for measured in recovered_shapes:
        add_patch(measured)
    for measured in style_report.get('regions',[]):
        add_patch(measured)

    # A strict-area containment graph is acyclic. Every ancestor has fewer
    # ancestors than its descendants, so a stable topological order follows.
    backgrounds=[e for e in result['elements'] if e['type'] in SHAPES]
    ancestors={e['id']:[p['id'] for p in backgrounds if p is not e and
               _area(p['bbox'])>_area(e['bbox']) and _contains(p['bbox'],e['bbox'])] for e in backgrounds}
    source_roles=deepcopy(roles)
    for element in backgrounds:
        ident=element['id']
        if roles.get(ident)=='page_background': continue
        parents=[by_id[p] for p in ancestors[ident] if roles.get(p)!='page_background']
        parent=min(parents,key=lambda p:_area(p['bbox'])) if parents else None
        raw_role=roles.get(ident,'container_background')
        if raw_role in {'title_band_candidate','title_band'} and parent:
            x,y,w,h=element['bbox'];px,py,pw,ph=parent['bbox']
            # Thin/wide alone is not a title: it must occupy the parent's top.
            roles[ident]=('title_band' if h<.45*ph and w>=.6*pw and y-py<=max(6,.08*ph)
                          else 'body_fill')
        elif raw_role in {'filled_region','background_fill','title_band_candidate'}:
            roles[ident]='body_fill' if parent else 'main_container'
        elif raw_role=='container_background':
            roles[ident]='child_card' if parent else 'main_container'
    rank={'page_background':0,'main_container':1,'container_background':2,
          'child_card':3,'background_fill':4,'body_fill':5,'title_band':6}
    backgrounds.sort(key=lambda e:(len(ancestors[e['id']]),-_area(e['bbox']),
                     rank.get(roles.get(e['id']),3),e['bbox'][1],e['bbox'][0],e['id']))
    for index,e in enumerate(backgrounds):
        e['z_index']=index
        report['containment'].append({'element_id':e['id'],'ancestor_ids':ancestors[e['id']],
            'z_index':index,'role_candidate':roles.get(e['id'],'container_background'),
            'source_role_candidate':source_roles.get(e['id'],'container_background')})
    for index,a in enumerate(backgrounds):
        for b in backgrounds[index+1:]:
            overlap=_intersection(a['bbox'],b['bbox'])
            if overlap and not (_contains(a['bbox'],b['bbox']) or _contains(b['bbox'],a['bbox'])):
                report['overlaps'].append({'element_ids':[a['id'],b['id']],
                    'intersection_area':overlap,'status':'unresolved_occlusion',
                    'policy':'Stable measured-background order only; no semantic occlusion assertion.'})
    offset=len(backgrounds)
    for e in result['elements']:
        if e['type'] in {'connector','arrow'}:
            e['z_index']=offset
        elif e['type'] in TEXT:
            e['z_index']=offset+1
    # No new region/ownership decisions for historical elements are made.
    for region in result['regions']:
        region['children']=[r['id'] for r in result['regions'] if r['parent_id']==region['id']]
        region['children'] += [e['id'] for e in result['elements'] if e['region_id']==region['id']]
    result['layered']=True
    validate_ir(result)
    report['paint_order']={'fills':[e['id'] for e in backgrounds],
        'borders':[e['id'] for e in backgrounds],
        'connections':[e['id'] for e in result['elements'] if e['type'] in {'connector','arrow'}]+[c['id'] for c in result['connections']],
        'text':[e['id'] for e in result['elements'] if 'raw_text' in e]}
    report['counts']={'original_elements':len(ir['elements']),'styled_elements':len(report['style_updates']),
        'new_visual_elements':len(result['elements'])-len(ir['elements']),
        'preserved_text_elements':sum('raw_text' in e for e in ir['elements']),
        'semantic_connections':len(result['connections']),'unresolved_overlaps':len(report['overlaps'])}
    report['preservation']={'existing_ids':True,'existing_bboxes':True,'existing_text':True,
        'existing_ownership':True,'connections':result['connections']==ir['connections']}
    return result,styles,report
