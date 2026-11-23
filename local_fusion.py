"""Deterministic OCR/geometry fusion; measured layout is not semantic topology.

OCR strings and boxes remain separate drawable text elements. Multiline groups
are reviewable annotations, so grouping cannot reflow, rewrite or drop a line.
"""
from copy import deepcopy
import math

from structured_ir.validator import inspect_ir
from structured_tikz.schema import COLORS, STYLE_LIMITS

SHAPES = {'rectangle', 'rounded_rectangle', 'circle', 'diamond', 'title_band'}


def _box(box, width, height):
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)):
        raise ValueError('Expected finite bbox [left, top, width, height].')
    x,y,w,h = box
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > width or y+h > height:
        raise ValueError('Detected bbox lies outside the source image; no silent clipping or text removal.')
    return list(box)


def _contains(a,b):
    return a[0] <= b[0] and a[1] <= b[1] and a[0]+a[2] >= b[0]+b[2] and a[1]+a[3] >= b[1]+b[3]


def _area(box):
    return box[2]*box[3]


def _intersection(a,b):
    return max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0])) * max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))


def _owner(box, regions, *, exclude=None):
    candidates = [r for r in regions if r['id'] != exclude and _contains(r['bbox'],box)
                  and (exclude is None or _area(r['bbox']) > _area(box))]
    # A smallest box is insufficient: it must be nested in every other candidate.
    deepest = [r for r in candidates if all(r['id']==s['id'] or
               (_contains(s['bbox'],r['bbox']) and _area(s['bbox'])>_area(r['bbox'])) for s in candidates)]
    return (deepest[0]['id'] if len(deepest)==1 else None), [r['id'] for r in candidates]


def _style(value):
    if not isinstance(value, dict):
        raise ValueError('Geometry style must be an object.')
    result = {}
    for key, item in value.items():
        if key in ('fill','stroke','text_color'):
            if item not in set(COLORS) | ({'none'} if key != 'text_color' else set()):
                raise ValueError('Geometry styles must use the existing Scene palette.')
        elif key in STYLE_LIMITS:
            lo,hi = STYLE_LIMITS[key]
            if isinstance(item,bool) or not isinstance(item,(int,float)) or not math.isfinite(item) or not lo <= item <= hi:
                raise ValueError('Geometry style outside existing Scene bounds.')
        elif key == 'bold':
            if not isinstance(item,bool): raise ValueError('bold must be boolean.')
        else:
            raise ValueError('Unknown geometry style property: '+key)
        result[key] = item
    return result


def _segment_hits_box(points, box):
    """Continuous segment/box clipping; no sampled gaps through OCR glyph boxes."""
    a,b = points; lower,upper = 0.,1.
    for axis in (0,1):
        delta = b[axis]-a[axis]; minimum=box[axis]; maximum=minimum+box[axis+2]
        if delta == 0:
            if not minimum <= a[axis] <= maximum: return False
        else:
            lo,hi = sorted(((minimum-a[axis])/delta,(maximum-a[axis])/delta))
            lower,upper=max(lower,lo),min(upper,hi)
            if lower > upper: return False
    return True


def build_local_ir(ocr, geometry):
    """Return (validated IR, partial fusion report, converter style_map).

Both inputs must bind the same source image. IDs come solely from this builder,
and children are derived after ownership is fixed. No semantic edges are guessed.
"""
    if not isinstance(ocr,dict) or not isinstance(geometry,dict):
        raise ValueError('OCR and geometry must be objects.')
    image = ocr.get('image',{})
    width,height = image.get('width'),image.get('height')
    if any(type(v) is not int or not 1 <= v <= 20000 for v in (width,height)):
        raise ValueError('Source canvas dimensions must be positive integers <=20000.')
    if (ocr.get('coordinate_system') != 'top-left' or ocr.get('bbox_format') != 'xywh'
            or geometry.get('coordinate_system') != 'top-left'):
        raise ValueError('Fusion requires explicit original-image top-left xywh coordinates.')
    if not isinstance(image.get('sha256'),str) or len(image['sha256']) != 64:
        raise ValueError('Source image SHA-256 is required.')
    if any(geometry.get('image',{}).get(k) != image[k] for k in ('width','height','sha256')):
        raise ValueError('OCR and geometry source image identities differ.')
    rows, shapes = ocr.get('rows'),geometry.get('shapes')
    if not isinstance(rows,list) or not isinstance(shapes,list):
        raise ValueError('OCR rows and geometry shapes must be lists.')
    rows = deepcopy(rows); source_ids=set()
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get('id'),str) or row['id'] in source_ids:
            raise ValueError('OCR source IDs must be unique strings.')
        source_ids.add(row['id'])
        if not isinstance(row.get('text'),str) or not 1 <= len(row['text']) <= 4000:
            raise ValueError('OCR text must be nonempty and within the existing IR limit; it is not silently dropped.')
        row['bbox'] = _box(row.get('bbox'),width,height)
        conf = row.get('confidence')
        if isinstance(conf,bool) or not isinstance(conf,(int,float)) or not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError('OCR confidence must be a measured finite value in [0,1].')
    rows.sort(key=lambda r:(r['bbox'][1],r['bbox'][0],r['id']))
    report = {'status':'partial','coordinate_system':'top-left','bbox_format':'xywh',
        'source_image_sha256':image['sha256'],'text_matches':[],'region_matches':[],
        'overlaps':[],'shape_deduplication':[],'unrendered_shapes':[],'line_decisions':[],
        'multiline_groups':[],'unconfirmed_arrow_candidates':deepcopy(geometry.get('arrow_candidates',[])),
        'diagnostics':[
            {'code':'semantic_connections_unverified','message':'Geometry does not establish directed topology; no semantic connections were invented.'},
            {'code':'appearance_estimated','message':'Font sizes derive from OCR box height; font family and exact glyph metrics are not recovered. Colors use the existing limited Scene palette.'}],
        'text_preservation':'Every OCR row becomes a separate text element without content rewriting.',
        'semantic_connection_count':0,'api_calls':0}
    if not rows:
        report['diagnostics'].append({'code':'empty_ocr','message':'OCR returned no text; geometric candidates are not a substitute for recognized content.'})
    accepted=[]; shape_ids=set()
    for raw in shapes:
        shape=deepcopy(raw)
        if not isinstance(shape,dict) or not isinstance(shape.get('id'),str) or shape['id'] in shape_ids:
            raise ValueError('Geometry source IDs must be unique strings.')
        shape_ids.add(shape['id']); shape['bbox']=_box(shape.get('bbox'),width,height)
        if shape.get('type') not in SHAPES:
            report['unrendered_shapes'].append({'source_id':shape['id'],'type':shape.get('type'),'bbox':shape['bbox'],'reason':'unconfirmed_shape_type'})
            continue
        duplicate=None
        for prior in accepted:
            overlap=_intersection(shape['bbox'],prior['bbox'])
            iou=overlap/(_area(shape['bbox'])+_area(prior['bbox'])-overlap)
            if shape['type']==prior['type'] and iou>=.97 and max(abs(a-b) for a,b in zip(shape['bbox'],prior['bbox']))<=2:
                duplicate=prior; break
        if duplicate:
            report['shape_deduplication'].append({'source_id':shape['id'],'retained_source_id':duplicate['id'],'reason':'same_shape_type_nearly_equal_measured_bbox'})
        else: accepted.append(shape)
    accepted.sort(key=lambda s:(-_area(s['bbox']),s['bbox'][1],s['bbox'][0],s['id']))
    regions=[]; elements=[]; styles={}
    for index,shape in enumerate(accepted,1):
        ident='local_region_%04d'%index
        shape['_region']=ident
        regions.append({'id':ident,'type':'unknown','bbox':shape['bbox'][:],'parent_id':None,
                        'uncertainty':'Measured spatial container candidate; semantic purpose is unverified.'})
    for shape,region in zip(accepted,regions):
        parent,candidates=_owner(region['bbox'],regions,exclude=region['id'])
        region['parent_id']=parent
        report['region_matches'].append({'source_id':shape['id'],'region_id':region['id'],'parent_id':parent,
            'containing_regions':candidates,'status':'ambiguous' if len(candidates)>1 and parent is None else 'geometric'})
        role='local_shape_style_%04d'%(len(elements)+1)
        styles[role]=_style(shape.get('style',{}))
        elements.append({'id':'local_shape_%04d'%(len(elements)+1),'type':shape['type'],'bbox':shape['bbox'][:],
            'region_id':region['id'],'style_role':role,'uncertainty':'Measured geometric candidate, not a confirmed semantic node.'})
    for i,a in enumerate(regions):
        for b in regions[i+1:]:
            overlap=_intersection(a['bbox'],b['bbox'])
            if overlap and not (_contains(a['bbox'],b['bbox']) or _contains(b['bbox'],a['bbox'])):
                report['overlaps'].append({'regions':[a['id'],b['id']],'intersection_area':overlap,'status':'ambiguous'})
    matches_by_region={}
    for order,row in enumerate(rows):
        owner,candidates=_owner(row['bbox'],regions)
        ident='local_text_%04d'%(order+1); role='local_text_style_%04d'%(order+1)
        styles[role]={'font_size':round(min(80,max(6,row['bbox'][3]*.82)),3),'text_color':'black'}
        item={'id':ident,'type':'multiline_text' if '\n' in row['text'] else 'text','bbox':row['bbox'][:],
              'region_id':owner,'raw_text':row['text'],'text_type':'plain','reading_order':order,
              'confidence':row['confidence'],'style_role':role}
        uncertainty=[]
        if owner is None and candidates: uncertainty.append('Multiple non-nested containers; ownership is unresolved.')
        if row['confidence']<.5: uncertainty.append('Low OCR confidence; text retained verbatim for review.')
        if uncertainty: item['uncertainty']=' '.join(uncertainty)
        elements.append(item)
        match={'source_id':row['id'],'element_id':ident,'region_id':owner,'containing_regions':candidates,
               'status':'ambiguous' if candidates and owner is None else 'contained' if owner else 'unassigned',
               'reading_order':order,'raw_text':row['text'],'bbox':row['bbox'][:],
               'ocr_reading_order_candidate':row.get('reading_order'),
               'confidence':row['confidence'],'role_candidate':'body_or_unclassified'}
        report['text_matches'].append(match)
        matches_by_region.setdefault(owner,[]).append(match)
    for owner,matches in matches_by_region.items():
        group=[]
        for index,match in enumerate(matches):
            match['region_reading_order']=index
            if owner is not None:
                box=next(r['bbox'] for r in regions if r['id']==owner)
                if match['bbox'][1]-box[1] <= .2*box[3]: match['role_candidate']='title_candidate'
            if group:
                previous=group[-1]; a,b=previous['bbox'],match['bbox']
                same_column=abs(a[0]-b[0]) <= .5*max(a[3],b[3])
                gap=b[1]-(a[1]+a[3])
                same_height=.6 <= b[3]/a[3] <= 1.65
                if owner is None or not (same_column and 0 <= gap <= 1.25*max(a[3],b[3]) and same_height):
                    if len(group)>1: report['multiline_groups'].append(_group(group,owner))
                    group=[]
            group.append(match)
        if len(group)>1: report['multiline_groups'].append(_group(group,owner))
    for index,line in enumerate(geometry.get('lines',[]),1):
        decision={'source_id':line.get('id'),'rendered':False,'reason':'unconfirmed_line'}
        report['line_decisions'].append(decision)
        pixel=line.get('pixel',{})
        if set(pixel) != {'x1','y1','x2','y2'}:
            decision['reason']='missing_measured_endpoints'; continue
        points=[[pixel['x1'],pixel['y1']],[pixel['x2'],pixel['y2']]]
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 or v>(width if axis==0 else height)
               for point in points for axis,v in enumerate(point)):
            raise ValueError('Line endpoints outside source image.')
        evidence=line.get('evidence',{})
        if (points[0]==points[1] or line.get('classification')!='geometry'
                or evidence.get('edge_coverage_fraction',0)<.9 or evidence.get('longest_continuous_support_fraction',0)<.75):
            decision['reason']='insufficient_continuous_nontext_support'; continue
        if any(_segment_hits_box(points,row['bbox']) for row in rows):
            decision['reason']='intersects_ocr_text'; continue
        border=False
        for shape in accepted:
            x,y,w,h=shape['bbox']
            if shape['type'] not in ('rectangle','rounded_rectangle','title_band'): continue
            for axis,edges,lo,hi in [(0,(x,x+w),y,y+h),(1,(y,y+h),x,x+w)]:
                if any(all(abs(point[axis]-edge)<=3 and lo-3<=point[1-axis]<=hi+3 for point in points) for edge in edges):
                    border=True
        if border:
            decision['reason']='already_represented_container_border'; continue
        x,y=min(p[0] for p in points),min(p[1] for p in points)
        w,h=max(p[0] for p in points)-x,max(p[1] for p in points)-y
        x=min(x,width-max(w,1)); y=min(y,height-max(h,1)); box=[x,y,max(w,1),max(h,1)]
        owner,_=_owner(box,regions)
        element={'id':'local_line_%04d'%index,'type':'connector','bbox':box,'points':points,'region_id':owner,
                 'uncertainty':'Measured line only; endpoint semantics and direction are unverified.'}
        elements.append(element); decision.update(rendered=True,element_id=element['id'],reason='continuous_geometric_line_not_semantic_connection')
    # One authoritative ownership model generates both sides of the relation.
    for region in regions:
        region['children']=[r['id'] for r in regions if r['parent_id']==region['id']]
        region['children'] += [e['id'] for e in elements if e['region_id']==region['id']]
    ir={'version':'1.0','canvas':{'id':'local_canvas','width':width,'height':height,'coordinate_system':'top-left'},
        'regions':regions,'elements':elements,'connections':[],
        'provenance':{'kind':'local_ocr_geometry','image_sha256':image['sha256'],'recognition':'local_ocr',
                      'topology':'unverified_no_inferred_connections'},
        'diagnostics':['Local recognition is partial: measured text and geometric candidates do not verify semantic topology.']}
    validation=inspect_ir(ir)
    if not validation['valid']:
        raise ValueError('Deterministic local IR failed validation: '+str(validation['errors']))
    report['counts']={'ocr_rows':len(rows),'preserved_text_elements':len(rows),'shape_candidates':len(shapes),
        'drawn_shapes':len(accepted),'regions':len(regions),'elements':len(elements),'connections':0,
        'ambiguous_text_ownership':sum(r['status']=='ambiguous' for r in report['text_matches']),
        'low_confidence_texts':sum(r['confidence']<.5 for r in rows),
        'drawn_line_candidates':sum(r['rendered'] for r in report['line_decisions'])}
    report['ir_validation']=validation
    return ir,report,styles


def _group(members,owner):
    x=min(m['bbox'][0] for m in members); y=min(m['bbox'][1] for m in members)
    right=max(m['bbox'][0]+m['bbox'][2] for m in members)
    bottom=max(m['bbox'][1]+m['bbox'][3] for m in members)
    return {'region_id':owner,'member_element_ids':[m['element_id'] for m in members],
            'member_source_ids':[m['source_id'] for m in members],
            'bbox':[x,y,right-x,bottom-y],
            'text':'\n'.join(m['raw_text'] for m in members),'status':'multiline_candidate',
            'render_policy':'original_rows_retained_without_reflow'}
