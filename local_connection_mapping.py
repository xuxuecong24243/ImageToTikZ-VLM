"""Map pixel-proven arrow routes to existing nodes without inventing topology.

Semantic side anchors and observed path coordinates are both retained. A small
gap from a measured arrow to its card is evidence, not permission to move either.
"""
from copy import deepcopy
import math

from local_fusion import _segment_hits_box
from structured_ir.validator import validate_ir
from structured_tikz.geometry import node_anchor

NODE_TYPES = {'rectangle', 'rounded_rectangle', 'circle', 'diamond'}


def _inside(inner, outer):
    return (inner[0]>=outer[0] and inner[1]>=outer[1] and
            inner[0]+inner[2]<=outer[0]+outer[2] and inner[1]+inner[3]<=outer[1]+outer[3])


def _unit(a,b):
    delta=[b[i]-a[i] for i in (0,1)];length=math.hypot(*delta)
    return [v/length for v in delta] if length else None


def _cardinal(tangent):
    axis=max(range(2),key=lambda i:abs(tangent[i]))
    if abs(tangent[1-axis])>.18:return None
    return axis,1 if tangent[axis]>0 else -1


def _ancestor(child, parent, regions):
    seen=set()
    while child in regions and child not in seen:
        if child==parent:return True
        seen.add(child);child=regions[child]['parent_id']
    return False


def _boundary(node, point, axis, side_sign):
    """Boundary on the required side at the observed perpendicular coordinate."""
    x,y,w,h=node['bbox'];center=[x+w/2,y+h/2];radii=[w/2,h/2]
    perpendicular=abs(point[1-axis]-center[1-axis])/radii[1-axis]
    if perpendicular>=.98:return None
    kind=node['type']
    if kind=='circle':factor=math.sqrt(max(0,1-perpendicular**2))
    elif kind=='diamond':factor=1-perpendicular
    else:factor=1.
    result=point[:];result[axis]=center[axis]+side_sign*radii[axis]*factor
    return result


def _endpoint_candidates(ir, point, tangent, role, distance_limit):
    cardinal=_cardinal(tangent)
    if cardinal is None:return {'status':'unsupported_endpoint_angle','candidates':[],'groups':[]}
    axis,travel=cardinal
    side_sign=travel if role=='source' else -travel
    anchor={ (0,1):'east',(0,-1):'west',(1,1):'south',(1,-1):'north'}[(axis,side_sign)]
    candidates=[]
    for node in ir['elements']:
        if node['type'] not in NODE_TYPES:continue
        boundary=_boundary(node,point,axis,side_sign)
        if boundary is None:continue
        gap=(point[axis]-boundary[axis])*side_sign
        # The end must be outside the facing boundary (with a small raster
        # tolerance), on the side predicted by the actual shaft tangent.
        if not -1.5<=gap<=distance_limit:continue
        center=[node['bbox'][i]+node['bbox'][i+2]/2 for i in (0,1)]
        if (point[axis]-center[axis])*side_sign<=0:continue
        candidates.append({'element_id':node['id'],'region_id':node['region_id'],
            'bbox':deepcopy(node['bbox']),'type':node['type'],'anchor':anchor,
            'boundary_point':boundary,'boundary_gap_px':gap,
            'perpendicular_position_fraction':(point[1-axis]-node['bbox'][1-axis])/node['bbox'][3-axis],
            'semantic_anchor_point':node_anchor(node,anchor)})
    regions={r['id']:r for r in ir['regions']}
    # Co-located outlines in a known parent/body family are one physical card.
    # Merely nearby or overlapping rectangles are deliberately not equivalent.
    groups=[]
    for candidate in sorted(candidates,key=lambda c:-(c['bbox'][2]*c['bbox'][3])):
        compatible=[]
        for group in groups:
            outer=group[0]
            related=(candidate['region_id'] is not None and outer['region_id'] is not None and
                     _ancestor(candidate['region_id'],outer['region_id'],regions))
            if (related and _inside(candidate['bbox'],outer['bbox']) and
                    abs(candidate['boundary_point'][axis]-outer['boundary_point'][axis])<=3):
                compatible.append(group)
        if len(compatible)==1:compatible[0].append(candidate)
        else:groups.append([candidate])
    result={'status':'unique' if len(groups)==1 else 'ambiguous' if groups else 'unmatched',
            'candidates':candidates,'groups':[{'canonical_element_id':g[0]['element_id'],
                'alias_element_ids':[c['element_id'] for c in g[1:]],
                'basis':'outer boundary of a nested region family with coincident facing sides'} for g in groups]}
    if len(groups)==1:result['selected']=groups[0][0]
    return result


def _route_valid(points, canvas):
    return (isinstance(points,list) and 2<=len(points)<=100 and
        all(isinstance(p,(list,tuple)) and len(p)==2 and
            all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=canvas['width' if i==0 else 'height']
                for i,v in enumerate(p)) for p in points) and
        all(a!=b for a,b in zip(points,points[1:])))


def _same_shaft(line, points, tolerance):
    """Only a whole, collinear old segment may be replaced by its proven arrow."""
    if line['type']!='connector' or len(line.get('points',[]))!=2:return False
    a,b=line['points']
    for start,end in zip(points,points[1:]):
        unit=_unit(start,end);length=math.dist(start,end)
        positions=[]
        for point in (a,b):
            offset=[point[i]-start[i] for i in (0,1)]
            along=sum(offset[i]*unit[i] for i in (0,1))
            across=abs(offset[0]*unit[1]-offset[1]*unit[0])
            if across>tolerance or not -tolerance<=along<=length+tolerance:break
            positions.append(along)
        if len(positions)==2 and abs(positions[1]-positions[0])>=length*.55:return True
    return False


def map_local_connections(ir, arrow_report, styles=None):
    """Return (IR copy, mapping report, styles copy), preserving observed routes.

Only detector-confirmed head + shaft evidence is admitted. Unmatched or ambiguous
confirmed arrows remain editable geometric arrows, with no invented endpoints.
"""
    validate_ir(ir)
    if not isinstance(arrow_report,dict) or not isinstance(arrow_report.get('arrows'),list):
        raise ValueError('Arrow report must contain an arrows array.')
    if styles is not None and not isinstance(styles,dict):raise ValueError('styles must be a mapping.')
    image=arrow_report.get('image',{})
    expected=ir.get('provenance',{}).get('image_sha256')
    if image.get('sha256') and expected and image['sha256']!=expected:
        raise ValueError('Arrow source image does not match IR.')
    for dimension in ('width','height'):
        if dimension in image and image[dimension]!=ir['canvas'][dimension]:
            raise ValueError('Arrow image dimensions do not match IR.')
    result=deepcopy(ir);style_map=deepcopy(styles or {})
    report={'status':'partial','mappings':[],'diagnostics':[],'deduplicated_lines':[],
            'coordinate_system':'top-left','path_policy':'Original measured points retained without rerouting.',
            'endpoint_policy':'Head direction, shaft tangent, side projection, signed boundary gap and region hierarchy; ambiguous families not guessed.',
            'api_calls':0}
    used={result['canvas']['id']}|{o['id'] for key in ('regions','elements','connections') for o in result[key]}
    def allocate(prefix):
        n=1
        while prefix+('%04d'%n) in used:n+=1
        ident=prefix+('%04d'%n);used.add(ident);return ident
    text_boxes=[e for e in ir['elements'] if e['type'] in {'text','multiline_text','code','math'} and e.get('raw_text')]
    referenced={c[k] for c in ir['connections'] for k in ('source_id','target_id')}
    seen_ids=set();seen_paths=[];removed=set()
    for arrow in arrow_report['arrows']:
        if not isinstance(arrow,dict):raise ValueError('Each arrow must be an object.')
        ident=arrow.get('id')
        if not isinstance(ident,str) or ident in seen_ids:raise ValueError('Arrow IDs must be unique strings.')
        seen_ids.add(ident)
        entry={'arrow_id':ident,'status':'rejected','diagnostics':[]};report['mappings'].append(entry)
        points=arrow.get('points')
        if arrow.get('status')!='confirmed_arrow' or not arrow.get('head') or not arrow.get('evidence'):
            entry['diagnostics'].append({'code':'unconfirmed_head_or_shaft'});continue
        if not _route_valid(points,ir['canvas']):
            entry['diagnostics'].append({'code':'invalid_measured_path'});continue
        if points in seen_paths:
            entry['diagnostics'].append({'code':'duplicate_detected_path'});continue
        seen_paths.append(deepcopy(points))
        head_tip=arrow['head'].get('tip')
        if (not isinstance(head_tip,(list,tuple)) or len(head_tip)!=2 or
                any(type(v) not in (int,float) or not math.isfinite(v) for v in head_tip) or
                math.dist(head_tip,points[-1])>1.5):
            entry['diagnostics'].append({'code':'head_tip_path_disagreement'});continue
        crossed=[e['id'] for e in text_boxes if any(_segment_hits_box([a,b],e['bbox']) for a,b in zip(points,points[1:]))]
        if crossed:
            entry['diagnostics'].append({'code':'path_crosses_existing_text','element_ids':crossed});continue
        total=sum(math.dist(a,b) for a,b in zip(points,points[1:]))
        head_width=arrow['head'].get('width_px',0)
        if type(head_width) not in (int,float) or not math.isfinite(head_width) or head_width<=0:
            entry['diagnostics'].append({'code':'invalid_head_measurement'});continue
        limit=min(40.,max(6.,head_width*1.5,total*.25))
        source=_endpoint_candidates(ir,points[0],_unit(points[0],points[1]),'source',limit)
        target=_endpoint_candidates(ir,points[-1],_unit(points[-2],points[-1]),'target',limit)
        # Existing diamond rendering requires exact vertex endpoints. Keep a
        # measured gapped/off-vertex arrow geometric instead of declaring a
        # semantic edge that the downstream renderer would then discard.
        for endpoint,point in ((source,points[0]),(target,points[-1])):
            chosen=endpoint.get('selected')
            if chosen and chosen['type']=='diamond' and math.dist(point,chosen['semantic_anchor_point'])>1e-6:
                endpoint['status']='unsupported_diamond_anchor_offset'
                entry['diagnostics'].append({'code':'diamond_endpoint_kept_geometric',
                    'element_id':chosen['element_id'],'message':'Observed path is not at the existing exact diamond anchor; no rerouting.'})
        entry.update(source=source,target=target,points=deepcopy(points),boundary_distance_limit_px=limit)
        if source['status']=='unique' and target['status']=='unique' and source['selected']['element_id']==target['selected']['element_id']:
            entry['diagnostics'].append({'code':'same_endpoint_node'});continue
        visual=deepcopy(arrow.get('style',{}))
        if set(visual)-{'stroke_color','line_width'}:
            raise ValueError('Arrow visual style supports only measured stroke_color and line_width.')
        if 'line_width' in visual:
            width=visual['line_width']
            if type(width) not in (int,float) or not math.isfinite(width) or width<=0:raise ValueError('Invalid arrow line width.')
            if not .1<=width<=10:
                entry['diagnostics'].append({'code':'stroke_width_clamped','measured_px':width})
                visual['line_width']=min(10,max(.1,width))
        if source['status']=='unique' and target['status']=='unique':
            a,b=source['selected'],target['selected']
            connection={'id':allocate('local_flow_'),'source_id':a['element_id'],'target_id':b['element_id'],
                'source_anchor':a['anchor'],'target_anchor':b['anchor'],'direction':'forward','kind':'flow',
                'points':deepcopy(points),'visual_style':visual,
                'uncertainty':'Topology is supported by measured arrowhead and boundary geometry; arbitrary semantic meaning is unverified.'}
            result['connections'].append(connection)
            entry.update(status='mapped',connection_id=connection['id'],source_id=a['element_id'],target_id=b['element_id'],
                source_anchor=a['anchor'],target_anchor=b['anchor'],
                cross_region=a['region_id']!=b['region_id'],
                anchor_offsets_px={'source':[points[0][i]-a['semantic_anchor_point'][i] for i in (0,1)],
                                   'target':[points[-1][i]-b['semantic_anchor_point'][i] for i in (0,1)]})
        else:
            x=min(p[0] for p in points);y=min(p[1] for p in points)
            w=max(1,max(p[0] for p in points)-x);h=max(1,max(p[1] for p in points)-y)
            x=min(x,ir['canvas']['width']-w);y=min(y,ir['canvas']['height']-h)
            element={'id':allocate('local_detected_arrow_'),'type':'arrow','bbox':[x,y,w,h],
                'region_id':None,'points':deepcopy(points),'visual_style':visual,
                'uncertainty':'Pixel-confirmed arrow; source or target is ambiguous/unmatched. No semantic endpoints inferred.'}
            result['elements'].append(element)
            entry.update(status='geometric_only',element_id=element['id'])
            entry['diagnostics'].append({'code':'semantic_endpoint_unresolved'})
        tolerance=min(2.,max(.75,visual.get('line_width',1)/2))
        for element in ir['elements']:
            if (element['id'] not in removed and element['id'] not in referenced and not element.get('raw_text')
                    and _same_shaft(element,points,tolerance)):
                removed.add(element['id'])
                report['deduplicated_lines'].append({'element':deepcopy(element),'arrow_id':ident,
                    'basis':'Complete old connector segment is collinear and enclosed by the measured arrow shaft.',
                    'tolerance_px':tolerance})
    if removed:
        result['elements']=[e for e in result['elements'] if e['id'] not in removed]
        for region in result['regions']:
            if 'children' in region:region['children']=[i for i in region['children'] if i not in removed]
    mapped=sum(m['status']=='mapped' for m in report['mappings'])
    report['counts']={'detected_arrows':len(arrow_report['arrows']),'mapped_connections':mapped,
        'geometric_only':sum(m['status']=='geometric_only' for m in report['mappings']),
        'rejected':sum(m['status']=='rejected' for m in report['mappings']),
        'deduplicated_line_elements':len(removed)}
    if mapped:
        result.setdefault('provenance',{})['local_connection_mapping']={
            'basis':'pixel_arrowhead_and_facing_boundaries','mapped_count':mapped,'semantic_labels':'unverified'}
    validate_ir(result)
    return result,report,style_map
