"""Pixel-supported filled arrowheads and continuous orthogonal shafts.

No graph edge is inferred here. Text, ordinary rules, closed borders, branching
paths and unsupported heads remain diagnostics. All coordinates are source pixels.
"""
from collections import deque
import math

from geometry_analyzer import _dependencies, _load_image
from local_geometry import _intersection


def _thin(mask, np):
    """Bounded Zhang-Suen skeletonization of a local binary component."""
    value=np.pad(mask.astype('uint8'),1)
    for _ in range(128):
        changed=False
        for phase in (0,1):
            p2=value[:-2,1:-1];p3=value[:-2,2:];p4=value[1:-1,2:];p5=value[2:,2:]
            p6=value[2:,1:-1];p7=value[2:,:-2];p8=value[1:-1,:-2];p9=value[:-2,:-2]
            ring=[p2,p3,p4,p5,p6,p7,p8,p9,p2]
            neighbors=sum(ring[:-1]);transitions=sum((a==0)&(b==1) for a,b in zip(ring,ring[1:]))
            remove=(value[1:-1,1:-1]==1)&(neighbors>=2)&(neighbors<=6)&(transitions==1)
            if phase==0:remove &= (p2*p4*p6==0)&(p4*p6*p8==0)
            else:remove &= (p2*p4*p8==0)&(p2*p6*p8==0)
            if remove.any():value[1:-1,1:-1][remove]=0;changed=True
        if not changed:return value[1:-1,1:-1]
    return None


def _heads(mask, cv2, np):
    contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:return []
    contour=max(contours,key=cv2.contourArea)
    # Absolute pixel tolerance preserves narrow shaft shoulders on long routes.
    poly=cv2.approxPolyDP(contour,1.1,True).reshape(-1,2).astype(float)
    result=[]
    if len(poly)<6:return result
    for i,tip in enumerate(poly):
        a,b=poly[(i-1)%len(poly)],poly[(i+1)%len(poly)]
        neck_a,neck_b=poly[(i-2)%len(poly)],poly[(i+2)%len(poly)]
        center=(a+b)/2;delta=tip-center;length=float(np.linalg.norm(delta))
        if length<3:continue
        axis=int(np.argmax(np.abs(delta)));other=1-axis
        if abs(delta[other])>max(1.5,abs(delta[axis])*.15):continue
        sign=1 if delta[axis]>0 else -1
        # Public detector results must be plain JSON/Python numbers even before
        # serialization; IR validation intentionally rejects NumPy scalar types.
        width=float(abs(a[other]-b[other])+1)
        shaft=float(abs(neck_a[other]-neck_b[other])+1)
        if width<6 or not .35<=length/width<=2.5 or shaft>width*.50:continue
        if abs(a[axis]-b[axis])>max(2,length*.18):continue
        if max(abs(neck_a[axis]-center[axis]),abs(neck_b[axis]-center[axis]))>max(2,length*.22):continue
        if abs((neck_a[other]+neck_b[other])/2-center[other])>max(1.5,width*.12):continue
        if not min(a[other],b[other])+width*.2<=tip[other]<=max(a[other],b[other])-width*.2:continue
        triangle=np.array([a,b,tip],dtype='int32')
        filled=np.zeros_like(mask,dtype='uint8');cv2.fillConvexPoly(filled,triangle,1)
        support=float(mask[filled>0].mean())
        if support<.80:continue
        result.append({'tip':tip.tolist(),'base_center':center.tolist(),'width_px':width,
            'length_px':length,'shaft_width_px':shaft,'axis':axis,'sign':sign,
            'triangle':triangle.tolist(),'fill_support':support})
    return result


def _shaft_path(mask, head, cv2, np):
    axis,sign=head['axis'],head['sign'];other=1-axis
    center=np.array(head['base_center']);coords=np.indices(mask.shape)[::-1]
    cut=((coords[axis]-center[axis])*sign>=-1)&(np.abs(coords[other]-center[other])<=head['width_px']*.65)
    shaft=mask.copy();shaft[cut]=0
    skeleton=_thin(shaft,np)
    if skeleton is None:return None,'skeleton_budget_exceeded'
    points={(int(x),int(y)) for y,x in np.argwhere(skeleton>0)}
    if len(points)<5:return None,'shaft_too_short'
    adjacency={}
    for x,y in points:
        neighbors=[]
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            q=(x+dx,y+dy)
            if q not in points:continue
            if dx and dy and ((x+dx,y) in points or (x,y+dy) in points):continue
            neighbors.append(q)
        adjacency[(x,y)]=neighbors
    endpoints=[p for p in points if len(adjacency[p])==1]
    if len(endpoints)!=2 or any(len(v)>2 for v in adjacency.values()):
        return None,'branching_or_closed_shaft'
    near=min(endpoints,key=lambda p:math.dist(p,center))
    if math.dist(near,center)>max(5,head['shaft_width_px']*2):return None,'head_shaft_disconnected'
    tail=next(p for p in endpoints if p!=near)
    prior={tail:None};pending=deque([tail])
    while pending:
        p=pending.popleft()
        for q in adjacency[p]:
            if q not in prior:prior[q]=p;pending.append(q)
    if near not in prior or len(prior)<len(points)*.95:return None,'disconnected_shaft'
    path=[];p=near
    while p is not None:path.append(p);p=prior[p]
    path.reverse()
    if len(path)<max(6,head['length_px']*.7):return None,'insufficient_shaft_length'
    simplified=cv2.approxPolyDP(np.array(path,dtype='float32'),max(1.25,head['shaft_width_px']*.6),False).reshape(-1,2).tolist()
    if len(simplified)>7:return None,'complex_shaft_route'
    for a,b in zip(simplified,simplified[1:]):
        delta=np.abs(np.array(b)-a)
        if min(delta)>max(2,max(delta)*.15):return None,'nonorthogonal_shaft_route'
    if len(simplified)>1:
        delta=np.array(simplified[-1])-simplified[-2]
        if delta[axis]*sign<=0 or abs(delta[other])>max(2,abs(delta[axis])*.15):
            return None,'head_direction_disagrees_with_shaft'
    # The final centerline is measured; retain the tip's actual source position.
    simplified[-1]=head['base_center'][:]
    simplified.append(head['tip'][:])
    return simplified,None


def _base_lines(points,geometry):
    matched=[]
    for line in (geometry or {}).get('lines',[]):
        p=line.get('pixel',{})
        if not all(k in p for k in ('x1','y1','x2','y2')):continue
        a=[p['x1'],p['y1']];b=[p['x2'],p['y2']]
        for u,v in zip(points,points[1:]):
            if math.dist(a,u)+math.dist(b,v)<=8 or math.dist(a,v)+math.dist(b,u)<=8:
                matched.append(line.get('id'));break
    return matched


def analyze_local_arrows(image_path,ocr_rows,geometry=None):
    """Detect supported heads and paths, independently of endpoint semantics."""
    cv2,np=_dependencies();pixels,image=_load_image(image_path)
    if geometry and any(geometry.get('image',{}).get(k)!=image[k] for k in ('width','height','sha256')):
        raise ValueError('Arrow geometry source image identity differs.')
    height,width=pixels.shape[:2];hsv=cv2.cvtColor(pixels,cv2.COLOR_BGR2HSV)
    # Saturated ink and neutral dark ink are separate masks. No page-specific hue.
    chromatic=cv2.inRange(hsv,np.array([0,40,25]),np.array([179,255,255]))
    neutral=((hsv[:,:,1]<40)&(hsv[:,:,2]<155)).astype('uint8')*255
    candidates=[];rejected=[];counts={'components':0,'text_components_excluded':0,'large_components_excluded':0,
                                    'thin_components_without_head':0}
    for method,mask in [('chromatic_component',chromatic),('neutral_dark_component',neutral)]:
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        for ident in range(1,count):
            x,y,w,h,area=map(int,stats[ident]);box=[x,y,w,h]
            if max(w,h)<10 or area<20:continue
            if min(w,h)<5:
                counts['thin_components_without_head']+=1;continue
            counts['components']+=1
            if w*h>width*height*.90 or area>width*height*.50 or w*h>2_000_000:
                counts['large_components_excluded']+=1
                rejected.append({'bbox':box,'reason':'large_component_resource_or_background_limit','status':'unconfirmed_geometry'})
                continue
            if any(_intersection(box,r['bbox'])/(w*h)>.35 for r in ocr_rows):
                counts['text_components_excluded']+=1;continue
            component=(labels[y:y+h,x:x+w]==ident).astype('uint8')
            heads=_heads(component,cv2,np)
            if not heads:
                rejected.append({'bbox':box,'method':method,'reason':'no_supported_filled_arrowhead','status':'unconfirmed_geometry'});continue
            if len(heads)!=1:
                rejected.append({'bbox':box,'reason':'multiple_arrowhead_candidates','status':'ambiguous'});continue
            head=heads[0];path,reason=_shaft_path(component,head,cv2,np)
            if path is None:
                rejected.append({'bbox':box,'reason':reason,'head':head,'status':'unconfirmed_geometry'});continue
            points=[[round(px+x,3),round(py+y,3)] for px,py in path]
            rgb=np.median(pixels[y:y+h,x:x+w][component>0][:,::-1],axis=0).round().astype(int).tolist()
            direction={ (0,1):'right',(0,-1):'left',(1,1):'down',(1,-1):'up'}[(head['axis'],head['sign'])]
            triangle=[[int(a+x),int(b+y)] for a,b in head['triangle']]
            xs=[p[0] for p in triangle];ys=[p[1] for p in triangle]
            candidates.append({'bbox':box,'points':points,'direction':direction,'status':'confirmed_arrow',
                'head':{'bbox':[min(xs),min(ys),max(xs)-min(xs)+1,max(ys)-min(ys)+1],
                        'tip':points[-1],'base_center':[head['base_center'][0]+x,head['base_center'][1]+y],
                        'width_px':head['width_px'],'length_px':head['length_px'],'triangle':triangle},
                'style':{'stroke_color':'#'+''.join('%02X'%v for v in rgb),'line_width':head['shaft_width_px']},
                'evidence':{'method':method+'_filled_head_and_continuous_skeleton',
                    'shaft_width_px':head['shaft_width_px'],'head_to_shaft_ratio':head['width_px']/head['shaft_width_px'],
                    'head_fill_support':head['fill_support'],'component_pixel_count':area,
                    'base_line_ids':_base_lines(points,geometry),'source':'original_pixels',
                    'route':'straight' if len(points)<=3 else 'orthogonal_polyline'},
                'score':.85,'score_type':'heuristic_evidence_rank_not_probability'})
    arrows=[]
    for candidate in sorted(candidates,key=lambda a:(a['bbox'][1],a['bbox'][0])):
        if any(math.dist(candidate['points'][-1],old['points'][-1])<3 and math.dist(candidate['points'][0],old['points'][0])<3 for old in arrows):continue
        arrows.append(dict(candidate,id='local_arrow_%04d'%(len(arrows)+1)))
    ordinary=[{'line_id':line.get('id'),'reason':'No confirmed arrowhead and continuous route assigned; remains ordinary geometry.'}
        for line in (geometry or {}).get('lines',[]) if line.get('id') not in {i for a in arrows for i in a['evidence']['base_line_ids']}]
    return {'version':'0.5','image':image,'coordinate_system':'top-left','bbox_format':'xywh',
        'arrows':arrows,'rejected_candidates':rejected,'ordinary_line_diagnostics':ordinary,
        'counts':dict(counts,confirmed_arrows=len(arrows)),
        'diagnostics':[{'code':'conservative_arrow_scope','message':'Filled cardinal heads with unbranched orthogonal shafts supported. Open, disconnected, border-attached or ambiguous heads remain partial; no nearest-node inference.'}],
        'api_calls':0}
