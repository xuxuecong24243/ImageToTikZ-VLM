"""Measured visual appearance only; no OCR text or semantic graph rewriting.

Colours are sampled in source pixels. Discrete colour components complement
the old edge detector, which cannot by itself expose a filled title strip.
Evidence and uncertainty accompany every estimate; RGB is never quantized to
the rendering palette. All thresholds describe image evidence, not page text.
"""
import math
from geometry_analyzer import _dependencies, _load_image


def _hex(rgb):
    return '#' + ''.join('%02X' % int(v) for v in rgb)


def _modal(values, np):
    if not len(values): return None, 0.
    values=values[::max(1,len(values)//150000)]
    codes=values.astype('int32')//8
    keys=codes[:,0]*1024+codes[:,1]*32+codes[:,2]
    unique,counts=np.unique(keys,return_counts=True)
    selected=values[keys==unique[counts.argmax()]]
    color=np.median(selected,axis=0).round().astype(int)
    fraction=float(np.mean(np.max(np.abs(values.astype(float)-color),axis=1)<=12))
    return color,fraction


def _box(box,width,height):
    if not isinstance(box,(list,tuple)) or len(box)!=4 or any(type(v) not in (float,int) or not math.isfinite(v) for v in box):
        raise ValueError('Visual style bbox must contain four finite original-pixel coordinates.')
    x,y,w,h=box
    if x<0 or y<0 or w<=0 or h<=0 or x+w>width or y+h>height:
        raise ValueError('Visual style bbox outside original image.')
    return [int(x),int(y),math.ceil(x+w)-int(x),math.ceil(y+h)-int(y)]


def _corner_measure(mask,cv2,np):
    """Four-corner inward offsets fit the circular rounded-rectangle relation.

Only symmetric corner evidence yields a radius. Top-only round title bands and
irregular components retain an uncertainty diagnostic instead of a fake radius.
"""
    h,w=mask.shape
    if min(h,w)<10: return None,{'status':'insufficient_corner_extent'}
    edge=[]
    for part in [mask,mask[:,::-1],mask[::-1,:],mask[::-1,::-1]]:
        # At the first supported row, x offset approximates the circle radius.
        offsets=[]
        for dy in range(min(3,h)):
            xx=np.flatnonzero(part[dy])
            if len(xx): offsets.append(int(xx[0]))
        edge.append(float(np.median(offsets)) if offsets else float(min(h,w)))
    maximum=min(h,w)*.4
    if max(edge)<=2:
        return 0.,{'status':'straight','corner_offsets_px':edge}
    if min(edge)>=2 and max(edge)-min(edge)<=max(3,float(np.median(edge))*.5):
        # Rows below the tangent lose sqrt(2*r*y) offset. Use the first row
        # separately to avoid conflating straight and shallow rounded corners.
        top_offsets=[]
        for part in [mask,mask[:,::-1],mask[::-1,:],mask[::-1,::-1]]:
            xx=np.flatnonzero(part[0]); top_offsets.append(int(xx[0]) if len(xx) else 0)
        radius=float(np.median(top_offsets))
        if 2<=radius<=min(50,maximum):
            return round(radius,1),{'status':'rounded_estimate','corner_offsets_px':edge,
                'radius_interval_px':[max(0,round(radius-2,1)),round(radius+2,1)]}
    return None,{'status':'uncertain_asymmetric_or_nonrectangular','corner_offsets_px':edge}


def _measure(rgb,box,excluded,cv2,np):
    x,y,w,h=box; crop=rgb[y:y+h,x:x+w]
    mask=excluded[y:y+h,x:x+w]
    inset=max(2,min(10,int(min(w,h)*.1)))
    interior=np.zeros((h,w),dtype=bool)
    if w>2*inset and h>2*inset: interior[inset:-inset,inset:-inset]=True
    interior &= ~mask
    fill,fraction=_modal(crop[interior],np)
    evidence={'fill':{'sample_count':int(interior.sum()),'dominant_fraction':round(fraction,4),
                      'method':'modal_local_RGB_excluding_text_and_boundary'}}
    style={'fill':'none','stroke':'none'}
    if fill is None:
        evidence['fill']['status']='insufficient_samples'
        return style,evidence,None
    evidence['fill']['rgb']=fill.tolist()
    reliable=fraction>=.65
    spread=float(np.percentile(np.max(np.abs(crop[interior].astype(float)-fill),axis=1),90)) if interior.any() else 0
    evidence['fill']['spread_p90_rgb']=round(spread,3)
    evidence['fill']['status']=('measured_uniform' if spread<=8 else 'approximate_nonuniform') if reliable else 'uncertain_nonuniform'
    bands=[]
    for a,b in zip(np.linspace(inset,h-inset,5).astype(int)[:-1],np.linspace(inset,h-inset,5).astype(int)[1:]):
        selected=crop[a:b][interior[a:b]]
        if len(selected):bands.append(np.median(selected,axis=0))
    if len(bands)==4:
        steps=np.diff(np.array(bands),axis=0)
        total=float(np.max(np.ptp(np.array(bands),axis=0)))
        smooth=(total>=10 and np.max(np.abs(steps))<=12 and
                np.all((steps>=-1).all(axis=0)|(steps<=1).all(axis=0)))
        evidence['fill']['smooth_vertical_variation']=bool(smooth)
    if reliable: style['fill_color']=_hex(fill)
    # Observe normal profiles along four straight mid-edge strips. Require
    # repeated support on multiple sides and a bounded consistent thickness.
    observations=[]; side_colors=[]
    for side in range(4):
        profiles=[]
        length=w if side<2 else h
        for pos in np.linspace(.2*length,.8*length,17).astype(int):
            pos=min(length-1,pos)
            if side==0: coordinates=[(j,pos) for j in range(min(10,h))]
            elif side==1: coordinates=[(h-1-j,pos) for j in range(min(10,h))]
            elif side==2: coordinates=[(pos,j) for j in range(min(10,w))]
            else: coordinates=[(pos,w-1-j) for j in range(min(10,w))]
            if any(mask[yy,xx] for yy,xx in coordinates):continue
            vals=np.array([crop[yy,xx] for yy,xx in coordinates])
            different=np.max(np.abs(vals.astype(float)-fill),axis=1)>12
            # A boundary must start at the supplied contour, not at a letter
            # or a separator several pixels inside the container.
            n=0
            while n<len(vals) and different[n]:n+=1
            if 0<n<len(vals) and n<=8:
                color,_=_modal(vals[:n],np); profiles.append((n,color))
        if len(profiles)>=7:
            widths=[p[0] for p in profiles]
            if max(widths)-min(widths)<=2:
                color,_=_modal(np.array([p[1] for p in profiles]),np)
                side_colors.append(color); observations.append({'side':side,'width_px':float(np.median(widths)),
                    'profile_count':len(profiles),'rgb':color.tolist()})
    evidence['stroke']={'status':'uncertain_no_consistent_border','profiles':observations}
    if len(observations)>=2:
        stroke,_=_modal(np.array(side_colors),np)
        supported=[o for o in observations if max(abs(a-b) for a,b in zip(o['rgb'],stroke))<=15]
        if len(supported)>=2:
            widths=[o['width_px'] for o in supported]
            style['stroke_color']=_hex(stroke)
            evidence['stroke'].update(status='measured_border',rgb=stroke.tolist())
            if max(widths)-min(widths)<=2:
                style['line_width']=float(np.median(widths))
                evidence['stroke']['width_interval_px']=[max(.1,min(widths)-1),max(widths)+1]
            else:evidence['stroke']['width_status']='uncertain_side_variation'
    # Use a filled outer silhouette, not glyph holes, to inspect all corners.
    occupancy=np.max(np.abs(crop.astype(float)-fill),axis=2)<=12
    if 'stroke_color' in style:
        stroke=np.array([int(style['stroke_color'][i:i+2],16) for i in (1,3,5)])
        occupancy |= np.max(np.abs(crop.astype(float)-stroke),axis=2)<=12
    contours,_=cv2.findContours(occupancy.astype('uint8')*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    silhouette=np.zeros((h,w),dtype='uint8')
    if contours:cv2.drawContours(silhouette,[max(contours,key=cv2.contourArea)],-1,1,cv2.FILLED)
    radius,corner=_corner_measure(silhouette,cv2,np)
    evidence['corner']=corner
    if radius is not None:style['rounded_radius']=radius
    return style,evidence,radius


def _intersection(a,b):
    return max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))


def _role(box,containers,peers,image_area):
    x,y,w,h=box
    parents=[b for b in containers if b[2]*b[3]>w*h*1.02 and
             b[0]<=x+3 and b[1]<=y+3 and b[0]+b[2]>=x+w-3 and b[1]+b[3]>=y+h-3]
    parents.sort(key=lambda b:b[2]*b[3])
    parent=parents[0] if parents else None
    below=any(abs(b[0]-x)<=6 and abs(b[2]-w)<=12 and abs(b[1]-(y+h))<=8 and b[3]>h*1.2 for b in peers if b!=box)
    above=any(abs(b[0]-x)<=6 and abs(b[2]-w)<=12 and abs(b[1]+b[3]-y)<=8 and h>b[3]*1.2 for b in peers if b!=box)
    if w/h>=3 and (below or (parent and y-parent[1]<=max(6,parent[3]*.06) and h<parent[3]*.3)):
        return 'title_band', 'Thin top-aligned fill adjacent to a larger body or within a measured container.'
    if above:return 'body_fill','Fill immediately below a narrower-height header with matching horizontal span.'
    if parent:return 'child_card','Contained visual rectangle; semantic purpose is unknown.'
    if w*h>=image_area*.05:return 'main_container','Large standalone visual container; semantic purpose is unknown.'
    return 'child_card','Standalone compact visual fill; card role remains a candidate.'


def extract_visual_styles(image_path,ocr_rows,geometry):
    """Return measured styles and new filled-component candidates, no IR edits."""
    cv2,np=_dependencies(); pixels,image=_load_image(image_path)
    width,height=image['width'],image['height']; rgb=cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB)
    if any(geometry.get('image',{}).get(k)!=image[k] for k in ('sha256','width','height')):
        raise ValueError('Style measurements must match the original geometry image.')
    excluded=np.zeros((height,width),dtype=bool)
    for row in ocr_rows:
        x,y,w,h=_box(row['bbox'],width,height);excluded[y:y+h,x:x+w]=True
    # Border samples identify page background without averaging page content.
    rim=max(1,min(width,height)//100)
    page_values=np.concatenate([rgb[:rim].reshape(-1,3),rgb[-rim:].reshape(-1,3),
        rgb[:,:rim].reshape(-1,3),rgb[:,-rim:].reshape(-1,3)])
    page,fraction=_modal(page_values,np)
    page_style={'fill':'none','stroke':'none'}
    if fraction>=.85:page_style['fill_color']=_hex(page)
    report={'version':'0.3','image':image,'coordinate_system':'top-left','bbox_format':'xywh',
        'shapes':[],'regions':[],'text_styles':{},'diagnostics':[],
        'page_background':{'bbox':[0,0,width,height],'type':'rectangle','role_candidate':'page_background',
            'visual_style':page_style,'evidence':{'fill':{'status':'measured_uniform' if fraction>=.85 else 'uncertain_nonuniform',
                'rgb':page.tolist(),'dominant_fraction':fraction,'method':'image_border_modal_pixels'}}},'api_calls':0}
    for shape in geometry['shapes']:
        box=_box(shape['bbox'],width,height)
        style,evidence,radius=_measure(rgb,box,excluded,cv2,np)
        item={'source_id':shape['id'],'bbox':shape['bbox'][:],'visual_style':style,'evidence':evidence,
            'role_candidate':'container_background','status':'measured_candidate'}
        if shape['type'] in ('rectangle','rounded_rectangle','unknown') and radius is not None:
            item['type']='rounded_rectangle' if radius>0 else 'rectangle'
        report['shapes'].append(item)
    # Quantized mode seeds are image-derived. Each component retains actual
    # median source RGB; narrow boundaries and glyph clusters fail area tests.
    sample=rgb[::2,::2].reshape(-1,3)
    codes=sample.astype('int32')//8
    keys=codes[:,0]*1024+codes[:,1]*32+codes[:,2]
    values,counts=np.unique(keys,return_counts=True)
    seeds=[]
    for k in values[np.argsort(counts)[::-1]][:40]:
        vals=sample[keys==k];color=np.median(vals,axis=0).round().astype(int)
        if max(abs(color-page))<=4 or any(max(abs(color-c))<=6 for c in seeds):continue
        if len(vals)<max(30,width*height/4*.00003):continue
        seeds.append(color)
    candidates=[]
    for seed in seeds:
        mask=cv2.inRange(rgb,np.maximum(0,seed-6).astype('uint8'),np.minimum(255,seed+6).astype('uint8'))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        # Remove narrow same-colour borders/lines that connect a solid header
        # strip to the whole panel. The original pixels remain the samples.
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour);area=w*h
            if w<max(24,width*.008) or h<max(14,height*.004) or area<max(300,width*height*.00007):continue
            if area>width*height*.90 or cv2.contourArea(contour)/area<.82:continue
            component=np.zeros((h,w),dtype='uint8')
            cv2.drawContours(component,[contour-[x,y]],-1,1,cv2.FILLED)
            filled=mask[y:y+h,x:x+w]>0; excluded_crop=excluded[y:y+h,x:x+w]
            available=(component>0)&~excluded_crop
            if available.sum()<area*.25:continue
            fraction=float(filled[available].mean())
            if fraction<.72:continue
            color=np.median(rgb[y:y+h,x:x+w][available&filled],axis=0).round().astype(int)
            # Reject same-page paper components left between cards/diagrams.
            if max(abs(color-page))<=4:continue
            radius,corner=_corner_measure(component,cv2,np)
            kind='rounded_rectangle' if radius and radius>0 else 'rectangle'
            style={'fill_color':_hex(color),'stroke':'none'}
            if radius is not None:style['rounded_radius']=radius
            role='filled_region'
            if w/h>=3 and h<height*.16:role='title_band_candidate'
            box=[x,y,w,h]
            if any(_intersection(box,c['bbox'])/(area+c['bbox'][2]*c['bbox'][3]-_intersection(box,c['bbox']))>.9 for c in candidates):continue
            candidates.append({'bbox':box,'type':kind,'visual_style':style,'role_candidate':role,'status':'measured_candidate',
                'evidence':{'fill':{'status':'measured_uniform_component','rgb':color.tolist(),'dominant_fraction':round(fraction,4),
                    'sample_count':int(available.sum()),'method':'local_connected_color_component_excluding_OCR'},
                    'corner':corner,'stroke':{'status':'component_interior_only_no_border_assertion'},
                    'component_area_fraction':round(cv2.contourArea(contour)/area,4)}})
    filtered=[]
    for candidate in candidates:
        b=candidate['bbox']
        gradient_parent=next((s for s in report['shapes'] if s['evidence']['fill'].get('smooth_vertical_variation')
            and abs(s['bbox'][0]-b[0])<=5 and abs(s['bbox'][2]-b[2])<=10
            and s['bbox'][1]<=b[1] and b[1]+b[3]<=s['bbox'][1]+s['bbox'][3]
            and b[3]<s['bbox'][3]*.8),None)
        if gradient_parent:
            report['diagnostics'].append({'code':'gradient_component_fragment','bbox':b,
                'parent_source_id':gradient_parent['source_id'],
                'message':'Quantized gradient stripe is not a separate panel; retain uncertain measured parent appearance.'})
        else:filtered.append(candidate)
    candidates=filtered
    candidates.sort(key=lambda c:(c['bbox'][1],c['bbox'][0],-c['bbox'][2]*c['bbox'][3]))
    if len(candidates)>160:
        report['diagnostics'].append({'code':'filled_region_limit','detected':len(candidates),'kept':160})
        candidates=sorted(candidates,key=lambda c:-c['bbox'][2]*c['bbox'][3])[:160]
        candidates.sort(key=lambda c:(c['bbox'][1],c['bbox'][0]))
    for i,candidate in enumerate(candidates,1):
        candidate['id']='visual_region_%04d'%i
    report['regions']=candidates
    boxes=[s['bbox'] for s in report['shapes']]+[s['bbox'] for s in candidates]
    for item in report['shapes']+candidates:
        item['role_candidate'],basis=_role(item['bbox'],boxes,[c['bbox'] for c in candidates],width*height)
        item['evidence']['role']={'status':'geometric_candidate_not_semantic_classification','basis':basis}
    # Per-box foreground differs from the local modal background. Dark and
    # light ink are both eligible; OCR strings are never consulted or changed.
    for row in ocr_rows:
        x,y,w,h=_box(row['bbox'],width,height);crop=rgb[y:y+h,x:x+w].reshape(-1,3)
        background,_=_modal(crop,np)
        delta=np.max(np.abs(crop.astype(float)-background),axis=1)
        fg=crop[delta>=35]
        item={'visual_style':{},'evidence':{'status':'uncertain_foreground','method':'contrasting_modal_pixels_in_OCR_bbox'}}
        if len(fg)>=max(3,len(crop)*.005):
            foreground,support=_modal(fg,np)
            if support>=.20:
                item={'visual_style':{'text_color_rgb':_hex(foreground)},
                    'evidence':{'status':'measured_foreground_candidate','rgb':foreground.tolist(),
                        'background_rgb':background.tolist(),'sample_count':len(fg),'dominant_fraction':support,
                        'method':'contrasting_modal_pixels_in_OCR_bbox'}}
        report['text_styles'][row['id']]=item
    report['diagnostics'].extend([
        {'code':'appearance_not_semantics','message':'Color and position produce visual role candidates only; text and connection semantics are unchanged.'},
        {'code':'gradient_limit','message':'Nonuniform fill is reported uncertain; no global average, raster embedding or invented opacity is used.'},
        {'code':'radius_limit','message':'Only symmetric source corner evidence produces a circular radius; asymmetric corners remain uncertain.'}])
    report['counts']={'measured_existing_shapes':len(report['shapes']),'new_filled_regions':len(candidates),
        'text_foreground_estimates':sum(bool(x['visual_style']) for x in report['text_styles'].values()),
        'uncertain_fills':sum(s['evidence']['fill']['status']!='measured_uniform' for s in report['shapes'])}
    return report
