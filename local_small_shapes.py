"""Measured small circles and audited local OCR; no ordinal completion.

Original OCR rows, IDs, typography and geometry are never rewritten. Circle
background and its text remain independent editable elements sharing a region.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from geometry_analyzer import _dependencies, _load_image
from local_geometry import _classify, _dominant
from local_ocr import recognize_crops
from structured_ir.validator import validate_ir

MAX_CIRCLES=32
MIN_CONFIDENCE=.85


def _save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def _contains(a,b):
    return a[0]<=b[0] and a[1]<=b[1] and a[0]+a[2]>=b[0]+b[2] and a[1]+a[3]>=b[1]+b[3]


def _iou(a,b):
    area=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    return area/(a[2]*a[3]+b[2]*b[3]-area) if area else 0


def _hex(rgb):return '#'+''.join('%02X'%int(v) for v in rgb)


def _inspect_circle(rgb,box,cv2,np):
    x,y,w,h=map(int,box);patch=rgb[y:y+h,x:x+w]
    yy,xx=np.mgrid[:h,:w];radius=min(w,h)/2
    distance=np.sqrt((xx-(w-1)/2)**2+(yy-(h-1)/2)**2)
    ring=(distance>=radius*.63)&(distance<=radius*.80)
    fill=_dominant(patch[ring],np)
    difference=np.max(np.abs(patch.astype(float)-fill),axis=2)
    ink=(difference>=18)&(distance<radius*.73)
    count,labels,stats,_=cv2.connectedComponentsWithStats(ink.astype('uint8'),8)
    accepted=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=2]
    mask=np.isin(labels,accepted)
    evidence={'fill_rgb':fill,'fill_sample_count':int(ring.sum()),
        'foreground_pixel_count':int(mask.sum()),'method':'Original RGB annulus mode; central contrasting connected components.',
        'source_pixel_bbox':box}
    outer=(distance>=radius*.83)&(distance<=radius*.96)
    stroke=_dominant(patch[outer],np)
    evidence['stroke_rgb']=stroke
    evidence['distinct_stroke']=max(abs(a-b) for a,b in zip(fill,stroke))>=20
    if not accepted or mask.sum()>.45*(math.pi*radius*radius):
        return evidence,None,None,None
    sy,sx=np.nonzero(mask);ix,iy,iw,ih=int(sx.min()),int(sy.min()),int(sx.max()-sx.min()+1),int(sy.max()-sy.min()+1)
    foreground=_dominant(patch[mask],np)
    evidence['text_rgb']=foreground
    evidence['ink_bbox']=[x+ix,y+iy,iw,ih]
    # Separate border sampling; absence of a distinct border is explicit.
    return evidence,[x+ix,y+iy,iw,ih],foreground,mask


def detect_small_circles(image_path,ir):
    """Reuse circles plus bounded original-pixel contour detection below old limits."""
    validate_ir(ir);cv2,np=_dependencies();pixels,image=_load_image(image_path)
    if any(image[k]!=ir['canvas'][k] for k in ('width','height')):raise ValueError('Small-shape canvas mismatch.')
    if ir.get('provenance',{}).get('image_sha256',image['sha256'])!=image['sha256']:raise ValueError('Small-shape image mismatch.')
    rgb=cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB);limit=max(64,min(160,min(image['width'],image['height'])*.10))
    candidates=[{'bbox':e['bbox'][:],'existing_element_id':e['id'],'method':'existing_validated_circle'}
                for e in ir['elements'] if e['type']=='circle' and 8<=min(e['bbox'][2:]) and max(e['bbox'][2:])<=limit]
    gray=cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY)
    contours,_=cv2.findContours(cv2.Canny(gray,25,90),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    added=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if not 8<=min(w,h)<=max(w,h)<=limit:continue
        kind=_classify(contour,cv2,np)
        if not kind or kind[0]!='circle':continue
        box=[x,y,w,h]
        if any(_iou(box,c['bbox'])>.65 for c in candidates):continue
        if any('raw_text' in e and _contains(e['bbox'],box) for e in ir['elements']):continue
        added.append({'bbox':box,'existing_element_id':None,'method':'small_original_pixel_circular_contour',
                      'contour_evidence':kind[2]})
    for item in sorted(added,key=lambda i:-i['bbox'][2]*i['bbox'][3]):
        if any(_iou(item['bbox'],c['bbox'])>.65 for c in candidates):continue
        # A new filled circle needs a real contrast against its outside corner;
        # glyph-sized loops and outline-only candidates remain unpromoted.
        x,y,w,h=item['bbox'];evidence,ink,fg,mask=_inspect_circle(rgb,item['bbox'],cv2,np)
        corners=[rgb[y,x],rgb[y,x+w-1],rgb[y+h-1,x],rgb[y+h-1,x+w-1]]
        outside=np.median(corners,axis=0)
        if max(abs(outside-np.asarray(evidence['fill_rgb'])))<20:continue
        candidates.append(item)
    candidates.sort(key=lambda i:(i['bbox'][1],i['bbox'][0]))
    diagnostics=[]
    if len(candidates)>MAX_CIRCLES:
        diagnostics.append({'code':'small_circle_budget','detected':len(candidates),'processed':MAX_CIRCLES})
        candidates=candidates[:MAX_CIRCLES]
    for index,c in enumerate(candidates,1):
        x,y,w,h=c['bbox'];evidence,ink,fg,mask=_inspect_circle(rgb,c['bbox'],cv2,np)
        c.update(id='small_circle_%04d'%index,center=[x+w/2,y+h/2],radius=min(w,h)/2,
                 measurements=evidence,ink_bbox=ink,fill_color=_hex(evidence['fill_rgb']),
                 text_color_rgb=_hex(fg) if fg else None,status='measured_circle',
                 score_type='geometric_evidence_not_probability')
        c['existing_text_ids']=[e['id'] for e in ir['elements'] if 'raw_text' in e and _contains(c['bbox'],e['bbox'])]
        if not ink:c['status']='no_reliable_internal_ink'
    return {'source_image':image,'coordinate_system':'top-left','bbox_format':'xywh','circles':candidates,
            'diagnostics':diagnostics,'api_calls':0,'budget':{'max_circles':MAX_CIRCLES,'max_ocr_batches':1}}


def _crops(image_path,report,folder):
    cv2,np=_dependencies();pixels,_=_load_image(image_path);rgb=cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB)
    folder.mkdir(exist_ok=False);requests=[]
    from PIL import Image
    for c in report['circles']:
        if not c['ink_bbox'] or c['existing_text_ids']:continue
        evidence,ink,fg,mask=_inspect_circle(rgb,c['bbox'],cv2,np)
        x,y,w,h=c['bbox'];ix,iy,iw,ih=ink
        pad=max(2,round(ih*.18));scale=4
        # Contrast normalization is based only on measured ink, never a glyph
        # template or target number. Preserve original and normalized crops.
        original=Image.fromarray(rgb[y:y+h,x:x+w]);original.save(folder/(c['id']+'_original.png'))
        normalized=np.full((ih+2*pad,iw+2*pad,3),255,dtype='uint8')
        selected=mask[iy-y:iy-y+ih,ix-x:ix-x+iw]
        normalized[pad:pad+ih,pad:pad+iw][selected]=0
        image=Image.fromarray(normalized).resize(((iw+2*pad)*scale,(ih+2*pad)*scale),Image.Resampling.LANCZOS)
        target=folder/(c['id']+'_ocr.png');image.save(target)
        requests.append({'id':c['id'],'path':str(target.resolve()),'original_crop':str((folder/(c['id']+'_original.png')).resolve()),
            'source_bbox':ink,'circle_bbox':c['bbox'],'scale':scale,'padding_before_scale':pad,
            'transform':'x_original=x_crop/scale-padding+source_bbox.x; y likewise',
            'preprocessing':'Observed foreground connected components -> black ink on white; 4x Lanczos'})
    return requests


def apply_small_shape_results(ir,report,ocr_result=None):
    """Pure append-only enrichment with auditable parent/child links."""
    result=deepcopy(ir);report=deepcopy(report)
    used={result['canvas']['id']}|{e['id'] for k in ('elements','regions','connections') for e in result[k]}
    by_id={e['id']:e for e in result['elements']}
    recognized={}
    if ocr_result:
        for request,pair in zip(ocr_result['requests'],ocr_result['engine_result']):
            recognized[request['id']]={'text':pair[0],'confidence':pair[1],'request':deepcopy(request)}
    report['associations']=[];report['added_element_ids']=[]
    top=max([e.get('z_index',0) for e in result['elements']]+[0])+1
    order=max([e.get('reading_order',0) for e in result['elements']]+[0])+1
    for circle in report['circles']:
        existing=circle['existing_element_id'];entry={'circle_candidate_id':circle['id'],'circle_element_id':existing,
            'text_element_id':None,'status':'unresolved'}
        report['associations'].append(entry)
        if not circle['ink_bbox']:
            entry['reason']='no_reliable_internal_ink';continue
        if circle['existing_text_ids']:
            entry.update(status='existing_text_retained',text_element_ids=circle['existing_text_ids']);continue
        rec=recognized.get(circle['id']);circle['ocr']=rec
        if not rec:
            entry['reason']='no_local_ocr_result';continue
        text,confidence=rec['text'],rec['confidence']
        if not text or len(text)>4 or not text.isalnum() or confidence<MIN_CONFIDENCE:
            entry.update(reason='uncertain_local_ocr',raw_text=text,confidence=confidence);continue
        if not existing:
            ident='local_'+circle['id'];region=ident+'_region'
            if ident in used or region in used:raise ValueError('Small-circle ID collision.')
            parents=[r for r in result['regions'] if _contains(r['bbox'],circle['bbox'])]
            deepest=[r for r in parents if all(r is s or _contains(s['bbox'],r['bbox']) and s['bbox']!=r['bbox'] for s in parents)]
            parent=deepest[0]['id'] if len(deepest)==1 else None
            result['regions'].append({'id':region,'type':'unknown','bbox':circle['bbox'][:],'parent_id':parent,'children':[],
                'uncertainty':'Measured circle-text spatial group; semantics are not inferred.'})
            element={'id':ident,'type':'circle','bbox':circle['bbox'][:],'region_id':region,'z_index':top,
                'visual_style':{'fill_color':circle['fill_color'],'stroke':'none'}}
            if circle['measurements'].get('distinct_stroke'):
                element['visual_style']['stroke_color']=_hex(circle['measurements']['stroke_rgb'])
            result['elements'].append(element);by_id[ident]=element;used.update([ident,region])
            report['added_element_ids'].append(ident);existing=ident;entry['circle_element_id']=ident
        owner=by_id[existing].get('region_id')
        ident='local_badge_text_'+circle['id'].rsplit('_',1)[1]
        if ident in used:raise ValueError('Small-text ID collision.')
        used.add(ident);box=circle['ink_bbox'][:]
        # Retain measured ink extent with the same measured circular center.
        # This moves only the newly recovered glyph, never an original OCR row.
        box[:2]=[circle['center'][0]-box[2]/2,circle['center'][1]-box[3]/2]
        style={'text_color_rgb':circle['text_color_rgb'],'font_family':'serif','font_size':max(6,min(80,box[3])),
               'bold':True,'line_spacing':1.25}
        new={'id':ident,'type':'text','bbox':box,'region_id':owner,'raw_text':text,'text_type':'plain',
            'reading_order':order,'confidence':confidence,'visual_style':style,'z_index':top+1,
            'uncertainty':'Literal local OCR; source crop, confidence and measured ink bbox retained in small_shape_detection.json.'}
        result['elements'].append(new);report['added_element_ids'].append(ident);order+=1
        entry.update(status='recovered_from_local_ocr',text_element_id=ident,region_id=owner,
            raw_text=text,confidence=confidence,source_ink_bbox=circle['ink_bbox'],render_bbox=box,
            centering='New text centered on the measured circle; source ink bbox retained separately.')
    for region in result['regions']:
        region['children']=[r['id'] for r in result['regions'] if r['parent_id']==region['id']]
        region['children'] += [e['id'] for e in result['elements'] if e.get('region_id')==region['id']]
    result['layered']=True
    validate_ir(result)
    report['status']='partial'
    report['counts']={'detected_circles':len(report['circles']),
        'recovered_texts':sum(a['status']=='recovered_from_local_ocr' for a in report['associations']),
        'existing_text_associations':sum(a['status']=='existing_text_retained' for a in report['associations']),
        'unresolved_circles':sum(a['status']=='unresolved' for a in report['associations'])}
    report['preservation']={'original_text_ids_bboxes_styles':True,'original_shapes':True,'ordinal_completion':False}
    return result,report


def _fit_new_text(ir,report,output,*,xelatex,timeout):
    """Measure only appended labels; all historical font choices stay untouched."""
    from structured_ir.converter import convert_ir
    from text_measurement import measure_text_candidates
    ids={a['text_element_id']:a for a in report['associations'] if a['status']=='recovered_from_local_ocr'}
    if not ids:return ir,{'status':'no_new_text','compile_attempts':0}
    scene,conversion=convert_ir(ir)
    if not conversion['complete']:return ir,{'status':'unavailable','reason':'source_scene_partial'}
    scene['nodes']=[n for n in scene['nodes'] if n['id'] in ids]
    scene['edges']=[];scene['labels']=[]
    for node in scene['nodes']:node['wrap']=False
    try:
        metrics=measure_text_candidates(scene,Path(output)/'badge_typography',xelatex=xelatex,timeout=timeout,
            families_by_id={i:['serif'] for i in ids})
    except (ValueError,OSError) as exc:
        return ir,{'status':'unavailable','reason':str(exc)}
    _save(Path(output)/'badge_text_measurement.json',metrics)
    if metrics['status']!='measured':return ir,metrics
    result=deepcopy(ir);by_id={e['id']:e for e in result['elements']};selected=[]
    for ident,association in ids.items():
        m=metrics['items'][ident]['candidates'].get('serif')
        if not m or m['height_px']<=0 or m['width_px']<=0:continue
        circle=by_id[association['circle_element_id']];diameter=min(circle['bbox'][2:])
        desired=association['source_ink_bbox'][3]
        ratio=min(desired/m['height_px'],diameter*.75/m['width_px'],diameter*.75/m['height_px'])
        size=max(6,min(240,m['effective_font_size']*ratio))
        by_id[ident]['visual_style'].update(font_size=round(min(80,size),5),font_scale=round(max(1,size/80),7))
        selected.append({'element_id':ident,'effective_font_size':size,'target_ink_height':desired,
                         'measured_reference':m,'source_circle_diameter':diameter})
    validate_ir(result)
    return result,{'status':'measured_new_badge_text_only','compile_attempts':metrics['compile_attempts'],
        'selections':selected,'report_path':str(Path(output)/'badge_text_measurement.json'),
        'original_typography_unchanged':True}


def recover_small_shapes(image_path,ir,output,*,ocr_python=None,ocr_timeout=300,
                         xelatex='xelatex',compile_timeout=60,measure_text=False):
    """Detect, recognize in one CPU batch and append independent editable text."""
    output=Path(output);report=detect_small_circles(image_path,ir);ocr_result=None
    pending=[c for c in report['circles'] if c['ink_bbox'] and not c['existing_text_ids']]
    if pending:
        requests=_crops(image_path,report,output/'small_shape_crops')
        try:
            ocr_result=recognize_crops(image_path,requests,output/'small_shape_ocr',python=ocr_python,timeout=ocr_timeout)
            report['ocr_evidence_path']=str(output/'small_shape_ocr/crop_ocr_raw.json')
        except (ValueError,OSError) as exc:
            report['diagnostics'].append({'code':'local_crop_ocr_failed','message':str(exc),'action':'No text invented.'})
    result,report=apply_small_shape_results(ir,report,ocr_result)
    report['typography']={'status':'source_ink_height_estimate','reason':'measurement_not_requested'}
    if measure_text:
        result,report['typography']=_fit_new_text(result,report,output,xelatex=xelatex,timeout=compile_timeout)
    _save(output/'small_shape_detection.json',report)
    return result,report
