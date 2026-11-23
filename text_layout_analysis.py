"""Source-pixel ink evidence and measured typography conflicts, not OCR repair.

Classification selects typography only. Strings, IDs, coordinates and graph
relations are never rewritten. Mathematical rendering is limited to exact
Unicode-to-TeX equivalents; ambiguous OCR formula tokens remain literal.
"""
from copy import deepcopy
import math
import re
import statistics

from geometry_analyzer import _dependencies, _load_image

_CJK = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')
_CODE = re.compile(r'\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b|\b__[A-Za-z0-9_]+__\b|\b[A-Za-z_]\w*\.[A-Za-z_]\w*|\b[A-Za-z_]\w*\s*[\[(]|\b[A-Za-z_]\w*\s*=(?!=)')
_MATH_SYMBOLS = {'∈':r'\in','×':r'\times','ℝ':r'\mathbb{R}','α':r'\alpha','β':r'\beta',
                 'γ':r'\gamma','δ':r'\delta','θ':r'\theta','σ':r'\sigma','≤':r'\le',
                 '≥':r'\ge','≠':r'\ne','±':r'\pm','⋅':r'\cdot'}
_SUPERSCRIPT = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺','0123456789-+')
_SUBSCRIPT = str.maketrans('₀₁₂₃₄₅₆₇₈₉₋₊','0123456789-+')
_SCRIPT_TOKEN = re.compile(r'[A-Za-z0-9]+(?:[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+|[₀₁₂₃₄₅₆₇₈₉₋₊]+)')


def equivalent_math_segments(text):
    """Only render printed symbols/scripts; never infer E=in, IR=R, or '^'."""
    segments=[]; plain=[]; position=0
    def flush():
        if plain:
            segments.append({'type':'text','content':''.join(plain)});plain.clear()
    while position<len(text):
        match=_SCRIPT_TOKEN.match(text,position)
        if match:
            token=match.group();base=re.match(r'[A-Za-z0-9]+',token).group()
            suffix=token[len(base):];sup=suffix[0] in '⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺'
            flush();segments.append({'type':'math','content':base+('^' if sup else '_')+'{'+suffix.translate(_SUPERSCRIPT if sup else _SUBSCRIPT)+'}'})
            position=match.end();continue
        character=text[position]
        if character in _MATH_SYMBOLS:
            flush();segments.append({'type':'math','content':_MATH_SYMBOLS[character]})
        else:plain.append(character)
        position+=1
    flush()
    return segments if any(s['type']=='math' for s in segments) else None


def classify_text(text, *, text_type='plain', heading=False):
    """Conservative typography category; classification never corrects OCR."""
    chinese=bool(_CJK.search(text)); latin=bool(re.search(r'[A-Za-z]',text))
    diagnostics=[]; segments=None
    ambiguous_formula=bool(re.search(r'[A-Za-z][A-Za-z0-9]*\s+[Ee]\s+(?:IR|R)\d',text)
                           or re.search(r'\d+x\d+',text) or re.search(r'[A-Za-z]\s*\^',text)
                           or re.fullmatch(r'\s*[A-Za-z]\s*=\s*[-+]?\d+(?:\.\d+)?\s*',text))
    if text_type=='math':
        kind='math_expression';families=['serif']
    elif text_type=='code' or re.search(r'\b[A-Za-z_]\w*\s*=\s*[\'\"]',text):
        kind='mixed_code_chinese' if chinese else 'python_code';families=['mono']
    else:
        segments=equivalent_math_segments(text)
        if segments:
            kind='math_uncertain' if ambiguous_formula else 'mixed_math_text' if chinese else 'math_expression'
            families=['serif','sans']
            diagnostics.append({'code':'literal_unicode_math_only','message':'Only printed Unicode symbols/scripts receive equivalent TeX; all other OCR tokens remain literal.'})
        elif ambiguous_formula:
            kind='math_uncertain';families=['serif','sans']
        elif _CODE.search(text):
            kind='mixed_code_chinese' if chinese else 'python_code';families=['mono']
        elif chinese:
            kind='chinese_title' if heading else 'mixed_chinese_english' if latin else 'chinese_body'
            families=['sans','serif'] if heading else ['serif','sans']
        else:
            kind='english_title' if heading else 'english_prose';families=['sans','serif'] if heading else ['serif','sans']
    if ambiguous_formula and text_type not in {'code','math'}:
        diagnostics.append({'code':'unresolved_ocr_formula','message':'Possible formula tokens remain literal; missing operators, scripts or dimension symbols cannot be recovered by typography.'})
    if re.search(r'[_\.\-][_.\-](?:[A-Za-z])|[A-Za-z][_\.\-]+\s*\(',text):
        diagnostics.append({'code':'identifier_ocr_uncertainty','message':'Identifier punctuation is preserved exactly; font changes do not establish OCR correctness.'})
    return {'text_kind':kind,'font_candidates':families,'render_segments':segments,'diagnostics':diagnostics}


def _box(value,width=None,height=None):
    if (not isinstance(value,(list,tuple)) or len(value)!=4 or
        any(type(v) not in (int,float) or not math.isfinite(v) for v in value) or value[2]<=0 or value[3]<=0):
        raise ValueError('Text bbox must contain four finite numbers with positive extent.')
    x,y,w,h=value
    if width is not None and (x<0 or y<0 or x+w>width or y+h>height):
        raise ValueError('Text bbox outside source image.')
    return list(value)


def _modal(values,np):
    keys=(values.astype('int32')//8)
    keys=keys[:,0]*1024+keys[:,1]*32+keys[:,2]
    unique,counts=np.unique(keys,return_counts=True)
    return np.median(values[keys==unique[counts.argmax()]],axis=0)


def _ink(rgb,box,style,cv2,np):
    x,y,w,h=box;left,top=math.floor(x),math.floor(y)
    right,bottom=math.ceil(x+w),math.ceil(y+h)
    crop=rgb[top:bottom,left:right];height,width=crop.shape[:2]
    values=crop.reshape(-1,3);background=_modal(values,np)
    evidence=style.get('evidence',{}) if isinstance(style,dict) else {}
    foreground=evidence.get('rgb') if evidence.get('status')=='measured_foreground_candidate' else None
    if foreground is not None:
        foreground=np.array(foreground,dtype=float)
        proposed=evidence.get('background_rgb')
        if proposed is not None:background=np.array(proposed,dtype=float)
    distance=np.max(np.abs(crop.astype(float)-background),axis=2)
    threshold=18.
    mask=distance>=threshold
    if foreground is not None and np.max(np.abs(foreground-background))>=30:
        contrast=float(np.max(np.abs(foreground-background)))
        threshold=max(18,contrast*.28)
        mask=(distance>=threshold)&(np.max(np.abs(crop.astype(float)-foreground),axis=2)<distance)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
    accepted=[];rejected=0
    for i in range(1,count):
        cx,cy,cw,ch,area=map(int,stats[i])
        line=(ch<=max(2,height*.045) and cw>=width*.65) or (cw<=2 and ch>=height*.85)
        if area<2 or line:rejected+=1;continue
        accepted.append((cx,cy,cw,ch,area))
    if not accepted:
        return None,{'status':'unavailable_no_reliable_ink','component_count':0,'rejected_components':rejected,
            'contrast_threshold_rgb':threshold,'message':'No ink measurement; OCR bbox is retained, not presented as measured ink.'}
    x0=min(a[0] for a in accepted);y0=min(a[1] for a in accepted)
    x1=max(a[0]+a[2] for a in accepted);y1=max(a[1]+a[3] for a in accepted)
    ink=[left+x0,top+y0,x1-x0,y1-y0]
    occupancy=sum(a[4] for a in accepted)/(width*height)
    status='measured_component_enclosure' if occupancy<=.65 else 'uncertain_dense_foreground'
    info={'status':status,'component_count':len(accepted),'rejected_components':rejected,
          'ink_pixel_count':sum(a[4] for a in accepted),'foreground_fraction':occupancy,
          'background_rgb':background.tolist(),'contrast_threshold_rgb':threshold,
          'method':'Original RGB contrast and connected components within the unchanged OCR bbox; thin border-like components excluded.'}
    return (ink if status=='measured_component_enclosure' else None),info


def analyze_source_text(image_path,ocr_rows,ir,fusion,visual_report):
    """Return measured source ink and conservative categories keyed by IR ID."""
    cv2,np=_dependencies();pixels,image=_load_image(image_path)
    width,height=image['width'],image['height']
    if ir['canvas']['width']!=width or ir['canvas']['height']!=height:
        raise ValueError('Source dimensions differ from IR canvas.')
    expected=ir.get('provenance',{}).get('image_sha256')
    if expected and expected!=image['sha256']:raise ValueError('Source image hash differs from IR provenance.')
    if visual_report.get('image',{}).get('sha256')!=image['sha256']:
        raise ValueError('Visual text evidence does not match source image.')
    rgb=cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB)
    rows={r['id']:r for r in ocr_rows}
    if len(rows)!=len(ocr_rows):raise ValueError('OCR source IDs must be unique.')
    elements={e['id']:e for e in ir['elements']}
    matches={m['element_id']:m['source_id'] for m in fusion.get('text_matches',[])}
    if len(matches)!=len(fusion.get('text_matches',[])) or set(matches.values())!=set(rows) or len(matches)!=len(rows):
        raise ValueError('Text fusion must preserve a one-to-one mapping of all OCR rows.')
    heights=[r['bbox'][3] for r in ocr_rows];median=statistics.median(heights) if heights else 1
    result={'source_image_sha256':image['sha256'],'coordinate_system':'top-left','bbox_format':'xywh',
            'items':{},'diagnostics':[],'api_calls':0,'text_rewritten':False}
    for ident,source in matches.items():
        if ident not in elements or source not in rows:raise ValueError('Text fusion reference does not resolve.')
        e,r=elements[ident],rows[source]
        box=_box(r['bbox'],width,height)
        if r['text']!=e.get('raw_text') or box!=e['bbox']:
            raise ValueError('Text content or bbox changed before source analysis.')
        ink,evidence=_ink(rgb,box,visual_report.get('text_styles',{}).get(source,{}),cv2,np)
        # Relative visual prominence and location, never page-specific words.
        heading=(box[3]>=median*1.25 and box[1]<height*.18)
        heading=heading or (box[3]>=median*1.10 and e.get('region_id') is not None and
            any(g['id']==e['region_id'] and box[1]-g['bbox'][1]<=max(box[3],g['bbox'][3]*.12) for g in ir['regions']))
        kind=classify_text(r['text'],text_type=e.get('text_type','plain'),heading=heading)
        item={'ocr_id':source,'raw_text':r['text'],'bbox':box,'ink_bbox':ink,'ink_status':evidence['status'],
              'ink_evidence':evidence,'text_kind':kind['text_kind'],'font_candidates':kind['font_candidates'],
              'diagnostics':kind['diagnostics'],'is_heading_candidate':heading}
        if kind['render_segments'] is not None:item['render_segments']=kind['render_segments']
        if '\n' in r['text']:
            item['diagnostics'].append({'code':'observed_multiline','message':'Existing newlines preserved; no additional wrapping inferred.'})
        result['items'][ident]=item
    result['counts']={'texts':len(result['items']),'measured_ink':sum(i['ink_bbox'] is not None for i in result['items'].values()),
                     'uncertain_math':sum(i['text_kind']=='math_uncertain' for i in result['items'].values())}
    return result


def _inside(a,b):
    return a[0]>=b[0] and a[1]>=b[1] and a[0]+a[2]<=b[0]+b[2] and a[1]+a[3]<=b[1]+b[3]


def _segment_hits_box(a,b,box):
    lower,upper=0.,1.
    for axis in (0,1):
        delta=b[axis]-a[axis];minimum=box[axis];maximum=minimum+box[axis+2]
        if delta==0:
            if not minimum<=a[axis]<=maximum:return False
        else:
            lo,hi=sorted(((minimum-a[axis])/delta,(maximum-a[axis])/delta))
            lower,upper=max(lower,lo),min(upper,hi)
            if lower>upper:return False
    return True


def analyze_text_collisions(ir,rendered_boxes):
    """Inspect measured rendered extents; no source box, graph or layout edits."""
    elements={e['id']:e for e in ir['elements']};regions={g['id']:g for g in ir['regions']}
    boxes={k:_box(v) for k,v in rendered_boxes.items()}
    if any(k not in elements or 'raw_text' not in elements[k] for k in boxes):
        raise ValueError('Rendered text boxes must reference text-bearing IR elements.')
    codes=['text_overlap','same_line_gap','container_overflow','border_proximity','page_overflow','graphic_boundary','connector_intersection']
    report={'counts':{k:0 for k in codes},'issues':[],'by_element':{k:[] for k in boxes},'diagnostics':[],
            'measured_boxes':deepcopy(boxes),'api_calls':0,'multiline_internal_spacing':'Whole-node measurements cannot resolve individual line spacing; source newlines are preserved.'}
    def issue(code,ids,**extra):
        severity='error' if code in {'text_overlap','container_overflow','page_overflow'} else 'warning'
        entry={'code':code,'element_ids':ids,'severity':severity,**extra};report['issues'].append(entry);report['counts'][code]+=1
        for ident in ids:
            if ident in report['by_element']:report['by_element'][ident].append(entry)
    canvas=[0,0,ir['canvas']['width'],ir['canvas']['height']]
    for ident,box in boxes.items():
        if not _inside(box,canvas):issue('page_overflow',[ident],bbox=box)
        owner=regions.get(elements[ident].get('region_id'))
        if owner:
            bounds=owner['bbox']
            margin=min(box[0]-bounds[0],box[1]-bounds[1],bounds[0]+bounds[2]-box[0]-box[2],bounds[1]+bounds[3]-box[1]-box[3])
            if margin<-.01:issue('container_overflow',[ident],region_id=owner['id'],minimum_margin_px=margin)
            elif margin<min(3,box[3]*.12):issue('border_proximity',[ident],region_id=owner['id'],minimum_margin_px=margin)
    pairs=list(boxes.items())
    for index,(ida,a) in enumerate(pairs):
        for idb,b in pairs[index+1:]:
            iw=min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]);ih=min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1])
            if iw>.5 and ih>.5:issue('text_overlap',[ida,idb],intersection_area_px=iw*ih)
            elif iw<=0 and ih>min(a[3],b[3])*.6 and -iw<min(a[3],b[3])*.12:
                issue('same_line_gap',[ida,idb],gap_px=-iw)
    for obj in ir['elements']+ir.get('connections',[]):
        if 'source_id' in obj and not obj.get('points'):
            report['diagnostics'].append({'code':'unmeasured_connection_route','graphic_id':obj['id'],
                'message':'No supplied rendered path; semantic endpoint proximity is not used as a guessed route.'})
        if obj.get('type') in {'connector','arrow'} or ('source_id' in obj and obj.get('points')):
            points=obj.get('points',[])
            for ident,box in boxes.items():
                if any(_segment_hits_box(a,b,box) for a,b in zip(points,points[1:])):
                    issue('connector_intersection',[ident],graphic_id=obj['id'])
        elif obj.get('type') in {'rectangle','rounded_rectangle','diamond'}:
            style=obj.get('visual_style',{})
            if style.get('stroke')=='none' and 'stroke_color' not in style:continue
            x,y,w,h=obj['bbox']
            vertices=([(x+w/2,y),(x+w,y+h/2),(x+w/2,y+h),(x,y+h/2)] if obj['type']=='diamond' else
                      [(x,y),(x+w,y),(x+w,y+h),(x,y+h)])
            for ident,box in boxes.items():
                if obj['id']==ident:continue
                if any(_segment_hits_box(a,b,box) for a,b in zip(vertices,vertices[1:]+vertices[:1])):
                    issue('graphic_boundary',[ident],graphic_id=obj['id'],shape_type=obj['type'])
    report['counts']['hard_conflicts']=sum(report['counts'][k] for k in ('text_overlap','container_overflow','page_overflow'))
    report['counts']['total_conflicts']=len(report['issues'])
    return report
