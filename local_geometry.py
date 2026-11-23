"""Measured local geometry candidates; never infer graph semantics from proximity.

The original analyzer is retained verbatim in ``base_geometry``. This adapter
adds conservative contour candidates without its nested-cell suppression, and
records measured colours plus quantization error against the existing palette.
OCR boxes are exclusions for colour statistics, not erasers of image borders.
"""
from copy import deepcopy
import math

from geometry_analyzer import analyze_image, _load_image, _dependencies
from structured_tikz.schema import COLORS
from text_region_detector import normalize_text_regions

MAX_SHAPES = 160


def _intersection(a, b):
    return max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))


def _contains(a, b):
    return a[0] <= b[0] and a[1] <= b[1] and a[0]+a[2] >= b[0]+b[2] and a[1]+a[3] >= b[1]+b[3]


def deduplicate_shapes(candidates):
    """Merge near-identical contour boundaries, never mere containment."""
    kept = []
    diagnostics = []
    for item in sorted(candidates, key=lambda c: (-c['score'], c['bbox'][1], c['bbox'][0], -c['bbox'][2]*c['bbox'][3])):
        box = item['bbox']
        duplicate = None
        for other in kept:
            old = other['bbox']
            area = _intersection(box, old)
            iou = area/(box[2]*box[3]+old[2]*old[3]-area)
            edges = lambda b: [b[0], b[1], b[0]+b[2], b[1]+b[3]]
            edge_delta = max(abs(a-b) for a,b in zip(edges(box), edges(old)))
            if iou >= .94 or (iou >= .75 and edge_delta <= 3):
                duplicate = other
                break
        if duplicate is None:
            kept.append(deepcopy(item))
        else:
            duplicate['evidence']['methods'] = sorted(set(duplicate['evidence']['methods']+item['evidence']['methods']))
            diagnostics.append({'code': 'duplicate_shape_boundary', 'bbox': box, 'retained_bbox': duplicate['bbox']})
    kept.sort(key=lambda c: (c['bbox'][1], c['bbox'][0], -c['bbox'][2]*c['bbox'][3]))
    return kept, diagnostics


def quantize_color(rgb):
    """Only approximate rendering colour; measured RGB is kept separately."""
    choices = {name: tuple(int(value[i:i+2], 16) for i in (0,2,4)) for name,value in COLORS.items()}
    name = min(choices, key=lambda n: sum((a-b)**2 for a,b in zip(rgb,choices[n])))
    error = math.sqrt(sum((a-b)**2 for a,b in zip(rgb, choices[name])))
    return {'rgb': [int(v) for v in rgb], 'palette_name': name, 'rgb_distance': round(error,3),
            'status': 'palette_approximation', 'meaning': 'Measured pixels; finite Scene palette is not exact colour reproduction.'}


def _classify(contour, cv2, np):
    x,y,w,h = cv2.boundingRect(contour)
    perimeter = cv2.arcLength(contour,True)
    area = cv2.contourArea(contour)
    if perimeter <= 0 or area <= 0:
        return None
    quad = cv2.approxPolyDP(contour, .018*perimeter, True)
    ratio = area/(w*h)
    circularity = 4*math.pi*area/(perimeter*perimeter)
    if len(quad)==4 and cv2.isContourConvex(quad):
        points = quad.reshape(-1,2).astype(float)
        normalized = (points-[x,y])/[max(1,w-1),max(1,h-1)]
        midpoints = np.array([[.5,0],[1,.5],[.5,1],[0,.5]])
        if .38 <= ratio <= .63 and all(min(np.linalg.norm(p-q) for q in midpoints) < .18 for p in normalized):
            return 'diamond', .86, {'area_to_bbox':round(ratio,4),'vertices':points.tolist()}
        sides = np.roll(points,-1,axis=0)-points
        lengths = np.linalg.norm(sides,axis=1)
        axis_error = np.max(np.min(np.abs(sides),axis=1)/np.maximum(lengths,1))
        if ratio >= .83 and axis_error <= .10:
            boundary=contour.reshape(-1,2).astype(float)
            corners=np.array([[x,y],[x+w-1,y],[x+w-1,y+h-1],[x,y+h-1]])
            gaps=[float(np.min(np.linalg.norm(boundary-corner,axis=1))) for corner in corners]
            rounded=min(gaps)>max(2,min(w,h)*.025)
            return ('rounded_rectangle' if rounded else 'rectangle'), .88, {
                'area_to_bbox':round(ratio,4),'axis_error':round(float(axis_error),4),
                'bbox_corner_gaps_px':[round(v,3) for v in gaps]}
    if circularity >= .82 and .9 <= w/h <= 1.11 and .65 <= ratio <= .85:
        return 'circle', .82, {'circularity':round(circularity,4),'area_to_bbox':round(ratio,4)}
    if 5 <= len(quad) <= 12 and .82 <= ratio <= .99 and cv2.isContourConvex(quad):
        # Rounded rectangles retain long axis-aligned straight sides.
        p = quad.reshape(-1,2).astype(float)
        delta = np.roll(p,-1,axis=0)-p
        length = np.linalg.norm(delta,axis=1)
        axis = np.min(np.abs(delta),axis=1)/np.maximum(length,1) < .08
        if float(length[axis].sum()/length.sum()) >= .60:
            return 'rounded_rectangle', .74, {'area_to_bbox':round(ratio,4),'corner_classification':'rounded_candidate_not_exact_radius'}
    return None


def _dominant(values, np):
    if not len(values):
        return None
    # Quantized modal colour tolerates antialiasing without averaging dark ink
    # into an otherwise pale fill. The reported representative is real pixels.
    codes = values.astype('int32')//8
    key = codes[:,0]*1024+codes[:,1]*32+codes[:,2]
    keys, counts = np.unique(key, return_counts=True)
    group = values[key == keys[counts.argmax()]]
    return np.median(group,axis=0).round().astype(int).tolist()


def _measure_style(pixels, box, ocr_mask, cv2, np):
    x,y,w,h = box
    rgb = cv2.cvtColor(pixels[y:y+h,x:x+w],cv2.COLOR_BGR2RGB)
    excluded = ocr_mask[y:y+h,x:x+w] > 0
    inset = max(2,min(8,int(min(w,h)*.08)))
    interior = np.zeros((h,w),dtype=bool)
    if h > 2*inset and w > 2*inset:
        interior[inset:-inset,inset:-inset] = True
    interior &= ~excluded
    fill = _dominant(rgb[interior],np)
    ring = np.ones((h,w),dtype=bool)
    if h > 6 and w > 6:
        ring[3:-3,3:-3] = False
    ring &= ~excluded
    border_values = rgb[ring]
    if fill is not None and len(border_values):
        different = np.linalg.norm(border_values.astype(float)-fill,axis=1) >= 18
        if different.any(): border_values=border_values[different]
    stroke = _dominant(border_values,np)
    measured = {'fill': quantize_color(fill) if fill is not None else None,
                'stroke': quantize_color(stroke) if stroke is not None else None,
                'method':'modal_original_pixels_excluding_OCR_bboxes',
                'fill_sample_count':int(interior.sum()),'stroke_sample_count':len(border_values)}
    style={'fill':measured['fill']['palette_name'] if fill is not None else 'none',
           'stroke':measured['stroke']['palette_name'] if stroke is not None else 'dark_blue'}
    return style,measured


def analyze_local_geometry(image_path, ocr_rows):
    """Return original pixel geometry, preserved base result and uncertainty.

Arrow candidates remain geometric hypotheses; this function never creates a
semantic source/target edge, guesses arrow direction, or recognizes OCR text.
"""
    cv2,np = _dependencies()
    pixels,image = _load_image(image_path)
    width,height=image['width'],image['height']
    supplied=[{'bbox_px':r['bbox'],'source':'ocr','confidence':r.get('confidence')} for r in ocr_rows]
    # Validate before calling the base analyzer, which intentionally catches
    # detector errors. A broken adapter must not silently become an empty OCR.
    normalize_text_regions(supplied,width,height)
    base=analyze_image(image_path,text_regions=supplied)
    ocr_mask=np.zeros((height,width),dtype=np.uint8)
    for row in supplied:
        x,y,w,h=row['bbox_px']
        ocr_mask[int(y):min(height,math.ceil(y+h)),int(x):min(width,math.ceil(x+w))]=255
    gray=cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),25,90)
    # Preserve original boundaries. OCR coverage is used to reject glyph-sized
    # contours only, never to erase a border that crosses a text rectangle.
    masks=[('low_threshold_canny',edges)]
    hsv=cv2.cvtColor(pixels,cv2.COLOR_BGR2HSV)
    chromatic=cv2.inRange(hsv,np.array([0,12,100]),np.array([179,255,255]))
    chromatic=cv2.morphologyEx(chromatic,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    masks.append(('chromatic_closed_component',chromatic))
    candidates=[]; triangles=[]; filtered_glyphs=0
    min_w,min_h=max(18,width*.006),max(14,height*.005)
    for method,mask in masks:
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour)
            perimeter=cv2.arcLength(contour,True)
            if perimeter <= 0: continue
            approx=cv2.approxPolyDP(contour,.035*perimeter,True)
            if len(approx)==3 and 5<=w<=max(35,width*.015) and 5<=h<=max(35,height*.015):
                if not any(_intersection([x,y,w,h],r['bbox'])/(w*h)>.5 for r in ocr_rows):
                    triangles.append({'bbox':[x,y,w,h],'vertices':approx.reshape(-1,2).tolist(),'method':method})
            if w<min_w or h<min_h or w*h>width*height*.98 or cv2.contourArea(contour)/(w*h)<.35:
                continue
            if any(_intersection([x,y,w,h],r['bbox'])/(w*h)>.75 and w*h<=1.5*r['bbox'][2]*r['bbox'][3] for r in ocr_rows):
                filtered_glyphs+=1
                continue
            classified=_classify(contour,cv2,np)
            if classified is None: continue
            kind,score,evidence=classified
            candidates.append({'bbox':[x,y,w,h],'type':kind,'score':score,
                'evidence':dict(evidence,methods=[method],meaning='geometric_candidate_not_semantic_node')})
    for rectangle in base['rectangles']:
        q=rectangle['bbox_px']; box=[q[k] for k in ('x','y','width','height')]
        methods=rectangle['evidence']['methods']
        kind='rectangle' if 'canny_four_corner_axis_aligned_contour' in methods else 'unknown'
        candidates.append({'bbox':box,'type':kind,'score':min(.70,rectangle['score']),
            'evidence':{'methods':methods,'meaning':'base_analyzer_candidate; chromatic bbox does not prove a rectangle'}})
    shapes,diagnostics=deduplicate_shapes(candidates)
    if len(shapes)>MAX_SHAPES:
        diagnostics.append({'code':'shape_resource_limit','detected':len(shapes),'kept':MAX_SHAPES})
        shapes=sorted(shapes,key=lambda s:(-s['score'],-s['bbox'][2]*s['bbox'][3]))[:MAX_SHAPES]
        shapes.sort(key=lambda s:(s['bbox'][1],s['bbox'][0]))
    for n,shape in enumerate(shapes,1):
        shape['id']='shape_%04d'%n
        shape['score_type']='heuristic_rank_not_probability'
        shape['style'],shape['color_measurements']=_measure_style(pixels,shape['bbox'],ocr_mask,cv2,np)
        shape['uncertainty']='Detected shape and palette approximation; semantic container role is not established.'
    backgrounds=[]
    for shape in shapes:
        x,y,w,h=shape['bbox']
        parents=[p for p in shapes if p is not shape and p['bbox']!=shape['bbox'] and _contains(p['bbox'],shape['bbox'])]
        if w*h>=width*height*.08:
            backgrounds.append({'shape_id':shape['id'],'bbox':shape['bbox'],'role_candidate':'background_or_panel','status':'unverified'})
        if parents and w/h>=3:
            parent=min(parents,key=lambda p:p['bbox'][2]*p['bbox'][3])
            if y-parent['bbox'][1]<=max(8,parent['bbox'][3]*.08) and h<parent['bbox'][3]*.4:
                backgrounds.append({'shape_id':shape['id'],'bbox':shape['bbox'],'parent_candidate':parent['id'],
                    'role_candidate':'title_band','status':'unverified','basis':'Thin top-contained shape; heading semantics not inferred.'})
    arrow_candidates=[]
    for triangle in triangles:
        if any(_intersection(triangle['bbox'],a['bbox'])/max(1,triangle['bbox'][2]*triangle['bbox'][3])>.8 for a in arrow_candidates): continue
        arrow_candidates.append(dict(triangle,id='arrow_candidate_%04d'%(len(arrow_candidates)+1),
            direction='unknown',status='unverified',meaning='Triangle-like contour may be an arrowhead or other symbol; no semantic mapping.'))
    diagnostics.extend([{'code':'geometry_is_not_semantics','message':'No source/target relationships inferred from shape or distance.'},
                        {'code':'original_borders_preserved','message':'OCR exclusion applied to colour samples and glyph-sized candidates, not source edge pixels.'},
                        {'code':'palette_quantization','message':'Actual RGB values and approximation errors are retained; exact colours may be unavailable in Scene.'}])
    return {'version':'0.2','image':image,'coordinate_system':'top-left','bbox_format':'xywh_original_pixels',
            'base_geometry':base,'shapes':shapes,'lines':deepcopy(base['lines']),
            'arrow_candidates':arrow_candidates,'background_regions':backgrounds,
            'diagnostics':diagnostics,'counts':{'shapes':len(shapes),'lines':len(base['lines']),
                'arrow_candidates':len(arrow_candidates),'glyph_sized_contours_excluded':filtered_glyphs},'api_calls':0}
