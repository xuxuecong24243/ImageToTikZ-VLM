"""Measured local typography with a fixed budget and immutable source geometry."""
from copy import deepcopy
import json
import math
from pathlib import Path

from auto_pipeline import save_json
from structured_ir.converter import convert_ir
from structured_ir.validator import validate_ir
from text_layout_analysis import analyze_source_text, analyze_text_collisions
from text_measurement import measure_text_candidates, parse_text_measurements


def _centered(element, width, height):
    x,y,w,h=element['bbox']
    return [x+(w-width)/2,y+(h-height)/2,width,height]


def _scene(ir,styles):
    scene,conversion=convert_ir(ir,style_map=styles)
    if scene is None or not conversion['complete']:
        raise ValueError('Text layout requires a completely representable validated source IR.')
    for node in scene['nodes']:
        if node['type']=='text' and '\n' not in node.get('text',''):node['wrap']=False
    return scene


def _font_fields(size,family,preserve_spaces):
    return {'font_size':round(min(80.,size),5),'font_scale':round(max(1.,size/80),7),
            'font_family':family,'line_spacing':1.25,'preserve_spaces':preserve_spaces}


def select_typography(ir,analysis,measurements):
    """Pure bounded selection: <=3 measured families, then <=2 shrink passes.

The source ink bounds are targets, while source OCR/owner/canvas bounds are hard
limits. No location, line break, non-text style or graph edits are permitted.
"""
    result=deepcopy(ir);elements={e['id']:e for e in result['elements']}
    original={e['id']:e for e in ir['elements']}
    regions={r['id']:r for r in ir['regions']}
    before={};after={};selections={};diagnostics=[]
    for ident,metrics in measurements['items'].items():
        if ident not in analysis['items']:continue
        e=elements[ident];info=analysis['items'][ident];base=metrics['baseline']
        before[ident]=_centered(e,base['width_px'],base['height_px'])
        after[ident]=before[ident][:]
        ink=info.get('ink_bbox')
        if ink is None or not metrics.get('candidates'):
            diagnostics.append({'code':'unmeasured_source_or_font','element_id':ident,'action':'original_typography_retained'})
            continue
        x,y,w,h=e['bbox'];cx,cy=x+w/2,y+h/2
        max_width,min_height=w,h
        owner=regions.get(e.get('region_id'))
        if owner:
            ox,oy,ow,oh=owner['bbox'];margin=min(3,h*.04)
            max_width=min(max_width,2*min(cx-ox-margin,ox+ow-margin-cx))
            min_height=min(min_height,2*min(cy-oy-margin,oy+oh-margin-cy))
        max_width=min(max_width,2*min(cx,ir['canvas']['width']-cx))
        min_height=min(min_height,2*min(cy,ir['canvas']['height']-cy))
        if max_width<=0 or min_height<=0:
            diagnostics.append({'code':'incompatible_fixed_anchor','element_id':ident,'action':'original_typography_retained'})
            continue
        choices=[]
        for preference,family in enumerate(info['font_candidates']):
            m=metrics['candidates'].get(family)
            if not m or m['width_px']<=0 or m['height_px']<=0:continue
            # Use both real dimensions, not a fixed multiplier of OCR height.
            target_w,target_h=ink[2],ink[3]
            ratio=min(target_w/m['width_px'],target_h/m['height_px'],
                      max_width*.98/m['width_px'],min_height*.98/m['height_px'])
            desired=max(6,min(240,m['effective_font_size']*ratio))
            factor=desired/m['effective_font_size'];mw,mh=m['width_px']*factor,m['height_px']*factor
            score=abs(math.log(max(.001,mw/target_w)))+1.4*abs(math.log(max(.001,mh/target_h)))+preference*.025
            choices.append((score,family,desired,mw,mh,m))
        if not choices:continue
        _,family,size,mw,mh,metric=min(choices,key=lambda v:v[0])
        preserve=info['text_kind'] in {'python_code','mixed_code_chinese'}
        e.setdefault('visual_style',{}).update(_font_fields(size,family,preserve))
        if info.get('render_segments'):e['render_segments']=deepcopy(info['render_segments'])
        after[ident]=_centered(e,mw,mh)
        selections[ident]={'font_family':family,'effective_font_size':size,'reference_metric':metric,
            'source_ink_bbox':ink,'predicted_width_px':mw,'predicted_height_px':mh,
            'preserve_spaces':preserve,'constraint_width_px':max_width,'constraint_height_px':min_height,
            'text_kind':info['text_kind'],'status':'selected_by_measured_dimensions'}
    baseline=analyze_text_collisions(ir,before)
    # Local, limited shrink can reduce overlapping measured boxes without moving
    # a card, anchor, arrow or original OCR bbox. Unresolved conflicts are kept.
    iterations=0
    for iteration in range(2):
        collision=analyze_text_collisions(result,after)
        offenders={i for issue in collision['issues'] if issue['code'] in
            {'text_overlap','container_overflow','page_overflow'} for i in issue['element_ids'] if i in selections}
        if not offenders:break
        iterations+=1
        for ident in offenders:
            selection=selections[ident];old=selection['effective_font_size'];size=max(6,old*.92)
            factor=size/old;selection['effective_font_size']=size
            selection['predicted_width_px']*=factor;selection['predicted_height_px']*=factor
            selection['status']='bounded_conflict_shrink'
            elements[ident]['visual_style'].update(_font_fields(size,selection['font_family'],selection['preserve_spaces']))
            after[ident]=_centered(elements[ident],selection['predicted_width_px'],selection['predicted_height_px'])
    collision=analyze_text_collisions(result,after)
    # Never accept a global increase in hard conflicts just to enlarge glyphs.
    if collision['counts']['hard_conflicts']>baseline['counts']['hard_conflicts']:
        diagnostics.append({'code':'candidate_increased_hard_conflicts','action':'retain_original_typography_for_all_texts'})
        result=deepcopy(ir);after=deepcopy(before);selections={}
        collision=deepcopy(baseline)
    result['measure_text']=True
    validate_ir(result)
    report={'status':'selected_partial','selections':selections,'diagnostics':diagnostics,
        'budget':{'max_font_candidates_per_text':3,'max_measurement_compiles':1,'max_shrink_passes':2,
                  'actual_shrink_passes':iterations,'final_document_compiles':1},
        'before':baseline,'predicted_after':collision,
        'raw_text_ids_boxes_graph_and_nontext_styles_unchanged':True,'api_calls':0}
    return result,report


def optimize_local_text(image_path,ocr,ir,fusion,styles,visual,output,*,xelatex='xelatex',timeout=60,enabled=True):
    output=Path(output)
    analysis=analyze_source_text(image_path,ocr['rows'],ir,fusion,visual)
    save_json(output/'text_source_analysis.json',analysis)
    if not enabled or not analysis['counts']['measured_ink']:
        report={'status':'skipped','reason':'--no-compile' if not enabled else 'no_reliable_source_ink',
                'items':{},'compile_attempts':0,'api_calls':0}
        save_json(output/'text_measurement_report.json',report)
        save_json(output/'font_mapping_report.json',{'status':'not_requested','reason':report['reason']})
        save_json(output/'text_collision_report.json',{'status':'unmeasured','reason':report['reason'],'api_calls':0})
        return ir,styles,report
    candidate=deepcopy(ir)
    for element in candidate['elements']:
        info=analysis['items'].get(element['id'])
        if not info:continue
        if info.get('render_segments'):element['render_segments']=deepcopy(info['render_segments'])
        if info['text_kind'] in {'python_code','mixed_code_chinese'}:
            element.setdefault('visual_style',{})['preserve_spaces']=True
    try:
        metrics=measure_text_candidates(_scene(ir,styles),output,xelatex=xelatex,timeout=timeout,
            families_by_id={i:v['font_candidates'] for i,v in analysis['items'].items()},candidate_scene=_scene(candidate,styles))
    except ValueError as exc:
        # Typography is optional; a safe partial Scene must still reach the
        # existing conversion contract with all unsupported-content diagnostics.
        metrics={'status':'failed','reason':'measurement_unavailable','items':{},'api_calls':0,
            'font_mapping':{'status':'unmeasured'},
            'diagnostics':[{'code':'measurement_unavailable','message':str(exc),
                            'action':'original_typography_retained'}]}
    save_json(output/'text_measurement_report.json',metrics)
    save_json(output/'font_mapping_report.json',metrics['font_mapping'])
    if metrics['status']!='measured':
        save_json(output/'text_collision_report.json',{'status':'unmeasured','reason':metrics['status'],'api_calls':0})
        return ir,styles,metrics
    result,selection=select_typography(ir,analysis,metrics)
    save_json(output/'text_layout_selection.json',selection)
    save_json(output/'text_collision_report.json',{'status':'predicted_pending_final_compile',
        'before':selection['before'],'predicted_after':selection['predicted_after'],'api_calls':0})
    return result,styles,{'status':'measured','selected_texts':len(selection['selections']),
        'compile_attempts':metrics['compile_attempts'],'api_calls':0,'budget':selection['budget']}


def verify_final_text_layout(ir,scene,output):
    """Use actual final node extents; no hidden correction/compile loop follows."""
    output=Path(output)
    if not scene['canvas'].get('measure_text'):return {'status':'not_requested'}
    prior=json.loads((output/'text_collision_report.json').read_text(encoding='utf-8'))
    try:
        log=(output/'generated.log').read_text(encoding='utf-8',errors='replace')
        measured=parse_text_measurements(log,scene['canvas'].get('scale',.4))
        texts={e['id']:e for e in ir['elements'] if e.get('raw_text')}
        if set(measured)!=set(texts):raise ValueError('Final text measurement markers do not cover every source text.')
    except (OSError,ValueError) as exc:
        # A failed diagnostic must not discard a valid PDF or prevent its PNG.
        result={**prior,'status':'unmeasured_final_compile','api_calls':0,
                'diagnostics':[{'code':'final_measurement_unavailable','message':str(exc)}]}
        save_json(output/'text_collision_report.json',result)
        return {'status':result['status'],'reason':str(exc)}
    boxes={i:_centered(texts[i],m['width_px'],m['height_px']) for i,m in measured.items()}
    actual=analyze_text_collisions(ir,boxes)
    result={**prior,'status':'verified_final_compile','actual_after':actual,'actual_metrics':measured,
        'missing_glyph_warnings':[l.strip() for l in log.splitlines() if 'Missing character:' in l],
        'no_extra_optimization_or_compile':True,'api_calls':0}
    save_json(output/'text_collision_report.json',result)
    return {'status':result['status'],'counts_before':result['before']['counts'],
            'counts_after':actual['counts'],'missing_glyph_warning_count':len(result['missing_glyph_warnings'])}
