"""One bounded XeLaTeX batch measures real typeset text, never OCR semantics.

Width/height are TikZ text-node extents (not raster ink). Source-pixel units are
obtained from TeX pt using the same Scene scale as the renderer. Candidate sizes
can be scaled locally, then checked against the final compiler's measurements.
"""
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

from auto_compile import compile_bounded
from latex_template import wrap_document
from structured_tikz import primitives
from structured_tikz.validator import validate_scene, resolve_style

FAMILIES = ('serif','sans','mono')
FONT_FILES = {
    'serif': ('FandolSong-Regular.otf','lmroman10-regular.otf'),
    'sans': ('FandolHei-Regular.otf','lmsans10-regular.otf'),
    'mono': ('FandolFang-Regular.otf','lmmono10-regular.otf'),
}
REFERENCE_SIZE = 32.
MAX_NODES = 500
MAX_MEASUREMENTS = 2000
_LOG_NUMBER = r'([+-]?(?:\d[\r\n]*)+(?:\.[\r\n]*(?:\d[\r\n]*)+)?)'
_MEASUREMENT = re.compile(r'ITTEXT\|([A-Za-z][A-Za-z0-9_\r\n]{0,95})\|\s*'
                          +_LOG_NUMBER+r'p\s*t\|\s*'+_LOG_NUMBER+r'p\s*t')


def parse_text_measurements(log, scale=.4):
    """Read complete unique log markers; malformed/duplicate markers fail closed."""
    if not isinstance(log,str):raise ValueError('Measurement log must be text.')
    if isinstance(scale,bool) or not isinstance(scale,(int,float)) or not math.isfinite(scale) or not .01<=scale<=2:
        raise ValueError('Measurement scale must match a valid Scene scale.')
    # TeX may wrap even within a long identifier or decimal at max_print_line.
    # Only log newlines are removed; arbitrary spaces in identifiers are illegal.
    matches=list(_MEASUREMENT.finditer(log)); result={}
    if log.count('ITTEXT|')!=len(matches):
        raise ValueError('Incomplete or malformed text measurement marker.')
    for match in matches:
        ident,width,height=(s.replace('\n','').replace('\r','') for s in match.groups())
        if re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}',ident) is None:
            raise ValueError('Invalid measurement identifier.')
        width,height=float(width),float(height)
        if ident in result:raise ValueError('Duplicate text measurement ID: '+ident)
        if not all(math.isfinite(v) and v>=0 for v in (width,height)):
            raise ValueError('Invalid measured text dimensions: '+ident)
        result[ident]={'width_pt':width,'height_pt':height,
                       'width_px':width/scale,'height_px':height/scale,
                       'basis':'actual_XeLaTeX_TikZ_text_node_extents'}
    return result


def _engine_path(xelatex):
    value=str(xelatex)
    candidate=Path(value).expanduser()
    if candidate.is_file():return str(candidate.resolve())
    return shutil.which(value)


def inspect_font_mapping(xelatex='xelatex'):
    """Inspect installed TeX files, never load project API configuration."""
    engine=_engine_path(xelatex)
    report={'engine':engine,'families':{},'downloads':0,
            'mapping':'Existing ctex fontset=fandol and Latin Modern; explicit rm/sf/tt family switches.',
            'fallback':'Unavailable candidate families are skipped; never invent their metrics.'}
    if not engine:
        report['status']='compiler_unavailable';return report
    kpse=Path(engine).with_name('kpsewhich.exe' if os.name=='nt' else 'kpsewhich')
    executable=str(kpse) if kpse.is_file() else shutil.which('kpsewhich')
    env={k:v for k,v in os.environ.items() if not k.startswith('API_') and k!='MODEL_NAME'}
    kwargs={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
    for family,files in FONT_FILES.items():
        records=[]
        for filename in files:
            found=None;error=None
            if executable:
                try:
                    completed=subprocess.run([executable,filename],capture_output=True,text=True,
                        encoding='utf-8',errors='replace',env=env,timeout=10,check=False,**kwargs)
                    lines=completed.stdout.strip().splitlines()
                    found=next((line for line in lines if Path(line).is_file()),None)
                    if completed.returncode and not found:error='Font file not found by kpsewhich.'
                except (OSError,subprocess.TimeoutExpired) as exc:error=type(exc).__name__
            else:error='kpsewhich unavailable; font availability cannot be verified.'
            records.append({'file':filename,'path':found,'error':error})
        report['families'][family]={'available':all(r['path'] for r in records),'files':records}
    report['status']='checked'
    return report


def _label_item(node, style, ident):
    item={k:deepcopy(node[k]) for k in ('text','segments','mode','align') if k in node}
    item.update(id=ident,x=32,y=32,anchor='center',style=deepcopy(style))
    # Painting/colour does not affect glyph extents; all metrics use black ink,
    # with no dependency on custom colour definitions or transparency.
    item['style'].update(fill='none',stroke='none',text_color='black')
    for k in ('fill_color','stroke_color','text_color_rgb','opacity'):
        item['style'].pop(k,None)
    return item


def build_measurement_batch(scene, *, families_by_id=None, candidate_scene=None, available_families=FAMILIES):
    """Pure batch generation; baseline and candidates use the shared renderer."""
    validate_scene(scene)
    candidate_scene=scene if candidate_scene is None else candidate_scene
    validate_scene(candidate_scene)
    if scene['canvas'].get('scale',.4)!=candidate_scene['canvas'].get('scale',.4):
        raise ValueError('Candidate and baseline Scene scales must agree.')
    if families_by_id is not None and not isinstance(families_by_id,dict):
        raise ValueError('families_by_id must map node IDs to family lists.')
    allowed=set(available_families)
    if allowed-set(FAMILIES):raise ValueError('Unsupported available font family.')
    nodes=[n for n in scene['nodes'] if n.get('text') or n.get('mode')=='rich']
    if len(nodes)>MAX_NODES:raise ValueError('Text measurement node budget exceeded.')
    by_id={n['id']:n for n in candidate_scene['nodes']}
    scale=scene['canvas'].get('scale',.4)
    measure_scene={'canvas':{'width':64,'height':64,'coordinate_system':'top-left','scale':scale},
                   'styles':{},'nodes':[],'edges':[],'labels':[]}
    lines=[r'\definecolor{sceneblack}{HTML}{111111}%',r'\begin{tikzpicture}']
    requests=[]
    for node in nodes:
        if node['id'] not in by_id:raise ValueError('Candidate Scene omits source text ID.')
        candidate=by_id[node['id']]
        if candidate.get('text')!=node.get('text'):
            raise ValueError('Candidate text provenance must remain unchanged.')
        families=FAMILIES if families_by_id is None else families_by_id.get(node['id'],FAMILIES)
        if not isinstance(families,(list,tuple)) or len(families)>3 or len(set(families))!=len(families) or set(families)-set(FAMILIES):
            raise ValueError('Each text needs at most three unique supported font candidates.')
        baseline=resolve_style(scene,node)
        variants=[('baseline',baseline,node)]
        for family in families:
            if family not in allowed:continue
            style=resolve_style(candidate_scene,candidate)
            style.update(font_size=REFERENCE_SIZE,font_family=family,font_scale=1.)
            variants.append((family,style,candidate))
        for variant,style,source in variants:
            ident='metric_%04d'%(len(requests)+1)
            item=_label_item(source,style,ident)
            width=source['bbox'][2] if variant=='baseline' and source['type']=='text' and source.get('wrap',True) else None
            # Measurement-only nodes overlap at one origin, so layout is bounded
            # by the longest measured label rather than the number of candidates.
            lines.append(primitives.label(item,measure_scene,name_prefix='m-',text_width=width))
            lines.extend(primitives.text_measurement_commands('m-'+ident,ident))
            requests.append({'measurement_id':ident,'element_id':node['id'],'variant':variant,
                'font_family':style.get('font_family','serif'),'font_size':style['font_size'],
                'font_scale':style.get('font_scale',1.),
                'effective_font_size':style['font_size']*style.get('font_scale',1.),
                'line_spacing':style.get('line_spacing',1.25),'preserve_spaces':style.get('preserve_spaces',False),
                'wrap_width_px':width,'render_mode':source.get('mode','text')})
    if len(requests)>MAX_MEASUREMENTS:raise ValueError('Text measurement candidate budget exceeded.')
    if not requests:lines.append(r'\path (0,0) rectangle (1,1);')
    lines.append(r'\end{tikzpicture}')
    fragment='\n'.join(lines)+'\n'
    return wrap_document(fragment),requests


def measure_text_candidates(scene, output_dir, *, xelatex='xelatex', timeout=60,
                            families_by_id=None, candidate_scene=None):
    """One compile, max 500 nodes / 2000 metrics; unavailable evidence is explicit."""
    validate_scene(scene)
    if isinstance(timeout,bool) or not isinstance(timeout,(float,int)) or not math.isfinite(timeout) or not 1<=timeout<=120:
        raise ValueError('Text measurement timeout must be in [1,120] seconds.')
    mapping=inspect_font_mapping(xelatex)
    report={'status':'unavailable','items':{},'font_mapping':mapping,'compile_attempts':0,
            'budget':{'max_nodes':MAX_NODES,'max_candidates_per_node':3,'max_compile_attempts':1},
            'diagnostics':[],'api_calls':0}
    if not mapping['engine']:
        report['diagnostics'].append({'code':'compiler_unavailable','message':'No measured dimensions; existing typography may be retained.'})
        return report
    available=[name for name,details in mapping['families'].items() if details['available']]
    if not available:
        report['diagnostics'].append({'code':'fonts_unavailable','message':'No verified font family available; no fake dimensions generated.'})
        return report
    document,requests=build_measurement_batch(scene,families_by_id=families_by_id,
        candidate_scene=candidate_scene,available_families=available)
    if not requests:
        report.update(status='empty',measurement_count=0);return report
    folder=Path(output_dir)/'text_metrics'
    folder.mkdir(parents=True,exist_ok=False)
    tex=folder/'measurement.tex';tex.write_text(document,encoding='utf-8')
    (folder/'requests.json').write_text(json.dumps(requests,ensure_ascii=False,indent=2),encoding='utf-8')
    report['paths']={'tex':str(tex),'requests':str(folder/'requests.json')}
    report['compile_attempts']=1
    result=compile_bounded(tex,xelatex=mapping['engine'],timeout=timeout)
    report['compile']={k:str(v) if isinstance(v,Path) else v for k,v in asdict(result).items()}
    report['paths']['log']=str(result.log_path)
    log=result.log_path.read_text(encoding='utf-8',errors='replace') if result.log_path.is_file() else ''
    warnings=[line.strip() for line in log.splitlines() if 'Missing character:' in line or ('Font' in line and 'Warning:' in line)]
    report['font_warnings']=warnings
    if not result.success:
        report['status']='failed'
        report['diagnostics'].append({'code':'measurement_compile_failed','message':result.errors})
        return report
    try:
        measured=parse_text_measurements(log,scene['canvas'].get('scale',.4))
    except ValueError as exc:
        report['status']='failed'
        report['diagnostics'].append({'code':'invalid_measurement_log','message':str(exc)})
        return report
    if set(measured)!={r['measurement_id'] for r in requests}:
        report.update(status='failed')
        report['diagnostics'].append({'code':'incomplete_measurements','message':'Compiler did not report every requested metric.'})
        return report
    for request in requests:
        item=report['items'].setdefault(request['element_id'],{'candidates':{}})
        metric=dict(measured[request['measurement_id']],**{k:v for k,v in request.items() if k not in ('element_id','variant')})
        if request['variant']=='baseline':item['baseline']=metric
        else:item['candidates'][request['variant']]=metric
    report.update(status='measured',measurement_count=len(measured),
        limitations=['Metrics are typeset node extents, not raster ink bounds.',
                     'Candidate-size scaling is a local approximation; final compilation must verify selected sizes.',
                     'Missing glyph warnings remain evidence of unsupported literal content; no OCR correction is performed.'])
    return report
