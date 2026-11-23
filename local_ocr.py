"""Isolated, CPU-only OCR adapter. No VLM, API configuration or runtime downloads.

The optional worker environment owns RapidOCR/ONNX dependencies. Original text,
quadrilaterals and engine confidence are retained; bbox is explicitly xywh.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def normalize_ocr_result(raw, image_size):
    """Validate engine results and enclose polygons in original-pixel bboxes."""
    width, height = image_size
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ValueError('OCR result must be a list or None.')
    rows, diagnostics = [], []
    for index, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError('OCR row {} requires polygon, text and confidence.'.format(index))
        polygon, text, confidence = item
        if (not isinstance(polygon, (list, tuple)) or len(polygon) != 4
                or any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in polygon)):
            raise ValueError('OCR row {} needs a four-corner polygon.'.format(index))
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for p in polygon for v in p):
            raise ValueError('OCR coordinates must be finite numbers.')
        if (not isinstance(text, str) or isinstance(confidence, bool)
                or not isinstance(confidence, (float, int)) or not math.isfinite(confidence)
                or not 0 <= confidence <= 1):
            raise ValueError('Invalid OCR text or engine confidence.')
        if any(not 0 <= x <= width or not 0 <= y <= height for x, y in polygon):
            raise ValueError('OCR polygon lies outside the original image.')
        x, y = math.floor(min(p[0] for p in polygon)), math.floor(min(p[1] for p in polygon))
        right, bottom = math.ceil(max(p[0] for p in polygon)), math.ceil(max(p[1] for p in polygon))
        if right <= x or bottom <= y:
            raise ValueError('OCR polygon has no positive extent.')
        if not text:
            diagnostics.append({'code': 'empty_recognized_string', 'raw_index': index,
                                'message': 'No recognized text; raw detection retained in engine_result.'})
            continue
        rows.append({'id': 'ocr_%04d' % (index+1), 'text': text, 'bbox': [x,y,right-x,bottom-y],
                     'polygon': polygon, 'confidence': confidence, 'engine_order': index,
                     'low_confidence': confidence < .65})
    # Candidate reading order, not a semantic document order claim.
    rows.sort(key=lambda r: (r['bbox'][1], r['bbox'][0], r['engine_order']))
    for order, row in enumerate(rows):
        row['reading_order'] = order
    return rows, diagnostics


def recognize_local(image_path, output_dir, *, python=None, timeout=300):
    """Invoke the separate installed runtime once; retain worker logs on failure."""
    image_path, output = Path(image_path).resolve(), Path(output_dir).resolve()
    if not math.isfinite(timeout) or not 1 <= timeout <= 900:
        raise ValueError('OCR timeout must be finite and between 1 and 900 seconds.')
    executable = Path(python) if python else ROOT/'.venv-ocr'/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    executable = executable.expanduser().resolve()
    if not executable.is_file():
        raise ValueError('Local OCR runtime is missing. Create .venv-ocr and install requirements-ocr.txt; see README.')
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output/'ocr_raw.json'
    if raw_path.exists():
        raise ValueError('OCR output exists; use a new output directory.')
    env = {k:v for k,v in os.environ.items() if not k.startswith('API_') and k not in {'MODEL_NAME','PYTHONPATH'}}
    env['PYTHONUTF8'] = '1'
    command = [str(executable), '-B', str(Path(__file__).resolve()), '--worker',
               '--input', str(image_path), '--output', str(raw_path)]
    kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    with (output/'ocr_worker.log').open('x', encoding='utf-8') as log:
        try:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                    env=env, timeout=timeout, check=False, **kwargs)
        except subprocess.TimeoutExpired as exc:
            raise ValueError('Local OCR timed out; inspect ocr_worker.log. No API fallback.') from exc
    if result.returncode or not raw_path.is_file():
        raise ValueError('Local OCR worker failed; inspect ocr_worker.log. No model output was fabricated.')
    result = json.loads(raw_path.read_text(encoding='utf-8'))
    if result.get('image', {}).get('sha256') != hashlib.sha256(image_path.read_bytes()).hexdigest():
        raise ValueError('OCR output does not match the source image.')
    normalized, _ = normalize_ocr_result(result['engine_result'],
                                         [result['image']['width'], result['image']['height']])
    if result['rows'] != normalized:
        raise ValueError('OCR normalization integrity check failed.')
    return result


def recognize_crops(image_path, crops, output_dir, *, python=None, timeout=300):
    """One offline recognition-only batch for measured, auditable image crops.

    Detection and source-coordinate restoration belong to the crop producer;
    the existing RapidOCR recognizer supplies only literal text and its score.
    """
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    if not 1 <= len(crops) <= 64:raise ValueError('Local crop OCR budget is 1..64 crops.')
    if not math.isfinite(timeout) or not 1 <= timeout <= 900:raise ValueError('Invalid OCR timeout.')
    executable=Path(python) if python else ROOT/'.venv-ocr'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    executable=executable.expanduser().resolve()
    if not executable.is_file():raise ValueError('Local OCR runtime is missing; no API fallback.')
    ids=set();manifest=[]
    for item in crops:
        if not isinstance(item.get('id'),str) or item['id'] in ids:raise ValueError('Crop IDs must be unique.')
        ids.add(item['id']);path=Path(item['path']).resolve()
        if not path.is_file():raise ValueError('Measured OCR crop does not exist.')
        manifest.append(dict(item,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    request=output/'requests.json';request.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    raw=output/'crop_ocr_raw.json'
    command=[str(executable),'-B',str(Path(__file__).resolve()),'--worker','--input',str(Path(image_path).resolve()),
             '--output',str(raw),'--crops-manifest',str(request)]
    env={k:v for k,v in os.environ.items() if not k.startswith('API_') and k not in {'MODEL_NAME','PYTHONPATH'}}
    env['PYTHONUTF8']='1'
    kwargs={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
    with (output/'ocr_worker.log').open('x',encoding='utf-8') as log:
        try:completed=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=env,timeout=timeout,check=False,**kwargs)
        except subprocess.TimeoutExpired as exc:raise ValueError('Local crop OCR timed out; no API fallback.') from exc
    if completed.returncode or not raw.is_file():raise ValueError('Local crop OCR failed; inspect crop OCR log.')
    result=json.loads(raw.read_text(encoding='utf-8'))
    if result['source_image_sha256']!=hashlib.sha256(Path(image_path).read_bytes()).hexdigest():raise ValueError('Crop OCR source mismatch.')
    if result['requests']!=manifest:raise ValueError('Crop OCR request integrity mismatch.')
    if len(result['engine_result'])!=len(crops):raise ValueError('Crop OCR result count mismatch.')
    for row in result['engine_result']:
        if not isinstance(row,list) or len(row)!=2 or not isinstance(row[0],str) or type(row[1]) not in (int,float) or not math.isfinite(row[1]) or not 0<=row[1]<=1:
            raise ValueError('Invalid crop OCR text or confidence.')
    return result


def _worker(image_path, output_path, crops_manifest=None):
    # All inference is offline, including third-party Python networking paths.
    import socket
    def forbidden(*args, **kwargs):
        raise RuntimeError('Network access is forbidden in local OCR inference.')
    socket.socket.connect = socket.socket.connect_ex = socket.socket.sendto = forbidden
    socket.create_connection = socket.getaddrinfo = forbidden
    import importlib.metadata
    import numpy as np
    import rapidocr_onnxruntime
    from rapidocr_onnxruntime import RapidOCR
    from PIL import Image
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        size = image.size
        rgba = image.convert('RGBA')
        rgb = Image.alpha_composite(Image.new('RGBA', size, 'white'), rgba).convert('RGB')
        # RapidOCR accepts BGR numpy; avoid its path decoder on Unicode paths.
        pixels = np.asarray(rgb)[:,:,::-1].copy()
    package = Path(rapidocr_onnxruntime.__file__).parent
    models = [{'file':p.name, 'bytes':p.stat().st_size,
               'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(package.rglob('*.onnx'))]
    if not models:
        raise RuntimeError('Bundled ONNX models are missing; offline mode will not download replacements.')
    started = time.monotonic()
    engine = RapidOCR(text_score=0.0, intra_op_num_threads=2, inter_op_num_threads=2)
    providers = {'text_det':engine.text_det.infer.session.get_providers(),
                 'text_cls':engine.text_cls.infer.session.get_providers(),
                 'text_rec':engine.text_rec.session.session.get_providers()}
    if any(value != ['CPUExecutionProvider'] for value in providers.values()):
        raise RuntimeError('Local OCR requires CPU-only execution providers.')
    if crops_manifest:
        requests=json.loads(Path(crops_manifest).read_text(encoding='utf-8'))
        if not 1<=len(requests)<=64:raise ValueError('Crop OCR request budget exceeded.')
        crops=[]
        for request in requests:
            path=Path(request['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=request['sha256']:raise ValueError('Crop pixels changed.')
            with Image.open(path) as crop:crops.append(np.asarray(crop.convert('RGB'))[:,:,::-1].copy())
        result,timing=engine.text_rec(crops)
        result=json.loads(json.dumps(result,default=lambda x:x.tolist() if hasattr(x,'tolist') else float(x)))
        report={'engine':'rapidocr-onnxruntime','engine_version':importlib.metadata.version('rapidocr-onnxruntime'),
            'mode':'existing_recognizer_on_measured_crops','source_image_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),
            'requests':requests,'engine_result':result,'engine_timing':timing,'models':models,
            'execution_providers':providers,'api_calls':0,'network_disabled':True,
            'confidence_meaning':'Uncalibrated recognition score, not accuracy; no sequence completion.'}
        with Path(output_path).open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
        return
    result, timing = engine(pixels)
    # Convert array/scalar objects, without changing recognized strings.
    result = json.loads(json.dumps(result, default=lambda x: x.tolist() if hasattr(x,'tolist') else float(x)))
    rows, diagnostics = normalize_ocr_result(result, size)
    report = {'engine':'rapidocr-onnxruntime', 'engine_version':importlib.metadata.version('rapidocr-onnxruntime'),
              'runtime_version':importlib.metadata.version('onnxruntime'), 'device':'CPU',
              'image':{'width':size[0], 'height':size[1], 'sha256':hashlib.sha256(image_path.read_bytes()).hexdigest()},
              'coordinate_system':'top-left', 'bbox_format':'xywh', 'engine_result':result,
              'rows':rows, 'diagnostics':diagnostics,
              'confidence_meaning':'Uncalibrated recognition score returned by the OCR engine; not accuracy.',
              'reading_order_meaning':'Top-left sort candidate; fusion refines per container.',
              'models':models, 'elapsed_seconds':round(time.monotonic()-started,3),
              'execution_providers':providers,
              'settings':{'text_score':engine.text_score, 'max_side_len':engine.max_side_len,
                          'min_side_len':engine.min_side_len, 'threads':2,
                          'coordinate_recovery':'Engine restores all polygons to original stored pixel coordinates.'},
              'engine_timing':timing, 'api_calls':0, 'network_disabled':True}
    with Path(output_path).open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Internal isolated local OCR worker')
    parser.add_argument('--worker',action='store_true',required=True)
    parser.add_argument('--input',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--crops-manifest')
    args=parser.parse_args(); _worker(args.input,args.output,args.crops_manifest)
