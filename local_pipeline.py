"""Local OCR + measured geometry -> program-built IR -> existing Scene."""
from pathlib import Path
from auto_pipeline import save_json
from structured_ir.validator import inspect_ir
from structured_ir.converter import convert_ir
from structured_ir.coverage import evaluate_ir_coverage
from structured_tikz.validator import validate_scene
from local_ocr import recognize_local


def prepare_local(image_path, output, *, ocr_python=None, ocr_timeout=300,
                  xelatex='xelatex', compile_timeout=60, text_layout=True):
    from local_geometry import analyze_local_geometry
    from local_fusion import build_local_ir
    from visual_style_extractor import extract_visual_styles
    from local_visual_layers import apply_visual_layers
    from local_text_layout import optimize_local_text
    from local_small_shapes import recover_small_shapes
    from local_arrow_detection import analyze_local_arrows
    from local_connection_mapping import map_local_connections
    output = Path(output)
    report = {'status':'failed', 'stage':'ocr', 'output_dir':str(output),
              'recognition':'local_ocr_geometry_candidates', 'recognition_status':'partial',
              'api_call_count':0, 'model':None}
    try:
        ocr = recognize_local(image_path, output, python=ocr_python, timeout=ocr_timeout)
        report['model'] = ocr['engine']+' '+ocr['engine_version']
        report['ocr_count'] = len(ocr['rows'])
        report['stage'] = 'geometry'
        geometry = analyze_local_geometry(image_path, ocr['rows'])
        save_json(output/'geometry_raw.json', geometry)
        report['geometry_count'] = len(geometry['shapes'])
        report['stage'] = 'fusion'
        ir, fusion, styles = build_local_ir(ocr, geometry)
        save_json(output/'fusion_result.json', fusion)
        report['stage'] = 'visual_style'
        visual = extract_visual_styles(image_path, ocr['rows'], geometry)
        save_json(output/'style_extraction_report.json', visual)
        report['stage'] = 'visual_layers'
        ir, styles, layers = apply_visual_layers(ir, fusion, styles, visual)
        save_json(output/'layer_analysis.json', layers)
        report['visual_style_status'] = 'measured_partial'
        report['visual_style_counts'] = visual.get('counts', {})
        report['stage'] = 'text_layout'
        ir, styles, typography = optimize_local_text(image_path,ocr,ir,fusion,styles,visual,output,
            xelatex=xelatex,timeout=compile_timeout,enabled=text_layout)
        report['typography'] = typography
        if typography['status'] != 'skipped': report['typography_version'] = '0.4'
        # Add measured small composites and topology after the existing text
        # optimizer, so new OCR rows cannot change any old text's font choice.
        report['stage'] = 'small_shapes'
        ir, small = recover_small_shapes(image_path, ir, output,
            ocr_python=ocr_python, ocr_timeout=ocr_timeout,
            xelatex=xelatex, compile_timeout=compile_timeout, measure_text=text_layout)
        save_json(output/'small_shape_detection.json', small)
        report['small_shapes'] = small.get('counts', {})
        report['stage'] = 'arrow_detection'
        arrows = analyze_local_arrows(image_path, ocr['rows'], geometry)
        save_json(output/'arrow_detection.json', arrows)
        report['stage'] = 'connection_mapping'
        ir, connections, styles = map_local_connections(ir, arrows, styles)
        save_json(output/'connection_mapping.json', connections)
        report['connections'] = connections.get('counts', {})
        report['local_features_version'] = '0.5'
        report['detected_arrow_count'] = len(arrows.get('arrows', []))
        save_json(output/'style_map.json', styles)
        save_json(output/'ir_generated.json', ir)
        report['stage'] = 'ir_validation'
        validation = inspect_ir(ir)
        save_json(output/'ir_validation.json', validation)
        if not validation['valid']:
            report['status'] = 'ir_invalid'
            return report
        report['stage'] = 'ir_to_scene'
        scene, conversion = convert_ir(ir, style_map=styles)
        save_json(output/'ir_to_scene_report.json', conversion)
        save_json(output/'coverage_report.json', evaluate_ir_coverage(ir, conversion_report=conversion))
        if scene is None:
            report.update(status='conversion_failed', error='Local IR could not produce a valid Scene.')
            return report
        # OCR detections are observed lines, not paragraphs. The existing Scene
        # flag avoids reflowing a single measured line inside a tight OCR bbox.
        # Do not change coordinates, strings, font sizes or the shared renderer.
        no_wrap = []
        for node in scene['nodes']:
            if node['type'] == 'text' and '\n' not in node.get('text', ''):
                node['wrap'] = False
                no_wrap.append(node['id'])
        validate_scene(scene)
        save_json(output/'render_policy.json', {'single_ocr_lines_no_wrap':no_wrap,
                  'text_coordinates_and_content_unchanged':True,
                  'reason':'Preserve measured line boundaries rather than infer paragraph wrapping.'})
        save_json(output/'scene_generated.json', scene)
        report.update(status='local_scene_ready', stage='scene_ready',
                      warning='Local recognition remains partial: geometric candidates and unconfirmed connections are not semantic reconstruction.')
        return report
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        return report
    finally:
        save_json(output/'local_pipeline_report.json', report)
