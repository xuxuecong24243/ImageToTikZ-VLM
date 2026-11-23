"""Local adapter contracts only; test doubles are never OCR acceptance data.

Real CPU OCR and real compilation are exercised separately in the v0.2 report.
Legacy VLM, historical Scene/TikZ and fusion/geometry checks remain in their
existing dedicated test modules rather than being duplicated here.
"""
from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
try:
    from .offline_support import OfflineTestCase
    from .ir_fixtures import example_ir
except ImportError:
    from offline_support import OfflineTestCase
    from ir_fixtures import example_ir

import convert
from latex_compiler import CompileResult
from local_ocr import normalize_ocr_result, recognize_local
from structured_ir.converter import convert_ir
from structured_ir.validator import inspect_ir
from structured_tikz.validator import validate_scene


def detection(text='x', confidence=.9, box=(10, 20, 30, 12)):
    x, y, w, h = box
    return [[[x,y], [x+w,y], [x+w,y+h], [x,y+h]], text, confidence]


class LocalOCRAndCLITests(OfflineTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='local-ocr-contract-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root/'中文 输入.png'
        Image.new('RGB', (640,420), 'white').save(self.source)
        self.counter = 0

    def output(self):
        self.counter += 1
        return self.root/('结果 %02d' % self.counter)

    def argv(self, *extra):
        return ['--input', str(self.source), '--output', str(self.output()), *extra]

    def worker_response(self, raw=None):
        raw = [detection()] if raw is None else raw
        rows, _ = normalize_ocr_result(raw, (640,420))
        return {'engine':'unit_test_worker', 'engine_version':'fixture',
                'image':{'width':640,'height':420,
                         'sha256':hashlib.sha256(self.source.read_bytes()).hexdigest()},
                'engine_result':raw, 'rows':rows}

    def test_bbox_is_original_pixel_xywh_enclosing_float_polygon(self):
        raw = [[[[10.2,20.8],[41.1,20.5],[40.9,33.2],[10.7,32.9]], '数学 x_1', .97]]
        rows, diagnostics = normalize_ocr_result(raw, (100,100))
        self.assertEqual(rows[0]['bbox'], [10,20,32,14])
        self.assertEqual(rows[0]['polygon'], raw[0][0])
        self.assertEqual(rows[0]['confidence'], .97)
        self.assertEqual(diagnostics, [])

    def test_low_confidence_preserves_code_case_punctuation_and_spaces(self):
        literal = '  __getitem__(Index_2) + tensor(0) 中文\n'
        rows, _ = normalize_ocr_result([detection(literal, .02)], (100,100))
        self.assertEqual(rows[0]['text'], literal)
        self.assertEqual(rows[0]['confidence'], .02)
        self.assertTrue(rows[0]['low_confidence'])

    def test_reading_order_candidate_does_not_mutate_raw_engine_result(self):
        raw = [detection('second', box=(60,20,20,10)), detection('first', box=(10,20,20,10)),
               detection('third', box=(10,50,20,10))]
        original = deepcopy(raw)
        rows, _ = normalize_ocr_result(raw, (100,100))
        self.assertEqual([r['text'] for r in rows], ['first','second','third'])
        self.assertEqual([r['reading_order'] for r in rows], [0,1,2])
        self.assertEqual([r['engine_order'] for r in rows], [1,0,2])
        self.assertEqual(len({r['id'] for r in rows}), 3)
        self.assertEqual(raw, original)

    def test_empty_ocr_and_empty_string_have_explicit_distinct_results(self):
        self.assertEqual(normalize_ocr_result(None, (100,100)), ([],[]))
        self.assertEqual(normalize_ocr_result([], (100,100)), ([],[]))
        rows, diagnostics = normalize_ocr_result([detection('')], (100,100))
        self.assertEqual(rows, [])
        self.assertEqual(diagnostics[0]['code'], 'empty_recognized_string')
        self.assertEqual(diagnostics[0]['raw_index'], 0)

    def test_invalid_polygons_fail_without_silent_clipping_or_success(self):
        cases = [detection(box=(-1,20,20,10)), detection(box=(90,20,20,10)),
                 detection(box=(10,20,0,10))]
        for value in (float('nan'), float('inf'), True):
            item = detection(); item[0][0][0] = value; cases.append(item)
        item = detection(); item[0].pop(); cases.append(item)
        for index, item in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                normalize_ocr_result([item], (100,100))

    def test_missing_or_invalid_engine_confidence_is_not_fabricated(self):
        for value in (None, True, -1, 1.01, float('nan'), '0.9'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_ocr_result([detection(confidence=value)], (100,100))
        for raw in ({}, [['bad row']], [[detection()[0], 123, .9]]):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                normalize_ocr_result(raw, (100,100))

    def test_missing_runtime_never_launches_worker_or_api_fallback(self):
        with patch('local_ocr.subprocess.run', side_effect=AssertionError('No worker launch')), \
             patch('vlm_client.APIConfig.from_env', side_effect=AssertionError('No config')):
            with self.assertRaisesRegex(ValueError, 'runtime is missing'):
                recognize_local(self.source, self.output(), python=self.root/'absent.exe')

    def test_worker_uses_isolated_environment_and_preserves_measured_result(self):
        payload = self.worker_response([detection('__len__()', .45)])
        observed = {}
        def worker(command, **kwargs):
            observed.update(command=command, kwargs=kwargs)
            Path(command[command.index('--output')+1]).write_text(json.dumps(payload), encoding='utf-8')
            return SimpleNamespace(returncode=0)
        with patch.dict(os.environ, {'API_KEY':'synthetic-secret', 'API_BASE_URL':'https://example.invalid',
                                     'MODEL_NAME':'not-used', 'PYTHONPATH':'not-inherited'}), \
             patch('local_ocr.subprocess.run', side_effect=worker) as process:
            result = recognize_local(self.source, self.output(), python=sys.executable, timeout=17)
        self.assertEqual(result, payload)
        self.assertEqual(process.call_count, 1)
        self.assertEqual(observed['kwargs']['timeout'], 17)
        env = observed['kwargs']['env']
        self.assertFalse(any(k.startswith('API_') for k in env))
        self.assertNotIn('MODEL_NAME', env)
        self.assertNotIn('PYTHONPATH', env)
        self.assertEqual(observed['command'][observed['command'].index('--input')+1], str(self.source.resolve()))

    def test_worker_hash_or_normalization_tampering_is_rejected(self):
        for case in ('hash', 'text'):
            payload = self.worker_response()
            if case == 'hash': payload['image']['sha256'] = '0'*64
            else: payload['rows'][0]['text'] = 'changed'
            def worker(command, **kwargs):
                Path(command[command.index('--output')+1]).write_text(json.dumps(payload), encoding='utf-8')
                return SimpleNamespace(returncode=0)
            with self.subTest(case=case), patch('local_ocr.subprocess.run', side_effect=worker), self.assertRaises(ValueError):
                recognize_local(self.source, self.output(), python=sys.executable)

    def test_worker_timeout_is_bounded_and_never_retried(self):
        with patch('local_ocr.subprocess.run', side_effect=subprocess.TimeoutExpired('test-only', 1)) as worker:
            with self.assertRaisesRegex(ValueError, 'timed out'):
                recognize_local(self.source, self.output(), python=sys.executable, timeout=1)
        self.assertEqual(worker.call_count, 1)

    def test_backend_default_is_vlm_and_reserved_hybrid_has_no_fallback(self):
        self.assertEqual(convert.build_parser().parse_args(self.argv()).backend, 'vlm')
        with patch('convert.run_two_stage', side_effect=AssertionError('No model request')), \
             patch('vlm_client.APIConfig.from_env', side_effect=AssertionError('No config')):
            args = convert.build_parser().parse_args(self.argv('--backend','hybrid','--allow-api'))
            report = convert.run_conversion(args)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('not implemented', report['errors'][0]['message'])
        self.assertEqual(report['api_request_count'], 0)

    def test_local_backend_rejects_api_and_ir_injection(self):
        combinations = [(['--allow-api'], None), (['--offline-ir','any.json'], None), ([], object())]
        for extra, client in combinations:
            args = convert.build_parser().parse_args(self.argv('--backend','local',*extra))
            with self.subTest(extra=extra, injected=client is not None), \
                 patch('local_pipeline.prepare_local', side_effect=AssertionError('Invalid mode must not run OCR')), \
                 patch('vlm_client.APIConfig.from_env', side_effect=AssertionError('No config')):
                report = convert.run_conversion(args, client=client)
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['api_request_count'], 0)
            self.assertIn('images only', report['errors'][0]['message'])

    def local_fixture(self, image, output, **kwargs):
        # Adapter fixture only: does not represent recognized image content.
        ir = example_ir(); scene, conversion = convert_ir(ir)
        files = {'ir_generated.json':ir, 'scene_generated.json':scene,
                 'ir_validation.json':inspect_ir(ir), 'ir_to_scene_report.json':conversion}
        for name, value in files.items():
            (Path(output)/name).write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        return {'status':'local_scene_ready','stage':'scene_ready','output_dir':str(output),
                'recognition':'local_ocr_geometry_candidates','recognition_status':'partial',
                'api_call_count':0,'model':'unit-test-only','ocr_count':3}

    def test_local_complete_conversion_remains_partial_after_previews(self):
        def compile_fixture(tex, **kwargs):
            tex = Path(tex); pdf=tex.with_suffix('.pdf'); log=tex.with_suffix('.log')
            pdf.write_bytes(b'Test-only preview stub'); log.write_text('stub',encoding='utf-8')
            return CompileResult(True,0,False,pdf,log,log,'',[])
        def image_fixture(source, target, *args, **kwargs):
            Image.new('RGB',(10,10),'white').save(target); return Path(target)
        # Comparison's third positional argument differs from render_pdf.
        def comparison_fixture(source, png, target):
            return image_fixture(source,target)
        argv = self.argv('--backend','local','--max-api-requests','0')
        terminal = io.StringIO()
        with patch('local_pipeline.prepare_local', side_effect=self.local_fixture), \
             patch('convert.run_two_stage', side_effect=AssertionError('No VLM pipeline')), \
             patch('vlm_client.APIConfig.from_env', side_effect=AssertionError('No config')), \
             patch('convert.compile_bounded', side_effect=compile_fixture), \
             patch('convert.render_pdf', side_effect=image_fixture), \
             patch('convert.make_comparison', side_effect=comparison_fixture), redirect_stdout(terminal):
            code = convert.main(argv)
        output = Path(argv[argv.index('--output')+1])
        report = json.loads((output/'report.json').read_text(encoding='utf-8'))
        self.assertEqual(code,2)
        self.assertEqual(report['backend'],'local')
        self.assertEqual(report['status'],'partial')
        self.assertEqual(report['conversion_status'],'complete')
        self.assertEqual(report['recognition_status'],'partial')
        self.assertEqual(report['semantic_status'],'unverified')
        self.assertEqual(report['api_request_count'],0)
        self.assertEqual(report['token_usage']['total_tokens'],0)
        self.assertEqual(report['png']['status'],'rendered')
        self.assertIn('[OK] Local OCR: 3 text regions',terminal.getvalue())
        self.assertTrue((output/'generated.tex').is_file())

    def test_local_ocr_failure_is_reported_without_vlm_or_fake_output(self):
        def failed(image, output, **kwargs):
            return {'status':'failed','stage':'ocr','output_dir':str(output),
                    'recognition':'not_performed','recognition_status':'partial',
                    'api_call_count':0,'error':'OCR runtime unavailable'}
        args = convert.build_parser().parse_args(self.argv('--backend','local','--no-compile'))
        with patch('local_pipeline.prepare_local', side_effect=failed), \
             patch('convert.run_two_stage', side_effect=AssertionError('No fallback')), \
             patch('vlm_client.APIConfig.from_env', side_effect=AssertionError('No config')):
            report = convert.run_conversion(args)
        self.assertEqual(report['status'],'failed')
        self.assertEqual(report['stage'],'ocr')
        self.assertIn('runtime unavailable',report['errors'][0]['message'])
        self.assertNotIn('tex',report['paths'])
        self.assertEqual(report['compile_attempts'],0)

    def test_local_pipeline_preserves_observed_single_and_multiline_boundaries(self):
        from local_pipeline import prepare_local
        ocr = self.worker_response([
            detection('__getitem__(Index_2)', .95, (25,30,180,22)),
            detection('第一行\nvalue_2 = x(0)', .87, (25,85,200,48))])
        ocr.update(coordinate_system='top-left', bbox_format='xywh')
        geometry = {'image':deepcopy(ocr['image']), 'coordinate_system':'top-left',
                    'shapes':[{'id':'container','type':'rectangle','bbox':[10,10,600,350],
                               'style':{'stroke':'dark_blue','fill':'none'}}],
                    'lines':[], 'arrow_candidates':[], 'diagnostics':[]}
        originals = deepcopy((ocr,geometry))
        output = self.output(); output.mkdir()
        with patch('local_pipeline.recognize_local', return_value=ocr), \
             patch('local_geometry.analyze_local_geometry', return_value=geometry), \
             patch('vlm_client.APIConfig.from_env', side_effect=AssertionError('No config')):
            report = prepare_local(self.source,output)
        self.assertEqual(report['status'],'local_scene_ready')
        self.assertEqual((ocr,geometry),originals)
        ir = json.loads((output/'ir_generated.json').read_text(encoding='utf-8'))
        scene = json.loads((output/'scene_generated.json').read_text(encoding='utf-8'))
        styles = json.loads((output/'style_map.json').read_text(encoding='utf-8'))
        policy = json.loads((output/'render_policy.json').read_text(encoding='utf-8'))
        self.assertTrue(inspect_ir(ir)['valid'])
        validate_scene(scene)
        expected_scene, conversion = convert_ir(ir,style_map=styles)
        self.assertTrue(conversion['complete'])
        source_by_text = {row['text']:row for row in ocr['rows']}
        nodes = {node['id']:node for node in scene['nodes']}
        single_ids = []
        for element in ir['elements']:
            if 'raw_text' not in element:
                continue
            node = nodes[element['id']]
            source = source_by_text[element['raw_text']]
            self.assertEqual(node['text'],source['text'])
            self.assertEqual(node['bbox'],source['bbox'])
            self.assertEqual(element['bbox'],source['bbox'])
            if '\n' in source['text']:
                self.assertNotIn('wrap',node)  # Existing paragraph behavior retained.
            else:
                self.assertIs(node['wrap'],False)
                single_ids.append(node['id'])
        self.assertEqual(policy['single_ocr_lines_no_wrap'],single_ids)
        self.assertTrue(policy['text_coordinates_and_content_unchanged'])
        for node in expected_scene['nodes']:
            if node['id'] in single_ids:
                node['wrap'] = False
        self.assertEqual(scene,expected_scene)  # No style, text, geometry or graph edits.
