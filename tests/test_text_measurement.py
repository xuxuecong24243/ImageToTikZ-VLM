"""Batch measurement contracts; real compiler smoke/acceptance runs are separate."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from latex_compiler import CompileResult
from text_measurement import (parse_text_measurements, build_measurement_batch,
                              measure_text_candidates, inspect_font_mapping)


def scene(text='中文 __getitem__(index)'):
    return {'canvas':{'width':700,'height':400,'coordinate_system':'top-left'},
        'styles':{},'nodes':[{'id':'text1','type':'text','bbox':[20,20,500,45],
        'text':text,'wrap':False,'style':{'font_size':30}}],'edges':[],'labels':[]}


def fonts():
    return {'engine':'mock-xelatex','status':'checked','families':{
        name:{'available':True,'files':[]} for name in ('serif','sans','mono')}}


class TextMeasurementTests(OfflineTestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(prefix='text-metrics-contract-')
        self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)

    def test_parse_dimensions_are_actual_pt_converted_to_source_pixels(self):
        r=parse_text_measurements('noise\nITTEXT|a|100.25pt|20.5pt\n',.5)
        self.assertEqual(r['a']['width_px'],200.5)
        self.assertEqual(r['a']['height_px'],41)
        self.assertEqual(r['a']['width_pt'],100.25)

    def test_parse_rejects_missing_duplicate_negative_and_bad_scale(self):
        for raw in ['ITTEXT|a|NaNpt|4pt','ITTEXT|a|-1pt|2pt',
                    'ITTEXT|a|1pt|2pt\nITTEXT|a|1pt|2pt','ITTEXT|a|1pt|']:
            with self.subTest(raw=raw),self.assertRaises(ValueError):parse_text_measurements(raw)
        for scale in [True,0,float('inf')]:
            with self.assertRaises(ValueError):parse_text_measurements('',scale)

    def test_parse_allows_log_wrap_after_separator(self):
        r=parse_text_measurements('ITTEXT|long_identifier|\n42.2pt|\n11.4pt',1.)
        self.assertEqual(r['long_identifier']['height_px'],11.4)
        long_id='a'*64
        wrapped='ITTEXT|'+long_id[:30]+'\n'+long_id[30:]+'|42.\n25pt|11.5p\nt'
        self.assertEqual(parse_text_measurements(wrapped,1.)[long_id]['width_px'],42.25)

    def test_batch_is_pure_preserves_baseline_and_has_three_bounded_candidates(self):
        original=scene();before=deepcopy(original)
        document,requests=build_measurement_batch(original)
        self.assertEqual(original,before)
        self.assertEqual(len(requests),4)
        self.assertEqual(requests[0]['font_size'],30)
        self.assertEqual({r['font_size'] for r in requests[1:]},{32.})
        self.assertEqual({r['font_scale'] for r in requests[1:]},{1.})
        self.assertEqual(document.count('ITTEXT|'),4)
        self.assertNotIn('\\includegraphics',document)

    def test_literal_code_and_multiline_content_use_shared_escape_once(self):
        value='  __getitem__(x[2]) {"A"} \\path\nNext_3'
        data=scene(value);data['nodes'][0]['style'].update(font_family='mono',preserve_spaces=True)
        doc,_=build_measurement_batch(data,families_by_id={'text1':['mono']})
        self.assertIn(r'\_\_getitem\_\_',doc)
        self.assertIn(r'\textbackslash{}path',doc)
        self.assertIn(r'\{"A"\}',doc)
        self.assertEqual(data['nodes'][0]['text'],value)

    def test_candidate_rich_math_keeps_raw_baseline_text(self):
        original=scene('x ∈ R');candidate=deepcopy(original)
        candidate['nodes'][0].update(mode='rich',segments=[{'type':'text','content':'x '},
            {'type':'math','content':r'\in'},{'type':'text','content':' R'}])
        doc,requests=build_measurement_batch(original,candidate_scene=candidate,families_by_id={'text1':['serif']})
        self.assertEqual(requests[0]['render_mode'],'text')
        self.assertEqual(requests[1]['render_mode'],'rich')
        self.assertIn('x ∈ R',doc);self.assertIn(r'$\in$',doc)

    def test_candidate_may_not_rewrite_source_and_family_budget_is_checked(self):
        original=scene();candidate=deepcopy(original);candidate['nodes'][0]['text']='changed'
        with self.assertRaises(ValueError):build_measurement_batch(original,candidate_scene=candidate)
        for families in [['serif','serif'],['system-font'],['serif','sans','mono','serif'],'sans']:
            with self.subTest(families=families),self.assertRaises(ValueError):
                build_measurement_batch(original,families_by_id={'text1':families})

    def test_missing_compiler_does_not_create_fake_measurements_or_launch(self):
        with patch('text_measurement._engine_path',return_value=None), \
             patch('text_measurement.compile_bounded',side_effect=AssertionError('No compile')):
            r=measure_text_candidates(scene(),self.root)
        self.assertEqual(r['status'],'unavailable');self.assertEqual(r['items'],{})
        self.assertEqual(r['compile_attempts'],0)
        self.assertFalse((self.root/'text_metrics').exists())

    def test_missing_font_candidates_are_skipped_with_explicit_mapping(self):
        mapping=fonts();mapping['families']['mono']['available']=False
        _,requests=build_measurement_batch(scene(),available_families=['serif','sans'])
        self.assertNotIn('mono',[r['variant'] for r in requests])
        mapping['families']['serif']['available']=False;mapping['families']['sans']['available']=False
        with patch('text_measurement.inspect_font_mapping',return_value=mapping), \
             patch('text_measurement.compile_bounded',side_effect=AssertionError('No fake fallback')):
            r=measure_text_candidates(scene(),self.root)
        self.assertEqual(r['diagnostics'][0]['code'],'fonts_unavailable')

    def test_one_compile_returns_all_measured_values_and_retains_request_provenance(self):
        calls=[]
        def compiler(tex,**kwargs):
            calls.append(kwargs);p=Path(tex);log=p.with_suffix('.log')
            requests=json.loads((p.parent/'requests.json').read_text(encoding='utf-8'))
            log.write_text('\n'.join('ITTEXT|%s|40pt|10pt'%r['measurement_id'] for r in requests),encoding='utf-8')
            pdf=p.with_suffix('.pdf');pdf.write_bytes(b'unit-fixture-not-acceptance')
            return CompileResult(True,0,False,pdf,log,log,'',[])
        with patch('text_measurement.inspect_font_mapping',return_value=fonts()),patch('text_measurement.compile_bounded',side_effect=compiler):
            r=measure_text_candidates(scene(),self.root)
        self.assertEqual(len(calls),1);self.assertEqual(r['compile_attempts'],1)
        self.assertEqual(r['status'],'measured');self.assertEqual(r['measurement_count'],4)
        self.assertEqual(r['items']['text1']['baseline']['width_px'],100)
        self.assertEqual(r['api_calls'],0)

    def test_compile_failure_stops_once_without_fabricated_dimensions(self):
        def failed(tex,**kwargs):
            p=Path(tex);log=p.with_suffix('.log');log.write_text('Missing font',encoding='utf-8')
            return CompileResult(False,1,False,None,log,log,'Missing font',[])
        with patch('text_measurement.inspect_font_mapping',return_value=fonts()),patch('text_measurement.compile_bounded',side_effect=failed) as process:
            r=measure_text_candidates(scene(),self.root)
        self.assertEqual(process.call_count,1);self.assertEqual(r['status'],'failed')
        self.assertEqual(r['items'],{});self.assertTrue((self.root/'text_metrics/measurement.tex').is_file())

    def test_incomplete_measurement_log_is_not_a_success(self):
        def incomplete(tex,**kwargs):
            p=Path(tex);log=p.with_suffix('.log');log.write_text('ITTEXT|metric_0001|4pt|2pt',encoding='utf-8')
            return CompileResult(True,0,False,p.with_suffix('.pdf'),log,log,'',[])
        with patch('text_measurement.inspect_font_mapping',return_value=fonts()),patch('text_measurement.compile_bounded',side_effect=incomplete):
            r=measure_text_candidates(scene(),self.root)
        self.assertEqual(r['status'],'failed');self.assertEqual(r['diagnostics'][0]['code'],'incomplete_measurements')

    def test_malformed_measurement_log_returns_failure_for_typography_fallback(self):
        def malformed(tex,**kwargs):
            p=Path(tex);log=p.with_suffix('.log');log.write_text('ITTEXT|metric_0001|bad|2pt',encoding='utf-8')
            return CompileResult(True,0,False,p.with_suffix('.pdf'),log,log,'',[])
        with patch('text_measurement.inspect_font_mapping',return_value=fonts()),patch('text_measurement.compile_bounded',side_effect=malformed) as process:
            r=measure_text_candidates(scene(),self.root)
        self.assertEqual(process.call_count,1);self.assertEqual(r['status'],'failed')
        self.assertEqual(r['items'],{});self.assertEqual(r['diagnostics'][0]['code'],'invalid_measurement_log')
