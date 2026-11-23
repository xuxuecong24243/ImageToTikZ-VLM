"""Bounded typography decisions using explicit synthetic font measurements.

These tests do not claim to recognize or compile the user's image contents.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from local_text_layout import select_typography,optimize_local_text,verify_final_text_layout
from structured_ir.validator import inspect_ir


def fixture():
    ir={'version':'1.0','canvas':{'id':'canvas','width':600,'height':400,'coordinate_system':'top-left'},
        'regions':[{'id':'panel','type':'panel','bbox':[20,20,560,340],'parent_id':None,
                    'children':['background','text_a','text_b']}],
        'elements':[{'id':'background','type':'rounded_rectangle','bbox':[20,20,560,340],'region_id':'panel',
                     'visual_style':{'fill_color':'#EEF5FB','stroke_color':'#254D75','line_width':2,'rounded_radius':15},'z_index':0},
                    {'id':'text_a','type':'text','bbox':[40,40,200,40],'region_id':'panel',
                     'raw_text':'__getitem__(Index_2)','text_type':'plain','reading_order':0,
                     'visual_style':{'text_color_rgb':'#254D75'},'z_index':2},
                    {'id':'text_b','type':'text','bbox':[320,140,180,40],'region_id':'panel',
                     'raw_text':'第二行 value_2','text_type':'plain','reading_order':1,'z_index':2},
                    {'id':'line','type':'connector','bbox':[50,280,450,1],'region_id':None,
                     'points':[[50,280],[500,280]],'z_index':1}],
        'connections':[{'id':'link','source_id':'text_a','target_id':'text_b','source_anchor':'east',
                        'target_anchor':'west','direction':'forward','kind':'flow'}], 'layered':True}
    analysis={'items':{},'counts':{'measured_ink':2}}
    metrics={'status':'measured','items':{},'compile_attempts':1,'font_mapping':{'status':'available'}}
    for ident,ink,kind in [('text_a',[50,45,180,28],'python_code'),('text_b',[325,145,160,28],'mixed_chinese_english')]:
        analysis['items'][ident]={'ink_bbox':ink,'font_candidates':['mono'] if kind=='python_code' else ['sans','serif'],
                                   'text_kind':kind}
        metrics['items'][ident]={'baseline':{'width_px':230,'height_px':45,'effective_font_size':32},
            'candidates':{'mono':{'width_px':180,'height_px':28,'effective_font_size':32},
                          'sans':{'width_px':160,'height_px':28,'effective_font_size':32},
                          'serif':{'width_px':130,'height_px':34,'effective_font_size':32}}}
    return ir,analysis,metrics


class LocalTextOptimizationTests(OfflineTestCase):
    def test_preserves_all_original_content_geometry_and_nontext_styles(self):
        ir,analysis,metrics=fixture();original=deepcopy((ir,analysis,metrics))
        result,report=select_typography(ir,analysis,metrics)
        self.assertEqual((ir,analysis,metrics),original)
        self.assertEqual(result['canvas'],ir['canvas']);self.assertEqual(result['regions'],ir['regions'])
        self.assertEqual(result['connections'],ir['connections'])
        after={e['id']:e for e in result['elements']}
        self.assertEqual(set(after),{e['id'] for e in ir['elements']})
        for element in ir['elements']:
            for field in ['raw_text','bbox','region_id','points','reading_order','text_type','z_index']:
                self.assertEqual(after[element['id']].get(field),element.get(field))
            if 'raw_text' not in element:self.assertEqual(after[element['id']],element)
        self.assertEqual(after['text_a']['visual_style']['text_color_rgb'],'#254D75')
        self.assertTrue(after['text_a']['visual_style']['preserve_spaces'])
        self.assertTrue(inspect_ir(result)['valid'])
        self.assertEqual(report['api_calls'],0)

    def test_selects_font_using_ink_aspect_not_fixed_ocr_height_ratio(self):
        ir,analysis,metrics=fixture()
        result,report=select_typography(ir,analysis,metrics)
        selected=report['selections']['text_b']
        self.assertEqual(selected['font_family'],'sans')
        self.assertAlmostEqual(selected['predicted_width_px'],160)
        self.assertAlmostEqual(selected['predicted_height_px'],28)
        self.assertAlmostEqual(selected['effective_font_size'],32)
        self.assertNotEqual(selected['effective_font_size'],ir['elements'][2]['bbox'][3]*.82)

    def test_large_measured_title_uses_optional_scale_without_breaking_font_size_limit(self):
        ir,analysis,metrics=fixture()
        ir['canvas']['width']=1200;ir['regions'][0]['bbox']=[20,20,1160,340]
        ir['elements'][0]['bbox']=[20,20,1160,340]
        title=ir['elements'][1];title['bbox']=[40,30,1000,150]
        ir['elements'][2]['bbox']=[330,240,180,40]
        analysis['items']['text_a'].update(ink_bbox=[60,40,850,125],text_kind='chinese_title',font_candidates=['sans'])
        metrics['items']['text_a']['candidates']['sans']={'width_px':220,'height_px':32,'effective_font_size':32}
        result,report=select_typography(ir,analysis,metrics)
        style=result['elements'][1]['visual_style']
        self.assertEqual(style['font_size'],80)
        self.assertGreater(style['font_scale'],1)
        self.assertLessEqual(style['font_scale'],3)
        self.assertGreater(report['selections']['text_a']['effective_font_size'],80)
        self.assertTrue(inspect_ir(result)['valid'])

    def test_missing_ink_or_font_candidate_keeps_existing_typography(self):
        ir,analysis,metrics=fixture()
        analysis['items']['text_a']['ink_bbox']=None
        metrics['items']['text_b']['candidates']={}
        result,report=select_typography(ir,analysis,metrics)
        self.assertEqual(result['elements'],ir['elements'])
        self.assertEqual(report['selections'],{})
        self.assertEqual(len(report['diagnostics']),2)

    def overlapping(self,separation):
        ir,analysis,metrics=fixture()
        for index,ident in enumerate(['text_a','text_b']):
            x=40+index*separation
            ir['elements'][index+1]['bbox']=[x,40,120,40]
            analysis['items'][ident].update(ink_bbox=[x+5,45,110,28],font_candidates=['sans'],text_kind='english_prose')
            metrics['items'][ident]={'baseline':{'width_px':70,'height_px':20,'effective_font_size':24},
                'candidates':{'sans':{'width_px':110,'height_px':28,'effective_font_size':32}}}
        return ir,analysis,metrics

    def test_two_shrink_passes_resolve_new_overlap_without_moving_anchors(self):
        ir,analysis,metrics=self.overlapping(95)
        result,report=select_typography(ir,analysis,metrics)
        self.assertEqual(report['budget']['actual_shrink_passes'],2)
        self.assertEqual(report['predicted_after']['counts']['hard_conflicts'],0)
        self.assertEqual(len(report['selections']),2)
        for selection in report['selections'].values():
            self.assertEqual(selection['status'],'bounded_conflict_shrink')
            self.assertLess(selection['effective_font_size'],32)
        self.assertEqual([e['bbox'] for e in result['elements']],[e['bbox'] for e in ir['elements']])

    def test_unresolved_increased_hard_conflicts_revert_all_typography(self):
        ir,analysis,metrics=self.overlapping(80)
        result,report=select_typography(ir,analysis,metrics)
        self.assertEqual(report['budget']['actual_shrink_passes'],2)
        self.assertEqual(report['selections'],{})
        self.assertEqual(result['elements'],ir['elements'])
        self.assertEqual(report['predicted_after']['counts']['hard_conflicts'],report['before']['counts']['hard_conflicts'])
        self.assertIn('candidate_increased_hard_conflicts',[d['code'] for d in report['diagnostics']])

    def test_equivalent_math_segments_do_not_change_raw_text_or_text_type(self):
        ir,analysis,metrics=fixture();ir['elements'][1]['raw_text']='x² ∈ ℝ'
        analysis['items']['text_a'].update(text_kind='math_expression',font_candidates=['serif'],
            render_segments=[{'type':'math','content':'x^{2}'},{'type':'text','content':' '},
                             {'type':'math','content':r'\in'},{'type':'text','content':' '},
                             {'type':'math','content':r'\mathbb{R}'}])
        result,report=select_typography(ir,analysis,metrics)
        element=result['elements'][1]
        self.assertEqual(element['raw_text'],'x² ∈ ℝ')
        self.assertEqual(element['text_type'],'plain')
        self.assertEqual(element['render_segments'],analysis['items']['text_a']['render_segments'])
        self.assertFalse(element['visual_style']['preserve_spaces'])

    def test_disabled_measurement_never_compiles_and_keeps_inputs(self):
        ir,analysis,metrics=fixture();styles={'code':{'font_size':24}}
        with tempfile.TemporaryDirectory() as directory, \
             patch('local_text_layout.analyze_source_text',return_value=analysis), \
             patch('local_text_layout.measure_text_candidates',side_effect=AssertionError('No compile')):
            result,newstyles,report=optimize_local_text('unused.png',{'rows':[]},ir,{},styles,{},directory,enabled=False)
            self.assertEqual(result,ir);self.assertEqual(newstyles,styles)
            self.assertEqual(report['status'],'skipped');self.assertEqual(report['compile_attempts'],0)
            saved=json.loads((Path(directory)/'text_collision_report.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['status'],'unmeasured')

    def test_missing_font_fallback_records_reason_without_fake_dimensions(self):
        ir,analysis,_=fixture();fallback={'status':'unavailable','items':{},'compile_attempts':0,
            'font_mapping':{'status':'missing','reason':'Required test font unavailable'},'api_calls':0}
        with tempfile.TemporaryDirectory() as directory, \
             patch('local_text_layout.analyze_source_text',return_value=analysis), \
             patch('local_text_layout.measure_text_candidates',return_value=fallback) as measurement:
            result,newstyles,report=optimize_local_text('unused.png',{'rows':[]},ir,{}, {},{},directory)
            self.assertEqual(result,ir);self.assertEqual(report['status'],'unavailable')
            self.assertEqual(measurement.call_count,1)
            self.assertEqual(json.loads((Path(directory)/'text_measurement_report.json').read_text(encoding='utf-8'))['items'],{})
            self.assertFalse((Path(directory)/'text_layout_selection.json').exists())

    def test_scene_measurement_failure_retains_original_typography(self):
        ir,analysis,_=fixture();styles={'original':{'font_size':24}}
        original=deepcopy((ir,styles))
        with tempfile.TemporaryDirectory() as directory, \
             patch('local_text_layout.analyze_source_text',return_value=analysis), \
             patch('local_text_layout._scene',side_effect=ValueError('Safe partial Scene is not measurable')), \
             patch('local_text_layout.measure_text_candidates',side_effect=AssertionError('No compiler')) as measure:
            result,newstyles,report=optimize_local_text('unused.png',{'rows':[]},ir,{},styles,{},directory)
            self.assertEqual((result,newstyles),original)
            self.assertEqual((ir,styles),original)
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['reason'],'measurement_unavailable')
            self.assertEqual(report['diagnostics'][0]['action'],'original_typography_retained')
            self.assertEqual(report['items'],{})
            self.assertEqual(report['font_mapping']['status'],'unmeasured')
            self.assertNotIn('compile_attempts',report)
            measure.assert_not_called()
            self.assertFalse((Path(directory)/'text_layout_selection.json').exists())

    def test_final_invalid_or_missing_markers_do_not_discard_existing_pdf(self):
        ir,_,_=fixture();scene={'canvas':{'measure_text':True,'scale':.4}}
        for log in [None,'No text dimension markers in this successful log.',
                    'ITTEXT|text_a|not-a-number pt|20pt']:
            with self.subTest(log=log),tempfile.TemporaryDirectory() as directory:
                output=Path(directory)
                prior={'status':'predicted_pending_final_compile','before':{'counts':{'hard_conflicts':0}}}
                (output/'text_collision_report.json').write_text(json.dumps(prior),encoding='utf-8')
                pdf=output/'generated.pdf';pdf.write_bytes(b'Existing compiler product retained verbatim')
                original=pdf.read_bytes()
                if log is not None:(output/'generated.log').write_text(log,encoding='utf-8')
                result=verify_final_text_layout(ir,scene,output)
                self.assertEqual(result['status'],'unmeasured_final_compile')
                self.assertTrue(result['reason'])
                saved=json.loads((output/'text_collision_report.json').read_text(encoding='utf-8'))
                self.assertEqual(saved['diagnostics'][0]['code'],'final_measurement_unavailable')
                self.assertEqual(saved['before'],prior['before'])
                self.assertNotIn('actual_metrics',saved)
                self.assertEqual(pdf.read_bytes(),original)
