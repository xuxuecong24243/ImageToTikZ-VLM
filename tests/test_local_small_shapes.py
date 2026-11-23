"""Measured small shapes and crop OCR contracts; real OCR acceptance is separate."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from PIL import Image,ImageDraw,ImageFont
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from local_small_shapes import detect_small_circles,apply_small_shape_results,recover_small_shapes,_fit_new_text
from local_ocr import recognize_crops
from structured_ir.validator import inspect_ir
from structured_ir.converter import convert_ir
from structured_tikz.generator import generate_tikz


class LocalSmallShapesTests(OfflineTestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)

    def image(self,*,letter='A',diameter=39,fill='#2464A5',ink='white',existing=True):
        image=Image.new('RGB',(300,180),'white');draw=ImageDraw.Draw(image)
        box=[30,35,diameter,diameter]
        draw.ellipse((30,35,30+diameter-1,35+diameter-1),fill=fill)
        if letter:
            font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',max(8,round(diameter*.55)))
            draw.text((30+diameter/2,35+diameter/2),letter,font=font,anchor='mm',fill=ink)
        path=self.root/'原图.png';image.save(path)
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        ir={'version':'1.0','canvas':{'id':'canvas','width':300,'height':180,'coordinate_system':'top-left'},
            'regions':[],'elements':[],'connections':[],'provenance':{'image_sha256':sha}}
        if existing:
            ir['regions']=[{'id':'circle_region','type':'unknown','bbox':box[:],'parent_id':None,'children':['badge']}]
            ir['elements']=[{'id':'badge','type':'circle','bbox':box[:],'region_id':'circle_region',
                'visual_style':{'fill_color':fill,'stroke':'none'},'z_index':2}]
        return path,ir

    def result(self,report,text='A',confidence=.97):
        return {'requests':[{'id':c['id'],'source_bbox':c['ink_bbox']} for c in report['circles']],
                'engine_result':[[text,confidence] for c in report['circles']]}

    def test_circle_measurement_retains_center_radius_fill_white_foreground_and_ink(self):
        path,ir=self.image();before=deepcopy(ir)
        report=detect_small_circles(path,ir);c=report['circles'][0]
        self.assertEqual(len(report['circles']),1)
        self.assertEqual(c['existing_element_id'],'badge');self.assertEqual(c['center'],[49.5,54.5])
        self.assertEqual(c['radius'],19.5);self.assertEqual(c['fill_color'],'#2464A5')
        self.assertEqual(c['text_color_rgb'],'#FFFFFF');self.assertIsNotNone(c['ink_bbox'])
        self.assertEqual(ir,before)

    def test_small_circle_below_old_analyzer_minimum_is_detected(self):
        path,ir=self.image(diameter=17,existing=False)
        report=detect_small_circles(path,ir)
        self.assertEqual(len(report['circles']),1)
        self.assertIsNone(report['circles'][0]['existing_element_id'])
        self.assertLess(report['circles'][0]['bbox'][2],20)

    def test_gray_circle_with_black_letter_has_measured_dark_ink(self):
        path,ir=self.image(fill='#999999',ink='black')
        circle=detect_small_circles(path,ir)['circles'][0]
        self.assertEqual(circle['fill_color'],'#999999')
        self.assertEqual(circle['text_color_rgb'],'#000000')

    def test_empty_circle_does_not_invent_label_or_invoke_ocr(self):
        path,ir=self.image(letter='');output=self.root/'out';output.mkdir()
        with patch('local_small_shapes.recognize_crops',side_effect=AssertionError('No absent glyph OCR')):
            result,report=recover_small_shapes(path,ir,output)
        self.assertEqual(result['elements'],ir['elements'])
        self.assertEqual(report['counts']['recovered_texts'],0)
        self.assertEqual(report['associations'][0]['reason'],'no_reliable_internal_ink')

    def test_existing_inside_text_is_associated_without_duplicate_or_modification(self):
        path,ir=self.image();ir['elements'].append({'id':'old_text','type':'text','bbox':[43,45,12,17],
            'region_id':'circle_region','raw_text':'A','text_type':'plain','reading_order':0})
        ir['regions'][0]['children'].append('old_text');before=deepcopy(ir)
        report=detect_small_circles(path,ir);result,report=apply_small_shape_results(ir,report)
        self.assertEqual(result['elements'],before['elements'])
        self.assertEqual(report['associations'][0]['status'],'existing_text_retained')

    def test_recovered_text_independent_centered_layered_and_children_consistent(self):
        path,ir=self.image();before=deepcopy(ir);detection=detect_small_circles(path,ir)
        result,report=apply_small_shape_results(ir,detection,self.result(detection))
        self.assertEqual(ir,before);self.assertEqual(result['elements'][0],before['elements'][0])
        text=result['elements'][-1];self.assertEqual(text['raw_text'],'A');self.assertEqual(text['confidence'],.97)
        self.assertEqual(text['visual_style']['text_color_rgb'],'#FFFFFF')
        self.assertEqual(text['region_id'],'circle_region')
        self.assertIn(text['id'],result['regions'][0]['children'])
        self.assertGreater(text['z_index'],result['elements'][0]['z_index'])
        self.assertEqual([text['bbox'][0]+text['bbox'][2]/2,text['bbox'][1]+text['bbox'][3]/2],[49.5,54.5])
        self.assertTrue(inspect_ir(result)['valid'])
        scene,converted=convert_ir(result);self.assertTrue(converted['complete'])
        tex=generate_tikz(scene)
        self.assertIn('FFFFFF',tex)
        self.assertLess(tex.index('% fill: badge'),tex.index('% text: '+text['id']))

    def test_new_small_circle_and_region_have_unique_bidirectional_references(self):
        path,ir=self.image(diameter=17,existing=False);detection=detect_small_circles(path,ir)
        result,report=apply_small_shape_results(ir,detection,self.result(detection,'7'))
        self.assertEqual(len(result['elements']),2);self.assertEqual(len(result['regions']),1)
        self.assertEqual(result['elements'][0]['type'],'circle')
        self.assertEqual(set(result['regions'][0]['children']),{e['id'] for e in result['elements']})
        self.assertEqual(len({e['id'] for e in result['elements']}),2)
        self.assertTrue(inspect_ir(result)['valid'])

    def test_low_confidence_empty_and_nonlabel_results_are_not_promoted(self):
        path,ir=self.image();detection=detect_small_circles(path,ir)
        for text,confidence in [('3',.4),('',.99),('A long label',.99),('?',.99)]:
            with self.subTest(text=text):
                result,report=apply_small_shape_results(ir,detection,self.result(detection,text,confidence))
                self.assertEqual(result['elements'],ir['elements'])
                self.assertEqual(report['associations'][0]['reason'],'uncertain_local_ocr')
                self.assertEqual(report['associations'][0]['raw_text'],text)

    def test_label_is_literal_not_completed_from_sequence(self):
        path,ir=self.image();detection=detect_small_circles(path,ir)
        result,report=apply_small_shape_results(ir,detection,self.result(detection,'9'))
        self.assertEqual(result['elements'][-1]['raw_text'],'9')
        self.assertFalse(report['preservation']['ordinal_completion'])

    def test_source_hash_mismatch_rejected_before_local_ocr(self):
        path,ir=self.image();ir['provenance']['image_sha256']='a'*64
        with self.assertRaises(ValueError):detect_small_circles(path,ir)

    def test_crop_ocr_failure_retains_original_scene_and_reports_reason(self):
        path,ir=self.image();output=self.root/'out';output.mkdir()
        with patch('local_small_shapes.recognize_crops',side_effect=ValueError('Missing runtime')):
            result,report=recover_small_shapes(path,ir,output)
        self.assertEqual(result['elements'],ir['elements'])
        self.assertEqual(report['diagnostics'][-1]['code'],'local_crop_ocr_failed')
        self.assertTrue((output/'small_shape_crops/small_circle_0001_original.png').is_file())
        self.assertTrue((output/'small_shape_detection.json').is_file())

    def test_new_ocr_batch_rejects_missing_runtime_and_no_network_fallback(self):
        with self.assertRaisesRegex(ValueError,'runtime is missing'):
            recognize_crops('missing.png',[{'id':'x','path':'missing.png'}],self.root/'missing',python=self.root/'not-python.exe')

    def test_decorative_rectangle_is_not_a_small_circle(self):
        path,ir=self.image(letter='',existing=False)
        image=Image.new('RGB',(300,180),'white');ImageDraw.Draw(image).rectangle((40,40,75,75),fill='#2464A5');image.save(path)
        ir['provenance']['image_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(detect_small_circles(path,ir)['circles'],[])

    def test_optional_measurement_fits_only_new_text_with_circle_capacity(self):
        path,ir=self.image();detection=detect_small_circles(path,ir)
        enriched,report=apply_small_shape_results(ir,detection,self.result(detection))
        ident=enriched['elements'][-1]['id'];before=deepcopy(enriched)
        metrics={'status':'measured','compile_attempts':1,'items':{ident:{'candidates':{'serif':{
            'height_px':16.,'width_px':12.,'effective_font_size':32.}}}}}
        output=self.root/'fit';output.mkdir()
        with patch('text_measurement.measure_text_candidates',return_value=metrics) as measured:
            result,fit=_fit_new_text(enriched,report,output,xelatex='unit-only',timeout=30)
        self.assertEqual(measured.call_count,1)
        self.assertEqual([n['id'] for n in measured.call_args.args[0]['nodes']],[ident])
        self.assertEqual(result['elements'][0],before['elements'][0])
        self.assertEqual(enriched,before)
        self.assertGreater(result['elements'][-1]['visual_style']['font_size'],before['elements'][-1]['visual_style']['font_size'])
        self.assertEqual(fit['status'],'measured_new_badge_text_only')

    def test_unavailable_new_text_metrics_retain_estimate(self):
        path,ir=self.image();detection=detect_small_circles(path,ir)
        enriched,report=apply_small_shape_results(ir,detection,self.result(detection));before=deepcopy(enriched)
        output=self.root/'fit';output.mkdir()
        with patch('text_measurement.measure_text_candidates',side_effect=ValueError('No usable font')):
            result,fit=_fit_new_text(enriched,report,output,xelatex='missing',timeout=30)
        self.assertEqual(result,before);self.assertEqual(fit['status'],'unavailable')
