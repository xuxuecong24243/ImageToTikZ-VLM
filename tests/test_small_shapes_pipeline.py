"""v0.5 integration: real CV/IR/TikZ with explicitly synthetic OCR responses.

These fixtures are unit tests, never acceptance evidence of real OCR accuracy.
"""
from copy import deepcopy
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from unittest.mock import patch
from PIL import Image,ImageDraw,ImageFont
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

import convert
from local_ocr import normalize_ocr_result
from local_arrow_detection import analyze_local_arrows
from local_connection_mapping import map_local_connections
from local_small_shapes import recover_small_shapes
from structured_ir.validator import inspect_ir
from structured_ir.converter import convert_ir
from structured_tikz.generator import generate_tikz
from structured_tikz.validator import validate_scene


class SmallShapesPipelineTests(OfflineTestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='small-pipeline-');self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.source=self.root/'流程 输入.png'
        im=Image.new('RGB',(720,400),'white');d=ImageDraw.Draw(im)
        for box in [(30,80,300,320),(420,80,690,320)]:d.rectangle(box,outline='#2464A5',width=2)
        d.ellipse((60,140,100,180),fill='#2464A5')
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
        d.text((80,160),'7',font=font,anchor='mm',fill='white')
        d.text((70,100),'Step_A',font=font,fill='black')
        d.line((307,210,398,210),fill='#2464A5',width=3)
        d.polygon([(392,200),(412,210),(392,220)],fill='#2464A5')
        im.save(self.source)
        self.sha=hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.raw=[[[[65,96],[185,96],[185,128],[65,128]],'Step_A',.96]]

    def ocr(self,image,output,**kwargs):
        rows,diagnostics=normalize_ocr_result(self.raw,(720,400))
        result={'engine':'synthetic_unit_fixture','engine_version':'test_only',
            'image':{'width':720,'height':400,'sha256':self.sha},'coordinate_system':'top-left',
            'bbox_format':'xywh','rows':rows,'engine_result':self.raw,'diagnostics':diagnostics}
        (Path(output)/'ocr_raw.json').write_text(json.dumps(result),encoding='utf8')
        return result

    def crop_ocr(self,image,requests,output,**kwargs):
        return {'source_image_sha256':self.sha,'requests':deepcopy(requests),
                'engine_result':[['7',.98] for _ in requests],'synthetic_unit_fixture':True}

    def test_local_stages_preserve_prior_text_and_reach_editable_output(self):
        output=self.root/'result';prior={}
        def capture(image,ir,folder,**kwargs):
            prior['ir']=deepcopy(ir)
            return recover_small_shapes(image,ir,folder,**kwargs)
        with patch('local_pipeline.recognize_local',side_effect=self.ocr), \
             patch('local_small_shapes.recognize_crops',side_effect=self.crop_ocr), \
             patch('local_small_shapes.recover_small_shapes',side_effect=capture), \
             patch('convert.run_two_stage',side_effect=AssertionError('No VLM')), \
             patch('vlm_client.APIConfig.from_env',side_effect=AssertionError('No real API configuration')), \
             redirect_stdout(io.StringIO()):
            code=convert.main(['--input',str(self.source),'--output',str(output),'--backend','local','--no-compile'])
        report=json.loads((output/'report.json').read_text(encoding='utf8'))
        self.assertEqual(code,2);self.assertEqual(report['status'],'partial')
        self.assertEqual(report['api_request_count'],0);self.assertEqual(report['local_features_version'],'0.5')
        for filename in ['small_shape_detection.json','arrow_detection.json','connection_mapping.json']:
            self.assertIn(filename,report['paths']);self.assertTrue((output/filename).is_file())
        ir=json.loads((output/'ir_generated.json').read_text(encoding='utf8'))
        self.assertTrue(inspect_ir(ir)['valid'])
        final={e['id']:e for e in ir['elements']}
        for old in prior['ir']['elements']:
            if 'raw_text' in old:self.assertEqual(old,final[old['id']])
        badge=next(e for e in ir['elements'] if e['id'].startswith('local_badge_text_'))
        self.assertEqual(badge['raw_text'],'7');self.assertEqual(badge['confidence'],.98)
        self.assertEqual(badge['visual_style']['text_color_rgb'],'#FFFFFF')
        self.assertEqual(len(ir['connections']),1)
        scene=json.loads((output/'scene_generated.json').read_text(encoding='utf8'));validate_scene(scene)
        self.assertEqual(next(n for n in scene['nodes'] if n['id']==badge['id'])['text'],'7')
        tex=(output/'generated.tex').read_text(encoding='utf8')
        self.assertIn('% text: '+badge['id'],tex);self.assertIn('FFFFFF',tex)
        self.assertIn('Latex',tex);self.assertNotIn('\\includegraphics',tex)

    def test_crop_ocr_failure_stays_partial_without_invented_digit(self):
        output=self.root/'failed_crop'
        with patch('local_pipeline.recognize_local',side_effect=self.ocr), \
             patch('local_small_shapes.recognize_crops',side_effect=ValueError('Unit fixture: OCR runtime absent')), \
             patch('convert.run_two_stage',side_effect=AssertionError('No API fallback')), \
             patch('vlm_client.APIConfig.from_env',side_effect=AssertionError('No configuration')):
            result=convert.run_conversion(convert.build_parser().parse_args(
                ['--input',str(self.source),'--output',str(output),'--backend','local','--no-compile']))
        self.assertEqual(result['status'],'partial');self.assertEqual(result['api_request_count'],0)
        self.assertEqual(result['latex']['status'],'generated')
        small=json.loads((output/'small_shape_detection.json').read_text(encoding='utf8'))
        self.assertEqual(small['counts']['recovered_texts'],0)
        self.assertTrue(any(d['code']=='local_crop_ocr_failed' for d in small['diagnostics']))
        ir=json.loads((output/'ir_generated.json').read_text(encoding='utf8'))
        self.assertFalse(any(e['id'].startswith('local_badge_text_') for e in ir['elements']))

    def test_real_detector_python_values_reach_mapper_without_json_roundtrip(self):
        detection=analyze_local_arrows(self.source,[])
        self.assertEqual(len(detection['arrows']),1)
        arrow=detection['arrows'][0]
        for value in [arrow['head']['width_px'],arrow['style']['line_width'],arrow['head']['length_px']]:
            self.assertIn(type(value),(float,int))
        ir={'version':'1.0','canvas':{'id':'canvas','width':720,'height':400,'coordinate_system':'top-left'},
            'regions':[],'elements':[
                {'id':'source','type':'rectangle','bbox':[30,80,270,240],'region_id':None},
                {'id':'target','type':'rectangle','bbox':[420,80,270,240],'region_id':None}],
            'connections':[],'provenance':{'image_sha256':self.sha}}
        enriched,mapping,styles=map_local_connections(ir,detection)
        self.assertEqual(len(enriched['connections']),1,mapping)
        edge=enriched['connections'][0]
        self.assertEqual((edge['source_id'],edge['target_id']),('source','target'))
        self.assertEqual(edge['points'],arrow['points'])
        self.assertTrue(inspect_ir(enriched)['valid'])
        scene,converted=convert_ir(enriched,style_map=styles)
        self.assertTrue(converted['scene_valid'])
        self.assertEqual(converted['converted_connections'],[edge['id']])
        # Existing converter truthfully reports observed gaps from ideal anchors
        # as partial; an integration must not reroute merely to claim complete.
        self.assertFalse(converted['complete'])
        self.assertTrue(any(d['code']=='connection_path_anchor_mismatch' for d in converted['diagnostics']))
        self.assertIn('Latex',generate_tikz(scene))

    def test_vlm_default_and_local_authorization_gate_precede_new_stages(self):
        parser=convert.build_parser()
        self.assertEqual(parser.parse_args(['--input','a.png','--output','out']).backend,'vlm')
        for label,extra in [('vlm',[]),('local_api',['--backend','local','--allow-api'])]:
            with self.subTest(label=label), \
                 patch('local_pipeline.prepare_local',side_effect=AssertionError('Gate must stop before local stages')), \
                 patch('convert.run_two_stage',side_effect=AssertionError('No paid call')), \
                 patch('vlm_client.APIConfig.from_env',side_effect=AssertionError('No configuration')):
                report=convert.run_conversion(parser.parse_args(
                    ['--input',str(self.source),'--output',str(self.root/label)]+extra))
            self.assertEqual(report['status'],'failed');self.assertEqual(report['api_request_count'],0)

