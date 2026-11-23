"""New visual stage integration with real CV and a clearly synthetic OCR fixture.

True OCR/compilation acceptance is separately executed on both user images.
"""
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from unittest.mock import patch

from PIL import Image, ImageDraw
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

import convert
from local_ocr import normalize_ocr_result
from structured_ir.validator import inspect_ir
from structured_tikz.validator import validate_scene


class LocalStylePipelineTests(OfflineTestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='visual-stage-contract-')
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.source=self.root/'样式 输入.png'
        image=Image.new('RGB',(600,400),'white')
        draw=ImageDraw.Draw(image)
        draw.rounded_rectangle((30,30,570,370),radius=18,fill='#f4f8fc',outline='#749db8',width=3)
        draw.rectangle((32,48,568,110),fill='#30577a')
        draw.text((70,70),'Header_1',fill='white')
        draw.text((70,160),'Code(x_2)',fill='#152940')
        image.save(self.source)
        self.raw=[[[[65,65],[160,65],[160,90],[65,90]],'Header_1',.91],
                  [[[65,155],[180,155],[180,180],[65,180]],'Code(x_2)',.88]]

    def ocr_fixture(self,image,output,**kwargs):
        rows,diagnostics=normalize_ocr_result(self.raw,(600,400))
        report={'engine':'synthetic_unit_test_only','engine_version':'fixture',
                'image':{'width':600,'height':400,'sha256':hashlib.sha256(self.source.read_bytes()).hexdigest()},
                'coordinate_system':'top-left','bbox_format':'xywh','rows':rows,
                'engine_result':self.raw,'diagnostics':diagnostics}
        (Path(output)/'ocr_raw.json').write_text(json.dumps(report),encoding='utf-8')
        return report

    def test_real_cv_style_stage_reaches_editable_cli_without_api(self):
        output=self.root/'结果'
        with patch('local_pipeline.recognize_local',side_effect=self.ocr_fixture), \
             patch('convert.run_two_stage',side_effect=AssertionError('No VLM pipeline')), \
             patch('vlm_client.APIConfig.from_env',side_effect=AssertionError('No configuration')), \
             redirect_stdout(io.StringIO()):
            code=convert.main(['--input',str(self.source),'--output',str(output),'--backend','local','--no-compile'])
        report=json.loads((output/'report.json').read_text(encoding='utf-8'))
        self.assertEqual(code,2)
        self.assertEqual(report['version'],'0.3')
        self.assertEqual(report['api_request_count'],0)
        self.assertEqual(report['latex']['status'],'generated')
        self.assertIn('style_extraction_report.json',report['paths'])
        self.assertIn('layer_analysis.json',report['paths'])
        ir=json.loads((output/'ir_generated.json').read_text(encoding='utf-8'))
        scene=json.loads((output/'scene_generated.json').read_text(encoding='utf-8'))
        self.assertTrue(inspect_ir(ir)['valid'])
        validate_scene(scene)
        self.assertTrue(ir['layered'])
        self.assertTrue(scene['canvas']['layered'])
        self.assertEqual([e['raw_text'] for e in ir['elements'] if 'raw_text' in e],['Header_1','Code(x_2)'])
        self.assertTrue(any(e.get('visual_style',{}).get('fill_color') for e in ir['elements']))
        tex=(output/'generated.tex').read_text(encoding='utf-8')
        self.assertIn('\\definecolor',tex)
        self.assertNotIn('\\includegraphics',tex)

    def test_style_failure_keeps_measurements_and_reports_stage(self):
        output=self.root/'failed'
        with patch('local_pipeline.recognize_local',side_effect=self.ocr_fixture), \
             patch('visual_style_extractor.extract_visual_styles',side_effect=ValueError('Style evidence invalid')), \
             patch('vlm_client.APIConfig.from_env',side_effect=AssertionError('No configuration')):
            report=convert.run_conversion(convert.build_parser().parse_args(
                ['--input',str(self.source),'--output',str(output),'--backend','local']))
        self.assertEqual(report['status'],'failed')
        self.assertEqual(report['api_request_count'],0)
        self.assertEqual(report['local_recognition']['stage'],'visual_style')
        self.assertIn('Style evidence invalid',report['local_recognition']['error'])
        self.assertTrue((output/'ocr_raw.json').is_file())
        self.assertTrue((output/'geometry_raw.json').is_file())
        self.assertFalse((output/'generated.tex').exists())
