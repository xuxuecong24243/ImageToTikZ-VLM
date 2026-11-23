"""Integration of first-response retention, root recovery and bounded stopping."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from unittest.mock import patch

from PIL import Image
from auto_pipeline import build_parser,run_pipeline
import test_auto_pipeline as fixtures
from test_auto_pipeline import OfflineClient,synthetic_scene


class RecordedFormatClient(OfflineClient):
    def complete_images(self,images,prompt):
        raw=super().complete_images(images,prompt)
        self.last_diagnostics['requested_response_format']={'type':'json_object'}
        return raw


class SceneContractPipelineTests(OfflineTestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.image=self.root/'input.png'
        Image.new('RGB',(640,320),'white').save(self.image)

    def args(self,*extra):
        return build_parser().parse_args(['--image',str(self.image),'--output-root',str(self.root/'runs'),
            '--geometry-mode','off','--allow-api','--max-local-rounds','0',*extra])

    def test_raw_and_initial_survive_structural_recovery_and_render(self):
        data=synthetic_scene();del data['canvas'];data['type']='json_object'
        before=copy.deepcopy(data);client=RecordedFormatClient([data])
        with ExitStack() as stack:
            compiler=fixtures.AutoPipelineTests.mock_outputs(self,stack)
            result=run_pipeline(self.args(),client=client)
        directory=Path(result['output_dir'])
        self.assertEqual(json.loads((directory/'scene_raw.json').read_text(encoding='utf-8')),before)
        initial=json.loads((directory/'initial_scene.json').read_text(encoding='utf-8'))
        self.assertNotIn('canvas',initial);self.assertEqual(initial['type'],'json_object')
        recovered=json.loads((directory/'scene_after_recovery.json').read_text(encoding='utf-8'))
        self.assertNotIn('type',recovered);self.assertEqual(recovered['canvas']['width'],640)
        self.assertEqual(result['structural_recovery']['status'],'recovered_structure_only')
        self.assertEqual(compiler.call_count,1);self.assertEqual(len(client.calls),1)
        self.assertEqual(result['quality'],'not_verified')

    def test_unknown_root_semantics_stops_before_unusable_api_patch(self):
        for extra in ({'type':'flowchart'}, {'type':'json_object'}):
            with self.subTest(extra=extra):
                data=synthetic_scene();data.update(extra)
                client=OfflineClient([data])  # No trusted format evidence: do not remove the field.
                with patch('auto_pipeline.compile_bounded',side_effect=AssertionError('must not compile')):
                    result=run_pipeline(self.args('--stage-strategy','adaptive','--allow-completion',
                        '--max-api-requests','3'),client=client)
                self.assertEqual(result['status'],'scene_invalid');self.assertEqual(len(client.calls),1)
                self.assertIn('completion_blocked_reason',result)

    def test_missing_collections_and_invalid_canvas_do_not_trigger_api_patch(self):
        for mutation in ('missing_nodes','invalid_canvas','invalid_styles'):
            with self.subTest(mutation=mutation):
                data=synthetic_scene()
                if mutation=='missing_nodes':del data['nodes']
                elif mutation=='invalid_canvas':data['canvas']=None
                else:data['styles']=[]
                client=OfflineClient([data])
                with patch('auto_pipeline.compile_bounded',side_effect=AssertionError):
                    result=run_pipeline(self.args('--stage-strategy','adaptive','--allow-completion',
                        '--max-api-requests','3'),client=client)
                self.assertEqual(result['status'],'scene_invalid');self.assertEqual(len(client.calls),1)

    def test_invalid_json_has_persisted_parse_diagnostic_and_original_response(self):
        client=OfflineClient([])
        client.complete_images=lambda images,prompt:'{"nodes":'
        result=run_pipeline(self.args(),client=client)
        directory=Path(result['output_dir'])
        self.assertEqual(result['parse_diagnostic']['code'],'invalid_json')
        self.assertEqual((directory/'request_01/response.txt').read_text(encoding='utf-8'),'{"nodes":')
        self.assertFalse((directory/'scene_raw.json').exists())


if __name__=='__main__':unittest.main()
