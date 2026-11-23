"""Offline unit checks of the single-shot experiment, never live API evidence."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from experiments import run_hybrid_experiment as runner
from latex_compiler import CompileResult
from structured_ir.converter import convert_ir
from structured_tikz.generator import generate_tikz


EMPTY_STRUCTURE = {'groups': [], 'connections': [], 'candidates': [], 'unresolved': []}


class HybridExperimentRunnerTests(OfflineTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='hybrid-runner-unit-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root/'新 实验'
        self.output.mkdir()
        self.args = SimpleNamespace(output=self.output, allow_api=True, api_timeout=4,
                                    max_tokens=256, xelatex='xelatex')
        self.config = SimpleNamespace(model='deepseek-flash', base_url='https://api.deepseek.com',
                                      api_key='synthetic-unit-key-never-real')

    def request_files(self):
        Image.new('RGB', (32, 32), 'white').save(self.output/'original.png')
        runner.save(self.output/'local_baseline.json', {'source_sha256': runner.sha(self.output/'original.png')})
        (self.output/'vlm_prompt.txt').write_text('Unit fixture: structure JSON only.', encoding='utf-8')

    def fake_client(self, visible=None, error=None, capture_path=None):
        visible = json.dumps(EMPTY_STRUCTURE) if visible is None else visible
        fake = Mock()
        fake.last_diagnostics = {'finish_reason': 'stop', 'usage': {'prompt_tokens': 100, 'completion_tokens': 20}}
        if capture_path:
            fake.last_diagnostics['capture'] = {'recoverable_exact': True, 'path': str(capture_path)}
        fake.complete_images.side_effect = error
        fake.complete_images.return_value = visible
        return fake

    def invoke_request(self, fake):
        with patch('vlm_client.APIConfig.from_env', return_value=self.config), \
             patch('vlm_client.VLMClient', return_value=fake), redirect_stdout(io.StringIO()):
            runner.request(self.args)

    def baseline_files(self):
        self.source = self.root/'技术 图片.png'
        Image.new('RGB', (200, 150), 'white').save(self.source)
        baseline = self.root/'existing_local'
        baseline.mkdir()
        ir = {'version': '1.0', 'canvas': {'id': 'canvas', 'width': 200, 'height': 150,
                                         'coordinate_system': 'top-left'},
              'regions': [], 'elements': [
                  {'id': 'old_box', 'type': 'rectangle', 'bbox': [10, 10, 180, 100], 'region_id': None},
                  {'id': 'old_text', 'type': 'text', 'bbox': [20, 20, 150, 25], 'region_id': None,
                   'raw_text': 'raw_Code(1)', 'text_type': 'plain', 'reading_order': 0}],
              'connections': []}
        runner.save(baseline/'ir_generated.json', ir)
        runner.save(baseline/'style_map.json', {})
        runner.save(baseline/'render_policy.json', {'single_ocr_lines_no_wrap': ['old_text']})
        scene, _ = convert_ir(ir)
        runner.apply_baseline_render_policy(scene, baseline)
        runner.save(baseline/'scene_generated.json', scene)
        (baseline/'generated.tikz').write_text(generate_tikz(scene), encoding='utf-8')
        runner.save(baseline/'report.json', {'input': {'sha256': runner.sha(self.source)},
                    'backend': 'local', 'api_request_count': 0})
        runner.save(baseline/'arrow_detection.json', {'arrows': []})
        return baseline, ir, scene

    def test_authorization_gate_precedes_configuration_and_network(self):
        self.args.allow_api = False
        with patch('vlm_client.APIConfig.from_env', side_effect=AssertionError('No config read')) as config, \
             patch('vlm_client.VLMClient', side_effect=AssertionError('No client')) as client:
            with self.assertRaisesRegex(ValueError, '--allow-api'):
                runner.request(self.args)
        config.assert_not_called()
        client.assert_not_called()
        self.assertFalse((self.output/'api_request_started.json').exists())

    def test_persistent_budget_survives_new_python_process(self):
        marker = self.output/'api_request_started.json'
        runner.save(marker, {'attempt': 1, 'maximum_attempts': 1})
        before = marker.read_bytes()
        # The child must exit at the ledger check, before any configuration import/read.
        proc = subprocess.run([sys.executable, '-B', str(Path(runner.__file__)), 'request',
                               '--allow-api', '--output', str(self.output)],
                              capture_output=True, text=True, timeout=20)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('budget has already been consumed', proc.stderr)
        self.assertEqual(marker.read_bytes(), before)
        self.assertFalse((self.output/'api_usage.json').exists())

    def test_first_response_body_and_visible_content_saved_without_rewrite(self):
        self.request_files()
        content = '{ "groups": [], "connections": [], "candidates": [], "unresolved": [] }\n'
        body = (' {"choices":[{"message":{"content":' + json.dumps(content) +
                '},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":20}}\n').encode()
        capture = self.root/'synthetic_provider_response.txt'
        capture.write_bytes(body)
        fake = self.fake_client(content, capture_path=capture)
        self.invoke_request(fake)
        fake.complete_images.assert_called_once()
        self.assertEqual((self.output/'vlm_response_raw.json').read_bytes(), body)
        self.assertEqual((self.output/'vlm_content_raw.txt').read_text(encoding='utf-8'), content)
        self.assertEqual(runner.read(self.output/'vlm_structure_parsed.json'), EMPTY_STRUCTURE)
        usage = runner.read(self.output/'api_usage.json')
        self.assertEqual(usage['request_count'], 1)
        self.assertEqual(usage['automatic_retries'], 0)
        self.assertIsNone(usage['cost']['actual_charge'])

    def test_failure_spends_budget_and_never_retries(self):
        self.request_files()
        fake = self.fake_client(error=TimeoutError('synthetic timeout'))
        self.invoke_request(fake)
        self.assertEqual(runner.read(self.output/'api_usage.json')['request_count'], 1)
        self.assertEqual(runner.read(self.output/'api_usage.json')['status'], 'failed')
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.invoke_request(fake)
        fake.complete_images.assert_called_once()

    def test_malformed_structure_preserves_response_and_no_repair(self):
        self.request_files()
        malformed = '{"groups":['
        fake = self.fake_client(malformed)
        self.invoke_request(fake)
        self.assertEqual((self.output/'vlm_content_raw.txt').read_text(encoding='utf-8'), malformed)
        self.assertFalse((self.output/'vlm_structure_parsed.json').exists())
        error = runner.read(self.output/'structure_parse_error.json')
        self.assertTrue(error['response_was_not_repaired'])
        self.assertTrue(error['no_retry'])
        fake.complete_images.assert_called_once()

    def test_unreviewed_model_stops_before_any_request(self):
        self.request_files()
        self.config.model = 'some-other-model'
        fake = self.fake_client()
        with self.assertRaisesRegex(ValueError, 'reviewed vision capability'):
            self.invoke_request(fake)
        fake.complete_images.assert_not_called()
        self.assertFalse((self.output/'api_request_started.json').exists())

    def test_prepare_replays_original_ir_and_exact_no_wrap_policy(self):
        baseline, ir, scene = self.baseline_files()
        original = {p.name:p.read_bytes() for p in baseline.iterdir()}
        self.args.input, self.args.baseline = self.source, baseline
        with patch.object(runner, 'snapshot', return_value={}), \
             patch('vlm_client.APIConfig.from_env', side_effect=AssertionError('No config')), \
             redirect_stdout(io.StringIO()):
            runner.prepare(self.args)
        self.assertEqual(runner.read(self.output/'baseline/ir_generated.json'), ir)
        self.assertEqual(runner.read(self.output/'baseline/scene_generated.json'), scene)
        self.assertEqual({p.name:p.read_bytes() for p in baseline.iterdir()}, original)
        index = runner.read(self.output/'element_index_map.json')
        self.assertEqual([e['id'] for e in index['elements']], ['old_box', 'old_text'])
        self.assertEqual((self.output/'original.png').read_bytes(), self.source.read_bytes())
        with self.assertRaisesRegex(ValueError, 'overwrit|exists|existing'):
            runner.prepare(self.args)

    def test_prepare_rejects_changed_scene_instead_of_certifying_baseline(self):
        baseline, _, scene = self.baseline_files()
        scene['nodes'][0]['bbox'][0] += 1
        (baseline/'scene_generated.json').write_text(json.dumps(scene), encoding='utf-8')
        self.args.input, self.args.baseline = self.source, baseline
        with patch.object(runner, 'snapshot', return_value={}):
            with self.assertRaisesRegex(ValueError, 'cannot be reproduced'):
                runner.prepare(self.args)
        self.assertFalse((self.output/'local_baseline.json').exists())

    def test_compile_failure_keeps_editable_tex_without_api_or_preview(self):
        baseline, ir, _ = self.baseline_files()
        baseline.rename(self.output/'baseline')
        Image.new('RGB', (200, 150), 'white').save(self.output/'original.png')
        runner.save(self.output/'vlm_structure_parsed.json', EMPTY_STRUCTURE)
        result = CompileResult(False, 1, False, None, self.output/'generated.log',
                               self.output/'generated.console.log', 'synthetic compile failure', [])
        with patch('experiments.hybrid_structure.fuse_structure', return_value=(deepcopy(ir), {'status':'unchanged'})), \
             patch('vlm_client.APIConfig.from_env', side_effect=AssertionError('No config')), \
             patch('vlm_client.VLMClient', side_effect=AssertionError('No API')), \
             patch('auto_compile.compile_bounded', return_value=result) as compile_call, \
             patch('image_compare.render_pdf', side_effect=AssertionError('No PNG after failed PDF')) as preview, \
             redirect_stdout(io.StringIO()):
            runner.render(self.args)
        compile_call.assert_called_once()
        preview.assert_not_called()
        self.assertIn('raw\\_Code', (self.output/'generated.tex').read_text(encoding='utf-8'))
        self.assertNotIn('\\includegraphics', (self.output/'generated.tex').read_text(encoding='utf-8'))
        self.assertEqual(runner.read(self.output/'ir_hybrid.json'), ir)
        report = runner.read(self.output/'render_report.json')
        self.assertEqual(report['api_calls_in_render'], 0)
        self.assertFalse(report['pdf_compiled'])
        self.assertFalse(report['png_generated'])
        self.assertEqual(runner.read(self.output/'compile_report.json')['log_path'], str(result.log_path))

