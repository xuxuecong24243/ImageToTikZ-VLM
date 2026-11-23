"""Offline synthetic pipeline; mocked AI/compile success is NOT recognition."""
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

from auto_compile import compile_bounded
from auto_pipeline import RequestBudgetError, RequestLedger, build_parser, run_pipeline
from latex_compiler import CompileResult


def synthetic_scene():
    return {'canvas': {'width': 640, 'height': 320, 'coordinate_system': 'top-left'},
            'styles': {'block': {'fill': 'light_blue', 'stroke': 'dark_blue'}},
            'nodes': [{'id': 'matrix', 'type': 'matrix', 'bbox': [70, 130, 120, 80],
                       'grid': {'rows': 2, 'cols': 3}, 'style': 'block'},
                      {'id': 'tensor', 'type': 'tensor3d', 'bbox': [430, 130, 120, 80],
                       'grid': {'rows': 2, 'cols': 3}, 'depth': {'dx': 22, 'dy': -18}, 'style': 'block'}],
            'labels': [{'id': 'title', 'text': '中文离线自检（非 AI 图片识别）', 'x': 320, 'y': 35,
                        'anchor': 'center', 'font_size': 26, 'mode': 'text',
                        'metadata': {'node_id': None, 'bbox': [80, 20, 480, 30]}},
                       {'id': 'formula', 'text': r'W_E\in\mathbb{R}^{50257\times256}',
                        'x': 320, 'y': 270, 'anchor': 'center', 'font_size': 24, 'mode': 'math',
                        'metadata': {'node_id': None, 'bbox': [150, 255, 340, 30]}}],
            'edges': [{'id': 'arrow', 'from': 'matrix', 'to': 'tensor', 'from_anchor': 'east',
                       'to_anchor': 'west', 'routing': 'straight', 'arrow': 'end'}],
            'metadata': {'source': 'handwritten_offline_fixture_not_model_output'}}


class OfflineClient:
    max_tokens = 32768

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.last_diagnostics = {}

    def complete_images(self, images, prompt):
        self.calls.append((list(images), prompt))
        self.last_diagnostics = {'finish_reason': 'stop', 'usage': {'prompt_tokens': 100, 'completion_tokens': 200,
                                                                  'total_tokens': 300}}
        return json.dumps(next(self.responses), ensure_ascii=False)


class AutoPipelineTests(OfflineTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root/'original.png'
        Image.new('RGB', (640, 320), 'white').save(self.image)

    def args(self, *extra):
        return build_parser().parse_args(['--image', str(self.image), '--output-root', str(self.root/'outputs'),
                                          '--geometry-mode', 'off', '--max-local-rounds', '0', *extra])

    def mock_outputs(self, stack, success=True):
        def compile_fake(tex, **kwargs):
            pdf, log, console = tex.with_suffix('.pdf'), tex.with_suffix('.log'), tex.with_suffix('.console.log')
            log.write_text('OFFLINE MOCK compiler', encoding='utf-8')
            console.write_text('OFFLINE MOCK compiler', encoding='utf-8')
            if success:
                pdf.write_bytes(b'%PDF-OFFLINE-MOCK')
            return CompileResult(success, 0 if success else 1, False, pdf if success else None,
                                 log, console, '' if success else 'Synthetic compile error', ['mock-xelatex'])

        def render_fake(pdf, png, **kwargs):
            Image.new('RGB', (640, 320), 'white').save(png)
            return png

        compiler = stack.enter_context(patch('auto_pipeline.compile_bounded', side_effect=compile_fake))
        stack.enter_context(patch('auto_pipeline.render_pdf', side_effect=render_fake))
        stack.enter_context(patch('auto_visual_feedback.compare_visual', return_value={
            'status': 'offline_mock_visual_not_assessed', 'needs_review': True}))
        stack.enter_context(patch('urllib.request.build_opener', side_effect=AssertionError('Real API forbidden')))
        return compiler

    def test_no_authorization_rejects_even_injected_client(self):
        client = OfflineClient([synthetic_scene()])
        with self.assertRaises(ValueError):
            run_pipeline(self.args(), client=client)
        self.assertEqual(client.calls, [])
        self.assertFalse((self.root/'outputs').exists())

    def test_dry_run_needs_no_api_configuration_and_never_compiles(self):
        with patch('auto_pipeline.VLMClient', side_effect=AssertionError), \
             patch('auto_pipeline.compile_bounded', side_effect=AssertionError):
            result = run_pipeline(self.args('--dry-run'))
        self.assertEqual(result['status'], 'prepared_offline')
        self.assertFalse(result['api_called'])
        self.assertTrue((Path(result['output_dir'])/'scene_prompt.txt').exists())

    def test_complete_mock_flow_one_request_preserves_original_and_logs(self):
        before = self.image.read_bytes()
        client = OfflineClient([synthetic_scene()])
        with ExitStack() as stack:
            compiler = self.mock_outputs(stack)
            result = run_pipeline(self.args('--allow-api'), client=client)
        self.assertEqual(result['status'], 'rendered_unverified')
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(compiler.call_count, 1)
        self.assertEqual(client.calls[0][0][0].read_bytes(), before)
        self.assertEqual(self.image.read_bytes(), before)
        self.assertEqual(result['api']['reported_total_usage_tokens'], 300)
        for key in ('final_scene', 'tex', 'pdf', 'png', 'comparison'):
            self.assertTrue(Path(result[key]).is_file())
        self.assertEqual(result['quality'], 'not_verified')

    def test_incomplete_scene_does_not_trigger_second_request_by_default(self):
        scene = synthetic_scene(); scene['nodes'][1]['depth'] = None
        client = OfflineClient([scene])
        with patch('auto_pipeline.compile_bounded', side_effect=AssertionError):
            result = run_pipeline(self.args('--allow-api', '--max-api-requests', '3'), client=client)
        self.assertEqual(result['status'], 'scene_invalid')
        self.assertEqual(len(client.calls), 1)

    def test_explicit_completion_repairs_geometry_then_compiles(self):
        scene = synthetic_scene(); scene['nodes'][1]['depth'] = None
        patch_data = {'operations': [{'op': 'set_node_geometry', 'id': 'tensor',
                                      'value': {'depth': {'dx': 22, 'dy': -18}}, 'reason': 'synthetic completion'}]}
        client = OfflineClient([scene, patch_data])
        with ExitStack() as stack:
            compiler = self.mock_outputs(stack)
            result = run_pipeline(self.args('--allow-api', '--stage-strategy', 'adaptive', '--allow-completion',
                                            '--max-api-requests', '2'), client=client)
        self.assertEqual(result['status'], 'rendered_unverified')
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(client.calls[1][0]), 1)
        self.assertEqual(compiler.call_count, 1)

    def test_visual_feedback_requires_separate_permission_and_two_images(self):
        patch_data = {'operations': [{'op': 'move_label', 'id': 'title', 'dx': 1, 'dy': 0,
                                      'reason': 'synthetic visual patch'}]}
        client = OfflineClient([synthetic_scene(), patch_data])
        with ExitStack() as stack:
            compiler = self.mock_outputs(stack)
            result = run_pipeline(self.args('--allow-api', '--allow-visual-refinement',
                                            '--max-api-requests', '2'), client=client)
        self.assertEqual(result['status'], 'rendered_unverified')
        self.assertEqual(compiler.call_count, 2)
        self.assertEqual([len(call[0]) for call in client.calls], [1, 2])
        self.assertEqual([r['phase'] for r in result['api']['requests']], ['generate', 'visual'])

    def test_visual_permission_does_not_override_request_limit(self):
        client = OfflineClient([synthetic_scene()])
        with ExitStack() as stack:
            self.mock_outputs(stack)
            result = run_pipeline(self.args('--allow-api', '--allow-visual-refinement'), client=client)
        self.assertEqual(len(client.calls), 1)
        self.assertIn('Maximum API', result['feedback_stopped_reason'])

    def test_compile_failure_stops_without_model_or_compile_retry(self):
        client = OfflineClient([synthetic_scene()])
        with ExitStack() as stack:
            compiler = self.mock_outputs(stack, success=False)
            result = run_pipeline(self.args('--allow-api', '--allow-visual-refinement'), client=client)
        self.assertEqual(result['status'], 'compile_failed')
        self.assertEqual(compiler.call_count, 1)
        self.assertEqual(len(client.calls), 1)

    def test_output_and_reported_usage_caps_prevent_next_request(self):
        for limit in ('output', 'usage'):
            client = OfflineClient([synthetic_scene()])
            extra = ['--max-output-budget', '32768'] if limit == 'output' else ['--max-total-usage-tokens', '300']
            args = self.args('--allow-api', '--max-api-requests', '2', '--allow-visual-refinement', *extra)
            directory = self.root/limit; directory.mkdir()
            ledger = RequestLedger(client, args, directory)
            ledger.request([self.image], 'synthetic', 'generate')
            with self.assertRaises(RequestBudgetError):
                ledger.request([self.image, self.image], 'synthetic', 'visual')
            self.assertEqual(len(client.calls), 1)

    def test_missing_usage_forbids_extra_call_and_truncation_is_not_retried(self):
        from vlm_client import APIResponseError
        client = OfflineClient([])
        def incomplete(images, prompt):
            client.last_diagnostics = {'finish_reason': 'length', 'usage': {}}
            raise APIResponseError('length')
        client.complete_images = incomplete
        with patch('auto_pipeline.compile_bounded', side_effect=AssertionError):
            result = run_pipeline(self.args('--allow-api'), client=client)
        self.assertEqual(result['status'], 'pipeline_failed')
        self.assertEqual(result['api']['request_count'], 1)
        self.assertFalse(result['api']['all_usage_reported'])

    def test_offline_scene_cannot_be_mixed_with_paid_permissions(self):
        with self.assertRaises(ValueError):
            run_pipeline(self.args('--offline-scene', 'unused.json', '--allow-api'))

    def test_invalid_limits_fail_before_subprocess(self):
        tex = self.root/'scene.tex'; tex.write_text('synthetic', encoding='utf-8')
        with patch('auto_compile.subprocess.Popen', side_effect=AssertionError):
            for kwargs in ({'timeout': float('nan')}, {'timeout': 0}, {'memory_mb': 128}):
                with self.assertRaises(ValueError):
                    compile_bounded(tex, **kwargs)

    def test_offline_input_preserves_human_source_and_review(self):
        scene = synthetic_scene()
        scene['metadata']['review_status'] = 'verified'
        path = self.root/'human.json'
        path.write_text(json.dumps(scene), encoding='utf-8')
        before = path.read_bytes()
        with ExitStack() as stack:
            self.mock_outputs(stack)
            result = run_pipeline(self.args('--offline-scene', str(path)))
        self.assertEqual(result['recognition'], 'not_performed_offline_supplied_scene')
        saved = json.loads((Path(result['output_dir'])/'initial_scene.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['metadata'], scene['metadata'])
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(result['api_called'])

    def test_rejected_patch_is_saved_and_original_scene_is_not_replaced(self):
        bad = {'operations': [{'op': 'move_label', 'id': 'title', 'dx': 10000, 'dy': 0,
                               'reason': 'synthetic invalid patch'}]}
        client = OfflineClient([synthetic_scene(), bad])
        with ExitStack() as stack:
            self.mock_outputs(stack)
            result = run_pipeline(self.args('--allow-api', '--allow-visual-refinement',
                                            '--max-api-requests', '2'), client=client)
        self.assertEqual(result['status'], 'pipeline_failed')
        directory = Path(result['output_dir'])/'iteration_01'
        self.assertTrue((directory/'model_patch.json').is_file())
        self.assertFalse(json.loads((directory/'patch_rejection.json').read_text())['applied'])
        saved = json.loads((directory/'scene.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['labels'][0]['x'], 320)
        self.assertEqual(result['api']['requests'][-1]['status'], 'patch_rejected')

    def test_api_key_is_redacted_from_diagnostic_artifacts(self):
        from types import SimpleNamespace
        client = OfflineClient([])
        client.config = SimpleNamespace(api_key='unit-test-secret-key')
        def error(images, prompt):
            client.last_diagnostics = {'message': 'unit-test-secret-key', 'usage': {}}
            raise ValueError('unit-test-secret-key')
        client.complete_images = error
        result = run_pipeline(self.args('--allow-api'), client=client)
        for path in Path(result['output_dir']).rglob('*'):
            if path.suffix in ('.json', '.txt'):
                self.assertNotIn('unit-test-secret-key', path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
