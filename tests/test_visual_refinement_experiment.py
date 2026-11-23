"""Offline entry/gate tests. Transport is a Mock; no compiler or model runs."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from unittest.mock import Mock, patch

from PIL import Image
from visual_refinement_experiment import build_parser, run_experiment


NO_CHANGE = {'status': 'no_change', 'observations': [], 'operations': []}


class VisualRefinementExperimentTests(OfflineTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'scene.json'
        self.reference, self.rendered = self.root/'reference.png', self.root/'rendered.png'
        for image in (self.reference, self.rendered):
            Image.new('RGB', (200, 100), 'white').save(image)
        scene = {'canvas': {'width': 200, 'height': 100, 'coordinate_system': 'top-left'},
                 'styles': {}, 'nodes': [{'id': 'box', 'type': 'rectangle', 'bbox': [140, 50, 20, 20]}],
                 'labels': [{'id': 'caption', 'x': 80, 'y': 40, 'text': '中文',
                             'metadata': {'bbox': [70, 30, 20, 20]}}], 'edges': []}
        self.source.write_text(json.dumps(scene, ensure_ascii=False), encoding='utf-8')
        self.original = self.source.read_bytes()
        self.feedback = self.root/'feedback.json'
        self.feedback.write_text(json.dumps(NO_CHANGE), encoding='utf-8')

    def args(self, *extra):
        return build_parser().parse_args(['--scene', str(self.source), '--reference', str(self.reference),
            '--rendered', str(self.rendered), '--output-root', str(self.root/'outputs'), *extra])

    def client(self):
        client = Mock()
        client.config = SimpleNamespace(api_key='offline-secret')
        client.max_tokens = 8192
        client.last_diagnostics = {'finish_reason': 'stop', 'usage': {'total_tokens': 10}}
        client.last_truncated_text = None
        client.complete_images.return_value = json.dumps(NO_CHANGE)
        return client

    def test_both_api_permissions_are_required_before_configuration_or_calls(self):
        for flags in ([], ['--allow-api'], ['--allow-visual-refinement']):
            with self.subTest(flags=flags), patch('visual_refinement_experiment.APIConfig.from_env') as config:
                with self.assertRaises(ValueError):
                    run_experiment(self.args(*flags))
                config.assert_not_called()

    def test_count_and_output_budget_exhaustion_make_no_request(self):
        for flags in (['--max-api-requests', '0'], ['--max-output-budget', '1']):
            client = self.client()
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                run_experiment(self.args('--allow-api', '--allow-visual-refinement', *flags), client=client)
            client.complete_images.assert_not_called()

    def test_mock_no_change_saves_response_without_api_compile_or_scene_after(self):
        with patch('visual_refinement_experiment.VLMClient') as transport, \
                patch('visual_refinement_experiment.compile_bounded') as compiler:
            report = run_experiment(self.args('--mock-feedback', str(self.feedback)))
        self.assertEqual(report['status'], 'no_change')
        self.assertFalse(report['api_called'])
        self.assertEqual(report['mode'], 'mock_offline')
        transport.assert_not_called(); compiler.assert_not_called()
        output = Path(report['output_dir'])
        self.assertTrue((output/'visual_feedback_raw.txt').is_file())
        self.assertTrue((output/'visual_feedback_parsed.json').is_file())
        self.assertTrue((output/'run_report.json').is_file())
        self.assertFalse((output/'scene_after.json').exists())
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_mock_mode_cannot_be_combined_with_api_authorization(self):
        with self.assertRaises(ValueError):
            run_experiment(self.args('--mock-feedback', str(self.feedback), '--allow-api'))

    def test_invalid_patch_is_saved_and_rejected_without_compilation(self):
        data = {'status': 'patch_proposed', 'observations': [{'object_id': 'caption', 'issue': 'mock',
            'reason': 'offline invalid fixture', 'confidence': .5}],
            'operations': [{'op': 'replace_scene', 'id': 'caption', 'value': {}}]}
        self.feedback.write_text(json.dumps(data), encoding='utf-8')
        with patch('visual_refinement_experiment.compile_bounded') as compiler:
            report = run_experiment(self.args('--mock-feedback', str(self.feedback)))
        self.assertEqual(report['status'], 'patch_rejected')
        compiler.assert_not_called()
        output = Path(report['output_dir'])
        validation = json.loads((output/'patch_validation.json').read_text(encoding='utf-8'))
        self.assertFalse(validation['accepted'])
        self.assertFalse((output/'scene_after.json').exists())
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_mock_transport_receives_original_then_rendered_once(self):
        client = self.client()
        with patch('visual_refinement_experiment.compile_bounded') as compiler:
            report = run_experiment(self.args('--allow-api', '--allow-visual-refinement'), client=client)
        self.assertEqual(report['status'], 'no_change')
        self.assertEqual(client.complete_images.call_count, 1)
        images, prompt = client.complete_images.call_args.args
        self.assertEqual(images, [self.reference.resolve(), self.rendered.resolve()])
        self.assertIn('ORIGINAL REFERENCE', prompt)
        self.assertIn('CURRENT TikZ', prompt)
        self.assertEqual(report['api']['request_count'], 1)
        self.assertEqual(report['api']['reported_total_usage_tokens'], 10)
        compiler.assert_not_called()

    def test_failed_mock_transport_has_no_retry_and_redacts_secret(self):
        client = self.client()
        client.complete_images.side_effect = RuntimeError('offline-secret synthetic error')
        report = run_experiment(self.args('--allow-api', '--allow-visual-refinement'), client=client)
        self.assertEqual(report['status'], 'request_failed')
        self.assertEqual(client.complete_images.call_count, 1)
        self.assertEqual(report['api']['request_count'], 1)
        for path in Path(report['output_dir']).rglob('*.json'):
            self.assertNotIn('offline-secret', path.read_text(encoding='utf-8'))

    def test_dry_run_prepares_prompt_without_api_or_compilation(self):
        with patch('visual_refinement_experiment.VLMClient') as transport, \
                patch('visual_refinement_experiment.compile_bounded') as compiler:
            report = run_experiment(self.args('--dry-run'))
        self.assertEqual(report['status'], 'prepared_offline')
        self.assertTrue((Path(report['output_dir'])/'visual_feedback_prompt.txt').is_file())
        transport.assert_not_called(); compiler.assert_not_called()


if __name__ == '__main__':
    unittest.main()
