"""Small offline contract suite; HTTP and compilation are always fake."""
import copy
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from contextlib import redirect_stderr
from unittest.mock import Mock, patch

from PIL import Image
import region_vlm_middle as middle
from structured_tikz.generator import generate_tikz
from vlm_client import APIConfig, APIError


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class FailedCompilation:
    success: bool
    log_path: str


class MiddleRegionTests(OfflineTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.original = self.root / 'original.png'
        self.crop = self.root / 'region_002.png'
        image = Image.new('RGB', (1536, 1024), 'white')
        image.putpixel((500, 100), (15, 40, 80))
        image.save(self.original)
        image.crop((444, 0, 1050, 1024)).save(self.crop)
        self.region = {'region_id': 'region_002', 'bbox': [456, 0, 582, 1024],
                       'crop_bbox': [444, 0, 606, 1024], 'offset': [444, 0],
                       'crop_path': self.crop.name}
        self.manifest = {'image': {'path': str(self.original), 'width': 1536,
                                  'height': 1024, 'sha256': sha(self.original)},
                         'regions': [copy.deepcopy(self.region)]}
        self.manifest_path = self.root / 'region_manifest.json'
        self.save_manifest()
        self.args = middle.build_parser().parse_args(['--manifest', str(self.manifest_path),
                                                      '--output-root', str(self.root / 'outputs')])
        self.args.baseline_scene = None
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')

    def fake_client(self, result=None, error=None):
        client = Mock()
        client.config = APIConfig('https://api.deepseek.com', 'offline-test-key', 'deepseek-flash')
        client.max_tokens = 32768
        client.last_diagnostics = {}
        client.last_truncated_text = None
        client.complete_images = Mock(return_value=result, side_effect=error)
        return client

    def authorized(self, client):
        self.args.allow_api = True
        environment = {'API_BASE_URL': 'https://api.deepseek.com',
                       'API_KEY': 'offline-test-key', 'MODEL_NAME': 'deepseek-flash'}
        with patch.dict(os.environ, environment, clear=True):
            return middle.run_experiment(self.args, client=client)

    def test_exact_input_dimensions_pixels_and_offset(self):
        original_hash = sha(self.original)
        manifest, crop, info = middle.load_middle_input(self.manifest_path)
        self.assertEqual(crop, self.crop)
        self.assertEqual(info['sha256'], sha(self.crop))
        self.assertEqual((info['width'], info['height']), (606, 1024))
        self.assertEqual(info['offset'], [444, 0])
        self.assertTrue(info['pixel_exact'])
        self.assertEqual(len(manifest['regions']), 1)
        self.assertEqual(sha(self.original), original_hash)

    def test_wrong_core_origin_is_not_accepted_as_crop_offset(self):
        self.manifest['regions'][0]['crop_bbox'][0] = 456
        self.manifest['regions'][0]['offset'][0] = 456
        self.save_manifest()
        with self.assertRaises(ValueError):
            middle.load_middle_input(self.manifest_path)

    def test_changed_crop_pixels_are_rejected(self):
        with Image.open(self.crop) as image:
            image = image.copy()
        image.putpixel((56, 100), (255, 0, 0))
        image.save(self.crop)
        with self.assertRaisesRegex(ValueError, 'pixel-identical'):
            middle.load_middle_input(self.manifest_path)

    def test_changed_original_hash_is_rejected(self):
        self.manifest['image']['sha256'] = '0' * 64
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            middle.load_middle_input(self.manifest_path)

    def test_wrong_canvas_and_invalid_math_are_rejected(self):
        scene = middle.mock_scene()
        scene['canvas']['width'] = 1536
        self.assertFalse(middle.validate_local_scene(scene, self.region)['schema_valid'])
        scene = middle.mock_scene()
        scene['labels'][0]['segments'][1]['content'] = r'\input{secret}'
        self.assertFalse(middle.validate_local_scene(scene, self.region)['schema_valid'])

    def test_strict_json_rejects_duplicates_truncation_and_nonfinite(self):
        for text in ('{"x":1,"x":2}', '{"canvas":', '{"x":NaN}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                middle._strict_json(text)

    def test_compatibility_is_minimal_and_keeps_original_and_none_stroke(self):
        original = middle.mock_scene()
        original['type'] = 'json_object'
        snapshot = copy.deepcopy(original)
        repaired, changes = middle.compatible_scene(original)
        self.assertEqual(original, snapshot)
        self.assertNotIn('type', repaired)
        self.assertEqual(repaired['styles']['plain']['stroke'], 'none')
        self.assertEqual(repaired['labels'], original['labels'])
        self.assertEqual(len(changes), 1)

    def test_rich_without_legacy_text_validates_and_renders_without_mutation(self):
        scene = middle.mock_scene()
        snapshot = copy.deepcopy(scene)
        report = middle.validate_local_scene(scene, self.region)
        self.assertTrue(report['schema_valid'])
        self.assertEqual(report['errors'], [])
        self.assertEqual(scene, snapshot)
        self.assertNotIn('text', scene['labels'][0])
        code = generate_tikz(scene)
        self.assertIn(r'W\in\mathbb{R}^{4\times8}', code)
        prompt = middle.build_middle_prompt()
        self.assertIn('segments', prompt)
        self.assertIn('606 x 1024', prompt)
        self.assertNotIn('seven steps', prompt.lower())

    def test_global_geometry_paths_owners_and_tensor_vectors(self):
        scene = middle.mock_scene()
        snapshot = copy.deepcopy(scene)
        global_scene, report = middle.map_to_global(scene, self.manifest)
        ids = report['id_map']['region_002']
        nodes = {n['id']: n for n in global_scene['nodes']}
        for node in scene['nodes']:
            expected = node['bbox'].copy()
            expected[0] += 444
            actual = nodes[ids[node['id']]]
            self.assertEqual(actual['bbox'], expected)
            if 'depth' in node:
                self.assertEqual(actual['depth'], node['depth'])
        label = global_scene['labels'][0]
        self.assertEqual([label['x'], label['y']], [474, 40])
        self.assertEqual(label['metadata']['bbox'], [474, 40, 450, 35])
        self.assertEqual(label['metadata']['node_id'], ids['matrix'])
        self.assertEqual(label['segments'], scene['labels'][0]['segments'])
        self.assertEqual(global_scene['edges'][0]['points'], [[584, 180], [664, 180]])
        self.assertEqual(global_scene['edges'][0]['metadata']['source_id'], ids['tensor'])
        self.assertEqual(scene, snapshot)
        self.assertTrue(report['schema_valid'])

    def test_mapping_does_not_merge_other_regions(self):
        invalid = copy.deepcopy(self.manifest)
        invalid['regions'].append(copy.deepcopy(self.region))
        with self.assertRaises(ValueError):
            middle.map_to_global(middle.mock_scene(), invalid)

    def test_default_offline_never_calls_api_or_compiler_and_preserves_inputs(self):
        protected = {p: sha(p) for p in [self.original, self.crop, self.manifest_path]}
        client = self.fake_client()
        with patch('region_vlm_middle.compile_bounded') as compiler:
            report = middle.run_experiment(self.args, client=client)
        client.complete_images.assert_not_called()
        compiler.assert_not_called()
        self.assertEqual(report['status'], 'offline_ready_api_not_authorized')
        self.assertEqual(report['api_request_count'], 0)
        self.assertEqual(report['compile_count'], 0)
        self.assertTrue(report['mock_schema_valid'])
        self.assertTrue(report['mock_global_mapping_valid'])
        self.assertFalse(report['api_config_ready'])
        self.assertEqual(protected, {p: sha(p) for p in protected})
        output = Path(report['output_dir'])
        _, saved_crop, saved_info = middle.load_middle_input(output / 'region_manifest.json')
        self.assertEqual(saved_crop, output / 'region_input.png')
        self.assertEqual(saved_info['sha256'], sha(self.crop))
        self.assertTrue((output / 'mock_scene.tex').is_file())
        self.assertFalse((output / 'response_raw.txt').exists())
        self.assertFalse((output / 'local_scene.json').exists())
        self.assertFalse((output / 'scene.pdf').exists())

    def test_cli_requires_explicit_authorization_and_fixed_budget(self):
        parser = middle.build_parser()
        self.assertFalse(parser.parse_args([]).allow_api)
        self.assertTrue(parser.parse_args(['--allow-api']).allow_api)
        for options in (['--max-api-requests', '2'], ['--max-tokens', '65536']):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(options)

    def test_authorized_failure_has_one_attempt_and_no_retry_or_compile(self):
        client = self.fake_client(error=APIError('API returned HTTP 429. offline-test-key'))
        with patch('region_vlm_middle.compile_bounded') as compiler:
            report = self.authorized(client)
        self.assertEqual(client.complete_images.call_count, 1)
        compiler.assert_not_called()
        self.assertEqual(report['api_request_count'], 1)
        self.assertEqual(report['http_status'], 429)
        self.assertEqual(report['compile_count'], 0)
        for path in Path(report['output_dir']).rglob('*'):
            if path.suffix in {'.json', '.txt'}:
                self.assertNotIn('offline-test-key', path.read_text(encoding='utf-8'))

    def test_invalid_response_is_saved_without_retry(self):
        client = self.fake_client(result='{"canvas":')
        with patch('region_vlm_middle.compile_bounded') as compiler:
            report = self.authorized(client)
        self.assertEqual(client.complete_images.call_count, 1)
        compiler.assert_not_called()
        self.assertEqual(report['api_request_count'], 1)
        self.assertEqual((Path(report['output_dir']) / 'response_raw.txt').read_text(encoding='utf-8'), '{"canvas":')
        self.assertTrue((Path(report['output_dir']) / 'validation.json').is_file())

    def test_fake_valid_response_compiles_only_local_once_and_stops_on_failure(self):
        client = self.fake_client(result=json.dumps(middle.mock_scene(), ensure_ascii=False))
        log = self.root / 'fake_compile.log'
        log.write_text('Synthetic compiler failure; no XeLaTeX was executed.', encoding='utf-8')
        with patch('region_vlm_middle.compile_bounded', return_value=FailedCompilation(False, str(log))) as compiler:
            report = self.authorized(client)
        self.assertEqual(client.complete_images.call_count, 1)
        self.assertEqual(compiler.call_count, 1)
        self.assertEqual(report['status'], 'compile_failed_no_retry')
        self.assertEqual(report['compile_count'], 1)
        output = Path(report['output_dir'])
        local = json.loads((output / 'local_scene.json').read_text(encoding='utf-8'))
        global_scene = json.loads((output / 'global_scene.json').read_text(encoding='utf-8'))
        self.assertEqual(local['canvas']['width'], 606)
        self.assertEqual(global_scene['canvas']['width'], 1536)
        self.assertEqual(compiler.call_args.args[0], output / 'scene.tex')
        request_images = client.complete_images.call_args.args[0]
        self.assertEqual(request_images, [output / 'region_input.png'])
        self.assertEqual(sha(request_images[0]), sha(self.crop))


if __name__ == '__main__':
    unittest.main()
