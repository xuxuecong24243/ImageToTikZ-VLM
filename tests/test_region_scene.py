"""Contract tests for local envelopes; no image recognition or compilation."""
import copy
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from region_scene import make_region_scene, validate_manifest, validate_region_scene


class RegionSceneTests(OfflineTestCase):
    def setUp(self):
        self.region = {'region_id': 'region_001', 'crop_bbox': [10, 20, 100, 80], 'offset': [10, 20]}
        self.scene = {'canvas': {'width': 100, 'height': 80, 'coordinate_system': 'top-left'},
                      'styles': {}, 'nodes': [{'id': 'a', 'type': 'rectangle', 'bbox': [5, 5, 20, 20]}],
                      'labels': [], 'edges': []}

    def test_envelope_copies_local_scene(self):
        before = copy.deepcopy(self.scene)
        bundle = make_region_scene(self.region, self.scene)
        bundle['scene']['nodes'][0]['bbox'][0] = 9
        self.assertEqual(self.scene, before)

    def test_full_image_canvas_cannot_masquerade_as_crop(self):
        self.scene['canvas']['width'] = 500
        with self.assertRaisesRegex(ValueError, 'crop dimensions'):
            make_region_scene(self.region, self.scene)

    def test_null_target_is_intermediate_only(self):
        pending = [{'id': 'cross', 'source': {'region_id': 'region_001', 'id': 'a'}, 'target': None}]
        bundle = make_region_scene(self.region, self.scene, pending_connections=pending)
        self.assertEqual(bundle['scene']['edges'], [])
        bundle['scene']['edges'].append({'id': 'illegal', 'from': 'a', 'to': None})
        with self.assertRaises(ValueError):
            validate_region_scene(bundle, self.region)

    def test_manifest_offset_cannot_disagree_with_crop(self):
        manifest = {'image': {'width': 200, 'height': 150}, 'regions': [self.region]}
        validate_manifest(manifest)
        manifest['regions'][0]['offset'] = [0, 0]
        with self.assertRaisesRegex(ValueError, 'crop origin'):
            validate_manifest(manifest)

    def test_pending_id_does_not_override_drawable_object(self):
        with self.assertRaisesRegex(ValueError, 'collides'):
            make_region_scene(self.region, self.scene, pending_connections=[{'id': 'a', 'target': None}])

    def test_mock_envelopes_integrate_with_merger_and_existing_generator(self):
        from region_mock_data import build_mock_region_scenes
        from region_scene_merger import merge_region_scenes
        from structured_tikz.generator import generate_tikz
        manifest = {'image': {'width': 900, 'height': 600}, 'regions': [
            {'region_id': 'region_001', 'crop_bbox': [0, 0, 312, 600], 'offset': [0, 0]},
            {'region_id': 'region_002', 'crop_bbox': [288, 0, 324, 600], 'offset': [288, 0]},
            {'region_id': 'region_003', 'crop_bbox': [588, 0, 312, 600], 'offset': [588, 0]}]}
        bundles = build_mock_region_scenes(manifest)
        snapshot = copy.deepcopy(bundles)
        scene, report = merge_region_scenes(manifest, bundles)
        self.assertEqual(bundles, snapshot)
        self.assertEqual(report['counts']['deduplicated'], 2)
        self.assertEqual(report['counts']['resolved_cross_region'], 1)
        self.assertEqual(report['counts']['pending_cross_region'], 1)
        self.assertEqual(sum(label.get('text') == '768' for label in scene['labels']), 3)
        self.assertTrue({'matrix', 'tensor3d'} <= {n['type'] for n in scene['nodes']})
        code = generate_tikz(scene)
        self.assertIn('Mock', code)
        self.assertIn(r'W\in\mathbb{R}^{4\times8}', code)
