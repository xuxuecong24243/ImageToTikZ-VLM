"""Synthetic region candidates and affine mapping; no OCR/API recognition."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from structured_ir.regions import (plan_regions, save_crops, local_to_global_bbox,
                                   local_to_global_points, remap_ir)


class IRRegionTests(OfflineTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'image.png'
        picture = Image.new('RGB', (200, 100))
        picture.putdata([(x % 256, y % 256, (x + y) % 256) for y in range(100) for x in range(200)])
        picture.save(self.image)
        self.geometry = {'image': {'width': 200, 'height': 100,
                                  'sha256': hashlib.sha256(self.image.read_bytes()).hexdigest()},
                         'rectangles': [], 'regions': [], 'lines': []}

    def records(self, boxes):
        self.geometry['rectangles'] = [{'id': 'r' + str(i), 'bbox_px': dict(zip(
            ('x', 'y', 'width', 'height'), box))} for i, box in enumerate(boxes)]

    def test_cache_reused_and_duplicate_sources_retained(self):
        self.records([[10, 10, 50, 40], [10, 10, 50, 40]])
        self.geometry['regions'] = [{'id': 'group', 'bbox_px': [10, 10, 50, 40]}]
        before = deepcopy(self.geometry)
        with patch('geometry_analyzer.analyze_image', side_effect=AssertionError('No CV rerun')):
            plan = plan_regions(self.image, self.geometry)
        self.assertEqual(plan['candidate_count'], 1)
        self.assertEqual(len(plan['candidates'][0]['sources']), 3)
        self.assertEqual(self.geometry, before)
        self.assertTrue(plan['geometry_reused'])
        self.assertEqual(plan['api_calls'], 0)

    def test_clip_partial_boundary_and_reject_outside_candidate(self):
        self.records([[-3, -2, 30, 20], [300, 20, 10, 10], [190.5, 90, 30, 30]])
        plan = plan_regions(self.image, self.geometry)
        self.assertEqual([c['bbox'] for c in plan['candidates']], [[0, 0, 27, 18], [190, 90, 10, 10]])
        self.assertTrue(any(d['code'] == 'invalid_candidate' for d in plan['diagnostics']))

    def test_nested_and_partial_overlaps_are_preserved_without_parent_inference(self):
        self.records([[5, 5, 100, 80], [10, 10, 30, 30], [90, 20, 100, 60]])
        plan = plan_regions(self.image, self.geometry)
        self.assertEqual(plan['candidate_count'], 3)
        self.assertEqual({o['relationship'] for o in plan['overlaps']},
                         {'first_contains_second', 'partial_overlap'})
        self.assertFalse(any('parent_id' in c for c in plan['candidates']))

    def test_uncertain_segmentation_keeps_candidates_and_no_grid(self):
        self.records([[2, 2, 50, 10], [80, 50, 15, 10]])
        plan = plan_regions(self.image, self.geometry)
        self.assertEqual(plan['existing_segmentation'][0]['source'], 'whole_image_fallback')
        self.assertEqual(plan['candidate_count'], 2)
        self.assertEqual(plan['status'], 'geometric_candidates_only')
        self.assertTrue(all(c['confidence'] is None for c in plan['candidates']))

    def test_empty_candidates_reuse_full_image_fallback(self):
        plan = plan_regions(self.image, self.geometry)
        self.assertEqual(plan['candidates'], [])
        self.assertEqual(plan['crop_regions'][0]['crop_bbox'], [0, 0, 200, 100])

    def test_wrong_cache_identity_and_missing_cache_rejected(self):
        self.geometry['image']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'hash/dimensions'):
            plan_regions(self.image, self.geometry)
        with self.assertRaisesRegex(ValueError, 'existing geometry'):
            plan_regions(self.image, None)

    def test_crop_pixels_and_local_recovery_are_exact_and_no_overwrite(self):
        self.records([[20, 30, 45, 25], [150, 80, 50, 20]])
        plan = plan_regions(self.image, self.geometry, overlap_px=8)
        for candidate in plan['candidates']:
            self.assertEqual(local_to_global_bbox(candidate['local_bbox'], candidate['transform']), candidate['bbox'])
        output = self.root / 'crops'
        manifest = save_crops(self.image, plan, output)
        with Image.open(self.image) as source:
            for region in manifest['regions']:
                x, y, w, h = region['crop_bbox']
                with Image.open(output / region['crop_path']) as crop:
                    self.assertEqual(crop.tobytes(), source.crop((x, y, x + w, y + h)).tobytes())
        with self.assertRaises(FileExistsError):
            save_crops(self.image, plan, output)

    def test_scaled_bbox_and_path_mapping_without_rounding(self):
        transform = {'offset': [17.5, 23], 'scale': [2, .5]}
        self.assertEqual(local_to_global_bbox([1.25, 4, 20, 16], transform), [20, 25, 40, 8])
        self.assertEqual(local_to_global_points([[0, 0], [1.25, 4]], transform), [[17.5, 23], [20, 25]])

    def test_invalid_transform_and_geometry_rejected(self):
        for transform in ({'offset': [0, 0], 'scale': [0, 1]},
                          {'offset': [False, 0], 'scale': [1, 1]},
                          {'offset': [0, 0], 'scale': [float('inf'), 1]}, {}):
            with self.subTest(transform=transform), self.assertRaises(ValueError):
                local_to_global_bbox([0, 0, 10, 10], transform)
        with self.assertRaises(ValueError):
            local_to_global_points([[0, float('nan')]], {'offset': [0, 0], 'scale': [1, 1]})

    def test_huge_integer_transform_is_a_validation_error(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            local_to_global_bbox([0, 0, 10, 10], {'offset': [10 ** 500, 0], 'scale': [1, 1]})

    def test_finite_inputs_cannot_produce_infinite_global_coordinates(self):
        transform = {'offset': [0, 0], 'scale': [1e308, 1]}
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            local_to_global_bbox([0, 0, 2, 1], transform)
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            local_to_global_points([[2, 0]], transform)

    def test_empty_wrong_type_transform_maps_are_not_silently_ignored(self):
        for key in ('region_transforms', 'connection_transforms'):
            for value in ([], '', False, 0):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'must be an object'):
                    remap_ir({}, {'offset': [0, 0], 'scale': [1, 1]}, {}, **{key: value})

    def test_cross_region_endpoint_references_labels_and_anchors_preserved(self):
        original = {'canvas': {'width': 50, 'height': 50}, 'regions': [],
                    'elements': [{'id': 'decision', 'type': 'diamond', 'bbox': [1, 2, 10, 10], 'region_id': 'left'},
                                 {'id': 'yes_node', 'bbox': [2, 3, 10, 10], 'region_id': 'right'}],
                    'connections': [{'id': 'branch', 'source_id': 'decision', 'target_id': 'yes_node',
                                     'source_anchor': 'east', 'target_anchor': 'west', 'branch_label': '是'}]}
        snapshot = deepcopy(original)
        canvas = {'width': 200, 'height': 100, 'coordinate_system': 'top-left'}
        result = remap_ir(original, {'offset': [0, 0], 'scale': [1, 1]}, canvas,
                          region_transforms={'left': {'offset': [10, 20], 'scale': [1, 1]},
                                             'right': {'offset': [100, 5], 'scale': [2, 2]}})
        self.assertEqual(result['elements'][0]['bbox'], [11, 22, 10, 10])
        self.assertEqual(result['elements'][1]['bbox'], [104, 11, 20, 20])
        self.assertEqual(result['connections'], original['connections'])
        self.assertEqual(original, snapshot)
        self.assertEqual(result['canvas'], canvas)

    def test_ambiguous_cross_region_path_frame_is_not_guessed(self):
        ir = {'connections': [{'id': 'edge', 'source_id': 'a', 'target_id': 'b',
                               'points': [[0, 0], [5, 5]], 'branch_label': 'yes', 'label_bbox': [1, 1, 2, 2]}]}
        transform = {'offset': [20, 30], 'scale': [1, 1]}
        with self.assertRaisesRegex(ValueError, 'connection transform'):
            remap_ir(ir, transform, {'width': 200, 'height': 100}, region_transforms={'crop': transform})
        mapped = remap_ir(ir, transform, {'width': 200, 'height': 100}, region_transforms={'crop': transform},
                          connection_transforms={'edge': transform})
        self.assertEqual(mapped['connections'][0]['points'], [[20, 30], [25, 35]])
        self.assertEqual(mapped['connections'][0]['label_bbox'], [21, 31, 2, 2])

    def test_nested_region_mapping_inherits_crop_transform_and_validates(self):
        from structured_ir.schema import IR_VERSION
        from structured_ir.validator import validate_ir
        ir = {'version': IR_VERSION,
              'canvas': {'id': 'canvas', 'width': 100, 'height': 100, 'coordinate_system': 'top-left'},
              'regions': [{'id': 'outer', 'type': 'panel', 'bbox': [0, 0, 100, 100], 'parent_id': None},
                          {'id': 'inner', 'type': 'panel', 'bbox': [10, 10, 50, 50], 'parent_id': 'outer'}],
              'elements': [{'id': 'box', 'type': 'rectangle', 'bbox': [20, 20, 10, 10], 'region_id': 'inner'}],
              'connections': []}
        validate_ir(ir)
        canvas = {'id': 'full', 'width': 300, 'height': 200, 'coordinate_system': 'top-left'}
        moved = remap_ir(ir, {'offset': [0, 0], 'scale': [1, 1]}, canvas,
                         region_transforms={'outer': {'offset': [100, 50], 'scale': [1, 1]}})
        self.assertEqual(moved['elements'][0]['bbox'], [120, 70, 10, 10])
        self.assertEqual(moved['regions'][1]['bbox'], [110, 60, 50, 50])
        validate_ir(moved)


if __name__ == '__main__':
    unittest.main()
