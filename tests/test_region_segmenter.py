"""Synthetic region geometry and pixel-copy contracts; no API or real CV run."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from unittest.mock import patch

from PIL import Image

from region_segmenter import segment_regions, save_region_crops


class RegionSegmenterTests(OfflineTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / 'source.png'
        image = Image.new('RGB', (300, 200))
        image.putdata([(x % 256, y % 256, (x + y) % 256) for y in range(200) for x in range(300)])
        image.save(self.image)
        self.original = self.image.read_bytes()

    def geometry(self, boxes):
        return {'image': {'width': 300, 'height': 200,
                          'sha256': hashlib.sha256(self.original).hexdigest()},
                'rectangles': [{'id': 'r_%d' % index,
                                'bbox_px': dict(zip(('x', 'y', 'width', 'height'), box))}
                               for index, box in enumerate(boxes)]}

    def columns(self):
        return self.geometry([[5, 20, 85, 175], [105, 20, 85, 175],
                              [205, 20, 90, 90], [205, 125, 90, 70],
                              [20, 50, 55, 80]])

    def test_detects_data_driven_columns_and_suppresses_inner_detail(self):
        geometry = self.columns()
        snapshot = deepcopy(geometry)
        regions = segment_regions(self.image, geometry=geometry)
        self.assertEqual(len(regions), 3)
        self.assertEqual([r['bbox'] for r in regions],
                         [[0, 0, 97, 200], [97, 0, 100, 200], [197, 0, 103, 200]])
        self.assertTrue(all(r['source'] == 'automatic_geometry_columns' for r in regions))
        self.assertIn('r_4', regions[0]['evidence']['suppressed_nested_candidate_ids'])
        self.assertEqual(geometry, snapshot)

    def test_region_count_is_not_fixed_at_three(self):
        geometry = self.geometry([[8, 20, 125, 175], [160, 20, 130, 175]])
        regions = segment_regions(self.image, geometry=geometry, overlap_px=0)
        self.assertEqual(len(regions), 2)
        self.assertEqual([r['bbox'] for r in regions], [[0, 0, 146, 200], [146, 0, 154, 200]])
        self.assertTrue(all(not r['has_overlap'] for r in regions))

    def test_crop_offsets_pixels_and_source_are_preserved(self):
        regions = segment_regions(self.image, geometry=self.columns(), overlap_px=12)
        manifest = save_region_crops(self.image, regions, self.root / 'output')
        self.assertTrue(manifest['automatic'])
        self.assertEqual(manifest['image']['sha256'], hashlib.sha256(self.original).hexdigest())
        with Image.open(self.image) as source:
            for record in manifest['regions']:
                x, y, width, height = record['crop_bbox']
                self.assertEqual(record['offset'], [x, y])
                self.assertTrue(record['has_overlap'])
                with Image.open(self.root / 'output' / record['crop_path']) as crop:
                    self.assertEqual(crop.size, (width, height))
                    self.assertEqual(crop.tobytes(), source.crop((x, y, x + width, y + height)).tobytes())
        self.assertEqual(self.image.read_bytes(), self.original)

    def test_padding_is_clamped_at_image_boundary(self):
        regions = segment_regions(self.image, geometry=self.columns(), overlap_px=30)
        self.assertEqual(regions[0]['crop_bbox'][:2], [0, 0])
        for record in regions:
            x, y, width, height = record['crop_bbox']
            self.assertGreaterEqual(x, 0); self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + width, 300); self.assertLessEqual(y + height, 200)
        self.assertEqual(regions[-1]['crop_bbox'][0] + regions[-1]['crop_bbox'][2], 300)

    def test_geometry_hash_or_size_mismatch_is_rejected(self):
        for field, bad in [('sha256', '0' * 64), ('width', 299)]:
            geometry = self.columns(); geometry['image'][field] = bad
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'hash/dimensions'):
                segment_regions(self.image, geometry=geometry)

    def test_insufficient_evidence_preserves_explicit_whole_image_fallback(self):
        regions = segment_regions(self.image, geometry=self.geometry([]))
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]['crop_bbox'], [0, 0, 300, 200])
        self.assertEqual(regions[0]['source'], 'whole_image_fallback')
        manifest = save_region_crops(self.image, regions, self.root / 'fallback')
        self.assertFalse(manifest['automatic'])
        self.assertEqual(manifest['segmentation_status'], 'insufficient_evidence_whole_image')

    def test_manual_configuration_is_explicit_and_bounded(self):
        regions = segment_regions(self.image, region_config=[[0, 0, 150, 200], [150, 0, 150, 200]])
        manifest = save_region_crops(self.image, regions, self.root / 'manual')
        self.assertFalse(manifest['automatic'])
        self.assertTrue(manifest['manual_configuration_used'])
        with self.assertRaises(ValueError):
            segment_regions(self.image, region_config=[[290, 0, 20, 200]])

    def test_existing_geometry_analyzer_is_reused_only_when_needed(self):
        with patch('geometry_analyzer.analyze_image', return_value=self.columns()) as analyze:
            self.assertEqual(len(segment_regions(self.image)), 3)
            analyze.assert_called_once_with(str(self.image.resolve()))
        with patch('geometry_analyzer.analyze_image', side_effect=AssertionError('Unexpected CV run')):
            segment_regions(self.image, geometry=self.columns())
            segment_regions(self.image, region_config=[[0, 0, 300, 200]])

    def test_output_is_not_overwritten_and_stale_regions_are_rejected(self):
        regions = segment_regions(self.image, geometry=self.columns())
        output = self.root / 'once'
        save_region_crops(self.image, regions, output)
        before = (output / 'regions/region_001.png').read_bytes()
        with self.assertRaises(FileExistsError):
            save_region_crops(self.image, regions, output)
        self.assertEqual((output / 'regions/region_001.png').read_bytes(), before)
        regions[0]['evidence']['image_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'different source'):
            save_region_crops(self.image, regions, self.root / 'stale')


if __name__ == '__main__':
    unittest.main()
