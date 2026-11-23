"""Synthetic image evidence only; not AI recognition or semantic accuracy tests."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from unittest.mock import patch

from PIL import Image, ImageDraw
from auto_visual_feedback import compare_visual


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'OpenCV not installed')
class AutoVisualTests(OfflineTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene = {'canvas': {'width': 200, 'height': 100}, 'nodes': [
            {'id': 'n', 'type': 'matrix', 'bbox': [20, 20, 60, 50]}], 'labels': [], 'edges': []}

    def image(self, name, shifted=0, blank=False, size=(200, 100)):
        image = Image.new('RGB', size, 'white')
        if not blank:
            ImageDraw.Draw(image).rectangle((20+shifted, 20, 80+shifted, 70), outline='black', width=2)
        path = self.root/name
        image.save(path)
        return path

    def test_identical_edges_are_not_a_semantic_success(self):
        a, b = self.image('a.png'), self.image('b.png')
        before = copy.deepcopy(self.scene)
        result = compare_visual(a, b, self.scene, self.root/'same')
        self.assertEqual(result['geometry']['original_edge_coverage'], 1)
        self.assertEqual(result['geometry']['rendered_edge_coverage'], 1)
        self.assertEqual(result['semantic_checks']['chinese_text'], 'unverified')
        self.assertTrue(result['needs_review'])
        self.assertTrue(Path(result['visual_diff']).is_file())
        self.assertEqual(self.scene, before)

    def test_shift_produces_local_difference(self):
        result = compare_visual(self.image('a.png'), self.image('b.png', shifted=40), self.scene, self.root/'shift')
        self.assertLess(result['geometry']['original_edge_coverage'], .65)
        self.assertEqual(result['regions'][0]['risk'], 'possible_local_difference')

    def test_blank_images_do_not_score_perfect(self):
        result = compare_visual(self.image('a.png', blank=True), self.image('b.png', blank=True), self.scene, self.root/'blank')
        self.assertIsNone(result['geometry']['original_edge_coverage'])
        self.assertIsNone(result['appearance']['original']['foreground_mean_rgb'])
        self.assertEqual(result['regions'][0]['risk'], 'insufficient_edges')

    def test_letterbox_preserves_aspect_ratio_and_discloses_margin(self):
        result = compare_visual(self.image('a.png'), self.image('b.png', size=(100, 100)), self.scene, self.root/'ratio')
        transform = result['registration']['rendered']
        self.assertEqual(transform['fitted_size'], [100, 100])
        self.assertEqual(transform['offset_px'], [50, 0])
        self.assertIsNone(result['registration']['residual_alignment_error_px'])

    def test_dependency_unavailable_and_source_size_mismatch(self):
        with patch('auto_visual_feedback._dependencies', side_effect=ImportError):
            self.assertEqual(compare_visual('unused', 'unused', {}, self.root)['status'], 'unavailable')
        a, b = self.image('a.png'), self.image('b.png')
        with self.assertRaises(ValueError):
            compare_visual(a, b, {'canvas': {'width': 99, 'height': 100}}, self.root/'bad')

    def test_historical_output_is_not_overwritten(self):
        a, b = self.image('a.png'), self.image('b.png')
        destination = self.root/'existing'
        destination.mkdir()
        path = destination/'visual_diff.png'
        path.write_bytes(b'history')
        with self.assertRaises(FileExistsError):
            compare_visual(a, b, self.scene, destination)
        self.assertEqual(path.read_bytes(), b'history')


if __name__ == '__main__':
    unittest.main()
