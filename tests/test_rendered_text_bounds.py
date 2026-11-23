"""Synthetic PDF/Scene measurements; no compilation, recognition or network."""
import copy
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from unittest.mock import MagicMock, patch

import fitz

from rendered_text_bounds import (associate_text_lines, infer_pdf_mapping,
                                  pdf_bbox_to_scene, measure_text_bounds)


def scene():
    return {'canvas': {'width': 400, 'height': 240, 'coordinate_system': 'top-left', 'scale': .4},
            'styles': {}, 'nodes': [
                {'id': 'a', 'type': 'rectangle', 'bbox': [20, 40, 50, 30]},
                {'id': 'b', 'type': 'rectangle', 'bbox': [100, 140, 60, 40]},
                {'id': 'c', 'type': 'rectangle', 'bbox': [260, 60, 70, 50]},
            ], 'labels': [], 'edges': []}


def mapping():
    return {'status': 'verified', 'scale': .4 * 72 / 72.27, 'tx': 17.0, 'ty': 6.0}


def pdf_box(box, transform=None):
    transform = transform or mapping()
    x, y, w, h = box
    s, tx, ty = transform['scale'], transform['tx'], transform['ty']
    return [x*s+tx, y*s+ty, (x+w)*s+tx, (y+h)*s+ty]


def drawings(source):
    result = []
    for node in source['nodes']:
        rect = fitz.Rect(pdf_box(node['bbox']))
        result.append({'items': [('re', rect, 1)], 'type': 'fs', 'rect': rect,
                       'fill': (.8, .9, 1), 'color': (0, 0, 0), 'width': .5})
    return result


def label(text='中文标签', **extra):
    result = {'id': 'caption', 'text': text, 'x': 100, 'y': 100,
              'anchor': 'center', 'mode': 'text',
              'metadata': {'bbox': [80, 90, 40, 20]}}
    result.update(extra)
    return result


class RenderedTextBoundsTests(OfflineTestCase):
    def test_pdf_spans_merge_prose_and_script_fragments_into_one_label(self):
        source = scene()
        source['labels'] = [label(mode='rich', segments=[
            {'type': 'text', 'content': '位置 '}, {'type': 'math', 'content': 'E_{pos}'}])]
        page, document = MagicMock(), MagicMock()
        page.rect = fitz.Rect(0, 0, 200, 120)
        page.get_drawings.return_value = drawings(source)
        page.get_text.return_value = {'blocks': [{'lines': [{
            'dir': (1.0, 0.0), 'bbox': pdf_box([70, 94, 60, 12]),
            'spans': [{'text': '位置 '}, {'text': 'E'}, {'text': 'pos'}]}]}]}
        document.page_count, document.needs_pass = 1, False
        document.__enter__.return_value = document
        document.__getitem__.return_value = page
        with patch('fitz.open', return_value=document):
            report = measure_text_bounds(source, 'synthetic.pdf')
        self.assertEqual(report['counts'], {'measured': 1, 'unresolved': 0})
        self.assertEqual(report['labels'][0]['pdf_text'], ['位置 Epos'])

    def test_mapping_recovers_translation_from_multiple_shape_anchors(self):
        source = scene()
        transform = infer_pdf_mapping(source, drawings(source))
        self.assertEqual(transform['status'], 'verified')
        self.assertAlmostEqual(transform['scale'], .4*72/72.27, places=6)
        self.assertAlmostEqual(transform['tx'], 17, places=3)
        self.assertAlmostEqual(transform['ty'], 6, places=3)

    def test_mapping_without_sufficient_geometry_is_unresolved(self):
        source = scene()
        self.assertEqual(infer_pdf_mapping(source, drawings(source)[:1])['status'], 'unresolved')

    def test_pdf_bbox_is_converted_not_treated_as_scene_pixels(self):
        original = [42, 63, 87, 14]
        measured = pdf_bbox_to_scene(pdf_box(original), mapping())
        for actual, expected in zip(measured, original):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertNotEqual(measured, pdf_box(original))

    def test_chinese_label_has_measured_bounds_and_source_is_unchanged(self):
        source = scene()
        source['labels'] = [label()]
        snapshot = copy.deepcopy(source)
        records = associate_text_lines(source, [{'text': '中文标签', 'bbox': pdf_box([80, 94, 40, 12])}], mapping())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['id'], 'caption')
        self.assertEqual(records[0]['status'], 'measured')
        for actual, expected in zip(records[0]['bbox_scene'], [80, 94, 40, 12]):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertEqual(source, snapshot)

    def test_math_characters_subscripts_and_superscripts_match_one_label(self):
        source = scene()
        source['labels'] = [label(r'E_{pos}\in\mathbb{R}^{4\times256}', mode='math')]
        records = associate_text_lines(source, [{'text': 'Epos ∈ R4×256',
                                                 'bbox': pdf_box([70, 94, 60, 12])}], mapping())
        self.assertEqual(records[0]['status'], 'measured')
        self.assertEqual(records[0]['line_count'], 1)

    def test_rich_label_combines_prose_and_math_lines_without_new_objects(self):
        source = scene()
        source['labels'] = [label('位置嵌入\n$E_{pos}$', mode='rich', segments=[
            {'type': 'text', 'content': '位置嵌入\n'}, {'type': 'math', 'content': 'E_{pos}'}])]
        lines = [{'text': '位置嵌入', 'bbox': pdf_box([80, 90, 40, 8])},
                 {'text': 'Epos', 'bbox': pdf_box([88, 102, 24, 8])}]
        records = associate_text_lines(source, lines, mapping())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['status'], 'measured')
        self.assertEqual(records[0]['line_count'], 2)
        for actual, expected in zip(records[0]['bbox_scene'], [80, 90, 40, 20]):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_matching_text_at_wrong_location_is_not_accepted_as_measurement(self):
        source = scene()
        source['labels'] = [label()]
        records = associate_text_lines(source, [{'text': '中文标签', 'bbox': pdf_box([320, 190, 40, 12])}], mapping())
        self.assertEqual(records[0]['status'], 'unresolved')
        self.assertIsNone(records[0].get('bbox_scene'))


if __name__ == '__main__':
    unittest.main()
