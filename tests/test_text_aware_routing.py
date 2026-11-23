"""Measured-box layout tests; no PDF compiler or external service is used."""
import copy
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from text_layout_diagnostics import (diagnose_text_layout, fix_dimension_wrapping,
                                     path_intersects, padded)
from text_aware_edge_router import route_text_aware_edges
from structured_tikz.generator import generate_tikz


def scene():
    return {'canvas': {'width': 400, 'height': 220, 'coordinate_system': 'top-left'},
            'styles': {}, 'nodes': [
                {'id': 'source', 'type': 'rectangle', 'bbox': [10, 80, 40, 40]},
                {'id': 'target', 'type': 'rectangle', 'bbox': [350, 80, 40, 40]},
            ], 'labels': [{'id': 'formula', 'x': 200, 'y': 100, 'mode': 'math',
                          'text': r'W_E\in\mathbb{R}^{50257\times256}'}],
            'edges': [{'id': 'link', 'points': [[60, 100], [340, 100]], 'arrow': 'end',
                       'metadata': {'source_id': 'source', 'target_id': 'target',
                                    'from_anchor': 'east', 'to_anchor': 'west'}}]}


def record(identifier='formula', box=None, **extra):
    result = {'id': identifier, 'collection': 'labels', 'status': 'measured',
              'bbox_scene': box or [160, 90, 80, 20], 'expected_bbox': [180, 90, 40, 20],
              'line_count': 1, 'method': 'synthetic_pdf_fixture'}
    result.update(extra)
    return result


def bounds(*records):
    return {'mapping': {'status': 'verified'}, 'labels': list(records or [record()])}


class TextLayoutDiagnosticsTests(OfflineTestCase):
    def test_no_wrap_changes_only_the_text_paragraph_option(self):
        source = scene()
        source['nodes'].append({'id': 'batch', 'type': 'text', 'bbox': [70, 30, 40, 20],
                                'mode': 'math', 'text': 'B = 8'})
        before = generate_tikz(source)
        source['nodes'][-1]['wrap'] = False
        after = generate_tikz(source)
        before_line = next(line for line in before.splitlines() if '(t-batch)' in line)
        after_line = next(line for line in after.splitlines() if '(t-batch)' in line)
        self.assertIn('text width=16pt', before_line)
        self.assertEqual(before_line.replace(',text width=16pt', ''), after_line)
        self.assertEqual(before.replace(before_line, after_line), after)

    def test_actual_width_exceeding_estimate_and_edge_crossing_are_reported(self):
        source = scene()
        report = diagnose_text_layout(source, bounds())
        self.assertTrue(report['labels'][0]['exceeds_expected_bbox'])
        self.assertEqual(report['summary']['edge_text_pairs'], 1)
        self.assertEqual(report['edge_conflicts'][0]['kind'], 'intersects_text_bounds')

    def test_owned_label_inside_its_node_is_not_a_geometry_error(self):
        source = scene()
        source['nodes'].append({'id': 'container', 'type': 'rounded_rectangle', 'bbox': [140, 70, 120, 60]})
        source['labels'][0]['metadata'] = {'node_id': 'container'}
        report = diagnose_text_layout(source, bounds())
        self.assertEqual(report['geometry_overlaps'], [])

    def test_two_measured_labels_overlapping_are_a_risk_not_auto_rewritten(self):
        source = scene()
        source['labels'].append({'id': 'caption', 'text': '维度', 'x': 220, 'y': 100})
        snapshot = copy.deepcopy(source)
        report = diagnose_text_layout(source, bounds(record(), record('caption', [210, 95, 30, 15])))
        self.assertEqual(report['summary']['text_overlap_pairs'], 1)
        self.assertEqual(report['text_overlaps'][0]['severity'], 'layout_risk')
        self.assertEqual(source, snapshot)

    def test_confirmed_dimension_wrap_changes_only_wrap_control(self):
        source = scene()
        node = {'id': 'batch', 'type': 'text', 'bbox': [70, 30, 40, 20],
                'mode': 'math', 'text': 'B = 8'}
        source['nodes'].append(node)
        snapshot = copy.deepcopy(source)
        measured = bounds(record('batch', [80, 24, 20, 32], collection='nodes', line_count=2))
        result, report = fix_dimension_wrapping(source, measured)
        self.assertEqual(len(report['changes']), 1)
        self.assertEqual(result['nodes'][-1], dict(node, wrap=False))
        self.assertEqual(source, snapshot)
        self.assertEqual(result['labels'], source['labels'])
        self.assertEqual(result['edges'], source['edges'])

    def test_unresolved_or_already_unwrapped_dimensions_are_not_changed(self):
        source = scene()
        source['nodes'].append({'id': 'batch', 'type': 'text', 'bbox': [70, 30, 40, 20],
                                'mode': 'math', 'text': 'B = 8'})
        result, report = fix_dimension_wrapping(source, bounds(record('batch', status='unresolved', bbox_scene=None, line_count=2)))
        self.assertEqual(result, source)
        self.assertEqual(report['changes'], [])
        source['nodes'][-1]['wrap'] = False
        result, report = fix_dimension_wrapping(source, bounds(record('batch', line_count=2)))
        self.assertEqual(result, source)
        self.assertEqual(report['changes'], [])


class TextAwareRoutingTests(OfflineTestCase):
    def test_orthogonal_detour_preserves_semantics_endpoints_and_original_scene(self):
        source = scene()
        snapshot = copy.deepcopy(source)
        result, report = route_text_aware_edges(source, bounds())
        self.assertEqual(len(report['changes']), 1)
        edge, original = result['edges'][0], source['edges'][0]
        self.assertNotEqual(edge['points'], original['points'])
        self.assertEqual(edge['points'][0], original['points'][0])
        self.assertEqual(edge['points'][-1], original['points'][-1])
        self.assertEqual({k:v for k,v in edge.items() if k != 'points'},
                         {k:v for k,v in original.items() if k != 'points'})
        self.assertFalse(path_intersects(edge['points'], [padded([160, 90, 80, 20])]))
        self.assertTrue(all(a[0] == b[0] or a[1] == b[1] for a,b in zip(edge['points'], edge['points'][1:])))
        self.assertEqual(result['nodes'], source['nodes'])
        self.assertEqual(result['labels'], source['labels'])
        self.assertEqual(source, snapshot)

    def test_endpoint_inside_text_cannot_be_routed_without_changing_topology(self):
        source = scene()
        result, report = route_text_aware_edges(source, bounds(record(box=[50, 90, 50, 20])))
        self.assertEqual(result, source)
        self.assertEqual(report['changes'], [])
        self.assertTrue(report['unresolved'])

    def test_unconflicted_or_only_unresolved_text_preserves_route(self):
        source = scene()
        for data in (bounds(record(box=[160, 20, 80, 20])),
                     bounds(record(status='unresolved', bbox_scene=None))):
            with self.subTest(data=data):
                result, report = route_text_aware_edges(source, data)
                self.assertEqual(result, source)
                self.assertEqual(report['changes'], [])


if __name__ == '__main__':
    unittest.main()
