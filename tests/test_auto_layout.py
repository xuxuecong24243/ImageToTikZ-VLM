"""Synthetic offline rules; passing does not establish diagram recognition quality."""
import copy
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from auto_scene_validator import diagnose_scene
from auto_layout_refiner import refine_scene
from structured_tikz.validator import validate_scene


def scene():
    return {'canvas': {'width': 300, 'height': 200, 'coordinate_system': 'top-left'},
            'styles': {}, 'nodes': [
                {'id': 'A', 'type': 'rectangle', 'bbox': [20, 60, 40, 40]},
                {'id': 'B', 'type': 'rectangle', 'bbox': [240, 60, 40, 40]}],
            'labels': [], 'edges': [], 'metadata': {}}


def pending():
    return {'id': 'link', 'from': 'A', 'to': 'B', 'from_anchor': 'east', 'to_anchor': 'west',
            'arrow': 'end', 'routing': 'unknown', 'metadata': {'source': 'vlm_proposal_not_measured'}}


class AutoLayoutTests(OfflineTestCase):
    def test_diagnostics_collect_missing_fields_without_filling(self):
        data = scene()
        data['nodes'][0].update(type='tensor3d', grid=None, depth=None)
        data['labels'] = [{'id': 'formula', 'text': r'\input{private}', 'mode': 'math', 'x': None, 'y': 10}]
        before = copy.deepcopy(data)
        report = diagnose_scene(data)
        self.assertEqual(data, before)
        codes = {e['code'] for e in report['errors']}
        self.assertTrue({'schema_invalid', 'missing_or_invalid_depth', 'missing_or_invalid_grid',
                         'label_position_missing', 'invalid_math'}.issubset(codes))

    def test_invalid_nested_types_do_not_crash(self):
        data = scene()
        data['nodes'][0]['type'] = []
        data['edges'] = [{'id': 'bad', 'from': {}, 'to': []}]
        data['labels'] = [{'id': 'label', 'text': 'abc', 'x': 5, 'y': 5, 'anchor': None}]
        self.assertFalse(diagnose_scene(data)['schema_valid'])
        self.assertTrue(diagnose_scene(None)['errors'])

    def test_valid_schema_still_does_not_prove_quality(self):
        report = diagnose_scene(scene(), image_size=(300, 200))
        self.assertTrue(report['schema_valid'])
        self.assertFalse(report['quality_verified'])
        self.assertTrue(report['unknowns'])
        self.assertEqual(diagnose_scene(scene(), (400, 200))['errors'][0]['code'], 'canvas_image_mismatch')

    def test_references_tensor_projection_and_pending_parameters(self):
        data = scene()
        data['nodes'][1].update(type='tensor3d', grid={'rows': 2, 'cols': 3}, depth={'dx': 30, 'dy': -20})
        data['edges'] = [{'id': 'bad', 'from': 'A', 'to': 'missing'}]
        data['metadata']['pending_parameters'] = [{'id': 'A', 'field': 'grid'}]
        codes = {e['code'] for e in diagnose_scene(data)['errors']}
        self.assertTrue({'node_outside_canvas', 'unknown_edge_node', 'unresolved_pending_parameters'}.issubset(codes))

    def test_overlap_and_path_crossing_are_risks(self):
        data = scene()
        data['nodes'].append({'id': 'blocker', 'type': 'rectangle', 'bbox': [120, 60, 40, 40]})
        data['edges'] = [dict(pending(), routing='straight')]
        data['labels'] = [{'id': 'text', 'text': 'mock', 'x': 140, 'y': 80, 'font_size': 20}]
        risks = diagnose_scene(data)['risks']
        self.assertTrue(any(r['code'] == 'edge_crosses_unrelated_node' for r in risks))
        self.assertTrue(any(r.get('evidence') == 'heuristic_extent_not_font_metrics' for r in risks))

    def test_small_boundary_clamp_is_audited_but_large_moves_and_tensors_are_not(self):
        data = scene()
        data['nodes'][0]['bbox'][0] = -3
        data['nodes'][1]['bbox'][0] = 290
        before = copy.deepcopy(data)
        after, changes = refine_scene(data)
        self.assertEqual(data, before)
        self.assertEqual(after['nodes'][0]['bbox'][0], 0)
        self.assertEqual(after['nodes'][1], before['nodes'][1])
        self.assertEqual(len(changes), 1)
        self.assertIn('before', changes[0])
        data['nodes'][0].update(type='tensor3d', grid={'rows': 2, 'cols': 2}, depth={'dx': 10, 'dy': -10})
        self.assertEqual(refine_scene(data)[0]['nodes'][0], data['nodes'][0])

    def test_protected_objects_unchanged(self):
        for state in ('verified', 'rejected'):
            data = scene()
            data['nodes'][0].update(bbox=[-3, 60, 40, 40], metadata={'review_status': state})
            self.assertEqual(refine_scene(data)[0]['nodes'][0], data['nodes'][0])
            data['metadata']['review_status'] = state
            self.assertEqual(refine_scene(data), (data, []))

    def test_small_label_move_requires_supplied_bbox(self):
        data = scene()
        data['labels'] = [{'id': 'caption', 'text': 'mock', 'x': 45, 'y': 57,
                           'metadata': {'bbox': [30, 50, 30, 14], 'source': 'vlm_proposal_not_measured'}}]
        result, changes = refine_scene(data)
        self.assertLess(result['labels'][0]['y'], 57)
        self.assertEqual(changes[0]['reason'], 'small_label_shift_from_supplied_bbox_conflict')
        del data['labels'][0]['metadata']['bbox']
        self.assertEqual(refine_scene(data)[0]['labels'], data['labels'])

    def test_owned_or_fully_contained_text_is_not_moved(self):
        data = scene()
        data['labels'] = [{'id': 'caption', 'text': 'mock', 'x': 40, 'y': 80,
                           'metadata': {'bbox': [25, 70, 30, 14], 'node_id': 'A'}}]
        self.assertEqual(refine_scene(data)[0]['labels'], data['labels'])
        del data['labels'][0]['metadata']['node_id']
        self.assertEqual(refine_scene(data)[0]['labels'], data['labels'])

    def test_pending_topology_routes_around_obstacle_with_provenance(self):
        data = scene()
        data['nodes'].append({'id': 'blocker', 'type': 'matrix', 'bbox': [120, 60, 40, 40],
                              'grid': {'rows': 2, 'cols': 2}})
        data['metadata']['pending_edges'] = [pending()]
        before = copy.deepcopy(data)
        result, changes = refine_scene(data)
        self.assertEqual(data, before)
        self.assertEqual(result['metadata']['pending_edges'], [])
        self.assertEqual(len(result['edges']), 1)
        edge = result['edges'][0]
        self.assertEqual(edge['points'][0], [60, 80])
        self.assertEqual(edge['points'][-1], [240, 80])
        self.assertTrue(any(p[1] != 80 for p in edge['points']))
        self.assertEqual(edge['metadata']['source'], 'deterministic_route_not_measured')
        self.assertEqual(edge['metadata']['review_status'], 'unverified')
        validate_scene(result)
        self.assertFalse(any(r['code'] == 'edge_crosses_unrelated_node' for r in diagnose_scene(result)['risks']))
        self.assertEqual(changes[0]['reason'], 'route_pending_semantic_relationship')

    def test_unroutable_or_invalid_relation_remains_pending(self):
        for alteration in ({'to': 'missing'}, {'arrow': 'unknown'}, {'from_anchor': None}):
            data = scene()
            item = dict(pending(), **alteration)
            data['metadata']['pending_edges'] = [item]
            result, _ = refine_scene(data)
            self.assertEqual(result['edges'], [])
            self.assertEqual(result['metadata']['pending_edges'], [item])

    def test_valid_explicit_path_preserved_but_colliding_reference_edge_rerouted(self):
        data = scene()
        data['edges'] = [{'id': 'existing', 'points': [[65, 80], [235, 80]], 'arrow': 'end'}]
        self.assertEqual(refine_scene(data)[0]['edges'], data['edges'])
        data['edges'] = [dict(pending(), routing='straight')]
        data['nodes'].append({'id': 'blocker', 'type': 'rectangle', 'bbox': [120, 60, 40, 40]})
        result, changes = refine_scene(data)
        self.assertIn('points', result['edges'][0])
        self.assertEqual(result['edges'][0]['metadata']['source_id'], 'A')
        self.assertEqual(changes[0]['reason'], 'reroute_reference_edge_around_known_boxes')


if __name__ == '__main__':
    unittest.main()
