"""Offline synthetic route evidence; not an image recognition quality claim."""
import copy
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from semantic_edge_router import route_edges
from semantic_schema import SemanticValidationError
from structured_tikz.validator import validate_scene


def scene():
    return {'canvas': {'width': 400, 'height': 300, 'coordinate_system': 'top-left'},
            'styles': {}, 'nodes': [
                {'id': 'box_A', 'type': 'rectangle', 'bbox': [20, 20, 50, 50]},
                {'id': 'box_B', 'type': 'rectangle', 'bbox': [240, 20, 50, 50]}],
            'edges': [], 'labels': [],
            'metadata': {'review_candidates': {'lines': [], 'text_regions': []}}}


def semantic():
    return {'objects': [], 'texts': [], 'unmatched_objects': [], 'unmatched_texts': [], 'warnings': [],
            'relations': [{'source_id': 'box_A', 'target_id': 'box_B', 'relation': 'arrow',
                           'direction': 'forward', 'confidence': 'high', 'review_required': True}]}


def line(identifier, a, b, grid=False):
    return {'id': identifier, 'points': [a, b], 'classification': 'geometry',
            'source_candidate': {'evidence': {'contained_in': ['box_A'] if grid else [],
                                              'possible_role': 'border_or_grid' if grid else 'unclassified_line'}}}


class EnrichEdgeTests(OfflineTestCase):
    def test_invalid_semantic_id_rejected(self):
        response = semantic(); response['relations'][0]['target_id'] = 'missing'
        with self.assertRaises(SemanticValidationError):
            route_edges(scene(), response)

    def test_straight_evidence_unverified_and_inputs_preserved(self):
        original = scene()
        original['metadata']['review_candidates']['lines'] = [line('line_1', [72, 45], [238, 45])]
        before = copy.deepcopy(original)
        edges, report = route_edges(original, semantic())
        self.assertEqual(original, before)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['points'], [[72, 45], [238, 45]])
        self.assertEqual(edges[0]['metadata']['review_status'], 'unverified')
        self.assertEqual(edges[0]['metadata']['arrow_source'], 'vlm_proposal_not_measured')
        self.assertEqual(report['entries'][0]['anchors']['source']['anchor'], 'east')
        original['edges'] = edges
        validate_scene(original)

    def test_two_segment_measured_intersection(self):
        original = scene(); original['nodes'][1]['bbox'] = [200, 180, 50, 50]
        original['metadata']['review_candidates']['lines'] = [
            line('horizontal', [72, 45], [225, 45]), line('vertical', [225, 46], [225, 178])]
        edges, report = route_edges(original, semantic())
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['points'], [[72, 45], [225, 45], [225, 178]])
        self.assertEqual(report['counts']['recovered_paths'], 1)

    def test_known_text_collision_retains_candidate_without_edge(self):
        original = scene()
        original['metadata']['review_candidates']['lines'] = [line('line_1', [72, 45], [238, 45])]
        edges, report = route_edges(original, semantic(), text_boxes=[[120, 35, 40, 20]])
        self.assertEqual(edges, [])
        self.assertEqual(report['entries'][0]['status'], 'conflict')
        self.assertEqual(report['entries'][0]['matched_line_ids'], ['line_1'])
        self.assertIn('known_text_box:0', report['entries'][0]['conflicts'])

    def test_no_geometry_and_missing_gap_remain_pending(self):
        for lines in ([], [line('partial', [72, 45], [150, 45])]):
            original = scene(); original['metadata']['review_candidates']['lines'] = lines
            edges, report = route_edges(original, semantic())
            self.assertEqual(edges, [])
            self.assertEqual(report['entries'][0]['status'], 'unresolved')

    def test_grid_and_text_stroke_candidates_are_rejected(self):
        original = scene()
        grid = line('grid', [72, 45], [238, 45], grid=True)
        stroke = line('stroke', [72, 45], [238, 45]); stroke['classification'] = 'text_stroke_candidate'
        original['metadata']['review_candidates']['lines'] = [grid, stroke]
        edges, report = route_edges(original, semantic())
        self.assertEqual(edges, [])
        self.assertEqual(len(report['excluded_line_candidates']), 2)

    def test_existing_review_states_are_preserved(self):
        for state in ('verified', 'rejected', 'unverified'):
            original = scene()
            original['edges'] = [{'id': 'human_edge', 'from': 'box_A', 'to': 'box_B', 'metadata': {'review_status': state}}]
            before = copy.deepcopy(original)
            edges, report = route_edges(original, semantic())
            self.assertEqual(edges, [])
            self.assertEqual(report['entries'][0]['status'], 'protected')
            self.assertEqual(original, before)

    def test_semantic_direction_is_not_claimed_as_measured(self):
        original = scene()
        original['metadata']['review_candidates']['lines'] = [line('line_1', [72, 45], [238, 45])]
        for direction, expected in [('forward', 'end'), ('reverse', 'start'), ('bidirectional', 'both')]:
            response = semantic(); response['relations'][0]['direction'] = direction
            edges, _ = route_edges(original, response)
            self.assertEqual(edges[0]['arrow'], expected)
        response = semantic(); response['relations'][0]['direction'] = 'unknown'
        edges, report = route_edges(original, response)
        self.assertEqual(edges, [])
        self.assertEqual(report['entries'][0]['reasons'], ['arrow_direction_unknown'])


if __name__ == '__main__':
    unittest.main()
