"""Synthetic retained geometry only: no images, models or reference Scene."""
import copy
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from semantic_shape_enricher import enrich_shapes


def _fixture(kind='matrix', bbox=None, lines=None, status='unverified'):
    bbox = bbox or [20, 20, 120, 80]
    node = {'id': 'shape_001', 'type': 'rectangle', 'bbox': bbox,
            'metadata': {'review_status': status, 'source_candidate': {'bbox_px': list(bbox)}}}
    records = [{'id': 'line_%03d' % index, 'points': points, 'classification': 'geometry',
                'source': 'opencv', 'review_status': 'unverified'}
               for index, points in enumerate(lines or [], 1)]
    scene = {'canvas': {'width': 300, 'height': 200, 'coordinate_system': 'top-left'},
             'styles': {}, 'nodes': [node], 'edges': [], 'labels': [],
             'metadata': {'review_status': 'unverified', 'review_candidates': {'lines': records}}}
    semantic = {'objects': [{'candidate_id': 'shape_001', 'semantic_type': kind,
                             'label': None, 'confidence': 'high', 'review_required': True}],
                'texts': [], 'relations': [], 'unmatched_objects': [], 'unmatched_texts': [], 'warnings': []}
    return scene, semantic


def _matrix_lines():
    return ([[[20, y], [140, y]] for y in (20, 60, 100)]+
            [[[x, 20], [x, 100]] for x in (20, 60, 100, 140)])


class ShapeEnrichmentTests(OfflineTestCase):
    def test_matrix_grid_and_measurement_trace_without_input_mutation(self):
        scene, semantic = _fixture(lines=_matrix_lines())
        before = copy.deepcopy(scene)
        proposals, report = enrich_shapes(scene, semantic)
        self.assertEqual(scene, before)
        self.assertEqual(proposals[0]['grid'], {'rows': 2, 'cols': 3})
        self.assertEqual(proposals[0]['id'], 'shape_001')
        meta = proposals[0]['metadata']
        self.assertEqual(meta['source_candidate'], before['nodes'][0]['metadata']['source_candidate'])
        self.assertEqual(meta['geometry_enrichment']['original_bbox'], [20, 20, 120, 80])
        self.assertEqual(meta['geometry_enrichment']['review_status'], 'unverified')
        self.assertEqual(report['entries'][0]['status'], 'completed')

    def test_missing_boundary_blocks_upgrade(self):
        lines = _matrix_lines()
        lines.pop(2)
        proposals, report = enrich_shapes(*_fixture(lines=lines))
        self.assertEqual(proposals, [])
        self.assertIn('grid.rows.end_boundary', report['entries'][0]['missing'])

    def test_single_cell_requires_all_four_measured_boundaries(self):
        lines = [[[20, 20], [140, 20]], [[20, 100], [140, 100]],
                 [[20, 20], [20, 100]], [[140, 20], [140, 100]]]
        proposals, _ = enrich_shapes(*_fixture(lines=lines))
        self.assertEqual(proposals[0]['grid'], {'rows': 1, 'cols': 1})
        proposals, _ = enrich_shapes(*_fixture(lines=lines[:-1]))
        self.assertEqual(proposals, [])

    def test_irregular_grid_is_unresolved(self):
        lines = _matrix_lines()+[[[20, 29], [140, 29]]]
        proposals, report = enrich_shapes(*_fixture(lines=lines))
        self.assertEqual(proposals, [])
        self.assertIn('grid.rows.regular_spacing', report['entries'][0]['missing'])

    def test_missing_tensor_depth_does_not_turn_bounding_box_into_front(self):
        scene, semantic = _fixture(kind='tensor3d', lines=_matrix_lines())
        before = copy.deepcopy(scene)
        proposals, report = enrich_shapes(scene, semantic)
        self.assertEqual(proposals, [])
        self.assertEqual(scene, before)
        self.assertIn('depth', report['entries'][0]['missing'])

    def test_supported_tensor_front_depth_and_grid(self):
        lines = ([[[20, y], [100, y]] for y in (40, 70, 100)]+
                 [[[x, 40], [x, 100]] for x in (20, 60, 100)]+
                 [[[20, 40], [30, 30]], [[100, 40], [110, 30]], [[100, 100], [110, 90]],
                  [[30, 30], [110, 30]], [[110, 30], [110, 90]]])
        scene, semantic = _fixture(kind='tensor3d', bbox=[20, 30, 90, 70], lines=lines)
        proposals, report = enrich_shapes(scene, semantic)
        self.assertEqual(report['entries'][0]['status'], 'completed')
        proposal = proposals[0]
        self.assertEqual(proposal['bbox'], [20, 40, 80, 60])
        self.assertEqual(proposal['depth'], {'dx': 10, 'dy': -10})
        self.assertEqual(proposal['grid'], {'rows': 2, 'cols': 2})
        self.assertEqual(proposal['metadata']['geometry_enrichment']['original_bbox'], [20, 30, 90, 70])
        self.assertEqual(proposal['metadata']['geometry_enrichment']['drawing_bbox_source'], 'derived_from_geometry')

    def test_protected_nodes_unchanged(self):
        for status in ('verified', 'rejected'):
            scene, semantic = _fixture(lines=_matrix_lines(), status=status)
            before = copy.deepcopy(scene)
            proposals, report = enrich_shapes(scene, semantic)
            self.assertEqual(scene, before)
            self.assertEqual(proposals, [])
            self.assertEqual(report['entries'][0]['status'], 'protected')

    def test_text_stroke_lines_are_not_grid_evidence(self):
        scene, semantic = _fixture(lines=_matrix_lines())
        scene['metadata']['review_candidates']['lines'][0]['classification'] = 'text_stroke_candidate'
        proposals, report = enrich_shapes(scene, semantic)
        self.assertEqual(proposals, [])
        self.assertIn('grid.rows.start_boundary', report['entries'][0]['missing'])


if __name__ == '__main__':
    unittest.main()
