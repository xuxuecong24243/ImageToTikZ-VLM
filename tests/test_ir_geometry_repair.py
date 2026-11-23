"""Synthetic evidence-gated repairs; no page-specific data or API requests."""
from copy import deepcopy
import unittest

try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from structured_ir.geometry_repair import ir_fingerprint, repair_geometry
from structured_ir.validator import inspect_ir
from structured_ir.converter import convert_ir
from structured_tikz.generator import generate_tikz


IMAGE_SHA = 'a' * 64


def sample_ir():
    return {
        'version': '1.0',
        'canvas': {'id': 'canvas', 'width': 400, 'height': 300,
                   'coordinate_system': 'top-left'},
        'regions': [
            {'id': 'left', 'type': 'panel', 'bbox': [10, 10, 160, 260], 'parent_id': None},
            {'id': 'right', 'type': 'panel', 'bbox': [220, 10, 170, 260], 'parent_id': None},
        ],
        'elements': [
            {'id': 'caption', 'type': 'multiline_text', 'bbox': [30, 40, 110, 35],
             'region_id': 'left', 'raw_text': 'x = data[0]\n下一行',
             'text_type': 'plain', 'reading_order': 0},
            {'id': 'source', 'type': 'rectangle', 'bbox': [30, 100, 70, 50],
             'region_id': 'left'},
            {'id': 'target', 'type': 'rounded_rectangle', 'bbox': [250, 100, 90, 50],
             'region_id': 'right', 'raw_text': '完成', 'text_type': 'plain', 'reading_order': 1},
        ],
        'connections': [
            {'id': 'flow', 'source_id': 'source', 'target_id': 'target',
             'direction': 'forward', 'source_anchor': 'east', 'target_anchor': 'west', 'kind': 'flow'},
        ],
        'provenance': {'kind': 'synthetic_unit_fixture'},
    }


def observation(ir, *, element_index=0, **updates):
    element = ir['elements'][element_index]
    value = {'element_id': element['id'], 'expected_bbox': deepcopy(element['bbox']),
             'expected_region_id': element['region_id'], 'confirmed': True,
             'basis': 'Reviewed synthetic source measurement, not inferred from proximity.',
             'action': 'reassign', 'ownership': 'unique_region'}
    value.update(updates)
    return value


def evidence(ir, *observations):
    return {'image_sha256': IMAGE_SHA, 'ir_sha256': ir_fingerprint(ir),
            'coordinate_system': 'top-left', 'canvas_size': [400, 300],
            'observations': list(observations)}


class IRGeometryRepairTests(OfflineTestCase):
    def run_repair(self, ir, proof=None):
        return repair_geometry(ir, proof, image_sha256=IMAGE_SHA)

    def wrong_owner(self):
        ir = sample_ir()
        ir['elements'][0]['bbox'] = [250, 40, 100, 35]
        return ir

    def assert_rejected_unchanged(self, ir, proof, message=None):
        before = deepcopy(ir)
        repaired, report = self.run_repair(ir, proof)
        self.assertEqual(ir, before)
        self.assertEqual(repaired, before)
        self.assertEqual(report['changes'], [])
        self.assertEqual(report['status'], 'unresolved')
        self.assertTrue(report['unresolved'])
        if message:
            self.assertTrue(any(message in d['message'] for d in report['unresolved']), report)
        return report

    def test_correct_parent_is_unchanged_without_evidence(self):
        ir = sample_ir()
        self.assertTrue(inspect_ir(ir)['valid'])
        repaired, report = self.run_repair(ir)
        self.assertEqual(report['status'], 'unchanged')
        self.assertTrue(report['after_validation']['valid'])
        self.assertEqual(repaired, ir)
        self.assertIsNot(repaired, ir)

    def test_wrong_parent_is_diagnosed_without_guessing(self):
        ir = self.wrong_owner()
        repaired, report = self.run_repair(ir)
        self.assertEqual(repaired, ir)
        self.assertEqual(report['status'], 'unresolved')
        self.assertEqual(report['before_validation']['errors'][0]['path'], '/elements/0/bbox')
        self.assertEqual(report['before_validation']['errors'][0]['code'], 'outside_region')
        self.assertEqual(report['unresolved'][0]['code'], 'insufficient_evidence')

    def test_unique_containing_region_changes_only_owner(self):
        ir = self.wrong_owner()
        repaired, report = self.run_repair(ir, evidence(ir, observation(ir)))
        self.assertTrue(report['after_validation']['valid'])
        self.assertEqual(report['status'], 'repaired')
        self.assertEqual(repaired['elements'][0]['region_id'], 'right')
        self.assertEqual(repaired['elements'][0]['bbox'], ir['elements'][0]['bbox'])
        self.assertEqual([d['path'] for d in report['changes']], ['/elements/0/region_id'])

    def test_explicit_children_are_synchronized_and_reported(self):
        ir = self.wrong_owner()
        ir['regions'][0]['children'] = ['caption', 'source']
        ir['regions'][1]['children'] = ['target']
        repaired, report = self.run_repair(ir, evidence(ir, observation(ir)))
        self.assertTrue(report['after_validation']['valid'])
        self.assertEqual(repaired['regions'][0]['children'], ['source'])
        self.assertEqual(repaired['regions'][1]['children'], ['target', 'caption'])
        self.assertEqual({d['path'] for d in report['changes']},
                         {'/elements/0/region_id', '/regions/0/children', '/regions/1/children'})

    def test_page_level_measurement_updates_box_and_owner_atomically(self):
        ir = sample_ir()
        ir['elements'][0]['bbox'] = [30, 265, 100, 20]
        proof = evidence(ir, observation(ir, measured_bbox=[80, 280, 150, 15],
                                         ownership='unassigned', page_level=True))
        repaired, report = self.run_repair(ir, proof)
        self.assertTrue(report['after_validation']['valid'])
        self.assertIsNone(repaired['elements'][0]['region_id'])
        self.assertEqual(repaired['elements'][0]['bbox'], [80, 280, 150, 15])
        self.assertEqual({d['path'] for d in report['changes']},
                         {'/elements/0/bbox', '/elements/0/region_id'})

    def test_page_level_requires_explicit_review(self):
        ir = self.wrong_owner()
        proof = evidence(ir, observation(ir, measured_bbox=[80, 280, 150, 15], ownership='unassigned'))
        self.assert_rejected_unchanged(ir, proof, 'Page-level ownership')

    def test_page_level_assignment_cannot_hide_region_overlap(self):
        ir = self.wrong_owner()
        proof = evidence(ir, observation(ir, ownership='unassigned', page_level=True))
        self.assert_rejected_unchanged(ir, proof, 'region overlap')

    def test_minimal_region_expansion_does_not_fill_entire_support(self):
        ir = sample_ir()
        ir['regions'][0]['bbox'] = [10, 10, 160, 100]
        proof = evidence(ir, observation(ir, element_index=1, action='expand_region',
                                         verified_owner=True, region_support_bbox=[10, 10, 160, 170]))
        repaired, report = self.run_repair(ir, proof)
        self.assertTrue(report['after_validation']['valid'])
        self.assertEqual(repaired['regions'][0]['bbox'], [10, 10, 160, 140])
        self.assertEqual(repaired['regions'][1], ir['regions'][1])
        self.assertEqual(repaired['elements'], ir['elements'])
        self.assertEqual([d['path'] for d in report['changes']], ['/regions/0/bbox'])

    def test_region_expansion_rejects_new_neighbor_overlap(self):
        ir = sample_ir()
        ir['regions'][0]['bbox'] = [10, 10, 100, 260]
        ir['regions'][1]['bbox'] = [140, 10, 250, 260]
        ir['elements'][0]['bbox'] = [30, 40, 70, 35]
        ir['elements'][1]['bbox'] = [90, 100, 60, 50]
        proof = evidence(ir, observation(ir, element_index=1, action='expand_region',
                                         verified_owner=True, region_support_bbox=[10, 10, 160, 260]))
        self.assert_rejected_unchanged(ir, proof, 'overlap')

    def test_region_expansion_requires_support_and_confirmed_owner(self):
        ir = sample_ir()
        ir['regions'][0]['bbox'] = [10, 10, 160, 100]
        for updates in ({'verified_owner': False, 'region_support_bbox': [10, 10, 160, 170]},
                        {'verified_owner': True, 'region_support_bbox': [10, 10, 160, 100]}):
            with self.subTest(updates=updates):
                proof = evidence(ir, observation(ir, element_index=1, action='expand_region', **updates))
                self.assert_rejected_unchanged(ir, proof)

    def test_out_of_canvas_measurement_is_not_clipped(self):
        ir = self.wrong_owner()
        proof = evidence(ir, observation(ir, measured_bbox=[390, 280, 20, 10]))
        self.assert_rejected_unchanged(ir, proof, 'outside the image')

    def test_verified_measurement_can_repair_out_of_canvas_element(self):
        ir = sample_ir()
        ir['elements'][0]['bbox'] = [390, 40, 30, 35]
        self.assertIn('out_of_bounds', {d['code'] for d in inspect_ir(ir)['errors']})
        proof = evidence(ir, observation(ir, action='measure_bbox', measured_bbox=[30, 40, 110, 35]))
        repaired, report = self.run_repair(ir, proof)
        self.assertTrue(report['after_validation']['valid'])
        self.assertEqual(repaired['elements'][0]['region_id'], 'left')
        self.assertEqual([d['path'] for d in report['changes']], ['/elements/0/bbox'])

    def test_overlapping_regions_reject_ambiguous_assignment_and_box_move(self):
        ir = sample_ir()
        ir['regions'][1]['bbox'] = [100, 10, 290, 260]
        ir['elements'][0]['bbox'] = [30, 265, 100, 20]
        proof = evidence(ir, observation(ir, measured_bbox=[120, 40, 30, 35]))
        self.assert_rejected_unchanged(ir, proof, 'ambiguous')

    def test_incorrect_source_binding_and_coordinate_frames_are_rejected(self):
        ir = self.wrong_owner()
        for field, value in [('image_sha256', 'b' * 64), ('ir_sha256', 'b' * 64),
                             ('coordinate_system', 'normalized'), ('canvas_size', [800, 600])]:
            proof = evidence(ir, observation(ir))
            proof[field] = value
            with self.subTest(field=field):
                report = self.assert_rejected_unchanged(ir, proof)
                self.assertEqual(report['unresolved'][0]['code'], 'invalid_evidence')

    def test_unconfirmed_or_unexplained_observation_is_rejected(self):
        ir = self.wrong_owner()
        for update in ({'confirmed': False}, {'basis': ''}, {'basis': '  '}):
            with self.subTest(update=update):
                self.assert_rejected_unchanged(ir, evidence(ir, observation(ir, **update)))

    def test_stale_geometry_preconditions_are_rejected(self):
        ir = self.wrong_owner()
        for update in ({'expected_bbox': [0, 0, 1, 1]}, {'expected_region_id': 'missing'}):
            with self.subTest(update=update):
                self.assert_rejected_unchanged(ir, evidence(ir, observation(ir, **update)), 'preconditions')

    def test_new_validation_errors_reject_entire_measurement(self):
        ir = sample_ir()
        proof = evidence(ir, observation(ir, action='measure_bbox', measured_bbox=[250, 40, 100, 35]))
        self.assert_rejected_unchanged(ir, proof, 'new validation errors')

    def test_input_evidence_semantics_and_connections_are_preserved(self):
        ir = self.wrong_owner()
        proof = evidence(ir, observation(ir))
        old_ir, old_proof = deepcopy(ir), deepcopy(proof)
        repaired, report = self.run_repair(ir, proof)
        self.assertEqual(ir, old_ir)
        self.assertEqual(proof, old_proof)
        self.assertEqual(repaired['connections'], old_ir['connections'])
        self.assertEqual([e['id'] for e in repaired['elements']], [e['id'] for e in old_ir['elements']])
        for old, new in zip(old_ir['elements'], repaired['elements']):
            self.assertEqual({k: v for k, v in old.items() if k not in {'bbox', 'region_id'}},
                             {k: v for k, v in new.items() if k not in {'bbox', 'region_id'}})
        self.assertTrue(report['semantic_content_preserved'])
        self.assertEqual(report['api_calls'], 0)
        repaired['connections'].clear()
        self.assertEqual(len(ir['connections']), 1)

    def test_structurally_invalid_ir_is_not_repaired(self):
        ir = sample_ir()
        del ir['version']
        repaired, report = self.run_repair(ir)
        self.assertEqual(repaired, ir)
        self.assertEqual(report['unresolved'][0]['code'], 'unsupported_input_error')
        self.assertFalse(report['after_validation']['valid'])

    def test_existing_valid_ir_scene_and_tikz_are_unchanged(self):
        ir = sample_ir()
        repaired, report = self.run_repair(ir)
        first_scene, first_conversion = convert_ir(ir)
        second_scene, second_conversion = convert_ir(repaired)
        self.assertTrue(first_conversion['complete'])
        self.assertEqual((first_scene, first_conversion), (second_scene, second_conversion))
        self.assertEqual(generate_tikz(first_scene), generate_tikz(second_scene))
        self.assertEqual(report['changes'], [])


if __name__ == '__main__':
    unittest.main()
