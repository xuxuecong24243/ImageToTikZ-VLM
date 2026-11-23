"""IR contract tests, deliberately synthetic rather than page_01 transcription."""
import copy
import json
import unittest
from unittest.mock import patch

try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from structured_ir.schema import ELEMENT_TYPES, ir_schema
from structured_ir.validator import IRValidationError, inspect_ir, validate_ir


def synthetic_ir():
    return {
        'version': '1.0',
        'canvas': {'id': 'canvas', 'width': 400, 'height': 300, 'coordinate_system': 'top-left'},
        'regions': [{'id': 'panel', 'type': 'panel', 'bbox': [0, 0, 400, 300], 'parent_id': None},
                    {'id': 'flow', 'type': 'flow', 'bbox': [10, 10, 380, 280], 'parent_id': 'panel'}],
        'elements': [
            {'id': 'decision', 'type': 'diamond', 'bbox': [150, 20, 100, 50], 'region_id': 'flow',
             'raw_text': 'condition?', 'text_type': 'plain', 'reading_order': 0},
            {'id': 'left', 'type': 'rectangle', 'bbox': [30, 100, 80, 50], 'region_id': 'flow'},
            {'id': 'right', 'type': 'rounded_rectangle', 'bbox': [270, 100, 80, 50], 'region_id': 'flow'},
            {'id': 'join', 'type': 'text', 'bbox': [150, 200, 100, 40], 'region_id': 'flow',
             'raw_text': 'continue', 'text_type': 'plain', 'reading_order': 1},
        ],
        'connections': [
            {'id': 'yes_edge', 'source_id': 'decision', 'target_id': 'left', 'kind': 'branch',
             'direction': 'forward', 'source_anchor': 'west', 'target_anchor': 'north', 'branch_label': 'yes'},
            {'id': 'no_edge', 'source_id': 'decision', 'target_id': 'right', 'kind': 'branch',
             'direction': 'forward', 'source_anchor': 'east', 'target_anchor': 'north', 'branch_label': 'no'},
            {'id': 'merge_left', 'source_id': 'left', 'target_id': 'join', 'kind': 'merge',
             'direction': 'forward', 'source_anchor': 'south', 'target_anchor': 'west'},
            {'id': 'merge_right', 'source_id': 'right', 'target_id': 'join', 'kind': 'merge',
             'direction': 'forward', 'source_anchor': 'south', 'target_anchor': 'east'},
        ],
        'provenance': {'kind': 'synthetic_unit_fixture', 'recognition': False},
    }


class IRSchemaTests(OfflineTestCase):
    def assert_issue(self, ir, path, code=None):
        report = inspect_ir(ir)
        self.assertFalse(report['valid'], report)
        self.assertTrue(any(e['path'] == path and (code is None or e['code'] == code)
                            for e in report['errors']), report)
        return report

    def test_valid_independent_ir_preserves_input(self):
        data = synthetic_ir()
        original = copy.deepcopy(data)
        self.assertIs(validate_ir(data), data)
        self.assertEqual(data, original)

    def test_schema_is_json_and_returns_independent_copy(self):
        schema = ir_schema()
        json.dumps(schema, allow_nan=False)
        schema['$defs']['element']['properties']['type']['enum'].clear()
        self.assertEqual(set(ir_schema()['$defs']['element']['properties']['type']['enum']), ELEMENT_TYPES)

    def test_validator_consumes_exported_schema(self):
        schema = ir_schema()
        schema['$defs']['canvas']['properties']['width']['maximum'] = 50
        with patch('structured_ir.validator.ir_schema', return_value=schema):
            self.assert_issue(synthetic_ir(), '/canvas/width', 'number_range')

    def test_missing_required_root_field(self):
        data = synthetic_ir()
        del data['canvas']
        self.assert_issue(data, '/canvas', 'missing_field')

    def test_unknown_field_is_preserved_and_reported(self):
        data = synthetic_ir()
        data['type'] = 'json_object'
        self.assert_issue(data, '/type', 'unknown_field')
        self.assertEqual(data['type'], 'json_object')

    def test_unknown_nested_field_json_pointer_escaping(self):
        data = synthetic_ir()
        data['elements'][3]['a/b~c'] = 1
        self.assert_issue(data, '/elements/3/a~1b~0c', 'unknown_field')

    def test_wrong_field_type(self):
        data = synthetic_ir()
        data['connections'] = {}
        self.assert_issue(data, '/connections', 'field_type')

    def test_missing_text_is_not_replaced_with_placeholder(self):
        data = synthetic_ir()
        del data['elements'][3]['raw_text']
        self.assert_issue(data, '/elements/3/raw_text', 'missing_field')
        self.assertNotIn('raw_text', data['elements'][3])

    def test_raw_text_requires_kind_and_reading_order(self):
        data = synthetic_ir()
        del data['elements'][0]['reading_order']
        self.assert_issue(data, '/elements/0/reading_order', 'missing_field')

    def test_code_and_math_require_matching_text_type(self):
        for kind in ('code', 'math'):
            with self.subTest(kind=kind):
                data = synthetic_ir()
                data['elements'][3]['type'] = kind
                self.assert_issue(data, '/elements/3/text_type', 'text_type_mismatch')
                data['elements'][3]['text_type'] = kind
                self.assertTrue(inspect_ir(data)['valid'])

    def test_diamond_branch_merge_graph_is_valid(self):
        data = synthetic_ir()
        self.assertTrue(inspect_ir(data)['valid'])
        self.assertEqual([e['branch_label'] for e in data['connections'][:2]], ['yes', 'no'])
        self.assertEqual({e['target_id'] for e in data['connections'][2:]}, {'join'})

    def test_all_declared_element_types_have_valid_representation(self):
        for kind in ELEMENT_TYPES:
            with self.subTest(kind=kind):
                data = synthetic_ir()
                item = data['elements'][3]
                item['type'] = kind
                item['text_type'] = {'code': 'code', 'math': 'math'}.get(kind, 'plain')
                if kind in ('connector', 'arrow'):
                    item['points'] = [[150, 200], [250, 240]]
                self.assertTrue(inspect_ir(data)['valid'], inspect_ir(data))

    def test_geometric_arrow_requires_points(self):
        data = synthetic_ir()
        data['elements'][3]['type'] = 'arrow'
        self.assert_issue(data, '/elements/3/points', 'missing_field')

    def test_points_are_rejected_on_non_line_element_without_silent_loss(self):
        data = synthetic_ir()
        data['elements'][3]['points'] = [[150, 200], [250, 240]]
        self.assert_issue(data, '/elements/3/points', 'forbidden_field')
        self.assertEqual(data['elements'][3]['points'], [[150, 200], [250, 240]])

    def test_unknown_element_type_rejected_without_loss(self):
        data = synthetic_ir()
        data['elements'][0]['type'] = 'hexagon'
        self.assert_issue(data, '/elements/0/type', 'unknown_type')
        self.assertEqual(len(data['elements']), 4)

    def test_bbox_dimensions_and_types(self):
        for box, pointer in [([1, 2, 0, 4], '/elements/3/bbox/2'),
                             ([1, 2, -1, 4], '/elements/3/bbox/2'),
                             ([True, 2, 3, 4], '/elements/3/bbox/0'),
                             ([1, 2, 3], '/elements/3/bbox')]:
            with self.subTest(box=box):
                data = synthetic_ir()
                data['elements'][3]['bbox'] = box
                self.assert_issue(data, pointer)

    def test_bbox_full_extent_inside_canvas(self):
        data = synthetic_ir()
        data['elements'][3]['bbox'] = [390, 200, 20, 40]
        self.assert_issue(data, '/elements/3/bbox', 'out_of_bounds')

    def test_region_ownership_bounds(self):
        data = synthetic_ir()
        data['elements'][3]['bbox'] = [0, 200, 20, 40]
        self.assert_issue(data, '/elements/3/bbox', 'outside_region')

    def test_parent_cycle_is_rejected(self):
        data = synthetic_ir()
        data['regions'][0]['parent_id'] = 'flow'
        self.assert_issue(data, '/regions/0/parent_id', 'parent_cycle')

    def test_semantic_graph_cycle_is_allowed(self):
        data = synthetic_ir()
        data['connections'].append({'id': 'loop', 'source_id': 'join', 'target_id': 'decision',
                                    'source_anchor': 'center', 'target_anchor': 'center',
                                    'kind': 'flow', 'direction': 'forward'})
        self.assertTrue(inspect_ir(data)['valid'])

    def test_invalid_owner_and_endpoint_references(self):
        for path, mutate in [('/elements/3/region_id', lambda d: d['elements'][3].update(region_id='missing')),
                             ('/connections/2/source_id', lambda d: d['connections'][2].update(source_id='panel')),
                             ('/regions/1/parent_id', lambda d: d['regions'][1].update(parent_id='missing'))]:
            with self.subTest(path=path):
                data = synthetic_ir()
                mutate(data)
                self.assert_issue(data, path, 'invalid_reference')

    def test_children_must_agree_with_actual_ownership(self):
        data = synthetic_ir()
        data['regions'][0]['children'] = ['flow']
        data['regions'][1]['children'] = ['decision', 'left', 'right', 'join']
        self.assertTrue(inspect_ir(data)['valid'])
        data['regions'][0]['children'].append('left')
        self.assert_issue(data, '/regions/0/children/1', 'ownership_mismatch')

    def test_children_cannot_omit_actual_child(self):
        data = synthetic_ir()
        data['regions'][0]['children'] = []
        self.assert_issue(data, '/regions/0/children', 'missing_children')

    def test_unknown_region_semantics_warning_not_false_success(self):
        data = synthetic_ir()
        data['regions'][0]['type'] = 'unknown'
        report = inspect_ir(data)
        self.assertTrue(report['valid'])
        self.assertEqual(report['warnings'][0]['code'], 'uncertain_region')

    def test_duplicate_ids_across_collections_and_canvas(self):
        for existing in ('canvas', 'panel', 'decision'):
            with self.subTest(existing=existing):
                data = synthetic_ir()
                data['connections'][0]['id'] = existing
                self.assert_issue(data, '/connections/0/id', 'duplicate_id')

    def test_invalid_id_and_anchor(self):
        data = synthetic_ir()
        data['elements'][3]['id'] = 'new\n'
        self.assert_issue(data, '/elements/3/id', 'identifier')
        data = synthetic_ir()
        data['connections'][0]['source_anchor'] = 'northwest'
        self.assert_issue(data, '/connections/0/source_anchor', 'enum')

    def test_branch_label_required_nonempty_unique(self):
        data = synthetic_ir()
        del data['connections'][0]['branch_label']
        self.assert_issue(data, '/connections/0/branch_label', 'missing_field')
        data['connections'][0]['branch_label'] = ' '
        self.assert_issue(data, '/connections/0/branch_label', 'empty_branch_label')
        data['connections'][0]['branch_label'] = 'no'
        self.assert_issue(data, '/connections/1/branch_label', 'ambiguous_branch_label')

    def test_path_points_bounds_and_no_consecutive_duplicates(self):
        for points, code in [([[20, 20], [410, 20]], 'out_of_bounds'),
                              ([[20, 20], [20, 20]], 'duplicate_point')]:
            with self.subTest(points=points):
                data = synthetic_ir()
                data['connections'][0]['points'] = points
                self.assert_issue(data, '/connections/0/points/1', code)

    def test_branch_label_bbox_bounds_and_requires_text(self):
        data = synthetic_ir()
        data['connections'][2]['label_bbox'] = [10, 20, 30, 20]
        self.assert_issue(data, '/connections/2/branch_label', 'missing_field')

    def test_confidence_range(self):
        data = synthetic_ir()
        data['elements'][3]['confidence'] = 1.2
        self.assert_issue(data, '/elements/3/confidence', 'number_range')

    def test_json_malformed_objects_do_not_crash(self):
        for value in (None, [], 'not json', {'canvas': []}, {'x': set()}, {3: 'badkey'}):
            with self.subTest(value=repr(value)):
                self.assertFalse(inspect_ir(value)['valid'])

    def test_nonfinite_and_cyclic_python_inputs_rejected(self):
        data = synthetic_ir()
        data['canvas']['width'] = float('nan')
        self.assert_issue(data, '/canvas/width', 'non_finite')
        data = synthetic_ir()
        data['provenance']['self'] = data
        self.assert_issue(data, '/provenance/self', 'cyclic_data')

    def test_enormous_integer_reports_range_instead_of_overflow(self):
        data = synthetic_ir()
        data['canvas']['width'] = 10 ** 500
        self.assert_issue(data, '/canvas/width', 'number_range')

    def test_exception_has_structured_diagnostics(self):
        with self.assertRaises(IRValidationError) as caught:
            validate_ir({})
        self.assertFalse(caught.exception.report['valid'])
        self.assertIn('/version', str(caught.exception))

    def test_raw_text_unicode_code_and_newlines_are_unchanged(self):
        data = synthetic_ir()
        text = '中文 f(x_1)\narray[0] == "yes"'
        data['elements'][3].update(type='multiline_text', raw_text=text)
        self.assertTrue(inspect_ir(data)['valid'])
        self.assertEqual(data['elements'][3]['raw_text'], text)


if __name__ == '__main__':
    unittest.main()
