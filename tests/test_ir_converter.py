from copy import deepcopy
import json
try:
    from .offline_support import OfflineTestCase
    from .ir_fixtures import example_ir, branch_ir
except ImportError:
    from offline_support import OfflineTestCase
    from ir_fixtures import example_ir, branch_ir
from structured_ir.converter import convert_ir
from structured_ir.validator import validate_ir
from structured_tikz.validator import validate_scene
from structured_tikz.generator import generate_tikz


class IRConverterTests(OfflineTestCase):
    def test_wrong_type_optional_maps_are_not_silently_defaulted(self):
        for key in ('style_map', 'config'):
            for value in ([], '', False, 0, [('scale', 1)]):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    convert_ir(example_ir(), **{key: value})

    def test_deterministic_content_geometry_and_hierarchy(self):
        ir = example_ir()
        before = deepcopy(ir)
        scene, report = convert_ir(ir)
        self.assertTrue(report['complete'])
        validate_scene(scene)
        self.assertEqual(ir, before)
        self.assertEqual(convert_ir(ir), (scene, report))
        self.assertEqual(scene['metadata']['regions'], ir['regions'])
        for a, b in zip(ir['elements'], scene['nodes']):
            self.assertEqual(a['raw_text'], b['text'])
            self.assertEqual(a['bbox'], b['bbox'])
        self.assertEqual(scene['edges'][0]['from'], 'start')
        tex = generate_tikz(scene)
        self.assertIn('\\node', tex)
        self.assertNotIn('includegraphics', tex)

    def test_diamond_and_its_incident_edges_map_without_substitution(self):
        ir = branch_ir()
        validate_ir(ir)
        scene, report = convert_ir(ir)
        self.assertTrue(report['complete'])
        self.assertTrue(report['scene_valid'])
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['unsupported_elements'], [])
        decision = next(n for n in scene['nodes'] if n['id'] == 'decision')
        self.assertEqual(decision['type'], 'diamond')
        self.assertEqual(decision['bbox'], ir['elements'][0]['bbox'])
        self.assertEqual(decision['text'], '有效？')
        self.assertEqual({e['id'] for e in scene['edges']}, {'yes', 'no', 'join_a', 'join_b'})
        self.assertEqual(scene['metadata']['ir_connections'], ir['connections'])
        self.assertEqual(report['unrendered_text'], [])

    def test_literal_code_math_multiline_and_circle(self):
        ir = example_ir()
        ir['elements'][0].update(type='multiline_text', raw_text='第一行\ncode_[1] "x"')
        ir['elements'][1].update(type='code', raw_text='a[0] = "_"')
        ir['elements'][2].update(type='math', raw_text='x_{1}+2', text_type='math')
        ir['elements'].append({'id': 'circle', 'type': 'circle', 'bbox': [60, 250, 40, 40], 'region_id': None})
        scene, report = convert_ir(ir)
        self.assertTrue(report['complete'])
        self.assertEqual(scene['nodes'][3]['background_shape'], 'circle')
        self.assertEqual(scene['nodes'][0]['text'], '第一行\ncode_[1] "x"')
        self.assertEqual(scene['nodes'][2]['mode'], 'math')
        generate_tikz(scene)

    def test_unsupported_math_stays_in_ir_and_partial_report(self):
        ir = example_ir()
        ir['elements'][0].update(type='math', text_type='math', raw_text=r'\unknown{a}')
        scene, report = convert_ir(ir)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['unrendered_text'][0]['raw_text'], r'\unknown{a}')
        self.assertNotIn('heading', [n['id'] for n in scene['nodes']])

    def test_center_anchor_path_preserves_endpoint_metadata(self):
        ir = example_ir()
        ir['connections'][0].update(source_anchor='center', direction='both')
        scene, report = convert_ir(ir)
        self.assertTrue(report['complete'])
        edge = scene['edges'][0]
        self.assertEqual(edge['points'], [[165, 157.5], [370, 157.5]])
        self.assertEqual(edge['metadata']['source_id'], 'start')
        self.assertEqual(edge['arrow'], 'both')

    def test_semantic_branch_label_requires_explicit_placement(self):
        ir = example_ir()
        ir['connections'][0].update(kind='branch', branch_label='是')
        scene, report = convert_ir(ir)
        self.assertFalse(report['complete'])
        self.assertEqual(scene['labels'], [])
        ir['connections'][0]['label_bbox'] = [290, 120, 50, 30]
        scene, report = convert_ir(ir)
        self.assertTrue(report['complete'])
        self.assertEqual(scene['labels'][0]['text'], '是')
        self.assertEqual(scene['labels'][0]['x'], 315)
        scene['labels'][0]['metadata']['bbox'][0] = 0
        self.assertEqual(ir['connections'][0]['label_bbox'][0], 290)

    def test_explicit_paths_title_band_and_styles(self):
        ir = example_ir()
        ir['elements'][0].update(type='title_band', style_role='heading')
        ir['connections'][0]['points'] = [[270, 157.5], [320, 157.5], [370, 157.5]]
        scene, report = convert_ir(ir, {'heading': {'fill': 'light_blue'}})
        self.assertTrue(report['complete'])
        self.assertEqual(scene['nodes'][0]['type'], 'rectangle')
        self.assertEqual(scene['nodes'][0]['style']['fill'], 'light_blue')
        self.assertEqual(scene['edges'][0]['points'], ir['connections'][0]['points'])
        self.assertEqual(scene['edges'][0]['metadata']['target_id'], 'finish')

    def test_invalid_style_and_non_square_circle_fail_explicitly(self):
        ir = example_ir()
        ir['elements'][0].update(type='circle')
        scene, report = convert_ir(ir)
        self.assertEqual(report['status'], 'partial')
        ir = example_ir()
        ir['elements'][0]['style_role'] = 'bad'
        scene, report = convert_ir(ir, {'bad': {'fill': 'not_a_color'}})
        self.assertEqual(report['status'], 'partial')
        self.assertTrue(report['scene_valid'])

    def test_geometric_arrow_does_not_invent_semantic_endpoints(self):
        ir = example_ir()
        ir['elements'].append({'id': 'line', 'type': 'arrow', 'bbox': [50, 300, 100, 1],
                               'points': [[50, 300], [150, 300]], 'region_id': None})
        scene, report = convert_ir(ir)
        self.assertTrue(report['complete'])
        edge = next(e for e in scene['edges'] if e['id'] == 'line')
        self.assertNotIn('from', edge)
        self.assertEqual(edge['arrow'], 'end')

    def test_disconnected_explicit_semantic_path_is_not_complete(self):
        ir = example_ir()
        ir['connections'][0]['points'] = [[10, 10], [20, 20]]
        scene, report = convert_ir(ir)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(scene['edges'][0]['points'], [[10, 10], [20, 20]])
        self.assertIn('connection_path_anchor_mismatch', [d['code'] for d in report['diagnostics']])

    def test_empty_but_legal_ir_cannot_be_complete_conversion(self):
        ir = example_ir()
        ir.update(regions=[], elements=[], connections=[])
        validate_ir(ir)
        scene, report = convert_ir(ir)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(scene['nodes'], [])
        self.assertEqual(report['diagnostics'][-1]['code'], 'no_drawable_content')
