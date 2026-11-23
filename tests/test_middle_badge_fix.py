"""Focused offline checks for the reviewed real Scene and opt-in circle text."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from middle_badge_repair import BADGE_IDS, repair_middle_scene
from structured_tikz.generator import generate_tikz
from structured_tikz.primitives import point
from structured_tikz.validator import SceneValidationError, validate_scene

SOURCE = Path(__file__).resolve().parent / 'fixtures' / 'historical' / 'middle_badge'


class MiddleBadgeFixTests(OfflineTestCase):
    def setUp(self):
        self.raw = (SOURCE / 'scene_repaired.json').read_bytes()
        self.scene = json.loads(self.raw)

    def test_legacy_whole_scene_output_is_byte_equivalent(self):
        self.assertEqual(generate_tikz(self.scene), (SOURCE / 'scene.tikz').read_text(encoding='utf-8'))

    def test_explicit_circle_is_one_editable_object(self):
        fixed, _ = repair_middle_scene(self.scene)
        tikz = generate_tikz(fixed)
        for digit, node_id in enumerate(BADGE_IDS, 1):
            lines = [line for line in tikz.splitlines() if '(n-' + node_id + ')' in line]
            self.assertEqual(len(lines), 1)
            self.assertIn(',circle,minimum size=16pt,', lines[0])
            self.assertIn('anchor=center,align=center', lines[0])
            self.assertNotIn('text width=', lines[0])
            self.assertTrue(lines[0].endswith('{' + str(digit) + '};'))
            self.assertNotIn('(t-' + node_id + ')', tikz)

    def test_blue_fill_and_white_text_are_preserved(self):
        fixed, _ = repair_middle_scene(self.scene)
        validate_scene(fixed)
        for line in generate_tikz(fixed).splitlines():
            if any('(n-' + node_id + ')' in line for node_id in BADGE_IDS):
                self.assertIn('fill=sceneblue', line)
                self.assertIn('text=scenewhite', line)
                self.assertIn('draw=none', line)
                self.assertIn(r'\fontsize{8.8pt}{11pt}\selectfont\bfseries', line)

    def test_all_three_centers_and_dimensions_are_unchanged(self):
        fixed, _ = repair_middle_scene(self.scene)
        before = {node['id']: node for node in self.scene['nodes']}
        after = {node['id']: node for node in fixed['nodes']}
        for node_id, center in zip(BADGE_IDS, ((55, 200), (55, 340), (55, 790))):
            self.assertEqual(before[node_id]['bbox'], after[node_id]['bbox'])
            x, y, w, h = after[node_id]['bbox']
            self.assertEqual((x + w / 2, y + h / 2), center)
            self.assertIn('(n-' + node_id + ') at ' + point(*center, fixed['canvas']), generate_tikz(fixed))

    def test_only_confirmed_duplicate_label_is_removed(self):
        extra = deepcopy(self.scene['labels'][0])
        extra.update(id='legitimate_elsewhere', x=100, y=900)
        extra['metadata'] = {'bbox': [100, 860, 400, 40]}
        self.scene['labels'].append(extra)
        fixed, changes = repair_middle_scene(self.scene)
        self.assertEqual(fixed['labels'], [extra])
        self.assertEqual([c['id'] for c in changes if c['operation'] == 'remove_label'], ['lbl_fragment'])
        self.assertEqual(changes[0]['original'], self.scene['labels'][0])
        self.assertEqual(next(n for n in fixed['nodes'] if n['id'] == 'top_title_label'),
                         next(n for n in self.scene['nodes'] if n['id'] == 'top_title_label'))

    def test_external_references_prevent_removal(self):
        for container in ('root', 'node', 'label'):
            with self.subTest(container=container):
                scene = deepcopy(self.scene)
                target = scene if container == 'root' else scene['nodes'][0] if container == 'node' else scene['labels'][0]
                target.setdefault('metadata', {})['reference'] = {'targets': ['lbl_fragment']}
                # A self-reference in the deleted label has no surviving owner;
                # use a second label to test an actual external reference.
                if container == 'label':
                    other = deepcopy(target)
                    other['id'] = 'other_label'
                    scene['labels'].append(other)
                with self.assertRaisesRegex(ValueError, 'referenced'):
                    repair_middle_scene(scene)

    def test_ambiguous_duplicate_is_not_deleted(self):
        for key, value in (('text', 'Different title'), ('x', 100), ('font_size', 25)):
            with self.subTest(key=key):
                scene = deepcopy(self.scene)
                scene['labels'][0][key] = value
                with self.assertRaisesRegex(ValueError, 'Cannot confirm'):
                    repair_middle_scene(scene)

    def test_rich_text_and_math_output_unchanged(self):
        fixed, _ = repair_middle_scene(self.scene)
        before_line = next(line for line in generate_tikz(self.scene).splitlines() if '(t-step2_title)' in line)
        after_line = next(line for line in generate_tikz(fixed).splitlines() if '(t-step2_title)' in line)
        self.assertEqual(before_line, after_line)
        self.assertIn(r'$b = 0,\ldots, L-1$', after_line)

    def test_repaired_scene_passes_schema_and_has_unique_ids(self):
        fixed, _ = repair_middle_scene(self.scene)
        validate_scene(fixed)
        ids = [item['id'] for collection in ('nodes', 'edges', 'labels') for item in fixed[collection]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(fixed['nodes']), 26)
        self.assertEqual(fixed['labels'], [])

    def test_input_object_and_original_file_unchanged(self):
        snapshot = deepcopy(self.scene)
        repair_middle_scene(self.scene)
        self.assertEqual(snapshot, self.scene)
        self.assertEqual(self.raw, (SOURCE / 'scene_repaired.json').read_bytes())

    def test_idempotence_without_extra_backgrounds(self):
        once, changes = repair_middle_scene(self.scene)
        twice, second_changes = repair_middle_scene(once)
        self.assertEqual(once, twice)
        self.assertEqual(len(changes), 4)
        self.assertEqual(second_changes, [])
        self.assertEqual(sum(n.get('background_shape') == 'circle' for n in twice['nodes']), 3)

    def test_other_nodes_text_styles_layout_and_topology_unchanged(self):
        fixed, _ = repair_middle_scene(self.scene)
        for old, new in zip(self.scene['nodes'], fixed['nodes']):
            expected = dict(old, background_shape='circle') if old['id'] in BADGE_IDS else old
            self.assertEqual(new, expected)
        for key in ('canvas', 'styles', 'edges', 'metadata'):
            self.assertEqual(self.scene[key], fixed[key])

    def test_invalid_background_contract_is_rejected(self):
        for mutation in ({'background_shape': 'ellipse'}, {'background_shape': None},
                         {'background_shape': 'circle', 'type': 'rectangle'},
                         {'background_shape': 'circle', 'bbox': [35, 180, 41, 40]}):
            with self.subTest(mutation=mutation):
                scene = deepcopy(self.scene)
                next(n for n in scene['nodes'] if n['id'] == 'step1_badge').update(mutation)
                with self.assertRaises(SceneValidationError):
                    validate_scene(scene)

    def test_circle_preserves_existing_node_reference_name(self):
        fixed, _ = repair_middle_scene(self.scene)
        fixed['edges'] = [{'id': 'test_link', 'from': 'step1_badge', 'to': 'step1_title', 'arrow': 'end'}]
        code = generate_tikz(fixed)
        self.assertIn('(n-step1_badge.east)', code)
        self.assertIn('(n-step1_title.west)', code)

    def test_circle_also_accepts_legal_inline_style(self):
        fixed, _ = repair_middle_scene(self.scene)
        node = next(n for n in fixed['nodes'] if n['id'] == 'step1_badge')
        node['style'] = deepcopy(fixed['styles']['badge'])
        code = generate_tikz(fixed)
        line = next(line for line in code.splitlines() if '(n-step1_badge)' in line)
        self.assertIn('fill=sceneblue', line)
        self.assertIn('text=scenewhite', line)


if __name__ == '__main__':
    unittest.main()
