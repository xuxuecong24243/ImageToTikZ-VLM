"""Synthetic text data only: no model, original figure or compiler is used."""
import copy
import json
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from scene_text_normalizer import normalize_scene_text
from structured_tikz.validator import validate_scene


def scene():
    return {'canvas': {'width': 600, 'height': 300, 'coordinate_system': 'top-left'},
            'styles': {},
            'nodes': [{'id': 'A', 'type': 'rectangle', 'bbox': [40, 100, 100, 80]},
                      {'id': 'B', 'type': 'rectangle', 'bbox': [400, 100, 100, 80]}],
            'edges': [], 'labels': [], 'metadata': {'review_status': 'unverified'}}


def label(identifier, text='中文标签', x=120, y=40, mode='text', owner=None):
    return {'id': identifier, 'text': text, 'x': x, 'y': y,
            'mode': mode, 'font_size': 20, 'anchor': 'center', 'align': 'center',
            'metadata': {'bbox': [x-50, y-10, 100, 20], 'node_id': owner}}


def text_node(identifier='caption', text='中文标签'):
    return {'id': identifier, 'type': 'text', 'bbox': [70, 30, 100, 20],
            'text': text, 'mode': 'text', 'font_size': 20, 'align': 'center'}


class SceneTextNormalizerTests(OfflineTestCase):
    def test_same_position_duplicate_labels_merge_with_audit(self):
        original = scene()
        original['labels'] = [label('first'), label('second')]
        normalized, report = normalize_scene_text(original)
        self.assertEqual(len(normalized['labels']), 1)
        self.assertEqual(normalized['labels'][0]['text'], '中文标签')
        self.assertTrue(report['changes'])
        audit = json.dumps(report, ensure_ascii=False)
        self.assertIn('first', audit)
        self.assertIn('second', audit)
        self.assertIn('中文标签', audit)
        validate_scene(normalized)

    def test_identical_text_at_different_positions_is_not_deleted(self):
        original = scene()
        original['labels'] = [label('left', x=120), label('right', x=470)]
        normalized, _ = normalize_scene_text(original)
        self.assertEqual(normalized['labels'], original['labels'])

    def test_text_node_duplicate_prefers_existing_label(self):
        original = scene()
        original['nodes'].append(text_node())
        original['labels'] = [label('caption_label')]
        normalized, report = normalize_scene_text(original)
        self.assertNotIn('caption', {node['id'] for node in normalized['nodes']})
        self.assertEqual(normalized['labels'][0]['id'], 'caption_label')
        self.assertEqual(normalized['labels'][0]['x'], 120)
        self.assertEqual(normalized['labels'][0]['y'], 40)
        self.assertTrue(report['changes'])
        validate_scene(normalized)

    def test_edge_referenced_text_node_and_edges_are_preserved(self):
        original = scene()
        original['nodes'].append(text_node())
        original['labels'] = [label('caption_label')]
        original['edges'] = [{'id': 'link', 'from': 'caption', 'to': 'B', 'arrow': 'end'}]
        normalized, _ = normalize_scene_text(original)
        self.assertIn('caption', {node['id'] for node in normalized['nodes']})
        self.assertEqual(normalized['edges'], original['edges'])
        validate_scene(normalized)

    def test_explicit_path_metadata_reference_also_protects_text_node(self):
        original = scene()
        original['nodes'].append(text_node())
        original['labels'] = [label('caption_label')]
        original['edges'] = [{'id': 'link', 'points': [[170, 40], [450, 40]], 'arrow': 'end',
                              'metadata': {'source_id': 'caption', 'target_id': 'B'}}]
        normalized, _ = normalize_scene_text(original)
        self.assertIn('caption', {node['id'] for node in normalized['nodes']})
        self.assertEqual(normalized['edges'], original['edges'])

    def test_whole_math_delimiters_change_only_text_and_mode(self):
        expression = r'W_E\in\mathbb{R}^{50257\times256}'
        for wrapped in ('$'+expression+'$', r'\('+expression+r'\)'):
            with self.subTest(wrapped=wrapped):
                original = scene()
                original['labels'] = [label('formula', wrapped)]
                normalized, report = normalize_scene_text(original)
                converted = normalized['labels'][0]
                self.assertEqual(converted['mode'], 'math')
                self.assertEqual(converted['text'], expression)
                for key in ('x', 'y', 'font_size', 'anchor', 'align'):
                    self.assertEqual(converted[key], original['labels'][0][key])
                self.assertTrue(report['changes'])
                validate_scene(normalized)

    def test_mixed_chinese_english_math_is_preserved_and_pending(self):
        original = scene()
        original['labels'] = [label('mixed', r'位置嵌入 Position $E_{pos}$')]
        normalized, report = normalize_scene_text(original)
        self.assertEqual(normalized['labels'][0]['text'], original['labels'][0]['text'])
        self.assertEqual(normalized['labels'][0]['mode'], 'text')
        self.assertTrue(any(item.get('reason') == 'mixed_text_math' for item in report['pending_review']))

    def test_illegal_delimited_math_never_becomes_executable_math(self):
        for text in (r'$\input{private}$', r'$E_{bad$'):
            with self.subTest(text=text):
                original = scene()
                original['labels'] = [label('invalid', text)]
                normalized, report = normalize_scene_text(original)
                self.assertEqual(normalized['labels'][0]['text'], text)
                self.assertEqual(normalized['labels'][0]['mode'], 'text')
                self.assertTrue(any(item.get('reason') == 'invalid_math' for item in report['pending_review']))
                validate_scene(normalized)

    def test_repeated_dimension_annotations_remain_at_distinct_positions(self):
        original = scene()
        original['labels'] = [label('batch_left', 'B=8', x=70, y=80, mode='math'),
                              label('batch_middle', 'B=8', x=270, y=80, mode='math'),
                              label('batch_right', 'B=8', x=470, y=80, mode='math')]
        normalized, _ = normalize_scene_text(original)
        self.assertEqual(normalized['labels'], original['labels'])

    def test_formula_characters_and_numeric_dimensions_are_preserved(self):
        expression = r' E_{\mathrm{input}}^{2}+\alpha \in \mathbb{R}^{8\times4\times256} '
        original = scene()
        original['labels'] = [label('formula', '$'+expression+'$')]
        normalized, _ = normalize_scene_text(original)
        self.assertEqual(normalized['labels'][0]['text'], expression)
        self.assertEqual(normalized['labels'][0]['mode'], 'math')

    def test_normalized_scene_is_idempotent_and_original_is_unchanged(self):
        original = scene()
        original['nodes'].append(text_node())
        original['labels'] = [label('caption_label'), label('formula', r'$W_E^{4}$', x=420)]
        snapshot = copy.deepcopy(original)
        once, _ = normalize_scene_text(original)
        twice, _ = normalize_scene_text(once)
        self.assertEqual(twice, once)
        self.assertEqual(original, snapshot)
        validate_scene(once)

    def test_matching_ignores_only_safe_whitespace_and_fullwidth_punctuation(self):
        original = scene()
        original['labels'] = [label('first', '位置嵌入（位置 信息）'),
                              label('second', '位置嵌入 (位置 信息)')]
        normalized, _ = normalize_scene_text(original)
        self.assertEqual(len(normalized['labels']), 1)
        self.assertIn(normalized['labels'][0]['text'], [x['text'] for x in original['labels']])

    def test_equivalent_single_token_math_scripts_match_without_rewriting(self):
        original = scene()
        original['labels'] = [label('first', r'W_E^4', mode='math'),
                              label('second', r'W_{E}^{4}', mode='math')]
        normalized, _ = normalize_scene_text(original)
        self.assertEqual(len(normalized['labels']), 1)
        self.assertIn(normalized['labels'][0]['text'], [r'W_E^4', r'W_{E}^{4}'])

    def test_conflicting_owners_prevent_duplicate_deletion(self):
        original = scene()
        original['labels'] = [label('owner_A', owner='A'), label('owner_B', owner='B')]
        normalized, _ = normalize_scene_text(original)
        self.assertEqual(normalized['labels'], original['labels'])

    def test_reviewed_object_and_existing_geometric_data_are_preserved(self):
        original = scene()
        original['labels'] = [label('reviewed'), label('duplicate')]
        original['labels'][0]['metadata']['review_status'] = 'verified'
        reviewed = copy.deepcopy(original['labels'][0])
        original['edges'] = [{'id': 'link', 'from': 'A', 'to': 'B', 'arrow': 'end'}]
        normalized, _ = normalize_scene_text(original)
        by_id = {item['id']: item for item in normalized['labels']}
        self.assertEqual(by_id['reviewed'], reviewed)
        self.assertEqual(normalized['nodes'], original['nodes'])
        self.assertEqual(normalized['edges'], original['edges'])
        validate_scene(normalized)

    def test_backslash_alone_does_not_make_plain_text_math(self):
        original = scene()
        original['labels'] = [label('literal', r'路径 C:\data 与命令 \alpha')]
        normalized, _ = normalize_scene_text(original)
        self.assertEqual(normalized['labels'][0]['text'], original['labels'][0]['text'])
        self.assertEqual(normalized['labels'][0]['mode'], 'text')

    def test_invalid_input_schema_is_not_silently_repaired(self):
        original = scene()
        original['labels'] = [label('broken')]
        original['labels'][0]['x'] = 1000
        with self.assertRaises(ValueError):
            normalize_scene_text(original)


if __name__ == '__main__':
    unittest.main()
