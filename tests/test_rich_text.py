"""Offline synthetic rich text contract tests; never call a model or compiler.

The real Chinese Scene compilation and PNG inspection are performed separately
once per experiment, rather than repeatedly from the unit suite.
"""
import copy
import json
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from rich_text_parser import RichTextParseError, parse_rich_text
from scene_text_normalizer import normalize_scene_text
from structured_tikz.generator import generate_tikz
from structured_tikz.validator import SceneValidationError, escape_text, validate_scene


def scene():
    return {
        'canvas': {'width': 600, 'height': 300, 'coordinate_system': 'top-left'},
        'styles': {},
        'nodes': [{'id': 'box', 'type': 'rectangle', 'bbox': [400, 100, 100, 80]}],
        'edges': [], 'labels': [],
    }


def label(text='位置嵌入 $E_{pos}$', **overrides):
    item = {'id': 'caption', 'text': text, 'mode': 'text', 'x': 120, 'y': 40,
            'anchor': 'north west', 'font_size': 20, 'align': 'left',
            'style': {'bold': True, 'text_color': 'blue'}}
    item.update(overrides)
    return item


def rich_label(segments, **overrides):
    item = label(mode='rich', segments=segments, **overrides)
    item.pop('text')
    return item


class RichTextParserTests(OfflineTestCase):
    def test_plain_chinese(self):
        self.assertEqual(parse_rich_text('位置嵌入（可训练）'),
                         [{'type': 'text', 'content': '位置嵌入（可训练）'}])

    def test_plain_english_and_ordinary_backslashes(self):
        text = r'Position embedding C:\data\file and \alpha'
        self.assertEqual(parse_rich_text(text), [{'type': 'text', 'content': text}])

    def test_chinese_with_formula(self):
        expression = r'E_{pos}\in\mathbb{R}^{4\times256}'
        self.assertEqual(parse_rich_text('位置嵌入 $' + expression + '$'), [
            {'type': 'text', 'content': '位置嵌入 '},
            {'type': 'math', 'content': expression},
        ])

    def test_english_with_parenthesis_delimited_formula(self):
        self.assertEqual(parse_rich_text(r'Embedding \(E_{pos}\).'), [
            {'type': 'text', 'content': 'Embedding '},
            {'type': 'math', 'content': 'E_{pos}'},
            {'type': 'text', 'content': '.'},
        ])

    def test_multiple_math_fragments_keep_order_and_newlines(self):
        self.assertEqual(parse_rich_text('词元 $W_E$\n位置 ' + r'\(W_P\)'), [
            {'type': 'text', 'content': '词元 '},
            {'type': 'math', 'content': 'W_E'},
            {'type': 'text', 'content': '\n位置 '},
            {'type': 'math', 'content': 'W_P'},
        ])

    def test_escaped_dollar_is_literal_not_a_delimiter(self):
        self.assertEqual(parse_rich_text(r'费用 \$5，嵌入 $E_{pos}$'), [
            {'type': 'text', 'content': '费用 $5，嵌入 '},
            {'type': 'math', 'content': 'E_{pos}'},
        ])

    def test_unclosed_or_unmatched_delimiters_are_rejected(self):
        for text in ('说明 $x', r'说明 \(x', r'说明 x\)'):
            with self.subTest(text=text), self.assertRaises(RichTextParseError):
                parse_rich_text(text)

    def test_empty_display_or_nested_math_is_rejected(self):
        for text in ('说明 $$', '说明 $ $', r'说明 \(\)', '$$x$$',
                     r'$x \(y\)$', r'\(x $y$\)'):
            with self.subTest(text=text), self.assertRaises(RichTextParseError):
                parse_rich_text(text)

    def test_math_commands_and_braces_use_existing_strict_validation(self):
        for text in (r'标签 $\input{secret}$', r'标签 $\write18{bad}$',
                     r'标签 $E_{broken$', r'标签 $x%comment$'):
            with self.subTest(text=text), self.assertRaises(RichTextParseError):
                parse_rich_text(text)

    def test_formula_characters_numbers_and_internal_spaces_are_unchanged(self):
        expression = r' E_{\mathrm{input}}^{2}+\alpha \in \mathbb{R}^{8\times4\times256} '
        parsed = parse_rich_text('输入 $' + expression + '$')
        self.assertEqual(parsed[1], {'type': 'math', 'content': expression})
        self.assertEqual(parse_rich_text(r'矩阵 $W_E\in\mathbb{R}^{50257\times256}$')[1]['content'],
                         r'W_E\in\mathbb{R}^{50257\times256}')

    def test_json_backslash_roundtrip_does_not_double_escape_math(self):
        value = {'text': r'位置嵌入 $E_{pos}\in\mathbb{R}^{4\times256}$'}
        decoded = json.loads(json.dumps(value, ensure_ascii=False))
        self.assertEqual(parse_rich_text(decoded['text']), parse_rich_text(value['text']))
        self.assertEqual(parse_rich_text(decoded['text'])[1]['content'],
                         r'E_{pos}\in\mathbb{R}^{4\times256}')


class RichTextNormalizationTests(OfflineTestCase):
    def test_normalization_is_opt_in_and_old_scene_remains_compatible(self):
        original = scene()
        original['labels'] = [label()]
        old_tikz = generate_tikz(original)
        result, _ = normalize_scene_text(original)
        self.assertEqual(result, original)
        self.assertEqual(generate_tikz(result), old_tikz)
        self.assertIn(escape_text(original['labels'][0]['text']), old_tikz)

    def test_rich_conversion_retains_original_text_and_layout_and_audit(self):
        original = scene()
        source = '位置嵌入 Position\n' + r'$E_{pos}\in\mathbb{R}^{4\times256}$'
        original['labels'] = [label(source)]
        snapshot = copy.deepcopy(original)
        result, report = normalize_scene_text(original, enable_rich_text=True)
        converted = result['labels'][0]
        self.assertEqual(converted['mode'], 'rich')
        self.assertEqual(converted['text'], source)
        self.assertEqual(converted['segments'], parse_rich_text(source))
        for key, value in original['labels'][0].items():
            if key != 'mode':
                self.assertEqual(converted[key], value)
        self.assertEqual(result['nodes'], original['nodes'])
        self.assertEqual(result['edges'], original['edges'])
        self.assertEqual(original, snapshot)
        self.assertTrue(report['changes'])
        self.assertIn('caption', json.dumps(report, ensure_ascii=False))
        validate_scene(result)

    def test_plain_chinese_and_english_are_not_changed(self):
        for text in ('中文说明', 'Position embedding', r'路径 C:\data 与 \alpha'):
            with self.subTest(text=text):
                original = scene()
                original['labels'] = [label(text)]
                result, _ = normalize_scene_text(original, enable_rich_text=True)
                self.assertEqual(result, original)

    def test_pure_math_uses_math_mode_and_existing_math_is_unchanged(self):
        expression = r'W_E\in\mathbb{R}^{50257\times256}'
        for text, mode in (('$' + expression + '$', 'text'),
                           (r'\(' + expression + r'\)', 'text'), (expression, 'math')):
            with self.subTest(mode=mode, text=text):
                original = scene()
                original['labels'] = [label(text, mode=mode)]
                result, _ = normalize_scene_text(original, enable_rich_text=True)
                self.assertEqual(result['labels'][0]['mode'], 'math')
                self.assertEqual(result['labels'][0]['text'], expression)
                self.assertNotIn('segments', result['labels'][0])

    def test_invalid_mixed_content_is_preserved_with_reason(self):
        for text in (r'位置 $\input{secret}$', '位置 $E_{pos}', r'位置 \(\)'):
            with self.subTest(text=text):
                original = scene()
                original['labels'] = [label(text)]
                result, report = normalize_scene_text(original, enable_rich_text=True)
                self.assertEqual(result['labels'], original['labels'])
                self.assertTrue(report['pending_review'])
                self.assertTrue(any(entry.get('reason') for entry in report['pending_review']))
                validate_scene(result)

    def test_rich_normalization_is_idempotent_and_keeps_references(self):
        original = scene()
        original['nodes'].append({'id': 'caption_node', 'type': 'text', 'bbox': [40, 30, 180, 60],
                                  'text': r'词元 $W_E$', 'mode': 'text'})
        original['edges'] = [{'id': 'arrow', 'from': 'caption_node', 'to': 'box', 'arrow': 'end'}]
        original['labels'] = [label(r'位置 $W_P$', x=350)]
        snapshot = copy.deepcopy(original)
        once, _ = normalize_scene_text(original, enable_rich_text=True)
        twice, _ = normalize_scene_text(once, enable_rich_text=True)
        self.assertEqual(once, twice)
        self.assertEqual(once['edges'], original['edges'])
        self.assertEqual([n['bbox'] for n in once['nodes']], [n['bbox'] for n in original['nodes']])
        self.assertEqual([n['id'] for n in once['nodes']], [n['id'] for n in original['nodes']])
        self.assertEqual(original, snapshot)
        validate_scene(once)


class RichTextSchemaAndRenderingTests(OfflineTestCase):
    def test_rich_label_without_legacy_text_renders_in_one_node(self):
        original = scene()
        expression = r'E_{pos}\in\mathbb{R}^{4\times256}'
        original['labels'] = [rich_label([
            {'type': 'text', 'content': '位置嵌入 '},
            {'type': 'math', 'content': expression},
        ])]
        validate_scene(original)
        fragment = generate_tikz(original)
        self.assertEqual(fragment.count('(l-caption)'), 1)
        self.assertIn('位置嵌入 $' + expression + '$', fragment)
        self.assertNotIn(r'\textbackslash{}in', fragment)
        self.assertNotIn(r'\$E', fragment)
        self.assertIn('anchor=north west,align=left', fragment)
        self.assertIn(r'\fontsize{8pt}{10pt}\selectfont\bfseries', fragment)
        self.assertIn('at (48pt,104pt)', fragment)

    def test_rich_text_node_segments_reach_renderer_without_legacy_text(self):
        original = scene()
        original['nodes'].append({'id': 'caption_node', 'type': 'text', 'bbox': [40, 30, 180, 60],
                                  'mode': 'rich', 'segments': [
                                      {'type': 'text', 'content': '词元 '},
                                      {'type': 'math', 'content': 'W_E'},
                                  ]})
        validate_scene(original)
        fragment = generate_tikz(original)
        self.assertEqual(fragment.count('(t-caption_node)'), 1)
        self.assertIn('词元 $W_E$', fragment)
        self.assertIn('text width=72pt', fragment)

    def test_text_segments_keep_escaping_and_newlines(self):
        original = scene()
        prose = '费用 $5 & 50%\n' + r'路径 C:\data \input{secret} '
        original['labels'] = [rich_label([
            {'type': 'text', 'content': prose}, {'type': 'math', 'content': 'x_1'},
        ])]
        fragment = generate_tikz(original)
        self.assertIn(escape_text(prose) + '$x_1$', fragment)
        self.assertNotIn(r'\input{secret}', fragment)
        self.assertIn(r'\textbackslash{}input\{secret\}', fragment)

    def test_schema_rejects_unsafe_math_even_if_parser_is_bypassed(self):
        original = scene()
        original['labels'] = [rich_label([
            {'type': 'text', 'content': '说明 '},
            {'type': 'math', 'content': r'\input{secret}'},
        ])]
        with self.assertRaises(SceneValidationError):
            validate_scene(original)
        with self.assertRaises(SceneValidationError):
            generate_tikz(original)

    def test_schema_rejects_malformed_segments(self):
        cases = [None, [], 'x', [{'type': 'raw', 'content': 'x'}],
                 [{'type': 'math', 'content': ''}],
                 [{'type': 'math', 'content': '$x$'}],
                 [{'type': 'text', 'content': 'x', 'raw_tex': True}],
                 [{'type': 'text', 'content': 123}],
                 [{'type': 'text'}]]
        for segments in cases:
            with self.subTest(segments=segments):
                original = scene()
                original['labels'] = [rich_label(segments)]
                with self.assertRaises(SceneValidationError):
                    validate_scene(original)

    def test_segments_cannot_silently_override_legacy_mode(self):
        for mode in ('text', 'math'):
            with self.subTest(mode=mode):
                original = scene()
                original['labels'] = [label('x', mode=mode, segments=[{'type': 'math', 'content': 'y'}])]
                with self.assertRaises(SceneValidationError):
                    validate_scene(original)


if __name__ == '__main__':
    unittest.main()
