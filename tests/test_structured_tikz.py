"""Deterministic scene tests. No network, model, or real TeX subprocess here."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from latex_template import wrap_document
from structured_tikz.generator import generate_tikz, load_scene
from structured_tikz.primitives import point
from structured_tikz.validator import SceneValidationError, validate_scene


def scene(kind='rectangle'):
    node = {'id': 'a', 'type': kind, 'bbox': [20, 40, 100, 60],
            'style': {'fill': 'light_blue', 'stroke': 'dark_blue'}}
    if kind in ('matrix', 'tensor3d'):
        node['grid'] = {'rows': 3, 'cols': 4}
    if kind == 'tensor3d':
        node['depth'] = {'dx': 15, 'dy': -10}
    return {'canvas': {'width': 400, 'height': 200, 'coordinate_system': 'top-left', 'scale': .4},
            'styles': {}, 'nodes': [node], 'edges': [], 'labels': []}


class StructuredTikzTests(OfflineTestCase):
    def test_pixel_mapping_preserves_scale_and_inverts_y_once(self):
        canvas = scene()['canvas']
        self.assertEqual(point(0, 0, canvas), '(0pt,80pt)')
        self.assertEqual(point(100, 50, canvas), '(40pt,60pt)')
        self.assertEqual(point(100, 200, canvas), '(40pt,0pt)')

    def test_rectangle_has_editable_node_and_exact_dimensions(self):
        code = generate_tikz(scene())
        self.assertIn('(n-a)', code)
        self.assertIn('minimum width=40pt,minimum height=24pt', code)
        self.assertIn('fill=scenelightblue', code)
        self.assertNotIn(r'\includegraphics', code)

    def test_rounded_rectangle(self):
        self.assertIn('rounded corners=', generate_tikz(scene('rounded_rectangle')))

    def test_matrix_grid_has_rows_minus_one_and_cols_minus_one_lines(self):
        self.assertEqual(generate_tikz(scene('matrix')).count(r'\draw['), 5)

    def test_tensor_has_three_faces_and_projected_grid(self):
        code = generate_tikz(scene('tensor3d'))
        for name in ('front', 'top', 'right'):
            self.assertIn('% face: ' + name, code)
        self.assertEqual(code.count(r'\path['), 3)
        self.assertEqual(code.count(r'\draw['), 10)
        self.assertIn('(8pt,64pt) -- (14pt,68pt)', code)  # Positive depth goes right AND up in TikZ.

    def test_chinese_plain_text_escapes_tex_and_preserves_newlines(self):
        data = scene('text')
        data['nodes'][0]['text'] = '中文 & 50%_#{} $ ~ ^ \\input\n第二行'
        code = generate_tikz(data)
        self.assertIn('中文', code)
        self.assertIn(r'\& 50\%\_\#\{\}', code)
        self.assertIn(r'\textbackslash{}input', code)
        self.assertIn('(t-a)', code)
        self.assertNotIn(r'\input', code)
        wrap_document(code)

    def test_math_commands_and_numbers_are_preserved_verbatim(self):
        data = scene()
        formula = r'E_{\mathrm{input}}\in\mathbb{R}^{8\times4\times256}'
        data['labels'].append({'id': 'formula', 'text': formula, 'x': 200, 'y': 150, 'mode': 'math'})
        code = generate_tikz(data)
        self.assertIn('$' + formula + '$', code)
        self.assertIn('(l-formula)', code)

    def test_named_node_anchors_and_explicit_arrow_direction(self):
        data = scene()
        data['nodes'].append({'id': 'b', 'type': 'rectangle', 'bbox': [250, 80, 80, 50]})
        data['edges'] = [{'id': 'join', 'from': 'a', 'to': 'b', 'from_anchor': 'south',
                          'to_anchor': 'north', 'routing': 'orthogonal', 'first_axis': 'y',
                          'arrow': 'end', 'dashed': True}]
        code = generate_tikz(data)
        self.assertIn('(n-a.south) |- (n-b.north)', code)
        self.assertIn('->,dashed', code)

    def test_explicit_polyline_points_and_no_implicit_arrow(self):
        data = scene()
        data['edges'] = [{'id': 'join', 'points': [[120, 70], [170, 70], [170, 130]]}]
        code = generate_tikz(data)
        self.assertIn('(48pt,52pt) -- (68pt,52pt) -- (68pt,28pt)', code)
        self.assertNotIn('->', code)

    def test_labels_are_last_and_output_is_deterministic_without_mutation(self):
        data = scene('tensor3d')
        data['nodes'][0]['text'] = '中文'
        data['labels'].append({'id': 'title', 'text': '标题', 'x': 200, 'y': 15})
        original = copy.deepcopy(data)
        first = generate_tikz(data)
        second = generate_tikz(json.loads(json.dumps(data, sort_keys=True)))
        self.assertEqual(first, second)
        self.assertEqual(data, original)
        self.assertLess(first.index('% face: front'), first.index('(t-a)'))
        self.assertLess(first.index('(t-a)'), first.index('(l-title)'))

    def test_styles_only_allow_named_safe_values(self):
        data = scene()
        data['styles']['module'] = {'fill': 'light_blue', 'stroke': 'blue'}
        data['nodes'][0]['style'] = 'module'
        self.assertIn('draw=sceneblue', generate_tikz(data))
        for style in ({'fill': 'red,execute at begin node=bad'}, {'unknown': 'value'}, 'missing'):
            data['nodes'][0]['style'] = style
            with self.subTest(style=style), self.assertRaises(SceneValidationError):
                generate_tikz(data)

    def test_no_stroke_preserves_fill_for_named_and_inline_styles(self):
        for named in (False, True):
            data = scene()
            style = {'fill': 'light_blue', 'stroke': 'none'}
            data['styles']['borderless'] = style
            data['nodes'][0]['style'] = 'borderless' if named else style
            original = copy.deepcopy(data)
            code = generate_tikz(data)
            drawing = next(line for line in code.splitlines() if '(n-a)' in line)
            self.assertIn('draw=none', drawing)
            self.assertIn('fill=scenelightblue', drawing)
            self.assertEqual(data, original)

    def test_no_stroke_does_not_allow_invisible_text(self):
        data = scene()
        data['nodes'][0]['style']['text_color'] = 'none'
        with self.assertRaises(SceneValidationError):
            validate_scene(data)

    def test_no_stroke_does_not_allow_arbitrary_tikz_options(self):
        for value in ('none,execute at begin node=bad', 'transparent', 'red'):
            data = scene()
            data['nodes'][0]['style']['stroke'] = value
            with self.subTest(value=value), self.assertRaises(SceneValidationError):
                generate_tikz(data)

    def test_invalid_schema_ids_numbers_grid_references_and_anchors(self):
        invalid = []
        item = scene(); item['nodes'].append(copy.deepcopy(item['nodes'][0])); invalid.append(item)
        item = scene(); item['unknown'] = 'ignored?'; invalid.append(item)
        item = scene(); item['nodes'][0]['type'] = 'raster'; invalid.append(item)
        item = scene(); item['nodes'][0]['bbox'][2] = 0; invalid.append(item)
        item = scene(); item['nodes'][0]['bbox'][0] = float('nan'); invalid.append(item)
        item = scene(); item['nodes'][0]['bbox'][1] = True; invalid.append(item)
        item = scene('matrix'); item['nodes'][0]['grid']['rows'] = 0; invalid.append(item)
        item = scene('matrix'); item['nodes'][0]['grid']['cols'] = 3.5; invalid.append(item)
        item = scene('tensor3d'); item['nodes'][0]['depth']['dy'] = 10; invalid.append(item)
        item = scene(); item['edges'] = [{'id':'e', 'from':'a', 'to':'missing'}]; invalid.append(item)
        item = scene(); item['edges'] = [{'id':'e', 'from':'a', 'to':'a', 'from_anchor':'random'}]; invalid.append(item)
        item = scene(); item['labels'] = [{'id':'l', 'text':'hi', 'x':10, 'y':10, 'font_size':float('inf')}]; invalid.append(item)
        for index, data in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(SceneValidationError):
                validate_scene(data)

    def test_math_rejects_command_injection_and_bad_braces(self):
        for formula in (r'\input{x}', r'\write18{bad}', r'\csname input\endcsname', r'^^5cinput{x}',
                        r'\text{\input{x}}', 'x{', '$x$', 'x%comment', r'\begin{matrix}x\end{matrix}'):
            data = scene()
            data['labels'] = [{'id':'l', 'text':formula, 'mode':'math', 'x':10, 'y':10}]
            with self.subTest(formula=formula), self.assertRaises(SceneValidationError):
                generate_tikz(data)

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'outputs') as directory:
            path = Path(directory) / 'bad.json'
            path.write_text('{"canvas":{},"canvas":{}}', encoding='utf-8')
            with self.assertRaises(SceneValidationError):
                load_scene(path)

    def test_manual_scene_and_fixed_xelatex_template_compatibility(self):
        path = Path(__file__).resolve().parents[1] / 'examples' / 'gpt_embedding_scene.json'
        data, _ = load_scene(path)
        code = generate_tikz(data)
        document = wrap_document(code)
        self.assertEqual(document.count(r'\documentclass'), 1)
        self.assertIn('fontset=fandol', document)
        self.assertIn('50257', document)
        self.assertIn('256', document)
        self.assertEqual(sum(node['type'] == 'tensor3d' for node in data['nodes']), 4)


if __name__ == '__main__':
    unittest.main()
