"""Strict outer-wrapper recovery; no API, compiler, or historical fixture reads."""
import json
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from auto_scene_generator import _strict_json, parse_scene_response
from auto_scene_validator import diagnose_scene
from structured_tikz.generator import generate_tikz
from vlm_client import APIResponseError, parse_response


def scene_fixture():
    return {'canvas': {'width': 240, 'height': 120, 'coordinate_system': 'top-left'},
            'styles': {},
            'nodes': [{'id': 'block', 'type': 'matrix', 'bbox': [20, 40, 80, 50],
                       'grid': {'rows': 2, 'cols': 3}}],
            'edges': [],
            'labels': [{'id': 'caption', 'text': '离线测试', 'mode': 'text',
                        'x': 60, 'y': 20, 'anchor': 'center', 'font_size': 12}],
            'metadata': {'review_status': 'unverified'}}


class JsonRecoveryTests(OfflineTestCase):
    def test_plain_json_object_retains_content(self):
        scene = scene_fixture()
        self.assertEqual(_strict_json(json.dumps(scene)), scene)
        parsed = parse_scene_response(json.dumps(scene, ensure_ascii=False))
        self.assertEqual(parsed['labels'][0]['text'], '离线测试')
        self.assertTrue(diagnose_scene(parsed)['schema_valid'])

    def test_one_complete_json_fence_recovers_only_its_wrapper(self):
        scene = scene_fixture()
        raw = json.dumps(scene, ensure_ascii=False, indent=2)
        for language, newline in [('json', '\n'), ('JSON', '\r\n')]:
            fenced = ' \n```'+language+newline+raw+newline+'```\n '
            self.assertEqual(_strict_json(fenced), scene)
            self.assertTrue(diagnose_scene(parse_scene_response(fenced))['schema_valid'])

    def test_explanations_outside_fence_or_object_are_rejected(self):
        raw = json.dumps(scene_fixture())
        fenced = '```json\n'+raw+'\n```'
        for text in ('Here is the scene:\n'+fenced, fenced+'\nDone.',
                     'Result: '+raw, raw+'\nExplanation', '<json>'+raw+'</json>'):
            with self.subTest(text=text[:30]), self.assertRaises(ValueError):
                _strict_json(text)

    def test_multiple_objects_and_multiple_fences_are_rejected(self):
        for text in ('{} {}', '{}\n{}', '```json\n{}\n{}\n```',
                     '```json\n{}\n```\n```json\n{}\n```',
                     '```json\n{}\n```\n{}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                _strict_json(text)

    def test_incomplete_json_or_fences_are_rejected(self):
        for text in ('{"nodes": [', '```json\n{"nodes": [\n```',
                     '```json\n{}', '{}\n```', '```json\n{}\n``',
                     '```json\n{\n```', '```json\n{"a":1,}\n```'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                _strict_json(text)

    def test_unlabelled_and_other_language_fences_are_not_accepted(self):
        for text in ('```\n{}\n```', '```python\n{}\n```', '~~~json\n{}\n~~~',
                     '```json {} ```', '```json extra\n{}\n```'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                _strict_json(text)

    def test_duplicate_keys_and_nonfinite_numbers_remain_rejected(self):
        for body in ('{"a":1,"a":2}', '{"nested":{"a":1,"a":2}}',
                     '{"a":NaN}', '{"a":Infinity}', '{"a":-Infinity}', '{"a":1e999}'):
            for text in (body, '```json\n'+body+'\n```'):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    _strict_json(text)

    def test_backticks_inside_json_strings_are_data_not_extra_wrappers(self):
        data = {'note': 'Literal ```json and ``` are not delimiter lines.'}
        raw = json.dumps(data)
        self.assertEqual(_strict_json(raw), data)
        self.assertEqual(_strict_json('```json\n'+raw+'\n```'), data)
        with self.assertRaises(ValueError):
            _strict_json('```json\n[{}]\n```')

    def test_wrapping_does_not_bypass_schema_unknown_fields_or_illegal_math(self):
        cases = []
        scene = scene_fixture(); scene['arbitrary_shell'] = 'do not execute'; cases.append(scene)
        scene = scene_fixture(); scene['nodes'][0]['command'] = 'not supported'; cases.append(scene)
        scene = scene_fixture(); scene['nodes'][0]['grid'] = None; cases.append(scene)
        scene = scene_fixture(); scene['labels'][0].update(mode='math', text=r'\input{secret}'); cases.append(scene)
        scene = scene_fixture(); scene['nodes'][0]['bbox'] = [200, 40, 80, 50]; cases.append(scene)
        for scene in cases:
            parsed = parse_scene_response('```json\n'+json.dumps(scene)+'\n```')
            diagnosis = diagnose_scene(parsed)
            self.assertFalse(diagnosis['schema_valid'])
            self.assertTrue(diagnosis['errors'])
            with self.assertRaises(ValueError):
                generate_tikz(parsed)

    def test_length_finish_reason_rejects_even_complete_parseable_payload(self):
        raw = json.dumps(scene_fixture())
        for content in (raw, '```json\n'+raw+'\n```'):
            body = {'choices': [{'finish_reason': 'length', 'message': {'content': content}}]}
            with self.assertRaisesRegex(APIResponseError, 'finish_reason=length'):
                parse_response(body)
        body['choices'][0]['finish_reason'] = 'stop'
        self.assertEqual(_strict_json(parse_response(body)), scene_fixture())


if __name__ == '__main__':
    unittest.main()
