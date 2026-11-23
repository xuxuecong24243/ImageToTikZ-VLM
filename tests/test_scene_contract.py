"""Offline compatibility checks for the exported drawing grammar and model prompts."""
import copy
import json
import re
from pathlib import Path
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from auto_scene_generator import build_scene_prompt, build_patch_prompt, _strict_json, SceneJSONError
from structured_tikz.schema import (scene_contract, OBJECT_CONTRACTS, NODE_TYPES, TEXT_MODES,
                                    STYLE_DEFAULTS, MATH_COMMANDS)
from structured_tikz.validator import validate_scene
from structured_tikz.generator import generate_tikz


def scene():
    return {'canvas':{'width':300,'height':200,'coordinate_system':'top-left'},
            'styles':{},'nodes':[{'id':'A','type':'rectangle','bbox':[10,20,50,40]},
                               {'id':'B','type':'text','bbox':[200,20,50,40],'text':'中文 a_b[0]'}],
            'edges':[{'id':'link','from':'A','to':'B','arrow':'end'}],
            'labels':[{'id':'caption','text':'标题','x':130,'y':100}]}


def prompt_contract(prompt):
    return json.loads(prompt.split('SCENE_CONTRACT_BEGIN\n',1)[1].split('\nSCENE_CONTRACT_END',1)[0])


class SceneContractTests(OfflineTestCase):
    def test_root_required_optional_and_shared_fields(self):
        contract=scene_contract()['json_schema']['$defs']
        self.assertEqual(contract,OBJECT_CONTRACTS)
        self.assertEqual(set(contract['scene']['required']),{'canvas','styles','nodes','edges','labels'})
        self.assertEqual(set(contract['scene']['properties'])-set(contract['scene']['required']),{'metadata'})
        self.assertNotIn('type',contract['scene']['properties'])

    def test_full_generated_contract_is_in_initial_and_patch_prompt(self):
        self.assertEqual(prompt_contract(build_scene_prompt((300,200))),scene_contract((300,200)))
        self.assertEqual(prompt_contract(build_patch_prompt(scene(),{})),scene_contract())
        self.assertIn('NOT an example',build_scene_prompt((300,200)))
        self.assertIn('no Markdown code fences',build_scene_prompt((300,200)))
        self.assertIn('unsupported_elements',build_scene_prompt((300,200)))

    def test_enums_and_defaults_are_current_not_an_independent_prompt_list(self):
        contract=scene_contract(); defs=contract['json_schema']['$defs']
        self.assertEqual(set(defs['node']['properties']['type']['enum']),NODE_TYPES)
        self.assertEqual(set(defs['node']['properties']['mode']['enum']),TEXT_MODES)
        self.assertEqual(contract['style_defaults'],STYLE_DEFAULTS)
        self.assertEqual(set(contract['math_commands']),MATH_COMMANDS)
        self.assertIn('wrap',defs['node']['properties'])
        self.assertIn('background_shape',defs['node']['properties'])
        self.assertIn('none',defs['style']['properties']['stroke']['enum'])

    def test_each_declared_native_node_can_validate_and_render(self):
        for kind in NODE_TYPES:
            with self.subTest(kind=kind):
                data=scene(); node=data['nodes'][0]; node['type']=kind
                if kind=='text':node['text']='可编辑'
                if kind in {'matrix','tensor3d'}:node['grid']={'rows':2,'cols':3}
                if kind=='tensor3d':node['depth']={'dx':5,'dy':-5}
                validate_scene(data)
                self.assertIn('% node: A ('+kind+')',generate_tikz(data))

    def test_rich_circle_wrap_optional_defaults_and_literal_code_render(self):
        data=scene(); node=data['nodes'][1]
        node.update(bbox=[200,20,40,40],mode='rich',segments=[{'type':'text','content':'中文_'},
            {'type':'math','content':r'x_1'}],background_shape='circle',wrap=False)
        data['labels'][0]['style']={'stroke':'none'}
        text=generate_tikz(data)
        self.assertIn('circle,minimum size=',text)
        self.assertIn(r'中文\_',text)
        self.assertIn('$x_1$',text)
        self.assertNotIn('a_b[0]',text)  # rich text provenance must not render a second time

    def test_unknown_fields_rejected_at_all_object_levels(self):
        paths=[(),('canvas',),('nodes',0),('edges',0),('labels',0)]
        for path in paths:
            with self.subTest(path=path):
                data=scene(); target=data
                for key in path:target=target[key]
                target['invented_field']=1
                with self.assertRaisesRegex(ValueError,'unknown fields'):validate_scene(data)
        data=scene();data['styles']['new']={'invented_field':1}
        with self.assertRaisesRegex(ValueError,'unknown fields'):validate_scene(data)

    def test_conditional_features_reject_incompatible_shapes(self):
        additions=[{'grid':{'rows':2,'cols':2}}, {'depth':{'dx':2,'dy':-2}}, {'wrap':False},
                   {'background_shape':'circle'}, {'segments':[{'type':'text','content':'x'}]}]
        for addition in additions:
            with self.subTest(addition=addition):
                data=scene();data['nodes'][0].update(addition)
                with self.assertRaises(ValueError):validate_scene(data)

    def test_references_duplicate_ids_and_unsupported_types_remain_errors(self):
        for mutation in ('style','node_reference','duplicate','polygon'):
            with self.subTest(mutation=mutation):
                data=scene()
                if mutation=='style':data['nodes'][0]['style']='unknown'
                if mutation=='node_reference':data['edges'][0]['to']='caption'
                if mutation=='duplicate':data['labels'][0]['id']='A'
                if mutation=='polygon':data['nodes'][0]['type']='polygon'
                with self.assertRaises(ValueError):generate_tikz(data)

    def test_id_pattern_rejects_trailing_newline_and_grid_float_requires_runtime_check(self):
        rule=scene_contract()['json_schema']['$defs']['node']['properties']['id']
        self.assertIsNone(re.search(rule['pattern'],'A\n'))
        data=scene();data['nodes'][0]['id']='A\n'
        with self.assertRaises(ValueError):validate_scene(data)
        data=scene();data['nodes'][0].update(type='matrix',grid={'rows':1.0,'cols':2})
        with self.assertRaises(ValueError):validate_scene(data)
        self.assertTrue(any('Python int' in item for item in scene_contract()['runtime_constraints']))

    def test_legacy_scene_valid_and_unchanged(self):
        path=Path(__file__).resolve().parents[1]/'examples/gpt_embedding_scene.json'
        data=json.loads(path.read_text(encoding='utf-8')); before=copy.deepcopy(data)
        validate_scene(data)
        self.assertEqual(generate_tikz(data),generate_tikz(before))
        self.assertEqual(data,before)

    def test_export_is_independent_and_never_hardcodes_test_image(self):
        contract=scene_contract((300,200));contract['json_schema']['$defs']['node']['properties'].clear()
        self.assertIn('bbox',scene_contract()['json_schema']['$defs']['node']['properties'])
        prompt=build_scene_prompt((301,201),'fallible hint')
        self.assertNotIn('SpamDataset',prompt)
        self.assertNotIn('page_01',prompt)
        self.assertEqual(prompt_contract(prompt)['json_schema']['$defs']['canvas']['properties']['width']['const'],301)

    def test_parse_errors_are_classified_without_silent_repair(self):
        cases=[('{"nodes":','invalid_json'),('[]','root_not_object'),
               ('{"a":1,"a":2}','duplicate_json_key'),('{"v":NaN}','nonfinite_number'),
               ('{"v":1e999}','nonfinite_number'),('```text\n{}\n```','response_wrapper')]
        for raw,code in cases:
            with self.subTest(raw=raw),self.assertRaises(SceneJSONError) as caught:_strict_json(raw)
            self.assertEqual(caught.exception.diagnostic['code'],code)

    def test_legacy_single_json_fence_is_normalized_not_truncated(self):
        original=scene();raw=json.dumps(original,ensure_ascii=False)
        self.assertEqual(_strict_json('```json\n'+raw+'\n```'),original)
        with self.assertRaises(SceneJSONError):_strict_json(raw+' trailing text')


if __name__=='__main__':unittest.main()
