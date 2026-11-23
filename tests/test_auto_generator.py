"""Synthetic interfaces only; these tests do not call an API or read references."""
import copy
import json
from pathlib import Path
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from unittest.mock import Mock

from auto_scene_generator import (apply_scene_patch, build_patch_prompt,
    build_scene_prompt, generate_scene, parse_patch_response, parse_scene_response,
    request_patch)
from vlm_client import APIResponseError, ApiNotAuthorized


def sample_scene():
    return {'canvas':{'width':300,'height':200,'coordinate_system':'top-left'},
            'styles':{}, 'nodes':[{'id':'A','type':'rectangle','bbox':[20,60,50,40]},
                                  {'id':'B','type':'rectangle','bbox':[200,60,50,40]}],
            'labels':[{'id':'caption','text':'合成测试','mode':'text','x':45,'y':40,
                       'metadata':{'node_id':'A','bbox':[20,30,50,20]}}], 'edges':[],
            'metadata':{'review_status':'unverified'}}


def operation(op, identifier, **kwargs):
    return {'op':op,'id':identifier,'reason':'synthetic test',**kwargs}


class AutoSceneGeneratorTests(OfflineTestCase):
    def test_original_image_and_full_scene_contract(self):
        client=Mock(); client.complete_images.return_value=json.dumps(sample_scene())
        image=Path('synthetic.png')
        scene,raw=generate_scene(client,image,(300,200),'candidate hint')
        args=client.complete_images.call_args.args
        self.assertEqual(args[0],[image])
        self.assertIn('FRONT FACE',args[1]);self.assertIn('candidate hint',args[1])
        self.assertIn('width:300,height:200',args[1])
        self.assertEqual(scene['labels'][0]['text'],'合成测试')
        self.assertEqual(json.loads(raw)['nodes'],sample_scene()['nodes'])

    def test_strict_json_no_content_repair_or_surrounding_prose(self):
        for text in ('[]','```markdown\n{}\n```','{"a":1,"a":2}','{"x":NaN}','{"x":1e999}','{} trailing'):
            with self.subTest(text=text),self.assertRaises(ValueError):parse_scene_response(text)

    def test_incomplete_geometry_preserved_for_diagnostics_and_model_verified_distrusted(self):
        scene=sample_scene();scene['metadata']['review_status']='verified'
        scene['nodes'][0].update(type='tensor3d',grid=None,depth=None,
                                metadata={'review_status':'verified','source':'measured'})
        parsed=parse_scene_response(json.dumps(scene))
        self.assertIsNone(parsed['nodes'][0]['grid']);self.assertIsNone(parsed['nodes'][0]['depth'])
        self.assertEqual(parsed['nodes'][0]['metadata']['source'],'vlm_proposal_not_measured')
        self.assertEqual(parsed['metadata']['review_status'],'unverified')

    def test_no_retry_of_unauthorized_or_truncated_client_response(self):
        for error in (ApiNotAuthorized('offline'),APIResponseError('length')):
            client=Mock();client.complete_images.side_effect=error
            with self.assertRaises(type(error)):generate_scene(client,'original.png',(300,200))
            self.assertEqual(client.complete_images.call_count,1)

    def test_bounded_geometry_and_image_dimensions(self):
        with self.assertRaises(ValueError):build_scene_prompt((300,200),'x'*6001)
        with self.assertRaises(ValueError):build_scene_prompt((0,200))

    def test_atomic_geometry_completion_without_input_mutation(self):
        scene=sample_scene();scene['nodes'][0].update(type='tensor3d',grid=None,depth=None)
        before=copy.deepcopy(scene)
        value={'grid':{'rows':2,'cols':2},'depth':{'dx':10,'dy':-10}}
        after,changes=apply_scene_patch(scene,{'operations':[operation('set_node_geometry','A',value=value)]})
        self.assertEqual(scene,before);self.assertEqual(after['nodes'][0]['depth'],value['depth'])
        self.assertEqual(changes[0]['before']['depth'],None)

    def test_invalid_atomic_patch_does_not_apply_first_valid_movement(self):
        scene=sample_scene();before=copy.deepcopy(scene)
        ops=[operation('move_label','caption',dx=10,dy=0),operation('move_label','caption',dx=1000,dy=0)]
        with self.assertRaises(ValueError):apply_scene_patch(scene,{'operations':ops})
        self.assertEqual(scene,before)

    def test_protected_objects_and_canvas_cannot_be_changed(self):
        for state in ('verified','rejected'):
            scene=sample_scene();scene['labels'][0]['metadata']['review_status']=state
            with self.assertRaises(ValueError):apply_scene_patch(scene,{'operations':[operation('move_label','caption',dx=1,dy=0)]})
        with self.assertRaises(ValueError):apply_scene_patch(sample_scene(),{'operations':[operation('set_canvas','canvas',value={})]})

    def test_move_preserves_content_and_moves_proposed_bbox(self):
        after,_=apply_scene_patch(sample_scene(),{'operations':[operation('move_label','caption',dx=5,dy=-5)]})
        self.assertEqual(after['labels'][0]['text'],'合成测试')
        self.assertEqual(after['labels'][0]['metadata']['bbox'],[25,25,50,20])

    def test_pending_relation_resolution_preserves_topology(self):
        scene=sample_scene()
        pending={'id':'link','from':'A','to':'B','from_anchor':'east','to_anchor':'west','arrow':'end'}
        scene['metadata']['pending_edges']=[pending]
        edge={'id':'link','points':[[70,80],[200,80]],'arrow':'end'}
        after,_=apply_scene_patch(scene,{'operations':[operation('resolve_pending_edge','link',value=edge)]})
        self.assertEqual(after['metadata']['pending_edges'],[])
        self.assertEqual(after['edges'][0]['metadata']['source_id'],'A')
        self.assertEqual(after['edges'][0]['metadata']['target_id'],'B')
        edge['arrow']='start'
        with self.assertRaises(ValueError):apply_scene_patch(scene,{'operations':[operation('resolve_pending_edge','link',value=edge)]})

    def test_patch_rejects_invalid_math_ids_and_extra_fields(self):
        for value in ({'id':'caption','text':r'\input{secret}','mode':'math','x':45,'y':40},
                      {'id':'other','text':'changed ID','x':45,'y':40}):
            with self.assertRaises(ValueError):apply_scene_patch(sample_scene(),{'operations':[operation('replace_label','caption',value=value)]})
        with self.assertRaises(ValueError):apply_scene_patch(sample_scene(),{'operations':[operation('move_label','caption',dx=1,dy=0,command='ignored')]})
        with self.assertRaises(ValueError):parse_patch_response(json.dumps({'operations':[{}]*33}))

    def test_visual_patch_gets_two_images_and_no_unconditional_extra_call(self):
        client=Mock();client.complete_images.return_value='{"operations":[]}'
        patch,_=request_patch(client,['original.png','preview.png'],sample_scene(),{},phase='visual')
        self.assertEqual(patch,{'operations':[]})
        self.assertEqual(client.complete_images.call_args.args[0],['original.png','preview.png'])
        self.assertEqual(client.complete_images.call_count,1)
        with self.assertRaises(ValueError):request_patch(client,['preview.png'],sample_scene(),{},phase='visual')
        self.assertIn('ONLY diagnosed',build_patch_prompt(sample_scene(),{},'complete'))


if __name__=='__main__':unittest.main()
