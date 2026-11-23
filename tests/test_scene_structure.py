"""Bounded deterministic recovery never supplies missing image semantics."""
import copy
import json
from pathlib import Path
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from scene_structure import inspect_structure, recover_scene_structure
from structured_tikz.generator import generate_tikz


def sample():
    return {'canvas':{'width':300,'height':200,'coordinate_system':'top-left'},
            'styles':{},'nodes':[{'id':'A','type':'text','bbox':[20,20,80,30],'text':'原文_1'}],
            'labels':[],'edges':[]}


class SceneStructureTests(OfflineTestCase):
    def recover(self,data,**kwargs):
        return recover_scene_structure(data,(300,200),coordinate_system='top-left',**kwargs)

    def test_canvas_from_authoritative_input_without_geometry_edits(self):
        data=sample();del data['canvas'];before=copy.deepcopy(data)
        after,report=self.recover(data)
        self.assertEqual(data,before);self.assertEqual(after['canvas'],sample()['canvas'])
        self.assertEqual(after['nodes'],data['nodes'])
        self.assertTrue(report['after']['structure_accepted']);self.assertFalse(report['content_restored'])

    def test_partial_canvas_missing_fields_only_and_scale_preserved(self):
        data=sample();data['canvas']={'width':300,'scale':.25}
        after,report=self.recover(data)
        self.assertEqual(after['canvas'],dict(width=300,height=200,scale=.25,coordinate_system='top-left'))
        self.assertEqual(len(report['changes']),2)

    def test_canvas_mismatch_or_null_is_not_overwritten(self):
        for value in (None,{'width':123,'height':200,'coordinate_system':'top-left'}):
            with self.subTest(canvas=value):
                data=sample();data['canvas']=value;after,report=self.recover(data)
                self.assertEqual(after['canvas'],value);self.assertFalse(report['after']['structure_accepted'])

    def test_coordinates_not_assumed_without_prompt_evidence(self):
        data=sample();del data['canvas']
        after,report=recover_scene_structure(data,(300,200))
        self.assertNotIn('canvas',after);self.assertFalse(report['after']['structure_accepted'])

    def test_absent_styles_uses_existing_defaults_only(self):
        data=sample();expected=generate_tikz(data);del data['styles']
        after,report=self.recover(data)
        self.assertEqual(after['styles'],{});self.assertEqual(generate_tikz(after),expected)
        self.assertEqual(report['changes'][0]['path'],'/styles')

    def test_missing_named_styles_not_invented_and_explicit_config_supported(self):
        data=sample();del data['styles'];data['nodes'][0]['style']='emphasis'
        after,report=self.recover(data)
        self.assertNotIn('styles',after);self.assertFalse(report['after']['schema_valid'])
        after,report=self.recover(data,default_styles={'emphasis':{'bold':True}})
        self.assertEqual(after['styles'],{'emphasis':{'bold':True}})
        self.assertTrue(report['after']['schema_valid'])

    def test_invalid_configured_styles_and_existing_null_are_retained(self):
        data=sample();del data['styles'];data['nodes'][0]['style']='emphasis'
        after,report=self.recover(data,default_styles={'emphasis':{'fill':'invented'}})
        self.assertNotIn('styles',after);self.assertFalse(report['after']['schema_valid'])
        data=sample();data['styles']=None;after,report=self.recover(data)
        self.assertIsNone(after['styles']);self.assertFalse(report['after']['schema_valid'])

    def test_transport_echo_removed_only_with_exact_trusted_request_format(self):
        data=sample();data['type']='json_object'
        for evidence in (None,{'type':'text'}, {'type':'json_object','extra':1}):
            with self.subTest(evidence=evidence):
                after,report=self.recover(data,response_format=evidence)
                self.assertIn('type',after);self.assertFalse(report['after']['schema_valid'])
        after,report=self.recover(data,response_format={'type':'json_object'})
        self.assertNotIn('type',after);self.assertTrue(report['after']['schema_valid'])
        self.assertEqual(report['changes'][0]['before'],'json_object');self.assertEqual(data['type'],'json_object')

    def test_unknown_type_values_and_other_fields_are_never_dropped(self):
        for value in ('flowchart','diamond',{'type':'json_object'},None):
            with self.subTest(value=value):
                data=sample();data.update(type=value,unrecognised={'meaning':'retain me'})
                after,report=self.recover(data,response_format={'type':'json_object'})
                self.assertEqual(after,data);self.assertFalse(report['after']['schema_valid'])

    def test_missing_drawing_collections_do_not_become_empty_placeholders(self):
        for key in ('nodes','labels','edges'):
            with self.subTest(key=key):
                data=sample();del data[key];after,report=self.recover(data)
                self.assertNotIn(key,after);self.assertFalse(report['after']['schema_valid'])

    def test_semantic_content_unknown_shapes_and_references_not_repaired(self):
        data=sample();data['nodes'][0]['type']='polygon'
        data['edges']=[{'id':'arrow','from':'A','to':'missing','arrow':'end'}]
        data['labels']=[{'id':'A','x':2,'y':2,'text':'duplicate id'}]
        after,report=self.recover(data)
        self.assertEqual(after,data);self.assertFalse(report['after']['schema_valid'])
        self.assertEqual(report['semantic_objects_added'],0)

    def test_idempotent_one_pass_and_valid_input_noop(self):
        data=sample();del data['canvas'];del data['styles']
        first,r1=self.recover(data);second,r2=self.recover(first)
        self.assertEqual(first,second);self.assertEqual(r2['changes'],[])
        self.assertEqual(r1['passes'],1);self.assertEqual(r1['max_passes'],1)

    def test_protected_reviewed_data_never_modified(self):
        for state in ('verified','rejected'):
            data=sample();del data['canvas'];data['metadata']={'review_status':state}
            after,report=self.recover(data);self.assertEqual(after,data);self.assertEqual(report['changes'],[])

    def test_legacy_free_metadata_with_nonstring_review_status_does_not_crash(self):
        for value in ([],{},None,True):
            with self.subTest(value=value):
                data=sample();data['metadata']={'review_status':value}
                after,report=self.recover(data)
                self.assertEqual(after,data);self.assertTrue(report['after']['schema_valid'])

    def test_legacy_scene_preserves_exact_generated_tikz(self):
        path=Path(__file__).resolve().parents[1]/'examples/gpt_embedding_scene.json'
        data=json.loads(path.read_text(encoding='utf-8'));old=generate_tikz(data)
        after,report=recover_scene_structure(data,(data['canvas']['width'],data['canvas']['height']),coordinate_system='top-left')
        self.assertEqual(after,data);self.assertEqual(generate_tikz(after),old);self.assertEqual(report['changes'],[])

    def test_bad_dimensions_rejected_and_root_type_diagnosed(self):
        for size in ((0,200),(300,True),(300,),(-2,200)):
            with self.subTest(size=size),self.assertRaises(ValueError):recover_scene_structure(sample(),size)
        after,report=self.recover([])
        self.assertEqual(after,[]);self.assertFalse(report['after']['schema_valid'])


if __name__=='__main__':unittest.main()
