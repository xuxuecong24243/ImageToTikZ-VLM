"""Synthetic semantic fixtures only. These are NOT recognized image contents."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from unittest.mock import Mock, patch

from PIL import Image

from semantic_analyzer import (analyze_semantics, build_semantic_context,
                               build_semantic_prompt, main, prepare_semantic_inputs)
from semantic_schema import SemanticValidationError, parse_semantic_response, validate_semantic_result
from semantic_scene_merger import merge_semantics
from structured_tikz.validator import validate_scene
from vlm_client import APIConfig, ApiNotAuthorized


def scene():
    return {'canvas':{'width':400,'height':200,'coordinate_system':'top-left'},'styles':{},
            'nodes':[{'id':'rect_A','type':'rectangle','bbox':[20,60,100,70],
                      'metadata':{'source':'opencv','source_id':'rect_A','review_status':'unverified'}},
                     {'id':'rect_B','type':'rectangle','bbox':[250,60,100,70],
                      'metadata':{'source':'opencv','source_id':'rect_B','review_status':'unverified'}}],
            'edges':[],'labels':[],
            'metadata':{'source':'opencv_geometry_scene_draft','review_status':'unverified',
                        'source_image':{'width':400,'height':200},
                        'review_candidates':{'lines':[], 'text_regions':[
                            {'id':'text_A','bbox':[20,20,100,25], 'recognized_text':None}]}}}


def response():
    return {'objects':[{'candidate_id':'rect_A','semantic_type':'tensor3d','label':'Mock object only',
                        'confidence':'low','review_required':True}],
            'relations':[{'source_id':'rect_A','target_id':'rect_B','relation':'unknown',
                          'direction':'unknown','confidence':'unknown','review_required':True}],
            'texts':[{'id':'mock_label','candidate_id':'rect_A','text_region_id':'text_A',
                      'raw_text':'离线Mock标签','text':'离线Mock标签','mode':'text','bbox':None,
                      'confidence':'low','review_required':True}],
            'unmatched_objects':[], 'unmatched_texts':[],
            'warnings':['OFFLINE MOCK: not a real image recognition result.']}


class SemanticSchemaAndMergeTests(OfflineTestCase):
    def test_strict_json_and_invalid_encodings(self):
        data=response()
        self.assertEqual(parse_semantic_response(json.dumps(data),scene()),data)
        for raw in ('```json\n'+json.dumps(data)+'\n```','explanation '+json.dumps(data),'[]','{',
                    '{"objects":[],"objects":[]}',json.dumps(data).replace('"low"','NaN',1)):
            with self.subTest(raw=raw[:40]),self.assertRaises(SemanticValidationError):
                parse_semantic_response(raw,scene())

    def test_invalid_candidate_ids_and_duplicates_rejected(self):
        cases=[]
        r=response();r['objects'][0]['candidate_id']='invented';cases.append(r)
        r=response();r['objects'].append(copy.deepcopy(r['objects'][0]));cases.append(r)
        r=response();r['relations'][0]['target_id']='invented';cases.append(r)
        r=response();r['texts'][0]['text_region_id']='invented';cases.append(r)
        r=response();r['texts'].append(copy.deepcopy(r['texts'][0]));cases.append(r)
        for r in cases:
            with self.subTest(r=r),self.assertRaises(SemanticValidationError):
                validate_semantic_result(r,scene())

    def test_types_review_confidence_and_object_bbox_cannot_be_invented(self):
        for field,value in [('semantic_type','neural_layer'),('confidence',.99),('review_required',False),('bbox',[0,0,1,1])]:
            r=response();r['objects'][0][field]=value
            with self.subTest(field=field),self.assertRaises(SemanticValidationError):
                validate_semantic_result(r,scene())

    def test_missing_text_and_invalid_math_are_rejected(self):
        for field,value in [('text',''),('raw_text',None),('bbox',[0,0,-1,10])]:
            r=response();r['texts'][0][field]=value
            with self.subTest(field=field),self.assertRaises(SemanticValidationError):
                validate_semantic_result(r,scene())
        r=response();r['texts'][0].update(mode='math',text=r'\input{private}')
        with self.assertRaises(SemanticValidationError):
            validate_semantic_result(r,scene())

    def test_unknown_relations_remain_candidates_and_never_edges(self):
        r=response();r['relations'][0].update(source_id=None,target_id=None)
        merged=merge_semantics(scene(),r)
        self.assertEqual(merged['edges'],[])
        self.assertEqual(merged['metadata']['semantic_review']['relations'][0]['direction'],'unknown')
        r=response();r['relations'][0].update(relation='proximity',direction='forward')
        with self.assertRaises(SemanticValidationError):
            validate_semantic_result(r,scene())

    def test_bbox_ids_and_input_preserved_tensor_type_deferred(self):
        draft=scene();before=copy.deepcopy(draft);r=response();r_before=copy.deepcopy(r)
        merged=merge_semantics(draft,r)
        self.assertEqual(draft,before);self.assertEqual(r,r_before)
        self.assertEqual([(n['id'],n['bbox']) for n in merged['nodes']],[(n['id'],n['bbox']) for n in draft['nodes']])
        self.assertEqual(merged['nodes'][0]['type'],'rectangle')
        self.assertEqual(merged['nodes'][0]['metadata']['semantic_candidate']['semantic_type'],'tensor3d')
        self.assertTrue(merged['metadata']['semantic_review']['deferred_types'])
        self.assertNotIn('depth',merged['nodes'][0]);self.assertNotIn('grid',merged['nodes'][0])
        self.assertEqual(merged['nodes'][0]['metadata']['source'],'opencv')
        validate_scene(merged)

    def test_schema_compatible_type_update_remains_unverified(self):
        r=response();r['objects'][0]['semantic_type']='rounded_rectangle'
        merged=merge_semantics(scene(),r)
        self.assertEqual(merged['nodes'][0]['type'],'rounded_rectangle')
        self.assertEqual(merged['metadata']['review_status'],'unverified')

    def test_verified_and_rejected_objects_are_unchanged(self):
        for status in ('verified','rejected'):
            draft=scene();draft['nodes'][0]['metadata']['review_status']=status
            before=copy.deepcopy(draft['nodes'][0])
            merged=merge_semantics(draft,response())
            self.assertEqual(merged['nodes'][0],before)
            self.assertEqual(merged['labels'],[])
            self.assertTrue(merged['metadata']['semantic_review']['protected_objects'])
        draft=scene();draft['metadata']['review_status']='verified'
        with self.assertRaises(SemanticValidationError):
            merge_semantics(draft,response())

    def test_protected_labels_and_colliding_ids_not_overwritten(self):
        draft=scene()
        existing={'id':'human_label','text':'human','mode':'text','x':30,'y':25,
                  'metadata':{'review_status':'verified'}}
        draft['labels']=[existing]
        self.assertEqual(merge_semantics(draft,response())['labels'],[existing])
        draft['labels'][0]=dict(existing,id='sem_mock_label',x=350,y=180)
        self.assertEqual(merge_semantics(draft,response())['labels'],draft['labels'])

    def test_text_positions_use_measured_region_before_model_estimate(self):
        r=response();r['texts'][0]['bbox']=[250,150,100,20]
        merged=merge_semantics(scene(),r)
        label=merged['labels'][0]
        self.assertEqual((label['x'],label['y']),(70,32.5))
        self.assertEqual(label['metadata']['position_source'],'existing_text_region_measurement')
        self.assertEqual(label['metadata']['model_proposed_bbox'],[250,150,100,20])
        self.assertEqual(label['metadata']['raw_text'],'离线Mock标签')

    def test_null_position_and_unmatched_objects_are_not_fabricated(self):
        r=response();r['texts'][0]['text_region_id']=None
        r['unmatched_objects']=[{'description':'Mock unmatched shape','semantic_type':'unknown',
                                'bbox':None,'confidence':'unknown','review_required':True}]
        merged=merge_semantics(scene(),r)
        self.assertEqual(merged['labels'],[])
        self.assertEqual(len(merged['nodes']),2)
        self.assertTrue(merged['metadata']['semantic_review']['pending_texts'])
        self.assertTrue(merged['metadata']['semantic_review']['unmatched_objects'])

    def test_formula_raw_text_and_latex_remain_separate(self):
        r=response();r['texts'][0].update(raw_text='Mock x squared',text=r'x^{2}+\alpha',mode='math')
        merged=merge_semantics(scene(),r)
        self.assertEqual(merged['labels'][0]['text'],r'x^{2}+\alpha')
        self.assertEqual(merged['labels'][0]['metadata']['raw_text'],'Mock x squared')


class SemanticInputAndAuthorizationTests(OfflineTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'outputs')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.image=self.root/'fixture.png'
        Image.new('RGB',(400,200),'white').save(self.image)
        self.source=self.root/'scene.json'
        self.source.write_text(json.dumps(scene()),encoding='utf-8')

    def test_reference_and_prior_labels_cannot_leak_into_prompt(self):
        draft=scene();sentinel='DO_NOT_SEND_REFERENCE_LABEL'
        draft['metadata']['manual_reference']=sentinel
        draft['nodes'][0]['text']=sentinel
        draft['labels']=[{'id':'old_label','text':sentinel,'x':10,'y':10}]
        draft['metadata']['review_candidates']['text_regions'][0]['recognized_text']=sentinel
        prompt=build_semantic_prompt(build_semantic_context(draft))
        self.assertNotIn(sentinel,prompt)
        self.assertNotIn('gpt_embedding_scene',prompt)
        self.assertIn('rect_A',prompt)
        draft['metadata']['source']='manual_annotation_from_inputs/test.png'
        with self.assertRaises(SemanticValidationError):
            build_semantic_context(draft)

    def test_line_hints_are_capped_not_full_raw_segments(self):
        draft=scene()
        draft['metadata']['review_candidates']['lines']=[
            {'id':'line_%03d'%i,'classification':'geometry','points':[[10,i+1],[100,i+1]],
             'source_candidate':{'score':.5}} for i in range(100)]
        context=build_semantic_context(draft)
        self.assertEqual(len(context['line_hints']),12)
        self.assertEqual(context['omitted_line_hint_count'],88)

    def test_annotation_same_size_and_no_overwrite_or_image_mutation(self):
        before=self.image.read_bytes()
        output=self.root/'prepared'
        prepared=prepare_semantic_inputs(self.image,scene(),output)
        with Image.open(prepared['annotation']) as im:
            self.assertEqual(im.size,(400,200))
        self.assertEqual(self.image.read_bytes(),before)
        with self.assertRaises(FileExistsError):
            prepare_semantic_inputs(self.image,scene(),output)

    def test_analysis_requires_authorization_before_preparation_or_client(self):
        client=Mock(allow_api=False)
        with patch('semantic_analyzer.prepare_semantic_inputs') as prepare,self.assertRaises(ApiNotAuthorized):
            analyze_semantics(self.image,scene(),client,output_dir=self.root/'no-call')
        prepare.assert_not_called();client.complete_images.assert_not_called()

    def test_mock_response_uses_original_and_python_annotation_once(self):
        client=Mock(allow_api=True,config=APIConfig('https://example.test','test-secret','user-model'))
        client.complete_images.return_value=json.dumps(response(),ensure_ascii=False)
        result=analyze_semantics(self.image,scene(),client,output_dir=self.root/'mock')
        self.assertEqual(result,response())
        client.complete_images.assert_called_once()
        images,prompt=client.complete_images.call_args.args
        self.assertEqual(images[0],self.image.resolve())
        self.assertEqual(images[1].name,'candidate_annotation.png')
        self.assertEqual(len(images),2)
        self.assertIn('Return one strict JSON',prompt)

    def test_cli_without_allow_api_only_prepares_even_with_key(self):
        env={'API_BASE_URL':'https://api.deepseek.com','API_KEY':'test-secret','MODEL_NAME':'deepseek-flash'}
        with patch.dict(os.environ,env,clear=True),patch('semantic_analyzer.VLMClient') as client:
            code=main(['--image',str(self.image),'--scene',str(self.source),'--output-dir',str(self.root/'dry')])
        client.assert_not_called();self.assertEqual(code,0)
        report=json.loads(next((self.root/'dry').glob('*/run.json')).read_text(encoding='utf-8'))
        self.assertFalse(report['api_called']);self.assertEqual(report['status'],'prepared_offline')
        self.assertNotIn('test-secret',json.dumps(report))
        self.assertEqual(report['thinking_mode'],'disabled')

    def test_missing_config_check_and_manual_reference_fail_without_network(self):
        with patch.dict(os.environ,{},clear=True),patch('semantic_analyzer.VLMClient') as client:
            self.assertEqual(main(['--check']),0)
            data=scene();data['metadata']['source']='manual_reference'
            self.source.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaises(SystemExit):
                main(['--image',str(self.image),'--scene',str(self.source),'--allow-api'])
        client.assert_not_called()


if __name__=='__main__':
    unittest.main()
