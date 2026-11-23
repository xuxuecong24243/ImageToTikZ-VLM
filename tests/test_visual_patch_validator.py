"""Synthetic atomic visual-patch contracts; no model, PDF or compiler calls."""
import copy
import json
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from visual_patch_validator import (VisualPatchError, apply_visual_patch,
                                    parse_visual_feedback)
from structured_tikz.generator import generate_tikz
from structured_tikz.validator import validate_scene


def scene():
    return {'canvas':{'width':400,'height':240,'coordinate_system':'top-left'},'styles':{},
            'nodes':[{'id':'a','type':'rectangle','bbox':[20,100,60,40]},
                     {'id':'b','type':'rectangle','bbox':[300,140,60,40]}],
            'labels':[{'id':'caption','x':200,'y':40,'text':r'W_E\in\mathbb{R}^{50257\times256}',
                       'mode':'math','metadata':{'bbox':[170,30,60,20]}}],
            'edges':[{'id':'link','points':[[90,120],[100,120],[100,160],[300,160]],'arrow':'end',
                      'metadata':{'source_id':'a','target_id':'b','from_anchor':'east','to_anchor':'west'}}]}


def proposal(*operations):
    return {'status':'patch_proposed','observations':[
        {'object_id':op['id'],'issue':'position_mismatch','confidence':.8,
         'reason':'Synthetic offline visual-evidence fixture, not a model assessment.'}
        for op in operations], 'operations':list(operations)}


def move(identifier='caption', dx=5, dy=0):
    return {'op':'move_label','id':identifier,'dx':dx,'dy':dy}


class VisualPatchValidatorTests(OfflineTestCase):
    def test_new_unowned_label_containment_is_rejected(self):
        original={'canvas':{'width':300,'height':200,'coordinate_system':'top-left'},
                  'styles':{},'nodes':[{'id':'box','type':'rectangle','bbox':[100,100,40,40]}],
                  'labels':[{'id':'small','text':'x','x':143.5,'y':115,'metadata':{'bbox':[141,110,5,10]}}],
                  'edges':[]}
        data=proposal({'op':'set_node_geometry','id':'box','value':{'bbox':[110,100,40,40]}})
        with self.assertRaises(VisualPatchError) as caught:apply_visual_patch(original,data)
        risks=caught.exception.report['layout_guard']['increased_risks']
        self.assertTrue(any(r['kind']=='text_geometry' and r['ids']==['small','box'] for r in risks))
        self.assertEqual(original['nodes'][0]['bbox'],[100,100,40,40])

    def test_independent_label_layout_hint_translates_but_owner_metadata_does_not(self):
        original=scene()
        original['labels'][0]['metadata'].update(node_id='a',source_id='a',note='preserve provenance')
        result,report=apply_visual_patch(original,proposal(move(dx=4,dy=-3)))
        expected=copy.deepcopy(original['labels'][0]['metadata'])
        expected['bbox']=[174,27,60,20]
        for key,value in expected.items():
            self.assertEqual(result['labels'][0]['metadata'][key],value)
        self.assertEqual(original['labels'][0]['metadata']['bbox'],[170,30,60,20])
        self.assertEqual(report['proposed_operations'],[move(dx=4,dy=-3)])

    def test_label_shift_preserves_math_and_source_scene(self):
        original=scene();snapshot=copy.deepcopy(original)
        result,report=apply_visual_patch(original,proposal(move()))
        self.assertEqual(result['labels'][0]['x'],205)
        self.assertEqual(result['labels'][0]['text'],original['labels'][0]['text'])
        self.assertEqual(original,snapshot)
        self.assertTrue(report['accepted']);self.assertTrue(report['applied'])
        self.assertEqual(report['operations'][0]['before']['x'],200)
        self.assertEqual(report['operations'][0]['after']['x'],205)
        validate_scene(result)
        self.assertIn('50257',generate_tikz(result))

    def test_text_node_can_move_without_changing_bbox_size_or_formula(self):
        original=scene()
        original['nodes'].append({'id':'dimension','type':'text','bbox':[80,20,60,20],
                                  'text':'B = 8','mode':'math','wrap':False})
        result,_=apply_visual_patch(original,proposal(move('dimension',-5,0)))
        self.assertEqual(result['nodes'][-1]['bbox'],[75,20,60,20])
        self.assertEqual(result['nodes'][-1]['text'],'B = 8')
        self.assertFalse(result['nodes'][-1]['wrap'])

    def test_legal_node_resize_reuses_existing_atomic_applicator(self):
        result,report=apply_visual_patch(scene(),proposal({'op':'set_node_geometry','id':'a',
                                                       'value':{'bbox':[20,100,66,44]}}))
        self.assertEqual(result['nodes'][0]['bbox'],[20,100,66,44])
        self.assertEqual(result['nodes'][0]['type'],'rectangle')
        self.assertTrue(report['invariants']['topology_and_arrows_unchanged'])

    def test_legal_explicit_path_preserves_endpoints_and_topology(self):
        original=scene()
        result,_=apply_visual_patch(original,proposal({'op':'set_edge_path','id':'link',
            'points':[[90,120],[110,120],[110,160],[300,160]]}))
        self.assertEqual(result['edges'][0]['points'][0],original['edges'][0]['points'][0])
        self.assertEqual(result['edges'][0]['points'][-1],original['edges'][0]['points'][-1])
        self.assertEqual(result['edges'][0]['arrow'],'end')
        self.assertEqual(result['edges'][0]['metadata']['source_id'],'a')

    def test_missing_target_id_is_rejected(self):
        with self.assertRaises(VisualPatchError):
            apply_visual_patch(scene(),proposal(move('missing')))

    def test_euclidean_movement_limit_not_per_axis(self):
        for op in (move(dx=21),move(dx=15,dy=15),move(dx=True),move(dx=float('inf'))):
            with self.subTest(operation=op),self.assertRaises(VisualPatchError):
                apply_visual_patch(scene(),proposal(op))

    def test_resize_and_node_movement_limits_are_enforced(self):
        for box in ([20,100,67,40],[41,100,60,40],[20,100,60,35]):
            with self.subTest(box=box),self.assertRaises(VisualPatchError):
                apply_visual_patch(scene(),proposal({'op':'set_node_geometry','id':'a','value':{'bbox':box}}))

    def test_math_modification_and_unsupported_operations_are_rejected(self):
        for op in (dict(move(),text='changed'),{'op':'replace_label','id':'caption','value':{'text':'changed'}},
                   {'op':'set_node_geometry','id':'a','value':{'grid':{'rows':2,'cols':2}}}):
            with self.subTest(operation=op),self.assertRaises(VisualPatchError):
                apply_visual_patch(scene(),proposal(op))

    def test_endpoint_and_semantic_changes_are_rejected(self):
        points=[[91,120],[110,120],[110,160],[300,160]]
        op={'op':'set_edge_path','id':'link','points':points}
        with self.assertRaises(VisualPatchError):apply_visual_patch(scene(),proposal(op))
        op['points'][0]=[90,120];op['source_id']='b'
        with self.assertRaises(VisualPatchError):apply_visual_patch(scene(),proposal(op))

    def test_schema_failure_rolls_back_entire_group(self):
        original=scene();original['labels'][0].update(y=10,metadata={'bbox':[170,0,60,20]})
        snapshot=copy.deepcopy(original)
        data=proposal({'op':'set_node_geometry','id':'a','value':{'bbox':[20,100,66,44]}},move(dx=0,dy=-20))
        with self.assertRaises(VisualPatchError) as caught:apply_visual_patch(original,data)
        self.assertEqual(original,snapshot)
        self.assertFalse(caught.exception.report['applied'])
        self.assertTrue(caught.exception.report['atomic_rollback'])

    def test_no_change_is_a_deep_copy_without_provenance_mutation(self):
        original=scene()
        result,report=apply_visual_patch(original,{'status':'no_change','observations':[],'operations':[]})
        self.assertEqual(result,original);self.assertIsNot(result,original)
        self.assertEqual(report['status'],'no_change');self.assertFalse(report['applied'])

    def test_duplicate_ids_cannot_accumulate_movements(self):
        data=proposal(move(dx=15));data['operations'].append(move(dx=15))
        with self.assertRaises(VisualPatchError):apply_visual_patch(scene(),data)

    def test_new_measured_overlap_is_rejected(self):
        original=scene()
        original['labels'].append({'id':'other','text':'中文','x':250,'y':40,
                                    'metadata':{'bbox':[240,30,20,20]}})
        with self.assertRaises(VisualPatchError) as caught:apply_visual_patch(original,proposal(move(dx=20)))
        self.assertTrue(caught.exception.report['layout_guard']['increased_risks'])

    def test_measured_bounds_translate_as_predictions_not_measurements(self):
        original=scene()
        measured={'mapping':{'status':'verified'},'labels':[{'id':'caption','status':'measured',
            'bbox_scene':[180,34,40,12],'line_boxes_scene':[[180,34,40,12]],
            'original_position':{'x':200,'y':40}}]}
        _,report=apply_visual_patch(original,proposal(move()),measured)
        extent=report['layout_guard']['candidate_text_extents']['caption']
        self.assertTrue(extent['projected']);self.assertTrue(extent['measured_input'])
        self.assertEqual(extent['boxes'],[[185,34,40,12]])
        self.assertTrue(extent['source'].startswith('translation_of_'))

    def test_strict_json_rejects_fence_duplicate_keys_prose_and_nonfinite(self):
        raw=json.dumps({'status':'no_change','observations':[],'operations':[]})
        self.assertEqual(parse_visual_feedback(raw)['status'],'no_change')
        for value in ('```json\n'+raw+'\n```','explanation '+raw,raw+raw,
                      '{"status":"no_change","status":"no_change","observations":[],"operations":[]}',
                      '{"status":"no_change","observations":[],"operations":[],"bad":NaN}'):
            with self.subTest(value=value),self.assertRaises(VisualPatchError):parse_visual_feedback(value)

    def test_missing_evidence_wrong_type_and_false_no_change_are_rejected(self):
        bad=proposal(move());bad['observations']=[]
        for data in (bad,proposal(move('a')),dict(proposal(move()),status='no_change')):
            with self.subTest(data=data),self.assertRaises(VisualPatchError):apply_visual_patch(scene(),data)

    def test_node_move_cannot_disconnect_existing_fixed_endpoint(self):
        with self.assertRaises(VisualPatchError):
            apply_visual_patch(scene(),proposal({'op':'set_node_geometry','id':'a','value':{'bbox':[0,100,60,40]}}))


if __name__=='__main__':
    unittest.main()
