from copy import deepcopy
try:
    from .offline_support import OfflineTestCase
    from .ir_fixtures import branch_ir
except ImportError:
    from offline_support import OfflineTestCase
    from ir_fixtures import branch_ir
from structured_ir.consistency import inspect_consistency, repair_reviewed_pairings
from structured_ir.geometry_repair import ir_fingerprint
from structured_ir.converter import convert_ir

SHA='a'*64
def fixture():
    return {'version':'1.0','canvas':{'id':'c','width':500,'height':300,'coordinate_system':'top-left'},'regions':[],
       'elements':[{'id':'a','type':'rectangle','bbox':[20,20,120,80],'region_id':None},
                   {'id':'b','type':'rectangle','bbox':[220,20,120,80],'region_id':None},
                   {'id':'ta','type':'text','bbox':[40,40,30,20],'region_id':None,'raw_text':'9','text_type':'plain','reading_order':0},
                   {'id':'tb','type':'text','bbox':[240,40,30,20],'region_id':None,'raw_text':'2','text_type':'plain','reading_order':1}],
       'connections':[{'id':'flow','source_id':'a','target_id':'b','direction':'forward','source_anchor':'east','target_anchor':'west','kind':'flow'}]}
def proof(ir,**updates):
    value={'image_sha256':SHA,'ir_sha256':ir_fingerprint(ir),'canvas_size':[500,300],'coordinate_system':'top-left',
           'confirmed':True,'basis':'Reviewed synthetic independent annotation.',
           'connections':[{'id':'flow','source_id':'a','target_id':'b','direction':'forward'}],
           'pairs':[{'shape_id':'a','text_id':'ta','text':'9','shape_bbox':[20,20,120,80],'text_bbox':[40,40,30,20]},
                    {'shape_id':'b','text_id':'tb','text':'2','shape_bbox':[220,20,120,80],'text_bbox':[240,40,30,20]}]}
    value.update(updates); return value
def run(ir,reference=None,scene=None):
    if scene is None: scene,conversion=convert_ir(ir)
    else: conversion={'status':'complete'}
    return inspect_consistency(ir,scene,conversion,reference,image_sha256=SHA)

class ConsistencyTests(OfflineTestCase):
    def test_complete_does_not_mean_verified(self):
        r=run(fixture()); self.assertEqual(r['conversion_status'],'complete'); self.assertEqual(r['semantic_status'],'unverified')
        self.assertEqual(r['connections'][0]['scene_mapping'],'verified'); self.assertEqual(r['connections'][0]['semantic_status'],'unverified')
    def test_reviewed_flow_and_local_pairing_verified(self):
        ir=fixture(); r=run(ir,proof(ir)); self.assertEqual(r['connections'][0]['semantic_status'],'verified')
        self.assertTrue(all(p['status']=='verified' for p in r['pairings'])); self.assertEqual(r['semantic_status'],'unverified')
    def test_invalid_source_and_target(self):
        for key in ('source_id','target_id'):
            ir=fixture(); ir['connections'][0][key]='missing'; r=inspect_consistency(ir)
            self.assertEqual(r['semantic_status'],'partial'); self.assertTrue(any(c.get('path','').endswith(key) for c in r['checks']))
    def test_scene_endpoint_mapping_mismatch(self):
        ir=fixture(); s,_=convert_ir(ir); s['edges'][0]['to']='a'; r=run(ir,scene=s)
        self.assertIn('target_id_mapping_mismatch',r['connections'][0]['mapping_errors'])
    def test_scene_direction_mismatch(self):
        ir=fixture(); s,_=convert_ir(ir); s['edges'][0]['arrow']='start'; r=run(ir,scene=s)
        self.assertIn('direction_mapping_mismatch',r['connections'][0]['mapping_errors'])
    def test_reference_direction_conflict(self):
        ir=fixture(); p=proof(ir); p['connections'][0]['direction']='reverse'; r=run(ir,p)
        self.assertEqual(r['connections'][0]['semantic_status'],'conflict')
    def test_typed_flow_rejects_mixing_without_business_names(self):
        ir=fixture(); p=proof(ir); p['connections'][0].update(source_type='type_a',target_type='type_b',allowed_type_pairs=[['type_a','type_c']])
        r=run(ir,p); self.assertTrue(any(c['code']=='dataflow_types' and c['status']=='conflict' for c in r['checks']))
        p['connections'][0]['allowed_type_pairs']=[['type_a','type_b']]; r=run(ir,p)
        self.assertTrue(any(c['code']=='dataflow_types' and c['status']=='verified' for c in r['checks']))
    def test_missing_type_evidence_not_verified(self):
        ir=fixture(); p=proof(ir); p['connections'][0].update(source_type='type_a',allowed_type_pairs=[])
        self.assertTrue(any(c['code']=='dataflow_types' and c['status']=='unverified' for c in run(ir,p)['checks']))
    def test_equal_boxes_need_evidence_to_classify(self):
        ir=fixture(); ir['elements'][1]['bbox']=ir['elements'][0]['bbox'][:]
        r=inspect_consistency(ir); self.assertTrue(any(c['code']=='bbox_collision' and c['status']=='unverified' for c in r['collisions']))
        r=run(ir,proof(ir)); self.assertTrue(any(c['code']=='bbox_collision' and c['status']=='conflict' for c in r['collisions']))
    def test_reviewed_legal_overlap_and_duplicate_detection(self):
        ir=fixture(); ir['elements'][1]['bbox']=ir['elements'][0]['bbox'][:]
        for kind,state in [('legal_overlap','verified'),('duplicate_detection','conflict')]:
            p=proof(ir,pairs=[],overlaps=[{'ids':['a','b'],'classification':kind}]); r=run(ir,p)
            self.assertTrue(any(c['code']==kind and c['status']==state for c in r['collisions']))
            self.assertEqual(len(ir['elements']),4)
    def test_container_containment_not_false_collision(self):
        r=run(fixture()); self.assertEqual(r['collisions'],[])
    def test_large_overlap_uncertain_not_automatically_error(self):
        ir=fixture(); ir['elements'][1]['bbox']=[50,20,120,80]
        r=inspect_consistency(ir); self.assertTrue(any(c['code']=='large_overlap' and c['status']=='unverified' for c in r['collisions']))
    def test_sequence_uses_position_not_token_numeric_value(self):
        ir=fixture(); r=run(ir,proof(ir)); self.assertEqual(r['pairings'][-1]['status'],'verified')
        ir['elements'][3]['reading_order']=0; r=run(ir,proof(ir)); self.assertEqual(r['pairings'][-1]['status'],'conflict')
    def test_pair_not_bijective_is_rejected(self):
        ir=fixture(); p=proof(ir); p['pairs'][1]['text_id']='ta'; r=run(ir,p)
        self.assertTrue(any('non_bijective_pairing' in c.get('errors',[]) for c in r['pairings']))
        after,report=repair_reviewed_pairings(ir,p,image_sha256=SHA); self.assertEqual(after,ir); self.assertEqual(report['changes'],[])
    def test_atomic_repair_preserves_all_semantics_and_references(self):
        ir=fixture(); ir['elements'][1]['bbox']=[20,20,120,80]; ir['elements'][3]['bbox']=[45,40,30,20]
        original=deepcopy(ir); p=proof(ir); reference=deepcopy(p)
        after,report=repair_reviewed_pairings(ir,p,image_sha256=SHA)
        self.assertEqual(report['status'],'repaired'); self.assertEqual(ir,original); self.assertEqual(p,reference)
        self.assertEqual(after['connections'],original['connections']); self.assertEqual(after,fixture())
        self.assertEqual(convert_ir(after)[1]['status'],'complete')
    def test_repair_no_evidence_no_changes(self):
        ir=fixture(); a,r=repair_reviewed_pairings(ir,None,image_sha256=SHA)
        self.assertEqual(a,ir); self.assertEqual(r['status'],'unverified')
    def test_stale_hash_wrong_canvas_and_unconfirmed_evidence_rejected(self):
        ir=fixture()
        for k,v in [('ir_sha256','b'*64),('image_sha256','b'*64),('canvas_size',[600,300]),('coordinate_system','normalized'),('confirmed',False)]:
            p=proof(ir); p[k]=v
            with self.subTest(k=k),self.assertRaises(ValueError): repair_reviewed_pairings(ir,p,image_sha256=SHA)
    def test_entire_group_rollback_on_invalid_measured_box(self):
        ir=fixture(); ir['elements'][1]['bbox']=[20,20,120,80]; p=proof(ir)
        p['pairs'][1]['shape_bbox']=[450,20,120,80]; p['pairs'][1]['text_bbox']=[470,40,30,20]
        a,r=repair_reviewed_pairings(ir,p,image_sha256=SHA); self.assertEqual(a,ir); self.assertEqual(r['changes'],[])
    def test_branch_label_mapping_and_diamond_survive(self):
        ir=branch_ir(); s,c=convert_ir(ir); r=inspect_consistency(ir,s,c)
        self.assertEqual(c['status'],'complete'); self.assertTrue(all(x['scene_mapping']=='verified' for x in r['connections']))
        s['labels'][0]['text']='wrong'; r=inspect_consistency(ir,s,c)
        self.assertTrue(any('branch_label_mapping_unverified' in x.get('mapping_errors',[]) for x in r['connections']))
    def test_bounds_and_duplicate_id_are_reported_without_mutation(self):
        for modification in ('bounds','id'):
            ir=fixture()
            if modification=='bounds': ir['elements'][0]['bbox']=[490,20,120,80]
            else: ir['elements'][1]['id']='a'
            old=deepcopy(ir); r=inspect_consistency(ir); self.assertEqual(r['semantic_status'],'partial'); self.assertEqual(ir,old)
    def test_duplicate_or_malformed_reference_is_not_silently_selected(self):
        ir=fixture(); p=proof(ir); p['connections'].append(deepcopy(p['connections'][0]))
        with self.assertRaises(ValueError): run(ir,p)
        p=proof(ir); p['connections'][0]['status']='not_reviewed'
        with self.assertRaises(ValueError): run(ir,p)
        p=proof(ir); p['overlaps']=[{'ids':['a','b'],'classification':'legal_overlap'},
                                  {'ids':['b','a'],'classification':'duplicate_detection'}]
        with self.assertRaises(ValueError): run(ir,p)
    def test_unknown_pair_does_not_verify_empty_or_partial_sequence(self):
        ir=fixture(); p=proof(ir); p['pairs'][1]['text_id']='missing'
        r=run(ir,p); self.assertEqual(r['pairings'][-1]['status'],'conflict')
        self.assertIn('unknown_pair_reference',r['pairings'][1]['errors'])
    def test_missing_type_evidence_keeps_edge_summary_unverified(self):
        ir=fixture(); p=proof(ir); p['connections'][0].update(source_type='type_a',allowed_type_pairs=[])
        self.assertEqual(run(ir,p)['connections'][0]['semantic_status'],'unverified')
    def test_reviewed_anchor_and_actual_label_position_are_checked(self):
        ir=fixture(); p=proof(ir); p['connections'][0]['source_anchor']='south'
        self.assertEqual(run(ir,p)['connections'][0]['semantic_status'],'conflict')
        ir=branch_ir(); s,c=convert_ir(ir); s['labels'][0]['x']+=1
        r=inspect_consistency(ir,s,c)
        self.assertTrue(any('branch_label_position_mismatch' in e.get('mapping_errors',[]) for e in r['connections']))
    def test_pair_requires_shape_and_text_roles(self):
        ir=fixture(); p=proof(ir); p['pairs'][0].update(shape_id='ta',shape_bbox=ir['elements'][2]['bbox'])
        r=run(ir,p); self.assertIn('invalid_pair_element_type',r['pairings'][0]['errors'])
        a,result=repair_reviewed_pairings(ir,p,image_sha256=SHA)
        self.assertEqual(a,ir); self.assertEqual(result['changes'],[])
    def test_sequence_report_retains_reference_order(self):
        ir=fixture(); p=proof(ir); r=run(ir,p)
        self.assertEqual(r['pairings'][-1]['ids'],['a','b'])
