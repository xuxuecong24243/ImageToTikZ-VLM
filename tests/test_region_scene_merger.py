"""Offline synthetic merge contracts; never call a model, CV or compiler."""
from copy import deepcopy
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from region_scene_merger import merge_region_scenes
from structured_tikz.generator import generate_tikz
from structured_tikz.validator import validate_scene


def manifest():
    return {'image':{'width':400,'height':300,'sha256':'a'*64},'regions':[
        {'region_id':'region_001','crop_bbox':[0,0,240,300],'offset':[0,0]},
        {'region_id':'region_002','crop_bbox':[180,0,220,300],'offset':[180,0]}]}


def fixture():
    bundles=[]
    for i, width in enumerate((240,220),1):
        bundles.append({'region_id':'region_%03d'%i,'scene':{
            'canvas':{'width':width,'height':300,'coordinate_system':'top-left','scale':.4},
            'styles':{'module':{'fill':'light_blue','stroke':'dark_blue'}},
            'nodes':[{'id':'a','type':'rectangle','bbox':[20,100,40,30],'style':'module'}],
            'labels':[],'edges':[]},'pending_connections':[],
            'metadata':{'source':'mock_not_model_recognition'}})
    return bundles


def duplicates(bundles):
    for b,x in zip(bundles,(200,20)):
        b['scene']['nodes'].append({'id':'shared','type':'rectangle','bbox':[x,40,20,20],'style':'module'})


class RegionSceneMergerTests(OfflineTestCase):
    def test_local_ids_are_unique_global_ids_and_positions_translate(self):
        scene, report=merge_region_scenes(manifest(),fixture())
        self.assertNotEqual(report['id_map']['region_001']['a'],report['id_map']['region_002']['a'])
        self.assertEqual([n['bbox'] for n in scene['nodes']],[[20,100,40,30],[200,100,40,30]])
        self.assertTrue(all('/' not in n['id'] and len(n['id'])<=64 for n in scene['nodes']))

    def test_label_bbox_owner_and_explicit_path_translate(self):
        bundles=fixture();s=bundles[1]['scene']
        s['labels']=[{'id':'label','text':'768','mode':'math','x':40,'y':160,
            'metadata':{'node_id':'a','bbox':[30,150,20,20],'original_position':{'x':40,'y':160}}}]
        s['edges']=[{'id':'path','points':[[10,180],[40,180],[40,210]],'arrow':'end',
                     'metadata':{'source_id':'a','points':[[10,180],[40,180]]}}]
        scene,report=merge_region_scenes(manifest(),bundles)
        label=scene['labels'][0]
        self.assertEqual([label['x'],label['y']],[220,160])
        self.assertEqual(label['metadata']['bbox'],[210,150,20,20])
        self.assertEqual(label['metadata']['original_position'],{'x':220,'y':160})
        self.assertEqual(label['metadata']['node_id'],report['id_map']['region_002']['a'])
        self.assertEqual(scene['edges'][0]['points'],[[190,180],[220,180],[220,210]])
        self.assertEqual(scene['edges'][0]['metadata']['points'],[[190,180],[220,180]])

    def test_tensor_depth_and_matrix_grid_are_not_translated(self):
        bundles=fixture();s=bundles[1]['scene']
        s['nodes']=[{'id':'tensor','type':'tensor3d','bbox':[30,80,30,30],
            'grid':{'rows':2,'cols':3},'depth':{'dx':10,'dy':-5}},
            {'id':'matrix','type':'matrix','bbox':[100,80,30,30],'grid':{'rows':2,'cols':2}}]
        scene,_=merge_region_scenes(manifest(),bundles)
        tensor=next(n for n in scene['nodes'] if n['type']=='tensor3d')
        self.assertEqual(tensor['bbox'],[210,80,30,30]);self.assertEqual(tensor['depth'],{'dx':10,'dy':-5})
        self.assertEqual(tensor['grid'],{'rows':2,'cols':3})
        self.assertEqual(next(n for n in scene['nodes'] if n['type']=='matrix')['bbox'],[280,80,30,30])

    def test_overlap_dedup_remaps_edges_and_label_owners(self):
        bundles=fixture();duplicates(bundles)
        for b,x in zip(bundles,(210,30)):
            b['scene']['labels']=[{'id':'caption','text':'768','mode':'math','x':x,'y':70,'metadata':{'node_id':'shared'}}]
            b['scene']['edges']=[{'id':'link','from':'a','to':'shared','arrow':'end'}]
        scene,report=merge_region_scenes(manifest(),bundles)
        self.assertEqual(len(scene['nodes']),3);self.assertEqual(len(scene['labels']),1)
        self.assertEqual(report['counts']['deduplicated'],2)
        shared=report['id_map']['region_001']['shared']
        self.assertEqual(shared,report['id_map']['region_002']['shared'])
        self.assertTrue(all(e['to']==shared for e in scene['edges']))
        self.assertEqual(scene['labels'][0]['metadata']['node_id'],shared)

    def test_same_text_at_different_positions_is_not_deduplicated(self):
        bundles=fixture()
        for b in bundles:b['scene']['labels']=[{'id':'d','text':'768','mode':'math','x':80,'y':180}]
        scene,report=merge_region_scenes(manifest(),bundles)
        self.assertEqual(len(scene['labels']),2);self.assertEqual(report['counts']['deduplicated'],0)

    def test_overlapping_different_content_or_style_is_preserved(self):
        bundles=fixture();duplicates(bundles)
        bundles[1]['scene']['styles']['module']['fill']='white'
        scene,report=merge_region_scenes(manifest(),bundles)
        self.assertEqual(len(scene['nodes']),4);self.assertEqual(len(scene['styles']),2)
        self.assertNotEqual(report['style_map']['region_001']['module'],report['style_map']['region_002']['module'])
        self.assertTrue(any(r['reason']=='possible_duplicate_conflicting_content_or_style'
                            for r in report['review_required']))

    def test_unknown_conflicting_owners_prevent_overlap_dedup(self):
        bundles=fixture();duplicates(bundles)
        bundles[0]['scene']['nodes'][1]['metadata']={'node_id':'unknown_owner_one'}
        bundles[1]['scene']['nodes'][1]['metadata']={'node_id':'unknown_owner_two'}
        scene,report=merge_region_scenes(manifest(),bundles)
        self.assertEqual(len(scene['nodes']),4)
        self.assertEqual(report['counts']['deduplicated'],0)
        self.assertNotEqual(report['id_map']['region_001']['shared'],report['id_map']['region_002']['shared'])
        self.assertEqual(sum(r['reason']=='unresolved_metadata_references' for r in report['review_required']),2)

    def test_same_style_content_is_reused(self):
        scene,report=merge_region_scenes(manifest(),fixture())
        self.assertEqual(len(scene['styles']),1)
        self.assertEqual(report['style_map']['region_001']['module'],report['style_map']['region_002']['module'])

    def test_ambiguous_duplicate_in_one_crop_is_preserved(self):
        bundles=fixture();duplicates(bundles)
        other=deepcopy(bundles[1]['scene']['nodes'][1]);other['id']='shared_again'
        bundles[1]['scene']['nodes'].append(other)
        scene,report=merge_region_scenes(manifest(),bundles)
        self.assertEqual(len(scene['nodes']),5)
        self.assertTrue(any(r['reason']=='ambiguous_duplicate_preserved' for r in report['review_required']))

    def test_confirmed_mock_cross_connection_and_pending_unknown(self):
        bundles=fixture()
        cross={'id':'bridge','source':{'region_id':'region_001','id':'a'},
            'target':{'region_id':'region_002','id':'a'},'from_anchor':'east','to_anchor':'west',
            'arrow':'end','evidence':{'kind':'mock_confirmed','confirmed':True}}
        unknown=deepcopy(cross);unknown.update(id='unknown',target=None)
        bundles[0]['pending_connections']=[cross,unknown]
        scene,report=merge_region_scenes(manifest(),bundles)
        self.assertEqual(len(scene['edges']),1)
        self.assertTrue(scene['edges'][0]['metadata']['mock_only'])
        self.assertEqual(report['counts']['resolved_cross_region'],1)
        self.assertEqual(report['counts']['pending_cross_region'],1)
        self.assertIsNotNone(scene['edges'][0]['from']);self.assertIsNotNone(scene['edges'][0]['to'])

    def test_unconfirmed_or_non_mock_assertion_does_not_create_edge(self):
        for origin,confirmed in [('mock_not_model_recognition',False),('real_unverified',True)]:
            bundles=fixture();bundles[0]['metadata']['source']=origin
            bundles[0]['pending_connections']=[{'id':'bridge','source':{'region_id':'region_001','id':'a'},
                'target':{'region_id':'region_002','id':'a'},'evidence':{'kind':'mock_confirmed','confirmed':confirmed}}]
            scene,report=merge_region_scenes(manifest(),bundles)
            self.assertEqual(scene['edges'],[]);self.assertEqual(len(report['pending_connections']),1)

    def test_confirmed_global_geometric_path_requires_exact_source_binding(self):
        bundles=fixture()
        bundles[0]['pending_connections']=[{'id':'bridge',
            'source':{'region_id':'region_001','id':'a'},'target':{'region_id':'region_002','id':'a'},
            'from_anchor':'east','to_anchor':'west','arrow':'end',
            'evidence':{'kind':'geometric_path','confirmed':True,'coordinate_system':'global',
                        'source_image_sha256':'a'*64,'points':[[60,115],[200,115]]}}]
        scene,report=merge_region_scenes(manifest(),bundles)
        self.assertEqual(len(scene['edges']),1)
        self.assertEqual(scene['edges'][0]['points'],[[60,115],[200,115]])
        self.assertNotIn('from',scene['edges'][0])
        self.assertEqual(scene['edges'][0]['metadata']['source_id'],report['id_map']['region_001']['a'])
        self.assertFalse(scene['edges'][0]['metadata']['mock_only'])
        self.assertEqual(report['counts']['resolved_cross_region'],1)

    def test_bad_geometric_path_binding_is_deferred_without_endpoint_guessing(self):
        for invalid in ('hash','endpoint','missing_coordinate_system','local_coordinate_system'):
            bundles=fixture()
            evidence={'kind':'geometric_path','confirmed':True,'coordinate_system':'global',
                      'source_image_sha256':'a'*64,'points':[[60,115],[200,115]]}
            if invalid=='hash':evidence['source_image_sha256']='b'*64
            elif invalid=='endpoint':evidence['points'][0]=[70,115]
            elif invalid=='missing_coordinate_system':del evidence['coordinate_system']
            else:evidence['coordinate_system']='local'
            bundles[0]['pending_connections']=[{'id':'bridge',
                'source':{'region_id':'region_001','id':'a'},'target':{'region_id':'region_002','id':'a'},
                'from_anchor':'east','to_anchor':'west','evidence':evidence}]
            with self.subTest(invalid=invalid):
                scene,report=merge_region_scenes(manifest(),bundles)
                self.assertEqual(scene['edges'],[])
                self.assertEqual(report['counts']['pending_cross_region'],1)
                self.assertIn('geometric_path_not_global_or_not_bound',report['pending_connections'][0]['reason'])

    def test_math_rich_text_and_inputs_preserved_and_generator_compatible(self):
        bundles=fixture();s=bundles[1]['scene']
        formula=r'W_E\in\mathbb{R}^{768\times768}'
        s['labels']=[{'id':'math','mode':'math','text':formula,'x':90,'y':190},
            {'id':'rich','mode':'rich','segments':[{'type':'text','content':'中文维度 '},
                {'type':'math','content':r'd=768'}],'x':90,'y':220,'metadata':{'node_id':'a'}}]
        before=deepcopy((manifest(),bundles))
        scene,report=merge_region_scenes(before[0],bundles)
        self.assertEqual((manifest(),bundles),before)
        self.assertEqual(scene['labels'][0]['text'],formula)
        self.assertEqual(scene['labels'][1]['segments'],s['labels'][1]['segments'])
        validate_scene(scene);code=generate_tikz(scene)
        self.assertIn(formula,code);self.assertIn('中文维度',code)
        again,audit=merge_region_scenes(manifest(),bundles)
        self.assertEqual(scene,again);self.assertEqual(report,audit)

    def test_unknown_metadata_geometry_and_owner_are_scoped_and_reported(self):
        bundles=fixture();bundles[1]['scene']['nodes'][0]['metadata']={
            'crop_bbox':[1,2,3,4],'node_id':'missing'}
        scene,report=merge_region_scenes(manifest(),bundles)
        node=scene['nodes'][1]
        self.assertNotIn('crop_bbox',node['metadata']);self.assertNotIn('node_id',node['metadata'])
        self.assertEqual(node['metadata']['region_local_metadata']['coordinate_system'],'local')
        self.assertTrue(report['review_required'])

    def test_missing_region_or_scale_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):merge_region_scenes(manifest(),fixture()[:1])
        bundles=fixture();bundles[1]['scene']['canvas']['scale']=.5
        with self.assertRaises(ValueError):merge_region_scenes(manifest(),bundles)


if __name__=='__main__':
    unittest.main()
