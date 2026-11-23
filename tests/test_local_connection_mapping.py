"""Pixel-arrow evidence is distinct from inferred endpoint ownership."""
from copy import deepcopy

try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from local_connection_mapping import map_local_connections
from structured_ir.validator import validate_ir
from structured_ir.converter import convert_ir
from structured_tikz.generator import generate_tikz


def graph():
    return {'version':'1.0','canvas':{'id':'canvas','width':500,'height':400,'coordinate_system':'top-left'},
        'regions':[{'id':'r1','type':'panel','bbox':[20,30,140,180],'parent_id':None,'children':['left','caption']},
                   {'id':'r2','type':'panel','bbox':[210,30,140,180],'parent_id':None,'children':['right']}],
        'elements':[{'id':'left','type':'rectangle','bbox':[20,30,140,180],'region_id':'r1',
                     'visual_style':{'fill_color':'#DBEAFE','stroke_color':'#123456'},'z_index':1},
                    {'id':'right','type':'rounded_rectangle','bbox':[210,30,140,180],'region_id':'r2',
                     'visual_style':{'fill_color':'#DBEAFE','rounded_radius':8},'z_index':2},
                    {'id':'caption','type':'text','bbox':[30,50,100,25],'region_id':'r1',
                     'raw_text':'原文 __ID_2','text_type':'plain','reading_order':0,
                     'visual_style':{'font_size':24,'font_family':'mono'},'z_index':3}],
        'connections':[],'layered':True,'provenance':{'image_sha256':'a'*64}}


def arrow(points=None,ident='detected_1'):
    points=points or [[166,120],[204,120]]
    return {'id':ident,'status':'confirmed_arrow','points':points,
        'head':{'tip':points[-1][:],'width_px':14},
        'evidence':{'method':'synthetic_pixel_fixture','shaft_width_px':2,'head_to_shaft_ratio':7},
        'style':{'stroke_color':'#2464A5','line_width':2}}


class LocalConnectionMappingTests(OfflineTestCase):
    def test_horizontal_evidence_maps_cross_region_and_preserves_source(self):
        ir=graph();before=deepcopy(ir);styles={'original':{'font_size':21}}
        result,report,copied=map_local_connections(ir,{'arrows':[arrow()]},styles)
        self.assertEqual(ir,before);self.assertEqual(result['elements'],before['elements'])
        self.assertEqual(result['regions'],before['regions']);self.assertEqual(copied,styles)
        edge=result['connections'][0]
        self.assertEqual((edge['source_id'],edge['target_id']),('left','right'))
        self.assertEqual((edge['source_anchor'],edge['target_anchor']),('east','west'))
        self.assertEqual(edge['points'],arrow()['points']);self.assertEqual(edge['direction'],'forward')
        self.assertTrue(report['mappings'][0]['cross_region']);validate_ir(result)
        scene,conversion=convert_ir(result,copied)
        self.assertTrue(conversion['scene_valid']);self.assertEqual(scene['edges'][0]['arrow'],'end')
        self.assertEqual(scene['edges'][0]['points'],edge['points'])
        self.assertEqual(conversion['status'],'partial') # measured 6px endpoint gaps are not rerouted
        self.assertIn('connection_path_anchor_mismatch',[d['code'] for d in conversion['diagnostics']])

    def test_nested_card_body_coincident_side_is_one_endpoint_family(self):
        ir=graph();ir['regions'].append({'id':'body','type':'panel','bbox':[21,90,138,119],
            'parent_id':'r1','children':['body_shape']})
        ir['regions'][0]['children'].append('body')
        ir['elements'].append({'id':'body_shape','type':'rectangle','bbox':[21,90,138,119],'region_id':'body'})
        result,report,_=map_local_connections(ir,{'arrows':[arrow()]})
        self.assertEqual(result['connections'][0]['source_id'],'left')
        group=report['mappings'][0]['source']['groups'][0]
        self.assertEqual(group['alias_element_ids'],['body_shape'])

    def test_multiple_non_nested_candidates_remain_unresolved(self):
        ir=graph();ir['elements'].append({'id':'other','type':'rectangle','bbox':[80,100,80,140],'region_id':None})
        result,report,_=map_local_connections(ir,{'arrows':[arrow()]})
        self.assertEqual(result['connections'],[])
        self.assertEqual(report['mappings'][0]['source']['status'],'ambiguous')
        self.assertEqual(report['mappings'][0]['status'],'geometric_only')
        self.assertEqual(result['elements'][-1]['type'],'arrow')
        scene,_=convert_ir(result);self.assertEqual(scene['edges'][-1]['arrow'],'end')

    def test_no_head_no_forced_connection_or_arrow(self):
        ir=graph();detected=arrow();detected['status']='line_candidate';detected.pop('head')
        result,report,_=map_local_connections(ir,{'arrows':[detected]})
        self.assertEqual(result,ir);self.assertEqual(report['counts']['rejected'],1)
        self.assertEqual(report['mappings'][0]['diagnostics'][0]['code'],'unconfirmed_head_or_shaft')

    def test_wrong_head_tip_and_invalid_coordinates_are_rejected(self):
        for point in [[205,130],[float('nan'),120],[True,120]]:
            candidate=arrow();candidate['head']['tip']=point
            result,report,_=map_local_connections(graph(),{'arrows':[candidate]})
            self.assertEqual(result['connections'],[]);self.assertEqual(report['counts']['rejected'],1)
        candidate=arrow([[-1,120],[204,120]])
        result,report,_=map_local_connections(graph(),{'arrows':[candidate]})
        self.assertEqual(report['mappings'][0]['diagnostics'][0]['code'],'invalid_measured_path')

    def test_vertical_direction_uses_south_north_anchors(self):
        ir=graph();ir.update(regions=[],elements=[
            {'id':'above','type':'rectangle','bbox':[100,20,100,100],'region_id':None},
            {'id':'below','type':'rectangle','bbox':[100,170,100,100],'region_id':None}])
        result,report,_=map_local_connections(ir,{'arrows':[arrow([[150,125],[150,165]])]})
        self.assertEqual((result['connections'][0]['source_anchor'],result['connections'][0]['target_anchor']),('south','north'))
        self.assertEqual(report['counts']['mapped_connections'],1)

    def test_reverse_direction_follows_tip_not_reading_order(self):
        candidate=arrow([[204,120],[166,120]])
        result,_,_=map_local_connections(graph(),{'arrows':[candidate]})
        edge=result['connections'][0]
        self.assertEqual((edge['source_id'],edge['target_id']),('right','left'))
        self.assertEqual((edge['source_anchor'],edge['target_anchor']),('west','east'))

    def test_elbow_route_preserves_every_bend_and_endpoint_tangents(self):
        ir=graph();ir.update(regions=[],elements=[
            {'id':'above','type':'rectangle','bbox':[20,20,100,100],'region_id':None},
            {'id':'below','type':'rectangle','bbox':[200,200,100,100],'region_id':None}])
        points=[[125,70],[250,70],[250,195]]
        result,report,_=map_local_connections(ir,{'arrows':[arrow(points)]})
        edge=result['connections'][0]
        self.assertEqual(edge['points'],points)
        self.assertEqual((edge['source_anchor'],edge['target_anchor']),('east','north'))
        self.assertEqual(report['counts']['mapped_connections'],1)

    def test_distance_and_side_projection_required_not_just_nearest_nodes(self):
        for points in [[[170,250],[200,250]],[[180,300],[220,300]]]:
            result,report,_=map_local_connections(graph(),{'arrows':[arrow(points)]})
            self.assertEqual(result['connections'],[])
            self.assertEqual(report['mappings'][0]['status'],'geometric_only')

    def test_text_crossing_is_diagnosed_without_moving_content(self):
        ir=graph();ir['elements'].append({'id':'between','type':'text','bbox':[174,110,20,20],
            'region_id':None,'raw_text':'Keep','text_type':'plain','reading_order':1})
        original=deepcopy(ir)
        result,report,_=map_local_connections(ir,{'arrows':[arrow()]})
        self.assertEqual(result,original)
        issue=report['mappings'][0]['diagnostics'][0]
        self.assertEqual(issue['code'],'path_crosses_existing_text');self.assertEqual(issue['element_ids'],['between'])

    def test_strict_duplicate_shaft_removed_but_other_lines_retained(self):
        ir=graph();ir['elements'].extend([
            {'id':'same','type':'connector','bbox':[167,120,28,1],'points':[[167,120],[195,120]],'region_id':None},
            {'id':'long','type':'connector','bbox':[165,125,45,1],'points':[[165,125],[210,125]],'region_id':None},
            {'id':'cross','type':'connector','bbox':[185,110,1,20],'points':[[185,110],[185,130]],'region_id':None}])
        result,report,_=map_local_connections(ir,{'arrows':[arrow()]})
        ids={e['id'] for e in result['elements']}
        self.assertNotIn('same',ids);self.assertIn('long',ids);self.assertIn('cross',ids)
        self.assertEqual(report['deduplicated_lines'][0]['element']['id'],'same')
        validate_ir(result)

    def test_duplicate_paths_and_ids_do_not_duplicate_arrows(self):
        result,report,_=map_local_connections(graph(),{'arrows':[arrow(),arrow(ident='second')]})
        self.assertEqual(len(result['connections']),1);self.assertEqual(report['counts']['rejected'],1)
        with self.assertRaises(ValueError):map_local_connections(graph(),{'arrows':[arrow(),arrow()]})

    def test_wrong_image_identity_rejected_and_output_deterministic(self):
        with self.assertRaises(ValueError):
            map_local_connections(graph(),{'arrows':[],'image':{'sha256':'b'*64}})
        with self.assertRaises(ValueError):
            map_local_connections(graph(),{'arrows':[],'image':{'width':501}})
        args=(graph(),{'arrows':[arrow()]})
        self.assertEqual(map_local_connections(*args),map_local_connections(*args))

    def test_new_arrow_uses_existing_layered_generator_after_background_before_text(self):
        result,report,_=map_local_connections(graph(),{'arrows':[arrow()]})
        scene,_=convert_ir(result);tikz=generate_tikz(scene)
        edge_id=report['mappings'][0]['connection_id']
        self.assertIn('edge: '+edge_id,tikz)
        self.assertLess(tikz.index('% fill: left'),tikz.index('% edge: '+edge_id))
        self.assertLess(tikz.index('% edge: '+edge_id),tikz.index('% text: caption'))
        self.assertIn('2464A5',tikz)
        self.assertNotIn('includegraphics',tikz)

    def test_measured_diamond_gap_stays_drawable_without_changing_diamond_contract(self):
        ir=graph();ir['elements'][0]['type']='diamond'
        result,report,_=map_local_connections(ir,{'arrows':[arrow()]})
        self.assertEqual(result['connections'],[])
        self.assertEqual(report['mappings'][0]['status'],'geometric_only')
        self.assertEqual(result['elements'][0],ir['elements'][0])
        scene,conversion=convert_ir(result)
        self.assertTrue(conversion['complete'])
        self.assertEqual(scene['edges'][-1]['arrow'],'end')
        self.assertEqual(scene['edges'][-1]['points'],arrow()['points'])
