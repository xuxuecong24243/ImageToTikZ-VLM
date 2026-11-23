"""Measured appearance integration, independent of any acceptance image."""
from copy import deepcopy
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from local_fusion import build_local_ir
from local_visual_layers import apply_visual_layers
from structured_ir.validator import inspect_ir
from structured_ir.converter import convert_ir


def sample():
    image={'width':600,'height':400,'sha256':'a'*64}
    ocr={'image':image,'coordinate_system':'top-left','bbox_format':'xywh','rows':[
        {'id':'line_1','text':'A__B(2)','bbox':[45,40,120,20],'confidence':.99,'reading_order':0}]}
    geometry={'image':image,'coordinate_system':'top-left','shapes':[
        {'id':'outer','type':'rectangle','bbox':[10,10,500,350],'style':{'fill':'white','stroke':'blue'}},
        {'id':'card','type':'rectangle','bbox':[30,30,240,120],'style':{'fill':'light_blue'}}],
        'lines':[],'arrow_candidates':[]}
    ir,fusion,styles=build_local_ir(ocr,geometry)
    measured={'image':image,'shapes':[
        {'source_id':'outer','bbox':[10,10,500,350],
         'visual_style':{'fill_color':'#FEFEFE','stroke_color':'#445566','line_width':2},'role_candidate':'main_container'},
        {'source_id':'card','bbox':[30,30,240,120],'type':'rounded_rectangle',
         'visual_style':{'fill_color':'#EFF5FC','stroke_color':'#BCD1E7','rounded_radius':6},'role_candidate':'child_card'}],
        'text_styles':{'line_1':{'visual_style':{'text_color_rgb':'#FFFFFF'},'evidence':{'status':'measured'}}},
        'page_background':{'bbox':[0,0,600,400],'visual_style':{'fill_color':'#FFFFFF','stroke':'none'}},
        'regions':[{'id':'visual_region_0001','bbox':[30,30,240,40],'type':'title_band',
                    'visual_style':{'fill_color':'#244E70','stroke':'none'},'role_candidate':'title_band'}]}
    return ir,fusion,styles,measured


class LocalVisualLayerTests(OfflineTestCase):
    def test_existing_content_identity_and_inputs_are_preserved(self):
        args=sample(); before=deepcopy(args)
        styled,styles,report=apply_visual_layers(*args)
        self.assertEqual(args,before)
        self.assertEqual(styles,args[2])
        original={e['id']:e for e in args[0]['elements']}
        for element in styled['elements']:
            if element['id'] in original:
                old=original[element['id']]
                for key in ('id','bbox','region_id','raw_text','text_type','reading_order','confidence'):
                    self.assertEqual(element.get(key),old.get(key))
        self.assertEqual(report['counts']['preserved_text_elements'],1)
        self.assertTrue(all(report['preservation'].values()))

    def test_nested_fills_follow_geometry_then_separate_strokes_and_text(self):
        args=sample(); styled,_,report=apply_visual_layers(*args)
        byid={e['id']:e for e in styled['elements']}
        fill_order=report['paint_order']['fills']
        self.assertEqual(fill_order[0],'local_visual_page_background')
        self.assertLess(byid['local_shape_0001']['z_index'],byid['local_shape_0002']['z_index'])
        self.assertLess(byid['local_shape_0002']['z_index'],byid['local_visual_visual_region_0001']['z_index'])
        self.assertGreater(byid['local_text_0001']['z_index'],max(byid[i]['z_index'] for i in fill_order))
        self.assertEqual(report['paint_order']['borders'],fill_order)
        self.assertEqual(report['paint_order']['text'],['local_text_0001'])

    def test_children_rebuilt_without_changing_old_ownership(self):
        styled,_,_=apply_visual_layers(*sample())
        for region in styled['regions']:
            expected=[r['id'] for r in styled['regions'] if r['parent_id']==region['id']]
            expected += [e['id'] for e in styled['elements'] if e['region_id']==region['id']]
            self.assertEqual(region['children'],expected)
        self.assertTrue(inspect_ir(styled)['valid'])
        ids=[styled['canvas']['id']]+[o['id'] for k in ('regions','elements','connections') for o in styled[k]]
        self.assertEqual(len(ids),len(set(ids)))

    def test_exact_color_maps_through_ir_to_scene_with_existing_font(self):
        styled,styles,_=apply_visual_layers(*sample())
        scene,report=convert_ir(styled,style_map=styles)
        self.assertTrue(report['complete'])
        text=next(n for n in scene['nodes'] if n['id']=='local_text_0001')
        self.assertEqual(text['text'],'A__B(2)')
        self.assertEqual(text['style']['text_color_rgb'],'#FFFFFF')
        self.assertEqual(text['style']['font_size'],16.4)
        card=next(n for n in scene['nodes'] if n['id']=='local_shape_0002')
        self.assertEqual(card['type'],'rounded_rectangle')
        self.assertEqual(card['style']['rounded_radius'],6)
        self.assertEqual(card['style']['fill_color'],'#EFF5FC')
        self.assertTrue(scene['canvas']['layered'])

    def test_repeated_boundary_reuses_original_id_and_box(self):
        args=sample(); args[3]['regions']=[{'id':'same','type':'rounded_rectangle','bbox':[30,30,240,119],
            'visual_style':{'fill_color':'#E0F0F8','stroke':'none','rounded_radius':7},'role_candidate':'child_card'}]
        styled,styles,report=apply_visual_layers(*args)
        self.assertEqual(len(report['matched_visual_regions']),1)
        card=next(e for e in styled['elements'] if e['id']=='local_shape_0002')
        self.assertEqual(card['bbox'],[30,30,240,120])
        self.assertEqual(card['visual_style']['fill_color'],'#E0F0F8')
        self.assertEqual(card['visual_style']['stroke_color'],'#BCD1E7')
        scene,converted=convert_ir(styled,style_map=styles)
        self.assertTrue(converted['complete'])
        actual=next(e for e in scene['nodes'] if e['id']==card['id'])
        self.assertEqual(actual['style']['stroke_color'],'#BCD1E7')
        self.assertEqual(len(styled['elements']),len(args[0]['elements'])+1)

    def test_new_evidence_can_render_previously_unknown_contour_without_reassigning_ids(self):
        args=sample()
        args[3]['shapes'].append({'source_id':'unknown_contour','bbox':[20,365,500,25],
            'type':'rounded_rectangle','visual_style':{'fill_color':'#ECF4FC','stroke':'none','rounded_radius':8},
            'role_candidate':'background_fill','evidence':{'fill':{'status':'approximate_nonuniform'}}})
        styled,_,report=apply_visual_layers(*args)
        new=next(e for e in styled['elements'] if e['id']=='local_visual_visual_shape_unknown_contour')
        self.assertEqual(new['bbox'],[20,365,500,25])
        self.assertEqual(new['type'],'rounded_rectangle')
        self.assertIsNone(new['region_id'])
        self.assertTrue({e['id'] for e in args[0]['elements']} <= {e['id'] for e in styled['elements']})
        self.assertTrue(inspect_ir(styled)['valid'])

    def test_missing_style_preserves_defaults_and_uncertain_fill_is_not_invented(self):
        ir,fusion,styles,_=sample()
        raw={'regions':[{'id':'unknown','bbox':[300,20,100,50],'type':'rectangle',
            'visual_style':{'fill':'none'},'evidence':{'fill':{'status':'uncertain_nonuniform'}}}]}
        styled,newstyles,report=apply_visual_layers(ir,fusion,styles,raw)
        self.assertEqual(newstyles,styles)
        self.assertEqual(len(styled['elements']),len(ir['elements']))
        self.assertTrue(all('visual_style' not in e for e in styled['elements']))
        self.assertTrue(any(d['code']=='uncertain_fill_not_added' for d in report['diagnostics']))

    def test_ambiguous_region_owner_and_non_nested_overlap_are_reported(self):
        ir,fusion,styles,raw=sample()
        # Add another legitimate, non-nested region that also contains the patch.
        ir['regions'].append({'id':'cross_region','type':'unknown','bbox':[20,20,300,150],
            'parent_id':'local_region_0001','children':[],'uncertainty':'candidate'})
        ir['regions'][0]['children'].append('cross_region')
        ir['elements'].append({'id':'cross_shape','type':'rectangle','bbox':[20,20,300,150],
            'region_id':'cross_region'})
        ir['regions'][-1]['children'].append('cross_shape')
        # The old card is contained here, so a new sibling overlap is needed.
        ir['regions'][-1]['bbox']=[20,20,220,150]
        ir['elements'][-1]['bbox']=[20,20,220,150]
        raw['regions']=[{'id':'overlap_patch','bbox':[45,40,150,40],'type':'title_band',
            'visual_style':{'fill_color':'#334466'}}]
        styled,_,report=apply_visual_layers(ir,fusion,styles,raw)
        patch=next(e for e in styled['elements'] if e['id']=='local_visual_overlap_patch')
        self.assertIsNone(patch['region_id'])
        self.assertTrue(report['overlaps'])
        self.assertTrue(any(d['code']=='ambiguous_patch_owner' for d in report['diagnostics']))
        self.assertTrue(inspect_ir(styled)['valid'])

    def test_semantic_connection_and_measured_path_remain_byte_equivalent(self):
        args=sample(); ir=args[0]
        ir['connections']=[{'id':'meaningful','source_id':'local_shape_0001','target_id':'local_shape_0002',
            'direction':'forward','source_anchor':'east','target_anchor':'west','kind':'flow',
            'points':[[510,185],[520,185],[520,90],[30,90]]}]
        ir['elements'].append({'id':'measured_line','type':'connector','bbox':[300,300,100,1],
            'region_id':'local_region_0001','points':[[300,300],[400,300]]})
        ir['regions'][0]['children'].append('measured_line')
        styled,_,report=apply_visual_layers(*args)
        self.assertEqual(styled['connections'],ir['connections'])
        line=next(e for e in styled['elements'] if e['id']=='measured_line')
        self.assertEqual(line['points'],[[300,300],[400,300]])
        self.assertIn('meaningful',report['paint_order']['connections'])

    def test_wrong_source_and_out_of_bounds_candidate_fail_without_clipping(self):
        for mode in ('hash','dimensions','bbox','page'):
            args=sample()
            if mode=='hash': args[3]['image']=dict(args[3]['image'],sha256='b'*64)
            if mode=='dimensions': args[3]['image']=dict(args[3]['image'],width=601)
            if mode=='bbox': args[3]['regions'][0]['bbox']=[590,10,30,30]
            if mode=='page': args[3]['page_background']['bbox']=[0,0,599,400]
            with self.subTest(mode=mode),self.assertRaises(ValueError): apply_visual_layers(*args)

    def test_invalid_style_or_text_mutation_cannot_bypass_validator(self):
        for mode in ('rgb','font','width','id'):
            args=sample()
            if mode=='rgb': args[3]['regions'][0]['visual_style']['fill_color']='#12;draw=red'
            if mode=='font': args[3]['text_styles']['line_1']['visual_style']['font_size']=80
            if mode=='width': args[3]['shapes'][0]['visual_style']['line_width']=float('nan')
            if mode=='id': args[3]['regions'][0]['id']='bad-id'
            with self.subTest(mode=mode),self.assertRaises(ValueError): apply_visual_layers(*args)

    def test_deterministic_layers_do_not_depend_on_candidate_input_order(self):
        args=sample()
        args[3]['regions'].append({'id':'another','bbox':[300,200,150,80],'type':'rectangle',
            'visual_style':{'fill_color':'#ECF2F8'},'role_candidate':'child_card'})
        first,_,report=apply_visual_layers(*args)
        args[3]['regions'].reverse()
        second,_,again=apply_visual_layers(*args)
        self.assertEqual(report['paint_order'],again['paint_order'])
        self.assertEqual({e['id']:e['z_index'] for e in first['elements']},
                         {e['id']:e['z_index'] for e in second['elements']})
