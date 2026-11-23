"""Literal visual styles and opt-in layering preserve the old rendering contract."""
from copy import deepcopy
import re
try:
    from .offline_support import OfflineTestCase
    from .ir_fixtures import example_ir, branch_ir
except ImportError:
    from offline_support import OfflineTestCase
    from ir_fixtures import example_ir, branch_ir

from structured_ir.schema import ir_schema
from structured_ir.validator import inspect_ir
from structured_ir.converter import convert_ir
from structured_tikz.schema import scene_contract
from structured_tikz.validator import validate_scene
from structured_tikz.generator import generate_tikz


def nested_scene():
    return {'canvas':{'width':300,'height':200,'coordinate_system':'top-left','layered':True},
            'styles':{},'nodes':[
                {'id':'child','type':'rounded_rectangle','bbox':[40,40,90,60],'z_index':2,
                 'style':{'fill_color':'#F3F8FC','stroke_color':'#B8CDD9','rounded_radius':7.5,'line_width':2.5}},
                {'id':'parent','type':'rectangle','bbox':[10,10,200,160],'z_index':0,
                 'style':{'fill_color':'#224F72','stroke_color':'#445566'}},
                {'id':'text','type':'text','text':'原文_9','bbox':[50,50,60,20],'z_index':3,
                 'style':{'text_color_rgb':'#FFFDFB'}}],
            'edges':[{'id':'route','points':[[20,130],[180,130]],'arrow':'end','style':{'stroke_color':'#ABCDEF'}}],
            'labels':[]}


class VisualStyleContractTests(OfflineTestCase):
    def test_ir_style_overlay_is_valid_and_does_not_change_text_or_geometry(self):
        data=example_ir(); data['layered']=True
        data['elements'][0].update(style_role='box',visual_style={'fill_color':'#E1ECF7','line_width':2,'opacity':.8},z_index=4)
        before=deepcopy(data)
        scene,report=convert_ir(data,style_map={'box':{'stroke':'blue','fill':'white'}})
        self.assertTrue(report['complete']);validate_scene(scene)
        self.assertEqual(data,before)
        self.assertEqual(scene['canvas']['layered'],True)
        node=scene['nodes'][0]
        self.assertEqual(node['style'],{'stroke':'blue','fill':'white','fill_color':'#E1ECF7','line_width':2,'opacity':.8})
        self.assertEqual(node['z_index'],4)
        self.assertEqual(node['bbox'],data['elements'][0]['bbox'])
        self.assertEqual(node['text'],data['elements'][0]['raw_text'])

    def test_ir_invalid_styles_report_specific_fields(self):
        cases=[('fill_color','#FFF'),('stroke_color','#123456\\input{x}'),('text_color_rgb','#FFFFFF\n'),
               ('fill_color',False),('line_width',True),('line_width',0),('rounded_radius',51),
               ('opacity',-0.01),('opacity',float('nan')),('font_size',81),('invented',1)]
        for field,value in cases:
            data=example_ir(); data['elements'][0]['visual_style']={field:value}
            with self.subTest(field=field,value=value):
                report=inspect_ir(data);self.assertFalse(report['valid'])
                self.assertTrue(any(e['path']=='/elements/0/visual_style/'+field for e in report['errors']))

    def test_ir_z_index_and_layered_are_strict_optional_types(self):
        for field,value in [('z_index',True),('z_index',1.0),('z_index',-1),('z_index',10001)]:
            data=example_ir();data['elements'][0][field]=value
            with self.subTest(value=value):self.assertFalse(inspect_ir(data)['valid'])
        data=example_ir();data['layered']='yes';self.assertFalse(inspect_ir(data)['valid'])
        self.assertNotIn('layered',ir_schema()['$defs']['ir']['required'])

    def test_scene_color_overrides_reject_code_and_named_field_stays_enum(self):
        data=nested_scene()
        for value in ['red','123456','#FFF','#123456; draw=none','#123456\n',None,False]:
            data['nodes'][0]['style']['fill_color']=value
            with self.subTest(value=value),self.assertRaises(ValueError):validate_scene(data)
        data=nested_scene();data['nodes'][0]['style']['fill']='#123456'
        with self.assertRaises(ValueError):validate_scene(data)
        self.assertIn('none',scene_contract()['json_schema']['$defs']['style']['properties']['stroke']['enum'])

    def test_scene_optional_numbers_reject_bool_nonfinite_and_out_of_range(self):
        for key,value in [('opacity',True),('opacity',float('inf')),('opacity',1.1),('line_width',11)]:
            data=nested_scene();data['nodes'][0]['style'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_scene(data)
        for collection in ['nodes','edges','labels']:
            data=nested_scene()
            if collection=='labels':data['labels']=[{'id':'lab','x':100,'y':100,'text':'x'}]
            data[collection][0]['z_index']=1.5
            with self.subTest(collection=collection),self.assertRaises(ValueError):validate_scene(data)
        data=nested_scene();data['canvas']['layered']=1
        with self.assertRaises(ValueError):validate_scene(data)

    def test_tikz_exact_color_width_radius_and_opacity(self):
        data=nested_scene();data['nodes'][0]['style']['opacity']=.6
        tex=generate_tikz(data)
        self.assertIn(r'\definecolor{scenergbF3F8FC}{HTML}{F3F8FC}',tex)
        exact_definitions=[line for line in tex.splitlines() if line.startswith(r'\definecolor{scenergb')]
        self.assertTrue(exact_definitions)
        self.assertTrue(all(line.endswith('%') for line in exact_definitions),
                        'Exact RGB definitions must not add newline glue before the TikZ hbox.')
        self.assertIn('fill=scenergbF3F8FC',tex)
        self.assertIn('draw=scenergbB8CDD9',tex)
        self.assertIn('line width=1pt',tex)
        self.assertIn('rounded corners=3pt',tex)
        self.assertIn('opacity=0.6',tex)
        self.assertIn('text=scenergbFFFDFB',tex)
        self.assertNotIn('includegraphics',tex)

    def test_nested_fills_then_borders_edges_and_text(self):
        data=nested_scene();before=deepcopy(data);tex=generate_tikz(data)
        self.assertEqual(before,data)
        order=['% fill: parent','% fill: child','% stroke: parent','% stroke: child','% edge: route','% text: text']
        positions=[tex.index(s) for s in order]
        self.assertEqual(positions,sorted(positions))
        self.assertIn(r'原文\_9',tex)
        fillpart=tex[tex.index('% fill: child'):tex.index('% stroke: parent')]
        self.assertIn('draw=none',fillpart)
        strokepart=tex[tex.index('% stroke: child'):tex.index('% edge: route')]
        self.assertIn('fill=none',strokepart)

    def test_layer_order_is_stable_for_equal_ranks_and_uses_supplied_ranks(self):
        data=nested_scene(); data['nodes'][0]['z_index']=0;data['nodes'][1]['z_index']=0
        tex=generate_tikz(data)
        self.assertLess(tex.index('% fill: child'),tex.index('% fill: parent'))
        data['nodes'][1]['z_index']=5
        self.assertLess(generate_tikz(data).index('% fill: child'),generate_tikz(data).index('% fill: parent'))

    def test_anchors_remain_unique_even_for_fill_named_nodes(self):
        data=nested_scene();data['nodes'].append({'id':'fill_parent','type':'rectangle','bbox':[230,10,20,30]})
        data['edges'].append({'id':'refs','from':'child','to':'fill_parent','arrow':'end'})
        tex=generate_tikz(data)
        names=re.findall(r'\((n-[A-Za-z0-9_-]+)\) at',tex)
        self.assertEqual(len(names),len(set(names)))
        self.assertIn('(n-child.east) -- (n-fill_parent.west)',tex)
        self.assertEqual(tex.count('(n-child) at'),1)

    def test_circle_glyphs_are_visible_only_after_edges_and_geometry_is_native(self):
        data=nested_scene();data['nodes'].append({'id':'circ','type':'text','background_shape':'circle',
            'bbox':[230,50,40,40],'text':'Circle_1','align':'right','style':{'fill_color':'#00AA88'}})
        tex=generate_tikz(data)
        self.assertEqual(tex.count('text opacity=0'),2)
        self.assertIn('circle,minimum size=16pt',tex)
        self.assertLess(tex.index('% edge: route'),tex.index('% text: circ'))
        final=tex[tex.index('% text: circ'):]
        self.assertNotIn('text opacity=0',final)
        self.assertIn('draw=none',final);self.assertIn('fill=none',final)
        self.assertIn('align=center',final);self.assertNotIn('align=right',final)
        self.assertIn(r'Circle\_1',final)

    def test_existing_matrix_and_diamond_semantics_survive_opt_in_layers(self):
        data=nested_scene();data['nodes'].append({'id':'mat','type':'matrix','bbox':[220,100,60,50],
            'grid':{'rows':2,'cols':3},'style':{'fill_color':'#D0D8FF'}})
        tex=generate_tikz(data);self.assertIn('% stroke: mat',tex)
        data=branch_ir();data['layered']=True
        scene,report=convert_ir(data);self.assertTrue(report['complete'])
        text=generate_tikz(scene)
        self.assertIn('diamond_text_fit decision',text)
        self.assertEqual(scene['metadata']['ir_connections'],data['connections'])

    def test_connection_style_and_order_are_mapped_without_endpoint_changes(self):
        data=example_ir();conn=data['connections'][0]
        conn.update(visual_style={'stroke_color':'#1122aa','opacity':.5},z_index=9)
        scene,report=convert_ir(data);self.assertTrue(report['complete'])
        edge=scene['edges'][0]
        self.assertEqual(edge['from'],conn['source_id']);self.assertEqual(edge['to'],conn['target_id'])
        self.assertEqual(edge['style'],conn['visual_style']);self.assertEqual(edge['z_index'],9)
        self.assertIn(r'\definecolor{scenergb1122AA}{HTML}{1122AA}',generate_tikz(scene))

    def test_absent_styles_or_explicit_false_keep_legacy_rendering(self):
        data=example_ir();scene,_=convert_ir(data)
        self.assertNotIn('layered',scene['canvas'])
        old=generate_tikz(scene)
        scene['canvas']['layered']=False
        self.assertEqual(old,generate_tikz(scene))
        self.assertNotIn('scenergb',old)
        self.assertNotIn('opacity=',old)

