"""Optional typography keeps original text, geometry and legacy render bytes."""
from copy import deepcopy
try:
    from .offline_support import OfflineTestCase
    from .ir_fixtures import example_ir
    from .test_visual_style_contract import nested_scene
except ImportError:
    from offline_support import OfflineTestCase
    from ir_fixtures import example_ir
    from test_visual_style_contract import nested_scene

from structured_ir.schema import ir_schema
from structured_ir.validator import inspect_ir
from structured_ir.converter import convert_ir
from structured_tikz.generator import generate_tikz
from structured_tikz.primitives import render_label_text,text_measurement_commands
from structured_tikz.validator import validate_scene,escape_text


class TextTypographyContractTests(OfflineTestCase):
    def scene(self,text='中文 title_2'):
        return {'canvas':{'width':400,'height':220,'coordinate_system':'top-left'},
            'styles':{},'nodes':[{'id':'text1','type':'text','text':text,'bbox':[30,30,300,70],'wrap':False}],
            'edges':[],'labels':[]}

    def test_explicit_families_and_baseline_spacing_only_change_text_options(self):
        for family,command in [('serif',r'\rmfamily'),('sans',r'\sffamily'),('mono',r'\ttfamily')]:
            scene=self.scene();scene['nodes'][0]['style']={'font_family':family,'font_size':30,'line_spacing':1.6}
            before=deepcopy(scene);tex=generate_tikz(scene)
            self.assertIn(r'\fontsize{12pt}{19.2pt}\selectfont'+command,tex)
            self.assertEqual(scene,before)
            self.assertIn('(72pt,62pt)',tex)

    def test_font_scale_is_local_and_keeps_old_size_limit(self):
        scene=self.scene();scene['nodes'][0]['style']={'font_size':80,'font_scale':1.5}
        tex=generate_tikz(scene)
        self.assertIn(r'\fontsize{48pt}{60pt}',tex)
        self.assertIn('(72pt,62pt)',tex)
        self.assertNotIn('scale=',tex)
        scene['nodes'][0]['style']['font_size']=81
        with self.assertRaises(ValueError):validate_scene(scene)

    def test_invalid_font_parameters_rejected_by_ir_and_scene(self):
        cases=[('font_family','Arial'),('font_family',False),('font_family',r'\input{x}'),
               ('line_spacing',.79),('line_spacing',2.21),('line_spacing',True),
               ('font_scale',.49),('font_scale',3.01),('font_scale',float('nan')),
               ('font_scale',False),('preserve_spaces',1)]
        for key,value in cases:
            with self.subTest(field=key,value=value):
                data=example_ir();data['elements'][0]['visual_style']={key:value}
                report=inspect_ir(data);self.assertFalse(report['valid'])
                self.assertTrue(any(e['path']=='/elements/0/visual_style/'+key for e in report['errors']))
                scene=self.scene();scene['nodes'][0]['style']={key:value}
                with self.assertRaises(ValueError):validate_scene(scene)

    def test_code_escapes_once_and_preserves_spaces_case_braces_and_quotes(self):
        literal='  __getitem__(Index_2)[0] = {"Path": r"C:\\x_1"} # 42'
        node={'mode':'text','text':literal}
        rendered=render_label_text(node,preserve_spaces=True)
        self.assertEqual(rendered,escape_text(literal).replace(' ',r'\ '))
        for fragment in [r'\_\_getitem\_\_(Index\_2)[0]',r'\{',r'\}',r'\textbackslash{}',r'\#',r'\ \ ']:
            self.assertIn(fragment,rendered)
        self.assertNotIn(r'\textbackslash{}_',rendered)
        self.assertEqual(node['text'],literal)
        scene=self.scene(literal);scene['nodes'][0]['style']={'font_family':'mono','preserve_spaces':True}
        self.assertIn(rendered,generate_tikz(scene))

    def test_multiline_and_mixed_text_keeps_newline_and_literal_content(self):
        text='中文 Code[x_2]\n  return "A_1"'
        scene=self.scene(text);scene['nodes'][0]['style']={'font_family':'mono','preserve_spaces':True,'line_spacing':1.4}
        tex=generate_tikz(scene)
        self.assertIn(r'中文\ Code[x\_2]\\\ \ return\ "A\_1"',tex)
        self.assertEqual(scene['nodes'][0]['text'],text)

    def test_ir_render_segments_preserve_raw_provenance_and_supported_math(self):
        data=example_ir();item=next(e for e in data['elements'] if e['type']=='text')
        item['raw_text']='原文 α x_2^3 / 2';item['render_segments']=[
            {'type':'text','content':'原文 '},
            {'type':'math','content':r'\alpha + \frac{x_2^3}{2}'},
            {'type':'text','content':'  Code[x_2]'}]
        item['visual_style']={'font_family':'sans','font_scale':1.1}
        data['measure_text']=True;before=deepcopy(data)
        self.assertTrue(inspect_ir(data)['valid'])
        scene,report=convert_ir(data);self.assertTrue(report['complete'])
        node=next(n for n in scene['nodes'] if n['id']==item['id'])
        self.assertEqual(node['text'],item['raw_text'])
        self.assertEqual(node['segments'],item['render_segments'])
        self.assertEqual(node['bbox'],item['bbox']);self.assertEqual(data,before)
        self.assertTrue(scene['canvas']['measure_text'])
        self.assertIn(r'$\alpha + \frac{x_2^3}{2}$',generate_tikz(scene))

    def test_render_segment_structure_and_math_injection_fail(self):
        cases=[[],[{'type':'unknown','content':'x'}],[{'type':'text','content':3}]]
        for segments in cases:
            data=example_ir();data['elements'][0]['render_segments']=segments
            with self.subTest(segments=segments):self.assertFalse(inspect_ir(data)['valid'])
        data=example_ir();item=next(e for e in data['elements'] if e['type']=='text')
        item['render_segments']=[{'type':'math','content':r'\input{private}'}]
        scene,report=convert_ir(data)
        self.assertFalse(report['complete'])
        self.assertTrue(any(d['element_id']==item['id'] and d['code']=='scene_representation_rejected' for d in report['diagnostics']))
        self.assertFalse(any(n['id']==item['id'] for n in scene['nodes']))

    def test_render_segments_require_original_text_and_text_element(self):
        data=example_ir();shape=next(e for e in data['elements'] if e['type'] not in {'text','code','math','multiline_text'})
        shape['render_segments']=[{'type':'text','content':'unowned text'}]
        self.assertFalse(inspect_ir(data)['valid'])
        self.assertNotIn('render_segments',ir_schema()['$defs']['element']['required'])

    def test_measurement_commands_are_safe_and_use_actual_anchor_extents(self):
        text='\n'.join(text_measurement_commands('t-node_1','node_1'))
        self.assertIn(r'\pgfpointanchor{t-node_1}{west}',text)
        self.assertIn(r'\pgfpointanchor{t-node_1}{east}',text)
        self.assertIn(r'\typeout{ITTEXT|node_1|\the\dimen0|\the\dimen2}',text)
        for name,ident in [('t-x};\\input{x}','safe'),('safe','x|20pt'),('safe',None)]:
            with self.assertRaises(ValueError):text_measurement_commands(name,ident)

    def test_opt_in_measurement_after_text_in_legacy_and_layered_scenes(self):
        for layered in (False,True):
            scene=self.scene();scene['canvas'].update(layered=layered,measure_text=True)
            scene['labels']=[{'id':'caption','x':100,'y':180,'text':'Caption'}]
            tex=generate_tikz(scene)
            self.assertEqual(tex.count('ITTEXT|text1|'),1)
            self.assertEqual(tex.count('ITTEXT|caption|'),1)
            self.assertLess(tex.index('(t-text1) at'),tex.index('ITTEXT|text1|'))
            self.assertLess(tex.index('(l-caption) at'),tex.index('ITTEXT|caption|'))

    def test_circle_measurement_is_only_after_visible_text_in_layered_mode(self):
        scene=nested_scene();scene['canvas']['measure_text']=True
        scene['nodes'].append({'id':'circ','type':'text','background_shape':'circle',
            'bbox':[240,100,30,30],'text':'X','style':{'font_size':6}})
        tex=generate_tikz(scene)
        self.assertEqual(tex.count('ITTEXT|circ|'),1)
        self.assertLess(tex.index('% text: circ'),tex.index('ITTEXT|circ|'))

    def test_measure_flag_is_strict_and_absence_preserves_defaults(self):
        scene=self.scene();old=generate_tikz(scene)
        self.assertNotIn('ITTEXT',old)
        self.assertNotIn(r'\rmfamily',old)
        scene['canvas']['measure_text']=False
        self.assertEqual(old,generate_tikz(scene))
        for bad in [1,'true',None]:
            scene['canvas']['measure_text']=bad
            with self.assertRaises(ValueError):validate_scene(scene)
            data=example_ir();data['measure_text']=bad
            self.assertFalse(inspect_ir(data)['valid'])
