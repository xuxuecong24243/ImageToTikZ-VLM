"""Synthetic source ink, typography classes and measured-layout constraints."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
from PIL import Image,ImageDraw
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from text_layout_analysis import analyze_source_text,analyze_text_collisions,classify_text,equivalent_math_segments


def diagram():
    return {'version':'1.0','canvas':{'id':'canvas','width':400,'height':300,'coordinate_system':'top-left'},
        'regions':[{'id':'panel','type':'panel','bbox':[20,20,250,200],'parent_id':None,'children':['a','b','border']}],
        'elements':[{'id':'a','type':'text','bbox':[35,35,80,20],'region_id':'panel','raw_text':'A','text_type':'plain','reading_order':0},
                    {'id':'b','type':'text','bbox':[150,35,80,20],'region_id':'panel','raw_text':'B','text_type':'plain','reading_order':1},
                    {'id':'border','type':'rectangle','bbox':[20,20,250,200],'region_id':'panel'},
                    {'id':'link','type':'arrow','bbox':[10,240,280,1],'region_id':None,'points':[[10,240],[290,240]]}],
        'connections':[]}


class TextLayoutAnalysisTests(OfflineTestCase):
    def test_chinese_title_body_and_mixed_categories(self):
        self.assertEqual(classify_text('中文标题',heading=True)['text_kind'],'chinese_title')
        self.assertEqual(classify_text('中文正文')['text_kind'],'chinese_body')
        self.assertEqual(classify_text('English prose')['text_kind'],'english_prose')
        self.assertEqual(classify_text('中文 English')['text_kind'],'mixed_chinese_english')
        self.assertEqual(classify_text('Header',heading=True)['font_candidates'][0],'sans')

    def test_python_identifiers_and_punctuation_preserved(self):
        literals=['__getitem__(index)','self.encoded_texts[index]','pad_token_id',
                  'value = {"key": "A_2", "path": "C:\\x"}',"value = 'case_2'","value = 'α×2'",'取 self.data[index]']
        for literal in literals:
            with self.subTest(literal=literal):
                out=classify_text(literal)
                self.assertIn(out['text_kind'],['python_code','mixed_code_chinese'])
                self.assertEqual(out['font_candidates'],['mono'])
                self.assertIsNone(out['render_segments'])

    def test_exact_unicode_scripts_and_symbols_without_formula_guessing(self):
        result=equivalent_math_segments('x² ∈ ℝ，α × 2，v₀')
        expressions=[s['content'] for s in result if s['type']=='math']
        for exact in ['x^{2}',r'\in',r'\mathbb{R}',r'\alpha',r'\times','v_{0}']:
            self.assertIn(exact,expressions)
        self.assertEqual(''.join(s['content'] for s in result if s['type']=='text'),'  ，  2，')
        self.assertIsNone(equivalent_math_segments('Epos E IR1x4x256'))

    def test_ambiguous_formula_stays_literal_and_warns(self):
        for value in ['Epos[None,:, :] E R1x4x256','We E IR50257×256','x^2','Q=12','m = -2.5']:
            classified=classify_text(value)
            self.assertEqual(classified['text_kind'],'math_uncertain')
            self.assertIn('unresolved_ocr_formula',[d['code'] for d in classified['diagnostics']])
        self.assertIsNone(classify_text('x_1',text_type='code')['render_segments'])

    def test_explicit_supported_math_uses_existing_contract(self):
        value=r'\frac{x_1}{\alpha^2}'
        result=classify_text(value,text_type='math')
        self.assertEqual(result['text_kind'],'math_expression')
        self.assertEqual(result['font_candidates'],['serif'])
        self.assertIsNone(result['render_segments'])

    def source_inputs(self,image,rows):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        path=Path(temporary.name)/'文字 源.png';image.save(path)
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        ir={'version':'1.0','canvas':{'id':'canvas','width':image.width,'height':image.height,'coordinate_system':'top-left'},
            'regions':[],'elements':[{'id':'t_'+r['id'],'type':'text','bbox':r['bbox'][:],'region_id':None,
             'raw_text':r['text'],'text_type':'plain','reading_order':i} for i,r in enumerate(rows)],'connections':[],
            'provenance':{'image_sha256':sha}}
        fusion={'text_matches':[{'source_id':r['id'],'element_id':'t_'+r['id']} for r in rows]}
        visual={'image':{'sha256':sha},'text_styles':{}}
        return path,ir,fusion,visual

    def test_source_ink_measured_inside_original_bbox_without_mutation(self):
        image=Image.new('RGB',(200,120),'white');draw=ImageDraw.Draw(image)
        draw.rectangle((35,30,43,49),fill='black');draw.rectangle((51,30,59,49),fill='black')
        draw.line((20,21,129,21),fill='black',width=1) # adjacent border, not ink
        rows=[{'id':'row','text':'AB','bbox':[20,20,110,40]}]
        path,ir,fusion,visual=self.source_inputs(image,rows);before=deepcopy((ir,rows))
        result=analyze_source_text(path,rows,ir,fusion,visual)
        self.assertEqual(result['items']['t_row']['ink_bbox'],[35,30,25,20])
        self.assertEqual(result['items']['t_row']['bbox'],rows[0]['bbox'])
        self.assertEqual((ir,rows),before)

    def test_light_ink_on_dark_background(self):
        image=Image.new('RGB',(160,80),'#254D75');draw=ImageDraw.Draw(image)
        draw.rectangle((35,22,60,42),fill='white')
        rows=[{'id':'row','text':'原文','bbox':[20,10,90,50]}]
        path,ir,fusion,visual=self.source_inputs(image,rows)
        visual['text_styles']['row']={'evidence':{'status':'measured_foreground_candidate','rgb':[255,255,255],'background_rgb':[37,77,117]}}
        result=analyze_source_text(path,rows,ir,fusion,visual)['items']['t_row']
        self.assertEqual(result['ink_bbox'],[35,22,26,21])

    def test_blank_ink_and_source_hash_mismatch_are_explicit(self):
        rows=[{'id':'row','text':'Unseen','bbox':[10,10,80,20]}]
        path,ir,fusion,visual=self.source_inputs(Image.new('RGB',(120,80),'white'),rows)
        result=analyze_source_text(path,rows,ir,fusion,visual)['items']['t_row']
        self.assertIsNone(result['ink_bbox']);self.assertEqual(result['ink_status'],'unavailable_no_reliable_ink')
        visual['image']['sha256']='x'*64
        with self.assertRaises(ValueError):analyze_source_text(path,rows,ir,fusion,visual)

    def test_relative_heading_and_multiline_retained(self):
        rows=[{'id':'a','text':'标题','bbox':[10,5,180,40]},
              {'id':'b','text':'第一行\nsecond','bbox':[10,80,100,20]},
              {'id':'c','text':'正文','bbox':[10,120,100,20]}]
        path,ir,fusion,visual=self.source_inputs(Image.new('RGB',(220,180),'white'),rows)
        result=analyze_source_text(path,rows,ir,fusion,visual)['items']
        self.assertEqual(result['t_a']['text_kind'],'chinese_title')
        self.assertEqual(result['t_b']['raw_text'],'第一行\nsecond')
        self.assertIn('observed_multiline',[d['code'] for d in result['t_b']['diagnostics']])

    def test_measured_overlap_and_spacing_use_actual_extents(self):
        ir=diagram();unchanged=deepcopy(ir)
        overlap=analyze_text_collisions(ir,{'a':[35,35,150,20],'b':[150,35,80,20]})
        self.assertEqual(overlap['counts']['text_overlap'],1)
        self.assertEqual(ir,unchanged)
        spaced=analyze_text_collisions(ir,{'a':[35,35,80,20],'b':[116,35,80,20]})
        self.assertEqual(spaced['counts']['same_line_gap'],1)

    def test_container_page_and_border_conflicts(self):
        ir=diagram()
        out=analyze_text_collisions(ir,{'a':[-2,35,90,20],'b':[21,40,80,20]})
        self.assertEqual(out['counts']['page_overflow'],1)
        self.assertEqual(out['counts']['container_overflow'],1)
        self.assertEqual(out['counts']['border_proximity'],1)
        self.assertGreater(out['counts']['graphic_boundary'],0)

    def test_arrow_and_diamond_boundary_crossings(self):
        ir=diagram();ir['elements'].append({'id':'decision','type':'diamond','bbox':[280,80,100,80],'region_id':None})
        out=analyze_text_collisions(ir,{'a':[50,232,100,20],'b':[320,78,30,15]})
        self.assertEqual(out['counts']['connector_intersection'],1)
        self.assertTrue(any(i.get('graphic_id')=='decision' for i in out['issues']))

    def test_background_containment_is_not_a_conflict(self):
        ir=diagram();ir['elements'][2]['visual_style']={'fill_color':'#BFDCEC','stroke':'none'}
        out=analyze_text_collisions(ir,{'a':[35,35,80,20],'b':[150,35,80,20]})
        self.assertEqual(out['counts']['total_conflicts'],0)

    def test_invalid_or_unknown_rendered_box_rejected(self):
        ir=diagram()
        for boxes in [{'missing':[1,1,20,20]},{'a':[1,2,True,20]},{'a':[1,1,float('nan'),20]}]:
            with self.assertRaises(ValueError):analyze_text_collisions(ir,boxes)
