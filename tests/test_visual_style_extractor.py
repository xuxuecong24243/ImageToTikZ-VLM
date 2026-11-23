"""Generic raster evidence tests, no fixed production-image expectations."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from PIL import Image,ImageDraw
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from visual_style_extractor import extract_visual_styles


@unittest.skipUnless(importlib.util.find_spec('cv2'),'OpenCV is optional')
class VisualStyleExtractionTests(OfflineTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'样式 图.png'
    def run_image(self,image,boxes=(),rows=()):
        image.save(self.path)
        geometry={'image':{'width':image.width,'height':image.height,
            'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest()},
            'shapes':[{'id':'shape_%d'%i,'type':'rectangle','bbox':list(box)} for i,box in enumerate(boxes)]}
        return extract_visual_styles(self.path,list(rows),geometry)
    def rectangle(self,fill='#BFDCEC',stroke='#283F50',width=3,radius=0):
        image=Image.new('RGB',(360,240),'white');d=ImageDraw.Draw(image)
        d.rounded_rectangle((30,30,329,209),radius=radius,fill=fill,outline=stroke,width=width)
        return image
    def test_pure_fill_measured_without_palette_quantization(self):
        r=self.run_image(self.rectangle(),[[30,30,300,180]])
        self.assertEqual(r['shapes'][0]['visual_style']['fill_color'],'#BFDCEC')
        self.assertEqual(r['shapes'][0]['evidence']['fill']['status'],'measured_uniform')
    def test_nearwhite_fill_is_not_replaced_by_page_white(self):
        r=self.run_image(self.rectangle(fill='#F1F6FB'),[[30,30,300,180]])
        self.assertEqual(r['shapes'][0]['visual_style']['fill_color'],'#F1F6FB')
        self.assertEqual(r['page_background']['visual_style']['fill_color'],'#FFFFFF')
    def test_border_color_and_source_pixel_thickness(self):
        r=self.run_image(self.rectangle(width=4),[[30,30,300,180]])
        s=r['shapes'][0]['visual_style']
        self.assertEqual(s['stroke_color'],'#283F50');self.assertEqual(s['line_width'],4)
        self.assertEqual(r['shapes'][0]['evidence']['stroke']['width_interval_px'],[3,5])
    def test_missing_border_does_not_invent_width(self):
        image=Image.new('RGB',(360,240),'white');ImageDraw.Draw(image).rectangle((30,30,329,209),fill='#BFDCEC')
        r=self.run_image(image,[[30,30,300,180]])['shapes'][0]
        self.assertEqual(r['visual_style']['stroke'],'none');self.assertNotIn('line_width',r['visual_style'])
    def test_measured_corner_distinguishes_sharp_and_round(self):
        straight=self.run_image(self.rectangle(),[[30,30,300,180]])['shapes'][0]
        rounded=self.run_image(self.rectangle(radius=22),[[30,30,300,180]])['shapes'][0]
        self.assertEqual(straight['visual_style']['rounded_radius'],0)
        self.assertGreater(rounded['visual_style']['rounded_radius'],8)
        self.assertLess(rounded['visual_style']['rounded_radius'],30)
        self.assertEqual(rounded['type'],'rounded_rectangle')
    def test_nonuniform_fill_is_explicit_and_not_global_average(self):
        image=Image.new('RGB',(360,240),'white');d=ImageDraw.Draw(image)
        for y in range(30,210):d.line((30,y,329,y),fill=(30+y,70,min(255,250-y)))
        r=self.run_image(image,[[30,30,300,180]])['shapes'][0]
        self.assertEqual(r['evidence']['fill']['status'],'uncertain_nonuniform')
        self.assertEqual(r['visual_style']['fill'],'none');self.assertNotIn('fill_color',r['visual_style'])
    def test_ocr_mask_excludes_dark_text_from_fill_statistics(self):
        image=self.rectangle();d=ImageDraw.Draw(image);d.rectangle((60,60,280,170),fill='#101020')
        row={'id':'text1','bbox':[60,60,221,111],'text':'Preserved __name__','confidence':.2}
        r=self.run_image(image,[[30,30,300,180]],[row])
        self.assertEqual(r['shapes'][0]['visual_style']['fill_color'],'#BFDCEC')
        self.assertEqual(row['text'],'Preserved __name__')
    def test_light_foreground_on_dark_header(self):
        image=Image.new('RGB',(360,240),'white');d=ImageDraw.Draw(image)
        d.rectangle((25,25,330,80),fill='#304878');d.rectangle((55,45,130,55),fill='white')
        row={'id':'ocrA','bbox':[45,35,110,35],'text':'ANY','confidence':.9}
        r=self.run_image(image,[],[row])
        self.assertEqual(r['text_styles']['ocrA']['visual_style']['text_color_rgb'],'#FFFFFF')
        self.assertTrue(any(s['visual_style'].get('fill_color')=='#304878' for s in r['regions']))
    def test_dark_strip_separates_from_same_color_panel_outline(self):
        image=Image.new('RGB',(500,350),'white');d=ImageDraw.Draw(image)
        d.rectangle((30,30,460,320),outline='#305070',width=2)
        d.rectangle((30,30,460,75),fill='#305070')
        r=self.run_image(image)
        strip=[s for s in r['regions'] if s['visual_style'].get('fill_color')=='#305070']
        self.assertEqual(len(strip),1);self.assertLess(strip[0]['bbox'][3],55)
        # Without a supplied enclosing panel, geometry only claims a compact
        # visual card; role never comes from the recognized heading string.
        self.assertIn(strip[0]['role_candidate'],{'title_band','child_card','main_container'})
    def test_nested_color_components_are_retained(self):
        image=Image.new('RGB',(500,350),'white');d=ImageDraw.Draw(image)
        d.rectangle((30,30,460,320),fill='#E2ECF8')
        d.rectangle((80,100,230,240),fill='#BFDAE8')
        r=self.run_image(image)
        colors={s['visual_style'].get('fill_color') for s in r['regions']}
        self.assertTrue({'#E2ECF8','#BFDAE8'}<=colors)
    def test_thin_separator_is_not_a_filled_panel(self):
        image=Image.new('RGB',(500,300),'white');d=ImageDraw.Draw(image)
        d.line((20,100,470,100),fill='#435567',width=2)
        r=self.run_image(image)
        self.assertEqual(r['regions'],[])
    def test_input_identity_mismatch_and_invalid_bbox_fail(self):
        image=self.rectangle();image.save(self.path)
        with self.assertRaises(ValueError):extract_visual_styles(self.path,[],{'image':{},'shapes':[]})
        with self.assertRaises(ValueError):self.run_image(image,[[350,20,50,50]])
    def test_deterministic_evidence_and_no_source_mutation(self):
        image=self.rectangle(radius=15);row={'id':'ocr1','bbox':[90,70,60,20],'text':'A_1','confidence':.7}
        a=self.run_image(image,[[30,30,300,180]],[row]);b=self.run_image(image,[[30,30,300,180]],[row])
        self.assertEqual(a,b);self.assertEqual(a['api_calls'],0)
        self.assertEqual(a['shapes'][0]['bbox'],[30,30,300,180]);self.assertEqual(row['text'],'A_1')
