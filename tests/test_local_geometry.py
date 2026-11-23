"""Synthetic tests for measured candidate geometry, not OCR quality claims."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from PIL import Image,ImageDraw
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from local_geometry import analyze_local_geometry,deduplicate_shapes,quantize_color


class LocalGeometryMathTests(OfflineTestCase):
    def test_palette_keeps_real_measurement_and_error(self):
        r=quantize_color([216,237,248])
        self.assertEqual(r['palette_name'],'light_blue'); self.assertEqual(r['rgb_distance'],0)
        self.assertGreater(quantize_color([240,100,100])['rgb_distance'],0)

    def test_dedup_retains_nested_containers(self):
        def item(box): return {'bbox':box,'score':.8,'type':'rectangle','evidence':{'methods':['test']}}
        original=[item([10,10,200,140]),item([11,11,198,138]),item([30,40,150,80])]
        result,diagnostics=deduplicate_shapes(original)
        self.assertEqual(len(result),2); self.assertTrue(diagnostics)
        self.assertEqual(len(original),3)


@unittest.skipUnless(importlib.util.find_spec('cv2'),'OpenCV dependency not installed')
class LocalGeometryDetectionTests(OfflineTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
    def detect(self,im,ocr=None):
        path=self.root/'中文 图片.png'; im.save(path)
        return analyze_local_geometry(path,ocr or [])
    def near(self,result,box,kind=None):
        return [s for s in result['shapes'] if (kind is None or s['type']==kind) and max(abs(a-b) for a,b in zip(s['bbox'],box))<8]
    def test_nested_real_contours_and_original_coordinate_frame(self):
        im=Image.new('RGB',(500,350),'white'); d=ImageDraw.Draw(im)
        d.rectangle((20,20,460,320),outline='#26445C',width=3)
        d.rectangle((80,80,300,220),outline='#26445C',width=3)
        r=self.detect(im)
        self.assertTrue(self.near(r,[20,20,441,301])); self.assertTrue(self.near(r,[80,80,221,141]))
        self.assertEqual(r['coordinate_system'],'top-left'); self.assertEqual(r['api_calls'],0)
        self.assertEqual(r['lines'],r['base_geometry']['lines'])
    def test_circle_diamond_and_round_rect_candidates(self):
        im=Image.new('RGB',(600,250),'white'); d=ImageDraw.Draw(im)
        d.ellipse((20,40,130,150),outline='#26445C',width=3)
        d.polygon([(250,25),(330,100),(250,175),(170,100)],outline='#26445C',width=3)
        d.rounded_rectangle((370,40,560,160),radius=20,outline='#26445C',width=3)
        r=self.detect(im)
        self.assertTrue(self.near(r,[20,40,111,111],'circle'))
        self.assertTrue(self.near(r,[170,25,161,151],'diamond'))
        self.assertTrue(self.near(r,[370,40,191,121],'rounded_rectangle'))
    def test_ocr_on_border_does_not_erase_original_edges(self):
        im=Image.new('RGB',(400,220),'white'); d=ImageDraw.Draw(im)
        d.rectangle((30,40,360,180),outline='#26445C',width=3)
        row={'id':'ocr1','text':'border overlap','bbox':[20,30,180,30],'confidence':.4}
        r=self.detect(im,[row])
        self.assertTrue(self.near(r,[30,40,331,141]))
        self.assertEqual(r['base_geometry']['text_detection']['ocr_regions'],1)
    def test_fill_measurement_and_background_candidates(self):
        im=Image.new('RGB',(400,240),'white'); d=ImageDraw.Draw(im)
        d.rectangle((30,30,360,200),fill='#D8EDF8',outline='#26445C',width=3)
        r=self.detect(im); candidates=self.near(r,[30,30,331,171])
        self.assertTrue(candidates)
        self.assertEqual(candidates[0]['style']['fill'],'light_blue')
        self.assertEqual(candidates[0]['color_measurements']['fill']['rgb'],[216,237,248])
        self.assertTrue(r['background_regions'])
    def test_blank_image_and_invalid_ocr_are_explicit(self):
        im=Image.new('RGB',(200,120),'white'); r=self.detect(im)
        self.assertEqual(r['shapes'],[]); self.assertEqual(r['arrow_candidates'],[])
        with self.assertRaises(ValueError): self.detect(im,[{'bbox':[195,10,20,20],'confidence':.5}])
    def test_triangle_is_only_unverified_arrow_candidate(self):
        im=Image.new('RGB',(300,180),'white'); d=ImageDraw.Draw(im)
        d.line((20,90,234,90),fill='#26445C',width=3)
        d.polygon([(240,82),(257,90),(240,98)],fill='#26445C')
        r=self.detect(im)
        self.assertTrue(r['lines'])
        self.assertTrue(r['arrow_candidates'])
        self.assertNotIn('connections',r)
        self.assertTrue(all(a['status']=='unverified' and a['direction']=='unknown' for a in r['arrow_candidates']))
