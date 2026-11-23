"""Real CV on generated geometric fixtures; no mocked detector results."""
from copy import deepcopy
from pathlib import Path
import tempfile
from PIL import Image,ImageDraw
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from local_arrow_detection import analyze_local_arrows


def arrow_fixture(direction='right',color='#2464A5',bent=False):
    image=Image.new('RGB',(180,180),'white');draw=ImageDraw.Draw(image)
    route=[(25,90),(125,90)] if not bent else [(35,25),(35,90),(125,90)]
    draw.line(route,fill=color,width=3)
    draw.polygon([(120,80),(138,90),(120,100)],fill=color)
    turns={'right':0,'up':1,'left':2,'down':3}[direction]
    return image.rotate(turns*90)


class LocalArrowDetectionTests(OfflineTestCase):
    def setUp(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.folder=Path(folder.name)

    def run_image(self,image,ocr=None,geometry=None):
        path=self.folder/'箭头 source.png';image.save(path)
        return analyze_local_arrows(path,ocr or [],geometry)

    def test_horizontal_head_and_original_pixel_path(self):
        report=self.run_image(arrow_fixture())
        self.assertEqual(len(report['arrows']),1)
        arrow=report['arrows'][0]
        self.assertEqual(arrow['direction'],'right')
        self.assertEqual(arrow['status'],'confirmed_arrow')
        self.assertLess(arrow['points'][0][0],arrow['points'][-1][0])
        self.assertAlmostEqual(arrow['points'][-1][0],138,delta=1)
        self.assertEqual(arrow['style']['stroke_color'],'#2464A5')
        self.assertGreater(arrow['evidence']['head_fill_support'],.8)
        self.assertGreater(arrow['evidence']['head_to_shaft_ratio'],2)
        self.assertEqual(report['api_calls'],0)

    def test_vertical_and_reverse_directions_follow_head_not_reading_order(self):
        for direction in ('down','up','left'):
            with self.subTest(direction=direction):
                arrow=self.run_image(arrow_fixture(direction))['arrows']
                self.assertEqual(len(arrow),1)
                self.assertEqual(arrow[0]['direction'],direction)

    def test_bent_arrow_retains_observed_elbow_and_direction(self):
        arrow=self.run_image(arrow_fixture(bent=True))['arrows']
        self.assertEqual(len(arrow),1)
        points=arrow[0]['points']
        self.assertGreaterEqual(len(points),4)
        self.assertLess(points[0][1],points[-1][1])
        self.assertEqual(arrow[0]['evidence']['route'],'orthogonal_polyline')
        self.assertEqual(arrow[0]['direction'],'right')

    def test_neutral_arrow_not_restricted_to_blue(self):
        for color in ('#222222','#B63655','#258942'):
            with self.subTest(color=color):
                r=self.run_image(arrow_fixture(color=color))
                self.assertEqual(len(r['arrows']),1)
                self.assertEqual(r['arrows'][0]['style']['stroke_color'],color)

    def test_ordinary_line_and_table_border_not_arrows(self):
        im=Image.new('RGB',(300,220),'white');d=ImageDraw.Draw(im)
        d.line((20,30,160,30),fill='#2464A5',width=3)
        d.rectangle((20,70,260,190),outline='#2464A5',width=3)
        d.line((120,70,120,190),fill='#2464A5',width=3)
        r=self.run_image(im)
        self.assertEqual(r['arrows'],[])
        self.assertTrue(r['rejected_candidates'])

    def test_triangle_without_shaft_is_not_a_connection(self):
        im=Image.new('RGB',(180,180),'white');d=ImageDraw.Draw(im)
        d.polygon([(40,50),(80,70),(40,90)],fill='#2464A5')
        r=self.run_image(im)
        self.assertEqual(r['arrows'],[])
        self.assertEqual(r['rejected_candidates'][0]['reason'],'no_supported_filled_arrowhead')

    def test_ocr_contained_arrow_glyph_is_not_a_flow_edge(self):
        rows=[{'bbox':[10,65,140,50],'text':'A → B','id':'text'}]
        before=deepcopy(rows);r=self.run_image(arrow_fixture(),rows)
        self.assertEqual(r['arrows'],[]);self.assertEqual(rows,before)
        self.assertEqual(r['counts']['text_components_excluded'],1)

    def test_branching_route_is_explicitly_uncertain(self):
        im=arrow_fixture();ImageDraw.Draw(im).line((65,90,65,145),fill='#2464A5',width=3)
        r=self.run_image(im)
        self.assertEqual(r['arrows'],[])
        self.assertTrue(any(x['reason']=='branching_or_closed_shaft' for x in r['rejected_candidates']))

    def test_open_head_not_silently_assumed_filled(self):
        im=Image.new('RGB',(180,180),'white');d=ImageDraw.Draw(im)
        d.line((25,90,138,90),fill='#2464A5',width=3)
        d.line((120,80,138,90,120,100),fill='#2464A5',width=3)
        r=self.run_image(im)
        self.assertEqual(r['arrows'],[])

    def test_disconnected_head_cannot_create_invented_shaft(self):
        im=Image.new('RGB',(180,180),'white');d=ImageDraw.Draw(im)
        d.line((25,90,105,90),fill='#2464A5',width=3)
        d.polygon([(120,80),(138,90),(120,100)],fill='#2464A5')
        self.assertEqual(self.run_image(im)['arrows'],[])

    def test_measured_base_line_evidence_reused_and_geometry_identity_checked(self):
        im=arrow_fixture();r=self.run_image(im)
        geometry={'image':r['image'],'lines':[{'id':'old_line','pixel':{'x1':25,'y1':90,'x2':120,'y2':90}}]}
        before=deepcopy(geometry);result=self.run_image(im,geometry=geometry)
        self.assertEqual(result['arrows'][0]['evidence']['base_line_ids'],['old_line'])
        self.assertEqual(geometry,before)
        geometry['image']['sha256']='0'*64
        with self.assertRaises(ValueError):self.run_image(im,geometry=geometry)

    def test_blank_input_and_deterministic_ids(self):
        self.assertEqual(self.run_image(Image.new('RGB',(100,100),'white'))['arrows'],[])
        one=self.run_image(arrow_fixture());two=self.run_image(arrow_fixture())
        self.assertEqual(one,two)

