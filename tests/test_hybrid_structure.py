"""Structure experiment tests use real CV on labeled synthetic pixels, no API."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from PIL import Image,ImageDraw
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from experiments.hybrid_structure import parse_structure, fuse_structure, structure_schema
from structured_ir.converter import convert_ir
from structured_ir.validator import validate_ir
from structured_tikz.generator import generate_tikz


def baseline():
    return {'version':'1.0','canvas':{'id':'canvas','width':320,'height':180,'coordinate_system':'top-left'},
        'regions':[], 'elements':[
            {'id':'left','type':'rectangle','bbox':[10,50,60,80],'region_id':None,
             'visual_style':{'fill_color':'#ECF5FF'}},
            {'id':'right','type':'rectangle','bbox':[190,50,60,80],'region_id':None},
            {'id':'text','type':'text','bbox':[15,65,40,20],'region_id':None,
             'raw_text':'中文 __ID_2','text_type':'plain','reading_order':0}],
        'connections':[], 'layered':True}


def supplement():
    return {'groups':[], 'connections':[{'source_id':'left','target_id':'right','source_anchor':'east',
        'target_anchor':'west','relation':'data_flow','line_ids':[],'evidence':'A visible filled arrow joins the faces.',
        'confidence':.97}], 'candidates':[],'unresolved':[]}


class HybridStructureTests(OfflineTestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name)/'混合 pixel source.png'
        self.image=Image.new('RGB',(320,180),'white')
        d=ImageDraw.Draw(self.image)
        d.line([(80,90),(162,90)],fill='#2464A5',width=3)
        d.polygon([(160,80),(180,90),(160,100)],fill='#2464A5')
        self.image.save(self.path)

    def run_fusion(self,s=None,ir=None):
        return fuse_structure(ir or baseline(),s or supplement(),image_path=self.path,arrow_report={})

    def test_strict_json_fence_and_schema(self):
        s=supplement()
        self.assertEqual(parse_structure(json.dumps(s)),s)
        self.assertEqual(parse_structure('```json\n'+json.dumps(s)+'\n```'),s)
        self.assertFalse(structure_schema()['additionalProperties'])
        for bad in ['prefix '+json.dumps(s),'{"groups":[],"groups":[]}',
                    json.dumps(dict(s,points=[[0,0],[1,1]]))]:
            with self.subTest(bad=bad[:40]),self.assertRaises(ValueError):parse_structure(bad)

    def test_reject_model_geometry_text_and_nonfinite(self):
        for key,value in [('points',[[70,90],[190,90]]),('bbox',[1,2,3,4]),('raw_text','new'),('confidence',float('nan'))]:
            s=supplement();s['connections'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):parse_structure(json.dumps(s))

    def test_pixel_confirmed_connection_preserves_every_old_object(self):
        ir=baseline();before=deepcopy(ir);s=supplement();sb=deepcopy(s)
        merged,r=self.run_fusion(s,ir)
        self.assertEqual(ir,before);self.assertEqual(s,sb)
        self.assertTrue(all(r['preservation'].values()));self.assertEqual(r['counts']['connections_added'],1)
        self.assertEqual(merged['elements'],before['elements']);validate_ir(merged)
        c=merged['connections'][0]
        self.assertEqual(c['points'][-1],[180.,90.]);self.assertEqual(c['visual_style']['stroke_color'],'#2464A5')
        scene,report=convert_ir(merged)
        self.assertTrue(report['scene_valid']);self.assertEqual(scene['edges'][0]['arrow'],'end')
        tikz=generate_tikz(scene)
        self.assertIn('% edge: '+c['id'],tikz);self.assertIn(',->]',tikz)

    def test_unknown_ids_are_rejected_without_dropping_old_content(self):
        for field in ('source_id','target_id'):
            s=supplement();s['connections'][0][field]='invented'
            merged,r=self.run_fusion(s)
            self.assertEqual(merged['elements'],baseline()['elements']);self.assertEqual(merged['connections'],[])
            self.assertEqual(r['connections'][0]['status'],'rejected')
        s=supplement();s['connections'][0]['line_ids']=['invented']
        self.assertEqual(self.run_fusion(s)[1]['connections'][0]['status'],'rejected')

    def test_low_confidence_is_not_promoted(self):
        s=supplement();s['connections'][0]['confidence']=.79
        merged,r=self.run_fusion(s)
        self.assertEqual(merged['connections'],[]);self.assertEqual(r['connections'][0]['reason'],'low_model_confidence')

    def test_plain_line_has_no_invented_arrow(self):
        im=Image.new('RGB',(320,180),'white');ImageDraw.Draw(im).line((80,90,180,90),fill='#2464A5',width=3)
        im.save(self.path)
        merged,r=self.run_fusion()
        self.assertEqual(merged['connections'],[]);self.assertEqual(r['connections'][0]['reason'],'no_supported_pixel_arrow')

    def test_wrong_direction_is_not_overridden(self):
        s=supplement();p=s['connections'][0]
        p.update(source_id='right',target_id='left',source_anchor='west',target_anchor='east')
        merged,r=self.run_fusion(s)
        self.assertEqual(merged['connections'],[]);self.assertEqual(r['connections'][0]['reason'],'no_supported_pixel_arrow')

    def test_old_connection_and_duplicate_proposal_remain_single(self):
        merged,r=self.run_fusion();old=deepcopy(merged['connections'])
        s=supplement();s['connections']*=2
        result,report=self.run_fusion(s,merged)
        self.assertEqual(result['connections'],old)
        self.assertEqual([c['status'] for c in report['connections']],['already_present','already_present'])

    def test_reverse_of_reliable_connection_is_conflict(self):
        merged,r=self.run_fusion();s=supplement();p=s['connections'][0]
        p.update(source_id='right',target_id='left',source_anchor='west',target_anchor='east')
        result,report=self.run_fusion(s,merged)
        self.assertEqual(result['connections'],merged['connections'])
        self.assertEqual(report['connections'][0]['reason'],'conflicts_with_existing_direction')

    def test_text_intersection_prevents_new_drawing(self):
        ir=baseline();ir['elements'].append(dict(ir['elements'][2],id='crossing',bbox=[110,80,20,20]))
        merged,r=self.run_fusion(ir=ir)
        self.assertEqual(merged['connections'],[]);self.assertEqual(r['connections'][0]['reason'],'path_crosses_existing_text')

    def test_existing_geometric_shaft_is_not_drawn_twice(self):
        ir=baseline();ir['elements'].append({'id':'shaft','type':'connector','bbox':[80,90,100,1],
            'region_id':None,'points':[[80,90],[180,90]]})
        s=supplement();s['connections'][0]['line_ids']=['shaft']
        merged,r=self.run_fusion(s,ir)
        self.assertEqual(merged['elements'],ir['elements']);self.assertEqual(merged['connections'],[])
        self.assertEqual(r['connections'][0]['status'],'semantic_only')

    def test_containment_verified_groups_are_metadata_only(self):
        s=supplement();s['connections']=[];s['groups']=[{'group_id':'module','element_ids':['text'],
            'container_id':'left','relation':'contains','evidence':'Text inside left box.','confidence':.96}]
        merged,r=self.run_fusion(s)
        self.assertEqual(r['groups'][0]['status'],'contains_verified');self.assertEqual(merged['regions'],[])
        self.assertEqual(merged['elements'],baseline()['elements'])
        s['groups'][0]['element_ids']=['right']
        self.assertEqual(self.run_fusion(s)[1]['groups'][0]['reason'],'containment_conflict')

    def test_cyclic_and_unknown_group_references_are_rejected(self):
        ir=baseline();ir['elements'][1]['bbox']=ir['elements'][0]['bbox'][:]
        s=supplement();s['connections']=[]
        s['groups']=[{'group_id':'g1','element_ids':['right'],'container_id':'left','relation':'contains','evidence':'fixture','confidence':.9},
                     {'group_id':'g2','element_ids':['left'],'container_id':'right','relation':'contains','evidence':'fixture','confidence':.9}]
        merged,r=self.run_fusion(s,ir)
        self.assertTrue(all(x['reason']=='containment_cycle' for x in r['groups']))
        s['groups'][0]['element_ids']=['missing'];s['groups']=s['groups'][:1]
        self.assertEqual(self.run_fusion(s,ir)[1]['groups'][0]['status'],'rejected')

    def test_candidates_do_not_create_verified_shapes_or_glyphs(self):
        s=supplement();s['connections']=[]
        s['candidates']=[{'type':kind,'related_element_ids':['left'],'evidence':'visible in test image',
            'uncertainty':'not measured'} for kind in ('curve','circle_plus','dashed_enclosure')]
        merged,r=self.run_fusion(s)
        self.assertEqual(merged['elements'],baseline()['elements']);self.assertEqual(r['counts']['candidates_unverified'],3)
        validate_ir(merged)

    def test_source_hash_mismatch_stops_fusion(self):
        ir=baseline();ir['provenance']={'image_sha256':'a'*64}
        with self.assertRaisesRegex(ValueError,'hash differs'):self.run_fusion(ir=ir)

    def test_ambiguous_two_pixel_arrows_do_not_choose_nearest(self):
        d=ImageDraw.Draw(self.image);d.line([(80,115),(162,115)],fill='#2464A5',width=3)
        d.polygon([(160,105),(180,115),(160,125)],fill='#2464A5');self.image.save(self.path)
        merged,r=self.run_fusion()
        self.assertEqual(merged['connections'],[]);self.assertEqual(r['connections'][0]['reason'],'ambiguous_pixel_arrows')

    def test_vertical_local_crop_uses_measured_direction(self):
        im=self.image.rotate(90,expand=True);im.save(self.path)
        ir=baseline();ir['canvas'].update(width=180,height=320)
        ir['elements']=ir['elements'][:2]
        ir['elements'][0]['bbox']=[50,250,80,60];ir['elements'][1]['bbox']=[50,70,80,60]
        s=supplement();s['connections'][0].update(source_anchor='north',target_anchor='south')
        merged,r=self.run_fusion(s,ir)
        self.assertEqual(r['counts']['connections_added'],1)
        self.assertGreater(merged['connections'][0]['points'][0][1],merged['connections'][0]['points'][-1][1])
