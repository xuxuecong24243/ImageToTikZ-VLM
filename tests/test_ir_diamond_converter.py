"""Native diamond mapping and evidence-based branch labels; fully offline."""
from copy import deepcopy
try:
    from .offline_support import OfflineTestCase
    from .ir_fixtures import branch_ir
except ImportError:
    from offline_support import OfflineTestCase
    from ir_fixtures import branch_ir
from structured_ir.converter import convert_ir
from structured_tikz.validator import validate_scene


class IRDiamondConverterTests(OfflineTestCase):
    def test_native_id_bbox_text_style_and_merge_preserved(self):
        ir = branch_ir(); ir['elements'][0]['style_role'] = 'decision'; before = deepcopy(ir)
        scene, report = convert_ir(ir, {'decision':{'fill':'light_blue'}})
        self.assertTrue(report['complete']); validate_scene(scene)
        node = scene['nodes'][0]
        for key in ('id','type','bbox'): self.assertEqual(node[key],ir['elements'][0][key])
        self.assertEqual(node['text'],'有效？'); self.assertEqual(node['style'],{'fill':'light_blue'})
        self.assertEqual(set(report['converted_connections']), {'yes','no','join_a','join_b'})
        self.assertEqual(scene['metadata']['ir_connections'],ir['connections']); self.assertEqual(ir,before)

    def test_all_arrow_directions_preserved(self):
        for direction, arrow in [('forward','end'),('reverse','start'),('both','both'),('none','none')]:
            ir = branch_ir(); ir['connections'][0]['direction'] = direction
            scene, report = convert_ir(ir); self.assertTrue(report['complete'])
            self.assertEqual(scene['edges'][0]['arrow'],arrow)

    def test_legal_points_preserved_exactly(self):
        ir = branch_ir(); points = [[300,110],[300,140],[160,140],[160,180]]
        ir['connections'][0]['points'] = points
        scene, report = convert_ir(ir); self.assertTrue(report['complete'])
        self.assertEqual(scene['edges'][0]['points'],points)

    def test_center_boundary_is_rejected_without_deleting_ir(self):
        ir = branch_ir(); ir['connections'][0]['source_anchor'] = 'center'; before = deepcopy(ir)
        scene, report = convert_ir(ir)
        self.assertEqual(report['status'],'partial'); self.assertEqual(ir,before)
        self.assertNotIn('yes',report['converted_connections'])
        self.assertIn('diamond_connection_geometry',[d['code'] for d in report['diagnostics']])

    def test_path_crosses_interior_is_rejected(self):
        ir = branch_ir(); ir['connections'][0].update(source_anchor='north',points=[[300,40],[300,140],[160,180]])
        scene, report = convert_ir(ir)
        self.assertEqual(report['status'],'partial'); self.assertNotIn('yes',report['converted_connections'])
        self.assertIn('diamond_connection_geometry',[d['code'] for d in report['diagnostics']])

    def test_mismatching_explicit_anchor_does_not_draw(self):
        ir = branch_ir(); ir['connections'][0]['points'] = [[360,75],[400,140],[160,180]]
        scene, report = convert_ir(ir)
        self.assertEqual(report['status'],'partial'); self.assertNotIn('yes',report['converted_connections'])
        self.assertIn('diamond_connection_anchor_mismatch',[d['code'] for d in report['diagnostics']])

    def test_exact_existing_branch_text_reused_once(self):
        ir = branch_ir(); label = {'id':'yes_text','type':'text','bbox':deepcopy(ir['connections'][0]['label_bbox']),
                                  'region_id':'panel','raw_text':'是','text_type':'plain','reading_order':4}
        ir['elements'].append(label); before = deepcopy(ir)
        scene, report = convert_ir(ir)
        self.assertTrue(report['complete']); self.assertEqual(ir,before)
        self.assertEqual(scene['edges'][0]['metadata']['branch_label_node_id'],'yes_text')
        self.assertEqual([l['text'] for l in scene['labels']],['否'])
        self.assertEqual(next(n['text'] for n in scene['nodes'] if n['id']=='yes_text'),'是')

    def test_ambiguous_matching_text_keeps_partial(self):
        ir = branch_ir()
        for index in (4,5):
            ir['elements'].append({'id':'label'+str(index),'type':'text','bbox':[140,120,40,40],
                                   'region_id':'panel','raw_text':'是','text_type':'plain','reading_order':index})
        scene, report = convert_ir(ir)
        self.assertEqual(report['status'],'partial')
        self.assertIn('ambiguous_branch_label',[d['code'] for d in report['diagnostics']])

    def test_circle_text_not_reused_as_branch_label(self):
        ir = branch_ir(); ir['elements'].append({'id':'circle','type':'circle','bbox':[140,120,40,40],
                            'region_id':'panel','raw_text':'是','text_type':'plain','reading_order':4})
        scene, report = convert_ir(ir)
        self.assertTrue(report['complete']); self.assertEqual(len(scene['labels']),2)

    def test_unplaced_label_remains_partial(self):
        ir = branch_ir(); del ir['connections'][0]['label_bbox']
        scene, report = convert_ir(ir)
        self.assertEqual(report['status'],'partial'); self.assertIn('yes',report['converted_connections'])
        self.assertIn('unplaced_branch_label',[d['code'] for d in report['diagnostics']])

    def test_unsupported_circle_and_math_not_hidden(self):
        for changes in ({'type':'circle'}, {'text_type':'math','raw_text':r'\unknown{x}'}):
            ir = branch_ir(); ir['elements'][0].update(changes)
            scene, report = convert_ir(ir)
            self.assertEqual(report['status'],'partial'); self.assertTrue(report['unsupported_elements'])
            self.assertEqual(len(ir['connections']),4)

    def test_invalid_ir_endpoints_and_anchors_are_rejected(self):
        for update in ({'target_id':'missing'},{'source_anchor':'invalid'}):
            ir = branch_ir(); ir['connections'][0].update(update)
            with self.assertRaises(ValueError): convert_ir(ir)
