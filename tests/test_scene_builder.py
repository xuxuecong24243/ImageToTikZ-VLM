"""Offline conversion tests; fixture candidates do not claim recognition accuracy."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from scene_builder import (SceneBuildError, build_scene_draft, build_report,
                           compare_reference, edge_from_candidate, main, read_geometry)
from structured_tikz.validator import validate_scene


def geometry():
    return {'schema_version':'1.0', 'image':{'width':400,'height':200},
            'coordinate_system':'top-left-normalized', 'axes':{'x':'right','y':'down'},
            'rectangles':[{'id':'rect_001','bbox_px':{'x':40,'y':20,'width':100,'height':60},
                           'bbox':{'x':.1,'y':.1,'width':.25,'height':.3},
                           'score':.72, 'evidence':{'methods':['chromatic_component_bounding_box']}}],
            'lines':[{'id':'line_001','pixel':{'x1':150,'y1':60,'x2':220,'y2':60},'classification':'geometry'},
                     {'id':'line_002','pixel':{'x1':150,'y1':70,'x2':220,'y2':70},'classification':'uncertain'},
                     {'id':'line_003','pixel':{'x1':150,'y1':80,'x2':170,'y2':80},'classification':'text_stroke_candidate'}],
            'text_regions':[{'id':'text_001','bbox_px':[40,5,100,10],
                             'source':'heuristic_connected_components','confidence':None}],
            'regions':[], 'custom_diagnostic':{'retain':'unknown fields are not silently discarded'}}


class SceneBuilderTests(OfflineTestCase):
    def test_pixel_bbox_and_source_id_are_preserved(self):
        result = build_scene_draft(geometry(),400,200)
        node = result['nodes'][0]
        self.assertEqual(node['id'],'rect_001')
        self.assertEqual(node['type'],'rectangle')
        self.assertEqual(node['bbox'],[40,20,100,60])
        self.assertEqual(node['metadata']['source_id'],'rect_001')
        self.assertEqual(node['metadata']['review_status'],'unverified')
        self.assertEqual(node['metadata']['source_candidate'],geometry()['rectangles'][0])
        self.assertEqual(result['metadata']['source_geometry_metadata']['custom_diagnostic'],geometry()['custom_diagnostic'])
        validate_scene(result)

    def test_normalized_only_bbox_and_lines_are_explicitly_converted(self):
        data = geometry()
        del data['rectangles'][0]['bbox_px']
        data['lines'] = [{'id':'line_001','x1':.25,'y1':.2,'x2':.5,'y2':.2,'classification':'geometry'}]
        result = build_scene_draft(data,400,200)
        self.assertEqual(result['nodes'][0]['bbox'],[40,20,100,60])
        self.assertEqual(result['nodes'][0]['metadata']['coordinate_mapping'],'normalized_bbox_scaled_to_pixels')
        self.assertEqual(result['metadata']['review_candidates']['lines'][0]['points'],[[100,40],[200,40]])

    def test_uncertain_preserved_text_strokes_excluded_but_auditable(self):
        result = build_scene_draft(geometry(),400,200)
        candidates = result['metadata']['review_candidates']
        self.assertEqual([c['id'] for c in candidates['lines']],['line_001','line_002'])
        self.assertEqual(candidates['lines'][1]['classification'],'uncertain')
        self.assertEqual(candidates['excluded_lines'][0]['source_candidate'],geometry()['lines'][2])
        self.assertEqual(result['edges'],[])
        self.assertTrue(all(c['arrow']=='unknown' for c in candidates['lines']))

    def test_text_boxes_never_become_recognized_labels(self):
        result = build_scene_draft(geometry(),400,200)
        self.assertEqual(result['labels'],[])
        self.assertIsNone(result['metadata']['review_candidates']['text_regions'][0]['recognized_text'])
        self.assertNotIn('text',result['nodes'][0])

    def test_selection_requires_explicit_candidate_and_does_not_infer_direction(self):
        result = build_scene_draft(geometry(),400,200)
        edge = edge_from_candidate(result,'line_002','edge_manual')
        self.assertEqual(edge['points'],[[150,70],[220,70]])
        self.assertEqual(edge['arrow'],'none')
        self.assertNotIn('from',edge)
        self.assertNotIn('to',edge)
        self.assertEqual(edge['metadata']['review_status'],'unverified')
        self.assertEqual(result['edges'],[])
        with self.assertRaises(SceneBuildError):
            edge_from_candidate(result,'line_003','e')

    def test_missing_classification_is_uncertain_and_reported(self):
        data=geometry();del data['lines'][0]['classification']
        result=build_scene_draft(data,400,200)
        self.assertEqual(result['metadata']['review_candidates']['lines'][0]['classification'],'uncertain')
        self.assertTrue(result['metadata']['conversion_warnings'])

    def test_deterministic_without_modifying_input(self):
        data=geometry();before=copy.deepcopy(data)
        self.assertEqual(build_scene_draft(data,400,200),build_scene_draft(copy.deepcopy(data),400,200))
        self.assertEqual(data,before)
        self.assertEqual(build_scene_draft(data,400,200)['metadata']['review_status'],'unverified')

    def test_invalid_ids_coordinates_and_dimensions_fail_explicitly(self):
        cases=[]
        data=geometry();data['lines'][0]['id']='rect_001';cases.append(data)
        data=geometry();data['rectangles'][0]['bbox_px']['width']=0;cases.append(data)
        data=geometry();data['rectangles'][0]['bbox_px']['x']=float('nan');cases.append(data)
        data=geometry();data['rectangles'][0]['bbox']['x']=.2;cases.append(data)
        data=geometry();data['lines'][0]['classification']='invented';cases.append(data)
        data=geometry();data['coordinate_system']='bottom-left';cases.append(data)
        for data in cases:
            with self.subTest(data=data),self.assertRaises(SceneBuildError):
                build_scene_draft(data,400,200)
        with self.assertRaises(SceneBuildError):
            build_scene_draft(geometry(),800,200)

    def test_file_reader_invalid_json_and_cli_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'outputs') as directory:
            root=Path(directory);source=root/'geometry.json';output=root/'new'/'scene_draft.json'
            source.write_text(json.dumps(geometry()),encoding='utf-8')
            self.assertEqual(read_geometry(source),geometry())
            self.assertEqual(main(['--geometry',str(source),'--output',str(output)]),0)
            original=output.read_bytes()
            with self.assertRaises(SystemExit) as caught:
                main(['--geometry',str(source),'--output',str(output)])
            self.assertEqual(caught.exception.code,2)
            self.assertEqual(output.read_bytes(),original)
            for text in ('{','[]','{"image":{},"image":{}}','{"x":NaN}'):
                source.write_text(text,encoding='utf-8')
                with self.subTest(text=text),self.assertRaises(SceneBuildError):
                    read_geometry(source)

    def test_reference_match_is_report_only_and_tensor_extent_is_not_front(self):
        draft=build_scene_draft(geometry(),400,200)
        reference={'canvas':draft['canvas'].copy(),'styles':{},'edges':[],'labels':[],
                   'nodes':[{'id':'human_tensor','type':'tensor3d','bbox':[40,30,90,50],
                             'grid':{'rows':3,'cols':4},'depth':{'dx':10,'dy':-10}}]}
        before=copy.deepcopy(draft)
        report=compare_reference(draft,reference)
        self.assertEqual(report['matched_nodes'],1)
        self.assertEqual(report['matches'][0]['iou_reference_extent'],1)
        self.assertLess(report['matches'][0]['iou_reference_front_bbox'],1)
        self.assertEqual(draft,before)
        self.assertEqual(draft['nodes'][0]['type'],'rectangle')
        self.assertEqual(report['automatically_recognized_tensor_count'],0)

    def test_report_counts_are_from_real_collections(self):
        report=build_report(build_scene_draft(geometry(),400,200))
        expected={'converted_nodes':1,'nodes_requiring_review':1,'input_lines':3,
                  'retained_line_candidates':2,'excluded_text_stroke_candidates':1,
                  'uncertain_line_candidates':1,'automatic_edges':0,'automatic_labels':0,
                  'unrecognized_text_regions':1}
        for key,value in expected.items():
            self.assertEqual(report[key],value)


if __name__=='__main__':
    unittest.main()
