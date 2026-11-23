"""Synthetic geometry contracts, separate from real local OCR acceptance."""
from copy import deepcopy
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from local_fusion import build_local_ir
from structured_ir.validator import inspect_ir
from structured_ir.converter import convert_ir


def inputs():
    image={'width':600,'height':400,'sha256':'a'*64}
    ocr={'image':image,'coordinate_system':'top-left','bbox_format':'xywh','rows':[
        {'id':'row_a','text':'class Example:','bbox':[35,35,160,20],'confidence':.98,'reading_order':0},
        {'id':'row_b','text':'    __getitem__(x_0)','bbox':[35,65,190,20],'confidence':.4,'reading_order':1}]}
    geometry={'image':deepcopy(image),'coordinate_system':'top-left','shapes':[
        {'id':'shape_outer','type':'rectangle','bbox':[10,10,500,350],'style':{'stroke':'dark_blue','fill':'white'}},
        {'id':'shape_inner','type':'rounded_rectangle','bbox':[25,25,240,110],'style':{'stroke':'blue'}}],
        'lines':[],'arrow_candidates':[],'diagnostics':[]}
    return ocr,geometry


class LocalFusionTests(OfflineTestCase):
    def test_nested_ownership_uses_unique_deepest_container(self):
        ocr,g=inputs(); ir,r,styles=build_local_ir(ocr,g)
        inner=next(x for x in ir['regions'] if x['bbox']==[25,25,240,110])
        outer=next(x for x in ir['regions'] if x['bbox']==[10,10,500,350])
        self.assertEqual(inner['parent_id'],outer['id'])
        texts=[e for e in ir['elements'] if 'raw_text' in e]
        self.assertTrue(all(t['region_id']==inner['id'] for t in texts))
        self.assertEqual(r['counts']['preserved_text_elements'],2)

    def test_overlapping_non_nested_owners_stay_ambiguous(self):
        ocr,g=inputs(); g['shapes']=[
            {'id':'a','type':'rectangle','bbox':[10,10,300,250]},
            {'id':'b','type':'rectangle','bbox':[20,20,300,250]}]
        ir,r,_=build_local_ir(ocr,g)
        self.assertTrue(all(e['region_id'] is None for e in ir['elements'] if 'raw_text' in e))
        self.assertTrue(all(t['status']=='ambiguous' for t in r['text_matches']))
        self.assertEqual(len(r['overlaps']),1)

    def test_children_exactly_derived_and_unique_ids(self):
        ocr,g=inputs(); ir,_,_=build_local_ir(ocr,g)
        for region in ir['regions']:
            expected={r['id'] for r in ir['regions'] if r['parent_id']==region['id']}
            expected.update(e['id'] for e in ir['elements'] if e['region_id']==region['id'])
            self.assertEqual(set(region['children']),expected)
        ids=[ir['canvas']['id']]+[v['id'] for k in ('regions','elements','connections') for v in ir[k]]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertTrue(inspect_ir(ir)['valid'])

    def test_multiline_groups_preserve_raw_rows_and_low_confidence(self):
        ocr,g=inputs(); ir,r,_=build_local_ir(ocr,g)
        self.assertEqual([e['raw_text'] for e in ir['elements'] if 'raw_text' in e],[x['text'] for x in ocr['rows']])
        self.assertEqual(r['multiline_groups'][0]['text'],'class Example:\n    __getitem__(x_0)')
        low=next(e for e in ir['elements'] if e.get('confidence')==.4)
        self.assertIn('Low OCR confidence',low['uncertainty'])
        self.assertEqual(r['status'],'partial')

    def test_literal_multiline_and_spatial_reading_order(self):
        ocr,g=inputs(); ocr['rows'][0]['text']='A_1\nB(2)'; ocr['rows'].reverse()
        ir,r,_=build_local_ir(ocr,g)
        text=[e for e in ir['elements'] if 'raw_text' in e]
        self.assertEqual(text[0]['raw_text'],'A_1\nB(2)')
        self.assertEqual(text[0]['type'],'multiline_text')
        self.assertEqual([e['reading_order'] for e in text],[0,1])

    def test_identical_shape_candidates_deduplicated_without_dropping_text(self):
        ocr,g=inputs(); duplicate=deepcopy(g['shapes'][1]); duplicate['id']='duplicate'; g['shapes'].append(duplicate)
        ir,r,_=build_local_ir(ocr,g)
        self.assertEqual(len(r['shape_deduplication']),1)
        self.assertEqual(len(ir['regions']),2)
        self.assertEqual(r['counts']['preserved_text_elements'],2)

    def test_text_outside_all_containers_is_retained_unassigned(self):
        ocr,g=inputs(); ocr['rows'][0]['bbox']=[520,20,70,20]
        ir,r,_=build_local_ir(ocr,g)
        entry=next(e for e in ir['elements'] if e.get('raw_text')=='class Example:')
        self.assertIsNone(entry['region_id'])
        self.assertEqual(entry['bbox'],[520,20,70,20])
        self.assertTrue(inspect_ir(ir)['valid'])

    def test_invalid_bbox_confidence_or_image_binding_rejected(self):
        for kind in ('bbox','confidence','hash','coordinate'):
            ocr,g=inputs()
            if kind=='bbox': ocr['rows'][0]['bbox']=[590,20,20,20]
            if kind=='confidence': ocr['rows'][0]['confidence']=float('nan')
            if kind=='hash': g['image']['sha256']='b'*64
            if kind=='coordinate': ocr['bbox_format']='xyxy'
            with self.subTest(kind=kind),self.assertRaises(ValueError): build_local_ir(ocr,g)

    def test_unknown_shape_is_reported_not_substituted(self):
        ocr,g=inputs(); g['shapes'][1]['type']='unknown'
        ir,r,_=build_local_ir(ocr,g)
        self.assertEqual(len(r['unrendered_shapes']),1)
        self.assertEqual(len(ir['regions']),1)
        self.assertEqual(r['counts']['preserved_text_elements'],2)

    def test_empty_ocr_does_not_fabricate_text_or_connections(self):
        ocr,g=inputs(); ocr['rows']=[]
        ir,r,_=build_local_ir(ocr,g)
        self.assertFalse(any('raw_text' in e for e in ir['elements']))
        self.assertEqual(ir['connections'],[])
        self.assertEqual(r['status'],'partial')
        self.assertEqual(r['counts']['ocr_rows'],0)

    def test_style_mapping_converts_without_mutating_inputs(self):
        ocr,g=inputs(); before=deepcopy((ocr,g)); ir,r,styles=build_local_ir(ocr,g)
        scene,conversion=convert_ir(ir,style_map=styles)
        self.assertTrue(conversion['complete'])
        self.assertEqual((ocr,g),before)
        self.assertTrue(any(n.get('style',{}).get('font_size') for n in scene['nodes']))
        self.assertEqual(build_local_ir(ocr,g),(ir,r,styles))

    def test_geometry_line_is_not_semantic_edge_and_text_crossing_retained_in_report(self):
        ocr,g=inputs()
        def line(ident,a,b):
            return {'id':ident,'pixel':dict(zip(('x1','y1','x2','y2'),a+b)),
                    'classification':'geometry','evidence':{'edge_coverage_fraction':1,'longest_continuous_support_fraction':1}}
        g['lines']=[line('clear',[280,200],[450,200]),line('text_crossing',[30,45],[240,45]),
                    line('border',[10,10],[510,10])]
        ir,r,_=build_local_ir(ocr,g)
        self.assertEqual(ir['connections'],[])
        self.assertEqual(sum(e['type']=='connector' for e in ir['elements']),1)
        self.assertEqual([d['reason'] for d in r['line_decisions']],
                         ['continuous_geometric_line_not_semantic_connection','intersects_ocr_text','already_represented_container_border'])
        self.assertTrue(inspect_ir(ir)['valid'])

    def test_duplicate_source_id_rejected_without_aliasing(self):
        ocr,g=inputs(); ocr['rows'][1]['id']=ocr['rows'][0]['id']
        with self.assertRaisesRegex(ValueError,'unique'): build_local_ir(ocr,g)

    def test_invalid_scene_style_cannot_bypass_style_contract(self):
        ocr,g=inputs(); g['shapes'][0]['style']['fill']='#abcdef'
        with self.assertRaisesRegex(ValueError,'palette'): build_local_ir(ocr,g)
