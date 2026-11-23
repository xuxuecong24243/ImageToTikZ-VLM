import copy
import json
from pathlib import Path
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from scene_coverage import evaluate_coverage


def example():
    return {'canvas':{'width':100,'height':100,'coordinate_system':'top-left'},'styles':{},
        'nodes':[{'id':'A','type':'text','text':'input','bbox':[5,5,20,20]},
                 {'id':'B','type':'text','text':'output','bbox':[60,5,20,20]}],
        'labels':[],'edges':[{'id':'link','from':'A','to':'B','arrow':'end'}]}

def expectations():
    return {'content':[{'id':'start','all_of':['input']},{'id':'end','all_of':['output']}],
        'relations':[{'id':'flow','source':'start','target':'end'}]}


class CoverageTests(OfflineTestCase):
    def test_missing_expected_text_and_arrows_warn_separately_from_valid_format(self):
        scene=example();scene['edges']=[];scene['nodes'][1]['text']='unrelated'
        report=evaluate_coverage(scene,expectations())
        self.assertTrue(report['format']['schema_valid']);self.assertTrue(report['quality_warning'])
        self.assertEqual({x['id'] for x in report['missing']},{'end','flow'})

    def test_no_reference_counts_are_not_semantic_failure(self):
        scene=example();scene['edges']=[]
        report=evaluate_coverage(scene,geometry={'text_regions':[{}]*203})
        self.assertFalse(report['quality_warning']);self.assertEqual(report['missing'],[])
        self.assertIsNone(report['content']['complete']);self.assertTrue(report['unknown'])
        self.assertTrue(report['geometry']['text_candidate_count_is_not_required_text_nodes'])

    def test_declared_direction_can_be_checked_without_claiming_visual_accuracy(self):
        report=evaluate_coverage(example(),expectations())
        self.assertEqual(report['topology']['relations'][0]['status'],'recognised')
        scene=example();scene['edges'][0]['arrow']='start'
        self.assertEqual(evaluate_coverage(scene,expectations())['topology']['relations'][0]['status'],'missing')

    def test_polyline_without_endpoint_metadata_is_unknown(self):
        scene=example();scene['edges']=[{'id':'line','points':[[25,15],[60,15]],'arrow':'end'}]
        report=evaluate_coverage(scene,expectations())
        self.assertEqual(report['topology']['relations'][0]['status'],'unknown')

    def test_unsupported_primitive_and_exact_branch_label(self):
        exp=expectations();exp['native_primitives']=['polygon']
        exp['content'].append({'id':'yes','exact':['是']})
        scene=example();scene['labels']=[{'id':'body','text':'这是正文','x':50,'y':80}]
        report=evaluate_coverage(scene,exp)
        self.assertEqual(report['unsupported'][0]['type'],'polygon')
        self.assertIn('yes',{x['id'] for x in report['missing']})

    def test_branch_label_presence_does_not_prove_edge_assignment(self):
        exp=expectations();exp['relations'][0]['branch_label']='是'
        self.assertEqual(evaluate_coverage(example(),exp)['topology']['relations'][0]['status'],'unknown')

    def test_readonly_rich_segments_and_invalid_references(self):
        scene=example();scene['nodes'][0].update(mode='rich',segments=[{'type':'text','content':'input'}])
        before=copy.deepcopy(scene);evaluate_coverage(scene,expectations());self.assertEqual(before,scene)
        exp=expectations();exp['relations'][0]['target']='undefined'
        with self.assertRaises(ValueError):evaluate_coverage(scene,exp)

    def test_fixture_specific_words_stay_in_evaluation_data(self):
        root=Path(__file__).resolve().parents[1]
        fixture=json.loads((root/'tests/fixtures/spamdataset_coverage.json').read_text(encoding='utf-8'))
        report=evaluate_coverage(example(),fixture)
        self.assertFalse(report['content']['complete'])
        self.assertIn('condition',{x['id'] for x in report['missing']})
        self.assertNotIn('SpamDataset',(root/'scene_coverage.py').read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main()
