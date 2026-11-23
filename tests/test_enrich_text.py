"""Small offline synthetic checks; no images, models or API clients are used."""
from copy import deepcopy
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

from semantic_text_aligner import align_texts, merge_text_regions


class TextAlignmentTests(OfflineTestCase):
    def test_adjacent_fragments_merge_but_separated_row_does_not(self):
        result = merge_text_regions([
            {'text_region_id': 'a', 'bbox': [4, 3, 18, 10]},
            {'text_region_id': 'b', 'bbox': [25, 3, 20, 10]},
            {'text_region_id': 'c', 'bbox': [90, 3, 20, 10]},
        ])
        self.assertEqual(result[0], {'bbox': [4, 3, 41, 10], 'region_ids': ['a', 'b']})
        self.assertEqual(len(result), 2)

    def test_different_semantic_seeds_do_not_join(self):
        regions = [{'text_region_id': 'a', 'bbox': [0, 0, 20, 10]},
                   {'text_region_id': 'b', 'bbox': [24, 0, 20, 10]}]
        self.assertEqual(len(merge_text_regions(regions, {'a': 'one', 'b': 'two'})), 2)

    def test_split_title_seed_expands_both_directions_without_mutation(self):
        scene = {'labels': [{'id': 'sem_title', 'text': 'Synthetic heading', 'mode': 'text',
                             'x': 30, 'y': 9, 'metadata': {'raw_text': 'original', 'custom': True}}]}
        semantic = {'texts': [{'id': 'title', 'text': 'Synthetic heading', 'text_region_id': 'middle',
                               'bbox': [20, 4, 42, 10], 'mode': 'text'}]}
        context = {'text_regions': [{'text_region_id': 'left', 'bbox': [2, 4, 15, 10]},
                                    {'text_region_id': 'middle', 'bbox': [20, 4, 20, 10]},
                                    {'text_region_id': 'right', 'bbox': [42, 4, 20, 10]}]}
        before = deepcopy((scene, semantic, context))
        labels, report = align_texts(scene, semantic, context)
        self.assertEqual(report['entries'][0]['bbox'], [2, 4, 60, 10])
        self.assertEqual(report['entries'][0]['position_source'], 'geometry_inference')
        self.assertTrue(labels[0]['metadata']['custom'])
        self.assertEqual(labels[0]['metadata']['raw_text'], 'original')
        self.assertEqual((scene, semantic, context), before)

    def test_unique_formula_association_and_ambiguous_boxes(self):
        semantic = {'objects': [{'candidate_id': 'matrix', 'label': 'Matrix Q_z'}],
                    'texts': [{'id': 'formula', 'mode': 'math', 'text': r'Q_{z} \in R^5'}]}
        context = {'candidates': [{'candidate_id': 'matrix', 'bbox': [100, 100, 60, 40]}],
                   'text_regions': [{'text_region_id': 'caption', 'bbox': [105, 80, 50, 12]}]}
        labels, report = align_texts({}, semantic, context)
        self.assertEqual(len(labels), 1)
        self.assertEqual(report['entries'][0]['position_source'], 'geometry_inference')
        self.assertEqual(report['counts']['newly_positioned'], 1)
        context['text_regions'].append({'text_region_id': 'other', 'bbox': [105, 60, 50, 12]})
        labels, report = align_texts({}, semantic, context)
        self.assertEqual(labels, [])
        self.assertEqual(report['entries'][0]['status'], 'ambiguous')

    def test_ambiguous_symbol_object_matches_are_not_forced(self):
        semantic = {'objects': [{'candidate_id': 'a', 'label': 'Source H_sub'},
                                {'candidate_id': 'b', 'label': 'Output H_sub'}],
                    'texts': [{'id': 'f', 'mode': 'math', 'text': 'H_{sub} = 1'}]}
        labels, report = align_texts({}, semantic, {})
        self.assertEqual(labels, [])
        self.assertEqual(report['entries'][0]['status'], 'ambiguous')

    def test_protected_labels_and_nodes_preserved(self):
        label = {'id': 'sem_a', 'text': 'Approved', 'x': 5, 'y': 7,
                 'metadata': {'review_status': 'verified', 'bbox': [0, 0, 10, 14]}}
        scene = {'labels': [label], 'nodes': [{'id': 'n', 'metadata': {'review_status': 'rejected'}}]}
        semantic = {'texts': [{'id': 'a', 'bbox': [20, 20, 20, 20]},
                              {'id': 'b', 'candidate_id': 'n', 'bbox': [30, 30, 10, 10]}]}
        labels, report = align_texts(scene, semantic, {})
        self.assertEqual(labels, [label])
        self.assertEqual([entry['status'] for entry in report['entries']], ['protected', 'protected'])

    def test_unanchored_text_does_not_receive_invented_position(self):
        labels, report = align_texts({}, {'texts': [{'id': 'new', 'text': 'Unlocated words'}]}, {})
        self.assertFalse(labels)
        self.assertEqual(report['counts']['remaining_unpositioned'], 1)
        self.assertEqual(report['entries'][0]['position_source'], 'unresolved')


if __name__ == '__main__':
    unittest.main()
