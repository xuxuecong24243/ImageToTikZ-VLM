"""Offline integration: synthetic data, no reference leakage or remote services."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from unittest.mock import patch

from PIL import Image

from scene_enricher import run
from semantic_enrichment_evaluation import evaluate_enrichment
from semantic_scene_merger import enrich_semantic_scene, merge_semantics
from semantic_schema import SemanticValidationError
from structured_tikz.validator import validate_scene


def fixture():
    scene = {'canvas': {'width': 300, 'height': 200, 'coordinate_system': 'top-left'},
             'styles': {}, 'nodes': [{'id': 'A', 'type': 'rectangle', 'bbox': [20, 60, 80, 80],
                                     'metadata': {'review_status': 'unverified'}}],
             'edges': [], 'labels': [], 'metadata': {'review_status': 'unverified',
                 'review_candidates': {'lines': [], 'text_regions': [
                     {'id': 'left', 'bbox': [20, 20, 30, 20]},
                     {'id': 'right', 'bbox': [55, 20, 30, 20]}]}}}
    semantic = {'objects': [{'candidate_id': 'A', 'semantic_type': 'matrix', 'label': None,
                            'confidence': 'high', 'review_required': True}],
                'relations': [], 'texts': [
                    {'id': 'heading', 'candidate_id': None, 'text_region_id': 'left',
                     'raw_text': 'Synthetic heading', 'text': 'Synthetic heading', 'mode': 'text',
                     'bbox': None, 'confidence': 'high', 'review_required': True}],
                'unmatched_objects': [], 'unmatched_texts': [], 'warnings': []}
    context = {'image': copy.deepcopy(scene['canvas']),
               'candidates': [{'candidate_id': 'A', 'bbox': [20, 60, 80, 80]}],
               'text_regions': [{'text_region_id': t['id'], 'bbox': list(t['bbox'])}
                                for t in scene['metadata']['review_candidates']['text_regions']]}
    return merge_semantics(scene, semantic), semantic, context


class EnrichmentPipelineTests(OfflineTestCase):
    def test_schema_original_measurements_and_semantics_survive(self):
        scene, semantic, context = fixture()
        before = copy.deepcopy((scene, semantic, context))
        enriched, reports = enrich_semantic_scene(scene, semantic, context)
        self.assertEqual((scene, semantic, context), before)
        self.assertEqual(enriched['metadata']['semantic_review'], scene['metadata']['semantic_review'])
        self.assertEqual(enriched['metadata']['review_candidates'], scene['metadata']['review_candidates'])
        self.assertEqual(enriched['nodes'], scene['nodes'])
        self.assertEqual(enriched['labels'][0]['metadata']['bbox'], [20, 20, 65, 20])
        self.assertEqual(reports['summary']['updated_labels_applied'], 1)
        self.assertEqual(enriched['metadata']['review_status'], 'unverified')
        validate_scene(enriched)

    def test_verified_and_rejected_objects_preserved_and_scenes_refused(self):
        for status in ('verified', 'rejected'):
            scene, semantic, context = fixture()
            for kind in ('nodes', 'labels'):
                scene[kind][0]['metadata']['review_status'] = status
            enriched, _ = enrich_semantic_scene(scene, semantic, context)
            self.assertEqual(enriched['nodes'], scene['nodes'])
            self.assertEqual(enriched['labels'], scene['labels'])
            scene['metadata']['review_status'] = status
            with self.assertRaises(SemanticValidationError):
                enrich_semantic_scene(scene, semantic, context)

    def test_context_measurements_and_ids_cannot_drift(self):
        for change in ('bbox', 'id', 'dimensions'):
            scene, semantic, context = fixture()
            if change == 'bbox':
                context['text_regions'][0]['bbox'][0] += 1
            elif change == 'id':
                context['candidates'][0]['candidate_id'] = 'invented'
            else:
                context['image']['width'] += 1
            with self.assertRaises(SemanticValidationError):
                enrich_semantic_scene(scene, semantic, context)

    def test_reference_only_loaded_after_artifacts_and_inputs_unchanged(self):
        scene, semantic, context = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, data in [('scene', scene), ('semantic', semantic), ('context', context)]:
                (root/(name+'.json')).write_text(json.dumps(data), encoding='utf-8')
            Image.new('RGB', (300, 200), 'white').save(root/'image.png')
            reference_path = root/'reference.json'
            reference_path.write_text(json.dumps(scene), encoding='utf-8')
            before_bytes = {p: p.read_bytes() for p in root.iterdir()}
            args = argparse.Namespace(scene=root/'scene.json', semantic=root/'semantic.json',
                                      context=root/'context.json', image=root/'image.png',
                                      output_root=root/'outputs', reference=reference_path)
            from scene_enricher import load_scene

            def read_checked(path):
                if Path(path) == reference_path:
                    output = next((root/'outputs').iterdir())
                    for name in ('scene_enriched_draft.json', 'text_alignment_report.json',
                                 'shape_enrichment_report.json', 'edge_routing_report.json',
                                 'scene_enriched_review.png', 'enrichment_report.json'):
                        self.assertTrue((output/name).is_file())
                return load_scene(path)

            with patch('scene_enricher.load_scene', side_effect=read_checked), patch('builtins.print'), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('No API allowed')), \
                 patch('subprocess.run', side_effect=AssertionError('No compiler allowed')):
                output = run(args)
            self.assertTrue((output/'reference_evaluation.json').is_file())
            self.assertFalse((output/'scene_verified.json').exists())
            self.assertFalse(list(output.glob('*.tex')))
            for path, raw in before_bytes.items():
                self.assertEqual(path.read_bytes(), raw)

    def test_reference_evaluation_is_read_only_and_missing_parameters_remain_missing(self):
        scene, semantic, context = fixture()
        enriched, reports = enrich_semantic_scene(scene, semantic, context)
        reference = copy.deepcopy(enriched)
        reference['nodes'][0].update(type='matrix', grid={'rows': 2, 'cols': 3})
        before = copy.deepcopy(enriched)
        evaluation = evaluate_enrichment(scene, enriched, semantic, reports, reference)
        self.assertEqual(enriched, before)
        self.assertFalse(evaluation['nodes'][0]['grid_equal'])
        self.assertNotIn('grid', enriched['nodes'][0])


if __name__ == '__main__':
    unittest.main()
