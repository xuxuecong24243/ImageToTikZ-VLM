"""Self-contained loader contract fixtures, not historical model/measurement data.

The saved transaction and font-bound records below are explicitly synthetic.
They exercise the unchanged loader's hash binding and mismatch rejection without
publishing private reference images, HTTP responses, or historical local paths.
No optimization, model request, PDF measurement or compilation is performed.
"""
from copy import deepcopy
import hashlib
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

import offline_visual_patch_optimization as entry


class OfflineVisualPatchInputTests(OfflineTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='synthetic-feedback-contract-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        baseline = self.root / 'baseline'
        feedback_run = self.root / 'feedback'
        for folder in (baseline, feedback_run / 'request_01', self.root / 'inputs'):
            folder.mkdir(parents=True, exist_ok=True)
        for name, value in (('ROOT', self.root), ('BASELINE', baseline), ('FEEDBACK_RUN', feedback_run)):
            replacement = patch.object(entry, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)

        def save(path, value):
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

        scene = {'canvas': {'width': 600, 'height': 300, 'coordinate_system': 'top-left'},
                 'styles': {}, 'nodes': [], 'edges': [], 'labels': [],
                 'metadata': {'fixture': 'synthetic_loader_contract_not_a_real_model_run'}}
        bounds = {'pdf': str(baseline / 'scene_after.pdf'),
                  'mapping': {'status': 'verified', 'scale': 1, 'tx': 0, 'ty': 0},
                  'labels': [], 'fixture': 'synthetic_font_bounds_not_pdf_measurements'}
        for index in range(6):
            identifier = 'caption_' + str(index + 1)
            box = [30 + index * 80, 40, 40, 12]
            scene['nodes'].append({'id': identifier, 'type': 'text', 'bbox': box,
                                   'text': 'Label ' + str(index + 1), 'mode': 'text'})
            bounds['labels'].append({'id': identifier, 'collection': 'nodes', 'status': 'measured',
                'bbox_scene': list(box), 'line_boxes_scene': [list(box)], 'line_count': 1,
                'expected_bbox': list(box),
                'original_position': {'x': box[0] + 20, 'y': 46, 'anchor': 'center'}})
        feedback = {'status': 'patch_proposed',
                    'observations': [{'object_id': n['id'], 'issue': 'edge_text_overlap',
                        'confidence': .8, 'reason': 'Synthetic format fixture, not an actual observed collision.'}
                        for n in scene['nodes']],
                    'operations': [{'op': 'move_label', 'id': n['id'], 'dx': 1, 'dy': 0}
                                   for n in scene['nodes']]}
        save(baseline / 'scene_after.json', scene)
        save(feedback_run / 'scene_before.json', scene)
        save(feedback_run / 'text_bounds_before.json', bounds)
        for name in ('visual_feedback_raw.txt', 'request_01/response.txt', 'visual_feedback_parsed.json'):
            save(feedback_run / name, feedback)
        save(feedback_run / 'patch_validation.json',
             {'accepted': False, 'applied': False, 'fixture': 'synthetic_prior_rejection'})
        # PNGs exercise actual source-byte hashes and reference dimensions. They
        # are generated placeholders for this unit test, not conversion results.
        Image.new('RGB', (600, 300), 'white').save(self.root / 'inputs/test.png')
        Image.new('RGB', (600, 300), '#eeeeee').save(baseline / 'scene_after.png')
        sources = (baseline / 'scene_after.json', self.root / 'inputs/test.png', baseline / 'scene_after.png')
        save(feedback_run / 'run_report.json', {
            'fixture': 'synthetic_transaction_flags_for_loader_contract_no_api_was_called',
            'api_called': True, 'mode': 'authorized_dual_image',
            'api': {'requests': [{'diagnostics': {'http_status': 200, 'finish_reason': 'stop'}}]},
            'source_hashes': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}})
        # Fixed expected diagnostics for six non-overlapping plain-text boxes.
        # Do not call diagnose_text_layout to manufacture its own expectation.
        diagnosed = {'labels': [], 'text_overlaps': [], 'geometry_overlaps': [],
            'edge_conflicts': [], 'mapping': dict(bounds['mapping']), 'padding_px': 3.0,
            'interpretation': 'Measured font boxes; intersection is a layout risk, not a verified ink collision.',
            'summary': {'measured': 6, 'unresolved': 0, 'unexpected_wraps': 0,
                'text_overlap_pairs': 0, 'geometry_overlap_pairs': 0, 'edge_text_pairs': 0,
                'edges_with_text_conflicts': 0, 'outside_canvas': 0}}
        for record in bounds['labels']:
            diagnosed['labels'].append(dict(deepcopy(record), conflicts=[], needs_manual_review=False,
                exceeds_expected_bbox=False, outside_canvas=False, unexpected_wrap=False,
                measurement_class='measured_font_envelope'))
        save(baseline / 'text_bounds_report.json', {'after': diagnosed, 'after_pdf': bounds['pdf'],
             'fixture': 'synthetic_combined_contract_report'})

    def args(self):
        return entry.build_parser().parse_args([
            '--bounds', str(entry.BASELINE / 'text_bounds_report.json')])

    def test_combined_report_matches_saved_measurements(self):
        args = self.args()
        _, _, feedback, bounds, hashes, provenance = entry._load_verified_inputs(args)
        self.assertEqual(len(feedback['operations']), 6)
        self.assertEqual(bounds, entry._read_json(entry.FEEDBACK_RUN / 'text_bounds_before.json'))
        self.assertIn(str(args.bounds.resolve()), hashes)
        self.assertEqual(provenance['bounds_selection'],
                         'after_diagnosis_verified_against_saved_measurements')

    def reject_changed_report(self, alter):
        args = self.args()
        reader = entry._read_json
        document = deepcopy(reader(args.bounds))
        alter(document)
        def read(path):
            return document if Path(path).resolve() == args.bounds.resolve() else reader(path)
        with patch.object(entry, '_read_json', side_effect=read):
            with self.assertRaisesRegex(ValueError, 'Combined bounds report does not match'):
                entry._load_verified_inputs(args)

    def test_changed_measured_box_is_rejected(self):
        def alter(document):
            document['after']['labels'][0]['bbox_scene'][0] += 1
        self.reject_changed_report(alter)

    def test_changed_pdf_is_rejected(self):
        self.reject_changed_report(lambda document: document.update(after_pdf='different.pdf'))

    def test_original_saved_report_still_works(self):
        _, _, _, bounds, _, provenance = entry._load_verified_inputs(entry.build_parser().parse_args([]))
        self.assertEqual(bounds, entry._read_json(entry.FEEDBACK_RUN / 'text_bounds_before.json'))
        self.assertEqual(provenance['bounds_selection'], 'saved_measurement_report')


if __name__ == '__main__':
    unittest.main()
