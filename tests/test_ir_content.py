from copy import deepcopy
import json
try:
    from .offline_support import OfflineTestCase
    from .ir_fixtures import example_ir, branch_ir
except ImportError:
    from offline_support import OfflineTestCase
    from ir_fixtures import example_ir, branch_ir
from structured_ir.schema import ir_schema
from structured_ir.prompts import build_ir_prompt, parse_ir_response
from structured_ir.coverage import evaluate_ir_coverage
from structured_ir.converter import convert_ir


class IRContentTests(OfflineTestCase):
    def test_prompt_uses_exact_current_schema_and_original_size(self):
        prompt = build_ir_prompt((640, 420))
        embedded = prompt.split('IR_SCHEMA_JSON:\n', 1)[1].split('\nREGION_CANDIDATE_EVIDENCE_JSON:', 1)[0]
        self.assertEqual(json.loads(embedded), ir_schema())
        self.assertIn('[640, 420]', prompt)
        self.assertIn('raw_text exactly', prompt)
        self.assertIn('kind=merge', prompt)
        self.assertNotIn('SpamDataset', prompt)

    def test_prompt_keeps_candidate_uncertainty_and_does_not_mutate(self):
        plan = {'canvas': {'width': 640, 'height': 420}, 'candidates': [{'bbox': [0, 0, 20, 20]}], 'diagnostics': ['uncertain']}
        before = deepcopy(plan)
        prompt = build_ir_prompt((640, 420), plan)
        self.assertIn('NOT recognized semantic regions', prompt)
        self.assertIn('uncertain', prompt)
        self.assertEqual(plan, before)

    def test_strict_parser_preserves_provenance_text_and_rejects_bad_json(self):
        original = example_ir()
        self.assertEqual(parse_ir_response(json.dumps(original)), original)
        for bad in ('{', '{"version":1,"version":2}', '{"x":NaN}', 'Here is JSON: {}'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_ir_response(bad)

    def test_complete_expectations_are_not_a_visual_success_claim(self):
        ir = example_ir()
        report = evaluate_ir_coverage(ir, {'region_types': ['flow'], 'texts': ['保存结果', 'value = data[0]'],
                                         'element_types': ['text', 'rectangle'], 'connections': [{'source_id': 'start', 'target_id': 'finish', 'direction': 'forward'}]})
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['content']['status'], 'expectations_present')
        self.assertEqual(report['geometry']['status'], 'not_verified')
        self.assertTrue(report['quality_warning'])

    def test_missing_regions_text_nodes_and_connections_reported_separately(self):
        report = evaluate_ir_coverage(example_ir(), {'region_types': ['title'], 'texts': ['missing_text'],
                    'element_types': ['diamond'], 'connections': [{'source_id': 'missing', 'target_id': 'finish'}],
                    'minimum_counts': {'connections': 2}})
        self.assertEqual(len(report['missing']), 5)
        self.assertEqual(report['relationships']['status'], 'missing_expected_items')

    def test_branch_labels_and_direction_checked(self):
        ir = branch_ir()
        report = evaluate_ir_coverage(ir, {'connections': [
            {'source_id': 'decision', 'target_id': 'left', 'direction': 'forward', 'kind': 'branch', 'branch_label': '是'},
            {'source_id': 'left', 'target_id': 'merge', 'kind': 'merge'},
            {'source_id': 'decision', 'target_id': 'right', 'branch_label': '是'},
            {'source_id': 'decision', 'target_id': 'left', 'direction': 'reverse'}]})
        self.assertEqual(len(report['recognized']), 2)
        self.assertEqual(len(report['missing']), 2)

    def test_no_expectations_cannot_claim_omissions_from_candidate_counts(self):
        ir = example_ir()
        ir['connections'] = []
        report = evaluate_ir_coverage(ir, geometry={'text_regions': [{}]*203, 'rectangles': [{}]*8, 'lines': [{}]*74})
        self.assertEqual(report['missing'], [])
        self.assertTrue(report['warnings'])
        self.assertNotIn('coverage_ratio', report)
        self.assertEqual(report['content']['status'], 'not_evaluated_without_expectations')

    def test_unsupported_shapes_and_raw_text_preserved(self):
        ir = branch_ir()
        ir['elements'][0]['type'] = 'circle'  # non-square remains unsupported after diamond extension
        before = deepcopy(ir)
        scene, conversion = convert_ir(ir)
        report = evaluate_ir_coverage(ir, conversion_report=conversion)
        self.assertEqual(report['unsupported'][0]['id'], 'decision')
        self.assertEqual(report['unrendered_text'][0]['raw_text'], '有效？')
        self.assertEqual(ir, before)

    def test_malformed_ir_still_has_format_diagnostic(self):
        for value in (None, [], {'elements': 3}, {'elements': [None]}):
            with self.subTest(value=value):
                report = evaluate_ir_coverage(value)
                self.assertFalse(report['format_validity']['valid'])

    def test_unknown_expectation_keys_do_not_silently_disable_checks(self):
        with self.assertRaises(ValueError):
            evaluate_ir_coverage(example_ir(), {'typo': []})

    def test_invalid_confidence_is_diagnostic_not_an_evaluator_crash(self):
        for value in ('low', None, {}, []):
            ir = example_ir()
            ir['elements'][0]['confidence'] = value
            with self.subTest(value=value):
                self.assertFalse(evaluate_ir_coverage(ir)['format_validity']['valid'])
