import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from unittest.mock import patch
from PIL import Image
try:
    from .offline_support import OfflineTestCase
    from .ir_fixtures import example_ir, branch_ir
except ImportError:
    from offline_support import OfflineTestCase
    from ir_fixtures import example_ir, branch_ir
from auto_pipeline import build_parser, run_pipeline, main


class IRPipelineTests(OfflineTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root/'input.png'
        Image.new('RGB', (640, 420), 'white').save(self.image)
        self.ir_path = self.root/'supplied.json'
        self.ir_path.write_text(json.dumps(example_ir(), ensure_ascii=False), encoding='utf-8')

    def args(self, *extra):
        return build_parser().parse_args(['--image', str(self.image), '--geometry-mode', 'off',
                                         '--output-root', str(self.root/'outputs'), *extra])

    def test_default_mode_and_legacy_authorization_unchanged(self):
        args = self.args()
        self.assertEqual(args.pipeline_mode, 'single-stage')
        with self.assertRaisesRegex(ValueError, 'allow-api'):
            run_pipeline(args)
        with self.assertRaisesRegex(ValueError, 'IR-specific'):
            run_pipeline(self.args('--offline-ir', str(self.ir_path)))

    def test_offline_supplied_ir_never_loads_configuration_or_requests(self):
        with patch('vlm_client.APIConfig.from_env', side_effect=AssertionError('No config read')), \
             patch('vlm_client.VLMClient.complete_images', side_effect=AssertionError('No API')):
            report = run_pipeline(self.args('--pipeline-mode', 'two-stage', '--offline-ir', str(self.ir_path), '--no-compile'))
        self.assertEqual(report['status'], 'tex_generated_unverified')
        self.assertEqual(report['api_call_count'], 0)
        self.assertEqual(report['recognition'], 'not_performed_offline_supplied_ir')
        output = Path(report['output_dir'])
        self.assertEqual((output/'supplied_ir.json').read_bytes(), self.ir_path.read_bytes())
        self.assertEqual(json.loads((output/'ir_raw.json').read_text(encoding='utf-8')), example_ir())
        self.assertTrue((output/'generated.tex').exists())
        self.assertNotIn('includegraphics', (output/'generated.tex').read_text(encoding='utf-8'))

    def test_missing_ir_stops_without_fabricated_output(self):
        with patch('vlm_client.APIConfig.from_env', side_effect=AssertionError('No config read')):
            report = run_pipeline(self.args('--pipeline-mode', 'two-stage'))
        self.assertEqual(report['status'], 'awaiting_ir')
        self.assertFalse((Path(report['output_dir'])/'ir_raw.json').exists())
        self.assertFalse((Path(report['output_dir'])/'scene.json').exists())

    def test_offline_and_paid_modes_cannot_mix(self):
        with self.assertRaisesRegex(ValueError, 'Offline IR'):
            run_pipeline(self.args('--pipeline-mode', 'two-stage', '--offline-ir', str(self.ir_path), '--allow-api'))

    def test_partial_conversion_retains_original_and_does_not_compile(self):
        ir = branch_ir()
        ir['elements'][0]['type'] = 'circle'  # non-square circle remains unsupported
        self.ir_path.write_text(json.dumps(ir), encoding='utf-8')
        with patch('two_stage_pipeline.compile_bounded', side_effect=AssertionError('Partial must not compile')):
            report = run_pipeline(self.args('--pipeline-mode', 'two-stage', '--offline-ir', str(self.ir_path)))
        self.assertEqual(report['status'], 'conversion_partial')
        output = Path(report['output_dir'])
        self.assertEqual(json.loads((output/'ir_raw.json').read_text(encoding='utf-8')), ir)
        self.assertTrue((output/'scene.json').exists())
        self.assertFalse((output/'generated.tex').exists())

    def test_invalid_json_and_schema_have_separate_stages(self):
        self.ir_path.write_text('{', encoding='utf-8')
        report = run_pipeline(self.args('--pipeline-mode', 'two-stage', '--offline-ir', str(self.ir_path)))
        self.assertEqual(report['stage'], 'ir_parse')
        self.assertTrue((Path(report['output_dir'])/'parse_error.json').exists())
        self.ir_path.write_text('{}', encoding='utf-8')
        report = run_pipeline(self.args('--pipeline-mode', 'two-stage', '--offline-ir', str(self.ir_path)))
        self.assertEqual(report['status'], 'ir_invalid')
        self.assertEqual(report['compile_attempt_count'], 0)

    def test_dimensions_must_match_supplied_image(self):
        ir = example_ir()
        ir['canvas']['width'] = 700
        self.ir_path.write_text(json.dumps(ir), encoding='utf-8')
        report = run_pipeline(self.args('--pipeline-mode', 'two-stage', '--offline-ir', str(self.ir_path)))
        self.assertEqual(report['status'], 'ir_invalid')
        validation = json.loads((Path(report['output_dir'])/'ir_validation_report.json').read_text())
        self.assertEqual(validation['errors'][-1]['code'], 'source_dimensions_mismatch')

    def test_mock_stage_one_one_call_then_local_stage_two(self):
        class StubClient:
            max_tokens = 1000
            last_diagnostics = {'usage': {'total_tokens': 123}, 'finish_reason': 'stop'}
            def __init__(self):
                self.calls = 0
            def complete_images(self, images, prompt):
                self.calls += 1
                if self.calls > 1:
                    raise AssertionError('Only one VLM stage allowed')
                self.prompt = prompt
                return json.dumps(example_ir())
        client = StubClient()
        report = run_pipeline(self.args('--pipeline-mode', 'two-stage', '--allow-api', '--no-compile'), client=client)
        self.assertEqual(report['status'], 'tex_generated_unverified')
        self.assertEqual(client.calls, 1)
        self.assertEqual(report['api_call_count'], 1)  # injected invocation, never real HTTP
        self.assertIn('IR', client.prompt)
        self.assertEqual(report['automatic_repair_count'], 0)

    def test_dry_run_generates_schema_prompt_only(self):
        report = run_pipeline(self.args('--pipeline-mode', 'two-stage', '--dry-run'))
        self.assertEqual(report['status'], 'prepared_offline')
        self.assertTrue((Path(report['output_dir'])/'ir_schema.json').exists())
        self.assertFalse((Path(report['output_dir'])/'ir_raw.json').exists())

    def test_empty_legal_ir_is_not_compiled_as_success(self):
        ir = example_ir()
        ir.update(regions=[], elements=[], connections=[])
        self.ir_path.write_text(json.dumps(ir), encoding='utf-8')
        with patch('two_stage_pipeline.compile_bounded', side_effect=AssertionError('No empty success')):
            report = run_pipeline(self.args('--pipeline-mode', 'two-stage', '--offline-ir', str(self.ir_path)))
        self.assertEqual(report['status'], 'conversion_partial')
        self.assertFalse((Path(report['output_dir'])/'generated.tex').exists())

    def test_cli_exit_codes_distinguish_prepared_complete_and_missing_ir(self):
        common = ['--image', str(self.image), '--pipeline-mode', 'two-stage',
                  '--geometry-mode', 'off', '--output-root', str(self.root/'cli')]
        with redirect_stdout(io.StringIO()), \
             patch('vlm_client.APIConfig.from_env', side_effect=AssertionError('No config read')):
            self.assertEqual(main(common + ['--dry-run']), 0)
            self.assertEqual(main(common), 1)
            self.assertEqual(main(common + ['--offline-ir', str(self.ir_path), '--no-compile']), 0)

    def test_cli_partial_and_invalid_ir_have_failure_exit_code(self):
        common = ['--image', str(self.image), '--pipeline-mode', 'two-stage',
                  '--geometry-mode', 'off', '--output-root', str(self.root/'cli'),
                  '--offline-ir', str(self.ir_path), '--no-compile']
        with redirect_stdout(io.StringIO()):
            ir = branch_ir()
            ir['elements'][0]['type'] = 'circle'
            self.ir_path.write_text(json.dumps(ir), encoding='utf-8')
            self.assertEqual(main(common), 1)
            self.ir_path.write_text('{}', encoding='utf-8')
            self.assertEqual(main(common), 1)

    def test_repeat_offline_runs_preserve_previous_artifacts(self):
        args = self.args('--pipeline-mode', 'two-stage', '--offline-ir', str(self.ir_path), '--no-compile')
        first = run_pipeline(args)
        first_dir = Path(first['output_dir'])
        before = {p.name: p.read_bytes() for p in first_dir.iterdir() if p.is_file()}
        second = run_pipeline(args)
        self.assertNotEqual(first['output_dir'], second['output_dir'])
        self.assertEqual(before, {p.name: p.read_bytes() for p in first_dir.iterdir() if p.is_file()})

    def test_invalid_utf8_retains_original_bytes_without_model_fallback(self):
        raw = b'{"raw_text": "\xff"}'
        self.ir_path.write_bytes(raw)
        with patch('vlm_client.APIConfig.from_env', side_effect=AssertionError('No config read')):
            report = run_pipeline(self.args('--pipeline-mode', 'two-stage', '--offline-ir', str(self.ir_path)))
        output = Path(report['output_dir'])
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['api_call_count'], 0)
        self.assertEqual((output/'supplied_ir.json').read_bytes(), raw)
        self.assertFalse((output/'scene.json').exists())
