"""MVP adapter tests. Compiler/network stand-ins are not online acceptance.

The full suite separately runs IRHistoricalCompatibilityTests (56 Scene
validations, 49 exact TikZ hashes) and legacy CLI tests; do not duplicate them.
"""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from PIL import Image
try:
    from .offline_support import OfflineTestCase
    from .ir_fixtures import example_ir
except ImportError:
    from offline_support import OfflineTestCase
    from ir_fixtures import example_ir

import convert
from latex_compiler import CompileResult
from vlm_client import APIConfig


class StubClient:
    """A test-only in-memory response source. No HTTP implementation."""
    max_tokens = 1000

    def __init__(self, response=None, error=None, usage=True):
        self.config = APIConfig('https://example.invalid/v1', 'synthetic-test-key', 'fixture-vision')
        self.response = json.dumps(example_ir()) if response is None else response
        self.error = error
        self.calls = 0
        self.last_diagnostics = {'requested_model': self.config.model, 'response_model': self.config.model, 'finish_reason': 'stop'}
        if usage:
            self.last_diagnostics['usage'] = {'prompt_tokens': 80, 'completion_tokens': 43, 'total_tokens': 123}

    def complete_images(self, images, prompt):
        self.calls += 1
        if self.calls > 1:
            raise AssertionError('MVP must not retry')
        self.images = list(images)
        self.prompt = prompt
        if self.error:
            raise self.error
        print('Detailed test-only client log')
        return self.response


class MVPConvertTests(OfflineTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='mvp-test-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.image = self.root/'输入 图片.png'
        Image.new('RGB', (640, 420), 'white').save(self.image)
        self.ir_path = self.root/'已有 IR.json'
        self.write_ir(example_ir())
        self.sequence = 0

    def write_ir(self, value):
        self.ir_path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')

    def argv(self, *extra, replay=True, preview=False, image=None):
        self.sequence += 1
        args = ['--input', str(image or self.image), '--output', str(self.root/('输出 结果 %d' % self.sequence))]
        if replay:
            args += ['--offline-ir', str(self.ir_path)]
        if not preview:
            args += ['--no-compile']
        return args + list(extra)

    def run_args(self, *extra, replay=True, preview=False, image=None, client=None):
        args = convert.build_parser().parse_args(self.argv(*extra, replay=replay, preview=preview, image=image))
        return convert.run_conversion(args, client=client)

    def successful_compile(self, tex, **kwargs):
        tex = Path(tex)
        pdf = tex.with_suffix('.pdf')
        pdf.write_bytes(b'%PDF-1.4\nTest-only compiler stand-in; not an acceptance PDF.\n')
        log = tex.with_suffix('.log')
        log.write_text('test compiler successful', encoding='utf-8')
        console = tex.parent/'test.console.log'
        console.write_text('test console', encoding='utf-8')
        return CompileResult(True, 0, False, pdf, log, console, '', ['test-compiler'])

    def failed_compile(self, tex, **kwargs):
        tex = Path(tex)
        log = tex.with_suffix('.log')
        log.write_text('test: XeLaTeX unavailable', encoding='utf-8')
        return CompileResult(False, 1, False, None, log, log, 'XeLaTeX executable not found', ['test-compiler'])

    def successful_render(self, pdf, png, **kwargs):
        self.assertTrue(Path(pdf).is_file())
        Image.new('RGB', (10, 10), 'white').save(png)
        return Path(png)

    def test_cli_defaults_help_and_bounded_request_argument(self):
        args = convert.build_parser().parse_args(['--input', '图 片.jpg', '--output', '新 目录'])
        self.assertEqual(args.input, Path('图 片.jpg'))
        self.assertFalse(args.allow_api)
        self.assertIsNone(args.offline_ir)
        self.assertEqual(args.max_api_requests, 1)
        self.assertFalse(args.no_compile)
        self.assertIn('--allow-api', convert.build_parser().format_help())
        for value in ('2', '-1', 'invalid'):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                convert.build_parser().parse_args(['--input', 'a.png', '--output', 'new', '--max-api-requests', value])

    def test_png_and_jpeg_unicode_space_paths_replay_without_configuration(self):
        for extension in ('png', 'jpg'):
            source = self.root/('中文 空格.'+extension)
            Image.new('RGB', (640, 420), 'white').save(source)
            with self.subTest(extension=extension), patch('vlm_client.APIConfig.from_env', side_effect=AssertionError('No config read')):
                report = self.run_args(image=source)
            self.assertEqual(report['status'], 'partial')
            self.assertEqual(report['input']['width'], 640)
            self.assertEqual(report['latex']['status'], 'generated')
            self.assertEqual(report['api_request_count'], 0)
            self.assertEqual(report['pdf']['status'], 'skipped')

    def test_missing_or_undecodable_image_saves_stage_error(self):
        broken = self.root/'broken.jpg'
        broken.write_bytes(b'not an image')
        for source in (self.root/'missing.png', broken):
            with self.subTest(source=source):
                report = self.run_args(image=source)
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['stage'], 'input')
                self.assertTrue(report['errors'])
                self.assertEqual(json.loads(Path(report['paths']['report']).read_text(encoding='utf-8')), report)
                self.assertNotIn('tex', report['paths'])

    def test_authorization_gate_precedes_config_even_with_injected_client(self):
        client = StubClient()
        with patch('vlm_client.APIConfig.from_env', side_effect=AssertionError('No config read')):
            report = self.run_args(replay=False, client=client)
        self.assertEqual(report['stage'], 'authorization')
        self.assertEqual(report['status'], 'failed')
        self.assertIn('--allow-api', report['errors'][0]['message'])
        self.assertEqual(client.calls, 0)

    def test_zero_budget_does_not_call_or_read_configuration(self):
        client = StubClient()
        with patch('vlm_client.APIConfig.from_env', side_effect=AssertionError('No config read')):
            report = self.run_args('--allow-api', '--max-api-requests', '0', replay=False, client=client)
        self.assertEqual(report['api_request_count'], 0)
        self.assertEqual(client.calls, 0)
        self.assertIn('budget is zero', report['errors'][0]['message'])

    def test_offline_and_online_flags_cannot_mix(self):
        report = self.run_args('--allow-api')
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['stage'], 'authorization')
        self.assertEqual(report['api_request_count'], 0)

    def test_missing_config_reports_names_without_request(self):
        # OfflineTestCase redirects the real loader to an absent temporary .env.
        report = self.run_args('--allow-api', replay=False)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['api_request_count'], 0)
        self.assertIn('API_BASE_URL', report['errors'][0]['message'])
        self.assertIn('API_KEY', report['errors'][0]['message'])
        self.assertIn('MODEL_NAME', report['errors'][0]['message'])

    def test_mock_recognition_single_request_usage_and_raw_response_saved(self):
        client = StubClient()
        # Verify the actual VLM diagnostics field, without a config fallback.
        del client.config
        report = self.run_args('--allow-api', replay=False, client=client)
        self.assertEqual(client.calls, 1)
        self.assertEqual(report['api_request_count'], 1)
        self.assertEqual(report['model'], 'fixture-vision')
        self.assertEqual(report['token_usage']['total_tokens'], 123)
        self.assertTrue(report['token_usage']['known'])
        self.assertEqual(Path(report['paths']['ir_response.txt']).read_text(encoding='utf-8'), client.response)
        self.assertIn('IR_SCHEMA_JSON', client.prompt)
        self.assertEqual(report['recognition'], 'model_ir_response_unverified')

    def test_failed_request_counted_once_and_key_redacted(self):
        client = StubClient(error=RuntimeError('provider failed synthetic-test-key'), usage=False)
        report = self.run_args('--allow-api', replay=False, client=client)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(client.calls, 1)
        self.assertEqual(report['api_request_count'], 1)
        self.assertFalse(report['token_usage']['known'])
        self.assertIsNone(report['token_usage']['total_tokens'])
        for path in Path(report['paths']['report']).parent.rglob('*.json'):
            self.assertNotIn('synthetic-test-key', path.read_text(encoding='utf-8'))
        self.assertNotIn('tex', report['paths'])

    def test_parse_and_ir_reference_failures_remain_distinct(self):
        self.ir_path.write_text('{', encoding='utf-8')
        report = self.run_args()
        self.assertEqual(report['stage'], 'ir_parse')
        self.assertTrue(Path(report['paths']['parse_error.json']).is_file())
        ir = example_ir()
        ir['connections'][0]['target_id'] = 'absent'
        self.write_ir(ir)
        report = self.run_args()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['stage'], 'ir_validation')
        self.assertFalse(report['ir_validation']['valid'])
        self.assertTrue(any(e['path'].endswith('target_id') for e in report['ir_validation']['errors']))
        self.assertEqual(report['recovery']['status'], 'unresolved')
        self.assertNotIn('tex', report['paths'])

    def test_partial_scene_continues_to_editable_tex_with_loss_diagnostics(self):
        ir = example_ir()
        ir['elements'][1]['type'] = 'circle'  # Non-square, intentionally unsupported.
        self.write_ir(ir)
        original = self.ir_path.read_bytes()
        with patch('convert.compile_bounded', side_effect=self.successful_compile), \
             patch('convert.render_pdf', side_effect=self.successful_render):
            report = self.run_args(preview=True)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['conversion_status'], 'partial')
        self.assertEqual(report['png']['status'], 'rendered')
        self.assertTrue(report['scene_validation']['valid'])
        self.assertEqual(report['latex']['status'], 'generated')
        self.assertTrue(report['conversion']['unsupported_elements'])
        self.assertTrue(report['warnings'])
        self.assertEqual(self.ir_path.read_bytes(), original)
        self.assertEqual(json.loads(Path(report['paths']['ir_raw.json']).read_text(encoding='utf-8')), ir)

    def test_scene_validation_failure_stops_before_latex(self):
        with patch('convert.validate_scene', side_effect=ValueError('invalid scene test')), \
             patch('convert.compile_bounded', side_effect=AssertionError('No compilation')):
            report = self.run_args()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['stage'], 'scene_validation')
        self.assertFalse(report['scene_validation']['valid'])
        self.assertNotIn('tex', report['paths'])

    def test_converter_exception_does_not_fall_back_to_fake_scene(self):
        with patch('two_stage_pipeline.convert_ir', side_effect=ValueError('conversion stopped')):
            report = self.run_args()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['stage'], 'ir_to_scene')
        self.assertNotIn('tex', report['paths'])
        self.assertEqual(report['compile_attempts'], 0)

    def test_empty_ir_does_not_generate_empty_success(self):
        ir = example_ir()
        ir.update(regions=[], elements=[], connections=[])
        self.write_ir(ir)
        report = self.run_args()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['errors'][0]['code'], 'no_drawable_content')
        self.assertNotIn('tex', report['paths'])

    def test_compile_failure_preserves_tex_and_log_without_retry(self):
        with patch('convert.compile_bounded', side_effect=self.failed_compile) as compiler, \
             patch('convert.render_pdf', side_effect=AssertionError('No render without PDF')):
            report = self.run_args(preview=True)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['latex']['status'], 'generated')
        self.assertEqual(report['pdf']['status'], 'failed')
        self.assertEqual(report['png']['status'], 'skipped')
        self.assertEqual(compiler.call_count, 1)
        self.assertEqual(report['compile_attempts'], 1)
        self.assertTrue(Path(report['pdf']['log']).is_file())
        self.assertTrue(Path(report['paths']['tex']).is_file())

    def test_successful_pdf_png_and_complete_report_fields(self):
        with patch('convert.compile_bounded', side_effect=self.successful_compile), \
             patch('convert.render_pdf', side_effect=self.successful_render):
            report = self.run_args(preview=True)
        self.assertEqual(report['status'], 'success')
        self.assertEqual(report['conversion_status'], 'complete')
        self.assertEqual(report['semantic_status'], 'unverified')
        self.assertEqual(report['pdf']['status'], 'compiled')
        self.assertEqual(report['png']['status'], 'rendered')
        required = {'input', 'model', 'api_request_count', 'token_usage', 'ir_validation', 'scene_validation',
                    'conversion_status', 'semantic_status', 'latex', 'pdf', 'png', 'paths', 'errors', 'warnings'}
        self.assertTrue(required <= report.keys())
        for key in ('tex', 'pdf', 'png', 'report'):
            self.assertTrue(Path(report['paths'][key]).is_file())
        self.assertEqual(json.loads(Path(report['paths']['report']).read_text(encoding='utf-8')), report)

    def test_png_failure_is_partial_and_preserves_pdf_tex(self):
        with patch('convert.compile_bounded', side_effect=self.successful_compile), \
             patch('convert.render_pdf', side_effect=RuntimeError('renderer unavailable')):
            report = self.run_args(preview=True)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['stage'], 'render')
        self.assertEqual(report['pdf']['status'], 'compiled')
        self.assertEqual(report['png']['status'], 'failed')
        self.assertTrue(Path(report['paths']['pdf']).is_file())
        self.assertTrue(Path(report['paths']['tex']).is_file())

    def test_compiler_false_success_cannot_claim_pdf_or_render(self):
        def no_pdf(tex, **kwargs):
            path = Path(tex)
            return CompileResult(True, 0, False, path.with_suffix('.pdf'), path.with_suffix('.log'), path.with_suffix('.log'), '', [])
        with patch('convert.compile_bounded', side_effect=no_pdf), \
             patch('convert.render_pdf', side_effect=AssertionError('No PDF')):
            report = self.run_args(preview=True)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['pdf']['status'], 'failed')
        self.assertNotIn('pdf', report['paths'])

    def test_offline_source_immutable_and_output_is_native_editable_tikz(self):
        original = self.ir_path.read_bytes()
        report = self.run_args()
        self.assertEqual(report['mode'], 'offline_replay')
        self.assertEqual(report['recognition'], 'not_performed_offline_supplied_ir')
        self.assertEqual(self.ir_path.read_bytes(), original)
        document = Path(report['paths']['tex']).read_text(encoding='utf-8')
        self.assertIn('\\node', document)
        self.assertIn('\\draw', document)
        self.assertNotIn('\\includegraphics', document)
        self.assertEqual(report['recovery']['changes'], [])

    def test_existing_output_is_never_overwritten(self):
        args = convert.build_parser().parse_args(self.argv())
        first = convert.run_conversion(args)
        before = {p.relative_to(args.output).as_posix(): p.read_bytes() for p in args.output.rglob('*') if p.is_file()}
        with self.assertRaisesRegex(ValueError, 'not empty'):
            convert.run_conversion(args)
        after = {p.relative_to(args.output).as_posix(): p.read_bytes() for p in args.output.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(first['status'], 'partial')

    def test_main_exit_codes_and_concise_terminal_capture(self):
        output = io.StringIO()
        with redirect_stdout(output):
            failed = convert.main(self.argv(replay=False))
            partial = convert.main(self.argv())
            with patch('convert.compile_bounded', side_effect=self.successful_compile), \
                 patch('convert.render_pdf', side_effect=self.successful_render):
                success = convert.main(self.argv(preview=True))
        self.assertEqual((failed, partial, success), (1, 2, 0))
        self.assertIn('[OK] LaTeX generated', output.getvalue())
        self.assertIn('report.json', output.getvalue())
        self.assertLess(len(output.getvalue().splitlines()), 50)

    def test_mock_client_stdout_is_retained_in_log_not_terminal(self):
        output = io.StringIO()
        with redirect_stdout(output):
            report = self.run_args('--allow-api', replay=False, client=StubClient())
        self.assertEqual(output.getvalue(), '')
        self.assertIn('Detailed test-only client log', Path(report['paths']['log']).read_text(encoding='utf-8'))
