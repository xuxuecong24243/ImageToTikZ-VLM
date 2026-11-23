"""Offline configuration contracts; fixture credentials never reach a provider."""
import os
from pathlib import Path
import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from unittest.mock import patch

from PIL import Image
import vlm_client
from vlm_client import (APIConfig, ConfigError, OutputConfig, ThinkingConfig,
                        VLMClient, ApiNotAuthorized, read_config_environment)


class ProjectConfigurationTests(OfflineTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.envfile = self.root / '.env'
        self.addCleanup(patch.stopall)
        patch('vlm_client.PROJECT_ENV_FILE', self.envfile).start()
        patch.dict(os.environ, {}, clear=True).start()
        self.base = 'API_BASE_URL=https://fixture.test/v1\nAPI_KEY=fixture-secret\nMODEL_NAME=fixture-model\n'

    def write(self, text):
        self.envfile.write_text(text, encoding='utf-8')

    def test_absent_settings_stop_without_provider_or_model_defaults(self):
        with patch('urllib.request.build_opener') as opener:
            with self.assertRaisesRegex(ConfigError, 'API_BASE_URL, API_KEY, MODEL_NAME'):
                APIConfig.from_env()
            opener.assert_not_called()

    def test_project_file_loads_from_an_unrelated_working_directory(self):
        self.write(self.base)
        elsewhere = self.root / 'elsewhere'
        elsewhere.mkdir()
        (elsewhere / '.env').write_text(self.base.replace('fixture-model', 'wrong-cwd-model'), encoding='utf-8')
        previous = Path.cwd()
        try:
            os.chdir(elsewhere)
            self.assertEqual(APIConfig.from_env().model, 'fixture-model')
        finally:
            os.chdir(previous)

    def test_unrelated_environment_credentials_are_not_used(self):
        os.environ['OTHER_SERVICE_API_KEY'] = 'unrelated-fixture-secret'
        self.assertNotIn('OTHER_SERVICE_API_KEY', read_config_environment())
        with self.assertRaisesRegex(ConfigError, 'API_KEY'):
            APIConfig.from_env()

    def test_process_values_override_file_without_global_environment_mutation(self):
        self.write(self.base)
        os.environ.update(API_KEY='process-fixture-secret', MODEL_NAME='process-model')
        before = dict(os.environ)
        config = APIConfig.from_env()
        self.assertEqual(config.api_key, 'process-fixture-secret')
        self.assertEqual(config.model, 'process-model')
        self.assertEqual(config.base_url, 'https://fixture.test/v1')
        self.assertEqual(dict(os.environ), before)
        self.assertNotIn('process-fixture-secret', repr(config))

    def test_explicit_environment_mapping_never_reads_project_file(self):
        self.write(self.base)
        with patch('vlm_client._project_config_values', side_effect=AssertionError('File read forbidden')):
            with self.assertRaisesRegex(ConfigError, 'API_KEY'):
                APIConfig.from_env({'API_BASE_URL': 'https://explicit.test', 'MODEL_NAME': 'explicit-model'})
            self.assertEqual(OutputConfig.from_env(environ={}).max_tokens, 12000)
            self.assertIsNone(ThinkingConfig.from_env(environ={}).thinking_mode)

    def test_legacy_base_name_is_normalized_in_explicit_mapping_and_file(self):
        self.write(self.base.replace('API_BASE_URL=', 'API_BASE='))
        self.assertEqual(APIConfig.from_env().base_url, 'https://fixture.test/v1')
        config = APIConfig.from_env({'API_BASE': 'https://legacy.test', 'API_KEY': 'fixture', 'MODEL_NAME': 'model'})
        self.assertEqual(config.base_url, 'https://legacy.test')

    def test_process_alias_beats_file_canonical_but_process_canonical_wins(self):
        self.write(self.base)
        os.environ['API_BASE'] = 'https://legacy-process.test'
        self.assertEqual(APIConfig.from_env().base_url, 'https://legacy-process.test')
        os.environ['API_BASE_URL'] = 'https://canonical-process.test'
        self.assertEqual(APIConfig.from_env().base_url, 'https://canonical-process.test')

    def test_explicit_empty_process_setting_does_not_fall_back_to_file_or_alias(self):
        self.write(self.base)
        os.environ.update(API_BASE_URL='', API_BASE='https://legacy-process.test')
        with self.assertRaisesRegex(ConfigError, 'API_BASE_URL'):
            APIConfig.from_env()

    def test_file_canonical_wins_over_file_alias(self):
        self.write(self.base + 'API_BASE=https://legacy-file.test\n')
        self.assertEqual(APIConfig.from_env().base_url, 'https://fixture.test/v1')

    def test_literal_quoted_values_comments_bom_and_export(self):
        self.envfile.write_text('\ufeff# comment\nexport API_BASE_URL="https://fixture.test/v1" # comment\n'
                                "API_KEY='${DO_NOT_EXPAND}$(do-not-execute)'\nMODEL_NAME=fixture-model # comment\n", encoding='utf-8')
        os.environ['DO_NOT_EXPAND'] = 'must-not-be-used'
        config = APIConfig.from_env()
        self.assertEqual(config.api_key, '${DO_NOT_EXPAND}$(do-not-execute)')
        self.assertEqual(config.model, 'fixture-model')

    def test_required_and_optional_configuration_share_source_and_cli_precedence(self):
        self.write(self.base + 'API_MAX_OUTPUT_TOKENS=2048\nAPI_THINKING_MODE=enabled\nAPI_REASONING_EFFORT=low\n')
        self.assertEqual(OutputConfig.from_env().max_tokens, 2048)
        self.assertEqual(ThinkingConfig.from_env().request_fields(), {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'low'})
        os.environ['API_MAX_OUTPUT_TOKENS'] = '4096'
        self.assertEqual(OutputConfig.from_env().max_tokens, 4096)
        self.assertEqual(OutputConfig.from_env(max_tokens=8192).max_tokens, 8192)
        self.assertEqual(ThinkingConfig.from_env(thinking_mode='disabled').thinking_mode, 'disabled')

    def test_config_loading_cannot_grant_api_authorization(self):
        self.write(self.base + 'ALLOW_API=true\n')
        self.assertNotIn('ALLOW_API', read_config_environment())
        image = self.root / 'fixture.png'
        Image.new('RGB', (2, 2), 'white').save(image)
        client = VLMClient(APIConfig.from_env())
        with patch('urllib.request.build_opener') as opener:
            with self.assertRaises(ApiNotAuthorized):
                client.generate(image)
            opener.assert_not_called()

    def test_example_file_is_never_a_credentials_source(self):
        (self.root / '.env.example').write_text(self.base, encoding='utf-8')
        with self.assertRaisesRegex(ConfigError, 'API_KEY'):
            APIConfig.from_env()

    def test_invalid_values_and_syntax_do_not_leak_key(self):
        for text in ['API_KEY="fixture-secret\n', 'API_KEY=fixture-secret\nAPI_KEY=duplicate-secret\n',
                     'fixture-secret-invalid-assignment\n', self.base.replace('https://fixture.test/v1', 'https://user:fixture-secret@fixture.test')]:
            with self.subTest(kind=text.split('=', 1)[0][:7]):
                self.write(text)
                with self.assertRaises(ConfigError) as caught:
                    APIConfig.from_env()
                self.assertNotIn('fixture-secret', str(caught.exception))
                self.assertNotIn('duplicate-secret', str(caught.exception))

    def test_invalid_encoding_and_oversized_files_have_value_free_errors(self):
        for raw in [b'API_KEY=fixture-secret\xff', b'API_KEY=fixture-secret' + b'x' * 65536]:
            self.envfile.write_bytes(raw)
            with self.assertRaises(ConfigError) as caught:
                APIConfig.from_env()
            self.assertNotIn('fixture-secret', str(caught.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
