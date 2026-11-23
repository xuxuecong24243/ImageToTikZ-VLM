"""Two-image requests use the existing client; all transport is mocked offline."""
import base64
import json
import os
from pathlib import Path
import tempfile
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from unittest.mock import Mock, patch

from PIL import Image

from prompts import GENERATION_PROMPT
from vlm_client import APIConfig, APIResponseError, ApiNotAuthorized, VLMClient


class SemanticClientTests(OfflineTestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'outputs')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / 'original.png'
        self.annotated = self.root / 'candidate_ids.png'
        Image.new('RGB', (30, 15), 'white').save(self.original)
        Image.new('RGB', (30, 15), 'orange').save(self.annotated)
        self.config = APIConfig('https://api.deepseek.com', 'private-test-key', 'deepseek-flash')

    @staticmethod
    def opener(reason='stop', content='{"objects":[]}'):
        body = {'choices': [{'finish_reason': reason, 'message': {'content': content}}],
                'usage': {'completion_tokens': 32768,
                          'completion_tokens_details': {'reasoning_tokens': 12}}}
        reply = Mock()
        reply.__enter__ = Mock(return_value=reply)
        reply.__exit__ = Mock(return_value=False)
        reply.read.return_value = json.dumps(body).encode('utf-8')
        opener = Mock()
        opener.open.return_value = reply
        return opener

    def test_two_images_keep_order_bytes_prompt_and_existing_configuration(self):
        opener = self.opener()
        client = VLMClient(self.config, allow_api=True, max_tokens=32768, thinking_mode='disabled')
        prompt = 'Return strict semantic JSON, not TikZ.'
        with patch('vlm_client.urllib.request.build_opener', return_value=opener):
            self.assertEqual(client.complete_images([self.original, self.annotated], prompt), '{"objects":[]}')
        self.assertEqual(opener.open.call_count, 1)
        payload = json.loads(opener.open.call_args.args[0].data)
        content = payload['messages'][0]['content']
        self.assertEqual(content[0], {'type': 'text', 'text': prompt})
        self.assertEqual(len(content), 3)
        for part, path in zip(content[1:], (self.original, self.annotated)):
            self.assertEqual(base64.b64decode(part['image_url']['url'].split(',', 1)[1]), path.read_bytes())
        self.assertEqual(payload['max_tokens'], 32768)
        self.assertEqual(payload['thinking'], {'type': 'disabled'})
        self.assertFalse(payload['stream'])
        self.assertNotIn('response_format', payload)
        self.assertEqual(client.last_diagnostics['finish_reason'], 'stop')

    def test_no_authorization_stops_before_image_io_or_network(self):
        client = VLMClient(self.config)
        with patch('vlm_client.validate_image_path') as image_io, \
             patch('vlm_client.urllib.request.build_opener') as network, \
             self.assertRaises(ApiNotAuthorized):
            client.complete_images([self.original, self.annotated], 'prompt')
        image_io.assert_not_called()
        network.assert_not_called()

    def test_truncation_rejects_even_parseable_json_and_keeps_usage(self):
        client = VLMClient(self.config, allow_api=True, max_tokens=32768)
        opener = self.opener(reason='length')
        with patch('vlm_client.urllib.request.build_opener', return_value=opener), \
             self.assertRaisesRegex(APIResponseError, 'finish_reason=length'):
            client.complete_images([self.original, self.annotated], 'prompt')
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(client.last_diagnostics['usage']['completion_tokens'], 32768)
        self.assertEqual(client.last_diagnostics['usage']['completion_tokens_details']['reasoning_tokens'], 12)
        self.assertIsNone(client.last_truncated_text)

    def test_invalid_arguments_reset_diagnostics_without_network(self):
        client = VLMClient(self.config, allow_api=True)
        cases = [([], 'prompt'), ([self.original] * 3, 'prompt'), (str(self.original), 'prompt'),
                 ([self.original], ''), ([self.original], None), ([self.root / 'missing.png'], 'prompt')]
        with patch('vlm_client.urllib.request.build_opener') as network:
            for paths, prompt in cases:
                client.last_diagnostics = {'stale': True}
                client.last_truncated_text = 'stale'
                with self.subTest(paths=paths), self.assertRaises(ValueError):
                    client.complete_images(paths, prompt)
                self.assertNotIn('stale', client.last_diagnostics)
                self.assertEqual(client.last_diagnostics['read_state'], 'not_started')
                self.assertIsNone(client.last_diagnostics['http_status'])
                self.assertIsNone(client.last_diagnostics['response_byte_length'])
                self.assertFalse(client.last_diagnostics['response_saved'])
                self.assertFalse(client.last_diagnostics['response_recoverable'])
                self.assertIsNone(client.last_truncated_text)
        network.assert_not_called()

    def test_old_single_image_generation_payload_remains_unchanged(self):
        opener = self.opener()
        client = VLMClient(self.config, allow_api=True, max_tokens=32768)
        with patch('vlm_client.urllib.request.build_opener', return_value=opener):
            client.generate(self.original)
        payload = json.loads(opener.open.call_args.args[0].data)
        self.assertEqual(payload['messages'], [{'role': 'user', 'content': [
            {'type': 'text', 'text': GENERATION_PROMPT},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
                base64.b64encode(self.original.read_bytes()).decode('ascii')}},
        ]}])


if __name__ == '__main__':
    unittest.main()
