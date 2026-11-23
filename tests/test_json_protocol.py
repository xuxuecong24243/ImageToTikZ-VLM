"""All HTTP responses are mocked; never contact an API to test JSON mode."""
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
from auto_scene_generator import complete_scene_images
from auto_pipeline import RequestLedger, build_parser
from vlm_client import APIConfig, APIResponseError, VLMClient


class JsonProtocolTests(OfflineTestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start(); self.addCleanup(environment.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root/'original.png'
        Image.new('RGB', (20, 10), 'white').save(self.image)

    def client(self, base='https://api.deepseek.com', model='deepseek-flash'):
        return VLMClient(APIConfig(base, 'offline-secret', model), allow_api=True, max_tokens=2048)

    def opener(self, reason='stop'):
        reply = Mock(status=200)
        reply.__enter__ = Mock(return_value=reply)
        reply.__exit__ = Mock(return_value=False)
        reply.read.return_value = json.dumps({'choices': [{'finish_reason': reason,
            'message': {'content': '{"canvas":{}}'}}], 'usage': {'total_tokens': 50}}).encode()
        return Mock(open=Mock(return_value=reply))

    def test_scene_only_enables_official_json_output_and_records_http_status(self):
        opener = self.opener(); client = self.client()
        with patch('vlm_client.urllib.request.build_opener', return_value=opener):
            complete_scene_images(client, [self.image], 'Return Scene JSON.')
        payload = json.loads(opener.open.call_args.args[0].data)
        self.assertEqual(payload['response_format'], {'type': 'json_object'})
        self.assertEqual(payload['thinking']['type'], 'disabled')
        self.assertEqual(client.last_diagnostics['http_status'], 200)
        self.assertEqual(client.last_diagnostics['requested_response_format'], {'type': 'json_object'})

    def test_pipeline_ledger_uses_scene_json_mode(self):
        opener = self.opener(); client = self.client()
        args = build_parser().parse_args(['--image', str(self.image), '--allow-api'])
        with patch('vlm_client.urllib.request.build_opener', return_value=opener):
            ledger = RequestLedger(client, args, self.root)
            ledger.request([self.image], 'Return Scene JSON.', 'generate')
        self.assertIn('response_format', json.loads(opener.open.call_args.args[0].data))
        self.assertEqual(ledger.summary()['request_count'], 1)

    def test_direct_tikz_and_unknown_services_remain_unchanged(self):
        for scenario in ('tikz', 'other_host', 'other_model'):
            opener = self.opener()
            client = self.client(base='https://example.test' if scenario == 'other_host' else 'https://api.deepseek.com',
                                 model='unverified-model' if scenario == 'other_model' else 'deepseek-flash')
            with patch('vlm_client.urllib.request.build_opener', return_value=opener):
                if scenario == 'tikz':
                    client.generate(self.image)
                else:
                    complete_scene_images(client, [self.image], 'Return Scene JSON.')
            self.assertNotIn('response_format', json.loads(opener.open.call_args.args[0].data))

    def test_json_mode_retains_truncation_rejection_and_authorization(self):
        client = self.client(); opener = self.opener(reason='length')
        with patch('vlm_client.urllib.request.build_opener', return_value=opener):
            with self.assertRaises(APIResponseError):
                complete_scene_images(client, [self.image], 'Return Scene JSON.')
        self.assertEqual(opener.open.call_count, 1)
        client.allow_api = False
        with patch('vlm_client.urllib.request.build_opener') as network:
            with self.assertRaises(APIResponseError.__bases__[0]):
                complete_scene_images(client, [self.image], 'Return Scene JSON.')
            network.assert_not_called()

    def test_arbitrary_response_fields_rejected_before_network(self):
        with patch('vlm_client.urllib.request.build_opener') as network:
            with self.assertRaises(ValueError):
                self.client().complete_images([self.image], 'JSON', response_format={'type': 'json_object', 'extra': True})
            network.assert_not_called()


if __name__ == '__main__':
    unittest.main()
