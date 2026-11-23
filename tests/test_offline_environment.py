import os
from pathlib import Path
import socket
import unittest
try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase
from unittest.mock import patch
import vlm_client
try:
    from .offline_support import isolated_test_environment
except ImportError:
    from offline_support import isolated_test_environment


class OfflineEnvironmentTests(OfflineTestCase):
    def test_root_file_and_process_credentials_not_visible(self):
        with patch.dict(os.environ,{'API_KEY':'fixture-not-a-real-key','MODEL_NAME':'fixture-model'}):
            with isolated_test_environment() as root:
                self.assertEqual(vlm_client.PROJECT_ENV_FILE,root/'.env')
                self.assertNotIn('API_KEY',os.environ)
                with self.assertRaises(vlm_client.ConfigError):vlm_client.APIConfig.from_env()
            self.assertEqual(os.environ['API_KEY'],'fixture-not-a-real-key')

    def test_temp_file_loading_and_environment_priority_still_work(self):
        with isolated_test_environment() as root:
            (root/'.env').write_text('API_BASE_URL=https://fixture.test\nAPI_KEY=fixture-key\nMODEL_NAME=file-model\n',encoding='utf-8')
            self.assertEqual(vlm_client.APIConfig.from_env().model,'file-model')
            os.environ['MODEL_NAME']='process-model'
            self.assertEqual(vlm_client.APIConfig.from_env().model,'process-model')

    def test_file_pointer_and_environment_restored_after_error(self):
        old=vlm_client.PROJECT_ENV_FILE;before=dict(os.environ)
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with isolated_test_environment():raise RuntimeError('fixture')
        self.assertEqual(vlm_client.PROJECT_ENV_FILE,old);self.assertEqual(dict(os.environ),before)

    def test_network_blocked_including_loopback_no_real_socket_operation(self):
        with isolated_test_environment():
            for call in (lambda:socket.create_connection(('127.0.0.1',1)),
                         lambda:socket.getaddrinfo('example.invalid',443)):
                with self.assertRaisesRegex(AssertionError,'Real network forbidden'):call()


if __name__=='__main__':unittest.main()
