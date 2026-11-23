"""Test-only isolation: no developer credentials, root .env or real networking.

The context encloses setUp, the test and cleanups. Per-test fixtures can override
the temporary configuration path. Production loaders are unchanged.
"""
from contextlib import contextmanager, ExitStack
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

CONFIG_NAMES = {'API_KEY','API_BASE_URL','API_BASE','MODEL_NAME','API_MAX_OUTPUT_TOKENS',
                'API_TOKEN_LIMIT_PARAMETER','API_MODEL_OUTPUT_LIMIT','API_THINKING_MODE','API_REASONING_EFFORT'}


@contextmanager
def isolated_test_environment():
    import vlm_client
    clean = {key:value for key,value in os.environ.items() if key not in CONFIG_NAMES}
    with tempfile.TemporaryDirectory(prefix='imagetotikz-offline-test-') as directory, ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ,clean,clear=True))
        stack.enter_context(patch.object(vlm_client,'PROJECT_ENV_FILE',Path(directory)/'.env'))
        for name in ('socket.socket.connect','socket.socket.connect_ex','socket.create_connection',
                     'socket.getaddrinfo','socket.socket.sendto'):
            stack.enter_context(patch(name,side_effect=AssertionError('Real network forbidden in offline tests')))
        yield Path(directory)


class OfflineTestCase(unittest.TestCase):
    def run(self,result=None):
        with isolated_test_environment():
            return super().run(result)
