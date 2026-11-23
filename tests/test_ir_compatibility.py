"""Historical validation/TikZ expectations with portable public fixtures.

Public copies redact only metadata machine paths. Original and public hashes
are distinct; the original-byte archive check remains independent and optional
when private outputs are absent. All 56 validation and 49 TikZ expectations run
on a clean checkout. No compilation, API, or historical artifact writes occur.
"""
import hashlib
import json
from pathlib import Path
import unittest

try:
    from .offline_support import OfflineTestCase
except ImportError:
    from offline_support import OfflineTestCase

import structured_tikz.schema as scene_schema
import structured_tikz.validator as scene_validator
import structured_tikz.generator as scene_generator


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'historical'
PREVIOUS = FIXTURES / 'compatibility_report.json'
# This is explicitly the one original package-export hash, not the full private
# archive or API-response manifest. See fixtures/historical/README.md.
BEFORE = FIXTURES / 'native_exports_sha256.json'


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _corpus():
    return json.loads(PREVIOUS.read_text(encoding='utf-8'))['results']


def _scene(record):
    return json.loads((FIXTURES / record['path']).read_text(encoding='utf-8-sig'))


class IRHistoricalCompatibilityTests(OfflineTestCase):
    def test_all_56_historical_scene_inputs_unchanged(self):
        records = _corpus()
        self.assertEqual(len(records), 56)
        if any(not (ROOT / record['original_path']).is_file() for record in records):
            self.skipTest('Private historical archive unavailable; public 56-scene/49-TikZ checks still run.')
        for record in records:
            with self.subTest(path=record['original_path']):
                self.assertEqual(_sha((ROOT / record['original_path']).read_bytes()), record['sha256'])
                self.assertEqual(record['sha256'], record['original_sha256'])

    def test_all_56_public_fixture_inputs_unchanged(self):
        records = _corpus()
        self.assertEqual(len(records), 56)
        for record in records:
            with self.subTest(path=record['path']):
                self.assertEqual(_sha((FIXTURES / record['path']).read_bytes()), record['public_sha256'])
                if not record['redacted_fields']:
                    self.assertEqual(record['public_sha256'], record['original_sha256'])

    def test_all_56_historical_validation_conclusions_unchanged(self):
        records = _corpus()
        valid_count, invalid_count = 0, 0
        for record in records:
            with self.subTest(path=record['path']):
                data = _scene(record)
                original = json.dumps(data, ensure_ascii=False, sort_keys=True)
                try:
                    scene_validator.validate_scene(data)
                    accepted, error_type = True, None
                except Exception as exc:
                    accepted, error_type = False, type(exc).__name__
                expected = record['new_validation']
                self.assertEqual(accepted, expected['accepted'])
                if not accepted:
                    self.assertEqual(error_type, expected['error_type'])
                self.assertEqual(json.dumps(data, ensure_ascii=False, sort_keys=True), original)
                valid_count += accepted
                invalid_count += not accepted
        self.assertEqual((valid_count, invalid_count), (49, 7))

    def test_all_49_valid_scene_tikz_outputs_byte_identical(self):
        checked = 0
        for record in _corpus():
            if not record['new_validation']['accepted']:
                continue
            with self.subTest(path=record['path']):
                data = _scene(record)
                original = json.dumps(data, ensure_ascii=False, sort_keys=True)
                result = scene_generator.generate_tikz(data)
                self.assertEqual(_sha(result.encode('utf-8')), record['new_render']['sha256'])
                self.assertEqual(len(result), record['new_render']['characters'])
                self.assertEqual(json.dumps(data, ensure_ascii=False, sort_keys=True), original)
                checked += 1
        self.assertEqual(checked, 49)

    def test_native_entry_points_and_package_exports_unchanged(self):
        manifest = json.loads(BEFORE.read_text(encoding='utf-8'))
        # Native diamond support intentionally changes implementation files.
        # Behavioral compatibility is checked above against all 49 exact output
        # hashes and 56 validation results; package exports remain untouched.
        for name in ('__init__',):
            relative = 'structured_tikz/' + name + '.py'
            with self.subTest(module=name):
                self.assertEqual(_sha((ROOT / relative).read_bytes()), manifest[relative])
        self.assertEqual(scene_validator.validate_scene.__module__, 'structured_tikz.validator')
        self.assertEqual(scene_generator.generate_tikz.__module__, 'structured_tikz.generator')
        self.assertEqual(scene_schema.scene_contract.__module__, 'structured_tikz.schema')


if __name__ == '__main__':
    unittest.main()
