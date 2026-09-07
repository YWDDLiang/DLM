"""A stale or false canary receipt must not authorize expensive 256 sampling."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / 'operations/r03_c3fd_main_20260907/trial_preflight.py'
SPEC = importlib.util.spec_from_file_location('r03_trial_preflight_fixture', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AcceptedCanaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'code' / 'source-fixture'
        self.source.mkdir(parents=True)
        (self.source / '_CODE_READY').touch()
        directory = self.root / 'canary_geometry_123'
        directory.mkdir()
        (directory / '_SUCCESS').touch()
        self.complete = self.root / 'GEOMETRY_CANARY_COMPLETE.json'
        self.complete.write_text(json.dumps({'source': str(self.source), 'directory': str(directory),
                                             'requests_per_arm': 16, 'roles': ['G', 'P']}))
        self.accepted = {'accepted': True, 'source': str(self.source), 'directory': str(directory),
                         'geometry_canary_complete_sha256': hashlib.sha256(self.complete.read_bytes()).hexdigest()}
        self.save_acceptance()

    def save_acceptance(self):
        (self.root / 'GEOMETRY_CANARY_ACCEPTED.json').write_text(json.dumps(self.accepted))

    def test_matching_explicit_acceptance_passes(self):
        self.assertEqual(MODULE.require_geometry_canary_acceptance(self.root, self.source), self.accepted)

    def test_false_acceptance_is_not_permission(self):
        self.accepted['accepted'] = False
        self.save_acceptance()
        with self.assertRaises(ValueError):
            MODULE.require_geometry_canary_acceptance(self.root, self.source)

    def test_replaced_completion_receipt_is_not_the_reviewed_canary(self):
        self.complete.write_text(self.complete.read_text() + '\n')
        with self.assertRaises(ValueError):
            MODULE.require_geometry_canary_acceptance(self.root, self.source)

    def test_new_archive_cannot_inherit_old_acceptance(self):
        alternate = self.root / 'code' / 'different-source'
        alternate.mkdir()
        (alternate / '_CODE_READY').touch()
        with self.assertRaises(ValueError):
            MODULE.require_geometry_canary_acceptance(self.root, alternate)

    def test_missing_source_is_not_inferred_from_working_directory(self):
        del self.accepted['source']
        self.save_acceptance()
        with self.assertRaisesRegex(ValueError, 'explicit absolute'):
            MODULE.require_geometry_canary_acceptance(self.root, self.source)


if __name__ == '__main__':
    unittest.main()
