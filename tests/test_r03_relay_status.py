"""An exact terminal nonce closes a finished command even after noisy scrollback."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'operations/r03_c3fd_main_20260907/tmux_relay.py'
SPEC = importlib.util.spec_from_file_location('r03_relay_status_fixture', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RelayCompletionTest(unittest.TestCase):
    def status(self, captured):
        output = io.StringIO()
        with patch('subprocess.check_output', return_value=captured), contextlib.redirect_stdout(output):
            exec(MODULE.outer_program('abc123', None), {})
        return json.loads(output.getvalue())

    def test_intact_frame_retains_only_actual_output(self):
        result = self.status('prompt\nR03_BEGIN_abc123\nreal-output\nR03_END_abc123 0\nprompt\n')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['output'], 'real-output')
        self.assertFalse(result['output_prefix_lost'])

    def test_lost_start_keeps_exit_and_explicitly_marks_truncated_output(self):
        result = self.status('tar: timestamp warning\n{"ready":true}\nR03_END_abc123 0\nprompt\n')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['returncode'], 0)
        self.assertTrue(result['output_prefix_lost'])
        self.assertIn('"ready":true', result['output'])

    def test_failed_command_cannot_be_reported_as_success(self):
        result = self.status('failure\nR03_END_abc123 7\nprompt\n')
        self.assertEqual(result['returncode'], 7)


if __name__ == '__main__':
    unittest.main()
