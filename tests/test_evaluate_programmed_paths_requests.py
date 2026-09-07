"""CLI/ledger contracts only: no N/U, model, GPU or physical evaluator runs."""
from __future__ import annotations

from contextlib import redirect_stderr
import importlib.util
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_programmed_paths_request_contract", ROOT / "scripts/evaluate_programmed_paths.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def arguments(requests):
    return [
        "evaluate_programmed_paths.py", "--paths-jsonl", "unused_paths.jsonl",
        "--labels-jsonl", "unused_labels/labels.jsonl", "--frozen-config", "unused_config.json",
        "--official-cache", "unused_cache", "--output-dir", "unused_output",
        "--expected-requests", str(requests), "--endpoint", "native",
        "--cohort-role", "independent_main", "--policy-stage", "final",
    ]


class ExpectedRequestsCLITest(unittest.TestCase):
    def test_registered_sizes_pass_cli_and_reach_allocation_boundary(self):
        for size in (16, 128, 256, 500, 512, 1000, 1200):
            with self.subTest(size=size), patch.object(sys, "argv", arguments(size)), \
                    patch.dict(MODULE.os.environ, {}, clear=True), \
                    patch.object(MODULE, "read_jsonl") as reader:
                with self.assertRaisesRegex(RuntimeError, "allocated workflow CPUs"):
                    MODULE.main()
                reader.assert_not_called()

    def test_nonpositive_or_noninteger_sizes_fail_cli_before_any_evaluation(self):
        for size in (0, -1, -1000, "1.5", "nan"):
            output = StringIO()
            with self.subTest(size=size), patch.object(sys, "argv", arguments(size)), \
                    redirect_stderr(output), patch.object(MODULE, "read_jsonl") as reader:
                with self.assertRaises(SystemExit) as raised:
                    MODULE.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("positive integer", output.getvalue())
                reader.assert_not_called()

    def test_actual_row_count_must_still_equal_declared_denominator(self):
        records = [{"trajectory_id": "r0", "evaluation_ordinal": 0, "sample_idx": 100}]
        with patch.object(sys, "argv", arguments(16)), \
                patch.dict(MODULE.os.environ, {"SLURM_JOB_ID": "cpu-fixture"}), \
                patch.object(MODULE, "read_jsonl", return_value=records):
            with self.assertRaisesRegex(ValueError, "request denominator changed"):
                MODULE.main()

    def test_actual_evaluation_order_must_still_cover_the_full_cohort(self):
        records = [
            {"trajectory_id": "r0", "evaluation_ordinal": 1, "sample_idx": 100},
            {"trajectory_id": "r1", "evaluation_ordinal": 0, "sample_idx": 101},
        ]
        with patch.object(sys, "argv", arguments(2)), \
                patch.dict(MODULE.os.environ, {"SLURM_JOB_ID": "cpu-fixture"}), \
                patch.object(MODULE, "read_jsonl", return_value=records):
            with self.assertRaisesRegex(ValueError, "evaluation source order changed"):
                MODULE.main()


if __name__ == "__main__":
    unittest.main()
