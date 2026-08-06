from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import assemble_dependency_margin
from evaluate_dependency_margin import (
    aggregate,
    bootstrap_mean_ci,
    select_donor_indices,
)


def mock_row(num_atoms: int, answer: str, identity: str) -> dict:
    return {
        "answer": answer,
        "training_pair_sha256": identity * 64,
        "plangraph": {"composition": {"N": num_atoms}},
    }


def arm_report(arm: str, margin: float) -> dict:
    rows = [
        {
            "panel_ordinal": ordinal,
            "group_pairs": 3,
            "dependency_margin": margin + ordinal / 100_000.0,
        }
        for ordinal in range(100)
    ]
    return {
        "status": "complete",
        "arm": arm,
        "checkpoint_adapter_sha256": assemble_dependency_margin.EXPECTED_CHECKPOINTS[
            arm
        ],
        "result": {
            "panel_rows": 100,
            "row_records": rows,
            "arm_margin": bootstrap_mean_ci(
                [row["dependency_margin"] for row in rows],
                replicates=100,
            ),
        },
    }


class DependencyMarginTests(unittest.TestCase):
    def test_donor_rule_forward_then_wrap(self) -> None:
        rows = [
            mock_row(4, "a", "a"),
            mock_row(5, "b", "b"),
            mock_row(4, "c", "c"),
            mock_row(5, "d", "d"),
        ]
        self.assertEqual(select_donor_indices(rows, panel_rows=4), [2, 3, 0, 1])

    def test_bootstrap_is_deterministic(self) -> None:
        left = bootstrap_mean_ci([0.1, 0.2, 0.4], replicates=100)
        right = bootstrap_mean_ci([0.1, 0.2, 0.4], replicates=100)
        self.assertEqual(left, right)
        self.assertAlmostEqual(left["mean"], 0.23333333333333334)

    def test_aggregate_retains_all_rows(self) -> None:
        scored = [
            {
                "panel_ordinal": ordinal,
                "active_group": "g",
                "dependency_margin": ordinal / 1000.0,
            }
            for ordinal in range(100)
        ]
        result = aggregate(scored)
        self.assertEqual(result["panel_rows"], 100)
        self.assertEqual(result["pair_count"], 100)

    def _run_assembly(self, *, b1: float, b2: float) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm, margin in (("B1", b1), ("B2", b2)):
                path = root / "arms" / arm / "dependency_report.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(arm_report(arm, margin)) + "\n")
            training = {
                "status": "complete",
                "training_complete": True,
                "B2_likelihood_gate_passed": True,
            }
            training_path = root / "training.json"
            training_path.write_text(json.dumps(training) + "\n")
            output = root / "terminal.json"
            argv = [
                "assemble_dependency_margin.py",
                "--run-root",
                str(root),
                "--dlm-training-terminal",
                str(training_path),
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(assemble_dependency_margin.main(), 0)
            return json.loads(output.read_text())

    def test_pass_keeps_completion_pending(self) -> None:
        terminal = self._run_assembly(b1=0.1, b2=0.2)
        self.assertTrue(terminal["dependency_gate_passed"])
        self.assertFalse(terminal["Bstar_selected"])
        self.assertTrue(terminal["conditional_body_completion_pending"])

    def test_scientific_stop(self) -> None:
        terminal = self._run_assembly(b1=0.2, b2=0.1)
        self.assertFalse(terminal["dependency_gate_passed"])
        self.assertEqual(terminal["decision"], "scientific_stop_retain_B0")


if __name__ == "__main__":
    unittest.main()
