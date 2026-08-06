from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/a800/run_wq_schedule_correct_bridge_parity_v1.py"
CONTRACT = (
    ROOT
    / "configs/experiments/wyckoff_codiffusion"
    / "wq_schedule_correct_bridge_parity_v1.json"
)


class ScheduleCorrectBridgeRunnerStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_runner_has_no_submission_training_or_network_calls(self) -> None:
        lowered = self.source.lower()
        for forbidden in (
            "sbatch",
            "srun",
            "subprocess",
            "requests.",
            "mprester",
            "chgnet",
            "mattersim",
            ".backward(",
            "optimizer",
        ):
            self.assertNotIn(forbidden, lowered)

    def test_runner_reuses_exact_source_and_never_generates_proposals(self) -> None:
        self.assertIn("source_generation", self.source)
        self.assertIn("proposal_state", self.source)
        self.assertNotIn(".propose(", self.source)
        self.assertNotIn(".generate(", self.source)
        self.assertFalse(self.contract["source_panel"]["new_generation"])
        self.assertFalse(
            self.contract["source_panel"]["selection_uses_scientific_outcomes"]
        )

    def test_runner_strict_loads_and_records_all_32_cells(self) -> None:
        self.assertIn("load_registered_csp", self.source)
        self.assertIn("strict_load", self.source)
        self.assertIn("BRIDGE_CELL_COUNT", self.source)
        self.assertIn("attempt_rows.append", self.source)
        self.assertEqual(self.contract["matrix"]["total_cells"], 32)
        self.assertEqual(
            self.contract["gates"]["successful_positive_volume_outputs"], 32
        )

    def test_failure_evidence_is_written_without_retry(self) -> None:
        self.assertIn('"status": "failed"', self.source)
        self.assertIn("terminal_report.json", self.source)
        self.assertIn('"retry_or_replacement_used": False', self.source)
        self.assertNotIn("while ", self.source)

    def test_resource_envelope_respects_hard_cpu_rule(self) -> None:
        resources = self.contract[
            "future_resource_envelope_not_authorized_for_submission"
        ]
        self.assertLessEqual(resources["cpus"], 8 * resources["a800"])


if __name__ == "__main__":
    unittest.main()
