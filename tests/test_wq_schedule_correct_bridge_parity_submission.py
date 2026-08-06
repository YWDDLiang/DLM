from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiments/wyckoff_codiffusion"
    / "wq_schedule_correct_bridge_parity_execution_v1.json"
)
CONTRACT = (
    ROOT
    / "configs/experiments/wyckoff_codiffusion"
    / "wq_schedule_correct_bridge_parity_v1.json"
)
JOB_DIR = ROOT / "scripts/a800/wq_schedule_correct_bridge_parity_v1"
SBATCH = JOB_DIR / "preflight.sbatch"
SUBMIT = JOB_DIR / "submit_once.sh"
AUTHORIZATION = (
    ROOT
    / "diagnostics/authorization_records"
    / "wq_schedule_correct_bridge_parity_v1_remote_execution.json"
)
INSTALLER = ROOT / "scripts/a800/install_authorized_patch.py"
GATE = ROOT / "crystal_dlm/wqcodiff/crysllmgen/gate.py"


class ScheduleCorrectBridgeParitySubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        cls.authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
        cls.sbatch = SBATCH.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")
        cls.installer = INSTALLER.read_text(encoding="utf-8")
        cls.gate = GATE.read_text(encoding="utf-8")

    def test_single_job_resource_gate_is_exact_and_fail_closed(self) -> None:
        self.assertIn("#SBATCH --partition=gpu", self.sbatch)
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", self.sbatch)
        self.assertIn("#SBATCH --cpus-per-task=8", self.sbatch)
        self.assertIn("#SBATCH --mem=64G", self.sbatch)
        self.assertIn("#SBATCH --time=01:00:00", self.sbatch)
        self.assertNotIn("#SBATCH --array", self.sbatch)
        policy = self.submit.index("cpus > 8 * gpus")
        claim = self.submit.index('path.open("x"')
        sbatch = self.submit.index("sbatch --parsable")
        self.assertLess(policy, claim)
        self.assertLess(claim, sbatch)

    def test_submission_is_unique_and_does_not_modify_other_jobs(self) -> None:
        self.assertIn('test ! -e "$RECORD"', self.submit)
        self.assertIn('test ! -e "$CLAIM"', self.submit)
        self.assertIn('test ! -e "$OUTPUT"', self.submit)
        self.assertEqual(self.submit.count("sbatch --parsable"), 2)
        self.assertNotIn("scancel", self.submit)
        self.assertNotIn("scontrol update", self.submit)
        self.assertNotIn("squeue --start", self.submit)
        self.assertIn("same bridge-parity job identity already exists", self.submit)

    def test_sbatch_is_evaluation_only_and_uses_frozen_inputs(self) -> None:
        lowered = self.sbatch.lower()
        self.assertIn("conda activate diff_meets_diff", self.sbatch)
        self.assertIn("run_wq_schedule_correct_bridge_parity_v1.py", self.sbatch)
        self.assertIn(
            self.contract["source_panel"]["generation_jsonl_sha256"],
            self.sbatch,
        )
        self.assertIn(self.contract["parent"]["checkpoint_sha256"], self.sbatch)
        self.assertNotIn("train.py", lowered)
        self.assertNotIn("torchrun", lowered)
        self.assertNotIn("optimizer", lowered)
        self.assertNotIn("mprester", lowered)
        self.assertNotIn("chgnet", lowered)
        self.assertNotIn("mattersim", lowered)

    def test_execution_plan_and_authorization_forbid_training_and_retry(self) -> None:
        self.assertEqual(self.plan["status"], "authorized_not_submitted")
        self.assertFalse(self.plan["job"]["training"])
        self.assertFalse(
            self.plan["submission_semantics"]["retry_or_replacement_allowed"]
        )
        self.assertEqual(self.plan["job"]["cpus"], 8)
        self.assertEqual(self.plan["job"]["a800"], 1)
        forbidden = "\n".join(
            self.authorization["interpretation"]["not_authorized_by_this_record"]
        )
        self.assertRegex(forbidden, re.compile(r"short bridge training"))
        self.assertRegex(forbidden, re.compile(r"long bridge training"))

    def test_terminal_verifier_requires_all_32_cells(self) -> None:
        self.assertIn('observed.get("terminal_cells") != 32', self.sbatch)
        self.assertIn(
            'observed.get("successful_positive_volume_outputs") != 32',
            self.sbatch,
        )
        self.assertIn('report.get("training_performed") is not False', self.sbatch)
        self.assertTrue(
            self.plan["terminal_semantics"]["pass_does_not_authorize_short_training"]
        )

    def test_exact_remote_authorization_is_registered_for_install_and_gate(self) -> None:
        identity = self.plan["authorization"]["identity"]
        self.assertIn(f'"{identity}"', self.installer)
        self.assertIn(f'"{identity}"', self.gate)


if __name__ == "__main__":
    unittest.main()
