from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/wyckoff_codiffusion"
OLD_CONTRACT = CONFIG / "wq_schedule_correct_bridge_parity_v1.json"
CONTRACT = CONFIG / "wq_schedule_correct_bridge_parity_sup28054_v1.json"
PLAN = CONFIG / "wq_schedule_correct_bridge_parity_sup28054_execution_v1.json"
JOB_DIR = ROOT / "scripts/a800/wq_schedule_correct_bridge_parity_sup28054_v1"
SBATCH = JOB_DIR / "preflight.sbatch"
SUBMIT = JOB_DIR / "submit_once.sh"
AUTHORIZATION_ARCHIVE = (
    ROOT
    / "diagnostics/authorization_records"
    / "wq_schedule_correct_bridge_parity_sup28054_v1_remote_execution.json"
)
AUTHORIZATION_LOCAL = (
    ROOT
    / "runs/remote_audit"
    / "20260726_wq_schedule_correct_bridge_parity_sup28054_v1"
    / "authorization_record.json"
)
AUTHORIZATION = (
    AUTHORIZATION_ARCHIVE
    if AUTHORIZATION_ARCHIVE.is_file()
    else AUTHORIZATION_LOCAL
)
FAILURE_AUDIT_ARCHIVE = (
    ROOT
    / "diagnostics/failure_audits"
    / "wq_schedule_correct_bridge_parity_job28054.json"
)
FAILURE_AUDIT_LOCAL = (
    ROOT
    / "runs/remote_audit"
    / "20260726_wq_schedule_correct_bridge_parity_v1"
    / "job28054_terminal_failure_audit.json"
)
FAILURE_AUDIT = (
    FAILURE_AUDIT_ARCHIVE
    if FAILURE_AUDIT_ARCHIVE.is_file()
    else FAILURE_AUDIT_LOCAL
)
RUNNER = ROOT / "scripts/a800/run_wq_schedule_correct_bridge_parity_v1.py"
INSTALLER = ROOT / "scripts/a800/install_authorized_patch.py"
GATE = ROOT / "crystal_dlm/wqcodiff/crysllmgen/gate.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ScheduleCorrectBridgeParitySup28054SubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old = json.loads(OLD_CONTRACT.read_text(encoding="utf-8"))
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))
        cls.authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
        cls.failure = json.loads(FAILURE_AUDIT.read_text(encoding="utf-8"))
        cls.sbatch = SBATCH.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")

    def test_job28054_evidence_is_frozen_and_science_never_started(self) -> None:
        self.assertEqual(
            sha256(FAILURE_AUDIT),
            "99b7b57b80c6d097adaa22bf3fa39d761573f4e76f6c74b89b55947081c95dca",
        )
        self.assertEqual(
            sha256(OLD_CONTRACT),
            "d4f18bf74a1814d7de6d7a4d4934c615857edef364a039f371723aa1763b4c6b",
        )
        boundary = self.failure["failure_boundary"]
        self.assertEqual(boundary["scientific_trajectory_attempts"], 0)
        self.assertEqual(boundary["bridge_cells_started"], 0)
        self.assertFalse(boundary["parent_strict_load_started"])
        self.assertFalse(
            self.failure["submission_identity"]["retry_or_replacement_used"]
        )

    def test_only_scientific_delta_is_exact_source_schema_binding(self) -> None:
        for key in (
            "parent",
            "matrix",
            "bridge_semantics",
            "gates",
            "model_selection",
        ):
            self.assertEqual(self.old[key], self.contract[key], key)
        old_panel = dict(self.old["source_panel"])
        new_panel = dict(self.contract["source_panel"])
        self.assertEqual(
            old_panel.pop("required_schema"),
            "wq_parent_csp_probe_attempt_v1",
        )
        self.assertEqual(
            new_panel.pop("required_schema"),
            "wqcodiff_generation_attempt_v1",
        )
        self.assertEqual(old_panel, new_panel)
        self.assertTrue(
            self.contract["supersession"][
                "scientific_delta_is_only_source_schema_binding"
            ]
        )

    def test_input_hashes_matrix_and_runner_are_reused_exactly(self) -> None:
        self.assertEqual(
            self.contract["source_panel"]["generation_jsonl_sha256"],
            "b6eb7f80a29da699407d8d19bbedeb2d657f5d7940cd767d6d71aecb6c58a598",
        )
        self.assertEqual(
            self.contract["parent"]["checkpoint_sha256"],
            "573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e",
        )
        self.assertEqual(self.contract["matrix"]["timesteps"], [100, 200, 400, 800])
        self.assertEqual(self.contract["matrix"]["total_cells"], 32)
        self.assertEqual(
            sha256(RUNNER),
            "22dd1ad3884fd6bf7fc8832e12372cecba53c35e89a2eca977fef121177f310b",
        )

    def test_resource_gate_is_exact_and_precedes_exclusive_claim(self) -> None:
        self.assertIn("#SBATCH --partition=gpu", self.sbatch)
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", self.sbatch)
        self.assertIn("#SBATCH --cpus-per-task=8", self.sbatch)
        self.assertIn("#SBATCH --mem=64G", self.sbatch)
        self.assertIn("#SBATCH --time=01:00:00", self.sbatch)
        self.assertNotIn("#SBATCH --array", self.sbatch)
        policy = self.submit.index("cpus > 8 * gpus")
        source_schema = self.submit.index("source rows do not match")
        claim = self.submit.index('path.open("x"')
        sbatch = self.submit.index("sbatch --parsable")
        self.assertLess(policy, source_schema)
        self.assertLess(source_schema, claim)
        self.assertLess(claim, sbatch)

    def test_submit_is_unique_and_preserves_old_identity(self) -> None:
        self.assertIn('test ! -e "$RECORD"', self.submit)
        self.assertIn('test ! -e "$CLAIM"', self.submit)
        self.assertIn('test ! -e "$OUTPUT"', self.submit)
        self.assertIn("$OLD_RECORD_SHA256", self.submit)
        self.assertIn("$OLD_CLAIM_SHA256", self.submit)
        self.assertIn("$OLD_STDOUT_SHA256", self.submit)
        self.assertIn("$OLD_STDERR_SHA256", self.submit)
        self.assertIn("$OLD_GPU_SHA256", self.submit)
        self.assertIn("$OLD_TERMINAL_SHA256", self.submit)
        self.assertEqual(self.submit.count("sbatch --parsable"), 2)
        self.assertNotIn("scancel", self.submit)
        self.assertNotIn("scontrol update", self.submit)
        self.assertNotIn("squeue --start", self.submit)
        self.assertIn(
            "same bridge-parity supersession job identity already exists",
            self.submit,
        )

    def test_sbatch_is_evaluation_only_and_fails_closed_on_schema(self) -> None:
        lowered = self.sbatch.lower()
        self.assertIn("conda activate diff_meets_diff", self.sbatch)
        self.assertIn("run_wq_schedule_correct_bridge_parity_v1.py", self.sbatch)
        self.assertIn("bridge_parity_sup28054_source_schema=PASS", self.sbatch)
        self.assertIn("wqcodiff_generation_attempt_v1", self.sbatch)
        self.assertNotRegex(lowered, re.compile(r"\btrain\.py\b"))
        self.assertNotIn("torchrun", lowered)
        self.assertNotIn("optimizer", lowered)
        self.assertNotIn("mprester", lowered)
        self.assertNotIn("chgnet", lowered)
        self.assertNotIn("mattersim", lowered)

    def test_authorization_and_plan_forbid_training_and_retry(self) -> None:
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
        self.assertIn("short bridge training", forbidden)
        self.assertIn("long bridge training", forbidden)
        self.assertIn("a second supersession submission", forbidden)

    def test_terminal_gate_requires_all_32_cells_and_no_training(self) -> None:
        self.assertIn('observed.get("terminal_cells") != 32', self.sbatch)
        self.assertIn(
            'observed.get("successful_positive_volume_outputs") != 32',
            self.sbatch,
        )
        self.assertIn('report.get("training_performed") is not False', self.sbatch)
        self.assertTrue(
            self.plan["terminal_semantics"]["pass_does_not_authorize_short_training"]
        )

    def test_exact_authorization_is_registered_for_install_and_gate(self) -> None:
        identity = self.plan["authorization"]["identity"]
        self.assertIn(
            f'"{identity}"',
            INSTALLER.read_text(encoding="utf-8"),
        )
        self.assertIn(f'"{identity}"', GATE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
