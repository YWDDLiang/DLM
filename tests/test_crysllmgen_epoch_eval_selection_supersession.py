from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BATCH = (
    ROOT
    / "configs/experiments/wyckoff_codiffusion/"
    "epoch_eval_selection_supersession27104_v1_dag.json"
)
AUTHORIZATION = (
    ROOT
    / "runs/remote_audit/"
    "20260723_user_authorization_epoch_eval_selection_supersession27104_v1.json"
)
FAILURE_AUDIT = (
    ROOT
    / "runs/remote_audit/"
    "20260723_refiner27103_pass_epoch_eval27104_identity_failure_v1.json"
)
JOB_DIR = ROOT / "scripts/a800/epoch_eval_selection_supersession27104_v1"
EVAL = JOB_DIR / "epoch_eval.sbatch"
SELECT = JOB_DIR / "select_epoch.sbatch"
SUBMIT = JOB_DIR / "submit_once.sh"
INSTALLER = ROOT / "scripts/a800/install_authorized_patch.py"
GATE = ROOT / "crystal_dlm/wqcodiff/crysllmgen/gate.py"
NLL = ROOT / "scripts/a800/evaluate_crysllmgen_lora_nll.py"
SAMPLE = ROOT / "scripts/a800/sample_crysllmgen_wq.py"
SAMPLING = ROOT / "crystal_dlm/wqcodiff/crysllmgen/wq_sampling.py"
EVIDENCE = ROOT / "scripts/a800/build_crysllmgen_epoch_evidence.py"
SELECTION_CORE = ROOT / "crystal_dlm/wqcodiff/crysllmgen/epoch_selection.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EpochEvalSelectionSupersessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.batch = json.loads(BATCH.read_text(encoding="utf-8"))
        cls.eval = EVAL.read_text(encoding="utf-8")
        cls.select = SELECT.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")

    def test_authorization_and_failure_lineage_are_exact(self) -> None:
        self.assertEqual(
            self.batch["authorization"]["sha256"],
            _sha256(AUTHORIZATION),
        )
        self.assertEqual(
            self.batch["failure_being_superseded"]["audit_sha256"],
            _sha256(FAILURE_AUDIT),
        )
        self.assertEqual(
            self.batch["failure_being_superseded"][
                "epoch_evaluation_array_job_id"
            ],
            "27104",
        )
        authorization_name = (
            "user_epoch_eval_selection_supersession27104_v1_2026-07-23"
        )
        self.assertIn(authorization_name, INSTALLER.read_text(encoding="utf-8"))
        self.assertIn(authorization_name, GATE.read_text(encoding="utf-8"))

    def test_only_evaluation_and_selection_are_submitted_once(self) -> None:
        accounting = self.batch["submission_accounting"]
        self.assertEqual(
            accounting["slurm_submit_slots_including_array_elements"],
            4,
        )
        self.assertEqual(accounting["maximum_concurrent_a800"], 2)
        self.assertEqual(len(self.batch["jobs"]), 2)
        self.assertEqual(self.submit.count("submit_or_record_failure eval_job"), 1)
        self.assertEqual(
            self.submit.count("submit_or_record_failure selection_job"),
            1,
        )
        self.assertNotIn("refiner_candidate.sbatch", self.submit)
        self.assertNotIn("formal_train.sbatch", self.submit)
        self.assertIn("preflight waits for zero preexisting user GPU jobs", self.submit)
        self.assertIn('--dependency="afterok:$eval_job"', self.submit)
        self.assertIn("--array=0-2%2", self.eval)
        self.assertIn("claimed_before_any_sbatch", self.submit)
        self.assertLess(
            self.submit.index('with Path(path).open("x"'),
            self.submit.index("submit_or_record_failure eval_job"),
        )
        self.assertIn('"sbatch_command": eval_command or None', self.submit)
        self.assertIn('"sbatch_command": selection_command or None', self.submit)

    def test_registered_job_scripts_match_frozen_hashes(self) -> None:
        for job in self.batch["jobs"]:
            path = ROOT / job["script"]
            self.assertEqual(job["script_sha256"], _sha256(path), path)
        self.assertIn(_sha256(BATCH), self.submit)

    def test_three_execution_identities_are_never_relabelled(self) -> None:
        adapter_patch = (
            self.batch["frozen_identity"][
                "formal_lora_training_execution_patch_sha256"
            ]
        )
        refiner_patch = (
            self.batch["frozen_identity"][
                "refiner_training_execution_patch_sha256"
            ]
        )
        self.assertIn(adapter_patch, self.eval)
        self.assertIn(refiner_patch, self.eval)
        self.assertIn(
            "--adapter-training-execution-patch-sha256",
            self.eval,
        )
        self.assertIn(
            "--refiner-training-execution-patch-sha256",
            self.eval,
        )
        self.assertIn("--evaluation-execution-patch-sha256", self.select)
        self.assertIn(
            "--adapter-training-execution-patch-sha256",
            NLL.read_text(encoding="utf-8"),
        )
        self.assertIn(
            "--refiner-training-execution-patch-sha256",
            SAMPLE.read_text(encoding="utf-8"),
        )
        self.assertIn(
            "refiner checkpoint identity mismatch: execution patch",
            SAMPLING.read_text(encoding="utf-8"),
        )
        self.assertIn(
            "crysllmgen_epoch_selection_evidence_v2",
            EVIDENCE.read_text(encoding="utf-8"),
        )
        self.assertIn(
            "require_separated_execution_identity",
            SELECTION_CORE.read_text(encoding="utf-8"),
        )

    def test_outputs_are_unique_and_failed_outputs_are_preserved(self) -> None:
        outputs = self.batch["outputs"]
        self.assertIn("sup27104_v1", outputs["epoch_panel"])
        self.assertIn("sup27104_v1", outputs["epoch_evidence"])
        self.assertIn("sup27104_v1", outputs["checkpoint_selection_lock"])
        for script in (self.eval, self.select, self.submit):
            self.assertNotIn("epoch_checkpoint_panel_sup26955_v2", script)
            self.assertNotIn(
                "epoch_checkpoint_selection_lock_sup26955_v2",
                script,
            )
        self.assertIn("scientific_attempts_started", json.dumps(self.batch))
        self.assertFalse(self.batch["scientific_protocol_changed"])


if __name__ == "__main__":
    unittest.main()
