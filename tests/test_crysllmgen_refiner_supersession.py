from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "configs/experiments/wyckoff_codiffusion/refiner_supersession26955_v2_dag.json"
AUTHORIZATION = ROOT / "runs/remote_audit/20260723_user_authorization_refiner_supersession26955_v2.json"
RUNNER = ROOT / "scripts/a800/refiner_supersession26955_v2/refiner_candidate.sbatch"
SUBMIT = ROOT / "scripts/a800/refiner_supersession26955_v2/submit_once.sh"
EVAL = ROOT / "scripts/a800/refiner_supersession26955_v2/epoch_eval.sbatch"
SELECT = ROOT / "scripts/a800/refiner_supersession26955_v2/select_epoch.sbatch"
INSTALLER = ROOT / "scripts/a800/install_authorized_patch.py"
GATE = ROOT / "crystal_dlm/wqcodiff/crysllmgen/gate.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RefinerSupersessionDagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.batch = json.loads(BATCH.read_text(encoding="utf-8"))
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")
        cls.eval = EVAL.read_text(encoding="utf-8")
        cls.select = SELECT.read_text(encoding="utf-8")

    def test_authorization_and_lineage_are_frozen(self) -> None:
        self.assertEqual(
            self.batch["authorization"]["sha256"],
            _sha256(AUTHORIZATION),
        )
        self.assertEqual(
            self.batch["frozen_identity"]["failed_refiner_job_id"],
            "26955",
        )
        self.assertIn("REPLACEMENT_OF_JOB_ID=26955", self.runner)
        self.assertIn("train_wq_refiner_seed11_formal_supersession26955_v2", self.runner)
        self.assertNotIn("train_wq_refiner_seed11_formal_supersession26679_v1", self.runner)
        authorization_name = "user_refiner_supersession26955_v2_2026-07-23"
        self.assertIn(authorization_name, INSTALLER.read_text(encoding="utf-8"))
        self.assertIn(authorization_name, GATE.read_text(encoding="utf-8"))

    def test_prefetch_reduction_and_smoke_gate_are_frozen(self) -> None:
        selection = self.batch["prefetch_selection"]
        self.assertEqual(selection["workers"], 7)
        self.assertEqual(selection["depth"], 14)
        self.assertIn("PREFETCH_WORKERS=7", self.runner)
        self.assertIn("PREFETCH_DEPTH=14", self.runner)
        jobs = {job["key"]: job for job in self.batch["jobs"]}
        self.assertFalse(jobs["prefetch_stability_smoke"]["scientific_attempt"])
        self.assertEqual(jobs["prefetch_stability_smoke"]["updates"], 100)
        self.assertEqual(
            jobs["formal_refiner_supersession26955"]["dependency"],
            "afterok:prefetch_stability_smoke",
        )
        self.assertIn('--dependency="afterok:$smoke_job"', self.submit)

    def test_all_registered_scripts_match_manifest_hashes(self) -> None:
        for job in self.batch["jobs"]:
            path = ROOT / job["script"]
            self.assertEqual(job["script_sha256"], _sha256(path), path)

    def test_submit_accounting_and_dependency_chain_are_bounded(self) -> None:
        accounting = self.batch["submission_accounting"]
        self.assertEqual(accounting["slurm_submit_slots_including_array_elements"], 6)
        self.assertEqual(accounting["maximum_concurrent_a800"], 2)
        self.assertIn('--dependency="afterok:$refiner_job"', self.submit)
        self.assertIn('--dependency="afterok:$eval_job"', self.submit)
        self.assertIn("--array=0-2%2", self.eval)
        self.assertIn("no resubmission", self.submit)

    def test_eval_and_selection_use_unique_overridable_outputs(self) -> None:
        self.assertIn("formal_supersession26955_v2/model_ema_final.pt", self.eval)
        self.assertIn("epoch_checkpoint_panel_sup26955_v2", self.eval)
        self.assertIn("wq-epoch-selection-panel-sup26955-v2", self.eval)
        self.assertIn("wq-epoch-selection-paired-noise-sup26955-v2", self.eval)
        self.assertIn("epoch_checkpoint_evidence_sup26955_v2.json", self.select)
        self.assertIn("epoch_checkpoint_selection_lock_sup26955_v2.json", self.select)
        outputs = self.batch["outputs"]
        self.assertIn(Path(outputs["formal_refiner"]).name, self.submit)
        self.assertIn(Path(outputs["epoch_panel"]).name, self.submit)
        self.assertIn(Path(outputs["checkpoint_selection_lock"]).name, self.submit)


if __name__ == "__main__":
    unittest.main()
