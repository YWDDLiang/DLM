from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BATCH = (
    ROOT
    / "configs/experiments/wyckoff_codiffusion/"
    "epoch_eval_selection_supersession27104_v2_dag.json"
)
AUTHORIZATION = (
    ROOT
    / "runs/remote_audit/"
    "20260724_user_authorization_epoch_eval_selection_supersession27104_v2_bash42.json"
)
FAILURE_AUDIT = (
    ROOT
    / "runs/remote_audit/"
    "20260723_epoch_eval_selection_supersession27104_v1_presbatch_failure_v1.json"
)
V1_SUBMIT = (
    ROOT
    / "scripts/a800/epoch_eval_selection_supersession27104_v1/submit_once.sh"
)
JOB_DIR = ROOT / "scripts/a800/epoch_eval_selection_supersession27104_v2"
EVAL = JOB_DIR / "epoch_eval.sbatch"
SELECT = JOB_DIR / "select_epoch.sbatch"
SUBMIT = JOB_DIR / "submit_once.sh"
INSTALLER = ROOT / "scripts/a800/install_authorized_patch.py"
GATE = ROOT / "crystal_dlm/wqcodiff/crysllmgen/gate.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EpochEvalSelectionBash42SupersessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.batch = json.loads(BATCH.read_text(encoding="utf-8"))
        cls.eval = EVAL.read_text(encoding="utf-8")
        cls.select = SELECT.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")

    def test_authorization_and_presbatch_failure_are_exact(self) -> None:
        self.assertEqual(
            self.batch["authorization"]["sha256"],
            _sha256(AUTHORIZATION),
        )
        failure = self.batch["failure_being_superseded"]
        self.assertEqual(failure["audit_sha256"], _sha256(FAILURE_AUDIT))
        self.assertEqual(
            failure["entrypoint_identity"],
            "epoch_eval_selection_supersession27104_v1",
        )
        self.assertFalse(failure["prior_claim_created"])
        self.assertFalse(failure["prior_sbatch_invoked"])
        self.assertEqual(failure["scientific_attempts_started"], 0)
        authorization_name = (
            "user_epoch_eval_selection_supersession27104_v2_bash42_2026-07-24"
        )
        self.assertIn(authorization_name, INSTALLER.read_text(encoding="utf-8"))
        self.assertIn(authorization_name, GATE.read_text(encoding="utf-8"))

    def test_v1_entrypoint_is_immutable_and_v2_paths_are_unique(self) -> None:
        self.assertEqual(
            _sha256(V1_SUBMIT),
            "a4a156d62e3d80ef7d0dc1d87dfe59f618f7feb805529dfaa25f87008d364ff5",
        )
        for value in self.batch["outputs"].values():
            self.assertIn("v2", value)
            self.assertNotIn("sup27104_v1", value)
        self.assertIn("PRIOR_RECORD=", self.submit)
        self.assertIn("test ! -e \"$PRIOR_RECORD\"", self.submit)
        self.assertNotIn("epoch_checkpoint_panel_sup27104_v1", self.eval)
        self.assertNotIn("epoch_checkpoint_panel_sup27104_v1", self.select)

    def test_bash42_queue_preflight_avoids_empty_arrays(self) -> None:
        self.assertIn("set -Eeuo pipefail", self.submit)
        self.assertNotIn("mapfile", self.submit)
        self.assertNotIn("readarray", self.submit)
        self.assertNotIn("existing_rows[@]", self.submit)
        self.assertNotIn("gpu_rows[@]", self.submit)
        self.assertNotIn("gpu_rows=(", self.submit)
        begin = self.submit.index("# BASH42_QUEUE_PREFLIGHT_BEGIN")
        end = self.submit.index("# BASH42_QUEUE_PREFLIGHT_END")
        block = self.submit[begin:end].split("\n", 1)[1]

        with tempfile.TemporaryDirectory() as tmp:
            bindir = Path(tmp)
            fake_squeue = bindir / "squeue"
            fake_squeue.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s' \"${FAKE_SQUEUE_ROWS:-}\"\n",
                encoding="utf-8",
            )
            fake_squeue.chmod(0o755)
            harness = bindir / "harness.sh"
            harness.write_text(
                "#!/usr/bin/env bash\n"
                "set -Eeuo pipefail\n"
                f"{block}\n"
                "printf 'existing=<%s> gpu=<%s>\\n' "
                "\"$existing_rows\" \"$gpu_rows\"\n",
                encoding="utf-8",
            )
            harness.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{bindir}:{env['PATH']}"
            env["USER"] = "queue-test"
            env.pop("FAKE_SQUEUE_ROWS", None)
            empty = subprocess.run(
                ["/bin/bash", str(harness)],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(empty.returncode, 0, empty.stderr)
            self.assertIn("existing=<> gpu=<>", empty.stdout)

            env["FAKE_SQUEUE_ROWS"] = "91|short|(null)|cpu-job"
            cpu_only = subprocess.run(
                ["/bin/bash", str(harness)],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(cpu_only.returncode, 0, cpu_only.stderr)
            self.assertIn("gpu=<>", cpu_only.stdout)

            env["FAKE_SQUEUE_ROWS"] = (
                "92|gpu|gpu:NVIDIAA800-SXM4-80GB:1|gpu-job"
            )
            gpu = subprocess.run(
                ["/bin/bash", str(harness)],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(gpu.returncode, 3)
            self.assertIn(
                "preflight waits for zero preexisting user GPU jobs",
                gpu.stderr,
            )

    def test_claim_is_exclusive_and_precedes_both_sbatch_calls(self) -> None:
        self.assertIn('with Path(path).open("x"', self.submit)
        claim_schema = self.submit.index(
            '"crysllmgen_epoch_eval_selection_submission_claim_v2"'
        )
        claim = self.submit.index('with Path(path).open("x"', claim_schema)
        first_submit = self.submit.index(
            "submit_or_record_failure eval_job eval_command"
        )
        second_submit = self.submit.index(
            "submit_or_record_failure selection_job selection_command"
        )
        self.assertLess(claim, first_submit)
        self.assertLess(first_submit, second_submit)
        self.assertEqual(
            self.submit.count(
                "submit_or_record_failure eval_job eval_command"
            ),
            1,
        )
        self.assertEqual(
            self.submit.count(
                "submit_or_record_failure selection_job selection_command"
            ),
            1,
        )
        self.assertIn('--dependency="afterok:$eval_job"', self.submit)

    def test_frozen_job_hashes_and_scientific_contract(self) -> None:
        self.assertEqual(
            self.batch["schema"],
            "crysllmgen_epoch_eval_selection_supersession_dag_v2",
        )
        for job in self.batch["jobs"]:
            path = ROOT / job["script"]
            self.assertEqual(job["script_sha256"], _sha256(path), path)
        accounting = self.batch["submission_accounting"]
        self.assertEqual(
            accounting["slurm_submit_slots_including_array_elements"],
            4,
        )
        self.assertEqual(accounting["maximum_concurrent_a800"], 2)
        self.assertFalse(self.batch["scientific_protocol_changed"])
        self.assertIn("--array=0-2%2", self.eval)
        self.assertIn("--attempts 256", self.eval)
        self.assertIn("R5-C A100 protocol on A800", self.eval)
        self.assertIn("paired bootstrap 10000", json.dumps(self.batch))
        self.assertFalse(self.batch["selection_policy"]["dft"])
        self.assertNotIn("formal_train.sbatch", self.submit)
        self.assertNotIn("refiner_candidate.sbatch", self.submit)


if __name__ == "__main__":
    unittest.main()
