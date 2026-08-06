import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiments/wyckoff_codiffusion/"
    "wq_parent_csp_probe_sup27407_v1.json"
)
SBATCH = (
    ROOT
    / "scripts/a800/wq_parent_csp_probe_sup27407_v1/probe.sbatch"
)
SUBMIT = (
    ROOT
    / "scripts/a800/wq_parent_csp_probe_sup27407_v1/submit_once.sh"
)
PROBE = ROOT / "crystal_dlm/wqcodiff/crysllmgen/wq_parent_probe.py"


class WQParentCSPProbeSup27407Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = json.loads(PLAN.read_text(encoding="utf-8"))
        self.sbatch = SBATCH.read_text(encoding="utf-8")
        self.submit = SUBMIT.read_text(encoding="utf-8")

    def test_supersession_is_new_and_scientifically_bounded(self) -> None:
        self.assertEqual(self.plan["supersession"]["job_id"], "27407")
        self.assertEqual(
            self.plan["supersession"]["scientific_attempts_started"],
            0,
        )
        self.assertEqual(self.plan["resources"]["a800"], 1)
        self.assertEqual(self.plan["resources"]["time_limit_minutes"], 45)
        self.assertEqual(
            self.plan["scientific_contract"]["ordinals"],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            self.plan["scientific_contract"]["pairing_id"],
            "wq-epoch-selection-paired-noise-sup27104-v2",
        )

    def test_submission_paths_do_not_reuse_job_27407_identity(self) -> None:
        identity = self.plan["immutable_identity"]
        self.assertIn("sup27407", identity["experiment_id"])
        self.assertIn("sup27407", identity["slurm_job_name"])
        self.assertIn("sup27407", identity["submission_record"])
        self.assertIn("sup27407", identity["output"])
        self.assertIn("#SBATCH --job-name=wq-parent-csp-sup27407-v1", self.sbatch)
        self.assertIn(
            "wq_parent_csp_same_proposal_sup27407_v1",
            self.sbatch,
        )
        self.assertIn(
            "wq_parent_csp_same_proposal_sup27407_v1",
            self.submit,
        )
        self.assertIn("supersedes_job_id", self.submit)

    def test_only_authorized_gate_identity_fix_is_present(self) -> None:
        source = PROBE.read_text(encoding="utf-8")
        self.assertIn(
            "execution_patch_manifest_sha256=(\n"
            "            config.diagnostic_execution_patch_sha256\n"
            "        )",
            source,
        )
        self.assertIn("--attempts 4", self.sbatch)
        self.assertNotIn("bridge_train", self.sbatch)
        self.assertIn('"long_training_submitted": False', self.submit)


if __name__ == "__main__":
    unittest.main()
