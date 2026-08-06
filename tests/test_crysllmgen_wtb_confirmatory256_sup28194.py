from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (
    ROOT
    / "scripts"
    / "a800"
    / "wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1"
)
PIPELINE = WRAPPER / "pipeline.sbatch"
SUBMIT = WRAPPER / "submit_once.sh"
AUTHORIZATION = (
    ROOT
    / "diagnostics"
    / "authorization_records"
    / "wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1.json"
)
PLAN = (
    ROOT
    / "docs"
    / "experiment_program"
    / "20260727_wtb256_job28194_audit_sidecar_supersession_plan.md"
)


class WTBConfirmatory256Sup28194Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = PIPELINE.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")
        cls.authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))

    def test_execution_identity_is_new_but_scientific_identity_is_frozen(
        self,
    ) -> None:
        self.assertEqual(
            self.authorization["execution_identity"],
            "wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1",
        )
        self.assertEqual(
            self.authorization["scientific_identity"],
            "wq_wyckoff_chart_retraction_confirmatory256_v1",
        )
        self.assertEqual(self.authorization["supersedes_failed_job_id"], 28194)
        self.assertIn(
            "CONTRACT=configs/experiments/wyckoff_codiffusion/"
            "wq_wyckoff_chart_retraction_confirmatory256_v1.json",
            self.pipeline,
        )
        self.assertIn(
            "CONTRACT_SHA256="
            "293c026d2f371b592a81e8e4d3982b4cb65ae3b0d90b82bf72a639caae24b77a",
            self.pipeline,
        )
        self.assertIn(
            "SCIENTIFIC_IDENTITY="
            "wq_wyckoff_chart_retraction_confirmatory256_v1",
            self.submit,
        )

    def test_pipeline_is_one_a800_eight_cpu_and_evaluation_only(self) -> None:
        self.assertRegex(
            self.pipeline,
            r"(?m)^#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1$",
        )
        self.assertRegex(
            self.pipeline,
            r"(?m)^#SBATCH --cpus-per-task=8$",
        )
        self.assertIn("#SBATCH --partition=gpu", self.pipeline)
        self.assertIn("#SBATCH --mem=96G", self.pipeline)
        self.assertIn("#SBATCH --time=18:00:00", self.pipeline)
        self.assertNotIn("#SBATCH --array", self.pipeline)
        lowered = self.pipeline.lower()
        for forbidden in (
            "optimizer.step",
            ".backward(",
            "trainer.train",
            "run_training",
            "scancel",
        ):
            self.assertNotIn(forbidden, lowered)
        self.assertNotIn("\nsbatch ", lowered)
        self.assertIn('"training_performed": False', self.pipeline)
        self.assertIn('"automatic_followup_submission": False', self.pipeline)

    def test_preclaim_gate_load_precedes_claim_and_sbatch(self) -> None:
        gate_position = self.submit.index("GateALock.load(")
        claim_position = self.submit.index('with Path(sys.argv[1]).open("x"')
        sbatch_position = self.submit.index("sbatch --parsable")
        self.assertLess(gate_position, claim_position)
        self.assertLess(claim_position, sbatch_position)
        self.assertIn(
            "set(ALLOWED_AUTHORIZATIONS) != "
            "set(PATCH_ALLOWED_AUTHORIZATIONS)",
            self.submit,
        )
        self.assertIn(
            "wtb256_sup28194_preclaim_gate_a_lock=PASS",
            self.submit,
        )

    def test_old_job_evidence_is_hash_locked_and_never_reused(self) -> None:
        for digest in (
            "e660623c3fcdb606667379a1d3659d16defdfda8340f64cce3ebb68cc54399a7",
            "a7870e476ddb2d69dfb9a517bbb6d811c64a4021695ae9cab6c5acf82223caea",
            "5790afb3a6c6d372e1429348b95b0d8e350ce663c7d0d5fe4d4ba404fa380ace",
            "8100c391370939d00edc497be6fad497ca576a3a771d400a787148ac3ce2c723",
        ):
            self.assertIn(digest, self.submit)
        self.assertIn("test ! -e \"$OUTPUT\"", self.submit)
        self.assertNotIn("rm ", self.submit)
        self.assertNotIn("mv ", self.submit)
        self.assertNotIn("scancel", self.submit)
        self.assertEqual(
            len(re.findall(r"\bsbatch --parsable\b", self.submit)),
            2,
        )

    def test_authorization_is_one_shot_and_excludes_training(self) -> None:
        self.assertEqual(self.authorization["user_quote"], "同意")
        self.assertFalse(
            self.authorization["scientific_scope_unchanged"]["training"]
        )
        self.assertFalse(
            self.authorization["scientific_scope_unchanged"][
                "retry_or_replacement_allowed"
            ]
        )
        forbidden = " ".join(self.authorization["not_authorized"]).lower()
        self.assertIn("training", forbidden)
        self.assertIn("retry", forbidden)
        self.assertIn("job28194", forbidden)
        self.assertTrue(PLAN.is_file())


if __name__ == "__main__":
    unittest.main()
