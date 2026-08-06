from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiments/wyckoff_codiffusion/"
    "wq_parent_csp_sun256_eval_sup27410_v1.json"
)
SBATCH = (
    ROOT
    / "scripts/a800/wq_parent_csp_sun256_eval_sup27410_v1/evaluate.sbatch"
)
SUBMIT = (
    ROOT
    / "scripts/a800/wq_parent_csp_sun256_eval_sup27410_v1/submit_once.sh"
)
SUN_RUNNER = ROOT / "scripts/a800/run_crysllmgen_a100_sun.py"


class ParentCSPSUN256EvaluationSupersessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = json.loads(PLAN.read_text(encoding="utf-8"))
        self.sbatch = SBATCH.read_text(encoding="utf-8")
        self.submit = SUBMIT.read_text(encoding="utf-8")
        self.sun_runner = SUN_RUNNER.read_text(encoding="utf-8")

    def test_reuses_exact_generation_without_regeneration(self) -> None:
        reuse = self.plan["scientific_reuse"]
        self.assertFalse(reuse["regenerate"])
        self.assertEqual(reuse["attempts"], 256)
        self.assertEqual(reuse["ordinals"], [256, 511])
        self.assertEqual(
            reuse["source_generation_sha256"],
            "b6eb7f80a29da699407d8d19bbedeb2d657f5d7940cd767d6d71aecb6c58a598",
        )
        self.assertNotIn("probe_wq_proposal_with_parent_csp.py", self.sbatch)

    def test_all_evaluation_is_locked_to_diff_meets_diff(self) -> None:
        environment = self.plan["evaluation_environment"]
        self.assertTrue(environment["single_environment_for_all_evaluation_steps"])
        self.assertEqual(environment["conda_environment"], "diff_meets_diff")
        self.assertEqual(
            self.sbatch.count("conda activate diff_meets_diff"), 1
        )
        self.assertNotIn("conda activate crysllm", self.sbatch)
        self.assertIn('REQUIRED_EVALUATION_ENV = "diff_meets_diff"', self.sun_runner)
        self.assertNotIn(
            'os.environ.get("CONDA_DEFAULT_ENV") != "crysllm"',
            self.sun_runner,
        )

    def test_environment_smoke_precedes_metrics_and_sun(self) -> None:
        smoke = self.sbatch.index("diff_meets_diff_environment_smoke=PASS")
        metrics = self.sbatch.index("run_crysllmgen_metrics.py")
        sun = self.sbatch.index("run_crysllmgen_a100_sun.py")
        self.assertLess(smoke, metrics)
        self.assertLess(metrics, sun)
        self.assertIn('md.version("torch-scatter")', self.sbatch)
        self.assertIn("from compute_metrics import Crystal, GenEval", self.sbatch)

    def test_cpu_per_a800_policy_is_fail_closed_before_claim(self) -> None:
        self.assertIn("#SBATCH --cpus-per-task=8", self.sbatch)
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", self.sbatch)
        policy = self.submit.index("job_cpus > 8 * job_gpus")
        claim = self.submit.index('path.open("x"')
        sbatch = self.submit.index("sbatch --parsable")
        self.assertLess(policy, claim)
        self.assertLess(claim, sbatch)

    def test_submit_is_unique_and_does_not_touch_other_jobs(self) -> None:
        self.assertIn("test ! -e \"$RECORD\"", self.submit)
        self.assertIn("test ! -e \"$CLAIM\"", self.submit)
        self.assertIn("test ! -e \"$OUTPUT\"", self.submit)
        self.assertNotIn("scancel", self.submit)
        self.assertIn("submission_failed_no_retry", self.submit)


if __name__ == "__main__":
    unittest.main()
