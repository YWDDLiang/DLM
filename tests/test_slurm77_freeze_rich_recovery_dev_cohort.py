from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "77_freeze_rich_recovery_dev_cohort.sbatch"


class FreezeRichRecoveryDevCohortSlurmTest(unittest.TestCase):
    def test_job_is_zero_gpu_and_all_inputs_are_hashed(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=8", text)
        self.assertNotIn("#SBATCH --gres", text)
        self.assertIn("gpu_jobs_used\\t0", text)
        for value in (
            "SOURCE_SHA",
            "MPTRAIN_SHA",
            "C3S17_SHA",
            "C3S18_SHA",
            "D3MAIN_SHA",
            "D3SEALED_SHA",
            "OLDH1_SHA",
            "BUILDER_SHA",
        ):
            self.assertIn(value, text)

    def test_selection_and_views_are_frozen(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("--count 256", text)
        self.assertIn("--rcf-shift 73", text)
        self.assertIn("for view in ledger M0 R0 RCF", text)
        self.assertIn('"${OUTPUT}/${view}.jsonl"', text)
        self.assertIn("outcomes_read\\tfalse", text)


if __name__ == "__main__":
    unittest.main()
