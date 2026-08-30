from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "78_freeze_rich_recovery_dev_cohort_v2.sbatch"


class FreezeRichRecoveryDevCohortV2SlurmTest(unittest.TestCase):
    def test_recovery_changes_only_certificate_adapter(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("h1a2_rich_recovery_dev_cohort_v2", text)
        self.assertIn("minimal_prompt\\td3po_c3fd_certificate_adapter", text)
        self.assertIn("--count 256", text)
        self.assertIn("--rcf-shift 73", text)
        self.assertIn("BUILDER_SHA=1c35e95e", text)
        self.assertIn("outcomes_read\\tfalse", text)
        self.assertNotIn("#SBATCH --gres", text)


if __name__ == "__main__":
    unittest.main()
