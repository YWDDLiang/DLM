from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "slurm/176_spad_model494_response_followup.sbatch"
).read_text(encoding="utf-8")


class SPADModel494ResponseFollowupSlurmTest(unittest.TestCase):
    def test_fixed_two_stream_response_then_tau800_protocol(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2", SOURCE)
        self.assertIn("apply_model494_response_backfill.py", SOURCE)
        self.assertIn("spad_prospective_generation_s17_39537", SOURCE)
        self.assertIn("spad_prospective_generation_s18_39538", SOURCE)
        self.assertIn("--diff-steps 800", SOURCE)
        self.assertIn("--seed-by-sample-index", SOURCE)
        self.assertIn('"guided_samples"] == 256', SOURCE)
        self.assertIn('"direct_evaluation": "DEFERRED_COST"', SOURCE)

    def test_does_not_resample_or_select_candidates(self):
        self.assertNotIn("sample_spad_prospective_plans", SOURCE)
        self.assertNotIn("--best-of", SOURCE.lower())
        self.assertNotIn("--rerank", SOURCE.lower())
        self.assertNotIn("mp-api", SOURCE.lower())


if __name__ == "__main__":
    unittest.main()
