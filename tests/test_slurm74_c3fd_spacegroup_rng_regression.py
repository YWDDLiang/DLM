from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "74_c3fd_spacegroup_rng_regression.sbatch"


class C3FDSpacegroupRNGSlurmTest(unittest.TestCase):
    def test_job_is_zero_gpu_and_hash_frozen(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=16", text)
        self.assertNotIn("#SBATCH --gres", text)
        self.assertIn("gpu_jobs_used\\t0", text)
        self.assertIn("readonly CONTRACT_SHA=", text)
        for value in (
            "SEED17_SHA",
            "SEED18_SHA",
            "VOCAB_SHA",
            "SAMPLER_SHA",
            "COMPARATOR_SHA",
            "LEGACY17_PLANS_SHA",
            "LEGACY18_PLANS_SHA",
        ):
            self.assertIn(value, text)

    def test_job_uses_frozen_legacy_and_corrected_head_mode(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("--spacegroup-mode independent_head", text)
        self.assertIn("--num-samples 32", text)
        self.assertIn("--temperature 0.9", text)
        self.assertIn("--reachability-mode pauling_bitset", text)
        self.assertIn('--pair "seed17=${LEGACY17},${RUN}/corrected_seed17"', text)
        self.assertIn('--pair "seed18=${LEGACY18},${RUN}/corrected_seed18"', text)


if __name__ == "__main__":
    unittest.main()
