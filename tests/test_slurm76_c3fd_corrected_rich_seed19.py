from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "76_c3fd_corrected_rich_seed19.sbatch"


class C3FDCorrectedRichSeed19SlurmTest(unittest.TestCase):
    def test_contract_is_cpu_only_and_outcome_blind(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=16", text)
        self.assertNotIn("#SBATCH --gres", text)
        self.assertIn("gpu_jobs_used\\t0", text)
        self.assertIn("outcomes_read\\tfalse", text)
        self.assertIn("readonly CONTRACT_SHA=", text)

    def test_sampling_is_frozen_to_new_seed_and_corrected_sg(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("planner_checkpoint_seed\\t17", text)
        self.assertIn("sampling_seed\\t19", text)
        self.assertIn("--num-samples 1000", text)
        self.assertIn("--seed 19", text)
        self.assertIn("--temperature 0.9", text)
        self.assertIn("--reachability-mode pauling_bitset", text)
        self.assertIn("--spacegroup-mode independent_head", text)


if __name__ == "__main__":
    unittest.main()
