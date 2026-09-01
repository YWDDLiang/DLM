from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Slurm134Test(unittest.TestCase):
    def test_single_immutable_1200_sampling_contract(self) -> None:
        text = (ROOT / "slurm/134_sample_fused_planner_1200.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", text)
        self.assertIn("readonly REQUESTED=1200", text)
        self.assertIn("readonly SEED=23", text)
        self.assertIn('--expected-requested "${REQUESTED}"', text)
        self.assertIn("trajectory_per_ordinal\t1", text)
        self.assertIn("outcomes_read\tfalse", text)
        self.assertIn("retry_filter_replacement_rerank_best_of_n\tfalse", text)
        self.assertNotIn("nvidia-smi", text)
        self.assertNotIn("--retry", text)


if __name__ == "__main__":
    unittest.main()
