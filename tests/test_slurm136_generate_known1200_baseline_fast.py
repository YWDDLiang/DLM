from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Slurm136Test(unittest.TestCase):
    def test_six_gpu_fast_only_known_plan_contract(self) -> None:
        text = (ROOT / "slurm/136_generate_known1200_baseline_fast.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6", text)
        self.assertEqual(text.count('run_shard "${index}"'), 1)
        self.assertIn("for index in 0 1 2 3 4", text)
        self.assertIn("run_shard 5 remainder", text)
        self.assertIn('local device="${gpu[$slot]}" out="${RUN}/${name}"', text)
        self.assertIn("--expected-denominator \"${count}\"", text)
        self.assertIn("model494_or_CHGNet_or_NU\tfalse", text)
        self.assertIn("expensive_direct_metrics_run':False", text)
        self.assertNotIn("nvidia-smi", text)
        self.assertNotIn("--retry", text)


if __name__ == "__main__":
    unittest.main()
