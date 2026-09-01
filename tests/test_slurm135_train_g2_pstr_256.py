from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Slurm135Test(unittest.TestCase):
    def test_single_frozen_residual_only_tail_risk_contract(self) -> None:
        text = (ROOT / "slurm/135_train_g2_pstr_256.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2", text)
        self.assertIn("optimizer_steps\t256", text)
        self.assertIn("trainable\tperiodic_relation_residual_only", text)
        self.assertIn("--max-train-steps 256", text)
        self.assertIn("--no-lora", text)
        self.assertIn("--lr 1e-6", text)
        self.assertIn("--periodic-overlap-tail-temperature 0.10", text)
        self.assertIn("--periodic-overlap-tail-mix 0.50", text)
        self.assertIn("--periodic-image-radius 2", text)
        self.assertNotIn("--periodic-relation-uncertainty-gate", text)
        self.assertNotIn("nvidia-smi", text)


if __name__ == "__main__":
    unittest.main()
