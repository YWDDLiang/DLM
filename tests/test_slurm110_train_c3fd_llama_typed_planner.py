from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "slurm/110_train_c3fd_llama_typed_planner.sbatch").read_text(encoding="utf-8")


class Slurm110Test(unittest.TestCase):
    def test_one_gpu_one_seed_one_epoch_final_only(self):
        self.assertIn("#SBATCH --cpus-per-task=16", SOURCE)
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", SOURCE)
        self.assertIn("--seed 85017 --epochs 1", SOURCE)
        self.assertIn("--batch-size 2 --grad-accum 8", SOURCE)
        self.assertIn("--lr 2e-5", SOURCE)
        self.assertIn("--warmup-steps 100", SOURCE)
        self.assertIn('eligible_checkpoint\"]==\"final_only', SOURCE)
        self.assertIn("final/llama_adapter/adapter_model.safetensors", SOURCE)
        self.assertIn("final/typed_residual_config.json", SOURCE)
        self.assertIn("final/_SUCCESS", SOURCE)
        self.assertNotIn("best", SOURCE.lower())

    def test_frozen_seed17_c3fd_and_hash_pins(self):
        self.assertIn("train_seed17/checkpoint.pt", SOURCE)
        self.assertIn("C3FD_SHA=87c1673", SOURCE)
        self.assertIn("C3FD_LLAMA_MODEL_TREE_SHA256", SOURCE)
        self.assertIn("C3FD_LLAMA_FUSED_MANIFEST_SHA256", SOURCE)
        self.assertIn("C3FD_LLAMA_TYPED_TRAINER_SHA256", SOURCE)
        self.assertIn('c[\"c3fd_frozen\"] is True', SOURCE)


if __name__ == "__main__":
    unittest.main()
