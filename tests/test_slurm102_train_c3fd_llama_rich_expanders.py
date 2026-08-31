from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "slurm/102_train_c3fd_llama_rich_expanders.sbatch").read_text(
    encoding="utf-8"
)


class Slurm102StaticTest(unittest.TestCase):
    def test_four_fixed_route_seed_cells_use_four_gpus(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4", SOURCE)
        for spec in ("F 84017", "F 84018", "M 84117", "M 84118"):
            self.assertIn(f"train_one {spec}", SOURCE)

    def test_M_only_uses_projector(self):
        self.assertIn("prefix_args=(--soft-prefix-length 4", SOURCE)
        self.assertIn("if route=='F'", SOURCE)
        self.assertIn("soft_prefix_projector.pt", SOURCE)

    def test_one_epoch_final_only_no_selection(self):
        self.assertIn("source_epochs\t1", SOURCE)
        self.assertIn("eligible_checkpoint\tfinal_only", SOURCE)
        self.assertIn("seed_checkpoint_selection\tnone", SOURCE)
        self.assertNotIn("best", SOURCE.lower())


if __name__ == "__main__":
    unittest.main()
