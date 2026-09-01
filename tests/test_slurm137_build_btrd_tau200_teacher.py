from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Slurm137Test(unittest.TestCase):
    def test_six_gpu_tau200_teacher_and_no_energy_selection(self) -> None:
        text = (ROOT / "slurm/137_build_btrd_tau200_teacher.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6", text)
        self.assertIn("tau200_rows\t6144", text)
        self.assertIn("anchor_rows\t2048", text)
        self.assertIn("--shard-size 1024", text)
        self.assertIn("--nproc_per_node=6", text)
        self.assertIn("--diff-steps 200", text)
        self.assertIn("energy_label_or_selection\tfalse", text)
        self.assertIn("fallback_original_MP20_anchor_no_row_deletion", text)
        self.assertNotIn("CHGNet", text)
        self.assertNotIn("nvidia-smi", text)


if __name__ == "__main__":
    unittest.main()
