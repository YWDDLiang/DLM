from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "slurm/109_build_c3fd_llama_fused_data.sbatch").read_text(encoding="utf-8")


class Slurm109Test(unittest.TestCase):
    def test_cpu_immutable_builder_and_pinned_inputs(self):
        self.assertIn("#SBATCH --cpus-per-task=16", SOURCE)
        self.assertNotIn("#SBATCH --gres=gpu", SOURCE)
        self.assertIn("c3fd_semantic_v21_step1b_20260828", SOURCE)
        self.assertIn("ctv_minimal_spec_v4_20260828", SOURCE)
        self.assertIn("C3FD_LLAMA_CTV_TRAIN_SHA256", SOURCE)
        self.assertIn("--semantic-dir", SOURCE)
        self.assertIn("--ctv-minimal-dir", SOURCE)
        self.assertIn("[[ ! -e \"${OUTPUT}\" ]]", SOURCE)
        self.assertIn("dev_test_prospective_rows", SOURCE)


if __name__ == "__main__":
    unittest.main()

