from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PotentialClosureNativeWrapperTest(unittest.TestCase):
    def test_stream17_wrapper_uses_frozen_native_schedule_only(self):
        text = (ROOT / "slurm" / "187_potential_closure_native_stream17.sbatch").read_text()
        self.assertIn("gpu:NVIDIAA800-SXM4-80GB:2", text)
        self.assertIn("--cpus-per-task=8", text)
        self.assertIn("--spad-cell-closure", text)
        self.assertIn("--spad-backfill", text)
        self.assertIn("--spad-max-anchor-revisions 2", text)
        self.assertIn("--temperature 0.7", text)
        self.assertIn("--seed \"${DLM_SEED}\"", text)
        self.assertNotIn("refine_dlm_with_crysllmgen.py", text)
        self.assertNotIn("CHGNet", text)


if __name__ == "__main__":
    unittest.main()
