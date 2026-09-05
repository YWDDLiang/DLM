from pathlib import Path
import unittest


class PMTRTrainingSlurmTest(unittest.TestCase):
    def test_fixed_head_only_preflight_contract(self):
        path = Path(__file__).resolve().parents[1] / "slurm" / "221_pmtr_preflight_train.sbatch"
        text = path.read_text(encoding="utf-8")
        self.assertIn("NVIDIAA800-SXM4-80GB:4", text)
        self.assertIn("--cpus-per-task=16", text)
        self.assertIn("--nproc_per_node=4", text)
        self.assertIn("--epochs 2", text)
        self.assertIn("--limit 384", text)
        self.assertIn("--batch-size 4", text)
        self.assertIn("pmtr_final.pt", text)
        self.assertNotIn("CHGNet", text)
        self.assertNotIn("nvidia-smi", text)


if __name__ == "__main__":
    unittest.main()
