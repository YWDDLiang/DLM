from pathlib import Path
import unittest


class PMTRCertificationSlurmTest(unittest.TestCase):
    def test_resource_and_scientific_contract(self):
        path = Path(__file__).resolve().parents[1] / "slurm" / "220_pmtr_preflight_certify.sbatch"
        text = path.read_text(encoding="utf-8")
        self.assertIn("NVIDIAA800-SXM4-80GB:4", text)
        self.assertIn("--cpus-per-task=16", text)
        self.assertIn("--max-proposals 4", text)
        self.assertIn("--batch-size 16", text)
        self.assertIn("--probe-fraction 0.10", text)
        self.assertIn("len(rows) != 2048", text)
        self.assertNotIn("nvidia-smi", text)
        self.assertNotIn("MP_API", text)


if __name__ == "__main__":
    unittest.main()
