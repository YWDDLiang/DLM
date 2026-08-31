from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "slurm" / "120_recover_g2_raw_direct.sbatch"


class G2RawRecoverySlurmTest(unittest.TestCase):
    def test_reuses_frozen_body_and_metric_contract(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn("source_body_job\t39112", text)
        self.assertIn("body_rerun\tfalse", text)
        self.assertIn("science_parameters_changed\tfalse", text)
        self.assertIn("OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8", text)
        self.assertIn("CRYSLLMGEN_METRICS_NUM_CPUS=1", text)
        self.assertNotIn("sample_sgtc", text)

    def test_full_endpoint_remains_required(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn("full_model494_CHGNet_required\ttrue", text)
        self.assertIn('"full_model494_CHGNet_required": True', text)
        self.assertIn("official_query\tfalse", text)
        self.assertIn('report["denominator"] == "all_generation_attempts"', text)


if __name__ == "__main__":
    unittest.main()
