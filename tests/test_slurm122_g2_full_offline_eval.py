from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "slurm" / "122_g2_full_offline_eval.sbatch"


class G2FullOfflineSlurmTest(unittest.TestCase):
    def test_reuses_raw_and_evaluates_both_chgnet_endpoints(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn("source_raw_direct_job\t39114", text)
        self.assertIn("source_model494_job\t39115", text)
        self.assertIn("raw_generation_reused\ttrue", text)
        self.assertIn("raw_direct_reused\ttrue", text)
        self.assertIn("chgnet\traw_and_refined", text)
        self.assertIn("raw/full_reconstructed", text)
        self.assertIn("refined/full_reconstructed", text)

    def test_fixed_denominator_and_no_official_query(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn("H1_ACTIVE_DENOMINATOR=256", text)
        self.assertIn("--denominator 256", text)
        self.assertIn("missing_policy\tpreserve_sample_idx", text)
        self.assertIn("official_query\tfalse", text)
        self.assertNotIn("mp-api", text.lower())


if __name__ == "__main__":
    unittest.main()
