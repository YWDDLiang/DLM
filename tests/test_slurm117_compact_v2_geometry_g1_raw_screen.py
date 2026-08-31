from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "slurm" / "117_compact_v2_geometry_g1_raw_screen.sbatch"


class G1RawScreenSlurmTest(unittest.TestCase):
    def test_reuses_frozen_plan_noise_and_base_without_model494(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn("c3fd_llama_typed_plans_39088", text)
        self.assertIn("periodic_geometry_g1_39103", text)
        self.assertIn("dlm_seed\t91117", text)
        self.assertIn("--temperature 0.7 --seed 91117", text)
        self.assertIn("BASE_REPORT_SHA=b5881e29", text)
        self.assertIn("model494\tfalse", text)
        self.assertIn("chgnet\tfalse", text)
        self.assertIn("official_query\tfalse", text)
        self.assertNotIn("refiner", text.lower())

    def test_gate_is_requested_denominator_and_strictly_positive_direct(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn('result["comp_valid"] >= 244', text)
        self.assertIn('result["body_parsed"] >= 246', text)
        self.assertIn('result["direct_joint"] > 106', text)
        self.assertIn("--denominator 256", text)
        self.assertIn("retry_filter_replacement_rerank_best_of_n\tfalse", text)


if __name__ == "__main__":
    unittest.main()
