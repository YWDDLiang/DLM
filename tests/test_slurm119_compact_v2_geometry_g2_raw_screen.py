from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "slurm" / "119_compact_v2_geometry_g2_raw_screen.sbatch"


class G2RawScreenSlurmTest(unittest.TestCase):
    def test_same_frozen_cell_with_relation_checkpoint(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn("c3fd_llama_typed_plans_39088", text)
        self.assertIn("periodic_relation_g2_39107", text)
        self.assertIn("--periodic-relation-checkpoint \"${POLICY}\"", text)
        self.assertIn("--periodic-relation-rank 64", text)
        self.assertIn("--temperature 0.7 --seed 91117", text)
        self.assertIn("BASE_REPORT_SHA=b5881e29", text)
        self.assertIn("--denominator 256", text)

    def test_full_endpoint_is_mandatory_but_not_mixed_into_raw_job(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn("full_endpoint_required\ttrue_regardless_raw_direction", text)
        self.assertIn('"full_model494_CHGNet_required": True', text)
        self.assertIn("model494\tfalse_in_this_raw_stage_but_mandatory_next", text)
        self.assertIn("official_query\tfalse", text)
        self.assertNotIn("refine_dlm_with_crysllmgen.py", text)


if __name__ == "__main__":
    unittest.main()
