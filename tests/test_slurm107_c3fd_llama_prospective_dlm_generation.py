from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "slurm/107_c3fd_llama_prospective_dlm_generation.sbatch").read_text(
    encoding="utf-8"
)


class Slurm107StaticTest(unittest.TestCase):
    def test_four_route_stream_cells_use_fixed_dlm(self):
        for spec in ("0 F 17", "1 M 17", "2 F 18", "3 M 18"):
            self.assertIn(f"run_cell {spec}", SOURCE)
        self.assertIn("old_H1A2_rich_DLM", SOURCE)
        self.assertIn("model494_tau800", SOURCE)

    def test_missing_plans_are_retained_without_top_up(self):
        self.assertIn("--allow-missing-plans", SOURCE)
        self.assertIn("planner_failures\tpreserved_without_DLM_call", SOURCE)
        self.assertIn("retry_replacement_rerank\tfalse", SOURCE)


if __name__ == "__main__":
    unittest.main()
