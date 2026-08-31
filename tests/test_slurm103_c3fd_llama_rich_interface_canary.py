from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "slurm/103_c3fd_llama_rich_interface_canary.sbatch").read_text(
    encoding="utf-8"
)


class Slurm103StaticTest(unittest.TestCase):
    def test_four_cells_share_seed_and_fixed_rows(self):
        for spec in ("F 84017", "F 84018", "M 84117", "M 84118"):
            self.assertIn(f"sample_one {spec}", SOURCE)
        self.assertIn("readonly SAMPLE_SEED=86017", SOURCE)
        self.assertIn("readonly REQUESTED=64", SOURCE)

    def test_canary_cannot_select_or_retry(self):
        self.assertIn("setting_selection\tfalse", SOURCE)
        self.assertIn("retry_replacement_rerank\tfalse", SOURCE)
        self.assertIn("formula_changed", SOURCE)
        self.assertIn("EXPANDER_SHA", SOURCE)
        self.assertEqual(SOURCE.count("readonly EXPANDER_SHA="), 1)


if __name__ == "__main__":
    unittest.main()
