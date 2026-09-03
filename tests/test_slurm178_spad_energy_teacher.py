from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).resolve().parents[1] / "slurm/178_spad_energy_teacher.sbatch"
).read_text(encoding="utf-8")


class SPADEnergyTeacherSlurmTest(unittest.TestCase):
    def test_bounded_k4_terminal_teacher(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4", SOURCE)
        self.assertIn("--num-samples 2048", SOURCE)
        self.assertIn("sample_spad_energy_backfill_actions.py", SOURCE)
        self.assertIn("--diff-steps 800", SOURCE)
        self.assertIn("--max-proposals 8192", SOURCE)
        self.assertIn("label_spad_energy_actions.py", SOURCE)

    def test_no_test_query_or_selection(self):
        self.assertNotIn("query_official", SOURCE)
        self.assertNotIn("--rerank", SOURCE)
        self.assertNotIn("--best-of", SOURCE)
        self.assertIn('"mandatory_noop": True', SOURCE)
        self.assertIn('"validity_is_lexicographic": True', SOURCE)


if __name__ == "__main__":
    unittest.main()
