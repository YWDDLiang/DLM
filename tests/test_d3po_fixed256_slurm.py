from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class D3POFixed256SlurmTest(unittest.TestCase):
    def test_generation_is_exactly_six_fixed_cells(self):
        text = (ROOT / "slurm/66_d3po_fixed256_generation.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --cpus-per-task=48", text)
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6", text)
        self.assertEqual(len(re.findall(r"^run_cell [0-5] ", text, re.MULTILINE)), 6)
        self.assertIn("--num-samples 256", text)
        self.assertIn("--diff-steps 800", text)
        self.assertNotIn("--num-samples 1000", text)
        self.assertIn("rerank_or_replacement\tfalse", text)
        self.assertIn("checkpoint_or_seed_selection\tfalse", text)
        self.assertIn(
            "21a20c8eca10c30953f486ee00301a872e3c32b853bb0acbe187be2f9d94d3f5",
            text,
        )
        self.assertIn("trap 'on_error", text)
        self.assertIn("ENGINEERING_FAILURE.tsv", text)
        self.assertIn("OUTPUTS.sha256", text)

    def test_eval_runs_refined_and_raw_six_cell_passes(self):
        text = (ROOT / "slurm/67_d3po_fixed256_eval.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --cpus-per-task=48", text)
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6", text)
        self.assertIn('run_six ""', text)
        self.assertIn('run_six "raw_"', text)
        self.assertIn("H1_ACTIVE_DENOMINATOR=256", text)
        self.assertIn("_OFFLINE_SUCCESS", text)
        self.assertIn("trap 'on_error", text)
        self.assertIn("ENGINEERING_FAILURE.tsv", text)
        self.assertIn("OUTPUTS.sha256", text)

    def test_common_rng_streams_are_frozen(self):
        text = (ROOT / "slurm/66_d3po_fixed256_generation.sbatch").read_text(
            encoding="utf-8"
        )
        for value in ("91117", "92117", "101117", "102117"):
            self.assertIn(value, text)
        self.assertEqual(text.count('d3po_seed81017 "${SEED_A}"'), 2)
        self.assertEqual(text.count('d3po_seed81018 "${SEED_B}"'), 2)


if __name__ == "__main__":
    unittest.main()
