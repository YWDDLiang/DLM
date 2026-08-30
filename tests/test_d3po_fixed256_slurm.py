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
            "93e223c3174539b4ac1e573bfe679f6bc42bfab1a8eb6f5fc36cbf5a3ccede40",
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

    def test_late_guidance_generation_is_one_frozen_sealed_setting(self):
        text = (ROOT / "slurm/68_d3po_late_guidance_generation.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --cpus-per-task=48", text)
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6", text)
        self.assertEqual(len(re.findall(r"^run_cell [0-5] ", text, re.MULTILINE)), 6)
        self.assertIn("--late-guidance-scale 0.5", text)
        self.assertIn("--late-guidance-remaining-mask-threshold 0.25", text)
        self.assertIn("--reference-checkpoint-path \"${BASE}\"", text)
        self.assertIn("1b7f7111f75ed5b26cb235274f1b8da70393676898732722c3a63a2f61a35ce0", text)
        self.assertIn("LATE_GUIDANCE_BURNED", text)
        self.assertIn("rerank_or_replacement\tfalse", text)
        self.assertIn("trap 'on_error", text)
        self.assertNotIn("guidance_args=()", text)
        self.assertIn("local sample_command=(", text)

    def test_late_guidance_eval_runs_refined_and_raw(self):
        text = (ROOT / "slurm/69_d3po_late_guidance_eval.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn('run_six ""', text)
        self.assertIn('run_six "raw_"', text)
        self.assertIn("H1_ACTIVE_DENOMINATOR=256", text)
        self.assertIn("_OFFLINE_SUCCESS", text)
        self.assertIn("ENGINEERING_FAILURE.tsv", text)


if __name__ == "__main__":
    unittest.main()
