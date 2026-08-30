from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "93_c3fd_native_alignment_pool_generation.sbatch"


class Slurm93C3FDNativeAlignmentPoolGenerationTest(unittest.TestCase):
    def test_two_policy_matched_generation(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2", text)
        self.assertIn("#SBATCH --cpus-per-task=16", text)
        self.assertIn('run_cell 0 82017 "${POLICY17}"', text)
        self.assertIn('run_cell 1 82018 "${POLICY18}"', text)
        self.assertIn("shared_dlm_seed\\t93117", text)
        self.assertIn("shared_refiner_seed\\t103117", text)
        self.assertIn("groups_per_policy\\t64", text)
        self.assertIn("candidates_per_group\\t4", text)

    def test_hashes_and_no_selection_or_search(self):
        text = WRAPPER.read_text(encoding="utf-8")
        for name in (
            "MANIFEST_SHA",
            "PLANS_SHA",
            "GROUPS_SHA",
            "POLICY17_SHA",
            "POLICY18_SHA",
            "REFINER_SHA",
            "SAMPLER_SHA",
            "REFINER_SCRIPT_SHA",
        ):
            self.assertIn(name, text)
        self.assertIn("retry_rerank_replacement\\tfalse", text)
        self.assertIn("official_query\\tfalse", text)
        self.assertNotIn("best_of", text.lower())

    def test_allocated_gpu_and_refiner_marker(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("ALLOCATED_GPU_LIST", text)
        self.assertIn('device="${GPU_DEVICES[${slot}]}"', text)
        self.assertIn("refinement_metrics.json", text)
        self.assertIn('["output_file"]', text)


if __name__ == "__main__":
    unittest.main()
