from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "81_rich_recovery_canary_eval.sbatch"


class RichRecoveryCanaryEvalSlurmTest(unittest.TestCase):
    def test_resources_order_and_denominator_are_frozen(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6", text)
        self.assertIn("#SBATCH --cpus-per-task=48", text)
        self.assertIn("order\\trefined_then_raw", text)
        self.assertLess(text.index('run_six ""'), text.index('run_six "raw_"'))
        self.assertIn("denominator\\t256", text)
        self.assertIn("missing_policy\\tpreserve_by_sample_idx", text)
        self.assertIn("official_query\\tfalse", text)

    def test_composite_refinement_mapping_is_explicit(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn('if [[ "${stream}/${arm}" == "17/M0" ]]', text)
        self.assertIn('"${PARENT}/stream17/M0/refine"', text)
        self.assertIn('"${RECOVERY}/stream${stream}/${arm}"', text)

    def test_all_twelve_cells_are_required(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("for arm in M0 RCF R0 raw_M0 raw_RCF raw_R0", text)
        self.assertIn("_OFFLINE_SUCCESS", text)
        self.assertIn("ALLOCATED_GPU_LIST", text)

    def test_input_hashes_cover_generation_and_evaluators(self):
        text = WRAPPER.read_text(encoding="utf-8")
        for value in (
            "GENERATION_FINAL_SHA",
            "ASSEMBLE_REFINED_SHA",
            "ASSEMBLE_RAW_SHA",
            "FULL_EVAL_SHA",
            "CONFIG_SHA",
            "DIRECT_METRICS_SHA",
            "GT_SHA",
            "CACHE_SHA",
        ):
            self.assertIn(value, text)


if __name__ == "__main__":
    unittest.main()
