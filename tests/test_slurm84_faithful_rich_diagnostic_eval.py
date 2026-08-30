from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "84_faithful_rich_diagnostic_eval.sbatch"


class Slurm84FaithfulRichDiagnosticEvalTest(unittest.TestCase):
    def test_resources_order_and_denominator(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4", text)
        self.assertIn("#SBATCH --cpus-per-task=32", text)
        self.assertIn("order\\traw_then_refined", text)
        self.assertLess(text.index('run_four "raw_"'), text.index('run_four ""'))
        self.assertIn("denominator\\t256", text)
        self.assertIn("official_query\\tfalse", text)

    def test_all_eight_cells_and_allocated_gpu_mapping(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("for arm in H0 R0S raw_H0 raw_R0S", text)
        self.assertIn("_OFFLINE_SUCCESS", text)
        self.assertIn("ALLOCATED_GPU_LIST", text)
        self.assertIn('device="${GPU_DEVICES[${slot}]}"', text)

    def test_inputs_are_hash_pinned(self):
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
