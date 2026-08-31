from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "94_c3fd_native_alignment_pool_eval.sbatch"


class Slurm94C3FDNativeAlignmentPoolEvalTest(unittest.TestCase):
    def test_two_gpu_raw_first_four_cells(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2", text)
        self.assertIn("#SBATCH --cpus-per-task=16", text)
        self.assertLess(text.index('run_two "raw_"'), text.index('run_two ""'))
        self.assertIn("for arm in policy82017 policy82018 raw_policy82017 raw_policy82018", text)
        self.assertIn("_OFFLINE_SUCCESS", text)

    def test_group_and_evaluator_inputs_are_pinned(self):
        text = WRAPPER.read_text(encoding="utf-8")
        for name in (
            "GENERATION_FINAL_SHA",
            "GROUPS_SHA",
            "ASSEMBLE_REFINED_SHA",
            "ASSEMBLE_RAW_SHA",
            "FULL_EVAL_SHA",
            "CONFIG_SHA",
            "DIRECT_METRICS_SHA",
            "GT_SHA",
            "CACHE_SHA",
        ):
            self.assertIn(name, text)
        self.assertIn("groups_per_policy\\t64", text)
        self.assertIn("candidates_per_group\\t4", text)
        self.assertIn("policy_selection\\tfalse", text)
        self.assertIn("official_query\\tfalse", text)

    def test_allocated_gpu_mapping(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("ALLOCATED_GPU_LIST", text)
        self.assertIn('device="${GPU_DEVICES[${slot}]}"', text)
        self.assertNotIn('CUDA_VISIBLE_DEVICES="${slot}"', text)


if __name__ == "__main__":
    unittest.main()
