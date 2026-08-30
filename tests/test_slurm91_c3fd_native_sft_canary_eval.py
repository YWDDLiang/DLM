from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "91_c3fd_native_sft_canary_eval.sbatch"


class Slurm91C3FDNativeSFTCanaryEvalTest(unittest.TestCase):
    def test_resources_raw_first_and_eight_cells(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4", text)
        self.assertIn("#SBATCH --cpus-per-task=32", text)
        self.assertIn("order\\traw_then_refined", text)
        self.assertLess(text.index('run_four "raw_"'), text.index('run_four ""'))
        self.assertIn(
            "for arm in policy82017 policy82018 raw_policy82017 raw_policy82018",
            text,
        )
        self.assertIn("_OFFLINE_SUCCESS", text)
        self.assertIn("official_query\\tfalse", text)

    def test_inputs_and_ledger_are_hash_pinned(self):
        text = WRAPPER.read_text(encoding="utf-8")
        for name in (
            "GENERATION_FINAL_SHA",
            "LEDGER_SHA",
            "ASSEMBLE_REFINED_SHA",
            "ASSEMBLE_RAW_SHA",
            "FULL_EVAL_SHA",
            "CONFIG_SHA",
            "DIRECT_METRICS_SHA",
            "GT_SHA",
            "CACHE_SHA",
        ):
            self.assertIn(name, text)
        self.assertIn("source_split\\ttrain128,val128_by_frozen_ledger", text)
        self.assertIn("policy_or_checkpoint_selection\\tfalse", text)

    def test_allocated_gpu_mapping_and_policy_symmetry(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("ALLOCATED_GPU_LIST", text)
        self.assertIn('device="${GPU_DEVICES[${slot}]}"', text)
        self.assertNotIn('CUDA_VISIBLE_DEVICES="${slot}"', text)
        for command in (
            'run_cell 0 17 "${prefix}policy82017" control',
            'run_cell 1 17 "${prefix}policy82018" candidate',
            'run_cell 2 18 "${prefix}policy82017" control',
            'run_cell 3 18 "${prefix}policy82018" candidate',
        ):
            self.assertIn(command, text)
        self.assertIn("candidate_labels_only", text)


if __name__ == "__main__":
    unittest.main()
