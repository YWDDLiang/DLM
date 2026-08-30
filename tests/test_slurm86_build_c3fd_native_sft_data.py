from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "86_build_c3fd_native_sft_data.sbatch"


class Slurm86BuildC3FDNativeSFTDataTest(unittest.TestCase):
    def test_cpu_only_formal_four_view_contract(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --gres", text)
        self.assertIn("#SBATCH --cpus-per-task=16", text)
        self.assertIn("predicted-native-seed17", text)
        self.assertIn("predicted-native-seed18", text)
        self.assertIn("planner_checkpoint_selection\\tnone", text)
        self.assertIn("legacy_single_prediction\\tfalse", text)
        self.assertNotIn("--allow-legacy-single-prediction-development", text)

    def test_all_inputs_and_output_are_pinned(self):
        text = WRAPPER.read_text(encoding="utf-8")
        for value in (
            "INPUT_TRAIN_SHA",
            "INPUT_VAL_SHA",
            "SEMANTIC_TRAIN_SHA",
            "SEMANTIC_VAL_SHA",
            "PREDICTED_TRAIN_SHA",
            "PREDICTED_VAL_SHA",
            "PREDICTED_MANIFEST_SHA",
            "BUILDER_SHA",
            "NATIVE_PLAN_SHA",
        ):
            self.assertIn(value, text)
        self.assertIn('[[ ! -e "${OUTPUT}" ]]', text)
        self.assertIn("sha256sum -c SHA256SUMS", text)
        self.assertIn("chemsys_overlap", text)


if __name__ == "__main__":
    unittest.main()
