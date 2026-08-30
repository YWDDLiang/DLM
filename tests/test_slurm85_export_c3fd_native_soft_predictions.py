from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "85_export_c3fd_native_soft_predictions.sbatch"


class Slurm85ExportC3FDNativeSoftPredictionsTest(unittest.TestCase):
    def test_cpu_only_two_checkpoint_contract(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --gres", text)
        self.assertIn("#SBATCH --cpus-per-task=16", text)
        self.assertIn('--checkpoint "seed17=${C3FD17}"', text)
        self.assertIn('--checkpoint "seed18=${C3FD18}"', text)
        self.assertIn("checkpoint_selection\\tnone", text)
        self.assertIn("outcomes_read\\tfalse", text)

    def test_inputs_and_output_are_immutable(self):
        text = WRAPPER.read_text(encoding="utf-8")
        for value in (
            "C3FD17_SHA",
            "C3FD18_SHA",
            "TRAIN_SHA",
            "VAL_SHA",
            "VOCAB_SHA",
            "MANIFEST_SHA",
            "EXPORTER_SHA",
        ):
            self.assertIn(value, text)
        self.assertIn('[[ ! -e "${OUTPUT}" ]]', text)
        self.assertIn("sha256sum -c SHA256SUMS", text)
        self.assertIn("_SUCCESS", text)


if __name__ == "__main__":
    unittest.main()
