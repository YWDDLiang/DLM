from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "99_freeze_c3fd_h1a2_fusion_prospective.sbatch"


class Slurm99FreezeC3FDH1A2FusionProspectiveTest(unittest.TestCase):
    def test_cpu_only_matched_views(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --gres", text)
        self.assertIn("#SBATCH --cpus-per-task=8", text)
        self.assertIn("views\\tH1A2_FULL,C3FD_V2", text)
        self.assertIn("matched_composition_order\\ttrue", text)
        self.assertIn("outcomes_read\\tfalse", text)
        self.assertIn("alignment_training\\tstopped_by_user", text)

    def test_inputs_are_hash_pinned(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("SOURCE_SHA", text)
        self.assertIn("MP20_SHA", text)
        self.assertIn("BUILDER_SHA", text)
        self.assertIn('[[ ! -e "${OUTPUT}" ]]', text)
        self.assertIn("sha256sum -c SHA256SUMS", text)


if __name__ == "__main__":
    unittest.main()
