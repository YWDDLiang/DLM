from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "95_build_c3fd_native_alignment_groups.sbatch"


class Slurm95BuildC3FDNativeAlignmentGroupsTest(unittest.TestCase):
    def test_cpu_only_safety_contract(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --gres", text)
        self.assertIn("#SBATCH --cpus-per-task=8", text)
        self.assertIn("raw_invalid_policy\\tlexicographic_worst", text)
        self.assertIn("valid_order\\tsame_group_refined_CHGNet", text)
        self.assertIn("best_anchor\\tbest_raw_valid_refined_energy", text)
        self.assertIn("cross_composition_energy\\tfalse", text)
        self.assertIn("old_3614_used\\tfalse", text)
        self.assertIn("prospective_outcomes_read\\tfalse", text)

    def test_inputs_and_outputs_are_immutable(self):
        text = WRAPPER.read_text(encoding="utf-8")
        for name in (
            "COHORT_MANIFEST_SHA",
            "GROUPS_SHA",
            "GENERATION_FINAL_SHA",
            "BUILDER_SHA",
        ):
            self.assertIn(name, text)
        self.assertIn('[[ ! -e "${OUTPUT}" ]]', text)
        self.assertIn("sha256sum -c SHA256SUMS", text)


if __name__ == "__main__":
    unittest.main()
