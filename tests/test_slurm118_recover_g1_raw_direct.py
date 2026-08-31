from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "slurm" / "118_recover_g1_raw_direct.sbatch"


class G1RawRecoverySlurmTest(unittest.TestCase):
    def test_reuses_body_and_changes_only_engineering_label(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn("source_body_job\t39105", text)
        self.assertIn("body_rerun\tfalse", text)
        self.assertIn("science_parameters_changed\tfalse", text)
        self.assertIn("--arm-label candidate", text)
        self.assertNotIn("sample_sgtc", text)
        self.assertIn("body_reused_from_job", text)

    def test_gate_and_no_downstream_energy(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn('result["comp_valid"] >= 244', text)
        self.assertIn('result["body_parsed"] >= 246', text)
        self.assertIn('result["direct_joint"] > 106', text)
        self.assertIn("model494\tfalse", text)
        self.assertIn("chgnet\tfalse", text)
        self.assertIn("official_query\tfalse", text)


if __name__ == "__main__":
    unittest.main()
