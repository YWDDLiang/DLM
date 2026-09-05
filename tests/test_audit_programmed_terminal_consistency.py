import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("audit_programmed_terminals", Path(__file__).resolve().parents[1] / "scripts/audit_programmed_terminal_consistency.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TerminalConsistencyTest(unittest.TestCase):
    def test_same_energy_under_representation_changes(self):
        self.assertEqual(MODULE.compare_energies(-5., [-5., -5.00001, -4.99999])["status"], "consistent")

    def test_stale_trajectory_energy_is_not_accepted(self):
        row = MODULE.compare_energies(-499., [-14., -14., -14.])
        self.assertFalse(row["stored_energy_matches_fresh"])
        self.assertTrue(row["periodic_representation_consistent"])
        self.assertEqual(row["status"], "inconsistent")

    def test_representation_artifact_is_not_accepted(self):
        row = MODULE.compare_energies(-499., [-499., -14., -499.])
        self.assertTrue(row["stored_energy_matches_fresh"])
        self.assertFalse(row["periodic_representation_consistent"])


if __name__ == "__main__":
    unittest.main()
