import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("apply_programmed_audit", Path(__file__).resolve().parents[1] / "scripts/apply_programmed_terminal_audit.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AuditApplicationTest(unittest.TestCase):
    def test_withheld_verification_preserves_energy_and_occurrence(self):
        labels = [{"trajectory_id": "a", "group_id": "g", "verified": True, "status": "verified", "terminal_energy": -499.},
                  {"trajectory_id": "b", "group_id": "g", "verified": False, "status": "not_converged", "terminal_energy": -2.}]
        checks = [{"trajectory_id": "a", "group_id": "g", "status": "inconsistent", "stored_terminal_energy_eV_atom": -499.}]
        result = MODULE.apply_audit(labels, checks)
        self.assertEqual(len(result), 2)
        self.assertFalse(result[0]["verified"])
        self.assertEqual(result[0]["terminal_energy"], -499.)
        self.assertEqual(result[1], labels[1])
        self.assertTrue(labels[0]["verified"])

    def test_missing_and_duplicate_audits_are_rejected(self):
        labels = [{"trajectory_id": "a", "verified": True}]
        for audits in ([], [{"trajectory_id": "a"}, {"trajectory_id": "a"}]):
            with self.assertRaises(ValueError):
                MODULE.apply_audit(labels, audits)


if __name__ == "__main__":
    unittest.main()
