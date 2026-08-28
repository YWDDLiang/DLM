import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_c3fd_v21_step1", ROOT / "scripts" / "audit_c3fd_v21_step1.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import audit_c3fd_v21_step1.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AuditC3FDV21Step1Test(unittest.TestCase):
    def row(self):
        return {
            "proposal_supervision": True,
            "proposal_targets": {"family": 0, "N": 2, "arity": 2},
            "plan_state": {"N": 2, "elements": ["O", "Fe"]},
            "composition_supervision": True,
            "ledger_steps": [
                {"remaining_atoms": 2, "net_charge": 0, "remaining_species": 2, "branch": "unset"},
                {"remaining_atoms": 2, "net_charge": 0, "remaining_species": 2, "branch": "unset"},
                {"remaining_atoms": 1, "net_charge": -2, "remaining_species": 1, "branch": "ionic"},
                {"remaining_atoms": 0, "net_charge": 0, "remaining_species": 0, "branch": "ionic"},
            ],
        }

    def test_exact_proposal_and_ledger_pass(self):
        self.assertEqual(
            MODULE.validate_row(self.row()),
            {"proposal_exact": True, "ledger_exact": True},
        )

    def test_nonzero_terminal_charge_fails(self):
        row = self.row()
        row["ledger_steps"][-1]["net_charge"] = 1
        self.assertFalse(MODULE.validate_row(row)["ledger_exact"])


if __name__ == "__main__":
    unittest.main()
