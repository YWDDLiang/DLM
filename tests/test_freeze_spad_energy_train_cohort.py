import unittest
import importlib.util
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "scripts/freeze_spad_energy_train_cohort.py"
SPEC = importlib.util.spec_from_file_location("freeze_spad_energy_train_cohort", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
deterministic_key = MODULE.deterministic_key


class FreezeSPADEnergyTrainCohortTest(unittest.TestCase):
    def test_hash_order_is_deterministic_and_uses_plan_identity(self):
        row = {"source_row_idx": 7, "plan_state": {"N": 2, "elements": ["Na", "Cl"]}}
        self.assertEqual(deterministic_key(row, 5), deterministic_key(row, 5))
        changed = {**row, "plan_state": {"N": 2, "elements": ["K", "Cl"]}}
        self.assertNotEqual(deterministic_key(row, 5), deterministic_key(changed, 5))


if __name__ == "__main__":
    unittest.main()
