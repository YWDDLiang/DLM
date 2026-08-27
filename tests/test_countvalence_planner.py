import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.r5_plan_state import parse_countvalence_plan_state, plan_state_to_countvalencefields


class CountValencePlannerTest(unittest.TestCase):
    def test_roundtrip_preserves_composition_and_soft_properties(self) -> None:
        plan = {
            "formula": "Li2O",
            "N": 3,
            "elements": ["H", "Li", "O"],
            "counts": [0, 2, 1],
            "oxidation_candidates": [0, 1, -2],
            "charge_bucket": "neutral_plausible",
            "lattice_system": "cubic",
            "spacegroup_bucket": "sg_195_230",
            "volume_per_atom_bin": "volpa_010_014",
        }
        text = plan_state_to_countvalencefields(plan)
        rebuilt = parse_countvalence_plan_state(text)
        self.assertEqual(rebuilt["formula"], "Li2O")
        self.assertEqual(rebuilt["N"], 3)
        self.assertEqual(rebuilt["elements"], ["Li", "O"])
        self.assertEqual(rebuilt["counts"], [2, 1])
        self.assertEqual(rebuilt["generated_charge_sum"], 0)
        self.assertEqual(rebuilt["lattice_system"], "cubic")
        self.assertEqual(rebuilt["spacegroup_bucket"], "sg_195_230")
        self.assertEqual(rebuilt["volume_per_atom_bin"], "volpa_010_014")


if __name__ == "__main__":
    unittest.main()
