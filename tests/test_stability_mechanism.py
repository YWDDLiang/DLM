import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.r5_dynamic_length import (
    exact_body_token_count,
    exact_dynamic_generation_schedule_joint_coordinates,
)
from crystal_dlm.r5_plan_state import (
    build_hard_anchor_body_prompt,
    hard_anchor_plan_state,
    parse_plan_state_json,
)


class StabilityMechanismTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = {
            "plan_state_version": "r5",
            "N": 7,
            "elements": ["O", "Fe"],
            "counts": [4, 3],
            "formula": "O4Fe3",
            "reduced_formula": "O4Fe3",
            "charge_bucket": "neutral_plausible",
            "oxidation_candidates": [-2, "mixed"],
            "anion_framework": "oxide",
            "lattice_system": "cubic",
            "spacegroup_bucket": "sg_195_230",
            "volume_per_atom_bin": "volpa_010_014",
            "prototype_key": "oxide|cubic",
        }

    def test_hard_anchor_prompt_preserves_only_composition_contract(self) -> None:
        hard = hard_anchor_plan_state(self.plan)
        for key in ("N", "elements", "counts", "formula", "reduced_formula"):
            self.assertEqual(hard[key], self.plan[key])
        self.assertEqual(hard["anion_framework"], "unknown")
        self.assertEqual(hard["lattice_system"], "unknown")
        self.assertEqual(hard["spacegroup_bucket"], "sg_unknown")
        self.assertEqual(hard["volume_per_atom_bin"], "volpa_unknown")
        rebuilt = parse_plan_state_json(build_hard_anchor_body_prompt(self.plan))
        self.assertEqual(rebuilt, hard)

    def test_joint_coordinate_schedule_covers_every_token_once(self) -> None:
        schedule = exact_dynamic_generation_schedule_joint_coordinates(7)
        flattened = [position for group in schedule for position in group]
        self.assertEqual(len(flattened), exact_body_token_count(7))
        self.assertEqual(set(flattened), set(range(exact_body_token_count(7))))
        self.assertEqual(len(schedule[-1]), 21)
        self.assertEqual(schedule[-1][:3], [8, 9, 10])


if __name__ == "__main__":
    unittest.main()
