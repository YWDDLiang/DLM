import importlib.util
from pathlib import Path
import unittest

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_potential_closure_states",
    ROOT / "scripts" / "freeze_potential_closure_states.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import potential closure state freezer")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def answer(species):
    text, _ = arrays_to_dynamic_answer(
        lengths=[4.0, 4.0, 4.0],
        angles=[90.0, 90.0, 90.0],
        species=species,
        frac_coords=[[index / len(species)] * 3 for index in range(len(species))],
    )
    return text


class PotentialClosureStateFreezerTest(unittest.TestCase):
    def test_anchor_slots_follow_actual_body_species_then_reverse(self):
        self.assertEqual(
            MODULE.limited_anchor_slots_for_species(
                ["Zn", "Zn", "Ba", "O"],
                ["Ba", "Zn", "O"],
            ),
            (0, 2),
        )

    def test_one_source_creates_four_matched_strata(self):
        plan = {
            "N": 3,
            "elements": ["Li", "O"],
            "counts": [2, 1],
        }
        row = {
            "sample_idx": 9,
            "parsed": True,
            "text": answer(["Li", "Li", "O"]),
            "conditioning_prompt": "prompt",
            "plan_state": plan,
            "prompt_record": {
                "source_row_idx": 17,
                "teacher_answer": answer(["O", "Li", "Li"]),
                "species_program": ["O", "Li"],
                "species_program_source": "test_pointer",
            },
        }
        states, manifest = MODULE.build_states(
            [row],
            requested_sources=1,
            selection_seed=5,
        )
        self.assertEqual(len(states), 4)
        self.assertEqual(
            [value["stratum"] for value in states],
            list(MODULE.STRATA),
        )
        self.assertEqual(manifest["groups_per_stratum"], {name: 1 for name in MODULE.STRATA})
        self.assertTrue(all(value["outcomes_read"] is False for value in states))
        self.assertEqual(states[1]["active_positions"], [12, 13, 14])
        self.assertEqual(states[3]["active_positions"], [8, 9, 10])


if __name__ == "__main__":
    unittest.main()
