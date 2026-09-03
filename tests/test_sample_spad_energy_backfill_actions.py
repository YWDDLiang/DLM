import unittest

from crystal_dlm.spad_program import (
    anchor_revision_slots,
    program_from_element_order,
)


class SampleSPADEnergyBackfillActionsTest(unittest.TestCase):
    def test_energy_action_uses_first_reverse_program_anchor(self):
        plan = {"N": 4, "elements": ["O", "Na", "Cl"], "counts": [2, 1, 1]}
        program = program_from_element_order(
            plan, ["Cl", "O", "Na"], order_source="frozen_planner_llama_pointer"
        )
        self.assertEqual(anchor_revision_slots(program), (2, 0, 3))


if __name__ == "__main__":
    unittest.main()
