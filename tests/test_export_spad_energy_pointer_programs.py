import unittest

from crystal_dlm.spad_program import program_from_element_order


class ExportSPADEnergyPointerProgramsTest(unittest.TestCase):
    def test_exported_program_contract_is_exact_permutation(self):
        plan = {"N": 4, "elements": ["O", "Na", "Cl"], "counts": [2, 1, 1]}
        program = program_from_element_order(
            plan, ["Cl", "O", "Na"], order_source="frozen_planner_llama_pointer"
        )
        self.assertEqual(program.element_order, ("Cl", "O", "Na"))
        self.assertEqual(sum(entry.count for entry in program.entries), 4)


if __name__ == "__main__":
    unittest.main()
