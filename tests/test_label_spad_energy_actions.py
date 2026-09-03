import unittest

from crystal_dlm.spad_program import coordinate_positions


class LabelSPADEnergyActionsTest(unittest.TestCase):
    def test_active_positions_are_one_exact_xyz_transaction(self):
        self.assertEqual(coordinate_positions(0), (8, 9, 10))
        self.assertEqual(coordinate_positions(3), (20, 21, 22))


if __name__ == "__main__":
    unittest.main()
