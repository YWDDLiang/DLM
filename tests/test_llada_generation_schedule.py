import unittest

from crystal_dlm.generation_schedule import (
    n_elements_coords_lattice_schedule,
    n_elements_sequential_rest_schedule,
)


class LLaDAGenerationScheduleTests(unittest.TestCase):
    def test_n_elements_coords_lattice_schedule_positions(self):
        schedule = n_elements_coords_lattice_schedule(max_atoms=20)
        self.assertEqual(len(schedule), 3)
        self.assertEqual(schedule[0], [0])
        self.assertEqual(schedule[1], [8 + 5 * i for i in range(20)])

        expected_third = set(range(1, 7))
        for slot_index in range(20):
            base = 7 + slot_index * 5
            expected_third.update([base + 2, base + 3, base + 4])
        self.assertEqual(set(schedule[2]), expected_third)

        scheduled = [position for group in schedule for position in group]
        self.assertEqual(len(scheduled), 87)
        self.assertEqual(len(scheduled), len(set(scheduled)))
        self.assertNotIn(7, scheduled)
        self.assertNotIn(12, scheduled)
        self.assertNotIn(102, scheduled)

    def test_n_elements_sequential_rest_schedule_positions(self):
        schedule = n_elements_sequential_rest_schedule(max_atoms=20)
        self.assertEqual(schedule[0], [0])
        self.assertEqual(schedule[1], [8 + 5 * i for i in range(20)])
        self.assertEqual(schedule[2:8], [[1], [2], [3], [4], [5], [6]])
        self.assertEqual(schedule[8:11], [[9], [10], [11]])

        scheduled = [position for group in schedule for position in group]
        self.assertEqual(len(scheduled), 87)
        self.assertEqual(len(scheduled), len(set(scheduled)))
        self.assertNotIn(7, scheduled)
        self.assertNotIn(12, scheduled)
        self.assertNotIn(102, scheduled)


if __name__ == "__main__":
    unittest.main()
