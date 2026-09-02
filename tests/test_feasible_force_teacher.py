import unittest

import numpy as np

from crystal_dlm.feasible_force_teacher import (
    adjacent_token_feasible_projection,
    bounded_force_displacement,
    minimum_image_vector,
    periodic_pair_summary,
    project_periodic_feasible,
)


class FeasibleForceTeacherTest(unittest.TestCase):
    def test_exact_pbc_vector_crosses_wrap_boundary(self):
        lattice = np.eye(3) * 10.0
        vector, distance = minimum_image_vector(
            np.array([0.99, 0.0, 0.0]),
            np.array([0.01, 0.0, 0.0]),
            lattice,
            image_radius=2,
        )
        self.assertAlmostEqual(distance, 0.2, places=7)
        np.testing.assert_allclose(vector, np.array([0.2, 0.0, 0.0]), atol=1e-7)

    def test_projection_resolves_severe_periodic_collision(self):
        lattice = np.eye(3) * 10.0
        coordinates = np.array([[0.99, 0.0, 0.0], [0.005, 0.0, 0.0]])
        projected, report = project_periodic_feasible(
            coordinates,
            lattice,
            [8, 8],
            image_radius=2,
        )
        minimum, violations = periodic_pair_summary(
            projected,
            lattice,
            [8, 8],
            image_radius=2,
        )
        self.assertLess(report.initial_minimum_distance_A, 0.5)
        self.assertGreaterEqual(minimum, 0.599)
        self.assertEqual(violations, 0)
        self.assertTrue(report.converged)

    def test_projection_leaves_feasible_coordinates_unchanged(self):
        lattice = np.eye(3) * 10.0
        coordinates = np.array([[0.1, 0.1, 0.1], [0.4, 0.4, 0.4]])
        projected, report = project_periodic_feasible(
            coordinates,
            lattice,
            [14, 8],
            image_radius=2,
        )
        np.testing.assert_allclose(projected, coordinates, atol=1e-12)
        self.assertEqual(report.initial_margin_violations, 0)
        self.assertEqual(report.final_margin_violations, 0)

    def test_force_displacement_is_translation_free_and_bounded(self):
        forces = np.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]])
        displacement = bounded_force_displacement(forces)
        np.testing.assert_allclose(displacement.mean(axis=0), np.zeros(3), atol=1e-12)
        self.assertLessEqual(
            float(np.linalg.norm(displacement, axis=1).max()),
            0.15 + 1e-12,
        )

    def test_adjacent_token_projection_resolves_rounding_trap(self):
        lattice = np.eye(3) * 7.0
        quantized = np.array([[0.0, 0.0, 0.0], [0.07, 0.0, 0.0]])
        selected, report = adjacent_token_feasible_projection(
            quantized,
            quantized,
            lattice,
            [8, 8],
        )
        minimum, _violations = periodic_pair_summary(
            selected,
            lattice,
            [8, 8],
            margin_scale=0.0,
            margin_floor_A=0.50,
            margin_ceiling_A=0.50,
        )
        self.assertTrue(report.attempted)
        self.assertTrue(report.resolved)
        self.assertGreaterEqual(minimum, 0.50)
        self.assertGreater(report.changed_coordinate_tokens, 0)


if __name__ == "__main__":
    unittest.main()
