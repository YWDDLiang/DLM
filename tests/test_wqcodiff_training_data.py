from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from crystal_dlm.wqcodiff.training_data import (
    _training_geometry_evidence,
    _wrapped_gaussian_score,
)


class WrappedGaussianScoreTests(unittest.TestCase):
    def test_score_is_periodic_and_odd(self) -> None:
        sigma = 0.37
        delta = np.asarray([0.07, 0.31, -0.19])
        score = _wrapped_gaussian_score(delta, sigma)
        periodic = _wrapped_gaussian_score(delta + 3.0, sigma)
        reflected = _wrapped_gaussian_score(-delta, sigma)
        np.testing.assert_allclose(score, periodic, atol=1.0e-12, rtol=0.0)
        np.testing.assert_allclose(score, -reflected, atol=1.0e-12, rtol=0.0)

    def test_score_vanishes_at_torus_symmetry_points(self) -> None:
        score = _wrapped_gaussian_score(np.asarray([0.0, 0.5, -0.5]), 0.5)
        np.testing.assert_allclose(score, np.zeros(3), atol=1.0e-12, rtol=0.0)

    def test_small_noise_limit_matches_unwrapped_gaussian(self) -> None:
        delta = np.asarray([0.013, -0.021])
        sigma = 0.05
        expected = -delta / sigma**2
        np.testing.assert_allclose(
            _wrapped_gaussian_score(delta, sigma),
            expected,
            atol=1.0e-10,
            rtol=1.0e-10,
        )

    def test_score_matches_log_density_finite_difference(self) -> None:
        sigma = 0.5
        delta = 0.29
        epsilon = 1.0e-6
        images = np.arange(-8, 9, dtype=np.float64)

        def log_density(value: float) -> float:
            lifted = value + images
            values = -0.5 * np.square(lifted / sigma)
            maximum = float(values.max())
            return maximum + math.log(float(np.exp(values - maximum).sum()))

        numerical = (log_density(delta + epsilon) - log_density(delta - epsilon)) / (
            2.0 * epsilon
        )
        analytic = float(_wrapped_gaussian_score(delta, sigma))
        self.assertAlmostEqual(analytic, numerical, places=8)


class GeometryEvidenceTests(unittest.TestCase):
    @staticmethod
    def _evidence(coordinates: np.ndarray) -> list[list[float]]:
        lattice = np.eye(3, dtype=np.float64) * 5.0
        return _training_geometry_evidence(
            space_group=1,
            primitive_coordinates=coordinates,
            primitive_lattice=lattice,
            atom_to_orbit=np.asarray([0, 1], dtype=np.int64),
            candidate_wyckoff=(None, None),
            primitive_lattice_transform=np.eye(3, dtype=np.float64),
            conventional_lattice=lattice,
            time=0.6,
        )

    def test_geometry_builder_has_no_corruption_label_channel(self) -> None:
        parameters = set(inspect.signature(_training_geometry_evidence).parameters)
        self.assertTrue(
            parameters.isdisjoint(
                {"is_false", "wrong_wyckoff", "wrong_species", "event", "target"}
            )
        )
        coordinates = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        first = self._evidence(coordinates)
        second = self._evidence(coordinates.copy())
        np.testing.assert_array_equal(first, second)

    def test_collision_signal_responds_only_to_actual_geometry(self) -> None:
        separated = self._evidence(
            np.asarray([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        )
        collided = self._evidence(
            np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
        )
        self.assertEqual(separated[0][0], 0.0)
        self.assertEqual(separated[1][0], 0.0)
        self.assertGreater(collided[0][0], 0.9)
        self.assertGreater(collided[1][0], 0.9)


if __name__ == "__main__":
    unittest.main()
