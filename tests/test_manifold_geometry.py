from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.manifold_geometry import (
    cartesian_to_fractional,
    fractional_to_cartesian,
    lattice_to_metric,
    metric_to_lattice,
    relative_spd_tangent,
    spd_congruence_update,
    spd_matrix_invsqrt,
    spd_matrix_log,
    spd_matrix_sqrt,
    symmetric_matrix_exp,
    wrap_fractional,
    wrapped_fractional_delta,
)


class ManifoldGeometryTest(unittest.TestCase):
    def test_spd_maps_are_positive_and_round_trip(self) -> None:
        raw = torch.tensor(
            [[2.0, 0.4, -0.1], [0.4, 1.5, 0.2], [-0.1, 0.2, 1.2]],
            dtype=torch.float64,
            requires_grad=True,
        )
        metric = raw @ raw.transpose(-1, -2) + 0.2 * torch.eye(3, dtype=raw.dtype)
        root = spd_matrix_sqrt(metric)
        inverse_root = spd_matrix_invsqrt(metric)
        logged = spd_matrix_log(metric)
        recovered = symmetric_matrix_exp(logged)
        identity = torch.eye(3, dtype=raw.dtype)
        self.assertTrue(torch.all(torch.linalg.eigvalsh(root) > 0.0))
        self.assertTrue(torch.allclose(root @ root, metric, atol=1.0e-10))
        self.assertTrue(torch.allclose(inverse_root @ metric @ inverse_root, identity, atol=1.0e-10))
        self.assertTrue(torch.allclose(recovered, metric, atol=1.0e-10))
        recovered.sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_relative_tangent_congruence_round_trip(self) -> None:
        first_lattice = torch.tensor(
            [[3.1, 0.0, 0.0], [0.6, 3.7, 0.0], [0.2, -0.4, 4.2]],
            dtype=torch.float64,
        )
        second_lattice = torch.tensor(
            [[3.4, 0.0, 0.0], [0.3, 3.5, 0.0], [-0.1, 0.5, 4.0]],
            dtype=torch.float64,
        )
        first = lattice_to_metric(first_lattice)
        second = lattice_to_metric(second_lattice)
        tangent = relative_spd_tangent(first, second)
        recovered = spd_congruence_update(first, tangent)
        self.assertTrue(torch.allclose(recovered, second, atol=1.0e-9))
        canonical = metric_to_lattice(recovered)
        self.assertTrue(torch.allclose(lattice_to_metric(canonical), second, atol=1.0e-9))

    def test_periodic_wrapping_and_translation(self) -> None:
        coordinates = torch.tensor(
            [[-0.1, 0.2, 1.3], [2.9, -1.8, 0.3]], dtype=torch.float64
        )
        translated = coordinates + torch.tensor([3.0, -2.0, 5.0])
        self.assertTrue(torch.allclose(wrap_fractional(coordinates), wrap_fractional(translated)))
        delta = torch.tensor([0.98, -0.98, 1.02], dtype=torch.float64)
        self.assertTrue(
            torch.allclose(
                wrapped_fractional_delta(delta),
                torch.tensor([-0.02, 0.02, 0.02], dtype=torch.float64),
            )
        )

    def test_cartesian_fractional_round_trip_unbatched_and_batched(self) -> None:
        lattice = torch.tensor(
            [[3.2, 0.0, 0.0], [0.7, 3.5, 0.0], [0.2, 0.4, 4.1]],
            dtype=torch.float64,
        )
        fractional = torch.tensor(
            [[0.04, 0.11, 0.19], [0.42, 0.63, 0.87]], dtype=torch.float64
        )
        cartesian = fractional_to_cartesian(fractional, lattice)
        self.assertTrue(
            torch.allclose(cartesian_to_fractional(cartesian, lattice), fractional, atol=1.0e-12)
        )

        batched_lattice = torch.stack((lattice, 1.2 * lattice))
        batched_fractional = torch.stack((fractional, 0.5 * fractional))
        batched_cartesian = fractional_to_cartesian(batched_fractional, batched_lattice)
        self.assertTrue(
            torch.allclose(
                cartesian_to_fractional(batched_cartesian, batched_lattice),
                batched_fractional,
                atol=1.0e-12,
            )
        )

    def test_non_spd_and_non_finite_inputs_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive definite"):
            spd_matrix_log(torch.diag(torch.tensor([1.0, 0.0, 2.0])))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            spd_matrix_sqrt(torch.diag(torch.tensor([1.0, float("nan"), 2.0])))


if __name__ == "__main__":
    unittest.main()
