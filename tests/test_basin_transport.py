import unittest

import torch

from crystal_dlm.basin_transport import BasinTransportConfig, basin_transport_loss


class BasinTransportTest(unittest.TestCase):
    def test_registered_config_accounting(self) -> None:
        BasinTransportConfig().validate()
        with self.assertRaises(ValueError):
            BasinTransportConfig(teacher_rows=6000).validate()

    def test_identical_geometry_has_zero_transport_and_finite_gradient(self) -> None:
        lattice = torch.eye(3, requires_grad=True) * 4.0
        coords = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], requires_grad=True
        )
        result = basin_transport_loss(
            lattice, coords, lattice.detach(), coords.detach(), image_radius=2
        )
        self.assertAlmostEqual(result["loss"].item(), 0.0, places=7)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(coords.grad).all())

    def test_periodic_translation_and_site_order_are_respected(self) -> None:
        lattice = torch.tensor(
            [[4.0, 0.0, 0.0], [1.0, 3.8, 0.0], [0.2, 0.4, 4.1]]
        )
        teacher = torch.tensor([[0.95, 0.1, 0.2], [0.25, 0.4, 0.8]])
        translated = teacher + torch.tensor([1.0, 0.0, 0.0])
        result = basin_transport_loss(
            lattice, translated, lattice, teacher, image_radius=2
        )
        self.assertLess(result["coordinates"].item(), 1.0e-10)
        with self.assertRaises(ValueError):
            basin_transport_loss(
                lattice, teacher[:1], lattice, teacher, image_radius=2
            )

    def test_metric_and_coordinate_transport_are_positive(self) -> None:
        teacher_lattice = torch.eye(3) * 4.0
        predicted_lattice = torch.eye(3) * 4.2
        teacher = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        predicted = teacher.clone()
        predicted[1, 0] += 0.1
        result = basin_transport_loss(
            predicted_lattice,
            predicted,
            teacher_lattice,
            teacher,
            image_radius=2,
        )
        self.assertGreater(result["metric"].item(), 0.0)
        self.assertGreater(result["coordinates"].item(), 0.0)
        self.assertTrue(torch.isfinite(result["loss"]))


if __name__ == "__main__":
    unittest.main()
