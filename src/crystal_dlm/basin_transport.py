"""Local basin-transport objective for periodic residual distillation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from crystal_dlm.periodic_geometry_ops import minimum_image_distances


@dataclass(frozen=True)
class BasinTransportConfig:
    """The sole pre-registered BTRD setting."""

    weight: float = 0.25
    image_radius: int = 2
    teacher_steps: int = 200
    selected_rows: int = 8192
    teacher_rows: int = 6144
    anchor_rows: int = 2048

    def validate(self) -> None:
        if self.weight <= 0:
            raise ValueError("transport weight must be positive")
        if self.image_radius != 2:
            raise ValueError("BTRD requires the audited 125-image PBC operator")
        if self.teacher_steps != 200:
            raise ValueError("BTRD teacher step count changed")
        if self.teacher_rows + self.anchor_rows != self.selected_rows:
            raise ValueError("BTRD teacher/anchor accounting changed")


def basin_transport_loss(
    predicted_lattice: torch.Tensor,
    predicted_fractional: torch.Tensor,
    teacher_lattice: torch.Tensor,
    teacher_fractional: torch.Tensor,
    *,
    image_radius: int = 2,
    epsilon: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    """Measure normalized metric and periodic site transport to a teacher basin.

    Atom identity and order are fixed by the 7+4N contract. Coordinate transport
    uses the teacher lattice and a strict bounded triclinic minimum image.
    """

    if tuple(predicted_lattice.shape) != (3, 3) or tuple(teacher_lattice.shape) != (3, 3):
        raise ValueError("lattices must be 3x3 row-vector matrices")
    if predicted_fractional.ndim != 2 or predicted_fractional.shape[-1] != 3:
        raise ValueError("predicted fractional coordinates must be [N,3]")
    if teacher_fractional.shape != predicted_fractional.shape:
        raise ValueError("teacher site identity/order changed")
    if int(predicted_fractional.shape[0]) <= 0:
        raise ValueError("BTRD requires at least one atom")
    if int(image_radius) != 2:
        raise ValueError("BTRD requires image_radius=2")

    predicted_metric = predicted_lattice @ predicted_lattice.transpose(0, 1)
    teacher_metric = teacher_lattice @ teacher_lattice.transpose(0, 1)
    metric_denominator = teacher_metric.square().sum().clamp_min(float(epsilon))
    metric = (predicted_metric - teacher_metric).square().sum() / metric_denominator

    fractional_delta = predicted_fractional - teacher_fractional
    site_distances = minimum_image_distances(
        fractional_delta,
        teacher_lattice,
        image_radius=int(image_radius),
    )
    volume = torch.linalg.det(teacher_lattice).abs().clamp_min(float(epsilon))
    characteristic_length_sq = (
        volume / int(predicted_fractional.shape[0])
    ).pow(2.0 / 3.0).clamp_min(float(epsilon))
    coordinates = site_distances.square().mean() / characteristic_length_sq
    total = metric + coordinates
    return {"metric": metric, "coordinates": coordinates, "loss": total}


__all__ = ["BasinTransportConfig", "basin_transport_loss"]
