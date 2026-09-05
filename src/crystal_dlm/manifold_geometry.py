"""Pure-Torch geometry primitives for periodic crystal manifolds.

The lattice convention matches the rest of :mod:`crystal_dlm`: lattice
vectors are rows, so fractional row vectors map to Cartesian coordinates as
``r = f @ lattice`` and the fractional metric is ``lattice @ lattice.T``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_matrix(matrix: Tensor, *, name: str) -> None:
    if not isinstance(matrix, Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"{name} must have shape [..., n, n]")
    if not matrix.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError(f"{name} contains non-finite values")


def _validate_vector(vector: Tensor, lattice: Tensor, *, name: str) -> None:
    if not isinstance(vector, Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if vector.ndim < 1 or vector.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [..., 3]")
    if not vector.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if vector.device != lattice.device or vector.dtype != lattice.dtype:
        raise ValueError(f"{name} and lattice must share device and dtype")
    if not bool(torch.isfinite(vector).all().item()):
        raise ValueError(f"{name} contains non-finite values")


def symmetric_projection(matrix: Tensor) -> Tensor:
    """Project a square matrix onto its symmetric part."""

    _validate_matrix(matrix, name="matrix")
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _symmetric_eigh(matrix: Tensor, *, name: str) -> tuple[Tensor, Tensor]:
    symmetric = symmetric_projection(matrix)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    if not bool(torch.isfinite(eigenvalues).all().item()):
        raise ValueError(f"{name} eigendecomposition produced non-finite values")
    return eigenvalues, eigenvectors


def _recompose(eigenvalues: Tensor, eigenvectors: Tensor) -> Tensor:
    return (eigenvectors * eigenvalues.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)


def _positive_eigh(
    matrix: Tensor,
    *,
    name: str,
    min_eigenvalue: float,
) -> tuple[Tensor, Tensor]:
    if float(min_eigenvalue) <= 0.0:
        raise ValueError("min_eigenvalue must be positive")
    eigenvalues, eigenvectors = _symmetric_eigh(matrix, name=name)
    if bool((eigenvalues <= 0.0).any().item()):
        raise ValueError(f"{name} must be positive definite")
    return eigenvalues.clamp_min(float(min_eigenvalue)), eigenvectors


def spd_matrix_sqrt(matrix: Tensor, *, min_eigenvalue: float = 1.0e-12) -> Tensor:
    """Return the principal square root of an SPD matrix."""

    values, vectors = _positive_eigh(
        matrix, name="matrix", min_eigenvalue=min_eigenvalue
    )
    return symmetric_projection(_recompose(torch.sqrt(values), vectors))


def spd_matrix_invsqrt(
    matrix: Tensor, *, min_eigenvalue: float = 1.0e-12
) -> Tensor:
    """Return the inverse principal square root of an SPD matrix."""

    values, vectors = _positive_eigh(
        matrix, name="matrix", min_eigenvalue=min_eigenvalue
    )
    return symmetric_projection(_recompose(torch.rsqrt(values), vectors))


def spd_matrix_log(matrix: Tensor, *, min_eigenvalue: float = 1.0e-12) -> Tensor:
    """Return the principal matrix logarithm of an SPD matrix."""

    values, vectors = _positive_eigh(
        matrix, name="matrix", min_eigenvalue=min_eigenvalue
    )
    return symmetric_projection(_recompose(torch.log(values), vectors))


def symmetric_matrix_exp(matrix: Tensor) -> Tensor:
    """Exponentiate a symmetric matrix, producing an SPD matrix."""

    values, vectors = _symmetric_eigh(matrix, name="matrix")
    result = _recompose(torch.exp(values), vectors)
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("matrix exponential overflowed")
    return symmetric_projection(result)


def relative_spd_tangent(
    metric_from: Tensor,
    metric_to: Tensor,
    *,
    min_eigenvalue: float = 1.0e-12,
) -> Tensor:
    """Return the congruence-frame log tangent from one SPD metric to another.

    The result ``H`` is defined by

    ``metric_to = sqrt(metric_from) @ exp(H) @ sqrt(metric_from)``.
    """

    _validate_matrix(metric_from, name="metric_from")
    _validate_matrix(metric_to, name="metric_to")
    if metric_from.shape != metric_to.shape:
        raise ValueError("metric_from and metric_to must have the same shape")
    if metric_from.device != metric_to.device or metric_from.dtype != metric_to.dtype:
        raise ValueError("metric_from and metric_to must share device and dtype")
    inverse_root = spd_matrix_invsqrt(
        metric_from, min_eigenvalue=min_eigenvalue
    )
    relative = inverse_root @ metric_to @ inverse_root
    return spd_matrix_log(relative, min_eigenvalue=min_eigenvalue)


def spd_congruence_update(
    metric: Tensor,
    tangent: Tensor,
    *,
    min_eigenvalue: float = 1.0e-12,
) -> Tensor:
    """Apply an SPD-preserving congruence update by a symmetric tangent."""

    _validate_matrix(metric, name="metric")
    _validate_matrix(tangent, name="tangent")
    if metric.shape != tangent.shape:
        raise ValueError("metric and tangent must have the same shape")
    if metric.device != tangent.device or metric.dtype != tangent.dtype:
        raise ValueError("metric and tangent must share device and dtype")
    root = spd_matrix_sqrt(metric, min_eigenvalue=min_eigenvalue)
    updated = root @ symmetric_matrix_exp(tangent) @ root
    return symmetric_projection(updated)


def lattice_to_metric(lattice: Tensor) -> Tensor:
    """Convert a row-vector lattice to its fractional-coordinate SPD metric."""

    _validate_matrix(lattice, name="lattice")
    if tuple(lattice.shape[-2:]) != (3, 3):
        raise ValueError("lattice must have shape [..., 3, 3]")
    metric = lattice @ lattice.transpose(-1, -2)
    # A singular lattice is not a valid crystal cell.
    if bool((torch.linalg.eigvalsh(metric) <= 0.0).any().item()):
        raise ValueError("lattice must be non-singular")
    return symmetric_projection(metric)


def metric_to_lattice(metric: Tensor) -> Tensor:
    """Return the lower-triangular canonical row lattice of an SPD metric."""

    _validate_matrix(metric, name="metric")
    if tuple(metric.shape[-2:]) != (3, 3):
        raise ValueError("metric must have shape [..., 3, 3]")
    # Cholesky both validates SPD and gives L @ L.T == metric.
    try:
        return torch.linalg.cholesky(symmetric_projection(metric))
    except RuntimeError as exc:
        raise ValueError("metric must be positive definite") from exc


def wrap_fractional(fractional: Tensor, *, period: float = 1.0) -> Tensor:
    """Wrap fractional coordinates into ``[0, period)``."""

    if not isinstance(fractional, Tensor):
        raise TypeError("fractional must be a torch tensor")
    if not fractional.is_floating_point():
        raise TypeError("fractional must be floating point")
    if float(period) <= 0.0:
        raise ValueError("period must be positive")
    if not bool(torch.isfinite(fractional).all().item()):
        raise ValueError("fractional contains non-finite values")
    return torch.remainder(fractional, float(period))


def wrapped_fractional_delta(delta: Tensor, *, period: float = 1.0) -> Tensor:
    """Map a periodic displacement to the centered minimum scalar interval."""

    if not isinstance(delta, Tensor):
        raise TypeError("delta must be a torch tensor")
    if not delta.is_floating_point():
        raise TypeError("delta must be floating point")
    if float(period) <= 0.0:
        raise ValueError("period must be positive")
    if not bool(torch.isfinite(delta).all().item()):
        raise ValueError("delta contains non-finite values")
    scaled = delta / float(period)
    return (scaled - torch.round(scaled)) * float(period)


def fractional_to_cartesian(fractional: Tensor, lattice: Tensor) -> Tensor:
    """Convert row-vector fractional coordinates to Cartesian coordinates."""

    _validate_matrix(lattice, name="lattice")
    if tuple(lattice.shape[-2:]) != (3, 3):
        raise ValueError("lattice must have shape [3, 3] or [B, 3, 3]")
    _validate_vector(fractional, lattice, name="fractional")
    if lattice.ndim == 2:
        return torch.einsum("...i,ij->...j", fractional, lattice)
    if lattice.ndim == 3:
        if fractional.ndim < 2 or fractional.shape[0] != lattice.shape[0]:
            raise ValueError("batched lattice requires fractional shape [B, ..., 3]")
        return torch.einsum("b...i,bij->b...j", fractional, lattice)
    raise ValueError("lattice must be rank two or three")


def cartesian_to_fractional(cartesian: Tensor, lattice: Tensor) -> Tensor:
    """Convert row-vector Cartesian coordinates to fractional coordinates."""

    _validate_matrix(lattice, name="lattice")
    if tuple(lattice.shape[-2:]) != (3, 3):
        raise ValueError("lattice must have shape [3, 3] or [B, 3, 3]")
    _validate_vector(cartesian, lattice, name="cartesian")
    if lattice.ndim == 2:
        flat = cartesian.reshape(-1, 3)
        solved = torch.linalg.solve(lattice.transpose(-1, -2), flat.transpose(0, 1))
        return solved.transpose(0, 1).reshape(cartesian.shape)
    if lattice.ndim == 3:
        if cartesian.ndim < 2 or cartesian.shape[0] != lattice.shape[0]:
            raise ValueError("batched lattice requires cartesian shape [B, ..., 3]")
        flat = cartesian.reshape(cartesian.shape[0], -1, 3)
        solved = torch.linalg.solve(
            lattice.transpose(-1, -2), flat.transpose(-1, -2)
        )
        return solved.transpose(-1, -2).reshape(cartesian.shape)
    raise ValueError("lattice must be rank two or three")


__all__ = [
    "cartesian_to_fractional",
    "fractional_to_cartesian",
    "lattice_to_metric",
    "metric_to_lattice",
    "relative_spd_tangent",
    "spd_congruence_update",
    "spd_matrix_invsqrt",
    "spd_matrix_log",
    "spd_matrix_sqrt",
    "symmetric_matrix_exp",
    "symmetric_projection",
    "wrap_fractional",
    "wrapped_fractional_delta",
]
