"""Numerical lattice/torus interfaces for the registered H-P33 candidate.

This module contains no model, codec, physical filter, job or dataset paths.
Lattices are rows: Cartesian coordinates are ``fractional @ lattice``.  The
physical dimension is three; leading batch dimensions and padded site counts
are arbitrary.  Source/chart operations detach their inputs and use float64.
Only ``geometry_denoising_risk`` retains a prediction autograd graph.

The sampler is the specified finite algorithm: reverse Euler intervals followed
by one newly evaluated terminal readout.  It is not an exact reverse diffusion
or a general torus posterior sampler.  A field callback must perform one joint
field evaluation, must not mutate its state, and owns any neural input casts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class MixedGeometryConfig:
    epsilon: float = 0.002
    euler_steps: int = 32
    image_radius: int = 8
    reference_length: float = 1.0
    normalizer_std_floor: float = 1e-6

    def __post_init__(self) -> None:
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon < 1:
            raise ValueError("epsilon must be finite and in (0, 1)")
        if isinstance(self.euler_steps, bool) or not isinstance(self.euler_steps, int) or self.euler_steps < 1:
            raise ValueError("euler_steps must be a positive integer")
        if isinstance(self.image_radius, bool) or not isinstance(self.image_radius, int) or self.image_radius < 2:
            raise ValueError("image_radius must be an integer >= 2")
        for name in ("reference_length", "normalizer_std_floor"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


DEFAULT_CONFIG = MixedGeometryConfig()


def _data64(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch Tensor")
    return value.detach().to(dtype=torch.float64)


def _finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains nonfinite values")


def _batch_value(value, shape: torch.Size, device: torch.device, name: str) -> Tensor:
    result = torch.as_tensor(value, dtype=torch.float64, device=device).detach()
    try:
        result = torch.broadcast_to(result, shape)
    except RuntimeError as error:
        raise ValueError(f"{name} must broadcast to batch shape {tuple(shape)}") from error
    _finite(result, name)
    return result


def _counts(value, shape: torch.Size, device: torch.device) -> Tensor:
    counts = _batch_value(value, shape, device, "num_atoms")
    if not bool(((counts > 0) & (counts == counts.round())).all()):
        raise ValueError("num_atoms must contain positive integer counts")
    return counts


def _time(value, shape: torch.Size, device: torch.device, *, positive: bool = False) -> Tensor:
    time = _batch_value(value, shape, device, "time")
    valid = (time > 0 if positive else time >= 0) & (time <= 1)
    if not bool(valid.all()):
        raise ValueError("time must be in (0, 1]" if positive else "time must be in [0, 1]")
    return time


def _fractional(value: Tensor, atom_mask: Tensor | None) -> tuple[Tensor, Tensor]:
    fractional = _data64(value, "fractional")
    if fractional.ndim < 2 or fractional.shape[-1] != 3 or fractional.shape[-2] < 1 or fractional.numel() == 0:
        raise ValueError("fractional must have shape [..., sites, 3]")
    if atom_mask is None:
        mask = torch.ones(fractional.shape[:-1], dtype=torch.bool, device=fractional.device)
    else:
        if atom_mask.dtype != torch.bool or atom_mask.shape != fractional.shape[:-1]:
            raise ValueError("atom_mask must be boolean with shape [..., sites]")
        mask = atom_mask.to(device=fractional.device)
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("each crystal must have at least one present site")
    # Padded NaNs are not physical observations and must not enter arithmetic.
    fractional = torch.where(mask[..., None], fractional, 0.0)
    _finite(fractional, "present fractional coordinates")
    return fractional.remainder(1.0), mask


@torch.no_grad()
def half_log_spd_basis(*, device=None) -> Tensor:
    """Frobenius-orthonormal basis in the P33 declared order, float64."""
    basis = torch.zeros((6, 3, 3), dtype=torch.float64, device=device)
    basis[0].diagonal().copy_(basis.new_tensor([1.0, -1.0, 0.0]) / math.sqrt(2))
    basis[1].diagonal().copy_(basis.new_tensor([1.0, 1.0, -2.0]) / math.sqrt(6))
    for index, (row, column) in enumerate(((0, 1), (0, 2), (1, 2)), start=2):
        basis[index, row, column] = basis[index, column, row] = 1 / math.sqrt(2)
    basis[5] = torch.eye(3, dtype=torch.float64, device=device) / math.sqrt(3)
    return basis


@torch.no_grad()
def lattice_to_chart(lattice: Tensor, num_atoms, *, reference_length: float = 1.0) -> Tensor:
    """Unstandardized y=[five shape coefficients, log(V/(N*l0^3))/sqrt(3)]."""
    if not math.isfinite(reference_length) or reference_length <= 0:
        raise ValueError("reference_length must be finite and positive")
    lattice = _data64(lattice, "lattice")
    if lattice.shape[-2:] != (3, 3):
        raise ValueError("lattice must have shape [..., 3, 3]")
    _finite(lattice, "lattice")
    counts = _counts(num_atoms, lattice.shape[:-2], lattice.device)
    sign, _ = torch.linalg.slogdet(lattice)
    if not bool((sign > 0).all()):
        raise ValueError("row lattices must have strictly positive determinant")
    metric = (lattice / reference_length) @ (lattice / reference_length).transpose(-1, -2)
    _finite(metric, "lattice metric")
    eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    if not bool((eigenvalues > 0).all()):
        raise ValueError("lattice metric is not numerically positive definite")
    half_log = (eigenvectors * (0.5 * eigenvalues.log()).unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)
    chart = torch.einsum("...ab,jab->...j", half_log, half_log_spd_basis(device=lattice.device))
    chart[..., 5] -= counts.log() / math.sqrt(3)
    _finite(chart, "lattice chart")
    return chart


@torch.no_grad()
def chart_to_lattice(chart: Tensor, num_atoms, *, reference_length: float = 1.0) -> Tensor:
    """Decode y to the canonical lower row lattice, without clipping/projection."""
    if not math.isfinite(reference_length) or reference_length <= 0:
        raise ValueError("reference_length must be finite and positive")
    chart = _data64(chart, "chart")
    if chart.ndim < 1 or chart.shape[-1] != 6:
        raise ValueError("chart must have shape [..., 6]")
    _finite(chart, "chart")
    counts = _counts(num_atoms, chart.shape[:-1], chart.device)
    coefficients = chart.clone()
    coefficients[..., 5] += counts.log() / math.sqrt(3)
    half_log = torch.einsum("...j,jab->...ab", coefficients, half_log_spd_basis(device=chart.device))
    eigenvalues, eigenvectors = torch.linalg.eigh(half_log)
    exponentials = (2 * eigenvalues).exp() * reference_length**2
    _finite(exponentials, "exponentiated lattice spectrum")
    metric = (eigenvectors * exponentials.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)
    metric = 0.5 * (metric + metric.transpose(-1, -2))
    _finite(metric, "decoded lattice metric")
    lattice, info = torch.linalg.cholesky_ex(metric)
    if bool((info != 0).any()):
        raise FloatingPointError("decoded metric is not numerically positive definite")
    _finite(lattice, "decoded lattice")
    return lattice


@dataclass(frozen=True)
class LatticeNormalizer:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    raw_std: tuple[float, ...]
    std_floor: float
    reference_length: float
    source_count: int
    fit_split: str = "train"

    def __post_init__(self) -> None:
        if (self.fit_split != "train" or isinstance(self.source_count, bool)
                or not isinstance(self.source_count, int) or self.source_count < 1):
            raise ValueError("normalizer must be fitted on a nonempty train split")
        if not math.isfinite(self.std_floor) or self.std_floor <= 0:
            raise ValueError("std_floor must be finite and positive")
        if not math.isfinite(self.reference_length) or self.reference_length <= 0:
            raise ValueError("reference_length must be finite and positive")
        if any(len(values) != 6 for values in (self.mean, self.std, self.raw_std)):
            raise ValueError("normalizer requires exactly six coefficients")
        if not all(math.isfinite(value) for values in (self.mean, self.std, self.raw_std) for value in values):
            raise ValueError("normalizer coefficients must be finite")
        if any(value < 0 for value in self.raw_std):
            raise ValueError("raw standard deviations cannot be negative")
        if any(not math.isclose(value, max(raw, self.std_floor), rel_tol=1e-12, abs_tol=0)
               for value, raw in zip(self.std, self.raw_std)):
            raise ValueError("stored std must equal max(population std, std_floor)")

    @property
    def floor_hits(self) -> tuple[bool, ...]:
        return tuple(value < self.std_floor for value in self.raw_std)

    @classmethod
    @torch.no_grad()
    def fit(cls, train_lattice: Tensor, num_atoms, *, split: str = "train",
            config: MixedGeometryConfig = DEFAULT_CONFIG) -> LatticeNormalizer:
        """Fit all supplied training sources with ddof=0; caller owns provenance."""
        if split != "train":
            raise ValueError("validation/test data must not fit the lattice normalizer")
        chart = lattice_to_chart(train_lattice, num_atoms, reference_length=config.reference_length).reshape(-1, 6)
        if chart.shape[0] == 0:
            raise ValueError("normalizer needs at least one training source")
        mean = chart.mean(dim=0)
        raw_std = ((chart - mean).square().mean(dim=0)).sqrt()
        std = raw_std.clamp_min(config.normalizer_std_floor)
        return cls(tuple(mean.tolist()), tuple(std.tolist()), tuple(raw_std.tolist()),
                   config.normalizer_std_floor, config.reference_length, chart.shape[0])

    @torch.no_grad()
    def encode(self, lattice: Tensor, num_atoms) -> Tensor:
        chart = lattice_to_chart(lattice, num_atoms, reference_length=self.reference_length)
        result = (chart - chart.new_tensor(self.mean)) / chart.new_tensor(self.std)
        _finite(result, "normalized lattice chart")
        return result

    @torch.no_grad()
    def decode(self, z: Tensor, num_atoms) -> Tensor:
        z = _data64(z, "z")
        return chart_to_lattice(z * z.new_tensor(self.std) + z.new_tensor(self.mean), num_atoms,
                                reference_length=self.reference_length)

    def to_dict(self) -> dict:
        return {"schema": "mixed_geometry_lattice_normalizer_v1", "ddof": 0,
                "basis": "P33_frobenius_traceless_then_identity_over_sqrt3",
                **asdict(self), "floor_hits": list(self.floor_hits)}

    @classmethod
    def from_dict(cls, value: dict) -> LatticeNormalizer:
        if (value.get("schema") != "mixed_geometry_lattice_normalizer_v1" or value.get("ddof") != 0
                or value.get("basis") != "P33_frobenius_traceless_then_identity_over_sqrt3"):
            raise ValueError("unsupported lattice normalizer contract")
        normalizer = cls(tuple(value["mean"]), tuple(value["std"]), tuple(value["raw_std"]),
                         value["std_floor"], value["reference_length"], value["source_count"], value["fit_split"])
        if list(normalizer.floor_hits) != value.get("floor_hits"):
            raise ValueError("normalizer floor-hit record is inconsistent")
        return normalizer

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> LatticeNormalizer:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class GeometryState:
    z: Tensor
    fractional: Tensor
    t: Tensor
    atom_mask: Tensor

    @property
    def num_atoms(self) -> Tensor:
        return self.atom_mask.sum(dim=-1)

    def as_dtype(self, dtype: torch.dtype) -> GeometryState:
        """Detached network/data copy; sampling always retains its float64 state."""
        return GeometryState(self.z.detach().to(dtype=dtype), self.fractional.detach().to(dtype=dtype),
                             self.t.detach().to(dtype=dtype), self.atom_mask)


def _state64(state: GeometryState) -> GeometryState:
    z = _data64(state.z, "z")
    fractional, mask = _fractional(state.fractional, state.atom_mask)
    if z.shape != fractional.shape[:-2] + (6,) or z.device != fractional.device:
        raise ValueError("z and fractional must share batch shape/device, with six lattice channels")
    _finite(z, "z")
    time = _time(state.t, z.shape[:-1], z.device)
    return GeometryState(z, fractional, time, mask)


@torch.no_grad()
def vp_coefficients(t: Tensor) -> tuple[Tensor, Tensor]:
    t = _data64(t, "time")
    _time(t, t.shape, t.device)
    alpha, sigma = (math.pi * t / 2).cos(), (math.pi * t / 2).sin()
    # Declare the mathematical endpoint exactly, not cos(pi/2) roundoff.
    alpha = torch.where(t == 0, 1.0, torch.where(t == 1, 0.0, alpha))
    sigma = torch.where(t == 0, 0.0, torch.where(t == 1, 1.0, sigma))
    return alpha, sigma


@torch.no_grad()
def sample_log_uniform_times(batch_shape: Sequence[int] = (), *, generator: torch.Generator | None = None,
                             device=None, config: MixedGeometryConfig = DEFAULT_CONFIG) -> Tensor:
    uniform = torch.rand(tuple(batch_shape), dtype=torch.float64, device=device, generator=generator)
    return (math.log(config.epsilon) * (1 - uniform)).exp()


@torch.no_grad()
def lattice_v_target(clean_z: Tensor, lattice_noise: Tensor, t) -> Tensor:
    clean_z, noise = _data64(clean_z, "clean_z"), _data64(lattice_noise, "lattice_noise")
    if clean_z.ndim < 1 or clean_z.shape[-1] != 6 or noise.shape != clean_z.shape or noise.device != clean_z.device:
        raise ValueError("clean_z and lattice_noise must share shape [..., 6] and device")
    _finite(clean_z, "clean_z")
    _finite(noise, "lattice_noise")
    alpha, sigma = vp_coefficients(_time(t, clean_z.shape[:-1], clean_z.device))
    target = alpha[..., None] * noise - sigma[..., None] * clean_z
    _finite(target, "lattice v target")
    return target


def wrapped_image_tail_bounds(*, image_radius: int = 8, sigma_max: float = 1.0) -> dict[str, float]:
    """Uniform scalar truncation bounds for |delta|<=1 and 0<sigma<=sigma_max.

    For omitted images, |delta+n|>=|n|-1.  The retained denominator is at
    least exp(-1/(8*sigma^2)); decreasing Gaussian sums are bounded by their
    first term plus a Mills-ratio integral bound.  Radius>=2, sigma_max<=1
    also makes the normalized first-moment bounds monotone in sigma.
    """
    if isinstance(image_radius, bool) or not isinstance(image_radius, int) or image_radius < 2:
        raise ValueError("tail bounds require integer image_radius >= 2")
    if not math.isfinite(sigma_max) or not 0 < sigma_max <= 1:
        raise ValueError("tail bounds require sigma_max in (0, 1]")
    radius, variance = float(image_radius), sigma_max**2
    decay = math.exp(-(radius**2 - 0.25) / (2 * variance))
    relative_mass = 2 * decay * (1 + variance / radius)
    offset = radius + 3
    u_error = 2 * decay / sigma_max * (radius + offset + variance + offset * variance / radius)
    return {"omitted_mass_over_retained_mass": relative_mass, "absolute_u_error": u_error}


@torch.no_grad()
def wrapped_normal_u_target(noisy_fractional: Tensor, clean_fractional: Tensor, t, *,
                            atom_mask: Tensor | None = None, image_radius: int = 8) -> Tensor:
    """Float64 conditional u=t*score, not MIC/t and not a posterior sample."""
    wrapped_image_tail_bounds(image_radius=image_radius)
    noisy, mask = _fractional(noisy_fractional, atom_mask)
    clean, _ = _fractional(clean_fractional, mask)
    if noisy.shape != clean.shape or noisy.device != clean.device:
        raise ValueError("clean and noisy fractional states must share shape and device")
    time = _time(t, noisy.shape[:-2], noisy.device, positive=True)
    images = torch.arange(-image_radius, image_radius + 1, dtype=torch.float64, device=noisy.device)
    lifted = (noisy - clean)[..., None] + images
    scaled = lifted / time[..., None, None, None]
    log_weights = -0.5 * scaled.square()
    _finite(log_weights, "wrapped-normal image logits")
    weights = (log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)).exp()
    target = -(weights * scaled).sum(dim=-1)
    target = torch.where(mask[..., None], target, 0.0)
    _finite(target, "wrapped-normal target")
    return target


@dataclass(frozen=True)
class NoisyGeometry:
    state: GeometryState
    v_target: Tensor
    u_target: Tensor


@torch.no_grad()
def corrupt_geometry(clean_z: Tensor, clean_fractional: Tensor, t, *, atom_mask: Tensor | None = None,
                     generator: torch.Generator | None = None, lattice_noise: Tensor | None = None,
                     coordinate_noise: Tensor | None = None, config: MixedGeometryConfig = DEFAULT_CONFIG,
                     state_dtype: torch.dtype = torch.float32, target_dtype: torch.dtype = torch.float32) -> NoisyGeometry:
    """Construct full-noise data/targets in FP64, then make declared FP32 copies.

    The caller supplies source-keyed RNG/time; no dataset/rank-dependent seeding
    occurs here.  Optional explicit noise supports deterministic numerical tests.
    """
    if state_dtype not in (torch.float32, torch.float64) or target_dtype not in (torch.float32, torch.float64):
        raise ValueError("constructed states and targets must use float32 or float64")
    clean_f, mask = _fractional(clean_fractional, atom_mask)
    clean = _state64(GeometryState(clean_z, clean_f, torch.as_tensor(t, dtype=torch.float64, device=clean_f.device), mask))
    _time(clean.t, clean.z.shape[:-1], clean.z.device, positive=True)
    epsilon_z = (torch.randn(clean.z.shape, dtype=torch.float64, device=clean.z.device, generator=generator)
                 if lattice_noise is None else _data64(lattice_noise, "lattice_noise"))
    epsilon_f = (torch.randn(clean.fractional.shape, dtype=torch.float64, device=clean.z.device, generator=generator)
                 if coordinate_noise is None else _data64(coordinate_noise, "coordinate_noise"))
    if epsilon_f.shape != clean.fractional.shape or epsilon_f.device != clean.z.device:
        raise ValueError("coordinate_noise must share fractional shape/device")
    epsilon_f = torch.where(mask[..., None], epsilon_f, 0.0)
    _finite(epsilon_f, "present coordinate_noise")
    v = lattice_v_target(clean.z, epsilon_z, clean.t)
    alpha, sigma = vp_coefficients(clean.t)
    noisy_z = alpha[..., None] * clean.z + sigma[..., None] * epsilon_z
    noisy_f = torch.where(mask[..., None], (clean.fractional + clean.t[..., None, None] * epsilon_f).remainder(1), 0.0)
    u = wrapped_normal_u_target(noisy_f, clean.fractional, clean.t, atom_mask=mask, image_radius=config.image_radius)
    state = GeometryState(noisy_z, noisy_f, clean.t, mask).as_dtype(state_dtype)
    v, u = v.to(dtype=target_dtype), u.to(dtype=target_dtype)
    for value, name in ((state.z, "stored noisy z"), (state.fractional, "stored noisy fractional"),
                        (v, "stored v target"), (u, "stored u target")):
        _finite(value, name)
    return NoisyGeometry(state, v, u)


@dataclass(frozen=True)
class GeometryRisk:
    total: Tensor
    lattice: Tensor
    coordinates: Tensor
    per_example: Tensor


def geometry_denoising_risk(v_prediction: Tensor, u_prediction: Tensor, v_target: Tensor, u_target: Tensor,
                            atom_mask: Tensor, *, source_weights: Tensor | None = None) -> GeometryRisk:
    """FP32 half-lattice/half-coordinate risk, preserving prediction gradients.

    Every source has equal default weight regardless of its number of atoms.
    Explicit source weights handle padding/accumulation.  Targets and metadata
    are detached; no lattice decode/eigendecomposition enters this loss graph.
    """
    if (v_prediction.ndim < 1 or v_prediction.shape[-1] != 6
            or u_prediction.ndim < 2 or u_prediction.shape[-1] != 3
            or v_prediction.shape[:-1] != u_prediction.shape[:-2]
            or v_target.shape != v_prediction.shape or u_target.shape != u_prediction.shape):
        raise ValueError("prediction/target shapes must be [...,6] and [...,sites,3]")
    if atom_mask.dtype != torch.bool or atom_mask.shape != u_prediction.shape[:-1]:
        raise ValueError("atom_mask must be boolean with shape [...,sites]")
    device = v_prediction.device
    if any(value.device != device for value in (u_prediction, v_target, u_target, atom_mask)):
        raise ValueError("risk tensors must be on the same device")
    counts = atom_mask.sum(dim=-1)
    if not bool((counts > 0).all()):
        raise ValueError("each crystal needs at least one present site")
    v_pred, v_clean = v_prediction.float(), v_target.detach().float()
    u_pred = torch.where(atom_mask[..., None], u_prediction.float(), 0.0)
    u_clean = torch.where(atom_mask[..., None], u_target.detach().float(), 0.0)
    for value, name in ((v_pred, "v_prediction"), (v_clean, "v_target"), (u_pred, "present u_prediction"), (u_clean, "present u_target")):
        _finite(value, name)
    lattice = (v_pred - v_clean).square().mean(dim=-1)
    coordinates = (u_pred - u_clean).square().sum(dim=(-1, -2)) / (3 * counts)
    per_example = 0.5 * (lattice + coordinates)
    _finite(per_example, "per-example geometry risk")
    if source_weights is None:
        weights = torch.ones_like(per_example)
    else:
        weights = _batch_value(source_weights, per_example.shape, device, "source_weights").float()
        if not bool((weights >= 0).all()):
            raise ValueError("source_weights cannot be negative")
    _finite(weights, "FP32 source_weights")
    # A wholly padded optimizer contribution is zero, with a valid pred graph.
    total_weight = weights.sum()
    denominator = torch.where(total_weight > 0, total_weight, torch.ones_like(total_weight))
    reduce = lambda values: (weights * values).sum() / denominator
    return GeometryRisk(reduce(per_example), reduce(lattice), reduce(coordinates), per_example)


@torch.no_grad()
def sample_prior(atom_mask: Tensor, *, generator: torch.Generator | None = None) -> GeometryState:
    """Exact Gaussian chart and uniform torus input; no rejection or re-draw."""
    if (atom_mask.dtype != torch.bool or atom_mask.ndim < 1 or atom_mask.numel() == 0
            or not bool(atom_mask.any(dim=-1).all())):
        raise ValueError("atom_mask must describe at least one site per crystal")
    z = torch.randn(atom_mask.shape[:-1] + (6,), dtype=torch.float64, device=atom_mask.device, generator=generator)
    fractional = torch.rand(atom_mask.shape + (3,), dtype=torch.float64, device=atom_mask.device, generator=generator)
    fractional = torch.where(atom_mask[..., None], fractional, 0.0)
    return GeometryState(z, fractional, torch.ones(atom_mask.shape[:-1], dtype=torch.float64, device=z.device), atom_mask)


def _field64(state: GeometryState, v: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
    v, u = _data64(v, "v field"), _data64(u, "u field")
    if v.shape != state.z.shape or u.shape != state.fractional.shape or v.device != state.z.device or u.device != state.z.device:
        raise ValueError("field outputs must share their state's shapes and device")
    u = torch.where(state.atom_mask[..., None], u, 0.0)
    _finite(v, "v field")
    _finite(u, "present u field")
    return v, u


@torch.no_grad()
def probability_flow_step(state: GeometryState, v: Tensor, u: Tensor, next_time) -> GeometryState:
    state = _state64(state)
    following = _time(next_time, state.t.shape, state.z.device)
    if not bool((following < state.t).all()):
        raise ValueError("reverse Euler requires next_time < current time")
    v, u = _field64(state, v, u)
    delta = state.t - following
    z = state.z - (math.pi / 2) * delta[..., None] * v
    fractional = torch.where(state.atom_mask[..., None],
                             (state.fractional + delta[..., None, None] * u).remainder(1), 0.0)
    return _state64(GeometryState(z, fractional, following, state.atom_mask))


@torch.no_grad()
def terminal_readout(state: GeometryState, v: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
    """Registered local-lift readout; can average incompatible posterior modes."""
    state = _state64(state)
    _time(state.t, state.t.shape, state.z.device, positive=True)
    v, u = _field64(state, v, u)
    alpha, sigma = vp_coefficients(state.t)
    z = alpha[..., None] * state.z - sigma[..., None] * v
    fractional = torch.where(state.atom_mask[..., None],
                             (state.fractional + state.t[..., None, None] * u).remainder(1), 0.0)
    _finite(z, "terminal z")
    _finite(fractional, "terminal fractional")
    return z, fractional


@dataclass(frozen=True)
class GeometrySample:
    z: Tensor
    fractional: Tensor
    atom_mask: Tensor
    pre_readout_state: GeometryState
    nfe: int
    time_grid: tuple[float, ...]


class GeometrySamplingError(RuntimeError):
    """Unsuccessful numerical sample; caller records failure without retrying."""
    def __init__(self, message: str, *, nfe: int, time: float):
        super().__init__(message)
        self.nfe = nfe
        self.time = time


@torch.no_grad()
def sample_geometry(field: Callable[[GeometryState], tuple[Tensor, Tensor]], initial_state: GeometryState, *,
                    config: MixedGeometryConfig = DEFAULT_CONFIG) -> GeometrySample:
    """32 Euler + one epsilon readout by default; nfe counts callback invocations.

    Initial randomness is supplied by ``sample_prior``/the caller.  There is no
    constructor, intermediate quantization, solver substitution, or random final
    step.  The callback must do exactly one joint neural evaluation for its nfe
    to equal neural NFE.  Errors expose attempted nfe; this function never retries.
    """
    state = _state64(initial_state)
    if not bool((state.t == 1).all()):
        raise ValueError("H-P33 starts at time 1; supply its prior state explicitly")
    grid = tuple(config.epsilon + (1 - config.epsilon) * (index / config.euler_steps)**2
                 for index in range(config.euler_steps, -1, -1))
    nfe = 0
    try:
        for following in grid[1:]:
            nfe += 1
            v, u = field(state)
            state = probability_flow_step(state, v, u, following)
        nfe += 1
        v, u = field(state)  # A fresh evaluation at the actual epsilon state.
        z, fractional = terminal_readout(state, v, u)
    except Exception as error:
        raise GeometrySamplingError(str(error), nfe=nfe, time=float(state.t.reshape(-1)[0])) from error
    return GeometrySample(z, fractional, state.atom_mask, state, nfe, grid)


__all__ = [
    "MixedGeometryConfig", "DEFAULT_CONFIG", "half_log_spd_basis", "lattice_to_chart", "chart_to_lattice",
    "LatticeNormalizer", "GeometryState", "vp_coefficients", "sample_log_uniform_times", "lattice_v_target",
    "wrapped_image_tail_bounds", "wrapped_normal_u_target", "NoisyGeometry", "corrupt_geometry", "GeometryRisk",
    "geometry_denoising_risk", "sample_prior", "probability_flow_step", "terminal_readout", "GeometrySample",
    "GeometrySamplingError", "sample_geometry",
]
