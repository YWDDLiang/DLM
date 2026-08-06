"""Schedule-correct, MLIP-free CrysLLMGen bridge preflight primitives.

The released parent was trained from a *forward-noised* geometry state.  A
clean WQ proposal is therefore retained as immutable conditioning evidence and
is never aliased to the reverse-process state.  This module intentionally
contains no training, selection, relaxation, or MLIP code.

NumPy helpers provide an independent, CPU-only reconstruction audit.  Torch is
imported lazily only by the A800 strict-load/trajectory path, so the local
contract tests do not masquerade as a parent-checkpoint execution.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


PARENT_SCHEDULER_TIMESTEPS = 1000
BRIDGE_TIMESTEPS = (100, 200, 400, 800)
ATTEMPTS_PER_TIMESTEP = 8
BRIDGE_CELL_COUNT = len(BRIDGE_TIMESTEPS) * ATTEMPTS_PER_TIMESTEP
BRIDGE_NOISE_IDENTITY = "wq-schedule-correct-bridge-parity-v1"


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_array(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _immutable_f64(
    value: Any,
    *,
    name: str,
    shape: tuple[int | None, ...],
) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != len(shape) or any(
        expected is not None and observed != expected
        for observed, expected in zip(array.shape, shape)
    ):
        raise ValueError(f"{name} has invalid shape {array.shape}; expected {shape}")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


@dataclasses.dataclass(frozen=True, slots=True)
class BridgeCell:
    """One pre-registered proposal/timestep cell.

    Forward and reverse noise seeds deliberately depend on the proposal panel
    index, not the timestep.  Thus the same proposal receives paired noise
    across all four schedule cells.
    """

    timestep: int
    panel_index: int
    cell_id: str
    forward_noise_seed: int
    reverse_noise_seed: int

    def __post_init__(self) -> None:
        if self.timestep not in BRIDGE_TIMESTEPS:
            raise ValueError("bridge timestep is outside the frozen matrix")
        if not 0 <= self.panel_index < ATTEMPTS_PER_TIMESTEP:
            raise ValueError("bridge panel index is outside the frozen matrix")
        if not self.cell_id:
            raise ValueError("bridge cell identity is required")
        for value in (self.forward_noise_seed, self.reverse_noise_seed):
            if not 0 <= int(value) < (1 << 63):
                raise ValueError("bridge seeds must be portable signed-63-bit values")


def _derived_seed(*, base_seed: int, panel_index: int, channel: str) -> int:
    payload = {
        "identity": BRIDGE_NOISE_IDENTITY,
        "base_seed": int(base_seed),
        "panel_index": int(panel_index),
        "channel": channel,
    }
    return int.from_bytes(
        hashlib.sha256(_canonical_json(payload)).digest()[:8], "big"
    ) & ((1 << 63) - 1)


def build_bridge_cells(
    *,
    base_seed: int,
    timesteps: Sequence[int] = BRIDGE_TIMESTEPS,
    attempts_per_timestep: int = ATTEMPTS_PER_TIMESTEP,
) -> tuple[BridgeCell, ...]:
    """Build the exact 4x8 matrix without outcome-dependent selection."""

    if tuple(int(value) for value in timesteps) != BRIDGE_TIMESTEPS:
        raise ValueError("bridge timesteps differ from the frozen matrix")
    if attempts_per_timestep != ATTEMPTS_PER_TIMESTEP:
        raise ValueError("attempts per timestep differ from the frozen matrix")
    cells: list[BridgeCell] = []
    for timestep in BRIDGE_TIMESTEPS:
        for panel_index in range(ATTEMPTS_PER_TIMESTEP):
            identity = {
                "identity": BRIDGE_NOISE_IDENTITY,
                "timestep": timestep,
                "panel_index": panel_index,
            }
            cells.append(
                BridgeCell(
                    timestep=timestep,
                    panel_index=panel_index,
                    cell_id=f"b-{hashlib.sha256(_canonical_json(identity)).hexdigest()[:24]}",
                    forward_noise_seed=_derived_seed(
                        base_seed=base_seed,
                        panel_index=panel_index,
                        channel="forward",
                    ),
                    reverse_noise_seed=_derived_seed(
                        base_seed=base_seed,
                        panel_index=panel_index,
                        channel="reverse",
                    ),
                )
            )
    if len(cells) != BRIDGE_CELL_COUNT or len({cell.cell_id for cell in cells}) != len(
        cells
    ):
        raise RuntimeError("bridge cell construction violated the 32-cell contract")
    return tuple(cells)


@dataclasses.dataclass(frozen=True, slots=True)
class ParentScheduleArrays:
    """Independent NumPy reconstruction of the two parent forward schedules."""

    alphas_cumprod: np.ndarray
    coordinate_sigmas: np.ndarray

    def __post_init__(self) -> None:
        alpha = _immutable_f64(
            self.alphas_cumprod,
            name="alphas_cumprod",
            shape=(PARENT_SCHEDULER_TIMESTEPS + 1,),
        )
        sigma = _immutable_f64(
            self.coordinate_sigmas,
            name="coordinate_sigmas",
            shape=(PARENT_SCHEDULER_TIMESTEPS + 1,),
        )
        if alpha[0] != 1.0 or sigma[0] != 0.0:
            raise ValueError("parent schedules require exact t=0 identity entries")
        if np.any(alpha <= 0.0) or np.any(alpha > 1.0):
            raise ValueError("parent cumulative alphas are outside (0,1]")
        if np.any(np.diff(alpha) > 0.0):
            raise ValueError("parent cumulative alphas must be non-increasing")
        if np.any(sigma < 0.0) or np.any(np.diff(sigma) < 0.0):
            raise ValueError("parent coordinate sigmas must be non-decreasing")
        object.__setattr__(self, "alphas_cumprod", alpha)
        object.__setattr__(self, "coordinate_sigmas", sigma)


def build_numpy_parent_schedules(
    *,
    timesteps: int = PARENT_SCHEDULER_TIMESTEPS,
    sigma_begin: float = 0.005,
    sigma_end: float = 0.5,
) -> ParentScheduleArrays:
    """Reproduce the released parent's cosine and wrapped-coordinate grids."""

    if timesteps != PARENT_SCHEDULER_TIMESTEPS:
        raise ValueError("released parent schedule must contain exactly 1000 steps")
    if not 0.0 < sigma_begin < sigma_end:
        raise ValueError("invalid parent coordinate sigma endpoints")
    smoothing = 0.008
    positions = np.linspace(0.0, float(timesteps), timesteps + 1, dtype=np.float64)
    raw_cumulative = np.cos(
        ((positions / timesteps) + smoothing)
        / (1.0 + smoothing)
        * math.pi
        * 0.5
    ) ** 2
    raw_cumulative = raw_cumulative / raw_cumulative[0]
    beta_body = 1.0 - raw_cumulative[1:] / raw_cumulative[:-1]
    beta_body = np.clip(beta_body, 0.0001, 0.9999)
    betas = np.concatenate((np.zeros(1, dtype=np.float64), beta_body))
    alphas_cumprod = np.cumprod(1.0 - betas)
    coordinate_body = np.exp(
        np.linspace(math.log(sigma_begin), math.log(sigma_end), timesteps)
    )
    coordinate_sigmas = np.concatenate(
        (np.zeros(1, dtype=np.float64), coordinate_body)
    )
    return ParentScheduleArrays(
        alphas_cumprod=alphas_cumprod,
        coordinate_sigmas=coordinate_sigmas,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class CleanProposalCondition:
    """Immutable clean proposal retained separately from the diffusion state."""

    frac_coords: np.ndarray
    lattice: np.ndarray

    def __post_init__(self) -> None:
        coordinates = _immutable_f64(
            self.frac_coords,
            name="clean proposal coordinates",
            shape=(None, 3),
        )
        lattice = _immutable_f64(
            self.lattice,
            name="clean proposal lattice",
            shape=(3, 3),
        )
        if not 1 <= coordinates.shape[0] <= 20:
            raise ValueError("clean proposal is outside parent MP20 atom support")
        if np.any(coordinates < 0.0) or np.any(coordinates >= 1.0):
            raise ValueError("clean proposal coordinates must be wrapped to [0,1)")
        if abs(float(np.linalg.det(lattice))) <= 1.0e-8:
            raise ValueError("clean proposal lattice is degenerate")
        if np.any(np.linalg.norm(lattice, axis=1) <= 1.0e-8):
            raise ValueError("clean proposal lattice vectors are degenerate")
        object.__setattr__(self, "frac_coords", coordinates)
        object.__setattr__(self, "lattice", lattice)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(
                {
                    "frac_coords_sha256": _sha256_array(self.frac_coords),
                    "lattice_sha256": _sha256_array(self.lattice),
                }
            )
        ).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class ForwardNoisedGeometryState:
    """One immutable parent-compatible forward-noised geometry state."""

    timestep: int
    frac_coords: np.ndarray
    lattice: np.ndarray
    coordinate_noise: np.ndarray
    lattice_noise: np.ndarray
    alpha_bar: float
    coordinate_sigma: float
    condition_sha256: str

    def __post_init__(self) -> None:
        if self.timestep not in BRIDGE_TIMESTEPS:
            raise ValueError("forward-noised state is outside the frozen matrix")
        coordinates = _immutable_f64(
            self.frac_coords,
            name="noisy coordinates",
            shape=(None, 3),
        )
        coordinate_noise = _immutable_f64(
            self.coordinate_noise,
            name="coordinate noise",
            shape=tuple(coordinates.shape),
        )
        lattice = _immutable_f64(
            self.lattice,
            name="noisy lattice",
            shape=(3, 3),
        )
        lattice_noise = _immutable_f64(
            self.lattice_noise,
            name="lattice noise",
            shape=(3, 3),
        )
        if not 0.0 < float(self.alpha_bar) <= 1.0:
            raise ValueError("invalid cumulative alpha")
        if float(self.coordinate_sigma) <= 0.0:
            raise ValueError("invalid coordinate sigma")
        if len(self.condition_sha256) != 64:
            raise ValueError("clean-condition SHA256 is required")
        object.__setattr__(self, "frac_coords", coordinates)
        object.__setattr__(self, "coordinate_noise", coordinate_noise)
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "lattice_noise", lattice_noise)


@dataclasses.dataclass(frozen=True, slots=True)
class ScheduleCorrectBridgeInput:
    """Clean condition and noisy reverse state with enforced non-aliasing."""

    condition: CleanProposalCondition
    state: ForwardNoisedGeometryState
    cell: BridgeCell

    def __post_init__(self) -> None:
        if self.state.timestep != self.cell.timestep:
            raise ValueError("cell and state timesteps differ")
        if self.state.condition_sha256 != self.condition.sha256:
            raise ValueError("noisy state refers to a different clean condition")
        for clean, noisy in (
            (self.condition.frac_coords, self.state.frac_coords),
            (self.condition.lattice, self.state.lattice),
        ):
            if np.shares_memory(clean, noisy):
                raise ValueError("clean condition and noisy state must not alias")


def paired_standard_normal(
    *,
    seed: int,
    shape: tuple[int, ...],
    channel: str,
) -> np.ndarray:
    """Draw deterministic PCG64 noise with a channel-separated seed."""

    if not shape or any(int(value) <= 0 for value in shape):
        raise ValueError("noise shape must be positive")
    if channel not in {"coordinate", "lattice"}:
        raise ValueError("unknown bridge noise channel")
    derived = int.from_bytes(
        hashlib.sha256(f"{int(seed)}:{channel}".encode("ascii")).digest()[:8],
        "big",
    )
    generator = np.random.Generator(np.random.PCG64(derived))
    return generator.standard_normal(shape, dtype=np.float64)


def forward_noise_numpy(
    condition: CleanProposalCondition,
    *,
    schedules: ParentScheduleArrays,
    cell: BridgeCell,
) -> ScheduleCorrectBridgeInput:
    """Apply the exact parent training-time forward-noise semantics."""

    coordinate_noise = paired_standard_normal(
        seed=cell.forward_noise_seed,
        shape=tuple(condition.frac_coords.shape),
        channel="coordinate",
    )
    lattice_noise = paired_standard_normal(
        seed=cell.forward_noise_seed,
        shape=(3, 3),
        channel="lattice",
    )
    alpha_bar = float(schedules.alphas_cumprod[cell.timestep])
    coordinate_sigma = float(schedules.coordinate_sigmas[cell.timestep])
    noisy_coordinates = (
        condition.frac_coords + coordinate_sigma * coordinate_noise
    ) % 1.0
    noisy_lattice = (
        math.sqrt(alpha_bar) * condition.lattice
        + math.sqrt(1.0 - alpha_bar) * lattice_noise
    )
    state = ForwardNoisedGeometryState(
        timestep=cell.timestep,
        frac_coords=noisy_coordinates,
        lattice=noisy_lattice,
        coordinate_noise=coordinate_noise,
        lattice_noise=lattice_noise,
        alpha_bar=alpha_bar,
        coordinate_sigma=coordinate_sigma,
        condition_sha256=condition.sha256,
    )
    return ScheduleCorrectBridgeInput(condition=condition, state=state, cell=cell)


def reconstruction_errors(bridge_input: ScheduleCorrectBridgeInput) -> dict[str, float]:
    """Invert the known forward noise and report periodic/lattice error."""

    state = bridge_input.state
    condition = bridge_input.condition
    reconstructed_coordinates = (
        state.frac_coords - state.coordinate_sigma * state.coordinate_noise
    ) % 1.0
    delta = np.abs(reconstructed_coordinates - condition.frac_coords)
    periodic_delta = np.minimum(delta, 1.0 - delta)
    reconstructed_lattice = (
        state.lattice
        - math.sqrt(1.0 - state.alpha_bar) * state.lattice_noise
    ) / math.sqrt(state.alpha_bar)
    return {
        "coordinate_periodic_max_abs_error": float(periodic_delta.max(initial=0.0)),
        "lattice_max_abs_error": float(
            np.max(np.abs(reconstructed_lattice - condition.lattice), initial=0.0)
        ),
    }


def respaced_timesteps(start_timestep: int, *, steps: int = 32) -> tuple[int, ...]:
    """Return the frozen descending parent grid for one bridge cell."""

    if start_timestep not in BRIDGE_TIMESTEPS:
        raise ValueError("reverse start timestep is outside the frozen matrix")
    if not 2 <= steps <= start_timestep:
        raise ValueError("invalid respaced reverse-step count")
    values = np.rint(np.linspace(start_timestep, 1, steps)).astype(int)
    result = tuple(int(value) for value in values)
    if (
        len(result) != steps
        or len(set(result)) != steps
        or result[0] != start_timestep
        or result[-1] != 1
        or any(first <= second for first, second in zip(result, result[1:]))
    ):
        raise RuntimeError("invalid schedule-correct respaced timestep grid")
    return result


def forward_noise_torch(
    *,
    clean_frac_coords: Any,
    clean_lattice: Any,
    coordinate_noise: Any,
    lattice_noise: Any,
    timestep: int,
    alphas_cumprod: Any,
    coordinate_sigmas: Any,
) -> dict[str, Any]:
    """Torch counterpart used only after strict-loading the released parent."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - A800-only execution path.
        raise RuntimeError("torch is required for A800 bridge parity") from exc
    if timestep not in BRIDGE_TIMESTEPS:
        raise ValueError("bridge timestep is outside the frozen matrix")
    if len(alphas_cumprod) != PARENT_SCHEDULER_TIMESTEPS + 1:
        raise ValueError("parent cumulative-alpha schedule length is not 1001")
    if len(coordinate_sigmas) != PARENT_SCHEDULER_TIMESTEPS + 1:
        raise ValueError("parent coordinate-sigma schedule length is not 1001")
    clean_x = clean_frac_coords.detach().clone()
    clean_l = clean_lattice.detach().clone()
    noise_x = coordinate_noise.detach().clone()
    noise_l = lattice_noise.detach().clone()
    if clean_x.ndim != 2 or clean_x.shape[1] != 3 or noise_x.shape != clean_x.shape:
        raise ValueError("torch coordinate/noise shapes differ")
    if tuple(clean_l.shape) != (1, 3, 3) or noise_l.shape != clean_l.shape:
        raise ValueError("torch lattice/noise shapes differ")
    for name, value in (
        ("clean coordinates", clean_x),
        ("clean lattice", clean_l),
        ("coordinate noise", noise_x),
        ("lattice noise", noise_l),
    ):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} is non-finite")
    alpha_bar = alphas_cumprod[timestep].to(
        device=clean_l.device, dtype=clean_l.dtype
    )
    coordinate_sigma = coordinate_sigmas[timestep].to(
        device=clean_x.device, dtype=clean_x.dtype
    )
    noisy_x = (clean_x + coordinate_sigma * noise_x) % 1.0
    noisy_l = (
        torch.sqrt(alpha_bar) * clean_l
        + torch.sqrt(1.0 - alpha_bar) * noise_l
    )
    if clean_x.data_ptr() == noisy_x.data_ptr() or clean_l.data_ptr() == noisy_l.data_ptr():
        raise RuntimeError("clean condition and torch noisy state unexpectedly alias")
    return {
        "condition_frac_coords": clean_x,
        "condition_lattice": clean_l,
        "state_frac_coords": noisy_x,
        "state_lattice": noisy_l,
        "coordinate_noise": noise_x,
        "lattice_noise": noise_l,
        "alpha_bar": alpha_bar,
        "coordinate_sigma": coordinate_sigma,
        "timestep": int(timestep),
    }


def lattice_valid_torch(lattice: Any) -> bool:
    """Fail-closed finite/non-degenerate lattice check for reverse states."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - A800-only execution path.
        raise RuntimeError("torch is required for A800 bridge parity") from exc
    if tuple(lattice.shape[-2:]) != (3, 3) or not torch.isfinite(lattice).all():
        return False
    determinants = torch.linalg.det(lattice)
    lengths = torch.linalg.vector_norm(lattice, dim=-1)
    return bool(
        (torch.abs(determinants) > 1.0e-8).all()
        and (lengths > 1.0e-8).all()
        and torch.isfinite(determinants).all()
    )


def run_parent_reverse_from_noisy_state(
    *,
    model: Any,
    batch: Any,
    bridge_state: Mapping[str, Any],
    reverse_steps: int = 32,
    corrector_step_lr: float = 1.0e-5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the released decoder from a correctly forward-noised state.

    This mirrors the registered respaced parent update but never substitutes
    the clean proposal for ``x_t``/``L_t``.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - A800-only execution path.
        raise RuntimeError("torch is required for A800 bridge parity") from exc
    timestep = int(bridge_state["timestep"])
    grid = respaced_timesteps(timestep, steps=reverse_steps)
    x = bridge_state["state_frac_coords"].detach().clone()
    lattice = bridge_state["state_lattice"].detach().clone()
    calls = 0
    first_reverse_lattice_valid: bool | None = None
    with torch.no_grad():
        for index, current in enumerate(grid):
            following = grid[index + 1] if index + 1 < len(grid) else 0
            times = torch.full((1,), current, device=x.device, dtype=torch.long)
            time_embedding = model.time_embedding(times)
            sigma_x = model.sigma_scheduler.sigmas[current]
            sigma_norm = model.sigma_scheduler.sigmas_norm[current]
            next_sigma_x = model.sigma_scheduler.sigmas[following]

            random_coordinate = (
                torch.randn_like(x) if following > 0 else torch.zeros_like(x)
            )
            corrector_step = corrector_step_lr * (
                sigma_x / model.sigma_scheduler.sigma_begin
            ).square()
            coordinate_noise = torch.sqrt(2.0 * corrector_step)
            predicted_lattice, predicted_coordinate = model.decoder(
                time_embedding,
                batch.atom_types,
                x,
                lattice,
                batch.num_atoms,
                batch.batch,
            )
            calls += 1
            predicted_coordinate = predicted_coordinate * torch.sqrt(sigma_norm)
            x_half = (
                x
                - corrector_step * predicted_coordinate
                + coordinate_noise * random_coordinate
            )

            predicted_lattice, predicted_coordinate = model.decoder(
                time_embedding,
                batch.atom_types,
                x_half,
                lattice,
                batch.num_atoms,
                batch.batch,
            )
            calls += 1
            predicted_coordinate = predicted_coordinate * torch.sqrt(sigma_norm)
            coordinate_step = sigma_x.square() - next_sigma_x.square()
            coordinate_std = torch.sqrt(
                (
                    next_sigma_x.square()
                    * coordinate_step
                    / sigma_x.square()
                ).clamp_min(0.0)
            )
            random_coordinate = (
                torch.randn_like(x) if following > 0 else torch.zeros_like(x)
            )
            x = (
                x_half
                - coordinate_step * predicted_coordinate
                + coordinate_std * random_coordinate
            ) % 1.0

            alpha_bar = model.beta_scheduler.alphas_cumprod[current]
            next_alpha_bar = model.beta_scheduler.alphas_cumprod[following]
            predicted_clean_lattice = (
                lattice - torch.sqrt(1.0 - alpha_bar) * predicted_lattice
            ) / torch.sqrt(alpha_bar)
            random_lattice = (
                torch.randn_like(lattice)
                if following > 0
                else torch.zeros_like(lattice)
            )
            lattice = (
                torch.sqrt(next_alpha_bar) * predicted_clean_lattice
                + torch.sqrt(1.0 - next_alpha_bar) * random_lattice
            )
            if index == 0:
                first_reverse_lattice_valid = lattice_valid_torch(lattice)
            if not torch.isfinite(x).all() or not torch.isfinite(lattice).all():
                raise FloatingPointError("non-finite schedule-correct parent state")
    if calls != 2 * reverse_steps or first_reverse_lattice_valid is None:
        raise RuntimeError("schedule-correct parent call accounting changed")
    return (
        {
            "num_atoms": batch.num_atoms,
            "atom_types": batch.atom_types,
            "frac_coords": x,
            "lattices": lattice,
        },
        {
            "decoder_calls": calls,
            "reverse_steps": reverse_steps,
            "timestep_grid": list(grid),
            "first_reverse_lattice_valid": first_reverse_lattice_valid,
            "all_trajectory_values_finite": True,
            "clean_condition_used_as_reverse_state": False,
        },
    )
