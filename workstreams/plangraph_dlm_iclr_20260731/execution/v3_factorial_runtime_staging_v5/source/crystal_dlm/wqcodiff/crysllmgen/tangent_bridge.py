"""Training-free Wyckoff tangent projection for released CrysLLMGen steps.

The released parent operates on a primitive full-atom graph.  The WQ state,
by contrast, stores a conventional-setting lattice chart and one free
coordinate chart per Wyckoff orbit.  This module is the dependency-light
adapter between those two existing representations:

* a proposed parent coordinate step is pulled back with the exact/fallback
  orbit Jacobians already produced by :func:`wqcodiff.runtime.expand_state`;
* a proposed parent primitive lattice is mapped to the conventional setting,
  projected into :class:`wqcodiff.charts.LatticeChartCodec`, and mapped back;
* all discrete WQ fields remain immutable.

There is deliberately no Torch, training, MLIP, retry, topology revision, or
model-selection code here.  A later A800 runner may copy one proposed parent
step to NumPy, call this projector, and copy the single projected state back.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..charts import LatticeChartCodec
from ..runtime import ExpandedState, expand_state, project_atom_scores
from ..state import OrbitState, StratifiedState


ORBIT_REGULARIZATION = 1.0e-8
LATTICE_REGULARIZATION = 1.0e-8
DEFAULT_LATTICE_FINITE_DIFFERENCE_STEP = 1.0e-6
DEFAULT_RANK_RELATIVE_TOLERANCE = 1.0e-10
DEFAULT_MINIMUM_VOLUME = 1.0e-10
DEFAULT_MAXIMUM_CONDITION_NUMBER = 1.0e12
DEFAULT_LATTICE_FRAME_TOLERANCE = 2.0e-6


class TangentProjectionError(RuntimeError):
    """A terminal, non-retryable projection-contract failure."""


def _readonly_f64(
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


def _matrix_rank(
    design: np.ndarray,
    *,
    relative_tolerance: float,
) -> int:
    singular = np.linalg.svd(design, compute_uv=False)
    if singular.size == 0:
        return 0
    tolerance = (
        max(design.shape)
        * float(singular[0])
        * float(relative_tolerance)
    )
    return int(np.sum(singular > tolerance))


def _minimum_image_delta(proposed: np.ndarray, current: np.ndarray) -> np.ndarray:
    delta = np.asarray(proposed, dtype=np.float64) - np.asarray(
        current, dtype=np.float64
    )
    delta -= np.rint(delta)
    return delta


def _discrete_signature(state: StratifiedState) -> tuple[Any, ...]:
    return (
        int(state.space_group),
        str(state.lattice_system),
        bool(state.space_group_committed),
        tuple(
            (
                orbit.orbit_id,
                int(orbit.wyckoff_type),
                int(orbit.species),
                int(orbit.multiplicity),
                int(orbit.primitive_multiplicity),
                int(orbit.chart_dimension),
            )
            for orbit in state.orbits
        ),
        int(state.atom_count),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class OrbitTangentAudit:
    orbit_id: str
    orbit_index: int
    chart_dimension: int
    primitive_atom_count: int
    jacobian_rank: int
    input_update_norm: float
    chart_update_norm: float
    tangent_update_norm: float
    normal_residual_norm: float
    fixed_site_drift: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class LatticeProjectionAudit:
    projection_method: str
    lattice_system: str
    chart_dimension: int
    jacobian_rank: int
    finite_difference_step: float
    regularization: float
    input_update_norm: float
    chart_update_norm: float
    tangent_update_norm: float
    retracted_update_norm: float
    normal_residual_norm: float
    conventional_volume_before: float
    conventional_volume_after: float
    condition_number_after: float
    chart_roundtrip_max_abs_error: float
    lattice_roundtrip_max_abs_error: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class LatticeProjectionResult:
    lattice_chart: tuple[float, ...]
    conventional_lattice: np.ndarray
    audit: LatticeProjectionAudit

    def __post_init__(self) -> None:
        matrix = _readonly_f64(
            self.conventional_lattice,
            name="projected conventional lattice",
            shape=(3, 3),
        )
        object.__setattr__(self, "conventional_lattice", matrix)


@dataclasses.dataclass(frozen=True, slots=True)
class TangentStepAudit:
    schema: str
    topology_hash_before: str
    topology_hash_after: str
    discrete_fields_unchanged: bool
    atom_count: int
    orbit_regularization: float
    maximum_orbit_normal_residual_norm: float
    orbit_chart_roundtrip_checked: bool
    maximum_orbit_chart_roundtrip_max_abs_error: float
    primitive_transform_consistency_max_abs_error: float
    primitive_lattice_consistency_max_abs_error: float
    primitive_lattice_consistency_relative_error: float
    primitive_lattice_scale: float
    primitive_volume_after: float
    retry_or_replacement_used: bool
    topology_revision_used: bool
    orbit_audits: tuple[OrbitTangentAudit, ...]
    lattice_audit: LatticeProjectionAudit

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["orbit_audits"] = [
            value.to_dict() for value in self.orbit_audits
        ]
        payload["lattice_audit"] = self.lattice_audit.to_dict()
        return payload


@dataclasses.dataclass(frozen=True, slots=True)
class TangentStepResult:
    state: StratifiedState
    projected_fractional_coordinates: np.ndarray
    projected_primitive_lattice: np.ndarray
    audit: TangentStepAudit

    def __post_init__(self) -> None:
        coordinates = _readonly_f64(
            self.projected_fractional_coordinates,
            name="projected primitive fractional coordinates",
            shape=(None, 3),
        )
        lattice = _readonly_f64(
            self.projected_primitive_lattice,
            name="projected primitive lattice",
            shape=(3, 3),
        )
        object.__setattr__(self, "projected_fractional_coordinates", coordinates)
        object.__setattr__(self, "projected_primitive_lattice", lattice)


@dataclasses.dataclass(frozen=True, slots=True)
class TangentNoiseSubspaceAudit:
    field_id: str
    ambient_dimension: int
    tangent_dimension: int
    jacobian_rank: int
    energy_scale: float
    input_noise_norm: float
    output_noise_norm: float
    orthonormality_max_abs_error: float
    tangent_residual_norm: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class ManifoldNoiseAudit:
    schema: str
    distribution_claim: str
    paired_parent_noise_transform: bool
    parent_schedule_changed: bool
    fixed_site_noise_zero: bool
    coordinate_subspaces: tuple[TangentNoiseSubspaceAudit, ...]
    lattice_subspace: TangentNoiseSubspaceAudit

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["coordinate_subspaces"] = [
            value.to_dict() for value in self.coordinate_subspaces
        ]
        payload["lattice_subspace"] = self.lattice_subspace.to_dict()
        return payload


@dataclasses.dataclass(frozen=True, slots=True)
class ManifoldNoiseResult:
    coordinate_noise: np.ndarray
    primitive_lattice_noise: np.ndarray
    orbit_chart_noise: tuple[np.ndarray, ...]
    lattice_chart_noise: np.ndarray
    audit: ManifoldNoiseAudit

    def __post_init__(self) -> None:
        coordinates = _readonly_f64(
            self.coordinate_noise,
            name="manifold primitive coordinate noise",
            shape=(None, 3),
        )
        lattice = _readonly_f64(
            self.primitive_lattice_noise,
            name="manifold primitive lattice noise",
            shape=(3, 3),
        )
        orbit_chart_noise: list[np.ndarray] = []
        for index, value in enumerate(self.orbit_chart_noise):
            array = _readonly_f64(
                value,
                name=f"manifold orbit {index} chart noise",
                shape=(None,),
            )
            orbit_chart_noise.append(array)
        lattice_chart_noise = _readonly_f64(
            self.lattice_chart_noise,
            name="manifold lattice chart noise",
            shape=(None,),
        )
        object.__setattr__(self, "coordinate_noise", coordinates)
        object.__setattr__(self, "primitive_lattice_noise", lattice)
        object.__setattr__(
            self,
            "orbit_chart_noise",
            tuple(orbit_chart_noise),
        )
        object.__setattr__(
            self,
            "lattice_chart_noise",
            lattice_chart_noise,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ParentLatticeFrameAudit:
    schema: str
    alignment_max_abs_error: float
    orthogonality_max_abs_error: float
    determinant: float
    wq_roundtrip_max_abs_error: float
    parent_roundtrip_max_abs_error: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class ParentLatticeFrame:
    """Rigid right-frame map between WQ and parent lattice matrices."""

    wq_reference_lattice: np.ndarray
    parent_reference_lattice: np.ndarray
    wq_to_parent_right: np.ndarray
    audit: ParentLatticeFrameAudit

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "wq_reference_lattice",
            _readonly_f64(
                self.wq_reference_lattice,
                name="WQ reference primitive lattice",
                shape=(3, 3),
            ),
        )
        object.__setattr__(
            self,
            "parent_reference_lattice",
            _readonly_f64(
                self.parent_reference_lattice,
                name="parent reference primitive lattice",
                shape=(3, 3),
            ),
        )
        object.__setattr__(
            self,
            "wq_to_parent_right",
            _readonly_f64(
                self.wq_to_parent_right,
                name="WQ-to-parent right frame",
                shape=(3, 3),
            ),
        )

    @classmethod
    def from_reference_lattices(
        cls,
        *,
        wq_reference_lattice: Any,
        parent_reference_lattice: Any,
        tolerance: float = DEFAULT_LATTICE_FRAME_TOLERANCE,
    ) -> "ParentLatticeFrame":
        wq = _readonly_f64(
            wq_reference_lattice,
            name="WQ reference primitive lattice",
            shape=(3, 3),
        )
        parent = _readonly_f64(
            parent_reference_lattice,
            name="parent reference primitive lattice",
            shape=(3, 3),
        )
        if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
            raise ValueError("lattice-frame tolerance must be positive")
        try:
            right = np.linalg.solve(wq, parent)
            inverse = np.linalg.inv(right)
        except np.linalg.LinAlgError as exc:
            raise TangentProjectionError(
                "WQ/parent lattice frame is singular"
            ) from exc
        alignment_error = float(
            np.max(np.abs(wq @ right - parent), initial=0.0)
        )
        orthogonality_error = float(
            np.max(
                np.abs(right @ right.T - np.eye(3, dtype=np.float64)),
                initial=0.0,
            )
        )
        determinant = float(np.linalg.det(right))
        wq_roundtrip_error = float(
            np.max(np.abs((wq @ right) @ inverse - wq), initial=0.0)
        )
        parent_roundtrip_error = float(
            np.max(
                np.abs((parent @ inverse) @ right - parent),
                initial=0.0,
            )
        )
        if (
            alignment_error > tolerance
            or orthogonality_error > tolerance
            or abs(abs(determinant) - 1.0) > tolerance
            or wq_roundtrip_error > tolerance
            or parent_roundtrip_error > tolerance
        ):
            raise TangentProjectionError(
                "WQ and parent lattices are not related by one rigid frame"
            )
        return cls(
            wq_reference_lattice=wq,
            parent_reference_lattice=parent,
            wq_to_parent_right=right,
            audit=ParentLatticeFrameAudit(
                schema="wq_parent_lattice_frame_audit_v1",
                alignment_max_abs_error=alignment_error,
                orthogonality_max_abs_error=orthogonality_error,
                determinant=determinant,
                wq_roundtrip_max_abs_error=wq_roundtrip_error,
                parent_roundtrip_max_abs_error=parent_roundtrip_error,
            ),
        )

    def to_parent(self, wq_lattice: Any) -> np.ndarray:
        matrix = _readonly_f64(
            wq_lattice,
            name="WQ-frame primitive lattice",
            shape=(3, 3),
        )
        return _readonly_f64(
            matrix @ self.wq_to_parent_right,
            name="parent-frame primitive lattice",
            shape=(3, 3),
        )

    def to_wq(self, parent_lattice: Any) -> np.ndarray:
        matrix = _readonly_f64(
            parent_lattice,
            name="parent-frame primitive lattice",
            shape=(3, 3),
        )
        try:
            result = matrix @ np.linalg.inv(self.wq_to_parent_right)
        except np.linalg.LinAlgError as exc:
            raise TangentProjectionError(
                "WQ-to-parent right frame became singular"
            ) from exc
        return _readonly_f64(
            result,
            name="WQ-frame primitive lattice",
            shape=(3, 3),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ManifoldForwardAudit:
    schema: str
    timestep: int
    alpha_bar: float
    coordinate_sigma: float
    condition_topology_hash: str
    noised_topology_hash: str
    discrete_fields_unchanged: bool
    coordinate_chart_reconstruction_max_abs_error: float
    lattice_chart_reconstruction_max_abs_error: float
    primitive_volume: float
    parent_schedule_changed: bool
    clean_condition_used_as_reverse_state: bool
    retry_or_replacement_used: bool
    noise_audit: ManifoldNoiseAudit
    lattice_frame_audit: ParentLatticeFrameAudit

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["noise_audit"] = self.noise_audit.to_dict()
        payload["lattice_frame_audit"] = self.lattice_frame_audit.to_dict()
        return payload


@dataclasses.dataclass(frozen=True, slots=True)
class ManifoldForwardState:
    state: StratifiedState
    fractional_coordinates: np.ndarray
    primitive_lattice: np.ndarray
    audit: ManifoldForwardAudit
    lattice_frame: ParentLatticeFrame

    def __post_init__(self) -> None:
        coordinates = _readonly_f64(
            self.fractional_coordinates,
            name="manifold-forward primitive coordinates",
            shape=(None, 3),
        )
        lattice = _readonly_f64(
            self.primitive_lattice,
            name="manifold-forward primitive lattice",
            shape=(3, 3),
        )
        object.__setattr__(self, "fractional_coordinates", coordinates)
        object.__setattr__(self, "primitive_lattice", lattice)


@dataclasses.dataclass(frozen=True, slots=True)
class ReverseInputIdentityAudit:
    schema: str
    topology_hash: str
    atom_count: int
    primitive_coordinate_max_abs_error: float
    primitive_lattice_max_abs_error: float
    batch_parent_lattice_max_abs_error: float
    noised_lattice_frame_roundtrip_max_abs_error: float
    state_timestep_max_abs_error: float
    batch_species_and_order_match: bool
    single_graph_batch_match: bool
    clean_batch_geometry_used_as_reverse_state: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def validate_manifold_reverse_identity(
    *,
    forward_state: ManifoldForwardState,
    projector: "WyckoffTangentProjector",
    batch_num_atoms: Any,
    batch_atomic_numbers: Any,
    batch_fractional_coordinates: Any,
    batch_assignments: Any,
    batch_clean_primitive_lattice: Any,
) -> ReverseInputIdentityAudit:
    """Fail closed unless the noised WQ state and parent batch are one sample.

    The parent batch contains the *clean* geometry used to build decoder graph
    metadata, whereas reverse diffusion must start from ``forward_state``.
    Therefore geometry values are checked against a fresh expansion of the
    noised WQ state, while the batch is checked for graph cardinality, species
    order, coordinate shape, finiteness, and single-graph assignments.
    """

    from .bridge_parity import PARENT_SCHEDULER_TIMESTEPS

    expanded = expand_state(
        forward_state.state,
        projector.catalog,
        redetect_space_group=False,
    )
    topology_hash = forward_state.state.topology_hash()
    if (
        topology_hash != forward_state.audit.condition_topology_hash
        or topology_hash != forward_state.audit.noised_topology_hash
        or not forward_state.audit.discrete_fields_unchanged
        or forward_state.audit.parent_schedule_changed
        or forward_state.audit.clean_condition_used_as_reverse_state
        or forward_state.audit.retry_or_replacement_used
    ):
        raise TangentProjectionError(
            "manifold-forward audit does not identify an unchanged WQ sample"
        )

    coordinate_error = float(
        np.max(
            np.abs(
                expanded.fractional_coordinates
                - forward_state.fractional_coordinates
            ),
            initial=0.0,
        )
    )
    lattice_error = float(
        np.max(
            np.abs(
                expanded.primitive_lattice
                - forward_state.primitive_lattice
            ),
            initial=0.0,
        )
    )
    tolerance = projector.primitive_consistency_tolerance
    if coordinate_error > tolerance or lattice_error > tolerance:
        raise TangentProjectionError(
            "stored manifold-forward geometry disagrees with WQ expansion"
        )
    batch_parent_lattice = _readonly_f64(
        batch_clean_primitive_lattice,
        name="parent batch clean primitive lattice",
        shape=(3, 3),
    )
    batch_parent_lattice_error = float(
        np.max(
            np.abs(
                batch_parent_lattice
                - forward_state.lattice_frame.parent_reference_lattice
            ),
            initial=0.0,
        )
    )
    noised_lattice_frame_roundtrip_error = float(
        np.max(
            np.abs(
                forward_state.lattice_frame.to_wq(
                    forward_state.lattice_frame.to_parent(
                        forward_state.primitive_lattice
                    )
                )
                - forward_state.primitive_lattice
            ),
            initial=0.0,
        )
    )
    if (
        batch_parent_lattice_error > DEFAULT_LATTICE_FRAME_TOLERANCE
        or noised_lattice_frame_roundtrip_error
        > DEFAULT_LATTICE_FRAME_TOLERANCE
    ):
        raise TangentProjectionError(
            "parent lattice frame disagrees with the CrysLLMGen batch"
        )

    expected_timestep = (
        forward_state.audit.timestep / PARENT_SCHEDULER_TIMESTEPS
    )
    timestep_error = abs(
        float(forward_state.state.timestep) - expected_timestep
    )
    if timestep_error > 1.0e-12:
        raise TangentProjectionError(
            "manifold-forward state timestep disagrees with its audit"
        )

    num_atoms = np.asarray(batch_num_atoms, dtype=np.int64).reshape(-1)
    atomic_numbers = np.asarray(
        batch_atomic_numbers,
        dtype=np.int64,
    ).reshape(-1)
    batch_coordinates = np.asarray(
        batch_fractional_coordinates,
        dtype=np.float64,
    )
    assignments = np.asarray(batch_assignments, dtype=np.int64).reshape(-1)
    atom_count = expanded.atom_count
    single_graph_match = bool(
        num_atoms.shape == (1,)
        and int(num_atoms[0]) == atom_count
        and assignments.shape == (atom_count,)
        and np.array_equal(assignments, np.zeros(atom_count, dtype=np.int64))
    )
    species_match = bool(
        atomic_numbers.shape == (atom_count,)
        and np.array_equal(atomic_numbers, expanded.atomic_numbers)
    )
    if batch_coordinates.shape != (atom_count, 3):
        raise TangentProjectionError(
            "parent batch coordinate shape disagrees with WQ expansion"
        )
    if not np.isfinite(batch_coordinates).all():
        raise FloatingPointError(
            "parent batch contains non-finite clean coordinates"
        )
    if not single_graph_match or not species_match:
        raise TangentProjectionError(
            "parent batch graph/species identity disagrees with WQ expansion"
        )

    return ReverseInputIdentityAudit(
        schema="wq_tangent_reverse_input_identity_audit_v1",
        topology_hash=topology_hash,
        atom_count=atom_count,
        primitive_coordinate_max_abs_error=coordinate_error,
        primitive_lattice_max_abs_error=lattice_error,
        batch_parent_lattice_max_abs_error=batch_parent_lattice_error,
        noised_lattice_frame_roundtrip_max_abs_error=(
            noised_lattice_frame_roundtrip_error
        ),
        state_timestep_max_abs_error=timestep_error,
        batch_species_and_order_match=species_match,
        single_graph_batch_match=single_graph_match,
        clean_batch_geometry_used_as_reverse_state=False,
    )


def _project_gaussian_to_tangent(
    *,
    field_id: str,
    design: np.ndarray,
    base_noise: np.ndarray,
    tangent_dimension: int,
    rank_relative_tolerance: float,
) -> tuple[np.ndarray, TangentNoiseSubspaceAudit]:
    """Project paired iid noise and match its ambient expected squared norm.

    If ``e ~ N(0, I_n)`` and ``U`` is an orthonormal tangent basis, then
    ``U U^T e`` has expected squared norm ``d`` rather than the parent's
    ambient ``n``.  Multiplying by ``sqrt(n/d)`` preserves the parent's
    per-field expected noise energy without leaving the tangent subspace.
    Fixed sites (``d=0``) receive exactly zero noise.
    """

    matrix = np.asarray(design, dtype=np.float64)
    vector = np.asarray(base_noise, dtype=np.float64).reshape(-1)
    if matrix.shape != (vector.size, tangent_dimension):
        raise ValueError(
            f"{field_id} tangent design {matrix.shape} differs from "
            f"({vector.size}, {tangent_dimension})"
        )
    if not np.isfinite(matrix).all() or not np.isfinite(vector).all():
        raise FloatingPointError(f"{field_id} tangent noise input is non-finite")
    if tangent_dimension == 0:
        output = np.zeros_like(vector)
        audit = TangentNoiseSubspaceAudit(
            field_id=field_id,
            ambient_dimension=vector.size,
            tangent_dimension=0,
            jacobian_rank=0,
            energy_scale=0.0,
            input_noise_norm=float(np.linalg.norm(vector)),
            output_noise_norm=0.0,
            orthonormality_max_abs_error=0.0,
            tangent_residual_norm=0.0,
        )
        return output, audit

    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    tolerance = (
        max(matrix.shape)
        * float(singular[0])
        * float(rank_relative_tolerance)
    )
    rank = int(np.sum(singular > tolerance))
    if rank != tangent_dimension:
        raise TangentProjectionError(
            f"{field_id} Jacobian rank {rank} != {tangent_dimension}"
        )
    basis = left[:, :tangent_dimension]
    orthonormality_error = float(
        np.max(
            np.abs(
                basis.T @ basis
                - np.eye(tangent_dimension, dtype=np.float64)
            ),
            initial=0.0,
        )
    )
    scale = math.sqrt(vector.size / tangent_dimension)
    output = scale * basis @ (basis.T @ vector)
    tangent_residual = float(
        np.linalg.norm(output - basis @ (basis.T @ output))
    )
    if not np.isfinite(output).all():
        raise FloatingPointError(f"{field_id} projected noise is non-finite")
    audit = TangentNoiseSubspaceAudit(
        field_id=field_id,
        ambient_dimension=vector.size,
        tangent_dimension=tangent_dimension,
        jacobian_rank=rank,
        energy_scale=scale,
        input_noise_norm=float(np.linalg.norm(vector)),
        output_noise_norm=float(np.linalg.norm(output)),
        orthonormality_max_abs_error=orthonormality_error,
        tangent_residual_norm=tangent_residual,
    )
    return output, audit


class LatticeChartProjector:
    """Pull one full conventional-lattice proposal into a registered chart."""

    def __init__(
        self,
        *,
        finite_difference_step: float = DEFAULT_LATTICE_FINITE_DIFFERENCE_STEP,
        regularization: float = LATTICE_REGULARIZATION,
        rank_relative_tolerance: float = DEFAULT_RANK_RELATIVE_TOLERANCE,
        minimum_volume: float = DEFAULT_MINIMUM_VOLUME,
        maximum_condition_number: float = DEFAULT_MAXIMUM_CONDITION_NUMBER,
    ) -> None:
        for name, value in (
            ("finite_difference_step", finite_difference_step),
            ("regularization", regularization),
            ("rank_relative_tolerance", rank_relative_tolerance),
            ("minimum_volume", minimum_volume),
            ("maximum_condition_number", maximum_condition_number),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.finite_difference_step = float(finite_difference_step)
        self.regularization = float(regularization)
        self.rank_relative_tolerance = float(rank_relative_tolerance)
        self.minimum_volume = float(minimum_volume)
        self.maximum_condition_number = float(maximum_condition_number)

    def jacobian(
        self,
        lattice_chart: tuple[float, ...],
        lattice_system: str,
    ) -> np.ndarray:
        dimension = LatticeChartCodec.dimension(lattice_system)
        if len(lattice_chart) != dimension:
            raise ValueError("lattice chart dimension differs from crystal system")
        jacobian = np.empty((9, dimension), dtype=np.float64)
        for axis in range(dimension):
            plus = list(lattice_chart)
            minus = list(lattice_chart)
            plus[axis] += self.finite_difference_step
            minus[axis] -= self.finite_difference_step
            decoded_plus = LatticeChartCodec.decode_matrix(plus, lattice_system)
            decoded_minus = LatticeChartCodec.decode_matrix(minus, lattice_system)
            jacobian[:, axis] = (
                (decoded_plus - decoded_minus)
                / (2.0 * self.finite_difference_step)
            ).reshape(-1)
        if not np.isfinite(jacobian).all():
            raise FloatingPointError("lattice chart Jacobian is non-finite")
        return jacobian

    def project(
        self,
        *,
        lattice_chart: tuple[float, ...],
        lattice_system: str,
        proposed_conventional_lattice: Any,
    ) -> LatticeProjectionResult:
        """Project one absolute proposed lattice without mutating either input."""

        chart = tuple(float(value) for value in lattice_chart)
        if not all(math.isfinite(value) for value in chart):
            raise FloatingPointError("current lattice chart is non-finite")
        proposed = _readonly_f64(
            proposed_conventional_lattice,
            name="proposed conventional lattice",
            shape=(3, 3),
        )
        current = LatticeChartCodec.decode_matrix(chart, lattice_system)
        current_volume = abs(float(np.linalg.det(current)))
        if current_volume <= self.minimum_volume:
            raise TangentProjectionError("current conventional lattice is degenerate")

        design = self.jacobian(chart, lattice_system)
        dimension = design.shape[1]
        rank = _matrix_rank(
            design,
            relative_tolerance=self.rank_relative_tolerance,
        )
        if rank != dimension:
            raise TangentProjectionError(
                f"lattice chart Jacobian rank {rank} != {dimension}"
            )

        requested_update = (proposed - current).reshape(-1)
        try:
            proposed_chart = tuple(
                float(value)
                for value in LatticeChartCodec.encode_matrix(
                    proposed,
                    lattice_system,
                )
            )
        except Exception as exc:
            raise TangentProjectionError(
                "proposed conventional lattice has no valid registered "
                "chart retraction"
            ) from exc
        if len(proposed_chart) != dimension or not all(
            math.isfinite(value) for value in proposed_chart
        ):
            raise TangentProjectionError(
                "proposed lattice chart dimension or finiteness changed"
            )
        chart_update = np.asarray(proposed_chart, dtype=np.float64) - np.asarray(
            chart,
            dtype=np.float64,
        )
        if not np.isfinite(chart_update).all():
            raise FloatingPointError("lattice chart update is non-finite")
        linearized_update = design @ chart_update
        updated_chart = proposed_chart
        projected = LatticeChartCodec.decode_matrix(
            updated_chart,
            lattice_system,
        )
        retracted_update = (projected - current).reshape(-1)
        residual = requested_update - retracted_update
        volume = abs(float(np.linalg.det(projected)))
        condition = float(np.linalg.cond(projected))
        if volume <= self.minimum_volume:
            raise TangentProjectionError("projected conventional lattice is degenerate")
        if (
            not math.isfinite(condition)
            or condition > self.maximum_condition_number
        ):
            raise TangentProjectionError(
                "projected conventional lattice is ill-conditioned"
            )

        encoded = LatticeChartCodec.encode_matrix(projected, lattice_system)
        if len(encoded) != dimension:
            raise TangentProjectionError("projected lattice chart dimension changed")
        chart_roundtrip = max(
            (abs(left - right) for left, right in zip(encoded, updated_chart)),
            default=0.0,
        )
        decoded_roundtrip = LatticeChartCodec.decode_matrix(
            encoded,
            lattice_system,
        )
        lattice_roundtrip = float(
            np.max(np.abs(decoded_roundtrip - projected), initial=0.0)
        )
        if not all(
            math.isfinite(value)
            for value in (chart_roundtrip, lattice_roundtrip)
        ):
            raise FloatingPointError("lattice round-trip audit is non-finite")

        audit = LatticeProjectionAudit(
            projection_method="global_chart_retraction_v1",
            lattice_system=lattice_system,
            chart_dimension=dimension,
            jacobian_rank=rank,
            finite_difference_step=self.finite_difference_step,
            regularization=self.regularization,
            input_update_norm=float(np.linalg.norm(requested_update)),
            chart_update_norm=float(np.linalg.norm(chart_update)),
            tangent_update_norm=float(np.linalg.norm(linearized_update)),
            retracted_update_norm=float(np.linalg.norm(retracted_update)),
            normal_residual_norm=float(np.linalg.norm(residual)),
            conventional_volume_before=current_volume,
            conventional_volume_after=volume,
            condition_number_after=condition,
            chart_roundtrip_max_abs_error=float(chart_roundtrip),
            lattice_roundtrip_max_abs_error=lattice_roundtrip,
        )
        return LatticeProjectionResult(
            lattice_chart=updated_chart,
            conventional_lattice=projected,
            audit=audit,
        )


class ManifoldNoiseProjector:
    """Transform paired parent iid noise into WQ tangent noise.

    The transformation is deterministic and contains no random-number
    generator.  A caller supplies the exact paired parent noise that arm U
    would use; arm T receives its registered orthogonal tangent transform.
    """

    def __init__(
        self,
        catalog: Any,
        *,
        lattice_projector: LatticeChartProjector | None = None,
        rank_relative_tolerance: float = DEFAULT_RANK_RELATIVE_TOLERANCE,
    ) -> None:
        if (
            not math.isfinite(float(rank_relative_tolerance))
            or float(rank_relative_tolerance) <= 0.0
        ):
            raise ValueError("rank_relative_tolerance must be finite and positive")
        self.catalog = catalog
        self.lattice_projector = lattice_projector or LatticeChartProjector()
        self.rank_relative_tolerance = float(rank_relative_tolerance)

    def project_parent_noise(
        self,
        *,
        state: StratifiedState,
        parent_coordinate_noise: Any,
        parent_primitive_lattice_noise: Any,
        lattice_frame: ParentLatticeFrame | None = None,
    ) -> ManifoldNoiseResult:
        """Project exact arm-U base noise without changing schedule scalars."""

        expanded = expand_state(
            state,
            self.catalog,
            redetect_space_group=False,
        )
        frame = lattice_frame or ParentLatticeFrame.from_reference_lattices(
            wq_reference_lattice=expanded.primitive_lattice,
            parent_reference_lattice=expanded.primitive_lattice,
        )
        frame_reference_error = float(
            np.max(
                np.abs(
                    frame.wq_reference_lattice
                    - expanded.primitive_lattice
                ),
                initial=0.0,
            )
        )
        if frame_reference_error > DEFAULT_LATTICE_FRAME_TOLERANCE:
            raise TangentProjectionError(
                "lattice frame belongs to a different clean WQ state"
            )
        coordinate_input = _readonly_f64(
            parent_coordinate_noise,
            name="paired parent coordinate noise",
            shape=tuple(expanded.fractional_coordinates.shape),
        )
        lattice_input = _readonly_f64(
            parent_primitive_lattice_noise,
            name="paired parent primitive lattice noise",
            shape=(3, 3),
        )

        coordinate_output = np.zeros_like(coordinate_input)
        coordinate_audits: list[TangentNoiseSubspaceAudit] = []
        fixed_site_noise_zero = True
        for orbit_index, (orbit, jacobian) in enumerate(
            zip(state.orbits, expanded.orbit_jacobians)
        ):
            selected = expanded.atom_to_orbit == orbit_index
            vector = coordinate_input[selected].reshape(-1)
            raw_jacobian = np.asarray(jacobian, dtype=np.float64)
            design = (
                raw_jacobian.reshape(-1, orbit.chart_dimension)
                if orbit.chart_dimension
                else np.zeros((vector.size, 0), dtype=np.float64)
            )
            projected, audit = _project_gaussian_to_tangent(
                field_id=f"orbit:{orbit.orbit_id}",
                design=design,
                base_noise=vector,
                tangent_dimension=orbit.chart_dimension,
                rank_relative_tolerance=self.rank_relative_tolerance,
            )
            coordinate_output[selected] = projected.reshape(-1, 3)
            coordinate_audits.append(audit)
            if orbit.chart_dimension == 0 and np.any(projected != 0.0):
                fixed_site_noise_zero = False
        orbit_chart_noise = project_atom_scores(
            expanded,
            coordinate_output,
        )

        try:
            primitive_transform = (
                expanded.primitive_lattice
                @ np.linalg.inv(expanded.conventional_lattice)
            )
        except np.linalg.LinAlgError as exc:
            raise TangentProjectionError(
                "primitive/conventional lattice transform is singular"
            ) from exc
        conventional_design = self.lattice_projector.jacobian(
            state.lattice_chart,
            state.lattice_system,
        )
        lattice_dimension = conventional_design.shape[1]
        primitive_design = np.empty_like(conventional_design)
        for axis in range(lattice_dimension):
            derivative_wq = (
                primitive_transform
                @ conventional_design[:, axis].reshape(3, 3)
            )
            primitive_design[:, axis] = (
                derivative_wq @ frame.wq_to_parent_right
            ).reshape(-1)
        lattice_output_flat, lattice_audit = _project_gaussian_to_tangent(
            field_id="parent_primitive_lattice",
            design=primitive_design,
            base_noise=lattice_input.reshape(-1),
            tangent_dimension=lattice_dimension,
            rank_relative_tolerance=self.rank_relative_tolerance,
        )
        lattice_gram = (
            primitive_design.T @ primitive_design
            + LATTICE_REGULARIZATION
            * np.eye(lattice_dimension, dtype=np.float64)
        )
        lattice_chart_noise = np.linalg.solve(
            lattice_gram,
            primitive_design.T @ lattice_output_flat,
        )
        if not np.isfinite(lattice_chart_noise).all():
            raise FloatingPointError("lattice chart noise is non-finite")
        audit = ManifoldNoiseAudit(
            schema="wq_manifold_noise_audit_v1",
            distribution_claim=(
                "parent_schedule_manifold_restricted_not_parent_iid"
            ),
            paired_parent_noise_transform=True,
            parent_schedule_changed=False,
            fixed_site_noise_zero=fixed_site_noise_zero,
            coordinate_subspaces=tuple(coordinate_audits),
            lattice_subspace=lattice_audit,
        )
        return ManifoldNoiseResult(
            coordinate_noise=coordinate_output,
            primitive_lattice_noise=lattice_output_flat.reshape(3, 3),
            orbit_chart_noise=orbit_chart_noise,
            lattice_chart_noise=lattice_chart_noise,
            audit=audit,
        )


def forward_noise_manifold_numpy(
    *,
    state: StratifiedState,
    catalog: Any,
    schedules: Any,
    cell: Any,
    noise_projector: ManifoldNoiseProjector | None = None,
    parent_clean_primitive_lattice: Any | None = None,
) -> ManifoldForwardState:
    """Build arm-T's chart-space forward state from arm-U's paired base noise.

    The scalar parent schedules and exact ``BridgeCell`` identity are reused.
    The distribution is intentionally chart-space/manifold-restricted:

    ``q_t = q_0 + sigma_t * eps_q`` (periodic)

    ``z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1-alpha_bar_t) * eps_z``

    This is not represented as parent iid full-atom/full-lattice noise, and
    the audit states that limitation explicitly.
    """

    from .bridge_parity import (  # Local import avoids a module-level cycle.
        BRIDGE_TIMESTEPS,
        PARENT_SCHEDULER_TIMESTEPS,
        paired_standard_normal,
    )

    timestep = int(cell.timestep)
    if timestep not in BRIDGE_TIMESTEPS:
        raise ValueError("manifold forward timestep is outside the frozen matrix")
    if len(schedules.alphas_cumprod) != PARENT_SCHEDULER_TIMESTEPS + 1:
        raise ValueError("parent alpha schedule length is not 1001")
    if len(schedules.coordinate_sigmas) != PARENT_SCHEDULER_TIMESTEPS + 1:
        raise ValueError("parent coordinate schedule length is not 1001")

    state_snapshot = state.to_dict()
    topology_before = state.topology_hash()
    signature_before = _discrete_signature(state)
    expanded = expand_state(state, catalog, redetect_space_group=False)
    parent_clean_lattice = (
        expanded.primitive_lattice
        if parent_clean_primitive_lattice is None
        else parent_clean_primitive_lattice
    )
    lattice_frame = ParentLatticeFrame.from_reference_lattices(
        wq_reference_lattice=expanded.primitive_lattice,
        parent_reference_lattice=parent_clean_lattice,
    )
    parent_coordinate_noise = paired_standard_normal(
        seed=int(cell.forward_noise_seed),
        shape=tuple(expanded.fractional_coordinates.shape),
        channel="coordinate",
    )
    parent_lattice_noise = paired_standard_normal(
        seed=int(cell.forward_noise_seed),
        shape=(3, 3),
        channel="lattice",
    )
    projector = noise_projector or ManifoldNoiseProjector(catalog)
    manifold_noise = projector.project_parent_noise(
        state=state,
        parent_coordinate_noise=parent_coordinate_noise,
        parent_primitive_lattice_noise=parent_lattice_noise,
        lattice_frame=lattice_frame,
    )

    alpha_bar = float(schedules.alphas_cumprod[timestep])
    coordinate_sigma = float(schedules.coordinate_sigmas[timestep])
    if not 0.0 < alpha_bar <= 1.0 or coordinate_sigma <= 0.0:
        raise ValueError("invalid parent schedule scalar")
    noised_orbits: list[OrbitState] = []
    coordinate_reconstruction_error = 0.0
    for orbit, chart_noise in zip(
        state.orbits,
        manifold_noise.orbit_chart_noise,
    ):
        if len(chart_noise) != orbit.chart_dimension:
            raise TangentProjectionError(
                f"orbit {orbit.orbit_id} chart-noise dimension changed"
            )
        noised_coordinate = tuple(
            float((value + coordinate_sigma * noise) % 1.0)
            for value, noise in zip(
                orbit.free_coordinate,
                chart_noise,
            )
        )
        reconstructed = tuple(
            float((value - coordinate_sigma * noise) % 1.0)
            for value, noise in zip(noised_coordinate, chart_noise)
        )
        coordinate_reconstruction_error = max(
            coordinate_reconstruction_error,
            max(
                (
                    abs(((left - right + 0.5) % 1.0) - 0.5)
                    for left, right in zip(
                        reconstructed,
                        orbit.free_coordinate,
                    )
                ),
                default=0.0,
            ),
        )
        noised_orbits.append(
            dataclasses.replace(
                orbit,
                free_coordinate=noised_coordinate,
            )
        )

    sqrt_alpha = math.sqrt(alpha_bar)
    sqrt_one_minus_alpha = math.sqrt(1.0 - alpha_bar)
    noised_lattice_chart = tuple(
        float(
            sqrt_alpha * value
            + sqrt_one_minus_alpha * noise
        )
        for value, noise in zip(
            state.lattice_chart,
            manifold_noise.lattice_chart_noise,
        )
    )
    reconstructed_lattice_chart = tuple(
        float(
            (value - sqrt_one_minus_alpha * noise)
            / sqrt_alpha
        )
        for value, noise in zip(
            noised_lattice_chart,
            manifold_noise.lattice_chart_noise,
        )
    )
    lattice_reconstruction_error = max(
        (
            abs(left - right)
            for left, right in zip(
                reconstructed_lattice_chart,
                state.lattice_chart,
            )
        ),
        default=0.0,
    )
    noised_state = dataclasses.replace(
        state,
        lattice_chart=noised_lattice_chart,
        orbits=tuple(noised_orbits),
        timestep=timestep / PARENT_SCHEDULER_TIMESTEPS,
    )
    if state.to_dict() != state_snapshot:
        raise TangentProjectionError("manifold forward process mutated its input")
    topology_after = noised_state.topology_hash()
    discrete_unchanged = _discrete_signature(noised_state) == signature_before
    if topology_after != topology_before or not discrete_unchanged:
        raise TangentProjectionError(
            "manifold forward process changed a discrete WQ field"
        )
    expanded_noised = expand_state(
        noised_state,
        catalog,
        redetect_space_group=False,
    )
    primitive_volume = abs(float(np.linalg.det(expanded_noised.primitive_lattice)))
    if primitive_volume <= DEFAULT_MINIMUM_VOLUME:
        raise TangentProjectionError(
            "manifold-forward primitive lattice is degenerate"
        )
    audit = ManifoldForwardAudit(
        schema="wq_manifold_forward_audit_v1",
        timestep=timestep,
        alpha_bar=alpha_bar,
        coordinate_sigma=coordinate_sigma,
        condition_topology_hash=topology_before,
        noised_topology_hash=topology_after,
        discrete_fields_unchanged=discrete_unchanged,
        coordinate_chart_reconstruction_max_abs_error=float(
            coordinate_reconstruction_error
        ),
        lattice_chart_reconstruction_max_abs_error=float(
            lattice_reconstruction_error
        ),
        primitive_volume=primitive_volume,
        parent_schedule_changed=False,
        clean_condition_used_as_reverse_state=False,
        retry_or_replacement_used=False,
        noise_audit=manifold_noise.audit,
        lattice_frame_audit=lattice_frame.audit,
    )
    return ManifoldForwardState(
        state=noised_state,
        fractional_coordinates=expanded_noised.fractional_coordinates,
        primitive_lattice=expanded_noised.primitive_lattice,
        audit=audit,
        lattice_frame=lattice_frame,
    )


def run_parent_reverse_on_wyckoff_manifold(
    *,
    model: Any,
    batch: Any,
    forward_state: ManifoldForwardState,
    projector: "WyckoffTangentProjector",
    reverse_steps: int = 32,
    corrector_step_lr: float = 1.0e-5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the released CrysLLMGen update with projection after every substep.

    The parent decoder, time embedding, scheduler tensors, corrector/predictor
    formula, decoder call count, and Torch random stream mirror
    :func:`bridge_parity.run_parent_reverse_from_noisy_state`.  The only new
    operation is a deterministic WQ projection after the corrector coordinate
    proposal and after the predictor coordinate/lattice proposal.

    Torch is imported lazily so local mathematical tests remain independent of
    the A800 environment.  Projection is intentionally CPU/NumPy in this first
    diagnostic implementation; it reuses the existing PyXtal/runtime charts
    rather than introducing a second Torch geometry representation.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - A800-only execution path.
        raise RuntimeError(
            "torch is required for the A800 tangent bridge"
        ) from exc
    from .bridge_parity import (
        PARENT_SCHEDULER_TIMESTEPS,
        lattice_valid_torch,
        respaced_timesteps,
    )

    data_utils = __import__(
        "models_ddpm.data_utils",
        fromlist=["lattice_params_to_matrix_torch"],
    )
    clean_parent_lattice = data_utils.lattice_params_to_matrix_torch(
        batch.lengths,
        batch.angles,
    )
    identity_audit = validate_manifold_reverse_identity(
        forward_state=forward_state,
        projector=projector,
        batch_num_atoms=batch.num_atoms.detach().cpu().numpy(),
        batch_atomic_numbers=batch.atom_types.detach().cpu().numpy(),
        batch_fractional_coordinates=(
            batch.frac_coords.detach().cpu().numpy()
        ),
        batch_assignments=batch.batch.detach().cpu().numpy(),
        batch_clean_primitive_lattice=(
            clean_parent_lattice[0].detach().double().cpu().numpy()
        ),
    )
    timestep = int(forward_state.audit.timestep)
    grid = respaced_timesteps(timestep, steps=reverse_steps)
    device = batch.frac_coords.device
    coordinate_dtype = batch.frac_coords.dtype
    lattice_dtype = batch.lengths.dtype
    x = torch.tensor(
        np.array(forward_state.fractional_coordinates, copy=True),
        dtype=coordinate_dtype,
        device=device,
    )
    lattice = torch.tensor(
        np.array(
            forward_state.lattice_frame.to_parent(
                forward_state.primitive_lattice
            ),
            copy=True,
        )[None, :, :],
        dtype=lattice_dtype,
        device=device,
    )
    wq_state = forward_state.state
    calls = 0
    first_reverse_lattice_valid: bool | None = None
    projection_records: list[dict[str, Any]] = []
    maximum_orbit_normal_residual = 0.0
    maximum_lattice_normal_residual = 0.0
    maximum_fixed_site_drift = 0.0
    maximum_orbit_chart_roundtrip_error = 0.0
    all_orbit_chart_roundtrips_checked = True
    maximum_lattice_roundtrip_error = 0.0

    def project_candidate(
        *,
        phase: str,
        step_index: int,
        current_timestep: int,
        following_timestep: int,
        current_state: StratifiedState,
        candidate_coordinates: Any,
        candidate_lattice: Any,
    ) -> tuple[StratifiedState, Any, Any]:
        nonlocal maximum_orbit_normal_residual
        nonlocal maximum_lattice_normal_residual
        nonlocal maximum_fixed_site_drift
        nonlocal maximum_orbit_chart_roundtrip_error
        nonlocal all_orbit_chart_roundtrips_checked
        nonlocal maximum_lattice_roundtrip_error
        result = projector.project_parent_step(
            state=current_state,
            proposed_fractional_coordinates=(
                candidate_coordinates.detach().double().cpu().numpy()
            ),
            proposed_primitive_lattice=(
                forward_state.lattice_frame.to_wq(
                    candidate_lattice[0].detach().double().cpu().numpy()
                )
            ),
        )
        audit = result.audit
        maximum_orbit_normal_residual = max(
            maximum_orbit_normal_residual,
            audit.maximum_orbit_normal_residual_norm,
        )
        maximum_lattice_normal_residual = max(
            maximum_lattice_normal_residual,
            audit.lattice_audit.normal_residual_norm,
        )
        maximum_fixed_site_drift = max(
            maximum_fixed_site_drift,
            max(
                (
                    value.fixed_site_drift
                    for value in audit.orbit_audits
                ),
                default=0.0,
            ),
        )
        maximum_orbit_chart_roundtrip_error = max(
            maximum_orbit_chart_roundtrip_error,
            audit.maximum_orbit_chart_roundtrip_max_abs_error,
        )
        all_orbit_chart_roundtrips_checked = bool(
            all_orbit_chart_roundtrips_checked
            and audit.orbit_chart_roundtrip_checked
        )
        maximum_lattice_roundtrip_error = max(
            maximum_lattice_roundtrip_error,
            audit.lattice_audit.lattice_roundtrip_max_abs_error,
        )
        projection_records.append(
            {
                "schema": "wq_tangent_reverse_projection_record_v1",
                "step_index": step_index,
                "phase": phase,
                "current_timestep": current_timestep,
                "following_timestep": following_timestep,
                "audit": audit.to_dict(),
            }
        )
        projected_x = torch.tensor(
            np.array(result.projected_fractional_coordinates, copy=True),
            dtype=coordinate_dtype,
            device=device,
        )
        projected_lattice = torch.tensor(
            np.array(
                forward_state.lattice_frame.to_parent(
                    result.projected_primitive_lattice
                ),
                copy=True,
            )[None, :, :],
            dtype=lattice_dtype,
            device=device,
        )
        next_timestep = (
            current_timestep
            if phase == "corrector"
            else following_timestep
        )
        projected_state = dataclasses.replace(
            result.state,
            timestep=next_timestep / PARENT_SCHEDULER_TIMESTEPS,
        )
        return projected_state, projected_x, projected_lattice

    with torch.no_grad():
        for index, current in enumerate(grid):
            following = grid[index + 1] if index + 1 < len(grid) else 0
            times = torch.full(
                (1,),
                current,
                device=device,
                dtype=torch.long,
            )
            time_embedding = model.time_embedding(times)
            sigma_x = model.sigma_scheduler.sigmas[current]
            sigma_norm = model.sigma_scheduler.sigmas_norm[current]
            next_sigma_x = model.sigma_scheduler.sigmas[following]

            random_coordinate = (
                torch.randn_like(x)
                if following > 0
                else torch.zeros_like(x)
            )
            corrector_step = corrector_step_lr * (
                sigma_x / model.sigma_scheduler.sigma_begin
            ).square()
            coordinate_noise = torch.sqrt(2.0 * corrector_step)
            _, predicted_coordinate = model.decoder(
                time_embedding,
                batch.atom_types,
                x,
                lattice,
                batch.num_atoms,
                batch.batch,
            )
            calls += 1
            predicted_coordinate = predicted_coordinate * torch.sqrt(
                sigma_norm
            )
            candidate_x_half = (
                x
                - corrector_step * predicted_coordinate
                + coordinate_noise * random_coordinate
            ) % 1.0
            wq_half, x_half, lattice_half = project_candidate(
                phase="corrector",
                step_index=index,
                current_timestep=current,
                following_timestep=following,
                current_state=wq_state,
                candidate_coordinates=candidate_x_half,
                candidate_lattice=lattice,
            )

            predicted_lattice, predicted_coordinate = model.decoder(
                time_embedding,
                batch.atom_types,
                x_half,
                lattice_half,
                batch.num_atoms,
                batch.batch,
            )
            calls += 1
            predicted_coordinate = predicted_coordinate * torch.sqrt(
                sigma_norm
            )
            coordinate_step = sigma_x.square() - next_sigma_x.square()
            coordinate_std = torch.sqrt(
                (
                    next_sigma_x.square()
                    * coordinate_step
                    / sigma_x.square()
                ).clamp_min(0.0)
            )
            random_coordinate = (
                torch.randn_like(x_half)
                if following > 0
                else torch.zeros_like(x_half)
            )
            candidate_x = (
                x_half
                - coordinate_step * predicted_coordinate
                + coordinate_std * random_coordinate
            ) % 1.0

            alpha_bar = model.beta_scheduler.alphas_cumprod[current]
            next_alpha_bar = model.beta_scheduler.alphas_cumprod[following]
            predicted_clean_lattice = (
                lattice_half
                - torch.sqrt(1.0 - alpha_bar) * predicted_lattice
            ) / torch.sqrt(alpha_bar)
            random_lattice = (
                torch.randn_like(lattice_half)
                if following > 0
                else torch.zeros_like(lattice_half)
            )
            candidate_lattice = (
                torch.sqrt(next_alpha_bar) * predicted_clean_lattice
                + torch.sqrt(1.0 - next_alpha_bar) * random_lattice
            )
            wq_state, x, lattice = project_candidate(
                phase="predictor",
                step_index=index,
                current_timestep=current,
                following_timestep=following,
                current_state=wq_half,
                candidate_coordinates=candidate_x,
                candidate_lattice=candidate_lattice,
            )
            if index == 0:
                first_reverse_lattice_valid = lattice_valid_torch(lattice)
            if not torch.isfinite(x).all() or not torch.isfinite(lattice).all():
                raise FloatingPointError(
                    "non-finite Wyckoff-tangent parent state"
                )

    if calls != 2 * reverse_steps or first_reverse_lattice_valid is None:
        raise RuntimeError("Wyckoff-tangent parent call accounting changed")
    if len(projection_records) != 2 * reverse_steps:
        raise RuntimeError("Wyckoff-tangent projection accounting changed")
    return (
        {
            "num_atoms": batch.num_atoms,
            "atom_types": batch.atom_types,
            "frac_coords": x,
            "lattices": lattice,
            "wq_state": wq_state.to_dict(),
        },
        {
            "schema": "wq_tangent_parent_trajectory_v1",
            "decoder_calls": calls,
            "projection_calls": len(projection_records),
            "reverse_steps": reverse_steps,
            "timestep_grid": list(grid),
            "first_reverse_lattice_valid": first_reverse_lattice_valid,
            "all_trajectory_values_finite": True,
            "clean_condition_used_as_reverse_state": False,
            "retry_or_replacement_used": False,
            "topology_revision_used": False,
            "input_identity_verified": True,
            "input_identity_audit": identity_audit.to_dict(),
            "maximum_orbit_normal_residual_norm": (
                maximum_orbit_normal_residual
            ),
            "maximum_lattice_normal_residual_norm": (
                maximum_lattice_normal_residual
            ),
            "all_orbit_chart_roundtrips_checked": (
                all_orbit_chart_roundtrips_checked
            ),
            "maximum_orbit_chart_roundtrip_max_abs_error": (
                maximum_orbit_chart_roundtrip_error
            ),
            "maximum_fixed_site_drift": maximum_fixed_site_drift,
            "maximum_lattice_roundtrip_max_abs_error": (
                maximum_lattice_roundtrip_error
            ),
            "projection_records": projection_records,
        },
    )


class WyckoffTangentProjector:
    """Project one CrysLLMGen primitive-geometry proposal into a fixed WQ state."""

    def __init__(
        self,
        catalog: Any,
        *,
        lattice_projector: LatticeChartProjector | None = None,
        rank_relative_tolerance: float = DEFAULT_RANK_RELATIVE_TOLERANCE,
        primitive_consistency_tolerance: float = 1.0e-8,
        primitive_relative_consistency_tolerance: float = 1.0e-12,
    ) -> None:
        if (
            not math.isfinite(float(rank_relative_tolerance))
            or float(rank_relative_tolerance) <= 0.0
        ):
            raise ValueError("rank_relative_tolerance must be finite and positive")
        if (
            not math.isfinite(float(primitive_consistency_tolerance))
            or float(primitive_consistency_tolerance) <= 0.0
        ):
            raise ValueError(
                "primitive_consistency_tolerance must be finite and positive"
            )
        if (
            not math.isfinite(float(primitive_relative_consistency_tolerance))
            or float(primitive_relative_consistency_tolerance) <= 0.0
        ):
            raise ValueError(
                "primitive_relative_consistency_tolerance must be finite "
                "and positive"
            )
        self.catalog = catalog
        self.lattice_projector = lattice_projector or LatticeChartProjector()
        self.rank_relative_tolerance = float(rank_relative_tolerance)
        self.primitive_consistency_tolerance = float(
            primitive_consistency_tolerance
        )
        self.primitive_relative_consistency_tolerance = float(
            primitive_relative_consistency_tolerance
        )

    def _orbit_chart_roundtrip(
        self,
        state: StratifiedState,
    ) -> tuple[bool, float]:
        decode = getattr(self.catalog, "decode_generator", None)
        encode = getattr(self.catalog, "encode_free", None)
        if not callable(decode) or not callable(encode):
            return False, 0.0
        maximum_error = 0.0
        for orbit in state.orbits:
            generator = decode(
                state.space_group,
                orbit.wyckoff_type,
                orbit.free_coordinate,
            )
            recovered = tuple(
                float(value)
                for value in encode(
                    state.space_group,
                    orbit.wyckoff_type,
                    generator,
                )
            )
            if len(recovered) != orbit.chart_dimension:
                raise TangentProjectionError(
                    f"orbit {orbit.orbit_id} round-trip dimension changed"
                )
            maximum_error = max(
                maximum_error,
                max(
                    (
                        abs(((left - right + 0.5) % 1.0) - 0.5)
                        for left, right in zip(
                            recovered,
                            orbit.free_coordinate,
                        )
                    ),
                    default=0.0,
                ),
            )
        if not math.isfinite(maximum_error):
            raise FloatingPointError("orbit chart round-trip is non-finite")
        return True, float(maximum_error)

    def _orbit_projection(
        self,
        *,
        state: StratifiedState,
        expanded: ExpandedState,
        requested_update: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], tuple[OrbitTangentAudit, ...]]:
        audits: list[OrbitTangentAudit] = []
        for orbit_index, (orbit, jacobian) in enumerate(
            zip(state.orbits, expanded.orbit_jacobians)
        ):
            selected = requested_update[
                expanded.atom_to_orbit == orbit_index
            ].reshape(-1)
            raw_jacobian = np.asarray(jacobian, dtype=np.float64)
            design = (
                raw_jacobian.reshape(-1, orbit.chart_dimension)
                if orbit.chart_dimension
                else np.zeros((selected.size, 0), dtype=np.float64)
            )
            rank = _matrix_rank(
                design,
                relative_tolerance=self.rank_relative_tolerance,
            )
            if rank != orbit.chart_dimension:
                raise TangentProjectionError(
                    f"orbit {orbit.orbit_id} Jacobian rank "
                    f"{rank} != {orbit.chart_dimension}"
                )

        chart_updates = project_atom_scores(expanded, requested_update)
        for orbit_index, (orbit, jacobian, chart_update) in enumerate(
            zip(state.orbits, expanded.orbit_jacobians, chart_updates)
        ):
            selected = requested_update[
                expanded.atom_to_orbit == orbit_index
            ].reshape(-1)
            raw_jacobian = np.asarray(jacobian, dtype=np.float64)
            design = (
                raw_jacobian.reshape(-1, orbit.chart_dimension)
                if orbit.chart_dimension
                else np.zeros((selected.size, 0), dtype=np.float64)
            )
            rank = _matrix_rank(
                design,
                relative_tolerance=self.rank_relative_tolerance,
            )
            projected = (
                design @ chart_update
                if orbit.chart_dimension
                else np.zeros_like(selected)
            )
            residual = selected - projected
            fixed_site_drift = (
                float(np.linalg.norm(projected))
                if orbit.chart_dimension == 0
                else 0.0
            )
            audits.append(
                OrbitTangentAudit(
                    orbit_id=orbit.orbit_id,
                    orbit_index=orbit_index,
                    chart_dimension=orbit.chart_dimension,
                    primitive_atom_count=int(
                        np.sum(expanded.atom_to_orbit == orbit_index)
                    ),
                    jacobian_rank=rank,
                    input_update_norm=float(np.linalg.norm(selected)),
                    chart_update_norm=float(np.linalg.norm(chart_update)),
                    tangent_update_norm=float(np.linalg.norm(projected)),
                    normal_residual_norm=float(np.linalg.norm(residual)),
                    fixed_site_drift=fixed_site_drift,
                )
            )
        return chart_updates, tuple(audits)

    def project_parent_step(
        self,
        *,
        state: StratifiedState,
        proposed_fractional_coordinates: Any,
        proposed_primitive_lattice: Any,
    ) -> TangentStepResult:
        """Project one absolute parent proposal and return a new immutable state."""

        state_snapshot: Mapping[str, Any] = state.to_dict()
        topology_before = state.topology_hash()
        signature_before = _discrete_signature(state)
        expanded_before = expand_state(
            state,
            self.catalog,
            redetect_space_group=False,
        )

        proposed_coordinates = _readonly_f64(
            proposed_fractional_coordinates,
            name="proposed primitive fractional coordinates",
            shape=tuple(expanded_before.fractional_coordinates.shape),
        )
        if np.any(proposed_coordinates < -1.0e-8) or np.any(
            proposed_coordinates >= 1.0 + 1.0e-8
        ):
            raise ValueError(
                "parent fractional-coordinate proposal must already be wrapped"
            )
        proposed_coordinates_wrapped = np.asarray(proposed_coordinates) % 1.0
        requested_update = _minimum_image_delta(
            proposed_coordinates_wrapped,
            expanded_before.fractional_coordinates,
        )
        chart_updates, orbit_audits = self._orbit_projection(
            state=state,
            expanded=expanded_before,
            requested_update=requested_update,
        )

        updated_orbits: list[OrbitState] = []
        for orbit, chart_update in zip(state.orbits, chart_updates):
            free_coordinate = tuple(
                float((value + delta) % 1.0)
                for value, delta in zip(
                    orbit.free_coordinate,
                    chart_update,
                )
            )
            updated_orbits.append(
                dataclasses.replace(
                    orbit,
                    free_coordinate=free_coordinate,
                )
            )

        proposed_primitive = _readonly_f64(
            proposed_primitive_lattice,
            name="proposed primitive lattice",
            shape=(3, 3),
        )
        primitive_transform = np.asarray(
            expanded_before.primitive_transform,
            dtype=np.float64,
        )
        try:
            proposed_conventional = np.linalg.solve(
                primitive_transform,
                proposed_primitive,
            )
        except np.linalg.LinAlgError as exc:
            raise TangentProjectionError(
                "primitive/conventional lattice transform is singular"
            ) from exc
        lattice_result = self.lattice_projector.project(
            lattice_chart=state.lattice_chart,
            lattice_system=state.lattice_system,
            proposed_conventional_lattice=proposed_conventional,
        )

        updated_state = dataclasses.replace(
            state,
            lattice_chart=lattice_result.lattice_chart,
            orbits=tuple(updated_orbits),
        )
        topology_after = updated_state.topology_hash()
        signature_after = _discrete_signature(updated_state)
        discrete_unchanged = signature_after == signature_before
        if topology_after != topology_before or not discrete_unchanged:
            raise TangentProjectionError(
                "tangent projection changed a discrete WQ field"
            )
        if state.to_dict() != state_snapshot:
            raise TangentProjectionError("tangent projection mutated its input state")

        expanded_after = expand_state(
            updated_state,
            self.catalog,
            redetect_space_group=False,
        )
        transform_after = np.asarray(
            expanded_after.primitive_transform,
            dtype=np.float64,
        )
        primitive_transform_consistency = float(
            np.max(
                np.abs(transform_after - primitive_transform),
                initial=0.0,
            )
        )
        if (
            primitive_transform_consistency
            > self.primitive_consistency_tolerance
        ):
            raise TangentProjectionError(
                "projected WQ expansion changed the primitive transform"
            )
        expected_primitive_lattice = (
            primitive_transform @ lattice_result.conventional_lattice
        )
        primitive_consistency = float(
            np.max(
                np.abs(
                    expanded_after.primitive_lattice
                    - expected_primitive_lattice
                ),
                initial=0.0,
            )
        )
        primitive_lattice_scale = max(
            float(
                np.max(
                    np.abs(expanded_after.primitive_lattice),
                    initial=0.0,
                )
            ),
            float(
                np.max(
                    np.abs(expected_primitive_lattice),
                    initial=0.0,
                )
            ),
            1.0,
        )
        primitive_relative_consistency = (
            primitive_consistency / primitive_lattice_scale
        )
        if (
            primitive_consistency > self.primitive_consistency_tolerance
            and primitive_relative_consistency
            > self.primitive_relative_consistency_tolerance
        ):
            raise TangentProjectionError(
                "projected primitive lattice disagrees with WQ expansion"
            )
        primitive_volume = abs(
            float(np.linalg.det(expanded_after.primitive_lattice))
        )
        if primitive_volume <= DEFAULT_MINIMUM_VOLUME:
            raise TangentProjectionError(
                "projected primitive lattice has non-positive volume"
            )
        if not np.array_equal(
            expanded_after.atomic_numbers,
            expanded_before.atomic_numbers,
        ):
            raise TangentProjectionError(
                "tangent projection changed primitive species ordering"
            )
        if not np.array_equal(
            expanded_after.atom_to_orbit,
            expanded_before.atom_to_orbit,
        ):
            raise TangentProjectionError(
                "tangent projection changed atom-to-orbit membership"
            )
        (
            orbit_chart_roundtrip_checked,
            maximum_orbit_chart_roundtrip_error,
        ) = self._orbit_chart_roundtrip(updated_state)

        audit = TangentStepAudit(
            schema="wq_wyckoff_tangent_step_audit_v2",
            topology_hash_before=topology_before,
            topology_hash_after=topology_after,
            discrete_fields_unchanged=discrete_unchanged,
            atom_count=updated_state.atom_count,
            orbit_regularization=ORBIT_REGULARIZATION,
            maximum_orbit_normal_residual_norm=max(
                (
                    value.normal_residual_norm
                    for value in orbit_audits
                ),
                default=0.0,
            ),
            orbit_chart_roundtrip_checked=orbit_chart_roundtrip_checked,
            maximum_orbit_chart_roundtrip_max_abs_error=(
                maximum_orbit_chart_roundtrip_error
            ),
            primitive_transform_consistency_max_abs_error=(
                primitive_transform_consistency
            ),
            primitive_lattice_consistency_max_abs_error=(
                primitive_consistency
            ),
            primitive_lattice_consistency_relative_error=(
                primitive_relative_consistency
            ),
            primitive_lattice_scale=primitive_lattice_scale,
            primitive_volume_after=primitive_volume,
            retry_or_replacement_used=False,
            topology_revision_used=False,
            orbit_audits=orbit_audits,
            lattice_audit=lattice_result.audit,
        )
        return TangentStepResult(
            state=updated_state,
            projected_fractional_coordinates=(
                expanded_after.fractional_coordinates
            ),
            projected_primitive_lattice=expanded_after.primitive_lattice,
            audit=audit,
        )
