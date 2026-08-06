"""Runtime expansion of a quotient state into one primitive crystal graph.

The preprocessing representation is conventional-setting Wyckoff metadata,
while the MP20 graph contract is explicitly primitive and capped at 20 atoms.
This module is the single conversion path used by sampling and evaluation so
the two cannot silently disagree about multiplicity, centering, or atom IDs.
"""

from __future__ import annotations

import dataclasses
import math
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .charts import LatticeChartCodec, periodic_cartesian_distance
from .state import GeometryEvidence, StratifiedState
from .vocabulary import atomic_number_to_input_id, crystal_system_from_space_group


class StateExpansionError(RuntimeError):
    pass


_PRIMITIVE_TRANSFORM_CACHE: dict[int, np.ndarray] = {}
_PRIMITIVE_TRANSFORM_CACHE_LOCK = threading.Lock()


@dataclasses.dataclass(frozen=True, slots=True)
class ExpandedState:
    conventional_lattice: np.ndarray
    primitive_lattice: np.ndarray
    fractional_coordinates: np.ndarray
    atomic_numbers: np.ndarray
    atom_to_orbit: np.ndarray
    orbit_jacobians: tuple[np.ndarray, ...]
    redetected_space_group: int
    primitive_transform: np.ndarray | None = None

    def __post_init__(self) -> None:
        atom_count = len(self.atomic_numbers)
        if self.conventional_lattice.shape != (3, 3):
            raise ValueError("conventional lattice must be 3x3")
        if self.primitive_lattice.shape != (3, 3):
            raise ValueError("primitive lattice must be 3x3")
        if self.fractional_coordinates.shape != (atom_count, 3):
            raise ValueError("fractional coordinate shape mismatch")
        if self.atom_to_orbit.shape != (atom_count,):
            raise ValueError("atom_to_orbit shape mismatch")
        if not 1 <= atom_count <= 20:
            raise ValueError("expanded MP20 graph must contain 1--20 atoms")
        if len(self.orbit_jacobians) == 0:
            raise ValueError("expanded graph must contain at least one orbit")
        if not all(
            np.all(np.isfinite(value))
            for value in (
                self.conventional_lattice,
                self.primitive_lattice,
                self.fractional_coordinates,
            )
        ):
            raise ValueError("expanded graph contains non-finite values")
        transform = self.primitive_transform
        if transform is None:
            try:
                transform = (
                    np.asarray(self.primitive_lattice, dtype=np.float64)
                    @ np.linalg.inv(
                        np.asarray(self.conventional_lattice, dtype=np.float64)
                    )
                )
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "expanded conventional lattice is singular"
                ) from exc
        transform = np.array(transform, dtype=np.float64, copy=True)
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            raise ValueError("primitive transform must be one finite 3x3 matrix")
        reconstructed = transform @ np.asarray(
            self.conventional_lattice,
            dtype=np.float64,
        )
        if not np.allclose(
            reconstructed,
            self.primitive_lattice,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError(
                "primitive transform disagrees with expanded lattice pair"
            )
        transform.setflags(write=False)
        object.__setattr__(self, "primitive_transform", transform)

    @property
    def atom_count(self) -> int:
        return int(len(self.atomic_numbers))

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.primitive_lattice)))

    def pymatgen_structure(self) -> Any:
        try:
            from pymatgen.core import Structure
        except ImportError as exc:  # pragma: no cover - server dependency
            raise RuntimeError("pymatgen is required to materialize a structure") from exc
        return Structure(
            self.primitive_lattice,
            [int(value) for value in self.atomic_numbers],
            self.fractional_coordinates,
            coords_are_cartesian=False,
            to_unit_cell=True,
        )

    def as_dict(self) -> Mapping[str, Any]:
        return self.pymatgen_structure().as_dict()


def _catalog_expand(
    catalog: Any,
    space_group: int,
    wyckoff_type: int,
    free_coordinate: Sequence[float],
) -> np.ndarray:
    expand = getattr(catalog, "expand", None)
    if expand is None:
        raise StateExpansionError("chart catalog does not implement runtime expansion")
    values = np.asarray(
        expand(space_group, wyckoff_type, free_coordinate),
        dtype=np.float64,
    )
    if values.ndim != 2 or values.shape[1:] != (3,) or not np.all(np.isfinite(values)):
        raise StateExpansionError("catalog returned invalid expanded coordinates")
    return values % 1.0


def _derive_primitive_transform(
    *,
    catalog: Any,
    space_group: int,
    expected_centering_factor: int,
) -> np.ndarray:
    if int(expected_centering_factor) == 1:
        # A declared P lattice is already primitive even when a particular
        # one-species/special-position decoration happens to admit an
        # additional translation.  Reducing that accidental sublattice would
        # violate the quotient state's registered multiplicities.
        return np.eye(3, dtype=np.float64)
    try:
        import spglib
    except ImportError as exc:  # pragma: no cover - server dependency
        raise RuntimeError("spglib is required for runtime primitive expansion") from exc
    candidates = []
    for wyckoff_type in catalog.types(space_group):
        spec = catalog.get(space_group, wyckoff_type)
        if int(spec.dimension) == 3:
            candidates.append(spec)
    if not candidates:
        raise StateExpansionError(
            f"space group {space_group} has no general-position chart"
        )
    general = max(
        candidates,
        key=lambda value: (int(value.multiplicity), -int(value.wyckoff_type)),
    )
    reference_coordinates = _catalog_expand(
        catalog,
        space_group,
        general.wyckoff_type,
        (0.17320508075688773, 0.2718281828459045, 0.4142135623730951),
    )
    system = crystal_system_from_space_group(space_group)
    reference_charts = {
        "triclinic": (math.log(5.0), math.log(6.0), math.log(7.0), 0.2, 0.3, 0.4),
        "monoclinic": (math.log(5.0), math.log(6.0), math.log(7.0), 0.3),
        "orthorhombic": (math.log(5.0), math.log(6.0), math.log(7.0)),
        "tetragonal": (math.log(5.0), math.log(7.0)),
        "trigonal": (math.log(5.0), math.log(7.0)),
        "hexagonal": (math.log(5.0), math.log(7.0)),
        "cubic": (math.log(5.0),),
    }
    reference_lattice = LatticeChartCodec.decode_matrix(
        reference_charts[system], system
    )
    cell = spglib.standardize_cell(
        (
            reference_lattice,
            reference_coordinates,
            np.ones(len(reference_coordinates), dtype=np.int32),
        ),
        to_primitive=True,
        no_idealize=True,
        symprec=1.0e-5,
        angle_tolerance=5.0,
    )
    if cell is None:
        raise StateExpansionError(
            "spglib failed to derive the declared centering transform"
        )
    lattice = np.asarray(cell[0], dtype=np.float64)
    return lattice @ np.linalg.inv(reference_lattice)


def _primitive_lattice(
    conventional_lattice: np.ndarray,
    *,
    catalog: Any,
    space_group: int,
    expected_centering_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    transform = _PRIMITIVE_TRANSFORM_CACHE.get(int(space_group))
    if transform is None:
        with _PRIMITIVE_TRANSFORM_CACHE_LOCK:
            transform = _PRIMITIVE_TRANSFORM_CACHE.get(int(space_group))
            if transform is None:
                transform = _derive_primitive_transform(
                    catalog=catalog,
                    space_group=space_group,
                    expected_centering_factor=expected_centering_factor,
                )
                transform[np.abs(transform) < 1.0e-12] = 0.0
                _PRIMITIVE_TRANSFORM_CACHE[int(space_group)] = transform.copy()
    determinant = abs(float(np.linalg.det(transform)))
    expected = 1.0 / int(expected_centering_factor)
    if not math.isclose(determinant, expected, rel_tol=1.0e-6, abs_tol=1.0e-8):
        raise StateExpansionError(
            f"declared centering transform determinant {determinant} != {expected}"
        )
    lattice = transform @ conventional_lattice
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise StateExpansionError("spglib returned an invalid primitive lattice")
    if abs(float(np.linalg.det(lattice))) <= 1.0e-10:
        raise StateExpansionError("primitive lattice is singular")
    return lattice, transform


def _exact_primitive_orbit(
    *,
    catalog: Any,
    state: StratifiedState,
    orbit_index: int,
    conventional_to_primitive: np.ndarray,
    primitive_lattice: np.ndarray,
    tolerance: float = 1.0e-5,
) -> tuple[np.ndarray, np.ndarray] | None:
    expand = getattr(catalog, "expand_with_jacobians", None)
    if expand is None:
        return None
    orbit = state.orbits[orbit_index]
    pairs = expand(
        state.space_group,
        orbit.wyckoff_type,
        orbit.free_coordinate,
    )
    points: list[np.ndarray] = []
    jacobians: list[np.ndarray] = []
    for conventional_point, conventional_jacobian in pairs:
        point = (np.asarray(conventional_point) @ conventional_to_primitive) % 1.0
        if any(
            periodic_cartesian_distance(point, existing, primitive_lattice)
            <= tolerance
            for existing in points
        ):
            continue
        jacobian = conventional_to_primitive.T @ np.asarray(
            conventional_jacobian, dtype=np.float64
        )
        points.append(point)
        jacobians.append(jacobian)
    return np.asarray(points, dtype=np.float64), np.asarray(jacobians, dtype=np.float64)


def _deduplicate(
    points: np.ndarray,
    lattice: np.ndarray,
    *,
    tolerance: float = 1.0e-5,
) -> np.ndarray:
    kept: list[np.ndarray] = []
    for point in points:
        wrapped = np.asarray(point, dtype=np.float64) % 1.0
        if not any(
            periodic_cartesian_distance(wrapped, other, lattice) <= tolerance
            for other in kept
        ):
            kept.append(wrapped)
    return np.asarray(kept, dtype=np.float64)


def _match_points(reference: np.ndarray, candidate: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    if len(reference) != len(candidate):
        raise StateExpansionError("finite-difference orbit multiplicity changed")
    remaining = set(range(len(candidate)))
    ordered: list[np.ndarray] = []
    for point in reference:
        choice = min(
            remaining,
            key=lambda index: periodic_cartesian_distance(point, candidate[index], lattice),
        )
        ordered.append(candidate[choice])
        remaining.remove(choice)
    return np.asarray(ordered, dtype=np.float64)


def _orbit_points_in_primitive(
    *,
    catalog: Any,
    state: StratifiedState,
    orbit_index: int,
    conventional_to_primitive: np.ndarray,
    primitive_lattice: np.ndarray,
    free_coordinate: Sequence[float] | None = None,
) -> np.ndarray:
    orbit = state.orbits[orbit_index]
    q = orbit.free_coordinate if free_coordinate is None else tuple(free_coordinate)
    conventional = _catalog_expand(
        catalog,
        state.space_group,
        orbit.wyckoff_type,
        q,
    )
    primitive = (conventional @ conventional_to_primitive) % 1.0
    return _deduplicate(primitive, primitive_lattice)


def _finite_difference_jacobian(
    *,
    catalog: Any,
    state: StratifiedState,
    orbit_index: int,
    base_points: np.ndarray,
    conventional_to_primitive: np.ndarray,
    primitive_lattice: np.ndarray,
    epsilon: float = 1.0e-5,
) -> np.ndarray:
    orbit = state.orbits[orbit_index]
    dimension = orbit.chart_dimension
    result = np.zeros((len(base_points), 3, dimension), dtype=np.float64)
    for axis in range(dimension):
        plus = list(orbit.free_coordinate)
        minus = list(orbit.free_coordinate)
        plus[axis] += epsilon
        minus[axis] -= epsilon
        plus_points = _match_points(
            base_points,
            _orbit_points_in_primitive(
                catalog=catalog,
                state=state,
                orbit_index=orbit_index,
                free_coordinate=plus,
                conventional_to_primitive=conventional_to_primitive,
                primitive_lattice=primitive_lattice,
            ),
            primitive_lattice,
        )
        minus_points = _match_points(
            base_points,
            _orbit_points_in_primitive(
                catalog=catalog,
                state=state,
                orbit_index=orbit_index,
                free_coordinate=minus,
                conventional_to_primitive=conventional_to_primitive,
                primitive_lattice=primitive_lattice,
            ),
            primitive_lattice,
        )
        delta = plus_points - minus_points
        delta -= np.rint(delta)
        result[:, :, axis] = delta / (2.0 * epsilon)
    return result


def _redetect_space_group(
    lattice: np.ndarray,
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
) -> int:
    try:
        import spglib
    except ImportError as exc:  # pragma: no cover - server dependency
        raise RuntimeError("spglib is required for symmetry redetection") from exc
    dataset = spglib.get_symmetry_dataset(
        (lattice, coordinates, atomic_numbers),
        symprec=0.1,
        angle_tolerance=5.0,
    )
    if dataset is None:
        return 0
    if hasattr(dataset, "number"):
        return int(dataset.number)
    return int(dataset["number"])


def expand_state(
    state: StratifiedState,
    catalog: Any,
    *,
    redetect_space_group: bool = True,
) -> ExpandedState:
    """Expand a state once; failures are terminal and never resampled."""

    try:
        conventional_lattice = LatticeChartCodec.decode_matrix(
            state.lattice_chart,
            state.lattice_system,
        )
        centering_factors = {
            int(orbit.multiplicity) // int(orbit.primitive_multiplicity)
            for orbit in state.orbits
        }
        if len(centering_factors) != 1:
            raise StateExpansionError("orbits disagree on conventional centering factor")
        centering_factor = next(iter(centering_factors))
        primitive_lattice, primitive_transform = _primitive_lattice(
            conventional_lattice,
            catalog=catalog,
            space_group=state.space_group,
            expected_centering_factor=centering_factor,
        )
        conventional_to_primitive = np.linalg.inv(primitive_transform)
        primitive_parts: list[np.ndarray] = []
        jacobians: list[np.ndarray] = []
        atomic_numbers: list[int] = []
        atom_to_orbit: list[int] = []
        for orbit_index, orbit in enumerate(state.orbits):
            exact = _exact_primitive_orbit(
                catalog=catalog,
                state=state,
                orbit_index=orbit_index,
                conventional_to_primitive=conventional_to_primitive,
                primitive_lattice=primitive_lattice,
            )
            if exact is None:
                points = _orbit_points_in_primitive(
                    catalog=catalog,
                    state=state,
                    orbit_index=orbit_index,
                    conventional_to_primitive=conventional_to_primitive,
                    primitive_lattice=primitive_lattice,
                )
                orbit_jacobian = _finite_difference_jacobian(
                    catalog=catalog,
                    state=state,
                    orbit_index=orbit_index,
                    base_points=points,
                    conventional_to_primitive=conventional_to_primitive,
                    primitive_lattice=primitive_lattice,
                )
            else:
                points, orbit_jacobian = exact
            if len(points) != int(orbit.primitive_multiplicity):
                raise StateExpansionError(
                    f"orbit {orbit.orbit_id} primitive multiplicity changed: "
                    f"{len(points)} != {orbit.primitive_multiplicity}"
                )
            primitive_parts.append(points)
            jacobians.append(orbit_jacobian)
            atomic_numbers.extend([orbit.species] * len(points))
            atom_to_orbit.extend([orbit_index] * len(points))
        coordinates = np.concatenate(primitive_parts, axis=0)
        numbers = np.asarray(atomic_numbers, dtype=np.int32)
        mapping = np.asarray(atom_to_orbit, dtype=np.int64)
        if len(coordinates) != state.atom_count:
            raise StateExpansionError(
                f"primitive atom-count mismatch: {len(coordinates)} != {state.atom_count}"
            )
        return ExpandedState(
            conventional_lattice=conventional_lattice,
            primitive_lattice=primitive_lattice,
            fractional_coordinates=coordinates,
            atomic_numbers=numbers,
            atom_to_orbit=mapping,
            orbit_jacobians=tuple(jacobians),
            redetected_space_group=(
                _redetect_space_group(
                    primitive_lattice,
                    coordinates,
                    numbers,
                )
                if redetect_space_group
                else 0
            ),
            primitive_transform=primitive_transform,
        )
    except StateExpansionError:
        raise
    except Exception as exc:
        raise StateExpansionError(f"{type(exc).__name__}:{exc}") from exc


def project_atom_scores(
    expanded: ExpandedState,
    atom_scores: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Project expanded atom scores back to every orbit tangent chart."""

    scores = np.asarray(atom_scores, dtype=np.float64)
    if scores.shape != expanded.fractional_coordinates.shape:
        raise ValueError("atom score shape mismatch")
    result: list[np.ndarray] = []
    for orbit_index, jacobian in enumerate(expanded.orbit_jacobians):
        dimension = jacobian.shape[-1]
        if dimension == 0:
            result.append(np.zeros(0, dtype=np.float64))
            continue
        selected = scores[expanded.atom_to_orbit == orbit_index].reshape(-1)
        design = jacobian.reshape(-1, dimension)
        # Tikhonov-regularized pseudoinverse is the registered tangent
        # projector; float64 keeps the numerical gate meaningful.
        gram = design.T @ design + 1.0e-8 * np.eye(dimension)
        result.append(np.linalg.solve(gram, design.T @ selected))
    return tuple(result)


def geometry_signals_from_graph(
    fractional_coordinates: np.ndarray,
    primitive_lattice: np.ndarray,
    atom_to_orbit: np.ndarray,
    orbit_count: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return label-free collision, coordination, and lattice signals.

    This is the single implementation shared by training-time corruptions and
    reverse-time states.  In particular, none of the synthetic corruption
    labels are accepted by this function, which prevents the revision head
    from receiving an oracle indicator through its geometry channel.
    """

    coordinates = np.asarray(fractional_coordinates, dtype=np.float64)
    lattice = np.asarray(primitive_lattice, dtype=np.float64)
    mapping = np.asarray(atom_to_orbit, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
        raise ValueError("fractional coordinates must have shape (N,3)")
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError("primitive lattice must be a finite 3x3 matrix")
    if mapping.shape != (len(coordinates),):
        raise ValueError("atom-to-orbit mapping shape mismatch")
    if not 1 <= int(orbit_count) <= len(coordinates):
        raise ValueError("orbit count must be in [1, atom_count]")
    if len(mapping) and (int(mapping.min()) < 0 or int(mapping.max()) >= orbit_count):
        raise ValueError("atom-to-orbit mapping is outside the orbit range")
    if abs(float(np.linalg.det(lattice))) <= 1.0e-12:
        raise ValueError("primitive lattice is singular")

    if len(coordinates) == 1:
        minimum_by_atom = np.asarray([10.0], dtype=np.float64)
        neighbor_count = np.zeros(1, dtype=np.int64)
    else:
        delta = coordinates[:, None, :] - coordinates[None, :, :]
        delta -= np.rint(delta)
        distances = np.linalg.norm(delta @ lattice, axis=-1)
        np.fill_diagonal(distances, np.inf)
        minimum_by_atom = np.min(distances, axis=1)
        neighbor_count = np.sum(distances <= 3.0, axis=1, dtype=np.int64)

    collision = np.zeros(orbit_count, dtype=np.float64)
    coordination = np.zeros(orbit_count, dtype=np.float64)
    for orbit_index in range(orbit_count):
        selected = mapping == orbit_index
        if not bool(np.any(selected)):
            raise ValueError(f"orbit {orbit_index} has no expanded atoms")
        min_distance = float(np.min(minimum_by_atom[selected]))
        collision[orbit_index] = max(0.0, min(1.0, (0.8 - min_distance) / 0.8))
        counts = neighbor_count[selected]
        coordination[orbit_index] = float(
            np.mean((counts < 1) | (counts > 12))
        )

    volume_per_atom = abs(float(np.linalg.det(lattice))) / len(coordinates)
    condition = float(np.linalg.cond(lattice))
    lattice_strain = min(
        1.0,
        abs(math.log(max(volume_per_atom, 1.0e-12) / 20.0)) / 4.0
        + max(0.0, math.log(max(condition, 1.0)) - math.log(8.0)) / 4.0,
    )
    return collision, coordination, float(lattice_strain)


def compute_geometry_evidence(
    state: StratifiedState,
    expanded: ExpandedState,
    *,
    score_norms: Sequence[float] | None = None,
    basin_uncertainties: Sequence[float] | None = None,
) -> tuple[GeometryEvidence, ...]:
    """Compute the six frozen non-MLIP revision signals per orbit."""

    count = len(state.orbits)
    score_norms = tuple(score_norms or (0.0,) * count)
    basin_uncertainties = tuple(basin_uncertainties or (0.0,) * count)
    if len(score_norms) != count or len(basin_uncertainties) != count:
        raise ValueError("geometry evidence length mismatch")
    collision, coordination, lattice_strain = geometry_signals_from_graph(
        expanded.fractional_coordinates,
        expanded.primitive_lattice,
        expanded.atom_to_orbit,
        count,
    )
    symmetry_residual = 0.0 if expanded.redetected_space_group == state.space_group else 1.0
    result: list[GeometryEvidence] = []
    for orbit_index in range(count):
        result.append(
            GeometryEvidence(
                collision_deficit=float(collision[orbit_index]),
                coordination_anomaly=float(coordination[orbit_index]),
                lattice_strain=lattice_strain,
                symmetry_residual=symmetry_residual,
                score_norm=float(score_norms[orbit_index]),
                basin_uncertainty=float(basin_uncertainties[orbit_index]),
            )
        )
    return tuple(result)


def tensorize_state(
    state: StratifiedState,
    expanded: ExpandedState,
    evidence: Sequence[GeometryEvidence],
    *,
    time: float,
    masked_species: frozenset[str] = frozenset(),
    masked_wyckoff: frozenset[str] = frozenset(),
) -> Any:
    """Build a one-graph ragged tensor batch without importing torch eagerly."""

    import torch

    from .model import WQTensorBatch

    if len(evidence) != len(state.orbits):
        raise ValueError("one geometry evidence vector is required per orbit")
    orbit_species = [
        0 if orbit.orbit_id in masked_species else atomic_number_to_input_id(orbit.species)
        for orbit in state.orbits
    ]
    orbit_wyckoff = [
        0 if orbit.orbit_id in masked_wyckoff else orbit.wyckoff_type + 1
        for orbit in state.orbits
    ]
    atom_species = [
        0
        if state.orbits[int(orbit_index)].orbit_id in masked_species
        else atomic_number_to_input_id(int(atomic_number))
        for atomic_number, orbit_index in zip(
            expanded.atomic_numbers,
            expanded.atom_to_orbit,
        )
    ]
    return WQTensorBatch(
        atom_species=torch.tensor(atom_species, dtype=torch.long),
        frac_coords=torch.tensor(expanded.fractional_coordinates, dtype=torch.float32),
        lattices=torch.tensor(expanded.primitive_lattice[None, :, :], dtype=torch.float32),
        atom_batch=torch.zeros(expanded.atom_count, dtype=torch.long),
        atom_to_orbit=torch.tensor(expanded.atom_to_orbit, dtype=torch.long),
        orbit_species=torch.tensor(orbit_species, dtype=torch.long),
        orbit_wyckoff=torch.tensor(orbit_wyckoff, dtype=torch.long),
        orbit_batch=torch.zeros(len(state.orbits), dtype=torch.long),
        space_group=torch.tensor([state.space_group], dtype=torch.long),
        time=torch.tensor([float(time)], dtype=torch.float32),
        geometry_evidence=torch.tensor(
            [item.as_tuple() for item in evidence], dtype=torch.float32
        ),
    )


def concatenate_tensor_batches(batches: Sequence[Any]) -> Any:
    """Concatenate one-graph ragged batches without semantic padding."""

    if not batches:
        raise ValueError("cannot concatenate an empty batch list")
    import torch

    from .model import WQTensorBatch

    if any(int(batch.space_group.numel()) != 1 for batch in batches):
        raise ValueError("sampling concatenation requires one graph per input batch")
    atom_offsets: list[torch.Tensor] = []
    atom_batches: list[torch.Tensor] = []
    orbit_batches: list[torch.Tensor] = []
    orbit_offset = 0
    for graph, batch in enumerate(batches):
        atom_offsets.append(batch.atom_to_orbit + orbit_offset)
        atom_batches.append(torch.full_like(batch.atom_batch, graph))
        orbit_batches.append(torch.full_like(batch.orbit_batch, graph))
        orbit_offset += int(batch.orbit_species.numel())
    return WQTensorBatch(
        atom_species=torch.cat([batch.atom_species for batch in batches]),
        frac_coords=torch.cat([batch.frac_coords for batch in batches]),
        lattices=torch.cat([batch.lattices for batch in batches]),
        atom_batch=torch.cat(atom_batches),
        atom_to_orbit=torch.cat(atom_offsets),
        orbit_species=torch.cat([batch.orbit_species for batch in batches]),
        orbit_wyckoff=torch.cat([batch.orbit_wyckoff for batch in batches]),
        orbit_batch=torch.cat(orbit_batches),
        space_group=torch.cat([batch.space_group for batch in batches]),
        time=torch.cat([batch.time for batch in batches]),
        geometry_evidence=torch.cat(
            [batch.geometry_evidence for batch in batches]
        ),
    )


def split_model_output(output: Any, batches: Sequence[Any]) -> tuple[Any, ...]:
    """Split a batched model output into one-graph views in input order."""

    from .model import WQModelOutput

    atom_start = orbit_start = 0
    result = []
    for graph, batch in enumerate(batches):
        atom_end = atom_start + int(batch.atom_species.numel())
        orbit_end = orbit_start + int(batch.orbit_species.numel())
        result.append(
            WQModelOutput(
                space_group_logits=output.space_group_logits[graph : graph + 1],
                species_logits=output.species_logits[orbit_start:orbit_end],
                wyckoff_logits=output.wyckoff_logits[orbit_start:orbit_end],
                event_logits=output.event_logits[graph : graph + 1],
                event_orbit_logits=output.event_orbit_logits[orbit_start:orbit_end],
                birth_species_logits=output.birth_species_logits[graph : graph + 1],
                birth_wyckoff_logits=output.birth_wyckoff_logits[graph : graph + 1],
                birth_coordinate_mean=output.birth_coordinate_mean[graph : graph + 1],
                birth_coordinate_log_scale=output.birth_coordinate_log_scale[
                    graph : graph + 1
                ],
                revision_logits=output.revision_logits[orbit_start:orbit_end],
                atom_coordinate_score=output.atom_coordinate_score[atom_start:atom_end],
                lattice_score=output.lattice_score[graph : graph + 1],
                bridge_mean=output.bridge_mean[orbit_start:orbit_end],
                bridge_log_scale=output.bridge_log_scale[orbit_start:orbit_end],
                orbit_features=output.orbit_features[orbit_start:orbit_end],
            )
        )
        atom_start = atom_end
        orbit_start = orbit_end
    return tuple(result)
