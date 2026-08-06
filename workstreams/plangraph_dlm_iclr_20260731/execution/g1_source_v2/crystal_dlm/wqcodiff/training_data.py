"""Deterministic ragged batching and registered synthetic corruptions."""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .charts import LatticeChartCodec
from .dataset import PRIMARY_SYMPREC, tolerance_tag
from .losses import WQLossTargets, WQPriorTargets
from .model import WQTensorBatch, WQVariant
from .runtime import geometry_signals_from_graph
from .vocabulary import (
    MP20_ATOMIC_NUMBERS,
    atomic_number_to_input_id,
    atomic_number_to_target,
    target_to_atomic_number,
)


_PERIODIC_SHIFTS = np.asarray(
    [
        (first, second, third)
        for first in (-1.0, 0.0, 1.0)
        for second in (-1.0, 0.0, 1.0)
        for third in (-1.0, 0.0, 1.0)
    ],
    dtype=np.float64,
)


class JsonlRecordIndex:
    """Offset index over selected immutable preprocessing records."""

    def __init__(self, paths: Sequence[str | os.PathLike[str]]) -> None:
        if not paths:
            raise ValueError("at least one dataset shard is required")
        self.paths = tuple(Path(path).resolve() for path in paths)
        self.entries: list[tuple[int, int]] = []
        self._handles: dict[int, Any] = {}
        for path_index, path in enumerate(self.paths):
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    payload = json.loads(line)
                    if payload.get("selected"):
                        self.entries.append((path_index, offset))
        if not self.entries:
            raise ValueError("dataset index contains no selected records")

    @classmethod
    def from_frozen_entries(
        cls,
        paths: Sequence[str | os.PathLike[str]],
        entries: Sequence[tuple[int, int]],
    ) -> "JsonlRecordIndex":
        """Reopen an already audited offset index without rescanning JSONL.

        Training prefetch workers receive this immutable index from the main
        process.  Each worker owns independent file handles, so concurrent
        seeks cannot race while record order remains exactly the registered
        ``EpochSampler`` order.
        """

        if not paths or not entries:
            raise ValueError("frozen dataset index cannot be empty")
        instance = cls.__new__(cls)
        instance.paths = tuple(Path(path).resolve() for path in paths)
        instance.entries = [(int(path_index), int(offset)) for path_index, offset in entries]
        if any(
            path_index < 0 or path_index >= len(instance.paths) or offset < 0
            for path_index, offset in instance.entries
        ):
            raise ValueError("frozen dataset index contains an invalid offset")
        instance._handles = {}
        return instance

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path_index, offset = self.entries[index]
        handle = self._handles.get(path_index)
        if handle is None or handle.closed:
            handle = self.paths[path_index].open("rb")
            self._handles[path_index] = handle
        handle.seek(offset)
        return json.loads(handle.readline())

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:  # pragma: no cover - best effort at interpreter exit
        self.close()


@dataclasses.dataclass(slots=True)
class CorruptedBatch:
    inputs: WQTensorBatch
    targets: WQLossTargets
    prior_targets: WQPriorTargets
    metadata: tuple[Mapping[str, Any], ...]

    def to(self, device: torch.device | str) -> "CorruptedBatch":
        return CorruptedBatch(
            inputs=self.inputs.to(device),
            targets=WQLossTargets(*(value.to(device) for value in self.targets)),
            prior_targets=WQPriorTargets(*(value.to(device) for value in self.prior_targets)),
            metadata=self.metadata,
        )


@dataclasses.dataclass(slots=True)
class _OrbitWork:
    payload: dict[str, Any]
    source_species: int
    input_species: int
    source_wyckoff: int
    input_wyckoff: int
    is_false: bool = False
    wrong_wyckoff: bool = False
    wrong_species: bool = False
    pointer_target: bool = False

    @property
    def multiplicity(self) -> int:
        return int(
            self.payload["orbit"].get(
                "primitive_multiplicity",
                self.payload["orbit"]["multiplicity"],
            )
        )


def _primary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return dict(payload["decompositions"][tolerance_tag(PRIMARY_SYMPREC)])


@functools.lru_cache(maxsize=128)
def _atomic_number(symbol: str) -> int:
    from pymatgen.core.periodic_table import Element

    return int(Element(symbol).Z)


def _atom_level_primary(primary: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a primitive crystal to a dynamic P1 atom set without padding."""

    primitive = primary["primitive_structure"]
    matrix = np.asarray(primitive["lattice"]["matrix"], dtype=np.float64)
    chart = LatticeChartCodec.encode_matrix(matrix, "triclinic")
    orbits: list[dict[str, Any]] = []
    for index, site in enumerate(primitive["sites"]):
        species = site.get("species", ())
        if len(species) != 1 or abs(float(species[0].get("occu", 1.0)) - 1.0) > 1.0e-8:
            raise ValueError("B-ATOM-JOINT requires an ordered primitive structure")
        symbol = str(species[0].get("element") or species[0].get("name"))
        coordinate = [float(value) % 1.0 for value in site["abc"]]
        orbit = {
            "orbit_id": f"a{index}",
            "wyckoff_type": 0,
            "species": _atomic_number(symbol),
            "multiplicity": 1,
            "primitive_multiplicity": 1,
            "chart_dimension": 3,
            "free_coordinate": coordinate,
        }
        orbits.append(
            {
                "orbit": orbit,
                "wyckoff_symbol": "1a",
                "representative": coordinate,
                "chart_origin": [0.0, 0.0, 0.0],
                "chart_basis": np.eye(3).tolist(),
                "chart_fit_residual_angstrom": 0.0,
                "expanded_fractional_coordinates": [coordinate],
                "expanded_chart_jacobians": [np.eye(3).tolist()],
                "primitive_fractional_coordinates": [coordinate],
                "primitive_chart_jacobians": [np.eye(3).tolist()],
            }
        )
    if not 1 <= len(orbits) <= 20:
        raise ValueError("B-ATOM-JOINT primitive atom count is outside MP20")
    return {
        **dict(primary),
        "state": {
            "space_group": 1,
            "lattice_system": "triclinic",
            "lattice_chart": list(chart),
            "orbits": [value["orbit"] for value in orbits],
            "space_group_committed": True,
            "timestep": 0.0,
        },
        "primitive_lattice_transform": np.eye(3).tolist(),
        "orbits": orbits,
    }


def _mask_probability(time: float) -> float:
    return math.sin(0.5 * math.pi * time) ** 2


def _coordinate_sigma(time: float) -> float:
    return math.exp((1.0 - time) * math.log(0.005) + time * math.log(0.5))


def _wrapped_gaussian_score(
    wrapped_delta: np.ndarray | Sequence[float] | float,
    sigma: float,
    *,
    image_radius: int = 8,
) -> np.ndarray:
    """Exact torus score up to a preregistered, negligible image tail.

    ``wrapped_delta`` is interpreted modulo one in ``[-0.5, 0.5)``.  At the
    largest registered sigma (0.5), omitted images have relative mass below
    ``exp(-144.5)``; at smaller sigmas the bound is tighter.
    """

    if not math.isfinite(float(sigma)) or float(sigma) <= 0.0:
        raise ValueError("wrapped Gaussian sigma must be finite and positive")
    if int(image_radius) != image_radius or image_radius < 1:
        raise ValueError("wrapped Gaussian image radius must be a positive integer")
    delta = np.asarray(wrapped_delta, dtype=np.float64)
    if not np.all(np.isfinite(delta)):
        raise ValueError("wrapped Gaussian displacement must be finite")
    delta = (delta + 0.5) % 1.0 - 0.5
    images = np.arange(-image_radius, image_radius + 1, dtype=np.float64)
    lifted = delta[..., None] + images
    log_weight = -0.5 * np.square(lifted / float(sigma))
    log_weight -= np.max(log_weight, axis=-1, keepdims=True)
    weight = np.exp(log_weight)
    mean_lift = np.sum(weight * lifted, axis=-1) / np.sum(weight, axis=-1)
    return -mean_lift / (float(sigma) ** 2)


@functools.lru_cache(maxsize=230)
def _legal_wyckoff_types(space_group: int) -> tuple[int, ...]:
    from .charts import PyXtalChartCatalog

    catalog = getattr(_legal_wyckoff_types, "_catalog", None)
    if catalog is None:
        catalog = PyXtalChartCatalog()
        setattr(_legal_wyckoff_types, "_catalog", catalog)
    return catalog.types(int(space_group))


@functools.lru_cache(maxsize=4096)
def _candidate_chart_images(
    space_group: int,
    wyckoff_type: int,
) -> tuple[int, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    """Cache every symmetry-equivalent affine chart image for one type."""

    from .charts import PyXtalChartCatalog

    catalog = getattr(_candidate_chart_images, "_catalog", None)
    if catalog is None:
        catalog = PyXtalChartCatalog()
        setattr(_candidate_chart_images, "_catalog", catalog)
    spec = catalog.get(int(space_group), int(wyckoff_type))
    pairs = catalog.expand_with_jacobians(
        int(space_group),
        int(wyckoff_type),
        (0.0,) * spec.dimension,
    )
    images = tuple(
        (
            np.asarray(origin, dtype=np.float64),
            np.asarray(basis, dtype=np.float64),
        )
        for origin, basis in pairs
    )
    return int(spec.primitive_multiplicity), images


def _affine_image_residual(
    point: np.ndarray,
    origin: np.ndarray,
    basis: np.ndarray,
    conventional_lattice: np.ndarray,
) -> float:
    """Metric-aware distance from one periodic point to an affine chart."""

    deltas = point[None, :] + _PERIODIC_SHIFTS - origin[None, :]
    targets = deltas @ conventional_lattice
    if basis.shape[1]:
        # Rows of ``basis.T @ lattice`` span all Cartesian displacements
        # allowed by this affine image.  The residual projector is calculated
        # once for all 27 periodic lifts instead of solving 27 least-squares
        # systems in Python.
        design = basis.T @ conventional_lattice
        projector = np.eye(3, dtype=np.float64) - np.linalg.pinv(
            design, rcond=1.0e-12
        ) @ design
        targets = targets @ projector
    return float(np.min(np.linalg.norm(targets, axis=1)))


def _wyckoff_geometry_residual(
    *,
    space_group: int,
    candidate_wyckoff: int | None,
    primitive_coordinates: np.ndarray,
    primitive_lattice_transform: np.ndarray,
    conventional_lattice: np.ndarray,
) -> float:
    """Measure candidate-type support using geometry only, never labels."""

    if candidate_wyckoff is None:
        # A MASK token is deliberately not supplied with its hidden type.
        return 0.0
    expected_count, images = _candidate_chart_images(
        int(space_group), int(candidate_wyckoff)
    )
    if len(primitive_coordinates) != expected_count:
        return 1.0
    if images and images[0][1].shape[1] == 3:
        # A full-dimensional chart supports every fractional point; its
        # primitive multiplicity already provides the topology constraint.
        return 0.0
    residual = 0.0
    for primitive_point in primitive_coordinates:
        conventional_point = (
            np.asarray(primitive_point, dtype=np.float64)
            @ primitive_lattice_transform
        ) % 1.0
        point_residual = math.inf
        for origin, basis in images:
            point_residual = min(
                point_residual,
                _affine_image_residual(
                    conventional_point, origin, basis, conventional_lattice
                ),
            )
            if point_residual < 1.0e-10:
                break
        residual = max(residual, point_residual)
    # 0.1 Angstrom is the largest registered redetection tolerance.  Values
    # are bounded so a rare pathological cell cannot dominate the projection.
    return float(min(1.0, residual / 0.1))


def _training_geometry_evidence(
    *,
    space_group: int | None,
    primitive_coordinates: np.ndarray,
    primitive_lattice: np.ndarray,
    atom_to_orbit: np.ndarray,
    candidate_wyckoff: Sequence[int | None],
    primitive_lattice_transform: np.ndarray,
    conventional_lattice: np.ndarray,
    time: float,
) -> list[list[float]]:
    """Construct the six non-MLIP signals without corruption-label access."""

    orbit_count = len(candidate_wyckoff)
    collision, coordination, lattice_strain = geometry_signals_from_graph(
        primitive_coordinates,
        primitive_lattice,
        atom_to_orbit,
        orbit_count,
    )
    result: list[list[float]] = []
    for orbit_index, candidate in enumerate(candidate_wyckoff):
        selected = atom_to_orbit == orbit_index
        symmetry = (
            0.0
            if space_group is None
            else _wyckoff_geometry_residual(
                space_group=space_group,
                candidate_wyckoff=candidate,
                primitive_coordinates=primitive_coordinates[selected],
                primitive_lattice_transform=primitive_lattice_transform,
                conventional_lattice=conventional_lattice,
            )
        )
        result.append(
            [
                float(collision[orbit_index]),
                float(coordination[orbit_index]),
                float(lattice_strain),
                float(symmetry),
                0.0,  # replaced by the model's current detached score norm
                float(time),
            ]
        )
    return result


def _random_other_species(rng: random.Random, original: int) -> int:
    candidates = [value for value in MP20_ATOMIC_NUMBERS if value != original]
    return candidates[rng.randrange(len(candidates))]


def _noise_lattice(
    chart: Sequence[float],
    crystal_system: str,
    time: float,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(chart, dtype=np.float64)
    noise = generator.standard_normal(values.shape)
    alpha = max(math.cos(0.5 * math.pi * time) ** 2, 1.0e-6)
    noisy = math.sqrt(alpha) * values + math.sqrt(1.0 - alpha) * noise
    matrix = LatticeChartCodec.decode_matrix(noisy, crystal_system)
    padded_noise = np.zeros(6, dtype=np.float32)
    padded_mask = np.zeros(6, dtype=np.bool_)
    padded_noise[: len(noise)] = noise.astype(np.float32)
    padded_mask[: len(noise)] = True
    return matrix.astype(np.float32), padded_noise, padded_mask


def _orbit_coordinates(
    orbit: _OrbitWork,
    time: float,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = orbit.payload
    dimension = int(payload["orbit"]["chart_dimension"])
    coordinates = np.asarray(payload["primitive_fractional_coordinates"], dtype=np.float64)
    jacobians = [np.asarray(value, dtype=np.float64) for value in payload["primitive_chart_jacobians"]]
    if dimension == 0 or orbit.is_false:
        target_score = np.zeros_like(coordinates, dtype=np.float32)
        mask = np.zeros(len(coordinates), dtype=np.bool_)
        return coordinates.astype(np.float32), target_score, mask, np.zeros(3, dtype=np.float32)
    sigma = _coordinate_sigma(time)
    epsilon = generator.standard_normal(dimension)
    delta = sigma * epsilon
    wrapped_delta = (delta + 0.5) % 1.0 - 0.5
    score_q = _wrapped_gaussian_score(wrapped_delta, sigma)
    noisy: list[np.ndarray] = []
    score_atoms: list[np.ndarray] = []
    for coordinate, jacobian in zip(coordinates, jacobians):
        noisy.append((coordinate + jacobian @ wrapped_delta) % 1.0)
        score_atoms.append(jacobian @ score_q)
    padded_q = np.zeros(3, dtype=np.float32)
    padded_q[:dimension] = np.asarray(payload["orbit"]["free_coordinate"], dtype=np.float32)
    return (
        np.asarray(noisy, dtype=np.float32),
        np.asarray(score_atoms, dtype=np.float32),
        np.ones(len(noisy), dtype=np.bool_),
        padded_q,
    )


def _make_false_orbit(source: _OrbitWork, rng: random.Random, generator: np.random.Generator) -> _OrbitWork:
    payload = json.loads(json.dumps(source.payload))
    dimension = int(payload["orbit"]["chart_dimension"])
    delta = generator.uniform(-0.35, 0.35, size=dimension)
    coordinates = np.asarray(payload["primitive_fractional_coordinates"], dtype=np.float64)
    jacobians = [np.asarray(value, dtype=np.float64) for value in payload["primitive_chart_jacobians"]]
    payload["primitive_fractional_coordinates"] = [
        ((coordinate + jacobian @ delta) % 1.0).tolist()
        for coordinate, jacobian in zip(coordinates, jacobians)
    ]
    original_species = source.source_species
    false_species = _random_other_species(rng, original_species)
    payload["orbit"]["orbit_id"] = f"false:{payload['orbit']['orbit_id']}"
    payload["orbit"]["species"] = false_species
    return _OrbitWork(
        payload=payload,
        source_species=false_species,
        input_species=false_species,
        source_wyckoff=source.source_wyckoff,
        input_wyckoff=source.source_wyckoff,
        is_false=True,
        pointer_target=True,
    )


def build_corrupted_batch(
    payloads: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    variant: WQVariant,
    representation_variant: WQVariant | None = None,
    enable_revision_training: bool,
    mask_discrete_fields: bool = True,
    enable_topology_corruption: bool = True,
) -> CorruptedBatch:
    if not payloads:
        raise ValueError("cannot build an empty batch")
    rng = random.Random(seed)
    generator = np.random.default_rng(seed)

    atom_species: list[int] = []
    frac_coords: list[np.ndarray] = []
    atom_batch: list[int] = []
    atom_to_orbit: list[int] = []
    orbit_species: list[int] = []
    orbit_wyckoff: list[int] = []
    orbit_batch: list[int] = []
    space_group: list[int] = []
    times: list[float] = []
    evidences: list[list[float]] = []
    lattices: list[np.ndarray] = []

    target_sg: list[int] = []
    target_species: list[int] = []
    target_wyckoff: list[int] = []
    target_event: list[int] = []
    event_orbit: list[float] = []
    event_orbit_mask: list[bool] = []
    birth_species: list[int] = []
    birth_wyckoff: list[int] = []
    birth_coordinate: list[np.ndarray] = []
    birth_coordinate_mask: list[np.ndarray] = []
    revision: list[list[float]] = []
    revision_mask: list[list[bool]] = []
    coordinate_score: list[np.ndarray] = []
    coordinate_mask: list[bool] = []
    coordinate_weight: list[float] = []
    lattice_score: list[np.ndarray] = []
    lattice_mask: list[np.ndarray] = []
    bridge_coordinate: list[np.ndarray] = []
    bridge_mask: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    prior_space_group: list[int] = []
    prior_species: list[int] = []
    prior_wyckoff: list[int] = []
    prior_coordinate: list[np.ndarray] = []
    prior_coordinate_mask: list[np.ndarray] = []
    prior_lattice: list[np.ndarray] = []
    prior_lattice_mask: list[np.ndarray] = []

    orbit_offset = 0
    for graph_index, record in enumerate(payloads):
        primary = _primary(record)
        if representation_variant is WQVariant.ATOM_JOINT:
            primary = _atom_level_primary(primary)
        state = primary["state"]
        time = min(max(rng.random(), 1.0e-4), 1.0 - 1.0e-4)
        probability = _mask_probability(time)
        times.append(time)
        system = str(state["lattice_system"])
        conventional_lattice, lattice_target, lattice_target_mask = _noise_lattice(
            state["lattice_chart"], system, time, generator
        )
        primitive_transform = np.asarray(primary["primitive_lattice_transform"], dtype=np.float64)
        primitive_lattice = primitive_transform @ conventional_lattice
        lattices.append(primitive_lattice.astype(np.float32))
        lattice_score.append(lattice_target)
        lattice_mask.append(lattice_target_mask)
        clean_lattice = np.zeros(6, dtype=np.float32)
        clean_lattice_mask = np.zeros(6, dtype=np.bool_)
        clean_lattice_values = np.asarray(state["lattice_chart"], dtype=np.float32)
        clean_lattice[: len(clean_lattice_values)] = clean_lattice_values
        clean_lattice_mask[: len(clean_lattice_values)] = True
        prior_lattice.append(clean_lattice)
        prior_lattice_mask.append(clean_lattice_mask)

        works = [
            _OrbitWork(
                payload=dict(value),
                source_species=int(value["orbit"]["species"]),
                input_species=int(value["orbit"]["species"]),
                source_wyckoff=int(value["orbit"]["wyckoff_type"]),
                input_wyckoff=int(value["orbit"]["wyckoff_type"]),
            )
            for value in primary["orbits"]
        ]
        prior_orbit = works[rng.randrange(len(works))]
        prior_space_group.append(int(state["space_group"]) - 1)
        prior_species.append(atomic_number_to_target(prior_orbit.source_species))
        prior_wyckoff.append(prior_orbit.source_wyckoff)
        prior_q = np.zeros(3, dtype=np.float32)
        prior_q_mask = np.zeros(3, dtype=np.bool_)
        prior_dimension = int(prior_orbit.payload["orbit"]["chart_dimension"])
        prior_q[:prior_dimension] = np.asarray(
            prior_orbit.payload["orbit"]["free_coordinate"], dtype=np.float32
        )
        prior_q_mask[:prior_dimension] = True
        prior_coordinate.append(prior_q)
        prior_coordinate_mask.append(prior_q_mask)
        rng.shuffle(works)
        original_atom_count = sum(work.multiplicity for work in works)
        event = 0
        event_name = "none"
        deleted: _OrbitWork | None = None
        pointer: _OrbitWork | None = None

        operators: list[str] = ["wrong_species"]
        if len(works) > 1:
            operators.append("deletion")
        if any(original_atom_count + work.multiplicity <= 20 for work in works):
            operators.append("false_insertion")
        if len({work.source_wyckoff for work in works}) > 1:
            operators.append("wrong_wyckoff")
        if enable_topology_corruption and rng.random() < probability:
            event_name = operators[rng.randrange(len(operators))]

        if event_name == "deletion":
            position = rng.randrange(len(works))
            deleted = works.pop(position)
            event = 1
        elif event_name == "false_insertion":
            sources = [work for work in works if original_atom_count + work.multiplicity <= 20]
            false_orbit = _make_false_orbit(sources[rng.randrange(len(sources))], rng, generator)
            works.append(false_orbit)
            rng.shuffle(works)
            pointer = false_orbit
            event = 2
        elif event_name == "wrong_wyckoff":
            pointer = works[rng.randrange(len(works))]
            choices = sorted({work.source_wyckoff for work in works} - {pointer.source_wyckoff})
            pointer.input_wyckoff = choices[rng.randrange(len(choices))]
            pointer.wrong_wyckoff = True
            pointer.pointer_target = True
            event = 3
        elif event_name == "wrong_species":
            pointer = works[rng.randrange(len(works))]
            pointer.input_species = _random_other_species(rng, pointer.source_species)
            pointer.wrong_species = True
            pointer.pointer_target = True
            event = 4

        sg = int(state["space_group"])
        mask_sg = mask_discrete_fields and rng.random() < probability
        space_group.append(0 if mask_sg else sg)
        target_sg.append(sg - 1 if mask_sg else -100)
        target_event.append(event)

        if deleted is None:
            birth_species.append(-100)
            birth_wyckoff.append(-100)
            birth_coordinate.append(np.zeros(3, dtype=np.float32))
            birth_coordinate_mask.append(np.zeros(3, dtype=np.bool_))
        else:
            birth_species.append(atomic_number_to_target(deleted.source_species))
            birth_wyckoff.append(deleted.source_wyckoff)
            dimension = int(deleted.payload["orbit"]["chart_dimension"])
            q = np.zeros(3, dtype=np.float32)
            mask = np.zeros(3, dtype=np.bool_)
            q[:dimension] = np.asarray(deleted.payload["orbit"]["free_coordinate"], dtype=np.float32)
            mask[:dimension] = True
            birth_coordinate.append(q)
            birth_coordinate_mask.append(mask)

        graph_atom_start = len(frac_coords)
        graph_candidate_wyckoff: list[int | None] = []

        # AR observes a random-order prefix; the other kernels apply their
        # registered independent corruption to an equivariant set.
        ar_cut = max(0, min(len(works), int(round((1.0 - time) * len(works)))))
        for local_index, work in enumerate(works):
            physical_species = work.input_species
            species_input = atomic_number_to_input_id(physical_species)
            wp_input = work.input_wyckoff + 1
            if not mask_discrete_fields:
                mask_species = mask_wp = False
            elif variant is WQVariant.AR:
                mask_species = mask_wp = local_index >= ar_cut
            else:
                mask_species = rng.random() < probability
                mask_wp = rng.random() < probability
            if work.wrong_species:
                mask_species = False
            if work.wrong_wyckoff:
                mask_wp = False

            if mask_species:
                if variant is WQVariant.D3PM:
                    species_input = rng.randrange(1, len(MP20_ATOMIC_NUMBERS) + 1)
                else:
                    species_input = 0
            if mask_wp:
                if variant is WQVariant.D3PM:
                    legal_types = _legal_wyckoff_types(int(state["space_group"]))
                    wp_input = legal_types[rng.randrange(len(legal_types))] + 1
                else:
                    wp_input = 0
            graph_candidate_wyckoff.append(None if wp_input == 0 else wp_input - 1)

            orbit_species.append(species_input)
            orbit_wyckoff.append(wp_input)
            orbit_batch.append(graph_index)
            target_species.append(
                atomic_number_to_target(work.source_species)
                if (mask_species or work.wrong_species) and not work.is_false
                else -100
            )
            target_wyckoff.append(
                work.source_wyckoff
                if (mask_wp or work.wrong_wyckoff) and not work.is_false
                else -100
            )
            is_pointer_event = event in {2, 3, 4}
            event_orbit.append(float(work is pointer))
            event_orbit_mask.append(is_pointer_event)

            revision.append(
                [
                    float(work.is_false),
                    float(work.wrong_wyckoff),
                    float(work.wrong_species),
                ]
            )
            enable_for_variant = variant in {WQVariant.STRAT_CONF, WQVariant.STRAT_GEO}
            revision_mask.append([enable_revision_training and enable_for_variant] * 3)

            coords, scores, coord_masks, padded_q = _orbit_coordinates(work, time, generator)
            score_weight = _coordinate_sigma(time) ** 2
            global_orbit = orbit_offset + local_index
            input_atom_species = 0 if mask_species else atomic_number_to_input_id(physical_species)
            for coordinate, score, score_mask in zip(coords, scores, coord_masks):
                atom_species.append(input_atom_species)
                frac_coords.append(coordinate)
                atom_batch.append(graph_index)
                atom_to_orbit.append(global_orbit)
                coordinate_score.append(score)
                coordinate_mask.append(bool(score_mask))
                coordinate_weight.append(score_weight)

            q_mask = np.zeros(3, dtype=np.bool_)
            if work.wrong_wyckoff and not work.is_false:
                q_mask[: int(work.payload["orbit"]["chart_dimension"])] = True
            bridge_coordinate.append(padded_q)
            bridge_mask.append(q_mask)

        if variant is WQVariant.STRAT_GEO:
            graph_coordinates = np.asarray(frac_coords[graph_atom_start:], dtype=np.float64)
            graph_mapping = (
                np.asarray(atom_to_orbit[graph_atom_start:], dtype=np.int64)
                - orbit_offset
            )
            evidences.extend(
                _training_geometry_evidence(
                    # The hidden target SG is never used to form evidence.
                    # Reverse-time sampling always has a committed SG, whereas
                    # a training MASK receives no SG-conditioned residual.
                    space_group=None if mask_sg else sg,
                    primitive_coordinates=graph_coordinates,
                    primitive_lattice=np.asarray(primitive_lattice, dtype=np.float64),
                    atom_to_orbit=graph_mapping,
                    candidate_wyckoff=graph_candidate_wyckoff,
                    primitive_lattice_transform=primitive_transform,
                    conventional_lattice=np.asarray(conventional_lattice, dtype=np.float64),
                    time=time,
                )
            )
        else:
            evidences.extend([[0.0] * 6 for _ in works])

        orbit_offset += len(works)
        metadata.append(
            {
                "material_id": record["material_id"],
                "time": time,
                "event": event_name,
                "input_orbits": len(works),
                "input_atoms": sum(work.multiplicity for work in works),
            }
        )

    inputs = WQTensorBatch(
        atom_species=torch.tensor(atom_species, dtype=torch.long),
        frac_coords=torch.tensor(np.asarray(frac_coords), dtype=torch.float32),
        lattices=torch.tensor(np.asarray(lattices), dtype=torch.float32),
        atom_batch=torch.tensor(atom_batch, dtype=torch.long),
        atom_to_orbit=torch.tensor(atom_to_orbit, dtype=torch.long),
        orbit_species=torch.tensor(orbit_species, dtype=torch.long),
        orbit_wyckoff=torch.tensor(orbit_wyckoff, dtype=torch.long),
        orbit_batch=torch.tensor(orbit_batch, dtype=torch.long),
        space_group=torch.tensor(space_group, dtype=torch.long),
        time=torch.tensor(times, dtype=torch.float32),
        geometry_evidence=torch.tensor(evidences, dtype=torch.float32),
    )
    targets = WQLossTargets(
        space_group=torch.tensor(target_sg, dtype=torch.long),
        species=torch.tensor(target_species, dtype=torch.long),
        wyckoff=torch.tensor(target_wyckoff, dtype=torch.long),
        event=torch.tensor(target_event, dtype=torch.long),
        event_orbit=torch.tensor(event_orbit, dtype=torch.float32),
        event_orbit_mask=torch.tensor(event_orbit_mask, dtype=torch.bool),
        birth_species=torch.tensor(birth_species, dtype=torch.long),
        birth_wyckoff=torch.tensor(birth_wyckoff, dtype=torch.long),
        birth_coordinate=torch.tensor(np.asarray(birth_coordinate), dtype=torch.float32),
        birth_coordinate_mask=torch.tensor(np.asarray(birth_coordinate_mask), dtype=torch.bool),
        revision=torch.tensor(revision, dtype=torch.float32),
        revision_mask=torch.tensor(revision_mask, dtype=torch.bool),
        coordinate_score=torch.tensor(np.asarray(coordinate_score), dtype=torch.float32),
        coordinate_mask=torch.tensor(coordinate_mask, dtype=torch.bool),
        coordinate_weight=torch.tensor(coordinate_weight, dtype=torch.float32),
        lattice_score=torch.tensor(np.asarray(lattice_score), dtype=torch.float32),
        lattice_mask=torch.tensor(np.asarray(lattice_mask), dtype=torch.bool),
        bridge_coordinate=torch.tensor(np.asarray(bridge_coordinate), dtype=torch.float32),
        bridge_mask=torch.tensor(np.asarray(bridge_mask), dtype=torch.bool),
    )
    prior_targets = WQPriorTargets(
        space_group=torch.tensor(prior_space_group, dtype=torch.long),
        first_species=torch.tensor(prior_species, dtype=torch.long),
        first_wyckoff=torch.tensor(prior_wyckoff, dtype=torch.long),
        first_coordinate=torch.tensor(np.asarray(prior_coordinate), dtype=torch.float32),
        first_coordinate_mask=torch.tensor(
            np.asarray(prior_coordinate_mask), dtype=torch.bool
        ),
        lattice_chart=torch.tensor(np.asarray(prior_lattice), dtype=torch.float32),
        lattice_chart_mask=torch.tensor(
            np.asarray(prior_lattice_mask), dtype=torch.bool
        ),
    )
    return CorruptedBatch(
        inputs=inputs,
        targets=targets,
        prior_targets=prior_targets,
        metadata=tuple(metadata),
    )
