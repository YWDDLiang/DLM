"""Symmetry-compatible lattice and Wyckoff free-coordinate charts."""

from __future__ import annotations

import dataclasses
import itertools
import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from .bridge import ChartCatalog, ChartSpec


CRYSTAL_SYSTEM_DIMENSIONS = {
    "triclinic": 6,
    "monoclinic": 4,
    "orthorhombic": 3,
    "tetragonal": 2,
    "trigonal": 2,
    "hexagonal": 2,
    "cubic": 1,
}


def _lengths_angles(matrix: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("lattice matrix must be 3x3")
    lengths = np.linalg.norm(matrix, axis=1)
    if np.any(lengths <= 0) or not np.all(np.isfinite(lengths)):
        raise ValueError("lattice lengths must be finite and positive")

    def angle(u: np.ndarray, v: np.ndarray) -> float:
        cosine = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    alpha = angle(matrix[1], matrix[2])
    beta = angle(matrix[0], matrix[2])
    gamma = angle(matrix[0], matrix[1])
    return tuple(float(v) for v in lengths), (alpha, beta, gamma)


def _matrix_from_parameters(
    a: float,
    b: float,
    c: float,
    alpha_deg: float,
    beta_deg: float,
    gamma_deg: float,
) -> np.ndarray:
    alpha, beta, gamma = map(math.radians, (alpha_deg, beta_deg, gamma_deg))
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1.0e-10:
        raise ValueError("degenerate lattice angle")
    va = np.array([a, 0.0, 0.0], dtype=np.float64)
    vb = np.array([b * math.cos(gamma), b * sin_gamma, 0.0], dtype=np.float64)
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    cz_sq = c * c - cx * cx - cy * cy
    if cz_sq <= 0:
        raise ValueError("lattice parameters are not positive definite")
    vc = np.array([cx, cy, math.sqrt(cz_sq)], dtype=np.float64)
    return np.stack([va, vb, vc])


class LatticeChartCodec:
    """Ragged 1--6D lattice charts with positive-definite decoding."""

    beta_min_deg = 30.0
    beta_max_deg = 150.0

    @classmethod
    def dimension(cls, crystal_system: str) -> int:
        try:
            return CRYSTAL_SYSTEM_DIMENSIONS[crystal_system.lower()]
        except KeyError as exc:
            raise ValueError(f"unsupported crystal system: {crystal_system}") from exc

    @classmethod
    def encode_matrix(cls, matrix: Sequence[Sequence[float]], crystal_system: str) -> tuple[float, ...]:
        system = crystal_system.lower()
        array = np.asarray(matrix, dtype=np.float64)
        lengths, angles = _lengths_angles(array)
        a, b, c = lengths
        if system == "triclinic":
            gram = array @ array.T
            lower = np.linalg.cholesky(gram)
            return (
                math.log(float(lower[0, 0])),
                math.log(float(lower[1, 1])),
                math.log(float(lower[2, 2])),
                float(lower[1, 0]),
                float(lower[2, 0]),
                float(lower[2, 1]),
            )
        if system == "monoclinic":
            beta = angles[1]
            fraction = (beta - cls.beta_min_deg) / (cls.beta_max_deg - cls.beta_min_deg)
            if not 0.0 < fraction < 1.0:
                raise ValueError(f"monoclinic beta outside registered bounds: {beta}")
            return math.log(a), math.log(b), math.log(c), math.log(fraction / (1.0 - fraction))
        if system == "orthorhombic":
            return math.log(a), math.log(b), math.log(c)
        if system in {"tetragonal", "hexagonal", "trigonal"}:
            # Standard conventional hexagonal setting is used for trigonal SGs.
            return math.log(math.sqrt(a * b)), math.log(c)
        if system == "cubic":
            return (math.log((a * b * c) ** (1.0 / 3.0)),)
        raise ValueError(f"unsupported crystal system: {crystal_system}")

    @classmethod
    def decode_matrix(cls, chart: Sequence[float], crystal_system: str) -> np.ndarray:
        system = crystal_system.lower()
        values = tuple(float(value) for value in chart)
        if len(values) != cls.dimension(system) or not all(math.isfinite(v) for v in values):
            raise ValueError("lattice chart has the wrong dimension or non-finite values")
        if system == "triclinic":
            lower = np.array(
                [
                    [math.exp(values[0]), 0.0, 0.0],
                    [values[3], math.exp(values[1]), 0.0],
                    [values[4], values[5], math.exp(values[2])],
                ],
                dtype=np.float64,
            )
            # Rows of this representative realize the encoded Gram matrix.
            return lower
        if system == "monoclinic":
            a, b, c = (math.exp(value) for value in values[:3])
            fraction = 1.0 / (1.0 + math.exp(-values[3]))
            beta = cls.beta_min_deg + fraction * (cls.beta_max_deg - cls.beta_min_deg)
            return _matrix_from_parameters(a, b, c, 90.0, beta, 90.0)
        if system == "orthorhombic":
            a, b, c = (math.exp(value) for value in values)
            return _matrix_from_parameters(a, b, c, 90.0, 90.0, 90.0)
        if system == "tetragonal":
            a, c = (math.exp(value) for value in values)
            return _matrix_from_parameters(a, a, c, 90.0, 90.0, 90.0)
        if system in {"hexagonal", "trigonal"}:
            a, c = (math.exp(value) for value in values)
            return _matrix_from_parameters(a, a, c, 90.0, 90.0, 120.0)
        if system == "cubic":
            a = math.exp(values[0])
            return _matrix_from_parameters(a, a, a, 90.0, 90.0, 90.0)
        raise ValueError(f"unsupported crystal system: {crystal_system}")


def periodic_cartesian_distance(
    first: Sequence[float],
    second: Sequence[float],
    lattice_matrix: Sequence[Sequence[float]],
) -> float:
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    delta -= np.rint(delta)
    return float(np.linalg.norm(delta @ np.asarray(lattice_matrix, dtype=np.float64)))


@dataclasses.dataclass(frozen=True, slots=True)
class AffineOrbitChart:
    """Local affine chart of a Wyckoff orbit representative."""

    origin: tuple[float, float, float]
    basis: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    dimension: int
    fit_residual: float

    def __post_init__(self) -> None:
        matrix = np.asarray(self.basis, dtype=np.float64)
        if matrix.shape != (3, self.dimension):
            raise ValueError("basis must have shape (3, dimension)")
        if self.dimension not in {0, 1, 2, 3}:
            raise ValueError("invalid orbit dimension")
        if self.fit_residual < 0 or not math.isfinite(self.fit_residual):
            raise ValueError("invalid fit residual")

    def encode(self, fractional_coordinate: Sequence[float]) -> tuple[float, ...]:
        if self.dimension == 0:
            return ()
        point = np.asarray(fractional_coordinate, dtype=np.float64)
        origin = np.asarray(self.origin, dtype=np.float64)
        basis = np.asarray(self.basis, dtype=np.float64)
        projector = np.linalg.pinv(basis, rcond=1.0e-12)
        best: tuple[float, np.ndarray] | None = None
        for shift in itertools.product((-1.0, 0.0, 1.0), repeat=3):
            delta = point + np.asarray(shift) - origin
            # PyXtal's affine Wyckoff basis is generally full-column-rank but
            # not orthonormal (for example, one free variable may drive two
            # fractional axes).  The pseudoinverse is therefore required;
            # ``basis.T`` is valid only for the diagnostic SVD chart below.
            q = projector @ delta
            residual = float(np.linalg.norm(delta - basis @ q))
            if best is None or residual < best[0]:
                best = residual, q
        assert best is not None
        return tuple(float(value) for value in best[1])

    def decode(self, free_coordinate: Sequence[float]) -> tuple[float, float, float]:
        values = np.asarray(free_coordinate, dtype=np.float64)
        if values.shape != (self.dimension,):
            raise ValueError("free-coordinate dimension mismatch")
        point = np.asarray(self.origin) + np.asarray(self.basis) @ values
        point %= 1.0
        return tuple(float(value) for value in point)

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": list(self.origin),
            "basis": [list(row) for row in self.basis],
            "dimension": self.dimension,
            "fit_residual": self.fit_residual,
        }


def derive_affine_orbit_chart(
    representative: Sequence[float],
    symmetry_operations: Sequence[Any],
    lattice_matrix: Sequence[Sequence[float]],
    *,
    symprec: float,
) -> AffineOrbitChart:
    """Derive the site-symmetry nullspace and one compatible affine origin."""

    point = np.asarray(representative, dtype=np.float64)
    lattice = np.asarray(lattice_matrix, dtype=np.float64)
    constraints: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for operation in symmetry_operations:
        rotation = np.asarray(operation.rotation_matrix, dtype=np.float64)
        translation = np.asarray(operation.translation_vector, dtype=np.float64)
        transformed = rotation @ point + translation
        if periodic_cartesian_distance(transformed, point, lattice) <= symprec:
            integer_shift = np.rint(transformed - point)
            constraints.append(rotation - np.eye(3))
            targets.append(integer_shift - translation)
    if not constraints:
        raise ValueError("site stabilizer unexpectedly empty")
    matrix = np.concatenate(constraints, axis=0)
    target = np.concatenate(targets, axis=0)
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    threshold = max(matrix.shape) * (singular[0] if singular.size else 1.0) * 1.0e-10
    rank = int(np.sum(singular > threshold))
    basis = vh[rank:].T
    origin, *_ = np.linalg.lstsq(matrix, target, rcond=1.0e-12)
    dimension = int(basis.shape[1])
    if dimension:
        # SVD already yields an orthonormal nullspace; normalize once to remove
        # tiny LAPACK differences from serialized charts.
        basis, _ = np.linalg.qr(basis)
    chart = AffineOrbitChart(
        origin=tuple(float(value) for value in origin),
        basis=tuple(tuple(float(value) for value in row) for row in basis),
        dimension=dimension,
        fit_residual=0.0,
    )
    q = chart.encode(point)
    decoded = chart.decode(q)
    residual = periodic_cartesian_distance(decoded, point, lattice)
    return dataclasses.replace(chart, fit_residual=residual)


def expand_representative(
    representative: Sequence[float],
    symmetry_operations: Sequence[Any],
    lattice_matrix: Sequence[Sequence[float]],
    *,
    symprec: float,
) -> tuple[tuple[float, float, float], ...]:
    lattice = np.asarray(lattice_matrix, dtype=np.float64)
    result: list[tuple[float, float, float]] = []
    for operation in symmetry_operations:
        point = np.asarray(operation.operate(representative), dtype=np.float64) % 1.0
        if not any(periodic_cartesian_distance(point, other, lattice) <= symprec for other in result):
            result.append(tuple(float(value) for value in point))
    return tuple(sorted(result, key=lambda value: tuple(round(item, 12) for item in value)))


def expand_representative_with_jacobians(
    representative: Sequence[float],
    symmetry_operations: Sequence[Any],
    lattice_matrix: Sequence[Sequence[float]],
    chart_basis: Sequence[Sequence[float]],
    *,
    symprec: float,
) -> tuple[tuple[tuple[float, float, float], tuple[tuple[float, ...], ...]], ...]:
    """Expand one generator and retain ``d x_atom / d q`` for each image."""

    lattice = np.asarray(lattice_matrix, dtype=np.float64)
    basis = np.asarray(chart_basis, dtype=np.float64)
    if basis.shape[0] != 3:
        raise ValueError("chart basis must have three fractional-coordinate rows")
    result: list[tuple[tuple[float, float, float], tuple[tuple[float, ...], ...]]] = []
    for operation in symmetry_operations:
        point = np.asarray(operation.operate(representative), dtype=np.float64) % 1.0
        if any(periodic_cartesian_distance(point, other[0], lattice) <= symprec for other in result):
            continue
        jacobian = np.asarray(operation.rotation_matrix, dtype=np.float64) @ basis
        result.append(
            (
                tuple(float(value) for value in point),
                tuple(tuple(float(value) for value in row) for row in jacobian),
            )
        )
    return tuple(
        sorted(result, key=lambda value: tuple(round(item, 12) for item in value[0]))
    )


def wyckoff_letter_to_type(letter: str) -> int:
    normalized = letter.strip()
    if len(normalized) != 1:
        raise ValueError(f"invalid Wyckoff letter: {letter}")
    if "a" <= normalized <= "z":
        return ord(normalized) - ord("a")
    # SG 47 (Pmmm) has 27 positions.  ITA/PyXtal names its general position
    # ``A`` after exhausting a..z; lower-casing would silently alias it to 1a.
    if normalized == "A":
        return 26
    raise ValueError(f"invalid Wyckoff letter: {letter}")


def wyckoff_type_to_letter(value: int) -> str:
    if not 0 <= int(value) < 27:
        raise ValueError(f"invalid Wyckoff type index: {value}")
    if int(value) == 26:
        return "A"
    return chr(ord("a") + int(value))


def centering_factor_from_symbol(symbol: str) -> int:
    match = re.search(r"[PABCIFR]", symbol.upper())
    if not match:
        raise ValueError(f"cannot determine lattice centering from symbol: {symbol}")
    return {"P": 1, "A": 2, "B": 2, "C": 2, "I": 2, "R": 3, "F": 4}[match.group(0)]


class PyXtalChartCatalog(ChartCatalog):
    """Lazy PyXtal metadata adapter using ITA letter indices as type IDs.

    The catalogue is deliberately pinned to PyXtal's ``spglib`` Hall style so
    that it shares the setting used by pymatgen's conventional structures.
    Preprocessing verifies the Hall number instead of assuming the settings
    happen to agree.
    """

    def __init__(self, *, hall_style: str = "spglib") -> None:
        if hall_style != "spglib":
            raise ValueError("the active protocol requires the spglib Hall style")
        self.hall_style = hall_style
        self._groups: dict[int, Any] = {}
        # PyXtal group metadata is immutable for a pinned Hall style.  Reverse
        # sampling queries the same positions and DoF thousands of times per
        # batch, so cache the validated adapters instead of repeatedly
        # rebuilding dictionaries and recomputing matrix ranks.
        self._positions_by_group: dict[int, dict[int, Any]] = {}
        self._specs: dict[tuple[int, int], ChartSpec] = {}
        self._types_by_group: dict[int, tuple[int, ...]] = {}
        self._unit_affine_charts: dict[tuple[int, int], AffineOrbitChart] = {}

    def _group(self, space_group: int) -> Any:
        if space_group not in self._groups:
            try:
                from pyxtal.symmetry import Group
            except ImportError as exc:  # pragma: no cover - server dependency
                raise RuntimeError("pyxtal==1.1.4 is required for the chart catalogue") from exc
            self._groups[space_group] = Group(space_group, style=self.hall_style)
        return self._groups[space_group]

    @staticmethod
    def _label(wp: Any) -> str:
        for candidate in (
            getattr(wp, "get_label", lambda: "")(),
            getattr(wp, "letter", ""),
        ):
            match = re.search(r"([a-zA-Z])\s*$", str(candidate))
            if match:
                return match.group(1)
        match = re.search(r"([a-zA-Z])\b", str(wp))
        if not match:
            raise ValueError(f"cannot determine Wyckoff label from {wp!r}")
        return match.group(1)

    def _positions(self, space_group: int) -> dict[int, Any]:
        key = int(space_group)
        cached = self._positions_by_group.get(key)
        if cached is None:
            positions = getattr(self._group(key), "Wyckoff_positions")
            cached = {
                wyckoff_letter_to_type(self._label(wp)): wp for wp in positions
            }
            self._positions_by_group[key] = cached
        return cached

    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        key = (int(space_group), int(wyckoff_type))
        cached = self._specs.get(key)
        if cached is not None:
            return cached
        wp = self._positions(key[0])[key[1]]
        letter = self._label(wp)
        multiplicity = int(getattr(wp, "multiplicity", len(getattr(wp, "ops", ()))))
        dimension = int(wp.get_dof())
        group = self._group(key[0])
        symbol = str(getattr(group, "symbol", group))
        factor = centering_factor_from_symbol(symbol)
        if multiplicity % factor:
            raise ValueError(f"Wyckoff multiplicity {multiplicity} not divisible by centering {factor}")
        cached = ChartSpec(
            key[0],
            key[1],
            letter,
            multiplicity,
            dimension,
            multiplicity // factor,
        )
        self._specs[key] = cached
        return cached

    def types(self, space_group: int) -> tuple[int, ...]:
        key = int(space_group)
        cached = self._types_by_group.get(key)
        if cached is None:
            cached = tuple(sorted(self._positions(key)))
            self._types_by_group[key] = cached
        return cached

    def hall_number(self, space_group: int) -> int:
        value = getattr(self._group(space_group), "hall_number", None)
        if value is None:
            raise ValueError(f"PyXtal did not expose a Hall number for SG {space_group}")
        return int(value)

    def encode_free(self, space_group: int, wyckoff_type: int, position: Sequence[float]) -> tuple[float, ...]:
        wp = self._positions(space_group)[int(wyckoff_type)]
        values = tuple(float(value) % 1.0 for value in wp.get_free_xyzs(position))
        expected = self.get(space_group, wyckoff_type).dimension
        if len(values) != expected:
            raise ValueError(
                f"PyXtal free-coordinate dimension mismatch for SG {space_group} "
                f"Wyckoff {wyckoff_type}: {len(values)} != {expected}"
            )
        return values

    def decode_generator(
        self, space_group: int, wyckoff_type: int, free_coordinate: Sequence[float]
    ) -> tuple[float, float, float]:
        wp = self._positions(space_group)[int(wyckoff_type)]
        value = wp.get_position_from_free_xyzs(list(free_coordinate))
        return tuple(float(item) % 1.0 for item in value)

    def expand(
        self, space_group: int, wyckoff_type: int, free_coordinate: Sequence[float]
    ) -> tuple[tuple[float, float, float], ...]:
        wp = self._positions(space_group)[int(wyckoff_type)]
        generator = self.decode_generator(space_group, wyckoff_type, free_coordinate)
        values = wp.get_all_positions(generator)
        if values is None:
            raise ValueError(
                f"PyXtal could not expand SG {space_group} Wyckoff {wyckoff_type}"
            )
        result = tuple(tuple(float(item) % 1.0 for item in point) for point in values)
        expected = self.get(space_group, wyckoff_type).multiplicity
        if len(result) != expected:
            raise ValueError(
                f"PyXtal expansion multiplicity mismatch for SG {space_group} "
                f"Wyckoff {wyckoff_type}: {len(result)} != {expected}"
            )
        return tuple(
            sorted(result, key=lambda point: tuple(round(value, 12) for value in point))
        )

    def expand_with_jacobians(
        self,
        space_group: int,
        wyckoff_type: int,
        free_coordinate: Sequence[float],
    ) -> tuple[
        tuple[tuple[float, float, float], tuple[tuple[float, ...], ...]], ...
    ]:
        """Expand a generator together with exact ``d x_atom / d q`` maps."""

        wp = self._positions(space_group)[int(wyckoff_type)]
        spec = self.get(space_group, wyckoff_type)
        key = (int(space_group), int(wyckoff_type))
        chart = self._unit_affine_charts.get(key)
        if chart is None:
            chart = self.affine_chart(key[0], key[1], np.eye(3))
            self._unit_affine_charts[key] = chart
        generator = self.decode_generator(space_group, wyckoff_type, free_coordinate)
        basis = np.asarray(chart.basis, dtype=np.float64)
        pairs = []
        for operation in wp.ops:
            point = np.asarray(operation.operate(generator), dtype=np.float64) % 1.0
            jacobian = np.asarray(operation.rotation_matrix, dtype=np.float64) @ basis
            pairs.append(
                (
                    tuple(float(value) for value in point),
                    tuple(tuple(float(value) for value in row) for row in jacobian),
                )
            )
        pairs.sort(key=lambda item: tuple(round(value, 12) for value in item[0]))
        if len(pairs) != spec.multiplicity:
            raise ValueError(
                f"PyXtal Jacobian expansion multiplicity mismatch for SG {space_group} "
                f"Wyckoff {wyckoff_type}: {len(pairs)} != {spec.multiplicity}"
            )
        return tuple(pairs)

    def affine_chart(
        self,
        space_group: int,
        wyckoff_type: int,
        lattice_matrix: Sequence[Sequence[float]],
    ) -> AffineOrbitChart:
        """Return the exact affine chart used by PyXtal 1.1.4.

        ``get_position_from_free_xyzs`` inserts the free values on the
        non-frozen Cartesian-fractional axes and applies ``wp.ops[0]``.  Its
        Jacobian is consequently the corresponding columns of that operator's
        rotation matrix.  Several deterministic probes guard this contract
        against dependency/API drift.
        """

        wp = self._positions(space_group)[int(wyckoff_type)]
        spec = self.get(space_group, wyckoff_type)
        frozen = {int(axis) for axis in wp.get_frozen_axis()}
        free_axes = tuple(axis for axis in range(3) if axis not in frozen)
        if len(free_axes) != spec.dimension:
            raise ValueError(
                f"PyXtal frozen-axis/DoF mismatch for SG {space_group} "
                f"Wyckoff {wyckoff_type}"
            )
        operation = wp.ops[0]
        rotation = np.asarray(operation.rotation_matrix, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("PyXtal Wyckoff affine operator is not 3x3")
        basis = rotation[:, free_axes]
        if spec.dimension and np.linalg.matrix_rank(basis, tol=1.0e-12) != spec.dimension:
            raise ValueError(
                f"rank-deficient PyXtal chart for SG {space_group} Wyckoff {wyckoff_type}"
            )
        origin = self.decode_generator(space_group, wyckoff_type, (0.0,) * spec.dimension)
        lattice = np.asarray(lattice_matrix, dtype=np.float64)
        probes: list[tuple[float, ...]] = [(0.0,) * spec.dimension]
        for axis in range(spec.dimension):
            value = [0.0] * spec.dimension
            value[axis] = 0.137
            probes.append(tuple(value))
        if spec.dimension:
            probes.append(tuple((axis + 1) * 0.173 % 1.0 for axis in range(spec.dimension)))
        residual = 0.0
        for free in probes:
            exact = self.decode_generator(space_group, wyckoff_type, free)
            affine = (np.asarray(origin) + basis @ np.asarray(free, dtype=np.float64)) % 1.0
            residual = max(
                residual,
                periodic_cartesian_distance(exact, affine, lattice),
            )
        if residual >= 1.0e-6:
            raise ValueError(
                f"PyXtal affine-chart residual {residual:.3e} Å is not <1e-6 for "
                f"SG {space_group} Wyckoff {wyckoff_type}"
            )
        return AffineOrbitChart(
            origin=tuple(float(value) for value in origin),
            basis=tuple(tuple(float(value) for value in row) for row in basis),
            dimension=spec.dimension,
            fit_residual=residual,
        )
