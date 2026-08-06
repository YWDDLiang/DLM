"""MP20 -> stratified Wyckoff-quotient preprocessing and audit records."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .charts import (
    LatticeChartCodec,
    PyXtalChartCatalog,
    expand_representative,
    expand_representative_with_jacobians,
    periodic_cartesian_distance,
    wyckoff_letter_to_type,
)
from .state import OrbitState, StratifiedState
from .formal import regularized_projector_error


SYMPREC_GRID = (1.0e-3, 1.0e-2, 1.0e-1)
PRIMARY_SYMPREC = 1.0e-2
ANGLE_TOLERANCE_DEG = 5.0
P1_COVERAGE_MIN = 0.95
P1_ROUNDTRIP_MIN = 0.99
P1_ATOM_COUNT_CONSISTENCY = 1.0
MP20_TOTAL_RECORDS = 45_229


def tolerance_tag(symprec: float) -> str:
    return f"symprec_{symprec:.0e}".replace("+", "")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _point_set_residual(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
    lattice_matrix: Sequence[Sequence[float]],
) -> float:
    """Symmetric worst-nearest periodic distance between two finite sets."""

    if len(first) != len(second) or not first:
        return math.inf
    return max(
        max(
            min(
                periodic_cartesian_distance(point, candidate, lattice_matrix)
                for candidate in second
            )
            for point in first
        ),
        max(
            min(
                periodic_cartesian_distance(point, candidate, lattice_matrix)
                for candidate in first
            )
            for point in second
        ),
    )


def material_family_from_symbols(symbols: Iterable[str]) -> str:
    """Apply the one registered, mutually-exclusive material-family rule."""

    values = {str(value) for value in symbols}
    anion_groups = {
        "hydride": {"H"},
        "oxide": {"O"},
        "halide": {"F", "Cl", "Br", "I"},
        "chalcogenide": {"S", "Se", "Te"},
        "pnictide": {"N", "P", "As", "Sb", "Bi"},
    }
    present = [name for name, members in anion_groups.items() if values & members]
    if len(present) == 1:
        return present[0]
    if len(present) > 1:
        return "mixed_anion"
    if values:
        try:
            from pymatgen.core.periodic_table import Element

            if all(bool(Element(symbol).is_metal) for symbol in values):
                return "intermetallic"
        except (ImportError, ValueError):
            pass
    return "other"


@dataclasses.dataclass(frozen=True, slots=True)
class OrbitDecomposition:
    orbit: OrbitState
    wyckoff_symbol: str
    representative: tuple[float, float, float]
    chart_origin: tuple[float, float, float]
    chart_basis: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    chart_fit_residual_angstrom: float
    expanded_fractional_coordinates: tuple[tuple[float, float, float], ...]
    expanded_chart_jacobians: tuple[tuple[tuple[float, ...], ...], ...]
    primitive_fractional_coordinates: tuple[tuple[float, float, float], ...]
    primitive_chart_jacobians: tuple[tuple[tuple[float, ...], ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "orbit": self.orbit.to_dict(),
            "wyckoff_symbol": self.wyckoff_symbol,
            "representative": list(self.representative),
            "chart_origin": list(self.chart_origin),
            "chart_basis": [list(row) for row in self.chart_basis],
            "chart_fit_residual_angstrom": self.chart_fit_residual_angstrom,
            "expanded_fractional_coordinates": [
                list(point) for point in self.expanded_fractional_coordinates
            ],
            "expanded_chart_jacobians": [
                [list(row) for row in jacobian] for jacobian in self.expanded_chart_jacobians
            ],
            "primitive_fractional_coordinates": [
                list(point) for point in self.primitive_fractional_coordinates
            ],
            "primitive_chart_jacobians": [
                [list(row) for row in jacobian] for jacobian in self.primitive_chart_jacobians
            ],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ToleranceDecomposition:
    symprec: float
    state: StratifiedState
    hall_number: int
    international_symbol: str
    transformation_matrix: tuple[tuple[float, float, float], ...]
    origin_shift: tuple[float, float, float]
    centering_factor: int
    primitive_lattice_transform: tuple[tuple[float, float, float], ...]
    primitive_structure: Mapping[str, Any]
    conventional_structure: Mapping[str, Any]
    orbits: tuple[OrbitDecomposition, ...]
    roundtrip_match: bool
    multiplicity_consistent: bool
    roundtrip_rms_angstrom: float
    flags: tuple[str, ...]

    @property
    def canonical_hash(self) -> str:
        return self.state.topology_hash(include_geometry=True)

    def to_dict(self) -> dict[str, Any]:
        ordered_orbits = tuple(sorted(self.orbits, key=lambda value: value.orbit.storage_key()))
        return {
            "symprec": self.symprec,
            "state": self.state.to_dict(),
            "hall_number": self.hall_number,
            "international_symbol": self.international_symbol,
            "transformation_matrix": [list(row) for row in self.transformation_matrix],
            "origin_shift": list(self.origin_shift),
            "centering_factor": self.centering_factor,
            "primitive_lattice_transform": [
                list(row) for row in self.primitive_lattice_transform
            ],
            "primitive_structure": dict(self.primitive_structure),
            "conventional_structure": dict(self.conventional_structure),
            "orbits": [orbit.to_dict() for orbit in ordered_orbits],
            "roundtrip_match": self.roundtrip_match,
            "multiplicity_consistent": self.multiplicity_consistent,
            "roundtrip_rms_angstrom": self.roundtrip_rms_angstrom,
            "canonical_hash": self.canonical_hash,
            "flags": list(self.flags),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class WQDatasetRecord:
    material_id: str
    split: str
    source_cif_hash: str
    source_elements: tuple[str, ...]
    material_family: str
    primary_symprec: float
    selected: bool
    ambiguous: bool
    primary_failure_reason: str
    decompositions: tuple[ToleranceDecomposition, ...]

    @property
    def primary(self) -> ToleranceDecomposition | None:
        return next(
            (item for item in self.decompositions if abs(item.symprec - self.primary_symprec) < 1.0e-15),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mp20_wq_v1",
            "material_id": self.material_id,
            "split": self.split,
            "source_cif_hash": self.source_cif_hash,
            "source_elements": list(self.source_elements),
            "material_family": self.material_family,
            "primary_symprec": self.primary_symprec,
            "selected": self.selected,
            "ambiguous": self.ambiguous,
            "primary_failure_reason": self.primary_failure_reason,
            "decompositions": {
                tolerance_tag(item.symprec): item.to_dict() for item in self.decompositions
            },
        }


def _dataset_value(dataset: Any, name: str, default: Any) -> Any:
    if hasattr(dataset, name):
        return getattr(dataset, name)
    if isinstance(dataset, Mapping):
        return dataset.get(name, default)
    return default


class PymatgenWyckoffCodec:
    """Numerical conventional-cell decomposition with explicit chart traces."""

    def __init__(
        self,
        *,
        angle_tolerance: float = ANGLE_TOLERANCE_DEG,
        chart_catalog: PyXtalChartCatalog | None = None,
    ) -> None:
        self.angle_tolerance = float(angle_tolerance)
        self.chart_catalog = chart_catalog or PyXtalChartCatalog()

    @staticmethod
    def _dependencies() -> tuple[Any, Any, Any, Any]:
        try:
            from pymatgen.analysis.structure_matcher import StructureMatcher
            from pymatgen.core import Structure
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            from pymatgen.core.periodic_table import Element
        except ImportError as exc:  # pragma: no cover - exercised on server
            raise RuntimeError("pymatgen is required for MP20 preprocessing") from exc
        return Structure, SpacegroupAnalyzer, StructureMatcher, Element

    def from_cif(
        self,
        *,
        cif: str,
        material_id: str,
        split: str,
        symprec_grid: Sequence[float] = SYMPREC_GRID,
        primary_symprec: float = PRIMARY_SYMPREC,
    ) -> WQDatasetRecord:
        Structure, _, _, _ = self._dependencies()
        source = Structure.from_str(cif, fmt="cif")
        source_elements = tuple(
            sorted(str(element.symbol) for element in source.composition.elements)
        )
        decompositions: list[ToleranceDecomposition] = []
        failures: dict[float, str] = {}
        for symprec in symprec_grid:
            try:
                decompositions.append(self.decompose(source, symprec=float(symprec)))
            except Exception as exc:
                failures[float(symprec)] = f"{type(exc).__name__}:{exc}"
        primary = next(
            (item for item in decompositions if abs(item.symprec - primary_symprec) < 1.0e-15),
            None,
        )
        selected = bool(
            primary is not None and primary.roundtrip_match and primary.multiplicity_consistent
        )
        hashes = {item.canonical_hash for item in decompositions}
        ambiguous = len(hashes) > 1 or len(decompositions) != len(tuple(symprec_grid))
        reason = "" if selected else failures.get(primary_symprec, "primary_roundtrip_or_multiplicity")
        return WQDatasetRecord(
            material_id=material_id,
            split=split,
            source_cif_hash=_sha256_text(cif),
            source_elements=source_elements,
            material_family=material_family_from_symbols(source_elements),
            primary_symprec=primary_symprec,
            selected=selected,
            ambiguous=ambiguous,
            primary_failure_reason=reason,
            decompositions=tuple(sorted(decompositions, key=lambda item: item.symprec)),
        )

    def decompose(self, structure: Any, *, symprec: float) -> ToleranceDecomposition:
        _, SpacegroupAnalyzer, StructureMatcher, _ = self._dependencies()
        if len(structure) < 1:
            raise ValueError("empty crystal")
        if not structure.is_ordered:
            raise ValueError("fractional occupancy/disorder is out of scope")

        source_analyzer = SpacegroupAnalyzer(
            structure,
            symprec=symprec,
            angle_tolerance=self.angle_tolerance,
        )
        source_dataset = source_analyzer.get_symmetry_dataset()
        # PyXtal's Wyckoff tables use the International Tables/spglib setting.
        # pymatgen's ``get_conventional_standard_structure`` instead applies
        # the Setyawan--Curtarolo band-structure convention and may permute
        # orthorhombic axes while retaining the same HM label (e.g. Pnma).
        # ``get_refined_structure`` is pymatgen's documented ITA conventional
        # path and is therefore the only registered quotient representation.
        conventional = source_analyzer.get_refined_structure()
        conventional_analyzer = SpacegroupAnalyzer(
            conventional,
            symprec=symprec,
            angle_tolerance=self.angle_tolerance,
        )
        primitive = conventional_analyzer.find_primitive()
        if not 1 <= len(primitive) <= 20:
            raise ValueError(f"MP20 primitive atom count outside [1,20]: {len(primitive)}")
        if len(conventional) % len(primitive):
            raise ValueError("conventional/primitive atom counts have non-integral centering ratio")
        centering_factor = len(conventional) // len(primitive)
        if centering_factor not in {1, 2, 3, 4}:
            raise ValueError(f"unexpected centering factor: {centering_factor}")
        conventional_matrix = np.asarray(conventional.lattice.matrix, dtype=np.float64)
        primitive_matrix = np.asarray(primitive.lattice.matrix, dtype=np.float64)
        conventional_to_primitive = conventional_matrix @ np.linalg.inv(primitive_matrix)
        primitive_lattice_transform = primitive_matrix @ np.linalg.inv(conventional_matrix)
        analyzer = SpacegroupAnalyzer(
            conventional,
            symprec=symprec,
            angle_tolerance=self.angle_tolerance,
        )
        symmetrized = analyzer.get_symmetrized_structure()
        operations = analyzer.get_symmetry_operations(cartesian=False)
        dataset = analyzer.get_symmetry_dataset()
        space_group = int(analyzer.get_space_group_number())
        hall_number = int(_dataset_value(dataset, "hall_number", 0))
        pyxtal_hall_number = self.chart_catalog.hall_number(space_group)
        if hall_number != pyxtal_hall_number:
            raise ValueError(
                f"pymatgen/PyXtal Hall-setting mismatch for SG {space_group}: "
                f"{hall_number} != {pyxtal_hall_number}"
            )
        crystal_system = str(analyzer.get_crystal_system())
        lattice_chart = LatticeChartCodec.encode_matrix(conventional.lattice.matrix, crystal_system)
        orbit_records: list[OrbitDecomposition] = []
        rebuilt_species: list[str] = []
        rebuilt_coords: list[tuple[float, float, float]] = []

        for serial, (indices, symbol) in enumerate(
            zip(symmetrized.equivalent_indices, symmetrized.wyckoff_symbols)
        ):
            sites = [conventional[index] for index in indices]
            species = {site.specie.symbol for site in sites}
            if len(species) != 1:
                raise ValueError("one symmetry orbit contains multiple ordered species")
            element = next(iter(species))
            match = re.search(r"([a-zA-Z])\s*$", str(symbol))
            if not match:
                raise ValueError(f"invalid Wyckoff symbol: {symbol}")
            wyckoff_type = wyckoff_letter_to_type(match.group(1))
            spec = self.chart_catalog.get(space_group, wyckoff_type)
            if spec.letter != match.group(1):
                raise ValueError("pymatgen/PyXtal Wyckoff-letter mismatch")
            if spec.multiplicity != len(indices):
                raise ValueError(
                    f"pymatgen/PyXtal orbit multiplicity mismatch for {symbol}: "
                    f"{len(indices)} != {spec.multiplicity}"
                )
            expected_primitive_multiplicity = len(indices) // centering_factor
            if spec.primitive_multiplicity != expected_primitive_multiplicity:
                raise ValueError(
                    f"pymatgen/PyXtal primitive multiplicity mismatch for {symbol}: "
                    f"{expected_primitive_multiplicity} != {spec.primitive_multiplicity}"
                )
            target_points = tuple(
                tuple(float(value) % 1.0 for value in site.frac_coords) for site in sites
            )
            # This is the protocol's single coordinate convention.  Both
            # preprocessing and runtime call PyXtal's official free-coordinate
            # methods; the custom SVD chart remains a diagnostic utility only.
            candidates: list[tuple[float, tuple[float, ...], Any, tuple[float, ...], tuple[float, float, float], tuple[tuple[float, float, float], ...]]] = []
            for site in sites:
                representative = tuple(
                    float(value) % 1.0 for value in site.frac_coords
                )
                try:
                    candidate_free = self.chart_catalog.encode_free(
                        space_group, wyckoff_type, representative
                    )
                    candidate_generator = self.chart_catalog.decode_generator(
                        space_group, wyckoff_type, candidate_free
                    )
                    candidate_expanded = self.chart_catalog.expand(
                        space_group, wyckoff_type, candidate_free
                    )
                    residual = _point_set_residual(
                        target_points, candidate_expanded, conventional_matrix
                    )
                    candidates.append(
                        (
                            residual,
                            tuple(round(value, 12) for value in representative),
                            site,
                            candidate_free,
                            candidate_generator,
                            candidate_expanded,
                        )
                    )
                except Exception:
                    continue
            if not candidates:
                raise ValueError(f"PyXtal could not encode any representative for {symbol}")
            (
                source_projection_residual,
                _,
                representative_site,
                free,
                generator,
                pyxtal_expanded,
            ) = min(candidates, key=lambda value: (value[0], value[1]))
            chart = self.chart_catalog.affine_chart(
                space_group, wyckoff_type, conventional.lattice.matrix
            )
            operation_dedup_tolerance = 1.0e-7
            expanded = expand_representative(
                generator,
                operations,
                conventional.lattice.matrix,
                symprec=operation_dedup_tolerance,
            )
            if len(expanded) != len(indices):
                raise ValueError(
                    f"orbit multiplicity mismatch for {symbol}: {len(expanded)} != {len(indices)}"
                )
            pyxtal_pymatgen_residual = _point_set_residual(
                pyxtal_expanded, expanded, conventional_matrix
            )
            source_projection_residual = max(
                source_projection_residual,
                _point_set_residual(target_points, expanded, conventional_matrix),
            )
            if pyxtal_pymatgen_residual >= 1.0e-6:
                raise ValueError(
                    f"PyXtal/pymatgen expansion residual {pyxtal_pymatgen_residual:.3e} Å "
                    f"is not <1e-6 for {symbol}"
                )
            if source_projection_residual > symprec:
                raise ValueError(
                    f"Wyckoff projection residual {source_projection_residual:.3e} Å exceeds "
                    f"symprec {symprec:.3e} for {symbol}"
                )
            expanded_with_jacobians = expand_representative_with_jacobians(
                generator,
                operations,
                conventional.lattice.matrix,
                chart.basis,
                symprec=operation_dedup_tolerance,
            )
            if tuple(point for point, _ in expanded_with_jacobians) != expanded:
                raise ValueError("expanded coordinate/Jacobian ordering mismatch")
            primitive_points: list[tuple[float, float, float]] = []
            primitive_jacobians: list[tuple[tuple[float, ...], ...]] = []
            for point, jacobian in expanded_with_jacobians:
                primitive_point = (np.asarray(point) @ conventional_to_primitive) % 1.0
                if any(
                    periodic_cartesian_distance(
                        primitive_point,
                        existing,
                        primitive_matrix,
                    ) <= symprec
                    for existing in primitive_points
                ):
                    continue
                # Coordinates are stored as row vectors while chart Jacobians
                # map q to column fractional coordinates.
                primitive_jacobian = conventional_to_primitive.T @ np.asarray(jacobian)
                primitive_points.append(tuple(float(value) for value in primitive_point))
                primitive_jacobians.append(
                    tuple(tuple(float(value) for value in row) for row in primitive_jacobian)
                )
            if len(indices) % centering_factor or len(primitive_points) != expected_primitive_multiplicity:
                raise ValueError(
                    f"primitive orbit multiplicity mismatch for {symbol}: "
                    f"{len(primitive_points)} != {expected_primitive_multiplicity}"
                )
            atomic_number = int(representative_site.specie.Z)
            orbit = OrbitState(
                orbit_id=f"o{serial}",
                wyckoff_type=wyckoff_type,
                species=atomic_number,
                multiplicity=len(indices),
                chart_dimension=chart.dimension,
                free_coordinate=free,
                primitive_multiplicity=expected_primitive_multiplicity,
            )
            orbit_records.append(
                OrbitDecomposition(
                    orbit=orbit,
                    wyckoff_symbol=str(symbol),
                    representative=generator,
                    chart_origin=chart.origin,
                    chart_basis=chart.basis,
                    chart_fit_residual_angstrom=chart.fit_residual,
                    expanded_fractional_coordinates=expanded,
                    expanded_chart_jacobians=tuple(
                        jacobian for _, jacobian in expanded_with_jacobians
                    ),
                    primitive_fractional_coordinates=tuple(primitive_points),
                    primitive_chart_jacobians=tuple(primitive_jacobians),
                )
            )
            rebuilt_species.extend([element] * len(expanded))
            rebuilt_coords.extend(expanded)

        state = StratifiedState(
            space_group=space_group,
            lattice_system=crystal_system,
            lattice_chart=lattice_chart,
            orbits=tuple(record.orbit for record in orbit_records),
        )
        rebuilt = conventional.__class__(conventional.lattice, rebuilt_species, rebuilt_coords)
        matcher = StructureMatcher(
            ltol=0.2,
            stol=0.3,
            angle_tol=5.0,
            primitive_cell=True,
            scale=True,
            attempt_supercell=False,
        )
        matched = bool(matcher.fit(conventional, rebuilt))
        rms = float("inf")
        if matched:
            rms_result = matcher.get_rms_dist(conventional, rebuilt)
            if rms_result is not None:
                rms = float(rms_result[0])
        flags: list[str] = []
        max_fit = max((record.chart_fit_residual_angstrom for record in orbit_records), default=0.0)
        if max_fit > 1.0e-6:
            flags.append("chart_fit_residual_gt_1e-6")
        if not matched:
            flags.append("roundtrip_structure_mismatch")
        multiplicity_consistent = (
            state.atom_count == len(primitive)
            and state.conventional_atom_count == len(conventional)
        )
        if not multiplicity_consistent:
            flags.append("multiplicity_atom_count_mismatch")

        # Preserve the actual source -> ITA-conventional standardization, not
        # the near-identity re-analysis transform of the already-refined cell.
        transformation = np.asarray(
            _dataset_value(source_dataset, "transformation_matrix", np.eye(3)),
            dtype=np.float64,
        )
        origin_shift = np.asarray(
            _dataset_value(source_dataset, "origin_shift", np.zeros(3))
        )
        return ToleranceDecomposition(
            symprec=symprec,
            state=state,
            hall_number=hall_number,
            international_symbol=str(
                _dataset_value(dataset, "international", analyzer.get_space_group_symbol())
            ),
            transformation_matrix=tuple(tuple(float(v) for v in row) for row in transformation),
            origin_shift=tuple(float(v) for v in origin_shift),
            centering_factor=centering_factor,
            primitive_lattice_transform=tuple(
                tuple(float(value) for value in row) for row in primitive_lattice_transform
            ),
            primitive_structure=primitive.as_dict(),
            conventional_structure=conventional.as_dict(),
            orbits=tuple(orbit_records),
            roundtrip_match=matched,
            multiplicity_consistent=multiplicity_consistent,
            roundtrip_rms_angstrom=rms,
            flags=tuple(flags),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class PreprocessSummary:
    split: str
    rows_seen: int
    rows_written: int
    decoded_records: int
    selected: int
    ambiguous: int
    failures: int
    output_path: str
    output_sha256: str

    @property
    def coverage(self) -> float:
        return self.selected / self.rows_seen if self.rows_seen else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "coverage": self.coverage}


def _fallback_source_metadata(cif: str) -> tuple[tuple[str, ...], str]:
    """Recover composition metadata when quotient decomposition itself fails."""

    try:
        from pymatgen.core import Structure

        structure = Structure.from_str(cif, fmt="cif")
        symbols = tuple(
            sorted(str(element.symbol) for element in structure.composition.elements)
        )
    except Exception:
        symbols = ()
    return symbols, material_family_from_symbols(symbols) if symbols else "unknown"


def preprocess_mp20_csv(
    *,
    csv_path: str | os.PathLike[str],
    split: str,
    output_path: str | os.PathLike[str],
    shard_index: int = 0,
    shard_count: int = 1,
    limit: int | None = None,
    codec: PymatgenWyckoffCodec | None = None,
) -> PreprocessSummary:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    codec = codec or PymatgenWyckoffCodec()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows_seen = rows_written = decoded_records = selected = ambiguous = failures = 0
    with Path(csv_path).open("r", encoding="utf-8", newline="") as source, destination.open(
        "x", encoding="utf-8"
    ) as target:
        for index, row in enumerate(csv.DictReader(source)):
            if index % shard_count != shard_index:
                continue
            if limit is not None and rows_seen >= limit:
                break
            rows_seen += 1
            material_id = str(row.get("material_id") or row.get("id") or f"row-{index}")
            cif = str(row.get("cif.conv") or row.get("cif") or "")
            try:
                record = codec.from_cif(cif=cif, material_id=material_id, split=split)
                payload = record.to_dict()
                decoded_records += 1
                selected += int(record.selected)
                ambiguous += int(record.ambiguous)
            except Exception as exc:
                failures += 1
                source_elements, material_family = _fallback_source_metadata(cif)
                payload = {
                    "schema": "mp20_wq_v1",
                    "material_id": material_id,
                    "split": split,
                    "source_cif_hash": _sha256_text(cif),
                    "source_elements": list(source_elements),
                    "material_family": material_family,
                    "primary_symprec": PRIMARY_SYMPREC,
                    "selected": False,
                    "ambiguous": True,
                    "primary_failure_reason": f"{type(exc).__name__}:{exc}",
                    "decompositions": {},
                }
            target.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            target.flush()
            rows_written += 1
    return PreprocessSummary(
        split=split,
        rows_seen=rows_seen,
        rows_written=rows_written,
        decoded_records=decoded_records,
        selected=selected,
        ambiguous=ambiguous,
        failures=failures,
        output_path=str(destination),
        output_sha256=sha256_file(destination),
    )


def build_hash_fixed_subset(
    paths: Sequence[str | os.PathLike[str]],
    *,
    output_path: str | os.PathLike[str],
    fraction: float | None = None,
    count: int | None = None,
    salt: str = "wqcodiff-hash-fixed-v1",
) -> dict[str, Any]:
    """Materialize an immutable subset chosen only by salted material-ID hash."""

    if not paths:
        raise ValueError("subset source paths are required")
    if (fraction is None) == (count is None):
        raise ValueError("specify exactly one of fraction or count")
    if fraction is not None and not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0,1]")
    if count is not None and count <= 0:
        raise ValueError("count must be positive")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    sources: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        sources.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not payload.get("selected"):
                    continue
                material_id = str(payload.get("material_id", ""))
                if not material_id or material_id in seen_ids:
                    raise ValueError(f"{path}:{line_number}: missing/duplicate material_id")
                seen_ids.add(material_id)
                rows.append(payload)
    if not rows:
        raise ValueError("subset source has no selected records")
    if count is not None and count > len(rows):
        raise ValueError(f"requested {count} records, source has only {len(rows)}")
    rows.sort(
        key=lambda payload: hashlib.sha256(
            f"{salt}:{payload['material_id']}".encode("utf-8")
        ).hexdigest()
    )
    requested = (
        int(count)
        if count is not None
        else max(1, int(round(len(rows) * float(fraction))))
    )
    selected = rows[:requested]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for payload in selected:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    result = {
        "schema": "wqcodiff_hash_fixed_subset_v1",
        "selection_key": "sha256(salt:material_id)",
        "salt": salt,
        "source_selected_records": len(rows),
        "requested_fraction": fraction,
        "requested_count": count,
        "selected_records": len(selected),
        "selected_material_id_hash": hashlib.sha256(
            "\n".join(str(payload["material_id"]) for payload in selected).encode("utf-8")
        ).hexdigest(),
        "output": str(destination.resolve()),
        "output_sha256": sha256_file(destination),
        "sources": sources,
    }
    summary_path = destination.with_suffix(destination.suffix + ".summary.json")
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    result["summary"] = str(summary_path.resolve())
    return result


def audit_split_leakage(paths_by_split: Mapping[str, Iterable[str | os.PathLike[str]]]) -> dict[str, Any]:
    hashes: dict[str, set[str]] = {}
    for split, paths in paths_by_split.items():
        values: set[str] = set()
        for path in paths:
            with Path(path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    payload = json.loads(line)
                    if not payload.get("selected"):
                        continue
                    primary = payload["decompositions"][tolerance_tag(PRIMARY_SYMPREC)]
                    values.add(str(primary["canonical_hash"]))
        hashes[split] = values
    overlaps: dict[str, list[str]] = {}
    names = sorted(hashes)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            intersection = sorted(hashes[first] & hashes[second])
            if intersection:
                overlaps[f"{first}:{second}"] = intersection
    return {
        "split_counts": {key: len(value) for key, value in sorted(hashes.items())},
        "overlap_count": sum(len(value) for value in overlaps.values()),
        "overlaps": overlaps,
    }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _jensen_shannon(first: Mapping[str, int], second: Mapping[str, int]) -> float | None:
    first_total = sum(first.values())
    second_total = sum(second.values())
    if not first_total or not second_total:
        return None
    support = sorted(set(first) | set(second))
    p = [first.get(key, 0) / first_total for key in support]
    q = [second.get(key, 0) / second_total for key in support]
    midpoint = [(left + right) / 2.0 for left, right in zip(p, q)]

    def divergence(values: Sequence[float]) -> float:
        return sum(
            value * math.log(value / middle)
            for value, middle in zip(values, midpoint)
            if value > 0.0 and middle > 0.0
        )

    return 0.5 * (divergence(p) + divergence(q))


def _primary_payload(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    decompositions = payload.get("decompositions")
    if not isinstance(decompositions, Mapping):
        return None
    primary = decompositions.get(tolerance_tag(PRIMARY_SYMPREC))
    return primary if isinstance(primary, Mapping) else None


def _primitive_atom_count(primary: Mapping[str, Any]) -> int | None:
    state = primary.get("state")
    if not isinstance(state, Mapping):
        return None
    orbits = state.get("orbits")
    if not isinstance(orbits, Sequence):
        return None
    try:
        return sum(
            int(orbit.get("primitive_multiplicity", orbit["multiplicity"]))
            for orbit in orbits
        )
    except (KeyError, TypeError, ValueError):
        return None


def audit_wq_dataset(
    paths_by_split: Mapping[str, Iterable[str | os.PathLike[str]]],
    *,
    expected_total: int = MP20_TOTAL_RECORDS,
    allow_nonpaper_counts: bool = False,
) -> dict[str, Any]:
    """Run all registered P1 gates over immutable preprocessing shards."""

    if not paths_by_split:
        raise ValueError("at least one split is required for the P1 audit")
    split_reports: dict[str, Any] = {}
    canonical_hashes: dict[str, set[str]] = {}
    file_records: list[dict[str, Any]] = []
    global_schema_errors: list[str] = []
    global_split_errors: list[str] = []

    for split, raw_paths in sorted(paths_by_split.items()):
        paths = tuple(Path(path).resolve() for path in raw_paths)
        if not paths:
            raise ValueError(f"split {split!r} has no files")
        total = selected = primary_count = roundtrip = consistent = ambiguous = 0
        atom_range_violations = selected_contract_violations = 0
        chart_contract_violations = 0
        chart_error_max = 0.0
        projector_contract_violations = 0
        projector_error_max = 0.0
        ids: Counter[str] = Counter()
        failure_reasons: Counter[str] = Counter()
        all_families: Counter[str] = Counter()
        selected_families: Counter[str] = Counter()
        hashes: set[str] = set()
        for path in paths:
            file_records.append(
                {
                    "split": split,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    total += 1
                    try:
                        payload = json.loads(line)
                    except Exception as exc:
                        global_schema_errors.append(f"{path}:{line_number}:json:{exc}")
                        continue
                    if payload.get("schema") != "mp20_wq_v1":
                        global_schema_errors.append(f"{path}:{line_number}:schema")
                    if str(payload.get("split")) != split:
                        global_split_errors.append(
                            f"{path}:{line_number}:{payload.get('split')}!={split}"
                        )
                    ids[str(payload.get("material_id", ""))] += 1
                    is_selected = bool(payload.get("selected"))
                    selected += int(is_selected)
                    ambiguous += int(bool(payload.get("ambiguous")))
                    source_elements = tuple(str(value) for value in payload.get("source_elements", ()))
                    family = (
                        str(payload.get("material_family"))
                        if source_elements
                        else "unknown"
                    )
                    all_families[family] += 1
                    if is_selected:
                        selected_families[family] += 1
                    primary = _primary_payload(payload)
                    if primary is None:
                        failure_reasons[str(payload.get("primary_failure_reason") or "missing_primary")] += 1
                        if is_selected:
                            selected_contract_violations += 1
                        continue
                    primary_count += 1
                    matched = bool(primary.get("roundtrip_match"))
                    multiplicity_ok = bool(primary.get("multiplicity_consistent"))
                    roundtrip += int(matched)
                    consistent += int(multiplicity_ok)
                    atom_count = _primitive_atom_count(primary)
                    if atom_count is None or not 1 <= atom_count <= 20:
                        atom_range_violations += 1
                    orbit_payloads = primary.get("orbits")
                    chart_valid = isinstance(orbit_payloads, Sequence) and bool(orbit_payloads)
                    if chart_valid:
                        for orbit_payload in orbit_payloads:
                            if not isinstance(orbit_payload, Mapping):
                                chart_valid = False
                                break
                            orbit_state = orbit_payload.get("orbit")
                            basis = orbit_payload.get("chart_basis")
                            primitive_jacobians = orbit_payload.get(
                                "primitive_chart_jacobians"
                            )
                            try:
                                dimension = int(orbit_state["chart_dimension"])
                                residual = float(
                                    orbit_payload["chart_fit_residual_angstrom"]
                                )
                                basis_array = np.asarray(basis, dtype=np.float64)
                                valid_shape = basis_array.shape == (3, dimension)
                                valid_rank = dimension == 0 or (
                                    np.linalg.matrix_rank(basis_array, tol=1.0e-12)
                                    == dimension
                                )
                                finite = bool(
                                    np.all(np.isfinite(basis_array))
                                    and math.isfinite(residual)
                                )
                                projector_error = regularized_projector_error(
                                    primitive_jacobians
                                )
                            except (KeyError, TypeError, ValueError):
                                chart_valid = False
                                break
                            if is_selected:
                                chart_error_max = max(chart_error_max, residual)
                                if math.isfinite(projector_error):
                                    projector_error_max = max(
                                        projector_error_max, projector_error
                                    )
                            if not (
                                dimension in {0, 1, 2, 3}
                                and valid_shape
                                and valid_rank
                                and finite
                                and 0.0 <= residual < 1.0e-6
                            ):
                                chart_valid = False
                                break
                            if not math.isfinite(projector_error) or projector_error >= 1.0e-6:
                                if is_selected:
                                    projector_contract_violations += 1
                    if is_selected and not chart_valid:
                        chart_contract_violations += 1
                    if is_selected and (not matched or not multiplicity_ok or atom_count is None):
                        selected_contract_violations += 1
                    if is_selected and not chart_valid:
                        selected_contract_violations += 1
                    if is_selected:
                        canonical_hash = primary.get("canonical_hash")
                        if not canonical_hash:
                            selected_contract_violations += 1
                        else:
                            hashes.add(str(canonical_hash))

        duplicate_material_ids = sorted(
            material_id for material_id, count in ids.items() if not material_id or count > 1
        )
        family_support = sorted(set(all_families) | set(selected_families))
        family_shift_pp = {
            family: 100.0
            * (
                _rate(selected_families.get(family, 0), selected)
                - _rate(all_families.get(family, 0), total)
            )
            for family in family_support
        }
        split_reports[split] = {
            "records": total,
            "selected": selected,
            "coverage": _rate(selected, total),
            "primary_decompositions": primary_count,
            "roundtrip_matches": roundtrip,
            "roundtrip_rate_given_primary": _rate(roundtrip, primary_count),
            "atom_count_consistent": consistent,
            "atom_count_consistency_given_primary": _rate(consistent, primary_count),
            "atom_range_violations": atom_range_violations,
            "chart_contract_violations": chart_contract_violations,
            "chart_error_max_angstrom": chart_error_max,
            "projector_contract_violations": projector_contract_violations,
            "projector_error_max": projector_error_max,
            "selected_contract_violations": selected_contract_violations,
            "ambiguous": ambiguous,
            "ambiguous_rate": _rate(ambiguous, total),
            "duplicate_material_ids": duplicate_material_ids,
            "failure_reasons": dict(failure_reasons.most_common()),
            "material_family_all": dict(sorted(all_families.items())),
            "material_family_selected": dict(sorted(selected_families.items())),
            "material_family_metadata_coverage": _rate(
                total - all_families.get("unknown", 0), total
            ),
            "material_family_shift_pp": family_shift_pp,
            "material_family_max_abs_shift_pp": max(
                (abs(value) for value in family_shift_pp.values()), default=0.0
            ),
            "material_family_selection_jsd": _jensen_shannon(
                all_families, selected_families
            ),
        }
        canonical_hashes[split] = hashes

    overlaps: dict[str, list[str]] = {}
    names = sorted(canonical_hashes)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            intersection = sorted(canonical_hashes[first] & canonical_hashes[second])
            if intersection:
                overlaps[f"{first}:{second}"] = intersection
    overlap_count = sum(len(value) for value in overlaps.values())
    observed_total = sum(report["records"] for report in split_reports.values())
    checks = {
        "record_count_exact": allow_nonpaper_counts or observed_total == expected_total,
        "coverage_each_split_ge_0p95": all(
            report["coverage"] >= P1_COVERAGE_MIN for report in split_reports.values()
        ),
        "roundtrip_each_split_ge_0p99": all(
            report["roundtrip_rate_given_primary"] >= P1_ROUNDTRIP_MIN
            for report in split_reports.values()
        ),
        "atom_count_consistency_each_split_eq_1": all(
            report["atom_count_consistency_given_primary"]
            >= P1_ATOM_COUNT_CONSISTENCY
            and report["atom_range_violations"] == 0
            for report in split_reports.values()
        ),
        "pyxtal_chart_contract_each_split_lt_1e-6": all(
            report["chart_contract_violations"] == 0
            and report["chart_error_max_angstrom"] < 1.0e-6
            for report in split_reports.values()
        ),
        "tangent_projector_each_split_lt_1e-6": all(
            report["projector_contract_violations"] == 0
            and report["projector_error_max"] < 1.0e-6
            for report in split_reports.values()
        ),
        "selected_record_contract": all(
            report["selected_contract_violations"] == 0
            for report in split_reports.values()
        ),
        "material_ids_unique_within_split": all(
            not report["duplicate_material_ids"] for report in split_reports.values()
        ),
        "canonicalized_cross_split_leakage_eq_0": overlap_count == 0,
        "schema_valid": not global_schema_errors,
        "declared_split_matches_file_group": not global_split_errors,
    }
    passed = all(checks.values())
    return {
        "schema": "wqcodiff_p1_dataset_audit_v1",
        "ok": passed,
        "gate_passed": passed,
        "paper_count_gate_enforced": not allow_nonpaper_counts,
        "expected_total_records": expected_total,
        "observed_total_records": observed_total,
        "thresholds": {
            "coverage_min": P1_COVERAGE_MIN,
            "roundtrip_structure_match_min": P1_ROUNDTRIP_MIN,
            "atom_count_consistency": P1_ATOM_COUNT_CONSISTENCY,
            "pyxtal_chart_error_max_angstrom": 1.0e-6,
            "tangent_projector_error_max": 1.0e-6,
            "canonicalized_split_leakage_max": 0,
        },
        "checks": checks,
        "splits": split_reports,
        "canonical_overlap_count": overlap_count,
        "canonical_overlaps": overlaps,
        "schema_errors": global_schema_errors,
        "split_errors": global_split_errors,
        "files": file_records,
    }
