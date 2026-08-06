"""Finite, non-repairing Wyckoff text grammar for the CrysLLMGen fork.

The grammar deliberately uses a compact finite code for continuous values.  It
therefore supports exact constrained decoding without adding tokenizer tokens,
while the inherited continuous refiner still receives ordinary floating-point
lattice and free-coordinate charts after parsing.
"""

from __future__ import annotations

import dataclasses
import math
import random
import re
from typing import Any, Iterable, Literal, Sequence

from ..bridge import ChartCatalog, ChartSpec
from ..charts import LatticeChartCodec
from ..state import OrbitState, StratifiedState
from ..vocabulary import atomic_number_to_input_id, target_to_atomic_number


HEX_BINS = 256
HEX_CODE = re.compile(r"^[0-9A-F]{2}$")
ORBIT_FIELD = re.compile(
    r"^O=(0|[1-9][0-9]*),W=(0|[1-9][0-9]*),"
    r"E=([1-9][0-9]*),U=(-|[0-9A-F]{2}(?:,[0-9A-F]{2})*)$"
)

# Signed mu-law preserves useful resolution near ordinary MP20 cells while
# keeping every decoded byte numerically safe.  Entries are (max_abs, mu).
_LOG_LENGTH = (20.0, 63.0)
_OFF_DIAGONAL = (256.0, 127.0)
_ANGLE_LOGIT = (30.0, 63.0)
LATTICE_COMPONENT_QUANTIZERS: dict[str, tuple[tuple[float, float], ...]] = {
    "triclinic": (_LOG_LENGTH, _LOG_LENGTH, _LOG_LENGTH, _OFF_DIAGONAL, _OFF_DIAGONAL, _OFF_DIAGONAL),
    "monoclinic": (_LOG_LENGTH, _LOG_LENGTH, _LOG_LENGTH, _ANGLE_LOGIT),
    "orthorhombic": (_LOG_LENGTH, _LOG_LENGTH, _LOG_LENGTH),
    "tetragonal": (_LOG_LENGTH, _LOG_LENGTH),
    "trigonal": (_LOG_LENGTH, _LOG_LENGTH),
    "hexagonal": (_LOG_LENGTH, _LOG_LENGTH),
    "cubic": (_LOG_LENGTH,),
}


class GrammarViolation(ValueError):
    """A terminal parser/grammar failure; callers must not repair or retry it."""


def crystal_system_for_space_group(space_group: int) -> str:
    value = int(space_group)
    if not 1 <= value <= 230:
        raise GrammarViolation("space group must be in [1,230]")
    if value <= 2:
        return "triclinic"
    if value <= 15:
        return "monoclinic"
    if value <= 74:
        return "orthorhombic"
    if value <= 142:
        return "tetragonal"
    if value <= 167:
        return "trigonal"
    if value <= 194:
        return "hexagonal"
    return "cubic"


def _encode_hex(value: int) -> str:
    if not 0 <= int(value) < HEX_BINS:
        raise GrammarViolation("continuous code must be one byte")
    return f"{int(value):02X}"


def _decode_hex(value: str) -> int:
    if HEX_CODE.fullmatch(value) is None:
        raise GrammarViolation(f"invalid uppercase hexadecimal byte: {value!r}")
    return int(value, 16)


def _encode_companded(value: float, maximum: float, mu: float) -> int:
    numeric = float(value)
    if (
        not math.isfinite(numeric)
        or not math.isfinite(maximum)
        or not math.isfinite(mu)
        or maximum <= 0
        or mu <= 0
    ):
        raise GrammarViolation("lattice chart and compander parameters must be finite")
    if abs(numeric) > maximum:
        raise GrammarViolation("lattice chart exceeds the frozen byte-code support")
    normalized = numeric / maximum
    compressed = math.copysign(
        math.log1p(mu * abs(normalized)) / math.log1p(mu), normalized
    )
    return min(255, max(0, int(math.floor((compressed + 1.0) * 127.5 + 0.5))))


def _decode_companded(code: int, maximum: float, mu: float) -> float:
    compressed = int(code) / 127.5 - 1.0
    normalized = math.copysign(
        math.expm1(abs(compressed) * math.log1p(mu)) / mu, compressed
    )
    return float(maximum * normalized)


def encode_lattice_chart(values: Sequence[float], crystal_system: str) -> tuple[str, ...]:
    system = str(crystal_system).lower()
    try:
        quantizers = LATTICE_COMPONENT_QUANTIZERS[system]
    except KeyError as exc:
        raise GrammarViolation(f"unsupported crystal system: {crystal_system}") from exc
    if len(values) != len(quantizers) or len(values) != LatticeChartCodec.dimension(system):
        raise GrammarViolation("lattice chart dimension does not match crystal system")
    return tuple(
        _encode_hex(_encode_companded(value, maximum, mu))
        for value, (maximum, mu) in zip(values, quantizers)
    )


def decode_lattice_chart(codes: Sequence[str], crystal_system: str) -> tuple[float, ...]:
    system = str(crystal_system).lower()
    try:
        quantizers = LATTICE_COMPONENT_QUANTIZERS[system]
    except KeyError as exc:
        raise GrammarViolation(f"unsupported crystal system: {crystal_system}") from exc
    if len(codes) != len(quantizers):
        raise GrammarViolation("lattice code dimension does not match crystal system")
    values = tuple(
        _decode_companded(_decode_hex(code), maximum, mu)
        for code, (maximum, mu) in zip(codes, quantizers)
    )
    # Decode once here so positive-definiteness and registered lattice
    # conventions are part of the parser boundary, not a later repair stage.
    try:
        LatticeChartCodec.decode_matrix(values, system)
    except (ValueError, OverflowError) as exc:
        raise GrammarViolation(f"invalid decoded lattice chart: {exc}") from exc
    return values


def encode_free_coordinate(values: Sequence[float]) -> tuple[str, ...]:
    if len(values) not in {0, 1, 2, 3}:
        raise GrammarViolation("orbit free-coordinate dimension must be in [0,3]")
    result = []
    for value in values:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise GrammarViolation("free coordinates must be finite")
        code = int(math.floor((numeric % 1.0) * HEX_BINS + 0.5)) % HEX_BINS
        result.append(_encode_hex(code))
    return tuple(result)


def decode_free_coordinate(codes: Sequence[str]) -> tuple[float, ...]:
    if len(codes) not in {0, 1, 2, 3}:
        raise GrammarViolation("orbit free-coordinate dimension must be in [0,3]")
    return tuple(_decode_hex(code) / HEX_BINS for code in codes)


def _legal_spec(catalog: ChartCatalog, space_group: int, wyckoff_type: int) -> ChartSpec:
    try:
        legal_types = {int(value) for value in catalog.types(space_group)}
    except Exception as exc:
        raise GrammarViolation(f"Wyckoff catalogue failure: {type(exc).__name__}:{exc}") from exc
    if int(wyckoff_type) not in legal_types:
        raise GrammarViolation("Wyckoff type is not legal for the emitted space group")
    try:
        return catalog.get(space_group, int(wyckoff_type))
    except Exception as exc:
        raise GrammarViolation(f"Wyckoff lookup failure: {type(exc).__name__}:{exc}") from exc


def serialize_wq_proposal(state: StratifiedState, catalog: ChartCatalog) -> str:
    """Serialize in the current orbit presentation order, never canonicalizing it."""

    system = crystal_system_for_space_group(state.space_group)
    if state.lattice_system.lower() != system:
        raise GrammarViolation("state lattice system disagrees with its space group")
    lattice = ",".join(encode_lattice_chart(state.lattice_chart, system))
    fields = [f"SG={state.space_group}", f"Q={lattice}"]
    primitive_atoms = 0
    for index, orbit in enumerate(state.orbits):
        spec = _legal_spec(catalog, state.space_group, orbit.wyckoff_type)
        if (
            orbit.multiplicity != spec.multiplicity
            or orbit.primitive_multiplicity != spec.primitive_multiplicity
            or orbit.chart_dimension != spec.dimension
        ):
            raise GrammarViolation("orbit metadata disagrees with the frozen catalogue")
        try:
            species_code = atomic_number_to_input_id(orbit.species)
        except ValueError as exc:
            raise GrammarViolation(str(exc)) from exc
        primitive_atoms += int(spec.primitive_multiplicity)
        if primitive_atoms > 20:
            raise GrammarViolation("proposal exceeds the MP20 primitive-atom limit")
        free = encode_free_coordinate(orbit.free_coordinate)
        fields.append(
            f"O={index},W={orbit.wyckoff_type},E={species_code},"
            f"U={','.join(free) if free else '-'}"
        )
    if not state.orbits or not 1 <= primitive_atoms <= 20:
        raise GrammarViolation("a proposal must contain 1-20 primitive atoms")
    fields.append("STOP")
    return ";".join(fields)


def parse_wq_proposal(
    text: str,
    catalog: ChartCatalog,
    *,
    attempt_id: str = "",
    timestep: float = 0.0,
) -> StratifiedState:
    """Parse the exact byte grammar once; whitespace or prose is invalid."""

    if not text or text != text.strip() or any(character.isspace() for character in text):
        raise GrammarViolation("proposal must be one whitespace-free canonical byte string")
    fields = text.split(";")
    if len(fields) < 4 or fields[-1] != "STOP" or fields.count("STOP") != 1:
        raise GrammarViolation("proposal must end once with STOP after at least one orbit")
    if re.fullmatch(r"SG=(0|[1-9][0-9]*)", fields[0]) is None:
        raise GrammarViolation("first field must be canonical SG=<1..230>")
    space_group = int(fields[0][3:])
    system = crystal_system_for_space_group(space_group)
    if not fields[1].startswith("Q="):
        raise GrammarViolation("second field must be the lattice chart")
    lattice_codes = fields[1][2:].split(",") if fields[1][2:] else []
    lattice_chart = decode_lattice_chart(lattice_codes, system)
    orbits: list[OrbitState] = []
    primitive_atoms = 0
    for expected_index, field in enumerate(fields[2:-1]):
        match = ORBIT_FIELD.fullmatch(field)
        if match is None:
            raise GrammarViolation(f"invalid orbit field at index {expected_index}")
        serialized_index = int(match.group(1))
        if serialized_index != expected_index:
            raise GrammarViolation("orbit indices must be unique, contiguous, and presentation ordered")
        wyckoff_type = int(match.group(2))
        species_code = int(match.group(3))
        if not 1 <= species_code <= 89:
            raise GrammarViolation("species code must be in [1,89]")
        species = target_to_atomic_number(species_code - 1)
        spec = _legal_spec(catalog, space_group, wyckoff_type)
        raw_free = match.group(4)
        free_codes = () if raw_free == "-" else tuple(raw_free.split(","))
        if len(free_codes) != spec.dimension:
            raise GrammarViolation("free-coordinate code count disagrees with Wyckoff DoF")
        primitive_atoms += int(spec.primitive_multiplicity)
        if primitive_atoms > 20:
            raise GrammarViolation("proposal exceeds the MP20 primitive-atom limit")
        orbits.append(
            OrbitState(
                orbit_id=f"o{expected_index}",
                wyckoff_type=wyckoff_type,
                species=species,
                multiplicity=spec.multiplicity,
                primitive_multiplicity=spec.primitive_multiplicity,
                chart_dimension=spec.dimension,
                free_coordinate=decode_free_coordinate(free_codes),
            )
        )
    if not orbits:
        raise GrammarViolation("proposal contains no semantic orbit")
    return StratifiedState(
        space_group=space_group,
        lattice_system=system,
        lattice_chart=lattice_chart,
        orbits=tuple(orbits),
        attempt_id=attempt_id,
        timestep=float(timestep),
        space_group_committed=True,
    )


EditKind = Literal["noop", "birth", "death", "type_change", "species_change"]


@dataclasses.dataclass(frozen=True, slots=True)
class TopologyEdit:
    kind: EditKind
    orbit_index: int | None = None
    wyckoff_type: int | None = None
    species: int | None = None


def serialize_topology_edit(edit: TopologyEdit) -> str:
    if edit.kind == "noop":
        return "NOOP"
    if edit.kind == "birth":
        if edit.species is None:
            raise GrammarViolation("birth edit has no species")
        return f"BIRTH;W={edit.wyckoff_type};E={atomic_number_to_input_id(edit.species)}"
    if edit.kind == "death":
        return f"DEATH;O={edit.orbit_index}"
    if edit.kind == "type_change":
        return f"TYPE;O={edit.orbit_index};W={edit.wyckoff_type}"
    if edit.kind == "species_change":
        if edit.species is None:
            raise GrammarViolation("species edit has no species")
        return f"SPECIES;O={edit.orbit_index};E={atomic_number_to_input_id(edit.species)}"
    raise GrammarViolation(f"unsupported topology edit: {edit.kind}")


def parse_topology_edit(
    text: str, state: StratifiedState, catalog: ChartCatalog
) -> TopologyEdit:
    """Parse and support-mask one direct edit against the current stratum."""

    if text != text.strip() or any(character.isspace() for character in text):
        raise GrammarViolation("edit command must use the exact whitespace-free grammar")
    if text == "NOOP":
        return TopologyEdit("noop")
    patterns = (
        ("birth", re.fullmatch(r"BIRTH;W=(0|[1-9][0-9]*);E=([1-9][0-9]*)", text)),
        ("death", re.fullmatch(r"DEATH;O=(0|[1-9][0-9]*)", text)),
        ("type_change", re.fullmatch(r"TYPE;O=(0|[1-9][0-9]*);W=(0|[1-9][0-9]*)", text)),
        ("species_change", re.fullmatch(r"SPECIES;O=(0|[1-9][0-9]*);E=([1-9][0-9]*)", text)),
    )
    selected = next(((kind, match) for kind, match in patterns if match is not None), None)
    if selected is None:
        raise GrammarViolation("edit command does not match the frozen direct-event grammar")
    kind, match = selected
    assert match is not None
    if kind == "birth":
        wyckoff_type, species_code = map(int, match.groups())
        if not 1 <= species_code <= 89:
            raise GrammarViolation("birth species code is outside [1,89]")
        species = target_to_atomic_number(species_code - 1)
        spec = _legal_spec(catalog, state.space_group, wyckoff_type)
        if state.atom_count + int(spec.primitive_multiplicity) > 20:
            raise GrammarViolation("birth would exceed the MP20 atom limit")
        return TopologyEdit("birth", wyckoff_type=wyckoff_type, species=species)
    orbit_index = int(match.group(1))
    if not 0 <= orbit_index < len(state.orbits):
        raise GrammarViolation("edit orbit pointer is outside the current presentation")
    source = state.orbits[orbit_index]
    if kind == "death":
        if len(state.orbits) == 1:
            raise GrammarViolation("death would produce an empty structure")
        return TopologyEdit("death", orbit_index=orbit_index)
    target = int(match.group(2))
    if kind == "type_change":
        if target == source.wyckoff_type:
            raise GrammarViolation("type change must change the Wyckoff type")
        spec = _legal_spec(catalog, state.space_group, target)
        new_count = (
            state.atom_count
            - int(source.primitive_multiplicity)
            + int(spec.primitive_multiplicity)
        )
        if not 1 <= new_count <= 20:
            raise GrammarViolation("type change would leave the MP20 support")
        return TopologyEdit("type_change", orbit_index=orbit_index, wyckoff_type=target)
    if not 1 <= target <= 89:
        raise GrammarViolation("species change code must be in [1,89]")
    target_species = target_to_atomic_number(target - 1)
    if target_species == source.species:
        raise GrammarViolation("species change must select a different MP20 element")
    return TopologyEdit(
        "species_change", orbit_index=orbit_index, species=target_species
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ProposalGrammarState:
    """Semantic FSM used by the formal support audit and constrained decoder."""

    phase: Literal["space_group", "lattice", "orbit_or_stop", "stopped"] = "space_group"
    space_group: int | None = None
    atom_count: int = 0
    orbit_count: int = 0

    def accept_space_group(self, value: int) -> "ProposalGrammarState":
        if self.phase != "space_group":
            raise GrammarViolation("space group is legal only at the initial state")
        crystal_system_for_space_group(value)
        return ProposalGrammarState("lattice", int(value), 0, 0)

    def accept_lattice(self, codes: Sequence[str]) -> "ProposalGrammarState":
        if self.phase != "lattice" or self.space_group is None:
            raise GrammarViolation("lattice is legal only after the space group")
        decode_lattice_chart(codes, crystal_system_for_space_group(self.space_group))
        return ProposalGrammarState("orbit_or_stop", self.space_group, 0, 0)

    def legal_wyckoff_types(self, catalog: ChartCatalog) -> tuple[int, ...]:
        if self.phase != "orbit_or_stop" or self.space_group is None:
            return ()
        remaining = 20 - self.atom_count
        result = []
        for value in catalog.types(self.space_group):
            spec = catalog.get(self.space_group, int(value))
            if int(spec.primitive_multiplicity) <= remaining:
                result.append(int(value))
        return tuple(sorted(result))

    def accept_orbit(
        self,
        *,
        wyckoff_type: int,
        species: int,
        free_codes: Sequence[str],
        catalog: ChartCatalog,
    ) -> "ProposalGrammarState":
        if self.phase != "orbit_or_stop" or self.space_group is None:
            raise GrammarViolation("orbit is legal only after the lattice")
        if not 1 <= int(species) <= 89:
            raise GrammarViolation("species must be in [1,89]")
        legal = self.legal_wyckoff_types(catalog)
        if int(wyckoff_type) not in legal:
            raise GrammarViolation("orbit is outside the current atom-count support")
        spec = _legal_spec(catalog, self.space_group, int(wyckoff_type))
        if len(free_codes) != spec.dimension:
            raise GrammarViolation("free-coordinate code count disagrees with Wyckoff DoF")
        decode_free_coordinate(free_codes)
        return ProposalGrammarState(
            "orbit_or_stop",
            self.space_group,
            self.atom_count + int(spec.primitive_multiplicity),
            self.orbit_count + 1,
        )

    def accept_stop(self) -> "ProposalGrammarState":
        if self.phase != "orbit_or_stop" or self.orbit_count < 1:
            raise GrammarViolation("STOP requires at least one legal orbit")
        return ProposalGrammarState(
            "stopped", self.space_group, self.atom_count, self.orbit_count
        )


def audit_synthetic_grammar_transitions(
    catalog: ChartCatalog,
    *,
    transitions: int,
    seed: int,
) -> dict[str, Any]:
    """Sample only from normalized legal support and audit every resulting state."""

    if transitions <= 0:
        raise ValueError("transitions must be positive")
    rng = random.Random(seed)
    state = ProposalGrammarState()
    illegal_generated = 0
    completed_proposals = 0
    phase_counts = {name: 0 for name in ("space_group", "lattice", "orbit", "stop")}
    for _ in range(transitions):
        try:
            if state.phase == "space_group":
                state = state.accept_space_group(rng.randint(1, 230))
                phase_counts["space_group"] += 1
            elif state.phase == "lattice":
                assert state.space_group is not None
                dimension = LatticeChartCodec.dimension(
                    crystal_system_for_space_group(state.space_group)
                )
                state = state.accept_lattice(
                    tuple(_encode_hex(rng.randrange(HEX_BINS)) for _ in range(dimension))
                )
                phase_counts["lattice"] += 1
            elif state.phase == "orbit_or_stop":
                legal = state.legal_wyckoff_types(catalog)
                should_stop = state.orbit_count > 0 and (
                    not legal or rng.random() < 0.30 or state.orbit_count >= 20
                )
                if should_stop:
                    state = state.accept_stop()
                    phase_counts["stop"] += 1
                else:
                    if not legal:
                        raise GrammarViolation("no orbit support before the first orbit")
                    wyckoff = legal[rng.randrange(len(legal))]
                    assert state.space_group is not None
                    spec = catalog.get(state.space_group, wyckoff)
                    state = state.accept_orbit(
                        wyckoff_type=wyckoff,
                        species=rng.randint(1, 89),
                        free_codes=tuple(
                            _encode_hex(rng.randrange(HEX_BINS))
                            for _ in range(spec.dimension)
                        ),
                        catalog=catalog,
                    )
                    phase_counts["orbit"] += 1
            else:
                completed_proposals += 1
                state = ProposalGrammarState().accept_space_group(rng.randint(1, 230))
                phase_counts["space_group"] += 1
        except (GrammarViolation, KeyError, ValueError):
            illegal_generated += 1
            state = ProposalGrammarState()
    return {
        "schema": "crysllmgen_wq_grammar_transition_audit_v1",
        "transitions": transitions,
        "seed": seed,
        "illegal_generated": illegal_generated,
        "completed_proposals": completed_proposals,
        "phase_counts": phase_counts,
        "passed": illegal_generated == 0 and sum(phase_counts.values()) == transitions,
    }
