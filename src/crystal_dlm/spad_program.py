"""Planner-programmed position schedules for exact-length crystal DLMs.

SPAD keeps the physical body in the canonical dynamic ``7 + 4N`` layout.  A
separate species program controls *when* non-contiguous site coordinates are
revealed.  This avoids sharing token IDs between the Planner Llama and the DLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from crystal_dlm.composition_identity import canonical_symbol_counts
from crystal_dlm.dynamic_crystal import dynamic_answer_token_count
from crystal_dlm.fixed_slot import Z_TO_SYMBOL


LATTICE_POSITIONS = tuple(range(1, 7))


def element_position(slot_index: int) -> int:
    return 7 + 4 * int(slot_index)


def coordinate_positions(slot_index: int) -> tuple[int, int, int]:
    base = element_position(slot_index)
    return base + 1, base + 2, base + 3


@dataclass(frozen=True)
class SpeciesProgramEntry:
    symbol: str
    count: int
    slot_indices: tuple[int, ...]

    @property
    def anchor_slot(self) -> int:
        return int(self.slot_indices[0])

    @property
    def remaining_slots(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.slot_indices[1:])


@dataclass(frozen=True)
class SpeciesProgram:
    """A permutation of the unique elements in one exact-composition Plan."""

    num_atoms: int
    entries: tuple[SpeciesProgramEntry, ...]
    order_source: str

    @property
    def element_order(self) -> tuple[str, ...]:
        return tuple(entry.symbol for entry in self.entries)

    @property
    def anchor_slots(self) -> tuple[int, ...]:
        return tuple(entry.anchor_slot for entry in self.entries)


@dataclass(frozen=True)
class AnchorRevision:
    """One suffix-visible (or diagnostic suffix-hidden) anchor transaction."""

    slot_index: int
    positions: tuple[int, int, int]
    previous_token_ids: tuple[int, int, int]
    masked_token_ids: tuple[int, ...]
    visible_positions: tuple[bool, ...]
    suffix_visible: bool

    def provisional_token_ids(self) -> tuple[int, ...]:
        """Return a geometry view that uses old XYZ until replacements commit."""

        values = list(self.masked_token_ids)
        for position, old in zip(
            self.positions, self.previous_token_ids, strict=True
        ):
            values[position] = int(old)
        return tuple(values)


def _expanded_plan_slots(
    plan: Mapping[str, Any],
) -> tuple[int, dict[str, tuple[int, ...]]]:
    num_atoms = int(plan.get("N") or 0)
    composition = canonical_symbol_counts(
        [str(value) for value in (plan.get("elements") or ())],
        [int(value) for value in (plan.get("counts") or ())],
    )
    if not 1 <= num_atoms <= 20:
        raise ValueError("SPAD Plan N must be in 1..20")
    if sum(count for _symbol, count in composition) != num_atoms:
        raise ValueError("SPAD Plan violates exact N/count conservation")
    slots: dict[str, tuple[int, ...]] = {}
    cursor = 0
    for symbol, count in composition:
        slots[symbol] = tuple(range(cursor, cursor + int(count)))
        cursor += int(count)
    return num_atoms, slots


def program_from_element_order(
    plan: Mapping[str, Any],
    element_order: Sequence[str],
    *,
    order_source: str,
) -> SpeciesProgram:
    """Compile an externally predicted element permutation into native slots."""

    num_atoms, slots = _expanded_plan_slots(plan)
    order = tuple(str(value) for value in element_order)
    if len(order) != len(set(order)):
        raise ValueError("species program contains duplicate elements")
    if set(order) != set(slots) or len(order) != len(slots):
        raise ValueError("species program must permute every unique Plan element")
    source = str(order_source).strip()
    if not source:
        raise ValueError("species program requires an explicit order source")
    entries = tuple(
        SpeciesProgramEntry(
            symbol=symbol,
            count=len(slots[symbol]),
            slot_indices=slots[symbol],
        )
        for symbol in order
    )
    return SpeciesProgram(num_atoms=num_atoms, entries=entries, order_source=source)


def program_from_planner_trace(
    plan: Mapping[str, Any],
    semantic_trace: Sequence[Mapping[str, Any]],
) -> SpeciesProgram:
    """Compile first-occurrence element order from a typed Planner trace.

    Oxidation-state variants of one element are merged.  This function checks
    exact agreement with the final Plan but does not claim that the trace order
    was unconstrained or learned; callers must preserve that distinction in
    ``order_source`` when constructing scientific cells.
    """

    order: list[str] = []
    counts: dict[str, int] = {}
    eos_seen = False
    for raw in semantic_trace:
        action = str(raw.get("action") or "")
        if action == "proposal":
            if int(raw.get("N") or plan.get("N") or 0) != int(plan.get("N") or 0):
                raise ValueError("Planner trace proposal N disagrees with Plan")
            continue
        if action == "EOS":
            eos_seen = True
            continue
        if action != "species":
            raise ValueError(f"unsupported Planner trace action {action!r}")
        if eos_seen:
            raise ValueError("Planner trace contains species after EOS")
        atomic_number = int(raw.get("atomic_number") or 0)
        if atomic_number not in Z_TO_SYMBOL:
            raise ValueError(f"unsupported Planner atomic number {atomic_number}")
        symbol = str(Z_TO_SYMBOL[atomic_number])
        count = int(raw.get("count") or 0)
        if count <= 0:
            raise ValueError("Planner species count must be positive")
        if symbol not in counts:
            order.append(symbol)
            counts[symbol] = 0
        counts[symbol] += count
    if not eos_seen:
        raise ValueError("Planner trace lacks EOS")
    expected = dict(
        canonical_symbol_counts(
            [str(value) for value in (plan.get("elements") or ())],
            [int(value) for value in (plan.get("counts") or ())],
        )
    )
    if counts != expected:
        raise ValueError("Planner trace composition disagrees with Plan")
    return program_from_element_order(
        plan,
        order,
        order_source="typed_planner_trace",
    )


def prefilled_positions(num_atoms: int) -> tuple[int, ...]:
    """N and all species slots are fixed by the Planner composition."""

    if not 1 <= int(num_atoms) <= 20:
        raise ValueError("num_atoms must be in 1..20")
    return (0, *(element_position(slot) for slot in range(int(num_atoms))))


def spad_predictor_position_groups(
    program: SpeciesProgram,
) -> tuple[tuple[int, ...], ...]:
    """Return exact coverage for prefill, lattice, anchors, then future sites.

    Lattice scalars and XYZ components are single-position transactions.  This
    ensures that the gamma feasibility check sees alpha/beta and that periodic
    coordinate checks see X/Y before a candidate Z is committed.
    """

    groups: list[tuple[int, ...]] = [prefilled_positions(program.num_atoms)]
    groups.extend((position,) for position in LATTICE_POSITIONS)
    for entry in program.entries:
        groups.extend((position,) for position in coordinate_positions(entry.anchor_slot))
    for entry in program.entries:
        for slot in entry.remaining_slots:
            groups.extend((position,) for position in coordinate_positions(slot))
    _validate_exact_coverage(groups, program.num_atoms)
    return tuple(groups)


def canonical_predictor_position_groups(
    num_atoms: int,
) -> tuple[tuple[int, ...], ...]:
    """Adjacent control: canonical lattice then complete sites in storage order."""

    groups: list[tuple[int, ...]] = [prefilled_positions(num_atoms)]
    groups.extend((position,) for position in LATTICE_POSITIONS)
    for slot in range(int(num_atoms)):
        groups.extend((position,) for position in coordinate_positions(slot))
    _validate_exact_coverage(groups, int(num_atoms))
    return tuple(groups)


def anchor_revision_slots(program: SpeciesProgram) -> tuple[int, ...]:
    """Early anchors are revised once, in reverse Planner-program order."""

    return tuple(reversed(program.anchor_slots))


def spad_site_slots(program: SpeciesProgram) -> tuple[int, ...]:
    """Return sites in the order in which SPAD first commits their geometry."""

    anchors = [entry.anchor_slot for entry in program.entries]
    remaining = [
        slot
        for entry in program.entries
        for slot in entry.remaining_slots
    ]
    slots = tuple(int(value) for value in (*anchors, *remaining))
    if len(slots) != int(program.num_atoms) or set(slots) != set(
        range(int(program.num_atoms))
    ):
        raise ValueError("SPAD site order is not an exact slot permutation")
    return slots


def response_revision_slots(program: SpeciesProgram) -> tuple[int, ...]:
    """Visit every site once in reverse SPAD order for response-aligned repair."""

    return tuple(reversed(spad_site_slots(program)))


def begin_anchor_revision(
    token_ids: Sequence[int],
    *,
    slot_index: int,
    mask_token_id: int,
    suffix_visible: bool,
) -> AnchorRevision:
    """Mask one complete anchor while preserving its previous geometry view."""

    values = tuple(int(value) for value in token_ids)
    positions = coordinate_positions(slot_index)
    if positions[-1] >= len(values):
        raise ValueError("anchor slot is outside the exact-length canvas")
    if any(values[position] == int(mask_token_id) for position in positions):
        raise ValueError("anchor revision requires a complete committed XYZ")
    masked = list(values)
    previous = tuple(values[position] for position in positions)
    for position in positions:
        masked[position] = int(mask_token_id)
    visible = [True] * len(values)
    if not suffix_visible:
        for position in range(positions[-1] + 1, len(values)):
            visible[position] = False
    for position in positions:
        visible[position] = True
    return AnchorRevision(
        slot_index=int(slot_index),
        positions=positions,
        previous_token_ids=previous,
        masked_token_ids=tuple(masked),
        visible_positions=tuple(visible),
        suffix_visible=bool(suffix_visible),
    )


def _validate_exact_coverage(
    groups: Sequence[Sequence[int]], num_atoms: int
) -> None:
    expected = set(range(dynamic_answer_token_count(int(num_atoms))))
    flattened = [int(position) for group in groups for position in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("SPAD predictor schedule contains duplicate positions")
    if set(flattened) != expected:
        missing = sorted(expected - set(flattened))
        extra = sorted(set(flattened) - expected)
        raise ValueError(
            f"SPAD predictor schedule coverage changed: missing={missing}, extra={extra}"
        )


__all__ = [
    "AnchorRevision",
    "LATTICE_POSITIONS",
    "SpeciesProgram",
    "SpeciesProgramEntry",
    "anchor_revision_slots",
    "begin_anchor_revision",
    "canonical_predictor_position_groups",
    "coordinate_positions",
    "element_position",
    "prefilled_positions",
    "program_from_element_order",
    "program_from_planner_trace",
    "response_revision_slots",
    "spad_site_slots",
    "spad_predictor_position_groups",
]
