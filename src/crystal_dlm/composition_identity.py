"""Canonical composition identities shared by C³FD and CTV-DLM audits."""

from __future__ import annotations

from functools import reduce
from math import gcd
from typing import Any, Mapping, Sequence

from crystal_dlm.fixed_slot import SYMBOL_TO_Z, Z_TO_SYMBOL


def canonical_symbol_counts(
    elements: Sequence[str], counts: Sequence[int]
) -> tuple[tuple[str, int], ...]:
    if len(elements) != len(counts) or not elements:
        raise ValueError("composition requires aligned non-empty elements/counts")
    merged: dict[int, int] = {}
    for raw_symbol, raw_count in zip(elements, counts):
        symbol = str(raw_symbol)
        if symbol not in SYMBOL_TO_Z:
            raise ValueError(f"unsupported element {symbol!r}")
        count = int(raw_count)
        if count <= 0:
            raise ValueError(f"nonpositive count for {symbol}")
        atomic_number = int(SYMBOL_TO_Z[symbol])
        merged[atomic_number] = merged.get(atomic_number, 0) + count
    return tuple((Z_TO_SYMBOL[z], int(merged[z])) for z in sorted(merged))


def reduced_composition_identity(
    elements: Sequence[str], counts: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    canonical = canonical_symbol_counts(elements, counts)
    divisor = reduce(gcd, (int(count) for _symbol, count in canonical))
    divisor = max(1, int(divisor))
    return tuple(
        (int(SYMBOL_TO_Z[symbol]), int(count) // divisor)
        for symbol, count in canonical
    )


def identity_from_plan_state(plan: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    return reduced_composition_identity(
        [str(value) for value in (plan.get("elements") or ())],
        [int(value) for value in (plan.get("counts") or ())],
    )


def identity_text(identity: Sequence[tuple[int, int]]) -> str:
    if not identity:
        raise ValueError("empty reduced composition identity")
    return "|".join(f"{int(z)}:{int(count)}" for z, count in identity)


def formula_from_symbol_counts(values: Sequence[tuple[str, int]]) -> str:
    if not values:
        raise ValueError("cannot render an empty formula")
    return "".join(
        str(symbol) if int(count) == 1 else f"{symbol}{int(count)}"
        for symbol, count in values
    )


__all__ = [
    "canonical_symbol_counts",
    "formula_from_symbol_counts",
    "identity_from_plan_state",
    "identity_text",
    "reduced_composition_identity",
]
