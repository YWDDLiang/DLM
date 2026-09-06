"""Canonical dynamic-body site order aligned with inference element prefill."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any, Mapping

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer, parse_dynamic_answer
from crystal_dlm.fixed_slot import tokenize_answer_text


def expanded_plan_species(plan: Mapping[str, Any]) -> list[str]:
    elements = [str(value) for value in plan.get("elements") or []]
    counts = [int(value) for value in plan.get("counts") or []]
    if not elements or len(elements) != len(counts):
        raise ValueError("plan lacks aligned elements/counts")
    expanded: list[str] = []
    for element, count in zip(elements, counts, strict=True):
        if count <= 0:
            raise ValueError("plan counts must be positive")
        expanded.extend([element] * count)
    if len(expanded) != int(plan["N"]):
        raise ValueError("expanded Plan species does not match N")
    return expanded


def canonicalize_dynamic_answer_to_plan(
    answer: str,
    plan: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Move complete site records to the same element order used at inference.

    Within each element, source order is stable.  The operation is a pure site
    permutation: lattice and every species-coordinate record are preserved.
    """

    arrays = parse_dynamic_answer(answer, strict=True)
    expected = expanded_plan_species(plan)
    actual = [str(value) for value in arrays["species"]]
    if Counter(expected) != Counter(actual):
        raise ValueError("answer and Plan species multisets differ")
    available: dict[str, deque[int]] = defaultdict(deque)
    for index, symbol in enumerate(actual):
        available[symbol].append(index)
    order = [available[symbol].popleft() for symbol in expected]
    canonical, _diagnostics = arrays_to_dynamic_answer(
        arrays["lengths"],
        arrays["angles"],
        expected,
        [arrays["frac_coords"][index] for index in order],
    )
    source_tokens = tokenize_answer_text(answer)
    canonical_tokens = tokenize_answer_text(canonical)
    if len(canonical_tokens) != len(source_tokens):
        raise RuntimeError("site permutation changed dynamic body length")
    parsed = parse_dynamic_answer(canonical, strict=True)
    if parsed["lengths"] != arrays["lengths"] or parsed["angles"] != arrays["angles"]:
        raise RuntimeError("site permutation changed lattice")
    source_records = Counter(
        (symbol, *map(float, coordinate))
        for symbol, coordinate in zip(actual, arrays["frac_coords"], strict=True)
    )
    canonical_records = Counter(
        (symbol, *map(float, coordinate))
        for symbol, coordinate in zip(
            parsed["species"], parsed["frac_coords"], strict=True
        )
    )
    if source_records != canonical_records:
        raise RuntimeError("site permutation changed species-coordinate records")
    return canonical, {
        "changed": canonical != answer,
        "mismatched_element_slots": sum(
            source != target for source, target in zip(actual, expected, strict=True)
        ),
        "site_permutation": order,
    }


__all__ = ["canonicalize_dynamic_answer_to_plan", "expanded_plan_species"]
