"""Family-aware viability kernel for C3FD semantic decoding.

The original v2.1 sampler combined an exact atom/charge oracle with a
separate family-prefix check.  The intersection of two individually reachable
sets need not itself be reachable.  This module keeps an action only when an
actual benchmark-compatible suffix exists that jointly satisfies family, N,
charge, and exact arity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from crystal_dlm.ccfd import FormulaToken
from crystal_dlm.ccfd_v2 import (
    BenchmarkReachability,
    BenchmarkValidator,
    CCFDv2State,
    default_benchmark_validator,
)
from crystal_dlm.composition_pair_prior import ValenceNode
from crystal_dlm.fixed_slot import Z_TO_SYMBOL
from crystal_dlm.r5_plan_state import anion_framework_from_symbols


FAMILY_FORBIDDEN = {
    "oxide": frozenset(),
    "sulfide": frozenset({"O"}),
    "chalcogenide": frozenset({"O", "S"}),
    "halide": frozenset({"O", "S", "Se", "Te"}),
    "nitride": frozenset({"O", "S", "Se", "Te", "F", "Cl", "Br", "I"}),
    "phosphide_or_phosphate": frozenset(
        {"O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N"}
    ),
    "other": frozenset(
        {"O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P"}
    ),
}

FAMILY_REQUIRED = {
    "oxide": frozenset({"O"}),
    "sulfide": frozenset({"S"}),
    "chalcogenide": frozenset({"Se", "Te"}),
    "halide": frozenset({"F", "Cl", "Br", "I"}),
    "nitride": frozenset({"N"}),
    "phosphide_or_phosphate": frozenset({"P"}),
    "other": frozenset(),
}


def element_allowed_for_family(symbol: str, family: str) -> bool:
    if family not in FAMILY_FORBIDDEN:
        raise ValueError(f"unknown proposal family {family!r}")
    return str(symbol) not in FAMILY_FORBIDDEN[family]


def state_symbols(state: CCFDv2State) -> tuple[str, ...]:
    return tuple(Z_TO_SYMBOL[int(value)] for value in state.distinct_elements)


def family_prefix_reachable(
    state: CCFDv2State,
    *,
    family: str,
    target_arity: int,
    vocabulary_nodes: Sequence[ValenceNode],
) -> bool:
    """Cheap necessary family-prefix condition used before exact search."""

    symbols = set(state_symbols(state))
    if any(not element_allowed_for_family(symbol, family) for symbol in symbols):
        return False
    required = FAMILY_REQUIRED[family]
    if not required or symbols.intersection(required):
        return True
    slots_left = int(target_arity) - len(state.tokens)
    if slots_left <= 0:
        return False
    last_element = max(state.distinct_elements, default=0)
    return any(
        int(node.atomic_number) > int(last_element)
        and Z_TO_SYMBOL[int(node.atomic_number)] in required
        and (
            state.branch is None
            or (state.branch == "alloy" and int(node.oxidation_state) == 0)
            or (state.branch == "ionic" and int(node.oxidation_state) != 0)
        )
        for node in vocabulary_nodes
    )


class FamilyAwareBenchmarkReachability:
    """Exact suffix-existence oracle over the benchmark semantic vocabulary.

    The generic dynamic program first removes paths that cannot close exact N,
    charge, branch, and arity.  A memoized depth-first viability search then
    intersects those paths with the requested family and the independent
    terminal benchmark certificate.  Consequently every returned action has
    at least one valid continuation; sampling one cannot create a semantic
    dead end later.
    """

    def __init__(self, nodes: Sequence[ValenceNode]) -> None:
        canonical = tuple(
            sorted(
                {
                    ValenceNode(int(node.atomic_number), int(node.oxidation_state))
                    for node in nodes
                }
            )
        )
        if not canonical:
            raise ValueError("family-aware reachability requires species nodes")
        self.nodes = canonical
        self.generic = BenchmarkReachability(
            tuple((node.atomic_number, node.oxidation_state) for node in canonical)
        )
        self._completion_cache: dict[tuple[Any, ...], bool] = {}
        self._stats = {
            "states_evaluated": 0,
            "cache_hits": 0,
            "terminal_checks": 0,
        }

    def clear_cache(self) -> None:
        self._completion_cache.clear()
        for key in self._stats:
            self._stats[key] = 0

    def stats(self) -> Mapping[str, int]:
        return {
            **self._stats,
            "cached_states": len(self._completion_cache),
        }

    @staticmethod
    def _terminal_is_valid(
        state: CCFDv2State,
        *,
        family: str,
        target_arity: int,
        benchmark_validator: BenchmarkValidator,
    ) -> bool:
        if (
            not state.conservation_complete
            or len(state.tokens) != int(target_arity)
            or anion_framework_from_symbols(state_symbols(state)) != str(family)
        ):
            return False
        return state.end().certificate(
            benchmark_validator=benchmark_validator
        ).benchmark_compatible

    def can_complete(
        self,
        state: CCFDv2State,
        *,
        family: str,
        target_arity: int,
        benchmark_validator: BenchmarkValidator = default_benchmark_validator,
        max_species: int = 7,
    ) -> bool:
        family = str(family)
        target = int(target_arity)
        if family not in FAMILY_FORBIDDEN:
            raise ValueError(f"unknown proposal family {family!r}")
        key = (
            state,
            family,
            target,
            int(max_species),
            id(benchmark_validator),
        )
        cached = self._completion_cache.get(key)
        if cached is not None:
            self._stats["cache_hits"] += 1
            return bool(cached)
        self._stats["states_evaluated"] += 1

        if (
            state.ended
            or state.target_atoms is None
            or state.remaining_atoms is None
            or target <= 0
            or target > int(max_species)
            or len(state.tokens) > target
            or not family_prefix_reachable(
                state,
                family=family,
                target_arity=target,
                vocabulary_nodes=self.nodes,
            )
        ):
            result = False
        elif int(state.remaining_atoms) == 0:
            self._stats["terminal_checks"] += 1
            result = self._terminal_is_valid(
                state,
                family=family,
                target_arity=target,
                benchmark_validator=benchmark_validator,
            )
        elif len(state.tokens) >= target:
            result = False
        else:
            # Existence needs one witness, whereas the public legality method
            # must return every viable immediate action.  Short-circuiting
            # here leaves that action set unchanged and avoids recursively
            # materializing the full continuation tree for each candidate.
            result = False
            for token in self.generic.legal_species_counts(
                state,
                benchmark_validator=benchmark_validator,
                max_species=max_species,
                target_arity=target,
            ):
                symbol = Z_TO_SYMBOL[int(token.atomic_number)]
                if not element_allowed_for_family(symbol, family):
                    continue
                candidate = state.apply(token, max_species=max_species)
                if self.can_complete(
                    candidate,
                    family=family,
                    target_arity=target,
                    benchmark_validator=benchmark_validator,
                    max_species=max_species,
                ):
                    result = True
                    break
        self._completion_cache[key] = bool(result)
        return bool(result)

    def legal_species_counts(
        self,
        state: CCFDv2State,
        *,
        family: str,
        target_arity: int,
        benchmark_validator: BenchmarkValidator = default_benchmark_validator,
        max_species: int = 7,
    ) -> tuple[FormulaToken, ...]:
        """Return actions whose joint family/ledger/certificate suffix exists."""

        generic_actions = self.generic.legal_species_counts(
            state,
            benchmark_validator=benchmark_validator,
            max_species=max_species,
            target_arity=target_arity,
        )
        legal: list[FormulaToken] = []
        for token in generic_actions:
            symbol = Z_TO_SYMBOL[int(token.atomic_number)]
            if not element_allowed_for_family(symbol, family):
                continue
            candidate = state.apply(token, max_species=max_species)
            if self.can_complete(
                candidate,
                family=family,
                target_arity=target_arity,
                benchmark_validator=benchmark_validator,
                max_species=max_species,
            ):
                legal.append(token)
        return tuple(legal)


__all__ = [
    "FAMILY_FORBIDDEN",
    "FAMILY_REQUIRED",
    "FamilyAwareBenchmarkReachability",
    "element_allowed_for_family",
    "family_prefix_reachable",
    "state_symbols",
]
