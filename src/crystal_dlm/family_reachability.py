"""Family-aware viability kernel for C3FD semantic decoding.

The original v2.1 sampler combined an exact atom/charge oracle with a
separate family-prefix check.  The intersection of two individually reachable
sets need not itself be reachable.  This module keeps an action only when an
actual benchmark-compatible suffix exists that jointly satisfies family, N,
charge, and exact arity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
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


class PaulingWitnessReachability:
    """Compiled finite-state viability oracle with an explicit SMACT witness.

    Instead of searching complete formulas and invoking the benchmark at every
    leaf, this oracle carries the sufficient certificate used by SMACT's
    Pauling screen: exact charge neutrality plus every cation being less
    electronegative than every anion.  Zero-valence multi-element branches are
    accepted only when every selected element is a metal; unary compositions
    retain the benchmark shortcut.

    The generated oxidation states are therefore themselves a constructive
    benchmark witness.  The sampler still invokes the independent benchmark
    certificate at EOS; this class only compiles its sufficient conditions into
    the online action mask.
    """

    def __init__(
        self,
        nodes: Sequence[ValenceNode],
        *,
        electronegativity_by_atomic_number: Mapping[int, float | None] | None = None,
        metal_atomic_numbers: Sequence[int] | set[int] | frozenset[int] | None = None,
    ) -> None:
        canonical = tuple(
            sorted(
                {
                    ValenceNode(int(node.atomic_number), int(node.oxidation_state))
                    for node in nodes
                }
            )
        )
        if not canonical:
            raise ValueError("Pauling reachability requires species nodes")
        grouped: dict[int, set[int]] = {}
        for node in canonical:
            grouped.setdefault(int(node.atomic_number), set()).add(
                int(node.oxidation_state)
            )
        self.nodes = canonical
        self.elements = tuple(sorted(grouped))
        self.states = tuple(
            tuple(sorted(grouped[atomic_number]))
            for atomic_number in self.elements
        )
        self.element_to_index = {
            atomic_number: index for index, atomic_number in enumerate(self.elements)
        }

        if electronegativity_by_atomic_number is None or metal_atomic_numbers is None:
            default_eneg, default_metals = self._load_smact_properties(self.elements)
            if electronegativity_by_atomic_number is None:
                electronegativity_by_atomic_number = default_eneg
            if metal_atomic_numbers is None:
                metal_atomic_numbers = default_metals
        self.electronegativity = {
            int(atomic_number): (
                None if value is None else float(value)
            )
            for atomic_number, value in electronegativity_by_atomic_number.items()
        }
        self.metals = frozenset(int(value) for value in metal_atomic_numbers)
        finite_eneg = sorted(
            {
                float(value)
                for atomic_number in self.elements
                for value in (self.electronegativity.get(int(atomic_number)),)
                if value is not None
            }
        )
        self.eneg_rank = {
            value: index for index, value in enumerate(finite_eneg)
        }
        self.element_eneg_rank = {
            atomic_number: (
                None
                if self.electronegativity.get(int(atomic_number)) is None
                else self.eneg_rank[
                    float(self.electronegativity[int(atomic_number)])
                ]
            )
            for atomic_number in self.elements
        }
        self.no_cation_rank = -1
        self.no_anion_rank = len(finite_eneg)
        self._states_evaluated = 0

    @staticmethod
    def _load_smact_properties(
        atomic_numbers: Sequence[int],
    ) -> tuple[dict[int, float | None], frozenset[int]]:
        import smact

        symbols = [Z_TO_SYMBOL[int(value)] for value in atomic_numbers]
        space = smact.element_dictionary(symbols)
        eneg = {
            int(atomic_number): (
                None
                if getattr(space.get(symbol), "pauling_eneg", None) is None
                else float(space[symbol].pauling_eneg)
            )
            for atomic_number, symbol in zip(atomic_numbers, symbols)
        }
        metals = frozenset(
            int(atomic_number)
            for atomic_number, symbol in zip(atomic_numbers, symbols)
            if symbol in smact.metals
        )
        return eneg, metals

    def clear_cache(self) -> None:
        self._can_suffix.cache_clear()
        self._states_evaluated = 0

    def stats(self) -> Mapping[str, int]:
        cache = self._can_suffix.cache_info()
        return {
            "states_evaluated": int(self._states_evaluated),
            "cache_hits": int(cache.hits),
            "cache_misses": int(cache.misses),
            "cached_states": int(cache.currsize),
        }

    def _state_summary(
        self,
        state: CCFDv2State,
        *,
        family: str,
        target_arity: int,
    ) -> tuple[int, int, int, int, str | None, bool, int, int, bool] | None:
        if (
            state.ended
            or state.target_atoms is None
            or state.remaining_atoms is None
            or len(state.tokens) != len(state.distinct_elements)
            or len(state.tokens) > int(target_arity)
        ):
            return None
        symbols = state_symbols(state)
        if any(not element_allowed_for_family(symbol, family) for symbol in symbols):
            return None
        required = FAMILY_REQUIRED[str(family)]
        required_hit = bool(not required or set(symbols).intersection(required))
        max_cation = self.no_cation_rank
        min_anion = self.no_anion_rank
        all_metal = True
        for token in state.tokens:
            atomic_number = int(token.atomic_number)
            oxidation = int(token.oxidation_state)
            all_metal = all_metal and atomic_number in self.metals
            if oxidation == 0:
                continue
            rank = self.element_eneg_rank.get(atomic_number)
            if rank is None:
                return None
            if oxidation > 0:
                max_cation = max(max_cation, int(rank))
            else:
                min_anion = min(min_anion, int(rank))
        last_element = max(state.distinct_elements, default=0)
        start = 0
        while start < len(self.elements) and self.elements[start] <= last_element:
            start += 1
        return (
            start,
            int(state.remaining_atoms),
            int(target_arity) - len(state.tokens),
            int(state.net_charge),
            state.branch,
            required_hit,
            max_cation,
            min_anion,
            all_metal,
        )

    def _terminal_witness(
        self,
        *,
        target_arity: int,
        branch: str | None,
        required_hit: bool,
        max_cation: int,
        min_anion: int,
        all_metal: bool,
    ) -> bool:
        if not required_hit or int(target_arity) <= 0:
            return False
        if int(target_arity) == 1:
            return branch == "alloy"
        if branch == "alloy":
            return bool(all_metal)
        if branch != "ionic":
            return False
        return bool(
            max_cation != self.no_cation_rank
            and min_anion != self.no_anion_rank
            and int(max_cation) < int(min_anion)
        )

    @lru_cache(maxsize=None)
    def _can_suffix(
        self,
        family: str,
        target_arity: int,
        start_index: int,
        remaining_atoms: int,
        remaining_slots: int,
        net_charge: int,
        branch: str | None,
        required_hit: bool,
        max_cation: int,
        min_anion: int,
        all_metal: bool,
    ) -> bool:
        self._states_evaluated += 1
        atoms = int(remaining_atoms)
        slots = int(remaining_slots)
        start = int(start_index)
        if atoms == 0:
            return bool(
                slots == 0
                and int(net_charge) == 0
                and self._terminal_witness(
                    target_arity=target_arity,
                    branch=branch,
                    required_hit=required_hit,
                    max_cation=max_cation,
                    min_anion=min_anion,
                    all_metal=all_metal,
                )
            )
        if (
            atoms < slots
            or slots <= 0
            or start >= len(self.elements)
            or len(self.elements) - start < slots
        ):
            return False

        required = FAMILY_REQUIRED[str(family)]
        for index in range(start, len(self.elements)):
            if len(self.elements) - index < slots:
                break
            atomic_number = int(self.elements[index])
            symbol = Z_TO_SYMBOL[atomic_number]
            if not element_allowed_for_family(symbol, family):
                continue
            for oxidation in self.states[index]:
                token_branch = "alloy" if int(oxidation) == 0 else "ionic"
                if branch is not None and branch != token_branch:
                    continue
                rank = self.element_eneg_rank.get(atomic_number)
                if token_branch == "ionic" and rank is None:
                    continue
                next_max_cation = int(max_cation)
                next_min_anion = int(min_anion)
                if int(oxidation) > 0:
                    next_max_cation = max(next_max_cation, int(rank))
                elif int(oxidation) < 0:
                    next_min_anion = min(next_min_anion, int(rank))
                # Once the Pauling inequality is violated, adding species
                # cannot restore it.
                if (
                    next_max_cation != self.no_cation_rank
                    and next_min_anion != self.no_anion_rank
                    and next_max_cation >= next_min_anion
                ):
                    continue
                next_required = bool(
                    required_hit or (required and symbol in required)
                )
                next_all_metal = bool(all_metal and atomic_number in self.metals)
                max_count = atoms - (slots - 1)
                for count in range(1, max_count + 1):
                    if self._can_suffix(
                        family,
                        int(target_arity),
                        index + 1,
                        atoms - count,
                        slots - 1,
                        int(net_charge) + int(oxidation) * count,
                        branch or token_branch,
                        next_required,
                        next_max_cation,
                        next_min_anion,
                        next_all_metal,
                    ):
                        return True
        return False

    def terminal_witness_valid(
        self,
        state: CCFDv2State,
        *,
        family: str,
        target_arity: int,
    ) -> bool:
        summary = self._state_summary(
            state, family=family, target_arity=target_arity
        )
        if summary is None:
            return False
        (
            _start,
            atoms,
            slots,
            charge,
            branch,
            required_hit,
            max_cation,
            min_anion,
            all_metal,
        ) = summary
        return bool(
            atoms == 0
            and slots == 0
            and charge == 0
            and self._terminal_witness(
                target_arity=target_arity,
                branch=branch,
                required_hit=required_hit,
                max_cation=max_cation,
                min_anion=min_anion,
                all_metal=all_metal,
            )
        )

    def can_complete(
        self,
        state: CCFDv2State,
        *,
        family: str,
        target_arity: int,
        max_species: int = 7,
    ) -> bool:
        family = str(family)
        target = int(target_arity)
        if family not in FAMILY_FORBIDDEN:
            raise ValueError(f"unknown proposal family {family!r}")
        if target <= 0 or target > int(max_species):
            return False
        summary = self._state_summary(state, family=family, target_arity=target)
        if summary is None:
            return False
        return self._can_suffix(family, target, *summary)

    def legal_species_counts(
        self,
        state: CCFDv2State,
        *,
        family: str,
        target_arity: int,
        max_species: int = 7,
    ) -> tuple[FormulaToken, ...]:
        family = str(family)
        target = int(target_arity)
        summary = self._state_summary(state, family=family, target_arity=target)
        if summary is None or target <= 0 or target > int(max_species):
            return ()
        start, atoms, slots, _charge, branch, *_rest = summary
        if atoms <= 0 or slots <= 0:
            return ()
        legal: list[FormulaToken] = []
        max_count = atoms - (slots - 1)
        for index in range(start, len(self.elements)):
            if len(self.elements) - index < slots:
                break
            atomic_number = int(self.elements[index])
            symbol = Z_TO_SYMBOL[atomic_number]
            if not element_allowed_for_family(symbol, family):
                continue
            for oxidation in self.states[index]:
                token_branch = "alloy" if int(oxidation) == 0 else "ionic"
                if branch is not None and branch != token_branch:
                    continue
                for count in range(1, max_count + 1):
                    token = FormulaToken(atomic_number, int(oxidation), count)
                    candidate = state.apply(token, max_species=max_species)
                    if self.can_complete(
                        candidate,
                        family=family,
                        target_arity=target,
                        max_species=max_species,
                    ):
                        legal.append(token)
        return tuple(legal)


__all__ = [
    "FAMILY_FORBIDDEN",
    "FAMILY_REQUIRED",
    "FamilyAwareBenchmarkReachability",
    "PaulingWitnessReachability",
    "element_allowed_for_family",
    "family_prefix_reachable",
    "state_symbols",
]
