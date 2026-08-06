"""Chemistry-Reachable Plan decoding for the frozen H1 formula Planner.

CR-Plan changes only the legal token support while the model is emitting the
value of the first ``formula:`` line.  It does not repair, retry, replace,
filter, or rerank samples.  The implementation deliberately separates:

* evaluator-aligned, one-oxidation-state-per-element neutral witnesses;
* mixed-valence-only neutral witnesses, which remain reachable but are not
  counted as primary composition-validity gains;
* unary, all-metal, and oxidation-table-missing non-applicable strata; and
* charge-invalid terminal formulae, whose newline is blocked.

The module has no import-time dependency on SMACT, Transformers, or PyTorch.
This keeps its finite-state and dynamic-programming logic testable in a small
CPU-only environment.  Frozen runtime packages are loaded explicitly by the
execution audit.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import dataclasses
import hashlib
import json
import math
from functools import lru_cache
import re
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from crystal_dlm.fixed_slot import CHEMICAL_SYMBOLS, SYMBOL_TO_Z


CRPLAN_SCHEMA = "h1_crplan_v2"
CRPLAN_MODES = ("grammar_only", "terminal_only", "full_prefix")
MISSING_STATE_POLICIES = ("allow_non_applicable", "fail_closed")
FORMULA_LABEL = "formula:"
FORMULA_SYMBOLS = tuple(
    symbol
    for symbol in CHEMICAL_SYMBOLS
    if symbol != "X" and symbol in SYMBOL_TO_Z
)
FORMULA_SYMBOL_SET = frozenset(FORMULA_SYMBOLS)
FORMULA_SYMBOL_PREFIXES = frozenset(
    symbol[:width]
    for symbol in FORMULA_SYMBOLS
    for width in range(1, len(symbol) + 1)
)


class CRPlanError(RuntimeError):
    """Base class for fail-closed CR-Plan errors."""


class FormulaGrammarError(CRPlanError):
    """Raised when a generated token leaves the frozen flat-formula grammar."""


class CRPlanDeadEndError(CRPlanError):
    """Raised when the tokenizer has no legal continuation."""


class TerminalChargeError(FormulaGrammarError):
    """Raised when a token tries to terminate a charge-invalid formula."""

    def __init__(self, certificate: "TerminalCertificate") -> None:
        super().__init__(
            "formula termination has no table-relative neutral witness: "
            f"{certificate.stratum}"
        )
        self.certificate = certificate


class CRPlanIdentityError(CRPlanError):
    """Raised when generated text/parser identity is not the masked formula."""


def _canonical_counts(
    counts: Mapping[str, int] | Iterable[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    values = counts.items() if isinstance(counts, Mapping) else counts
    counter: Counter[str] = Counter()
    for symbol, raw_count in values:
        symbol = str(symbol)
        count = int(raw_count)
        if symbol not in FORMULA_SYMBOL_SET:
            raise ValueError(f"unsupported formula element {symbol!r}")
        if count <= 0:
            raise ValueError(f"non-positive formula count for {symbol}: {count}")
        counter[symbol] += count
    return tuple(
        (symbol, int(counter[symbol]))
        for symbol in sorted(counter, key=lambda value: SYMBOL_TO_Z[value])
    )


def _allocation_dict(
    states: Sequence[int],
    allocation: Sequence[int],
) -> dict[str, int]:
    return {
        str(int(state)): int(count)
        for state, count in zip(states, allocation)
        if int(count) > 0
    }


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalCertificate:
    """A reproducible, table-relative terminal applicability certificate."""

    counts: tuple[tuple[str, int], ...]
    total_atoms: int
    stratum: str
    terminal_allowed: bool
    charge_applicable: bool
    primary_charge_witness: bool
    uniform_oxidation_witness: tuple[tuple[str, int], ...] = ()
    mixed_valence_witness: tuple[
        tuple[str, tuple[tuple[int, int], ...]], ...
    ] = ()
    missing_elements: tuple[str, ...] = ()
    missing_state_policy: str = "allow_non_applicable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "h1_crplan_terminal_certificate_v1",
            "counts": [
                {"element": symbol, "count": count}
                for symbol, count in self.counts
            ],
            "total_atoms": self.total_atoms,
            "stratum": self.stratum,
            "terminal_allowed": self.terminal_allowed,
            "charge_applicable": self.charge_applicable,
            "primary_charge_witness": self.primary_charge_witness,
            "uniform_oxidation_witness": {
                symbol: oxidation
                for symbol, oxidation in self.uniform_oxidation_witness
            },
            "mixed_valence_witness": {
                symbol: {
                    str(oxidation): allocation
                    for oxidation, allocation in values
                }
                for symbol, values in self.mixed_valence_witness
            },
            "missing_elements": list(self.missing_elements),
            "missing_state_policy": self.missing_state_policy,
            "relative_to_frozen_oxidation_table_only": True,
            "pauling_checked": False,
            "body_condition_changed": False,
        }


@dataclasses.dataclass(slots=True)
class ReachabilityDiagnostics:
    queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    states_created: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "queries": int(self.queries),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "states_created": int(self.states_created),
        }


class OxidationReachability:
    """Memoized neutral-charge reachability over a frozen oxidation table."""

    def __init__(
        self,
        oxidation_states: Mapping[str, Sequence[int]],
        *,
        metals: Iterable[str] = (),
        max_atoms: int = 20,
        table_source: str = "unspecified",
        table_version: str = "unspecified",
        missing_state_policy: str = "allow_non_applicable",
    ) -> None:
        if int(max_atoms) < 1:
            raise ValueError("max_atoms must be positive")
        if str(missing_state_policy) not in MISSING_STATE_POLICIES:
            raise ValueError(
                "missing_state_policy must be one of "
                f"{MISSING_STATE_POLICIES}, got {missing_state_policy!r}"
            )
        self.max_atoms = int(max_atoms)
        self.missing_state_policy = str(missing_state_policy)
        self.oxidation_states = {
            symbol: tuple(sorted(set(int(value) for value in oxidation_states.get(symbol, ()))))
            for symbol in FORMULA_SYMBOLS
        }
        self.metals = frozenset(str(value) for value in metals)
        self.table_source = str(table_source)
        self.table_version = str(table_version)
        canonical = {
            "schema": "h1_crplan_oxidation_table_v1",
            "source": self.table_source,
            "version": self.table_version,
            "max_atoms": self.max_atoms,
            "states": {
                symbol: list(self.oxidation_states[symbol])
                for symbol in FORMULA_SYMBOLS
            },
            "metals": sorted(self.metals),
        }
        self.table_sha256 = hashlib.sha256(
            json.dumps(
                canonical,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.constraint_contract_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "schema": CRPLAN_SCHEMA,
                    "table_sha256": self.table_sha256,
                    "missing_state_policy": self.missing_state_policy,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.diagnostics = ReachabilityDiagnostics()
        self._terminal_cache: dict[
            tuple[tuple[str, int], ...], TerminalCertificate
        ] = {}
        self._prefix_cache: dict[tuple[tuple[str, int], ...], bool] = {}
        self._mixed_charge_cache: dict[
            tuple[tuple[str, int], ...], frozenset[int]
        ] = {}
        self._terminal_decision_cache: dict[
            tuple[tuple[str, int], ...], TerminalCertificate
        ] = {}
        self._element_charge_mask_cache: dict[
            tuple[str, int], int
        ] = {}
        self._mixed_charge_mask_cache: dict[
            tuple[tuple[str, int], ...], int
        ] = {}
        atom_charges = {
            state
            for states in self.oxidation_states.values()
            for state in states
        }
        self._atom_charges = tuple(sorted(atom_charges))
        self._missing_state_elements = tuple(
            symbol
            for symbol, states in self.oxidation_states.items()
            if not states
        )
        exact: list[frozenset[int]] = [frozenset((0,))]
        for _ in range(self.max_atoms):
            previous = exact[-1]
            current = frozenset(
                int(total + charge)
                for total in previous
                for charge in self._atom_charges
            )
            exact.append(current)
        self._suffix_exact_charge_sets = tuple(exact)
        max_abs_charge = max(
            (abs(int(value)) for value in self._atom_charges),
            default=0,
        )
        self._charge_offset = int(max_abs_charge * self.max_atoms)
        self._charge_domain_mask = (
            (1 << (2 * self._charge_offset + 1)) - 1
        )
        self._charge_zero_bit = 1 << self._charge_offset
        suffix_union: set[int] = set()
        suffix_negated_masks: list[int] = [0]
        for future_atoms in range(1, self.max_atoms + 1):
            suffix_union.update(
                self._suffix_exact_charge_sets[future_atoms]
            )
            mask = 0
            for charge in suffix_union:
                bit_index = self._charge_offset - int(charge)
                if 0 <= bit_index <= 2 * self._charge_offset:
                    mask |= 1 << bit_index
            suffix_negated_masks.append(mask)
        self._suffix_negated_union_masks = tuple(suffix_negated_masks)

    def table_report(self) -> dict[str, Any]:
        return {
            "schema": "h1_crplan_oxidation_table_report_v1",
            "source": self.table_source,
            "version": self.table_version,
            "sha256": self.table_sha256,
            "constraint_contract_sha256": self.constraint_contract_sha256,
            "missing_state_policy": self.missing_state_policy,
            "max_atoms": self.max_atoms,
            "element_count": len(self.oxidation_states),
            "elements_with_states": sum(
                int(bool(values)) for values in self.oxidation_states.values()
            ),
            "elements_without_states": [
                symbol
                for symbol, values in self.oxidation_states.items()
                if not values
            ],
            "metal_count": len(self.metals),
            "atom_charge_union": list(self._atom_charges),
        }

    def cache_report(self) -> dict[str, int]:
        element_cache = self._element_charge_allocations_cached.cache_info()
        return {
            "terminal_certificate_entries": len(self._terminal_cache),
            "terminal_decision_entries": len(
                self._terminal_decision_cache
            ),
            "prefix_reachability_entries": len(self._prefix_cache),
            "mixed_charge_set_entries": len(self._mixed_charge_cache),
            "mixed_charge_mask_entries": len(
                self._mixed_charge_mask_cache
            ),
            "element_charge_mask_entries": len(
                self._element_charge_mask_cache
            ),
            "element_allocation_entries_global": int(element_cache.currsize),
        }

    def _shift_charge_mask(self, mask: int, charge: int) -> int:
        if int(charge) >= 0:
            return (
                int(mask) << int(charge)
            ) & self._charge_domain_mask
        return int(mask) >> -int(charge)

    def _element_charge_mask(self, symbol: str, count: int) -> int:
        key = (str(symbol), int(count))
        cached = self._element_charge_mask_cache.get(key)
        self.diagnostics.queries += 1
        if cached is not None:
            self.diagnostics.cache_hits += 1
            return cached
        self.diagnostics.cache_misses += 1
        states = self.oxidation_states[str(symbol)]
        if not states:
            self._element_charge_mask_cache[key] = 0
            return 0
        reachable = self._charge_zero_bit
        for _ in range(int(count)):
            updated = 0
            for state in states:
                updated |= self._shift_charge_mask(
                    reachable,
                    int(state),
                )
            reachable = updated
        self.diagnostics.states_created += int(reachable.bit_count())
        self._element_charge_mask_cache[key] = int(reachable)
        return int(reachable)

    def _convolve_charge_masks(self, left: int, right: int) -> int:
        if not left or not right:
            return 0
        updated = 0
        remaining = int(right)
        while remaining:
            bit = remaining & -remaining
            bit_index = bit.bit_length() - 1
            updated |= self._shift_charge_mask(
                left,
                bit_index - self._charge_offset,
            )
            remaining ^= bit
        return int(updated)

    def _mixed_charge_mask(
        self,
        counts: tuple[tuple[str, int], ...],
    ) -> int:
        cached = self._mixed_charge_mask_cache.get(counts)
        if cached is not None:
            return cached
        reachable = self._charge_zero_bit
        for symbol, count in counts:
            element = self._element_charge_mask(symbol, count)
            if not element:
                reachable = 0
                break
            reachable = self._convolve_charge_masks(
                reachable,
                element,
            )
            self.diagnostics.states_created += int(
                reachable.bit_count()
            )
        self._mixed_charge_mask_cache[counts] = int(reachable)
        return int(reachable)

    def _uniform_neutral_reachable(
        self,
        counts: tuple[tuple[str, int], ...],
    ) -> bool:
        reachable = self._charge_zero_bit
        for symbol, count in counts:
            updated = 0
            for state in self.oxidation_states[symbol]:
                updated |= self._shift_charge_mask(
                    reachable,
                    int(count) * int(state),
                )
            reachable = updated
            self.diagnostics.states_created += int(
                reachable.bit_count()
            )
            if not reachable:
                return False
        return bool(reachable & self._charge_zero_bit)

    @staticmethod
    @lru_cache(maxsize=None)
    def _element_charge_allocations_cached(
        states: tuple[int, ...],
        count: int,
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        if int(count) < 0:
            raise ValueError("element count cannot be negative")
        start = tuple(0 for _ in states)
        reachable: dict[int, tuple[int, ...]] = {0: start}
        for _ in range(int(count)):
            updated: dict[int, tuple[int, ...]] = {}
            for total, allocation in reachable.items():
                for index, state in enumerate(states):
                    next_allocation = list(allocation)
                    next_allocation[index] += 1
                    updated.setdefault(
                        int(total + state),
                        tuple(next_allocation),
                    )
            reachable = updated
        return tuple(sorted(reachable.items()))

    def element_charge_allocations(
        self,
        symbol: str,
        count: int,
    ) -> dict[int, tuple[int, ...]]:
        states = self.oxidation_states[str(symbol)]
        if not states:
            return {}
        before = self._element_charge_allocations_cached.cache_info()
        result = dict(
            self._element_charge_allocations_cached(states, int(count))
        )
        after = self._element_charge_allocations_cached.cache_info()
        self.diagnostics.queries += 1
        if after.hits > before.hits:
            self.diagnostics.cache_hits += 1
        else:
            self.diagnostics.cache_misses += 1
            self.diagnostics.states_created += len(result)
        return result

    def _uniform_witness(
        self,
        counts: tuple[tuple[str, int], ...],
    ) -> tuple[tuple[str, int], ...] | None:
        reachable: dict[int, tuple[tuple[str, int], ...]] = {0: ()}
        for symbol, count in counts:
            updated: dict[int, tuple[tuple[str, int], ...]] = {}
            for total, witness in reachable.items():
                for state in self.oxidation_states[symbol]:
                    updated.setdefault(
                        int(total + count * state),
                        (*witness, (symbol, int(state))),
                    )
            reachable = updated
            self.diagnostics.states_created += len(updated)
        return reachable.get(0)

    def _mixed_witness(
        self,
        counts: tuple[tuple[str, int], ...],
    ) -> tuple[tuple[str, tuple[tuple[int, int], ...]], ...] | None:
        reachable: dict[
            int,
            tuple[tuple[str, tuple[tuple[int, int], ...]], ...],
        ] = {0: ()}
        for symbol, count in counts:
            states = self.oxidation_states[symbol]
            allocations = self.element_charge_allocations(symbol, count)
            updated: dict[
                int,
                tuple[tuple[str, tuple[tuple[int, int], ...]], ...],
            ] = {}
            for total, witness in reachable.items():
                for charge, allocation in allocations.items():
                    row = tuple(
                        (int(state), int(number))
                        for state, number in zip(states, allocation)
                        if int(number) > 0
                    )
                    updated.setdefault(
                        int(total + charge),
                        (*witness, (symbol, row)),
                    )
            reachable = updated
            self.diagnostics.states_created += len(updated)
        return reachable.get(0)

    def terminal_certificate(
        self,
        counts: Mapping[str, int] | Iterable[tuple[str, int]],
    ) -> TerminalCertificate:
        canonical = _canonical_counts(counts)
        cached = self._terminal_cache.get(canonical)
        self.diagnostics.queries += 1
        if cached is not None:
            self.diagnostics.cache_hits += 1
            return cached
        self.diagnostics.cache_misses += 1
        total_atoms = sum(count for _, count in canonical)
        if not 1 <= total_atoms <= self.max_atoms:
            certificate = TerminalCertificate(
                counts=canonical,
                total_atoms=total_atoms,
                stratum="atom_budget_invalid",
                terminal_allowed=False,
                charge_applicable=False,
                primary_charge_witness=False,
            )
        elif len(canonical) == 1:
            certificate = TerminalCertificate(
                counts=canonical,
                total_atoms=total_atoms,
                stratum="charge_not_applicable_unary",
                terminal_allowed=True,
                charge_applicable=False,
                primary_charge_witness=False,
            )
        elif all(symbol in self.metals for symbol, _ in canonical):
            certificate = TerminalCertificate(
                counts=canonical,
                total_atoms=total_atoms,
                stratum="charge_not_applicable_all_metal",
                terminal_allowed=True,
                charge_applicable=False,
                primary_charge_witness=False,
            )
        else:
            missing = tuple(
                symbol
                for symbol, _ in canonical
                if not self.oxidation_states[symbol]
            )
            if missing:
                if self.missing_state_policy == "allow_non_applicable":
                    certificate = TerminalCertificate(
                        counts=canonical,
                        total_atoms=total_atoms,
                        stratum="charge_not_applicable_table_missing",
                        terminal_allowed=True,
                        charge_applicable=False,
                        primary_charge_witness=False,
                        missing_elements=missing,
                    )
                else:
                    certificate = TerminalCertificate(
                        counts=canonical,
                        total_atoms=total_atoms,
                        stratum="charge_applicable_oxidation_state_missing",
                        terminal_allowed=False,
                        charge_applicable=True,
                        primary_charge_witness=False,
                        missing_elements=missing,
                    )
            else:
                uniform = self._uniform_witness(canonical)
                if uniform is not None:
                    certificate = TerminalCertificate(
                        counts=canonical,
                        total_atoms=total_atoms,
                        stratum="charge_applicable_uniform_neutral",
                        terminal_allowed=True,
                        charge_applicable=True,
                        primary_charge_witness=True,
                        uniform_oxidation_witness=uniform,
                    )
                else:
                    mixed_charges = self._mixed_charge_set(canonical)
                    if 0 in mixed_charges:
                        mixed = self._mixed_witness(canonical)
                        if mixed is None:
                            raise AssertionError(
                                "mixed charge set contained zero without a witness"
                            )
                        certificate = TerminalCertificate(
                            counts=canonical,
                            total_atoms=total_atoms,
                            stratum="charge_applicable_mixed_valence_only",
                            terminal_allowed=True,
                            charge_applicable=True,
                            primary_charge_witness=False,
                            mixed_valence_witness=mixed,
                        )
                    else:
                        certificate = TerminalCertificate(
                            counts=canonical,
                            total_atoms=total_atoms,
                            stratum="charge_applicable_no_neutral_witness",
                            terminal_allowed=False,
                            charge_applicable=True,
                            primary_charge_witness=False,
                        )
        certificate = dataclasses.replace(
            certificate,
            missing_state_policy=self.missing_state_policy,
        )
        self._terminal_cache[canonical] = certificate
        return certificate

    def terminal_decision(
        self,
        counts: Mapping[str, int] | Iterable[tuple[str, int]],
    ) -> TerminalCertificate:
        """Return the exact terminal stratum without constructing witnesses.

        Trie support enumeration only needs terminal legality and the frozen
        stratum.  Full deterministic witnesses are still constructed by
        ``terminal_certificate`` for the one formula that is actually sampled.
        """

        canonical = _canonical_counts(counts)
        cached = self._terminal_decision_cache.get(canonical)
        self.diagnostics.queries += 1
        if cached is not None:
            self.diagnostics.cache_hits += 1
            return cached
        self.diagnostics.cache_misses += 1
        total_atoms = sum(count for _, count in canonical)
        if not 1 <= total_atoms <= self.max_atoms:
            certificate = TerminalCertificate(
                counts=canonical,
                total_atoms=total_atoms,
                stratum="atom_budget_invalid",
                terminal_allowed=False,
                charge_applicable=False,
                primary_charge_witness=False,
            )
        elif len(canonical) == 1:
            certificate = TerminalCertificate(
                counts=canonical,
                total_atoms=total_atoms,
                stratum="charge_not_applicable_unary",
                terminal_allowed=True,
                charge_applicable=False,
                primary_charge_witness=False,
            )
        elif all(symbol in self.metals for symbol, _ in canonical):
            certificate = TerminalCertificate(
                counts=canonical,
                total_atoms=total_atoms,
                stratum="charge_not_applicable_all_metal",
                terminal_allowed=True,
                charge_applicable=False,
                primary_charge_witness=False,
            )
        else:
            missing = tuple(
                symbol
                for symbol, _ in canonical
                if not self.oxidation_states[symbol]
            )
            if missing:
                if self.missing_state_policy == "allow_non_applicable":
                    certificate = TerminalCertificate(
                        counts=canonical,
                        total_atoms=total_atoms,
                        stratum="charge_not_applicable_table_missing",
                        terminal_allowed=True,
                        charge_applicable=False,
                        primary_charge_witness=False,
                        missing_elements=missing,
                    )
                else:
                    certificate = TerminalCertificate(
                        counts=canonical,
                        total_atoms=total_atoms,
                        stratum="charge_applicable_oxidation_state_missing",
                        terminal_allowed=False,
                        charge_applicable=True,
                        primary_charge_witness=False,
                        missing_elements=missing,
                    )
            elif self._uniform_neutral_reachable(canonical):
                certificate = TerminalCertificate(
                    counts=canonical,
                    total_atoms=total_atoms,
                    stratum="charge_applicable_uniform_neutral",
                    terminal_allowed=True,
                    charge_applicable=True,
                    primary_charge_witness=True,
                )
            elif (
                self._mixed_charge_mask(canonical)
                & self._charge_zero_bit
            ):
                certificate = TerminalCertificate(
                    counts=canonical,
                    total_atoms=total_atoms,
                    stratum="charge_applicable_mixed_valence_only",
                    terminal_allowed=True,
                    charge_applicable=True,
                    primary_charge_witness=False,
                )
            else:
                certificate = TerminalCertificate(
                    counts=canonical,
                    total_atoms=total_atoms,
                    stratum="charge_applicable_no_neutral_witness",
                    terminal_allowed=False,
                    charge_applicable=True,
                    primary_charge_witness=False,
                )
        certificate = dataclasses.replace(
            certificate,
            missing_state_policy=self.missing_state_policy,
        )
        self._terminal_decision_cache[canonical] = certificate
        return certificate

    def _mixed_charge_set(
        self,
        counts: tuple[tuple[str, int], ...],
    ) -> frozenset[int]:
        cached = self._mixed_charge_cache.get(counts)
        if cached is not None:
            return cached
        reachable = frozenset((0,))
        for symbol, count in counts:
            allocations = self.element_charge_allocations(symbol, count)
            if not allocations:
                reachable = frozenset()
                break
            reachable = frozenset(
                int(left + right)
                for left in reachable
                for right in allocations
            )
            self.diagnostics.states_created += len(reachable)
        self._mixed_charge_cache[counts] = reachable
        return reachable

    def materialized_prefix_reachable(
        self,
        counts: Mapping[str, int] | Iterable[tuple[str, int]],
    ) -> bool:
        """Return whether a materialized prefix has any neutral legal suffix.

        Future atoms may use any element/oxidation state in the frozen table.
        This is an exact mixed-valence charge-set DP for the declared prefix
        abstraction.  It is intentionally conservative with respect to element
        identity: false-positive continuations can remain legal, but a known
        neutral mixed-valence path is never falsely removed.
        """

        canonical = _canonical_counts(counts)
        cached = self._prefix_cache.get(canonical)
        self.diagnostics.queries += 1
        if cached is not None:
            self.diagnostics.cache_hits += 1
            return cached
        self.diagnostics.cache_misses += 1
        total = sum(count for _, count in canonical)
        if not 1 <= total <= self.max_atoms:
            self._prefix_cache[canonical] = False
            return False
        if len(canonical) == 1:
            self._prefix_cache[canonical] = True
            return True
        if all(symbol in self.metals for symbol, _ in canonical):
            self._prefix_cache[canonical] = True
            return True
        missing = tuple(
            symbol
            for symbol, _ in canonical
            if not self.oxidation_states[symbol]
        )
        if missing:
            allowed = self.missing_state_policy == "allow_non_applicable"
            self._prefix_cache[canonical] = allowed
            return allowed
        remaining = self.max_atoms - total
        if (
            remaining >= 1
            and self._missing_state_elements
            and self.missing_state_policy == "allow_non_applicable"
        ):
            # The frozen terminal contract explicitly allows table-missing
            # formulae as a non-primary, charge-not-applicable stratum.  A
            # missing-state atom is therefore a valid suffix and must be part
            # of prefix reachability, even though it has no numeric charge.
            self._prefix_cache[canonical] = True
            return True
        current_charges = self._mixed_charge_mask(canonical)
        reachable = bool(
            current_charges & self._charge_zero_bit
            or (
                remaining >= 1
                and current_charges
                & self._suffix_negated_union_masks[remaining]
            )
        )
        self._prefix_cache[canonical] = bool(reachable)
        return bool(reachable)


@dataclasses.dataclass(frozen=True, slots=True)
class FormulaValueCursor:
    """Character-level cursor for one flat integer-count formula value."""

    max_atoms: int = 20
    committed_counts: tuple[tuple[str, int], ...] = ()
    pending_symbol_prefix: str = ""
    count_digits: str = ""
    seen_element: bool = False
    done: bool = False
    certificate: TerminalCertificate | None = None

    @property
    def committed_total(self) -> int:
        return sum(count for _, count in self.committed_counts)

    def signature(self) -> tuple[Any, ...]:
        return (
            self.max_atoms,
            self.committed_counts,
            self.pending_symbol_prefix,
            self.count_digits,
            self.seen_element,
            self.done,
            None if self.certificate is None else self.certificate.stratum,
        )

    def possible_symbols(self) -> tuple[str, ...]:
        if not self.pending_symbol_prefix:
            return ()
        return tuple(
            symbol
            for symbol in FORMULA_SYMBOLS
            if symbol.startswith(self.pending_symbol_prefix)
        )

    def _count_options(self, budget: int) -> tuple[int, ...]:
        if int(budget) < 1:
            return ()
        if not self.count_digits:
            return tuple(range(1, int(budget) + 1))
        return tuple(
            value
            for value in range(1, int(budget) + 1)
            if str(value).startswith(self.count_digits)
        )

    def _merged_pending_counts(
        self,
        symbol: str,
        count: int,
    ) -> tuple[tuple[str, int], ...]:
        merged = dict(self.committed_counts)
        merged[str(symbol)] = (
            int(merged.get(str(symbol), 0)) + int(count)
        )
        return tuple(
            (value, int(merged[value]))
            for value in sorted(
                merged,
                key=lambda item: SYMBOL_TO_Z[item],
            )
        )

    def iter_materializations(
        self,
    ) -> Iterable[tuple[tuple[str, int], ...]]:
        if self.done:
            yield (
                self.certificate.counts
                if self.certificate is not None
                else self.committed_counts
            )
            return
        if not self.pending_symbol_prefix:
            return
        budget = self.max_atoms - self.committed_total
        for symbol in self.possible_symbols():
            for count in self._count_options(budget):
                yield self._merged_pending_counts(symbol, count)

    def materializations(self) -> tuple[tuple[tuple[str, int], ...], ...]:
        """Enumerate finite ways the current partial element/count can finish."""

        return tuple(sorted(set(self.iter_materializations())))

    def grammar_prefix_reachable(self) -> bool:
        if self.done:
            return True
        if not self.seen_element and not self.pending_symbol_prefix:
            return True
        if not self.pending_symbol_prefix:
            return False
        budget = self.max_atoms - self.committed_total
        return bool(
            self.possible_symbols()
            and self._count_options(budget)
        )

    def chemistry_prefix_reachable(
        self,
        reachability: OxidationReachability,
    ) -> bool:
        if self.done:
            return bool(self.certificate and self.certificate.terminal_allowed)
        if not self.seen_element and not self.pending_symbol_prefix:
            return True
        if not self.committed_counts:
            # Any syntactically complete first element is a unary terminal,
            # which is an explicit frozen shortcut irrespective of table
            # coverage.
            return self.grammar_prefix_reachable()
        return any(
            reachability.materialized_prefix_reachable(counts)
            for counts in self.iter_materializations()
        )

    def _commit_pending(self) -> "FormulaValueCursor":
        prefix = self.pending_symbol_prefix
        if prefix not in FORMULA_SYMBOL_SET:
            raise FormulaGrammarError(
                f"incomplete or unsupported element symbol {prefix!r}"
            )
        count = int(self.count_digits) if self.count_digits else 1
        if count <= 0:
            raise FormulaGrammarError("element count must be positive")
        merged = Counter(dict(self.committed_counts))
        merged[prefix] += count
        canonical = _canonical_counts(merged)
        if sum(value for _, value in canonical) > self.max_atoms:
            raise FormulaGrammarError("formula exceeds frozen atom budget")
        return dataclasses.replace(
            self,
            committed_counts=canonical,
            pending_symbol_prefix="",
            count_digits="",
            seen_element=True,
        )

    def _terminate(
        self,
        *,
        mode: str,
        reachability: OxidationReachability,
        speculative: bool = False,
    ) -> "FormulaValueCursor":
        cursor = self._commit_pending()
        certificate = (
            reachability.terminal_decision(cursor.committed_counts)
            if speculative
            else reachability.terminal_certificate(
                cursor.committed_counts
            )
        )
        if mode in ("terminal_only", "full_prefix") and not certificate.terminal_allowed:
            raise TerminalChargeError(certificate)
        return dataclasses.replace(
            cursor,
            done=True,
            certificate=certificate,
        )

    def feed_character(
        self,
        character: str,
        *,
        mode: str,
        reachability: OxidationReachability,
        speculative: bool = False,
    ) -> "FormulaValueCursor":
        if mode not in CRPLAN_MODES:
            raise ValueError(f"unknown CR-Plan mode {mode!r}")
        if len(character) != 1:
            raise ValueError("feed_character expects exactly one character")
        if self.done:
            return self
        cursor = self
        if not cursor.seen_element and not cursor.pending_symbol_prefix:
            if character in (" ", "\t"):
                return cursor
            if not character.isupper() or not character.isascii():
                raise FormulaGrammarError(
                    "formula must begin with an ASCII element symbol"
                )
            if character not in FORMULA_SYMBOL_PREFIXES:
                raise FormulaGrammarError(
                    f"unsupported element-symbol prefix {character!r}"
                )
            cursor = dataclasses.replace(
                cursor,
                pending_symbol_prefix=character,
                seen_element=True,
            )
        elif character in ("\n", "\r"):
            if not cursor.pending_symbol_prefix:
                raise FormulaGrammarError("formula line ended without an element")
            return cursor._terminate(
                mode=mode,
                reachability=reachability,
                speculative=speculative,
            )
        elif character in (" ", "\t"):
            raise FormulaGrammarError(
                "whitespace is allowed only before the formula value"
            )
        elif character.isupper() and character.isascii():
            cursor = cursor._commit_pending()
            if cursor.committed_total >= cursor.max_atoms:
                raise FormulaGrammarError(
                    "new element would exceed frozen atom budget"
                )
            if character not in FORMULA_SYMBOL_PREFIXES:
                raise FormulaGrammarError(
                    f"unsupported element-symbol prefix {character!r}"
                )
            cursor = dataclasses.replace(
                cursor,
                pending_symbol_prefix=character,
            )
        elif character.islower() and character.isascii():
            if not cursor.pending_symbol_prefix or len(cursor.pending_symbol_prefix) != 1:
                raise FormulaGrammarError(
                    "lowercase formula byte is outside an element symbol"
                )
            if cursor.count_digits:
                raise FormulaGrammarError(
                    "lowercase element byte cannot follow a count"
                )
            symbol = cursor.pending_symbol_prefix + character
            if symbol not in FORMULA_SYMBOL_SET:
                raise FormulaGrammarError(f"unsupported element symbol {symbol!r}")
            cursor = dataclasses.replace(
                cursor,
                pending_symbol_prefix=symbol,
            )
        elif character.isdigit() and character.isascii():
            if not cursor.pending_symbol_prefix:
                raise FormulaGrammarError("formula count has no element")
            if len(cursor.pending_symbol_prefix) == 1 and (
                cursor.pending_symbol_prefix not in FORMULA_SYMBOL_SET
            ):
                raise FormulaGrammarError(
                    "count follows an incomplete element symbol"
                )
            digits = cursor.count_digits + character
            if digits.startswith("0"):
                raise FormulaGrammarError("formula count cannot start with zero")
            count = int(digits)
            if cursor.committed_total + count > cursor.max_atoms:
                raise FormulaGrammarError("formula count exceeds atom budget")
            cursor = dataclasses.replace(cursor, count_digits=digits)
        else:
            raise FormulaGrammarError(
                f"character {character!r} is outside flat formula grammar"
            )
        if mode == "full_prefix" and not cursor.chemistry_prefix_reachable(reachability):
            raise FormulaGrammarError(
                "formula prefix has no neutral completion within atom budget"
            )
        if not cursor.grammar_prefix_reachable():
            raise FormulaGrammarError(
                "formula prefix has no syntactically valid completion"
            )
        return cursor

    def feed(
        self,
        fragment: str,
        *,
        mode: str,
        reachability: OxidationReachability,
        speculative: bool = False,
    ) -> "FormulaValueCursor":
        cursor = self
        for character in str(fragment):
            cursor = cursor.feed_character(
                character,
                mode=mode,
                reachability=reachability,
                speculative=speculative,
            )
        return cursor


@dataclasses.dataclass(frozen=True, slots=True)
class PlanFormulaCursor:
    """Find the first formula label, then enforce its value until newline."""

    mode: str
    reachability: OxidationReachability
    phase: str = "seek_formula_label"
    seek_suffix: str = ""
    value: FormulaValueCursor | None = None

    def signature(self) -> tuple[Any, ...]:
        return (
            self.mode,
            self.reachability.constraint_contract_sha256,
            self.phase,
            self.seek_suffix,
            None if self.value is None else self.value.signature(),
        )

    @staticmethod
    def _next_seek_suffix(previous: str, character: str) -> tuple[str, bool]:
        candidate = (str(previous) + str(character).lower())[-len(FORMULA_LABEL) :]
        if candidate.endswith(FORMULA_LABEL):
            return "", True
        suffix = ""
        for width in range(1, min(len(candidate), len(FORMULA_LABEL) - 1) + 1):
            probe = candidate[-width:]
            if FORMULA_LABEL.startswith(probe):
                suffix = probe
        return suffix, False

    def feed(
        self,
        fragment: str,
        *,
        speculative: bool = False,
    ) -> "PlanFormulaCursor":
        cursor = self
        for character in str(fragment):
            if cursor.phase == "after_formula":
                continue
            if cursor.phase == "seek_formula_label":
                suffix, completed = self._next_seek_suffix(
                    cursor.seek_suffix,
                    character,
                )
                if completed:
                    cursor = dataclasses.replace(
                        cursor,
                        phase="formula_value",
                        seek_suffix="",
                        value=FormulaValueCursor(
                            max_atoms=cursor.reachability.max_atoms
                        ),
                    )
                else:
                    cursor = dataclasses.replace(cursor, seek_suffix=suffix)
                continue
            assert cursor.value is not None
            value = cursor.value.feed_character(
                character,
                mode=cursor.mode,
                reachability=cursor.reachability,
                speculative=speculative,
            )
            cursor = dataclasses.replace(cursor, value=value)
            if value.done:
                cursor = dataclasses.replace(cursor, phase="after_formula")
        return cursor

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        mode: str,
        reachability: OxidationReachability,
    ) -> "PlanFormulaCursor":
        return cls(mode=mode, reachability=reachability).feed(str(text))


@dataclasses.dataclass(frozen=True, slots=True)
class TokenSupport:
    token_ids: tuple[int, ...]
    rejection_counts: tuple[tuple[str, int], ...]
    terminal_token_ids: tuple[int, ...]

    def rejection_dict(self) -> dict[str, int]:
        return dict(self.rejection_counts)


@dataclasses.dataclass(frozen=True, slots=True)
class TokenSupportBundle:
    grammar_only: TokenSupport
    terminal_only: TokenSupport
    full_prefix: TokenSupport

    def for_mode(self, mode: str) -> TokenSupport:
        if mode not in CRPLAN_MODES:
            raise ValueError(f"unknown CR-Plan mode {mode!r}")
        return getattr(self, str(mode))


class CRPlanTokenVocabulary:
    """Frozen decoded token fragments and memoized cursor support.

    Formula-relevant tokenizer fragments are evaluated through a character
    trie.  Tokens that share a lexical prefix therefore share the exact same
    immutable FSM/DP transition instead of recomputing it independently.
    ``support_scalar_reference`` retains the original token-by-token
    implementation as a parity oracle for focused tests and release audits.
    """

    def __init__(
        self,
        fragments: Sequence[str],
        *,
        eos_token_id: int | None,
    ) -> None:
        self.fragments = tuple(str(value) for value in fragments)
        self.eos_token_id = (
            None if eos_token_id is None else int(eos_token_id)
        )
        self.vocab_size = len(self.fragments)
        digest = hashlib.sha256()
        for token_id, fragment in enumerate(self.fragments):
            digest.update(str(token_id).encode("ascii"))
            digest.update(b"\0")
            digest.update(fragment.encode("utf-8", errors="surrogatepass"))
            digest.update(b"\0")
        self.fragment_sha256 = digest.hexdigest()
        self._support_cache: dict[tuple[Any, ...], TokenSupport] = {}
        self._support_bundle_cache: dict[
            tuple[Any, ...], TokenSupportBundle
        ] = {}
        by_first: dict[str, list[int]] = defaultdict(list)
        for token_id, fragment in enumerate(self.fragments):
            if fragment:
                by_first[fragment[0]].append(token_id)
        self._ids_by_first = {
            character: tuple(values)
            for character, values in by_first.items()
        }
        trie_children: list[dict[str, int]] = [{}]
        trie_token_ids: list[list[int]] = [[]]
        for token_id, fragment in enumerate(self.fragments):
            node = 0
            for character in fragment:
                child = trie_children[node].get(character)
                if child is None:
                    child = len(trie_children)
                    trie_children[node][character] = child
                    trie_children.append({})
                    trie_token_ids.append([])
                node = child
            trie_token_ids[node].append(int(token_id))
        subtree_token_counts = [0 for _ in trie_children]
        subtree_eos_counts = [0 for _ in trie_children]
        for node in range(len(trie_children) - 1, -1, -1):
            subtree_token_counts[node] = len(trie_token_ids[node]) + sum(
                subtree_token_counts[child]
                for child in trie_children[node].values()
            )
            subtree_eos_counts[node] = sum(
                int(token_id == self.eos_token_id)
                for token_id in trie_token_ids[node]
            ) + sum(
                subtree_eos_counts[child]
                for child in trie_children[node].values()
            )
        self._trie_children = tuple(
            dict(values) for values in trie_children
        )
        self._trie_token_ids = tuple(
            tuple(values) for values in trie_token_ids
        )
        self._trie_subtree_token_counts = tuple(subtree_token_counts)
        self._trie_subtree_eos_counts = tuple(subtree_eos_counts)
        dfs_token_ids: list[int] = []
        subtree_starts = [0 for _ in trie_children]
        subtree_ends = [0 for _ in trie_children]

        def append_subtree(node: int) -> None:
            subtree_starts[node] = len(dfs_token_ids)
            dfs_token_ids.extend(trie_token_ids[node])
            for character in sorted(trie_children[node]):
                append_subtree(trie_children[node][character])
            subtree_ends[node] = len(dfs_token_ids)

        append_subtree(0)
        self._trie_dfs_token_ids = tuple(dfs_token_ids)
        self._trie_subtree_starts = tuple(subtree_starts)
        self._trie_subtree_ends = tuple(subtree_ends)

    def _formula_candidate_ids(
        self,
        cursor: PlanFormulaCursor,
    ) -> tuple[int, ...]:
        assert cursor.phase == "formula_value"
        assert cursor.value is not None
        value = cursor.value
        first_characters: set[str] = set()
        if not value.seen_element and not value.pending_symbol_prefix:
            first_characters.update((" ", "\t"))
            first_characters.update(
                prefix
                for prefix in FORMULA_SYMBOL_PREFIXES
                if len(prefix) == 1
            )
        else:
            first_characters.update(("\n", "\r"))
            if value.committed_total < value.max_atoms:
                first_characters.update(
                    prefix
                    for prefix in FORMULA_SYMBOL_PREFIXES
                    if len(prefix) == 1
                )
            if (
                value.pending_symbol_prefix
                and len(value.pending_symbol_prefix) == 1
                and not value.count_digits
            ):
                first_characters.update(
                    symbol[1]
                    for symbol in value.possible_symbols()
                    if len(symbol) == 2
                )
            if value.pending_symbol_prefix:
                first_characters.update("0123456789")
        return tuple(
            sorted(
                {
                    token_id
                    for character in first_characters
                    for token_id in self._ids_by_first.get(character, ())
                }
            )
        )

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "CRPlanTokenVocabulary":
        fragments = tuple(
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for token_id in range(len(tokenizer))
        )
        return cls(fragments, eos_token_id=tokenizer.eos_token_id)

    def _support_scalar_uncached(
        self,
        cursor: PlanFormulaCursor,
    ) -> TokenSupport:
        if cursor.phase == "after_formula":
            return TokenSupport(
                token_ids=tuple(range(self.vocab_size)),
                rejection_counts=(),
                terminal_token_ids=(),
            )
        candidate_ids = (
            tuple(range(self.vocab_size))
            if cursor.phase == "seek_formula_label"
            else self._formula_candidate_ids(cursor)
        )
        allowed: list[int] = []
        terminal: list[int] = []
        reasons: Counter[str] = Counter(
            {"grammar_block": self.vocab_size - len(candidate_ids)}
        )
        for token_id in candidate_ids:
            fragment = self.fragments[token_id]
            if (
                cursor.phase == "formula_value"
                and (token_id == self.eos_token_id or not fragment)
            ):
                reasons["empty_or_eos_inside_plan"] += 1
                continue
            try:
                updated = cursor.feed(fragment)
            except TerminalChargeError:
                reasons["terminal_charge_block"] += 1
                continue
            except FormulaGrammarError as exc:
                message = str(exc)
                if "no neutral completion" in message:
                    reasons["prefix_reachability_block"] += 1
                else:
                    reasons["grammar_block"] += 1
                continue
            allowed.append(token_id)
            if (
                cursor.phase == "formula_value"
                and updated.phase == "after_formula"
            ):
                terminal.append(token_id)
        return TokenSupport(
            token_ids=tuple(allowed),
            rejection_counts=tuple(sorted(reasons.items())),
            terminal_token_ids=tuple(terminal),
        )

    def support_scalar_reference(
        self,
        cursor: PlanFormulaCursor,
    ) -> TokenSupport:
        """Return the original scalar support for parity auditing only."""

        return self._support_scalar_uncached(cursor)

    def _support_trie_uncached(
        self,
        cursor: PlanFormulaCursor,
    ) -> TokenSupport:
        if cursor.phase == "after_formula":
            return TokenSupport(
                token_ids=tuple(range(self.vocab_size)),
                rejection_counts=(),
                terminal_token_ids=(),
            )
        formula_phase = cursor.phase == "formula_value"
        allowed: list[int] = []
        terminal: list[int] = []
        reasons: Counter[str] = Counter({"grammar_block": 0})
        if formula_phase:
            candidate_ids = self._formula_candidate_ids(cursor)
            reasons["grammar_block"] = self.vocab_size - len(candidate_ids)
            candidate_first_characters = {
                self.fragments[token_id][0]
                for token_id in candidate_ids
                if self.fragments[token_id]
            }
            root_characters = set(self._trie_children[0]).intersection(
                candidate_first_characters
            )
            root_token_ids: tuple[int, ...] = ()
        else:
            root_characters = set(self._trie_children[0])
            root_token_ids = self._trie_token_ids[0]

        for token_id in root_token_ids:
            allowed.append(int(token_id))

        def reject_subtree(node: int, reason: str) -> None:
            eos_count = (
                self._trie_subtree_eos_counts[node]
                if formula_phase
                else 0
            )
            if eos_count:
                reasons["empty_or_eos_inside_plan"] += eos_count
            regular = self._trie_subtree_token_counts[node] - eos_count
            if regular:
                reasons[reason] += regular

        def visit(node: int, state: PlanFormulaCursor) -> None:
            for token_id in self._trie_token_ids[node]:
                if formula_phase and token_id == self.eos_token_id:
                    reasons["empty_or_eos_inside_plan"] += 1
                    continue
                allowed.append(int(token_id))
                if formula_phase and state.phase == "after_formula":
                    terminal.append(int(token_id))
            for character, child in self._trie_children[node].items():
                try:
                    updated = state.feed(character)
                except TerminalChargeError:
                    reject_subtree(child, "terminal_charge_block")
                    continue
                except FormulaGrammarError as exc:
                    reason = (
                        "prefix_reachability_block"
                        if "no neutral completion" in str(exc)
                        else "grammar_block"
                    )
                    reject_subtree(child, reason)
                    continue
                visit(child, updated)

        for character in sorted(root_characters):
            child = self._trie_children[0][character]
            try:
                updated = cursor.feed(character)
            except TerminalChargeError:
                reject_subtree(child, "terminal_charge_block")
                continue
            except FormulaGrammarError as exc:
                reason = (
                    "prefix_reachability_block"
                    if "no neutral completion" in str(exc)
                    else "grammar_block"
                )
                reject_subtree(child, reason)
                continue
            visit(child, updated)

        return TokenSupport(
            token_ids=tuple(sorted(allowed)),
            rejection_counts=tuple(
                sorted(
                    (key, int(value))
                    for key, value in reasons.items()
                )
            ),
            terminal_token_ids=tuple(sorted(terminal)),
        )

    @staticmethod
    def _bundle_signature(
        cursor: PlanFormulaCursor,
    ) -> tuple[Any, ...]:
        return (
            cursor.reachability.constraint_contract_sha256,
            cursor.phase,
            cursor.seek_suffix,
            None if cursor.value is None else cursor.value.signature(),
        )

    def _subtree_token_ids(self, node: int) -> tuple[int, ...]:
        return self._trie_dfs_token_ids[
            self._trie_subtree_starts[node] :
            self._trie_subtree_ends[node]
        ]

    def _support_bundle_uncached(
        self,
        cursor: PlanFormulaCursor,
    ) -> TokenSupportBundle:
        if cursor.phase == "after_formula":
            support = TokenSupport(
                token_ids=tuple(range(self.vocab_size)),
                rejection_counts=(),
                terminal_token_ids=(),
            )
            return TokenSupportBundle(
                grammar_only=support,
                terminal_only=support,
                full_prefix=support,
            )

        formula_phase = cursor.phase == "formula_value"
        allowed: dict[str, list[int]] = {
            mode: [] for mode in CRPLAN_MODES
        }
        terminal: dict[str, list[int]] = {
            mode: [] for mode in CRPLAN_MODES
        }
        reasons: dict[str, Counter[str]] = {
            mode: Counter({"grammar_block": 0})
            for mode in CRPLAN_MODES
        }
        states = {
            mode: dataclasses.replace(cursor, mode=mode)
            for mode in CRPLAN_MODES
        }
        if formula_phase:
            candidate_ids = self._formula_candidate_ids(cursor)
            grammar_block = self.vocab_size - len(candidate_ids)
            for mode in CRPLAN_MODES:
                reasons[mode]["grammar_block"] = grammar_block
            candidate_first_characters = {
                self.fragments[token_id][0]
                for token_id in candidate_ids
                if self.fragments[token_id]
            }
            root_characters = set(self._trie_children[0]).intersection(
                candidate_first_characters
            )
            root_token_ids: tuple[int, ...] = ()
        else:
            root_characters = set(self._trie_children[0])
            root_token_ids = self._trie_token_ids[0]

        for token_id in root_token_ids:
            for mode in CRPLAN_MODES:
                allowed[mode].append(int(token_id))

        def reject_subtree(mode: str, node: int, reason: str) -> None:
            eos_count = (
                self._trie_subtree_eos_counts[node]
                if formula_phase
                else 0
            )
            if eos_count:
                reasons[mode]["empty_or_eos_inside_plan"] += eos_count
            regular = self._trie_subtree_token_counts[node] - eos_count
            if regular:
                reasons[mode][reason] += regular

        def accept_after_formula_subtree(mode: str, node: int) -> None:
            for token_id in self._subtree_token_ids(node):
                if formula_phase and token_id == self.eos_token_id:
                    reasons[mode]["empty_or_eos_inside_plan"] += 1
                    continue
                allowed[mode].append(int(token_id))
                if formula_phase:
                    terminal[mode].append(int(token_id))

        def advance(
            child: int,
            character: str,
            active: Mapping[str, PlanFormulaCursor],
        ) -> None:
            updated_states: dict[str, PlanFormulaCursor] = {}
            for mode, state in active.items():
                try:
                    updated = state.feed(
                        character,
                        speculative=True,
                    )
                except TerminalChargeError:
                    reject_subtree(
                        mode,
                        child,
                        "terminal_charge_block",
                    )
                    continue
                except FormulaGrammarError as exc:
                    reason = (
                        "prefix_reachability_block"
                        if "no neutral completion" in str(exc)
                        else "grammar_block"
                    )
                    reject_subtree(mode, child, reason)
                    continue
                if updated.phase == "after_formula":
                    accept_after_formula_subtree(mode, child)
                    continue
                updated_states[mode] = updated
            if updated_states:
                visit(child, updated_states)

        def visit(
            node: int,
            active: Mapping[str, PlanFormulaCursor],
        ) -> None:
            for token_id in self._trie_token_ids[node]:
                for mode in active:
                    if formula_phase and token_id == self.eos_token_id:
                        reasons[mode]["empty_or_eos_inside_plan"] += 1
                        continue
                    allowed[mode].append(int(token_id))
            for character, child in self._trie_children[node].items():
                advance(child, character, active)

        for character in sorted(root_characters):
            child = self._trie_children[0][character]
            advance(child, character, states)

        supports = {
            mode: TokenSupport(
                token_ids=tuple(sorted(allowed[mode])),
                rejection_counts=tuple(
                    sorted(
                        (key, int(value))
                        for key, value in reasons[mode].items()
                    )
                ),
                terminal_token_ids=tuple(sorted(terminal[mode])),
            )
            for mode in CRPLAN_MODES
        }
        return TokenSupportBundle(
            grammar_only=supports["grammar_only"],
            terminal_only=supports["terminal_only"],
            full_prefix=supports["full_prefix"],
        )

    def support_bundle(
        self,
        cursor: PlanFormulaCursor,
    ) -> TokenSupportBundle:
        signature = self._bundle_signature(cursor)
        cached = self._support_bundle_cache.get(signature)
        if cached is not None:
            return cached
        bundle = self._support_bundle_uncached(cursor)
        self._support_bundle_cache[signature] = bundle
        return bundle

    def support(self, cursor: PlanFormulaCursor) -> TokenSupport:
        signature = cursor.signature()
        cached = self._support_cache.get(signature)
        if cached is not None:
            return cached
        support = self.support_bundle(cursor).for_mode(cursor.mode)
        self._support_cache[signature] = support
        return support


class CRPlanLogitsProcessor:
    """Transformers-compatible fail-closed logit mask with telemetry."""

    def __init__(
        self,
        tokenizer: Any,
        token_vocabulary: CRPlanTokenVocabulary,
        reachability: OxidationReachability,
        *,
        prompt_width: int,
        mode: str,
        attempt_ordinal: int,
    ) -> None:
        if mode not in CRPLAN_MODES:
            raise ValueError(f"unknown CR-Plan mode {mode!r}")
        self.tokenizer = tokenizer
        self.token_vocabulary = token_vocabulary
        self.reachability = reachability
        self.prompt_width = int(prompt_width)
        self.mode = str(mode)
        self.attempt_ordinal = int(attempt_ordinal)
        self.steps: list[dict[str, Any]] = []
        self._dp_start = self.reachability.diagnostics.snapshot()
        self._cache_start = self.reachability.cache_report()
        self._cursor_by_generated: dict[tuple[int, ...], PlanFormulaCursor] = {
            (): PlanFormulaCursor(mode=self.mode, reachability=self.reachability)
        }
        self.dead_end: dict[str, Any] | None = None

    def _cursor(self, generated: tuple[int, ...]) -> PlanFormulaCursor:
        cached = self._cursor_by_generated.get(generated)
        if cached is not None:
            return cached
        parent = self._cursor(generated[:-1])
        token_id = int(generated[-1])
        if not 0 <= token_id < self.token_vocabulary.vocab_size:
            raise FormulaGrammarError(
                f"generated token id {token_id} outside tokenizer vocabulary"
            )
        cursor = parent.feed(self.token_vocabulary.fragments[token_id])
        self._cursor_by_generated[generated] = cursor
        return cursor

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        if int(input_ids.shape[0]) != 1 or int(scores.shape[0]) != 1:
            raise ValueError("CR-Plan requires stateless batch-size-one generation")
        generated = tuple(
            int(value)
            for value in input_ids[0, self.prompt_width :].tolist()
        )
        cursor = self._cursor(generated)
        if cursor.phase == "after_formula":
            return scores
        support_bundle = self.token_vocabulary.support_bundle(cursor)
        support = support_bundle.for_mode(self.mode)
        allowed = tuple(
            token_id
            for token_id in support.token_ids
            if token_id < int(scores.shape[-1])
        )
        if not allowed:
            self.dead_end = {
                "attempt_ordinal": self.attempt_ordinal,
                "generated_token_count": len(generated),
                "cursor_signature": repr(cursor.signature()),
                "rejection_counts": support.rejection_dict(),
            }
            raise CRPlanDeadEndError(
                "CR-Plan legal support is empty; attempt failed closed"
            )
        original = torch.softmax(scores[0].float(), dim=-1)
        allowed_tensor = torch.tensor(
            allowed,
            dtype=torch.long,
            device=scores.device,
        )
        masked = torch.full_like(scores, -torch.inf)
        masked[0, allowed_tensor] = scores[0, allowed_tensor]
        post = torch.softmax(masked[0].float(), dim=-1)
        finite = post > 0
        entropy_tensor = (
            -(post[finite] * torch.log(post[finite]))
        ).sum()
        grammar_support = support_bundle.grammar_only
        grammar_allowed = tuple(
            token_id
            for token_id in grammar_support.token_ids
            if token_id < int(scores.shape[-1])
        )
        grammar_tensor = torch.tensor(
            grammar_allowed,
            dtype=torch.long,
            device=scores.device,
        )
        reference_mode = (
            "terminal_only" if self.mode == "full_prefix" else "grammar_only"
        )
        reference_support = support_bundle.for_mode(reference_mode)
        reference_allowed = tuple(
            token_id
            for token_id in reference_support.token_ids
            if token_id < int(scores.shape[-1])
        )
        reference_tensor = torch.tensor(
            reference_allowed,
            dtype=torch.long,
            device=scores.device,
        )
        (
            retained_mass,
            grammar_mass,
            reference_mass,
            entropy,
        ) = (
            torch.stack(
                (
                    original.index_select(0, allowed_tensor).sum(),
                    original.index_select(0, grammar_tensor).sum(),
                    original.index_select(0, reference_tensor).sum(),
                    entropy_tensor,
                )
            )
            .detach()
            .cpu()
            .tolist()
        )
        self.steps.append(
            {
                "step": len(generated),
                "phase": cursor.phase,
                "formula_signature": (
                    None if cursor.value is None else repr(cursor.value.signature())
                ),
                "vocab_size": int(scores.shape[-1]),
                "grammar_support_size": len(grammar_allowed),
                "prefix_reference_mode": reference_mode,
                "prefix_reference_support_size": len(reference_allowed),
                "legal_support_size": len(allowed),
                "removed_probability_mass": max(0.0, 1.0 - retained_mass),
                "reachability_removed_probability_mass": max(
                    0.0,
                    grammar_mass - retained_mass,
                ),
                "prefix_only_removed_probability_mass": max(
                    0.0,
                    reference_mass - retained_mass,
                ),
                "mask_entropy": entropy,
                "terminal_token_support_size": len(
                    [
                        value
                        for value in support.terminal_token_ids
                        if value < int(scores.shape[-1])
                    ]
                ),
                "preterminal_support_difference": bool(
                    self.mode == "full_prefix"
                    and
                    cursor.phase == "formula_value"
                    and not (cursor.value and cursor.value.done)
                    and allowed != reference_allowed
                ),
                "rejection_counts": support.rejection_dict(),
            }
        )
        return masked

    def diagnostics(self, generated_ids: Sequence[int] | None = None) -> dict[str, Any]:
        cursor: PlanFormulaCursor | None = None
        cursor_error: str | None = None
        if generated_ids is not None:
            try:
                cursor = self._cursor(tuple(int(value) for value in generated_ids))
            except CRPlanError as exc:
                cursor_error = f"{type(exc).__name__}: {exc}"
        dp_end = self.reachability.diagnostics.snapshot()
        cache_end = self.reachability.cache_report()
        return {
            "schema": "h1_crplan_attempt_diagnostics_v1",
            "mode": self.mode,
            "attempt_ordinal": self.attempt_ordinal,
            "steps": list(self.steps),
            "masked_step_count": len(self.steps),
            "legal_support_enforcement": "mask_or_raise",
            "mask_application_count": len(self.steps),
            "empty_support_error_raised": self.dead_end is not None,
            "preterminal_support_difference_steps": sum(
                int(value["preterminal_support_difference"])
                for value in self.steps
            ),
            "blocked_newline_token_count": sum(
                int(value["rejection_counts"].get("terminal_charge_block", 0))
                for value in self.steps
            ),
            "dead_end": self.dead_end,
            "final_cursor_phase": None if cursor is None else cursor.phase,
            "final_cursor_error": cursor_error,
            "terminal_certificate": (
                None
                if cursor is None
                or cursor.value is None
                or cursor.value.certificate is None
                else cursor.value.certificate.to_dict()
            ),
            "dp": {
                "start": dict(self._dp_start),
                "end": dp_end,
                "attempt_delta": {
                    key: int(dp_end[key]) - int(self._dp_start[key])
                    for key in self._dp_start
                },
                "cache_start": dict(self._cache_start),
                "cache_end": cache_end,
                "attempt_peak_cache_entries": sum(cache_end.values()),
            },
            "silent_fallback_used_by_decoder": False,
            "retry_replacement_repair_filter_or_rerank_used": False,
        }


def validate_crplan_parsed_identity(
    *,
    raw_model_text: str,
    prompt_style: str,
    parsed_symbols: Sequence[str],
    parsed_counts: Sequence[int],
    diagnostics: Mapping[str, Any] | None,
    mode: str,
) -> dict[str, Any]:
    """Bind the masked FSM certificate to the exact formula used by parser.

    This closes three otherwise silent gaps: a spaced label that the final
    parser accepts but the FSM never sees, a later duplicate formula line that
    overwrites the first parser field, and prompt-prefill modes whose formula
    label is outside the generated continuation.
    """

    if mode not in CRPLAN_MODES:
        raise CRPlanIdentityError(f"identity validation requires CR-Plan mode, got {mode!r}")
    if str(prompt_style) == "formula_prefill_v1":
        raise CRPlanIdentityError(
            "formula-prefill cannot prove generated-continuation FSM identity"
        )
    normalized = str(raw_model_text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in normalized.strip().splitlines() if line.strip()]
    expected_fields = (
        (
            "formula",
            "anion",
            "charge",
            "lattice",
            "spacegroup",
            "volume",
            "end",
        )
        if str(prompt_style) == "h1_rich_plan_v1"
        else (
            "formula",
            "anion",
            "lattice",
            "spacegroup",
            "volume",
            "end",
        )
        if str(prompt_style) == "h1_rich_nocharge_plan_v1"
        else ("formula", "end")
    )
    observed_fields: list[str] = []
    for line in lines:
        match = re.match(r"(?i)^([a-z]+)\s*:", line)
        if match is None:
            raise CRPlanIdentityError(
                f"non-field line in constrained plan: {line!r}"
            )
        observed_fields.append(match.group(1).lower())
    if tuple(observed_fields) != tuple(expected_fields):
        raise CRPlanIdentityError(
            "constrained plan fields are not the exact unique frozen order: "
            f"{observed_fields}"
        )
    if not re.match(r"(?i)^formula:", lines[0]):
        raise CRPlanIdentityError(
            "formula label must be contiguous with colon for FSM identity"
        )
    if not isinstance(diagnostics, Mapping):
        raise CRPlanIdentityError("missing CR-Plan diagnostics")
    if diagnostics.get("dead_end") is not None:
        raise CRPlanIdentityError("dead-end attempt cannot have parsed identity")
    if diagnostics.get("final_cursor_phase") != "after_formula":
        raise CRPlanIdentityError(
            f"final cursor did not certify formula: {diagnostics.get('final_cursor_phase')!r}"
        )
    if diagnostics.get("final_cursor_error") is not None:
        raise CRPlanIdentityError(
            f"final cursor error: {diagnostics.get('final_cursor_error')}"
        )
    certificate = diagnostics.get("terminal_certificate")
    if not isinstance(certificate, Mapping):
        raise CRPlanIdentityError("final FSM certificate is missing")
    parsed = _canonical_counts(zip(parsed_symbols, parsed_counts))
    certified = _canonical_counts(
        (
            (str(value["element"]), int(value["count"]))
            for value in certificate.get("counts", ())
        )
    )
    if parsed != certified:
        raise CRPlanIdentityError(
            f"FSM/parser composition mismatch: certified={certified}, parsed={parsed}"
        )
    if mode in ("terminal_only", "full_prefix") and (
        certificate.get("terminal_allowed") is not True
    ):
        raise CRPlanIdentityError(
            "terminal/full-prefix parsed a formula without an allowed certificate"
        )
    raw_formula = lines[0].split(":", 1)[1].strip()
    if re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", raw_formula) is None:
        raise CRPlanIdentityError(
            f"raw formula is not the exact frozen flat grammar: {raw_formula!r}"
        )
    lexical_tokens = re.findall(r"([A-Z][a-z]?)(\d*)", raw_formula)
    raw_counts = _canonical_counts(
        (
            (symbol, int(digits) if digits else 1)
            for symbol, digits in lexical_tokens
        )
    )
    if raw_counts != parsed:
        raise CRPlanIdentityError(
            f"raw formula/parser composition mismatch: raw={raw_counts}, parsed={parsed}"
        )
    lexical_elements = [symbol for symbol, _ in lexical_tokens]
    repeated = sorted(
        symbol
        for symbol, count in Counter(lexical_elements).items()
        if count > 1
    )
    return {
        "schema": "h1_crplan_parser_fsm_identity_v1",
        "verified": True,
        "observed_fields": observed_fields,
        "formula_line_count": observed_fields.count("formula"),
        "fsm_counts_equal_parser_counts": True,
        "raw_formula": raw_formula,
        "raw_repeated_elements": repeated,
        "repeated_elements_allowed_then_canonicalized": True,
    }


def load_frozen_smact_table(
    *,
    max_atoms: int = 20,
    missing_state_policy: str = "allow_non_applicable",
) -> OxidationReachability:
    """Load the exact SMACT table used by the frozen Direct evaluator."""

    from importlib import metadata
    import smact

    states: MutableMapping[str, tuple[int, ...]] = {}
    for symbol in FORMULA_SYMBOLS:
        try:
            element = smact.element_dictionary((symbol,))[symbol]
        except (KeyError, ValueError):
            states[symbol] = ()
            continue
        states[symbol] = tuple(
            sorted(set(int(value) for value in (element.oxidation_states or ())))
        )
    return OxidationReachability(
        states,
        metals=tuple(str(value) for value in smact.metals),
        max_atoms=max_atoms,
        table_source="smact.Element.oxidation_states",
        table_version=metadata.version("SMACT"),
        missing_state_policy=missing_state_policy,
    )


def certificate_for_symbol_counts(
    reachability: OxidationReachability,
    symbols: Sequence[str],
    counts: Sequence[int],
) -> dict[str, Any]:
    if len(symbols) != len(counts):
        raise ValueError("symbols and counts must be aligned")
    return reachability.terminal_certificate(
        zip(
            (str(value) for value in symbols),
            (int(value) for value in counts),
        )
    ).to_dict()


__all__ = [
    "CRPLAN_MODES",
    "CRPLAN_SCHEMA",
    "CRPlanDeadEndError",
    "CRPlanError",
    "CRPlanIdentityError",
    "CRPlanLogitsProcessor",
    "CRPlanTokenVocabulary",
    "FORMULA_SYMBOLS",
    "FormulaGrammarError",
    "MISSING_STATE_POLICIES",
    "FormulaValueCursor",
    "OxidationReachability",
    "PlanFormulaCursor",
    "TerminalCertificate",
    "TerminalChargeError",
    "TokenSupport",
    "certificate_for_symbol_counts",
    "load_frozen_smact_table",
    "validate_crplan_parsed_identity",
]
