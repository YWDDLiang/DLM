"""Semantic composition compiler for conservation-constrained decoding v2.

CCFD-v1 wrapped the frozen text tokenizer and only checked whether a completed
formula was representable by an expanded oxidation-state catalogue.  That
improved the internal assignment rate, but did not guarantee the independent
SMACT/CrysLLMGen composition-valid endpoint.  V2 makes the generated actions
semantic:

1. emit and lock the atom count ``N``;
2. emit canonical ``(element, oxidation state, count)`` actions;
3. maintain exact atom and charge ledgers plus continuation reachability; and
4. end with one of two explicit certificates:
   ``benchmark_compatible`` or ``extended_only``.

``extended_only`` is deliberately an unknown outcome.  It is never promoted
to a benchmark-valid training target or headline success.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, reduce
from math import gcd
from typing import Any, Callable, Iterable, Mapping, Sequence

from crystal_dlm.ccfd import FormulaToken
from crystal_dlm.composition_validity import formula_from_composition
from crystal_dlm.fixed_slot import SYMBOL_TO_Z, Z_TO_SYMBOL


CERTIFICATE_BENCHMARK = "benchmark_compatible"
CERTIFICATE_EXTENDED_ONLY = "extended_only"
CERTIFICATE_BENCHMARK_UNKNOWN = "benchmark_unknown"
CERTIFICATE_INCOMPLETE = "incomplete"
CERTIFICATE_INVALID = "invalid"

CERTIFICATE_CLASSES = (
    CERTIFICATE_BENCHMARK,
    CERTIFICATE_EXTENDED_ONLY,
    CERTIFICATE_BENCHMARK_UNKNOWN,
    CERTIFICATE_INCOMPLETE,
    CERTIFICATE_INVALID,
)

BenchmarkValidator = Callable[[Sequence[int], Sequence[int]], Mapping[str, Any] | bool]
ExtendedAssigner = Callable[[Sequence[str], Sequence[int], int], Sequence[FormulaToken] | None]


@dataclass(frozen=True, order=True)
class SetAtomCount:
    """First semantic action.  The chosen atom count cannot later change."""

    total_atoms: int

    def __post_init__(self) -> None:
        if int(self.total_atoms) <= 0:
            raise ValueError("atom count must be positive")


@dataclass(frozen=True, order=True)
class EndComposition:
    """Explicit semantic EOS action."""


END_COMPOSITION = EndComposition()
SemanticAction = SetAtomCount | FormulaToken | EndComposition


@dataclass(frozen=True)
class CompositionCertificate:
    certificate_class: str
    reason: str
    benchmark_valid: bool | None
    benchmark_reason: str
    extended_valid: bool
    formula: str | None
    elements: tuple[int, ...]
    counts: tuple[int, ...]
    reduced_counts: tuple[int, ...]
    target_atoms: int | None
    emitted_atoms: int
    net_charge: int
    branch: str | None

    def __post_init__(self) -> None:
        if self.certificate_class not in CERTIFICATE_CLASSES:
            raise ValueError(f"unknown certificate class {self.certificate_class!r}")

    @property
    def benchmark_compatible(self) -> bool:
        return self.certificate_class == CERTIFICATE_BENCHMARK

    @property
    def usable_as_positive(self) -> bool:
        """Only the independent benchmark certificate is a positive label."""

        return self.benchmark_compatible

    def to_dict(self) -> dict[str, Any]:
        return {
            "certificate_class": self.certificate_class,
            "reason": self.reason,
            "benchmark_valid": self.benchmark_valid,
            "benchmark_reason": self.benchmark_reason,
            "extended_valid": self.extended_valid,
            "formula": self.formula,
            "elements": list(self.elements),
            "counts": list(self.counts),
            "reduced_counts": list(self.reduced_counts),
            "target_atoms": self.target_atoms,
            "emitted_atoms": int(self.emitted_atoms),
            "net_charge": int(self.net_charge),
            "branch": self.branch,
            "usable_as_positive": self.usable_as_positive,
        }


def _canonical_composition(tokens: Sequence[FormulaToken]) -> tuple[tuple[int, int], ...]:
    merged: dict[int, int] = {}
    for token in tokens:
        atomic_number = int(token.atomic_number)
        merged[atomic_number] = merged.get(atomic_number, 0) + int(token.count)
    return tuple(sorted((atomic_number, count) for atomic_number, count in merged.items() if count > 0))


def _reduced_counts(counts: Sequence[int]) -> tuple[int, ...]:
    divisor = reduce(gcd, (int(value) for value in counts)) if counts else 1
    divisor = max(1, int(divisor))
    return tuple(int(value) // divisor for value in counts)


def _normalize_validator_result(result: Mapping[str, Any] | bool) -> tuple[bool | None, str]:
    if isinstance(result, bool):
        return bool(result), "validator_boolean"
    valid = result.get("valid")
    if valid is not True and valid is not False:
        return None, str(result.get("reason") or "validator_unknown")
    return bool(valid), str(result.get("reason") or "validator_result")


def default_benchmark_validator(
    elements: Sequence[int], counts: Sequence[int]
) -> Mapping[str, Any]:
    """Lazy independent validator matching the frozen CrysLLMGen endpoint."""

    from crystal_dlm.composition_validity import classify_smact_validity

    return classify_smact_validity(elements, counts)


def default_extended_assigner(
    symbols: Sequence[str], counts: Sequence[int], max_species: int
) -> Sequence[FormulaToken] | None:
    """Find a broader exact SMACT-state witness for diagnostic unknowns.

    Unlike the formal benchmark certificate, this permits adjacent same-sign
    mixed valence within one element.  It uses the installed SMACT oxidation
    states rather than any external model-specific catalogue, and it does not
    convert a failed Pauling test into a positive label.
    """

    import smact

    canonical = [(str(symbol), int(count)) for symbol, count in zip(symbols, counts)]
    if not canonical or any(count <= 0 for _symbol, count in canonical):
        return None
    if len(canonical) == 1:
        symbol, count = canonical[0]
        return (FormulaToken.from_symbol(symbol, 0, count),)
    if all(symbol in smact.metals for symbol, _count in canonical):
        return tuple(
            FormulaToken.from_symbol(symbol, 0, count)
            for symbol, count in canonical
        )

    element_space = smact.element_dictionary([symbol for symbol, _count in canonical])
    states_by_symbol: dict[str, tuple[int, ...]] = {}
    for symbol, _count in canonical:
        element = element_space.get(symbol)
        states = tuple(
            sorted(
                {
                    int(value)
                    for value in (getattr(element, "oxidation_states", None) or ())
                    if int(value) != 0
                }
            )
        )
        if not states:
            return None
        states_by_symbol[symbol] = states

    frontier: dict[int, tuple[FormulaToken, ...]] = {0: ()}
    for symbol, count in canonical:
        states = states_by_symbol[symbol]
        options: list[tuple[FormulaToken, ...]] = [
            (FormulaToken.from_symbol(symbol, oxidation, count),)
            for oxidation in states
        ]
        if count > 1:
            for left, right in zip(states, states[1:]):
                if left * right < 0:
                    continue
                for left_count in range(1, count):
                    options.append(
                        tuple(
                            sorted(
                                (
                                    FormulaToken.from_symbol(symbol, left, left_count),
                                    FormulaToken.from_symbol(
                                        symbol, right, count - left_count
                                    ),
                                )
                            )
                        )
                    )
        next_frontier: dict[int, tuple[FormulaToken, ...]] = {}
        for partial_charge, path in frontier.items():
            for option in options:
                candidate = tuple(sorted((*path, *option)))
                if len(candidate) > int(max_species):
                    continue
                charge = int(partial_charge) + sum(token.charge_delta for token in option)
                incumbent = next_frontier.get(charge)
                if incumbent is None or (
                    len(candidate), candidate
                ) < (
                    len(incumbent), incumbent
                ):
                    next_frontier[charge] = candidate
        frontier = next_frontier
        if not frontier:
            return None
    return frontier.get(0)


@dataclass(frozen=True)
class CCFDv2State:
    """Immutable semantic state with a hard atom-count and charge ledger."""

    target_atoms: int | None = None
    remaining_atoms: int | None = None
    net_charge: int = 0
    branch: str | None = None
    tokens: tuple[FormulaToken, ...] = ()
    ended: bool = False

    @classmethod
    def start(cls) -> "CCFDv2State":
        return cls()

    @property
    def needs_atom_count(self) -> bool:
        return self.target_atoms is None

    @property
    def emitted_atoms(self) -> int:
        return sum(int(token.count) for token in self.tokens)

    @property
    def last_species_key(self) -> tuple[int, int] | None:
        return None if not self.tokens else self.tokens[-1].species_key

    @property
    def distinct_elements(self) -> tuple[int, ...]:
        return tuple(sorted({int(token.atomic_number) for token in self.tokens}))

    @property
    def conservation_complete(self) -> bool:
        if self.target_atoms is None or self.remaining_atoms is None or not self.tokens:
            return False
        if self.remaining_atoms != 0 or self.net_charge != 0:
            return False
        if self.branch == "alloy":
            return all(int(token.oxidation_state) == 0 for token in self.tokens)
        if self.branch == "ionic":
            return len(self.distinct_elements) >= 2 and all(
                int(token.oxidation_state) != 0 for token in self.tokens
            )
        return False

    @property
    def eos_legal(self) -> bool:
        return not self.ended and self.conservation_complete

    def set_atom_count(self, total_atoms: int, *, max_atoms: int = 20) -> "CCFDv2State":
        if self.ended:
            raise ValueError("composition already ended")
        if self.target_atoms is not None:
            raise ValueError("atom count is already locked")
        total = int(total_atoms)
        if total <= 0 or total > int(max_atoms):
            raise ValueError(f"atom count {total} outside 1..{int(max_atoms)}")
        return CCFDv2State(target_atoms=total, remaining_atoms=total)

    def append_species(
        self, token: FormulaToken, *, max_species: int = 7
    ) -> "CCFDv2State":
        if self.ended:
            raise ValueError("composition already ended")
        if self.target_atoms is None or self.remaining_atoms is None:
            raise ValueError("atom count must be emitted before species")
        if len(self.tokens) >= int(max_species):
            raise ValueError("maximum species count reached")
        if int(token.count) > int(self.remaining_atoms):
            raise ValueError("species count exceeds remaining atom budget")
        if self.last_species_key is not None and token.species_key <= self.last_species_key:
            raise ValueError("species actions must follow canonical increasing order")

        token_branch = "alloy" if int(token.oxidation_state) == 0 else "ionic"
        if self.branch is not None and self.branch != token_branch:
            raise ValueError("zero- and nonzero-valence species cannot share a branch")

        same_element = [
            existing
            for existing in self.tokens
            if int(existing.atomic_number) == int(token.atomic_number)
        ]
        if same_element and int(token.oxidation_state) != 0:
            signs = {
                1 if int(existing.oxidation_state) > 0 else -1
                for existing in same_element
                if int(existing.oxidation_state) != 0
            }
            signs.add(1 if int(token.oxidation_state) > 0 else -1)
            if len(signs) > 1:
                raise ValueError("one element cannot mix positive and negative valence")

        return CCFDv2State(
            target_atoms=int(self.target_atoms),
            remaining_atoms=int(self.remaining_atoms) - int(token.count),
            net_charge=int(self.net_charge) + int(token.charge_delta),
            branch=self.branch or token_branch,
            tokens=(*self.tokens, token),
        )

    def end(self) -> "CCFDv2State":
        if self.ended:
            raise ValueError("composition already ended")
        if not self.conservation_complete:
            raise ValueError("cannot end before exact atom and charge conservation")
        return CCFDv2State(
            target_atoms=self.target_atoms,
            remaining_atoms=self.remaining_atoms,
            net_charge=self.net_charge,
            branch=self.branch,
            tokens=self.tokens,
            ended=True,
        )

    def apply(
        self,
        action: SemanticAction,
        *,
        max_atoms: int = 20,
        max_species: int = 7,
    ) -> "CCFDv2State":
        if isinstance(action, SetAtomCount):
            return self.set_atom_count(action.total_atoms, max_atoms=max_atoms)
        if isinstance(action, FormulaToken):
            return self.append_species(action, max_species=max_species)
        if isinstance(action, EndComposition):
            return self.end()
        raise TypeError(f"unsupported semantic action {type(action).__name__}")

    def certificate(
        self,
        *,
        benchmark_validator: BenchmarkValidator = default_benchmark_validator,
    ) -> CompositionCertificate:
        composition = _canonical_composition(self.tokens)
        elements = tuple(int(value[0]) for value in composition)
        counts = tuple(int(value[1]) for value in composition)
        reduced = _reduced_counts(counts)
        formula = formula_from_composition(elements, counts) if elements else None
        common = {
            "formula": formula,
            "elements": elements,
            "counts": counts,
            "reduced_counts": reduced,
            "target_atoms": self.target_atoms,
            "emitted_atoms": self.emitted_atoms,
            "net_charge": int(self.net_charge),
            "branch": self.branch,
        }
        if not self.ended:
            return CompositionCertificate(
                certificate_class=CERTIFICATE_INCOMPLETE,
                reason="explicit_end_not_emitted",
                benchmark_valid=None,
                benchmark_reason="not_evaluated",
                extended_valid=False,
                **common,
            )
        if not self.conservation_complete:
            return CompositionCertificate(
                certificate_class=CERTIFICATE_INVALID,
                reason="conservation_failed",
                benchmark_valid=None,
                benchmark_reason="not_evaluated",
                extended_valid=False,
                **common,
            )
        try:
            benchmark_valid, benchmark_reason = _normalize_validator_result(
                benchmark_validator(elements, reduced)
            )
        except Exception as exc:  # noqa: BLE001 - unavailable validators are explicit unknowns.
            benchmark_valid = None
            benchmark_reason = f"validator_error:{type(exc).__name__}"
        if benchmark_valid is True:
            certificate_class = CERTIFICATE_BENCHMARK
            reason = "independent_benchmark_pass"
        elif benchmark_valid is False:
            certificate_class = CERTIFICATE_EXTENDED_ONLY
            reason = "exact_extended_ledger_but_independent_benchmark_failed"
        else:
            certificate_class = CERTIFICATE_BENCHMARK_UNKNOWN
            reason = "independent_benchmark_unavailable"
        return CompositionCertificate(
            certificate_class=certificate_class,
            reason=reason,
            benchmark_valid=benchmark_valid,
            benchmark_reason=benchmark_reason,
            extended_valid=True,
            **common,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_atoms": self.target_atoms,
            "remaining_atoms": self.remaining_atoms,
            "emitted_atoms": self.emitted_atoms,
            "net_charge": int(self.net_charge),
            "branch": self.branch,
            "ended": self.ended,
            "eos_legal": self.eos_legal,
            "tokens": [token.to_dict() for token in self.tokens],
        }


def replay_actions(
    actions: Sequence[SemanticAction],
    *,
    max_atoms: int = 20,
    max_species: int = 7,
) -> CCFDv2State:
    state = CCFDv2State.start()
    for action in actions:
        state = state.apply(action, max_atoms=max_atoms, max_species=max_species)
    return state


def _plan_symbol_counts(plan: Mapping[str, Any]) -> tuple[list[str], list[int], int]:
    raw_symbols = list(plan.get("elements") or ())
    raw_counts = list(plan.get("counts") or ())
    if len(raw_symbols) != len(raw_counts) or not raw_symbols:
        raise ValueError("plan must contain aligned non-empty elements and counts")
    merged: dict[str, int] = {}
    for raw_symbol, raw_count in zip(raw_symbols, raw_counts):
        symbol = str(raw_symbol)
        count = int(raw_count)
        if symbol not in SYMBOL_TO_Z:
            raise ValueError(f"unsupported element {symbol!r}")
        if count <= 0:
            raise ValueError(f"nonpositive count {count} for {symbol}")
        merged[symbol] = merged.get(symbol, 0) + count
    symbols = sorted(merged, key=lambda value: int(SYMBOL_TO_Z[value]))
    counts = [int(merged[symbol]) for symbol in symbols]
    derived_n = sum(counts)
    if "N" not in plan or plan.get("N") in (None, ""):
        raise ValueError("plan must explicitly provide N")
    source_n = int(plan["N"])
    if source_n != derived_n:
        raise ValueError(f"plan N={source_n} does not equal count sum {derived_n}")
    return symbols, counts, source_n


def compile_plan_actions(
    plan: Mapping[str, Any],
    *,
    benchmark_validator: BenchmarkValidator = default_benchmark_validator,
    extended_assigner: ExtendedAssigner = default_extended_assigner,
    max_atoms: int = 20,
    max_species: int = 7,
) -> tuple[tuple[SemanticAction, ...], dict[str, Any]]:
    """Compile an existing Plan into strict semantic actions without repair.

    Benchmark-compatible formulas use the independent validator's own
    oxidation-state witness when available.  Formulas rejected by that
    endpoint may still receive an expanded-catalog witness, but are labelled
    ``extended_only`` by the terminal certificate.
    """

    symbols, counts, target_atoms = _plan_symbol_counts(plan)
    if target_atoms > int(max_atoms):
        raise ValueError(f"plan N={target_atoms} exceeds max_atoms={int(max_atoms)}")
    elements = [int(SYMBOL_TO_Z[symbol]) for symbol in symbols]
    reduced = _reduced_counts(counts)
    raw_benchmark = benchmark_validator(elements, reduced)
    benchmark_valid, benchmark_reason = _normalize_validator_result(raw_benchmark)

    tokens: list[FormulaToken] = []
    assignment_source = "none"
    direct_witness_error: str | None = None
    if benchmark_valid is True:
        if benchmark_reason in {"single_element_shortcut", "all_metal_shortcut"}:
            tokens = [
                FormulaToken.from_symbol(symbol, 0, count)
                for symbol, count in zip(symbols, counts)
            ]
            assignment_source = "benchmark_zero_branch"
        elif isinstance(raw_benchmark, Mapping):
            oxidation_states = tuple(raw_benchmark.get("oxidation_states") or ())
            if len(oxidation_states) == len(symbols):
                tokens = [
                    FormulaToken.from_symbol(symbol, int(oxidation), count)
                    for symbol, oxidation, count in zip(symbols, oxidation_states, counts)
                ]
                assignment_source = "benchmark_oxidation_witness"
            else:
                direct_witness_error = "benchmark_witness_missing_or_wrong_length"

    if tokens:
        try:
            replay_actions(
                (
                    SetAtomCount(target_atoms),
                    *tuple(sorted(tokens)),
                    END_COMPOSITION,
                ),
                max_atoms=max_atoms,
                max_species=max_species,
            )
        except ValueError as exc:
            direct_witness_error = f"benchmark_witness_not_replayable:{exc}"
            tokens = []

    if not tokens:
        assigned = extended_assigner(symbols, counts, int(max_species))
        if not assigned:
            reason = direct_witness_error or "no_charge_neutral_smact_state_assignment"
            raise ValueError(f"composition has no semantic valence witness: {reason}")
        tokens = list(assigned)
        assignment_source = "smact_adjacent_mixed_diagnostic"

    actions: tuple[SemanticAction, ...] = (
        SetAtomCount(target_atoms),
        *tuple(sorted(tokens)),
        END_COMPOSITION,
    )
    state = replay_actions(actions, max_atoms=max_atoms, max_species=max_species)
    certificate = state.certificate(benchmark_validator=benchmark_validator)
    if certificate.formula is None:
        raise ValueError("semantic compiler produced an empty formula")
    compiled_composition = tuple(zip(certificate.elements, certificate.counts))
    expected_composition = tuple(zip(elements, counts))
    if compiled_composition != expected_composition:
        raise ValueError(
            f"semantic compiler changed composition {expected_composition!r} -> {compiled_composition!r}"
        )
    return actions, {
        "assignment_source": assignment_source,
        "benchmark_valid": benchmark_valid,
        "benchmark_reason": benchmark_reason,
        "direct_witness_error": direct_witness_error,
        "certificate": certificate.to_dict(),
    }


def _compatible_species(
    state: CCFDv2State,
    catalog: Sequence[FormulaToken],
    *,
    allow_mixed_valence: bool,
    max_species: int,
) -> Iterable[CCFDv2State]:
    for token in catalog:
        if not allow_mixed_valence and int(token.atomic_number) in state.distinct_elements:
            continue
        try:
            yield state.append_species(token, max_species=max_species)
        except ValueError:
            continue


def can_complete(
    state: CCFDv2State,
    catalog: Sequence[FormulaToken],
    *,
    required_certificate: str = CERTIFICATE_BENCHMARK,
    benchmark_validator: BenchmarkValidator = default_benchmark_validator,
    max_species: int = 7,
) -> bool:
    """Exact semantic look-ahead for a finite action catalogue.

    This is intended for the semantic decoder layer and CPU audits.  It is not
    a text/BPE prefix heuristic.  The catalogue should contain only actions
    representable by the selected model vocabulary.
    """

    if required_certificate not in {CERTIFICATE_BENCHMARK, CERTIFICATE_EXTENDED_ONLY, "any_exact"}:
        raise ValueError(f"unsupported required certificate {required_certificate!r}")
    if state.target_atoms is None or state.remaining_atoms is None or state.ended:
        return False
    ordered_catalog = tuple(sorted(set(catalog)))

    @lru_cache(maxsize=None)
    def visit(current: CCFDv2State) -> bool:
        if current.eos_legal:
            terminal = current.end()
            certificate = terminal.certificate(benchmark_validator=benchmark_validator)
            if required_certificate == "any_exact":
                return certificate.extended_valid
            if required_certificate == CERTIFICATE_BENCHMARK:
                return certificate.benchmark_compatible
            return certificate.certificate_class == CERTIFICATE_EXTENDED_ONLY
        if current.remaining_atoms is None or current.remaining_atoms <= 0:
            return False
        if len(current.tokens) >= int(max_species):
            return False
        for next_state in _compatible_species(
            current,
            ordered_catalog,
            allow_mixed_valence=required_certificate != CERTIFICATE_BENCHMARK,
            max_species=max_species,
        ):
            if visit(next_state):
                return True
        return False

    return visit(state)


def legal_next_actions(
    state: CCFDv2State,
    catalog: Sequence[FormulaToken],
    *,
    atom_count_choices: Sequence[int] = tuple(range(1, 21)),
    required_certificate: str = CERTIFICATE_BENCHMARK,
    benchmark_validator: BenchmarkValidator = default_benchmark_validator,
    max_atoms: int = 20,
    max_species: int = 7,
) -> tuple[SemanticAction, ...]:
    """Return only actions that retain at least one certified completion."""

    if state.ended:
        return ()
    if state.needs_atom_count:
        legal_n: list[SemanticAction] = []
        for total in sorted(set(int(value) for value in atom_count_choices)):
            try:
                candidate = state.set_atom_count(total, max_atoms=max_atoms)
            except ValueError:
                continue
            if can_complete(
                candidate,
                catalog,
                required_certificate=required_certificate,
                benchmark_validator=benchmark_validator,
                max_species=max_species,
            ):
                legal_n.append(SetAtomCount(total))
        return tuple(legal_n)
    if state.eos_legal:
        terminal = state.end().certificate(benchmark_validator=benchmark_validator)
        if required_certificate == "any_exact" and terminal.extended_valid:
            return (END_COMPOSITION,)
        if required_certificate == CERTIFICATE_BENCHMARK and terminal.benchmark_compatible:
            return (END_COMPOSITION,)
        if required_certificate == CERTIFICATE_EXTENDED_ONLY and terminal.certificate_class == CERTIFICATE_EXTENDED_ONLY:
            return (END_COMPOSITION,)

    legal: list[SemanticAction] = []
    ordered_catalog = tuple(sorted(set(catalog)))
    for token in ordered_catalog:
        if (
            required_certificate == CERTIFICATE_BENCHMARK
            and int(token.atomic_number) in state.distinct_elements
        ):
            continue
        try:
            candidate = state.append_species(token, max_species=max_species)
        except ValueError:
            continue
        if can_complete(
            candidate,
            ordered_catalog,
            required_certificate=required_certificate,
            benchmark_validator=benchmark_validator,
            max_species=max_species,
        ):
            legal.append(token)
    return tuple(legal)


class BenchmarkReachability:
    """Fast strict-N/charge look-ahead over a typed species vocabulary.

    The dynamic program chooses at most one oxidation state per future element,
    matching the formal benchmark branch.  Pauling/metric compatibility is
    still checked independently at EOS; this oracle never substitutes for the
    terminal benchmark certificate.
    """

    def __init__(self, nodes: Sequence[tuple[int, int]]) -> None:
        grouped: dict[int, set[int]] = {}
        for atomic_number, oxidation in nodes:
            grouped.setdefault(int(atomic_number), set()).add(int(oxidation))
        self.elements = tuple(sorted(grouped))
        self.states = tuple(
            tuple(sorted(grouped[atomic_number])) for atomic_number in self.elements
        )
        self.element_to_group = {
            atomic_number: index for index, atomic_number in enumerate(self.elements)
        }

    @lru_cache(maxsize=None)
    def _reachable_charges(
        self,
        group_index: int,
        remaining_atoms: int,
        remaining_slots: int,
        branch: str,
    ) -> frozenset[int]:
        atoms = int(remaining_atoms)
        slots = int(remaining_slots)
        index = int(group_index)
        if atoms == 0:
            return frozenset({0})
        if atoms < 0 or slots <= 0 or index >= len(self.elements):
            return frozenset()
        reachable = set(
            self._reachable_charges(index + 1, atoms, slots, branch)
        )
        for oxidation in self.states[index]:
            if branch == "ionic" and oxidation == 0:
                continue
            if branch == "alloy" and oxidation != 0:
                continue
            for count in range(1, atoms + 1):
                suffix = self._reachable_charges(
                    index + 1, atoms - count, slots - 1, branch
                )
                reachable.update(int(oxidation) * count + value for value in suffix)
        return frozenset(reachable)

    @lru_cache(maxsize=None)
    def _reachable_charges_exact(
        self,
        group_index: int,
        remaining_atoms: int,
        remaining_slots: int,
        branch: str,
    ) -> frozenset[int]:
        atoms = int(remaining_atoms)
        slots = int(remaining_slots)
        index = int(group_index)
        if atoms == 0:
            return frozenset({0}) if slots == 0 else frozenset()
        if atoms < 0 or slots <= 0 or index >= len(self.elements):
            return frozenset()
        reachable = set(
            self._reachable_charges_exact(index + 1, atoms, slots, branch)
        )
        for oxidation in self.states[index]:
            if branch == "ionic" and oxidation == 0:
                continue
            if branch == "alloy" and oxidation != 0:
                continue
            for count in range(1, atoms + 1):
                suffix = self._reachable_charges_exact(
                    index + 1, atoms - count, slots - 1, branch
                )
                reachable.update(int(oxidation) * count + value for value in suffix)
        return frozenset(reachable)

    def can_complete(
        self,
        state: CCFDv2State,
        *,
        max_species: int = 7,
        target_arity: int | None = None,
    ) -> bool:
        if state.ended or state.target_atoms is None or state.remaining_atoms is None:
            return False
        if target_arity is not None:
            target = int(target_arity)
            if target <= 0 or target > int(max_species) or len(state.tokens) > target:
                return False
        else:
            target = int(max_species)
        if state.remaining_atoms == 0:
            return bool(
                state.conservation_complete
                and (target_arity is None or len(state.tokens) == target)
            )
        if state.branch not in {"ionic", "alloy"}:
            return False
        remaining_slots = target - len(state.tokens)
        if remaining_slots <= 0:
            return False
        last_element = max(state.distinct_elements, default=0)
        start = 0
        while start < len(self.elements) and self.elements[start] <= last_element:
            start += 1
        if target_arity is None:
            charges = self._reachable_charges(
                start,
                int(state.remaining_atoms),
                remaining_slots,
                str(state.branch),
            )
        else:
            charges = self._reachable_charges_exact(
                start,
                int(state.remaining_atoms),
                remaining_slots,
                str(state.branch),
            )
        return -int(state.net_charge) in charges

    def legal_species_counts(
        self,
        state: CCFDv2State,
        *,
        benchmark_validator: BenchmarkValidator = default_benchmark_validator,
        max_species: int = 7,
        target_arity: int | None = None,
    ) -> tuple[FormulaToken, ...]:
        if state.ended or state.target_atoms is None or state.remaining_atoms is None:
            return ()
        last_element = max(state.distinct_elements, default=0)
        if target_arity is not None:
            target = int(target_arity)
            if target <= 0 or target > int(max_species) or len(state.tokens) >= target:
                return ()
        else:
            target = int(max_species)
        legal: list[FormulaToken] = []
        for atomic_number, states in zip(self.elements, self.states):
            if int(atomic_number) <= int(last_element):
                continue
            for oxidation in states:
                for count in range(1, int(state.remaining_atoms) + 1):
                    token = FormulaToken(int(atomic_number), int(oxidation), int(count))
                    try:
                        candidate = state.append_species(token, max_species=max_species)
                    except ValueError:
                        continue
                    if candidate.remaining_atoms == 0:
                        if (
                            candidate.conservation_complete
                            and (target_arity is None or len(candidate.tokens) == target)
                            and candidate.end()
                            .certificate(benchmark_validator=benchmark_validator)
                            .benchmark_compatible
                        ):
                            legal.append(token)
                    elif self.can_complete(
                        candidate,
                        max_species=max_species,
                        target_arity=target_arity,
                    ):
                        legal.append(token)
        return tuple(legal)


def state_to_plan_state(
    state: CCFDv2State,
    *,
    soft_fields: Mapping[str, Any],
    benchmark_validator: BenchmarkValidator = default_benchmark_validator,
) -> dict[str, Any]:
    """Render one ended semantic state into the existing rich-Plan state."""

    certificate = state.certificate(benchmark_validator=benchmark_validator)
    if not certificate.extended_valid:
        raise ValueError("only an exact ended composition can be rendered")
    symbols = [Z_TO_SYMBOL[int(value)] for value in certificate.elements]
    counts = [int(value) for value in certificate.counts]
    plan = dict(soft_fields)
    plan.update(
        {
            "N": int(state.target_atoms or 0),
            "elements": symbols,
            "counts": counts,
            "formula": str(certificate.formula),
            "reduced_formula": formula_from_composition(
                certificate.elements, certificate.reduced_counts
            ),
            "valence_species": [
                {
                    "element": Z_TO_SYMBOL[int(token.atomic_number)],
                    "oxidation_state": int(token.oxidation_state),
                    "count": int(token.count),
                }
                for token in state.tokens
            ],
            "ccfd_v2_certificate": certificate.to_dict(),
        }
    )
    if certificate.benchmark_compatible:
        if certificate.benchmark_reason == "single_element_shortcut":
            plan["charge_bucket"] = "single_element"
        elif certificate.benchmark_reason == "all_metal_shortcut":
            plan["charge_bucket"] = "all_metal"
        else:
            plan["charge_bucket"] = "neutral_plausible"
    else:
        # Extended chemistry is kept explicit and cannot masquerade as a
        # benchmark-compatible charge bucket.
        plan["charge_bucket"] = "validator_unavailable"
    return plan


def render_rich_plan(
    state: CCFDv2State,
    *,
    soft_fields: Mapping[str, Any],
    benchmark_validator: BenchmarkValidator = default_benchmark_validator,
) -> str:
    from crystal_dlm.r5_plan_body import H1_RICH_PLAN_FORMAT, format_composition_plan

    plan = state_to_plan_state(
        state,
        soft_fields=soft_fields,
        benchmark_validator=benchmark_validator,
    )
    return format_composition_plan(plan, plan_style=H1_RICH_PLAN_FORMAT)


__all__ = [
    "CERTIFICATE_BENCHMARK",
    "CERTIFICATE_BENCHMARK_UNKNOWN",
    "CERTIFICATE_CLASSES",
    "CERTIFICATE_EXTENDED_ONLY",
    "CERTIFICATE_INCOMPLETE",
    "CERTIFICATE_INVALID",
    "CCFDv2State",
    "BenchmarkReachability",
    "CompositionCertificate",
    "END_COMPOSITION",
    "EndComposition",
    "SemanticAction",
    "SetAtomCount",
    "can_complete",
    "compile_plan_actions",
    "default_benchmark_validator",
    "default_extended_assigner",
    "legal_next_actions",
    "render_rich_plan",
    "replay_actions",
    "state_to_plan_state",
]
