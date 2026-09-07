"""Finite-language reference for the native H1A2/C3FD token bridge.

This intentionally does not import a production parser or reachability kernel.
For a tiny synthetic chemical vocabulary it exhaustively enumerates *complete*
compositions, checks every single-valence-per-element witness, and spells each
accepted composition in every element order.  Prefix membership is then a
plain string-prefix lookup.  It is a test oracle, never a runtime decoder or
scientific composition-validity implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations, product
from typing import Iterable


@dataclass(frozen=True)
class SyntheticElement:
    symbol: str
    atomic_number: int
    oxidation_states: tuple[int, ...]
    electronegativity: float | None
    metal: bool = False


@dataclass(frozen=True)
class TerminalComposition:
    terms: tuple[tuple[str, int], ...]
    witnesses: tuple[tuple[int, ...], ...]
    family: str

    @property
    def target(self) -> tuple[int, str, int]:
        """The latent (N, family, arity) stratum; no stratum is sampled."""
        return sum(count for _symbol, count in self.terms), self.family, len(self.terms)


def _positive_count_tuples(slots: int, atom_budget: int) -> Iterable[tuple[int, ...]]:
    if slots == 0:
        yield ()
        return
    for first in range(1, atom_budget - (slots - 1) + 1):
        for tail in _positive_count_tuples(slots - 1, atom_budget - first):
            yield (first, *tail)


def _family(symbols: set[str]) -> str:
    # Written independently to expose production family-ledger mistakes.
    for family, required in (
        ("oxide", {"O"}),
        ("sulfide", {"S"}),
        ("chalcogenide", {"Se", "Te"}),
        ("halide", {"F", "Cl", "Br", "I"}),
        ("nitride", {"N"}),
        ("phosphide_or_phosphate", {"P"}),
    ):
        if symbols & required:
            return family
    return "other"


def _witnesses(
    elements: tuple[SyntheticElement, ...], counts: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    accepted: list[tuple[int, ...]] = []
    for oxidation in product(*(element.oxidation_states for element in elements)):
        if sum(count * state for count, state in zip(counts, oxidation)) != 0:
            continue
        if all(state == 0 for state in oxidation):
            if len(elements) == 1 or all(element.metal for element in elements):
                accepted.append(oxidation)
            continue
        if any(state == 0 for state in oxidation):
            continue
        cations = [
            element.electronegativity
            for element, state in zip(elements, oxidation)
            if state > 0
        ]
        anions = [
            element.electronegativity
            for element, state in zip(elements, oxidation)
            if state < 0
        ]
        if not cations or not anions or None in cations or None in anions:
            continue
        if max(cations) < min(anions):
            accepted.append(oxidation)
    return tuple(accepted)


def _spellings(terms: tuple[tuple[str, int], ...]) -> Iterable[str]:
    for ordered in permutations(terms):
        variants = [
            (symbol, symbol + "1") if count == 1 else (symbol + str(count),)
            for symbol, count in ordered
        ]
        for pieces in product(*variants):
            yield "".join(pieces)


class BruteForceFormulaLanguage:
    """Exact prefix language under an explicitly small synthetic vocabulary."""

    def __init__(
        self,
        elements: tuple[SyntheticElement, ...],
        *,
        max_atoms: int,
        max_species: int,
        allowed_strata: frozenset[tuple[int, int, str]] | None = None,
    ) -> None:
        self.elements = tuple(elements)
        self.max_atoms = int(max_atoms)
        self.max_species = int(max_species)
        if len({element.symbol for element in elements}) != len(elements):
            raise ValueError("synthetic symbols must be distinct")
        records: list[TerminalComposition] = []
        spellings: dict[str, TerminalComposition] = {}
        for arity in range(1, min(len(elements), self.max_species) + 1):
            for chosen in combinations(elements, arity):
                for counts in _positive_count_tuples(arity, self.max_atoms):
                    witnesses = _witnesses(chosen, counts)
                    if not witnesses:
                        continue
                    record = TerminalComposition(
                        terms=tuple((element.symbol, count) for element, count in zip(chosen, counts)),
                        witnesses=witnesses,
                        family=_family({element.symbol for element in chosen}),
                    )
                    stratum = (sum(counts), arity, record.family)
                    if allowed_strata is not None and stratum not in allowed_strata:
                        continue
                    records.append(record)
                    for text in _spellings(record.terms):
                        spellings[text] = record
        self.records = tuple(records)
        self.spellings = spellings
        self.terminals = frozenset(spellings)
        self.prefixes = frozenset(
            text[:end] for text in spellings for end in range(len(text) + 1)
        )

    def is_terminal_valid(self, text: str) -> bool:
        return str(text) in self.terminals

    def is_prefix_viable(self, text: str) -> bool:
        return str(text) in self.prefixes

    def compatible_targets(self, text: str) -> frozenset[tuple[int, str, int]]:
        return frozenset(
            record.target for spelling, record in self.spellings.items() if spelling.startswith(text)
        )

    def fragment_legal(self, current_value: str, fragment: str) -> bool:
        """One optional generated leading space; first newline ends chemistry."""
        combined = str(current_value) + str(fragment)
        value = combined[1:] if combined.startswith(" ") else combined
        if "\n" in value:
            return self.is_terminal_valid(value.split("\n", 1)[0])
        return self.is_prefix_viable(value)

    def native_field_fragment_legal(self, current_value: str, fragment: str) -> bool:
        """The frozen rich parser permits whitespace around its field value.

        Interior whitespace must not become a way to discard chemical terms.
        Once whitespace follows a completed formula only more padding or the
        newline can follow; the formula can no longer grow a count/element.
        """
        value = (str(current_value) + str(fragment)).lstrip(" \t")
        if "\n" in value:
            return self.is_terminal_valid(value.split("\n", 1)[0].strip(" \t\r"))
        if value.endswith((" ", "\t", "\r")):
            return self.is_terminal_valid(value.rstrip(" \t\r"))
        return self.is_prefix_viable(value)


CORE_ELEMENTS = (
    SyntheticElement("Li", 3, (0, 1), 0.98, True),
    SyntheticElement("O", 8, (-2, 0), 3.44),
    SyntheticElement("F", 9, (-1, 0), 3.98),
    SyntheticElement("Fe", 26, (0, 2, 3), 1.83, True),
)


class PieceTokenizer:
    """Tiny additive tokenizer with deliberately hostile boundary fragments."""

    eos_token_id = 0
    pieces = (
        "", "formula", ":", " ", " Fe", "Fe", "F", "e", "Li", "L", "i",
        "O", "2", "3", "1", "0", "10", "11", "20", "21", "2O3",
        "\n", "\nanion:", "2O3\nanion: oxide", " Fe2O3\nanion:",
        " Fe2O4\nanion:", "Xx", " Fe0", "-", "  Fe", "for", "mula: Fe",
        "mula: Xx\n", "formula: Fe2O3\nanion:", "anion: oxide\n", " H",
        "Fe2O3", "FeO", "OFe", "F2Fe", "F2F", "\nformula: Fe", ": Fe",
        "\nvolume: nonsense", " Li1", "Fe2O4", "O3Fe2", "Na", " Cl",
    )

    def __len__(self) -> int:
        return len(self.pieces)

    def decode(self, token_ids, **_kwargs) -> str:
        return "".join(self.pieces[int(token_id)] for token_id in token_ids)

    def token_id(self, piece: str) -> int:
        return self.pieces.index(piece)
