"""Conservation-Constrained Formula Decoding primitives.

The module is deliberately model-agnostic.  It defines exact atom/charge
accounting and a finite-state legality mask that can later be attached to an
autoregressive formula sampler without changing its weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

from crystal_dlm.fixed_slot import SYMBOL_TO_Z


BRANCHES = ("ionic", "alloy")


@dataclass(frozen=True, order=True)
class FormulaToken:
    """One element/oxidation/count action with exact conservation metadata."""

    atomic_number: int
    oxidation_state: int
    count: int

    def __post_init__(self) -> None:
        if int(self.atomic_number) not in SYMBOL_TO_Z.values():
            raise ValueError(f"unsupported atomic number {self.atomic_number}")
        if int(self.count) <= 0:
            raise ValueError("formula token count must be positive")

    @classmethod
    def from_symbol(cls, element: str, oxidation_state: int, count: int) -> "FormulaToken":
        symbol = str(element)
        if symbol not in SYMBOL_TO_Z:
            raise ValueError(f"unsupported element {symbol!r}")
        return cls(
            atomic_number=int(SYMBOL_TO_Z[symbol]),
            oxidation_state=int(oxidation_state),
            count=int(count),
        )

    @property
    def atom_delta(self) -> int:
        return int(self.count)

    @property
    def charge_delta(self) -> int:
        return int(self.oxidation_state) * int(self.count)

    @property
    def species_key(self) -> tuple[int, int]:
        return int(self.atomic_number), int(self.oxidation_state)

    def to_dict(self) -> dict[str, int]:
        return {
            "atomic_number": int(self.atomic_number),
            "oxidation_state": int(self.oxidation_state),
            "count": int(self.count),
            "atom_delta": self.atom_delta,
            "charge_delta": self.charge_delta,
        }


@dataclass(frozen=True)
class CCFDState:
    """Finite-state conservation ledger for one requested formula."""

    target_atoms: int
    remaining_atoms: int
    remaining_charge: int = 0
    branch: str | None = None
    tokens: tuple[FormulaToken, ...] = ()

    @classmethod
    def start(cls, target_atoms: int) -> "CCFDState":
        target = int(target_atoms)
        if target <= 0:
            raise ValueError("target atom count must be positive")
        return cls(target_atoms=target, remaining_atoms=target)

    @property
    def last_species_key(self) -> tuple[int, int] | None:
        return None if not self.tokens else self.tokens[-1].species_key

    @property
    def distinct_elements(self) -> tuple[int, ...]:
        return tuple(sorted({int(token.atomic_number) for token in self.tokens}))

    @property
    def eos_legal(self) -> bool:
        if not self.tokens or self.remaining_atoms != 0 or self.remaining_charge != 0:
            return False
        if self.branch == "alloy":
            return all(token.oxidation_state == 0 for token in self.tokens)
        if self.branch == "ionic":
            return len(self.distinct_elements) >= 2 and all(
                token.oxidation_state != 0 for token in self.tokens
            )
        return False

    def append(self, token: FormulaToken, *, max_species: int = 7) -> "CCFDState":
        if len(self.tokens) >= int(max_species):
            raise ValueError("maximum species count reached")
        if token.count > self.remaining_atoms:
            raise ValueError("token exceeds remaining atom budget")
        if self.last_species_key is not None and token.species_key <= self.last_species_key:
            raise ValueError("species tokens must follow canonical increasing order")

        token_branch = "alloy" if token.oxidation_state == 0 else "ionic"
        if self.branch is not None and self.branch != token_branch:
            raise ValueError("zero- and nonzero-valence species cannot share a branch")

        same_element = [
            existing
            for existing in self.tokens
            if existing.atomic_number == token.atomic_number
        ]
        if same_element:
            signs = {
                1 if existing.oxidation_state > 0 else -1
                for existing in same_element
                if existing.oxidation_state != 0
            }
            if token.oxidation_state != 0:
                signs.add(1 if token.oxidation_state > 0 else -1)
            if len(signs) > 1:
                raise ValueError("one element cannot use positive and negative valence together")

        return CCFDState(
            target_atoms=self.target_atoms,
            remaining_atoms=self.remaining_atoms - token.atom_delta,
            remaining_charge=self.remaining_charge - token.charge_delta,
            branch=self.branch or token_branch,
            tokens=(*self.tokens, token),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "target_atoms": int(self.target_atoms),
            "remaining_atoms": int(self.remaining_atoms),
            "remaining_charge": int(self.remaining_charge),
            "branch": self.branch,
            "eos_legal": self.eos_legal,
            "tokens": [token.to_dict() for token in self.tokens],
        }


def _compatible_candidates(
    state: CCFDState,
    catalog: Sequence[FormulaToken],
    *,
    max_species: int,
) -> Iterable[FormulaToken]:
    for token in catalog:
        try:
            state.append(token, max_species=max_species)
        except ValueError:
            continue
        yield token


def can_complete(
    state: CCFDState,
    catalog: Sequence[FormulaToken],
    *,
    max_species: int = 7,
) -> bool:
    """Return whether at least one legal continuation reaches a valid EOS."""

    ordered_catalog = tuple(sorted(set(catalog)))

    @lru_cache(maxsize=None)
    def visit(current: CCFDState) -> bool:
        if current.eos_legal:
            return True
        if current.remaining_atoms <= 0 or len(current.tokens) >= int(max_species):
            return False
        for candidate in _compatible_candidates(
            current, ordered_catalog, max_species=max_species
        ):
            next_state = current.append(candidate, max_species=max_species)
            if visit(next_state):
                return True
        return False

    return visit(state)


def legal_next_tokens(
    state: CCFDState,
    catalog: Sequence[FormulaToken],
    *,
    max_species: int = 7,
) -> tuple[FormulaToken, ...]:
    """Mask out actions that cannot lead to a complete conserved formula."""

    legal: list[FormulaToken] = []
    ordered_catalog = tuple(sorted(set(catalog)))
    for token in _compatible_candidates(state, ordered_catalog, max_species=max_species):
        next_state = state.append(token, max_species=max_species)
        if next_state.eos_legal or can_complete(
            next_state, ordered_catalog, max_species=max_species
        ):
            legal.append(token)
    return tuple(legal)


def replay_tokens(
    target_atoms: int,
    tokens: Sequence[FormulaToken],
    *,
    max_species: int = 7,
) -> CCFDState:
    state = CCFDState.start(target_atoms)
    for token in tokens:
        state = state.append(token, max_species=max_species)
    return state


__all__ = [
    "BRANCHES",
    "CCFDState",
    "FormulaToken",
    "can_complete",
    "legal_next_tokens",
    "replay_tokens",
]
