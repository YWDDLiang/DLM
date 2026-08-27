"""Tokenizer-preserving online CCFD mask for AR formula generation."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, Callable, Sequence

from crystal_dlm.ccfd_text import analyze_formula_prefix
from crystal_dlm.valence_assignment import assign_crysvcd_valences


FormulaValidator = Callable[[Sequence[str], Sequence[int]], bool]
_TOKEN_FRAGMENT_CACHE: dict[
    tuple[int, int, int], tuple[dict[int, str], dict[str, tuple[int, ...]]]
] = {}


@lru_cache(maxsize=200_000)
def _phase0_assignment_cached(elements: tuple[str, ...], counts: tuple[int, ...]) -> bool:
    # Elemental crystals are a dedicated zero-charge unary branch, matching
    # the benchmark's single-element validity contract.
    if len(elements) == 1 and len(counts) == 1 and int(counts[0]) > 0:
        return True
    assignment = assign_crysvcd_valences(elements, counts, max_species=7)
    return assignment.get("assigned") is True


def phase0_assignment_validator(elements: Sequence[str], counts: Sequence[int]) -> bool:
    """Use exactly the frozen Phase-0 representability contract at EOS."""

    return _phase0_assignment_cached(
        tuple(str(value) for value in elements),
        tuple(int(value) for value in counts),
    )


def _formula_value(text: str) -> str | None:
    """Permit the one leading space produced after a ``formula:`` prefill."""

    value = str(text)
    if value.startswith(" "):
        value = value[1:]
    if value.startswith(" "):
        return None
    return value


def _token_fragment_index(
    tokenizer: Any, eos_token_id: int
) -> tuple[dict[int, str], dict[str, tuple[int, ...]]]:
    cache_key = (id(tokenizer), len(tokenizer), int(eos_token_id))
    cached = _TOKEN_FRAGMENT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    fragments: dict[int, str] = {}
    grouped: dict[str, list[int]] = {}
    for token_id in range(len(tokenizer)):
        if token_id == int(eos_token_id):
            continue
        fragment = str(
            tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
        if not fragment:
            continue
        first = fragment[0]
        if first not in " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\n":
            continue
        formula_part = fragment.split("\n", 1)[0]
        if first == " ":
            # A leading-space token is legal only as the first formula token;
            # discard the enormous class of ordinary lowercase word tokens.
            value = formula_part[1:]
            if not value or not value[0].isupper() or not value.isalnum():
                continue
        elif first == "\n":
            pass
        elif not formula_part.isalnum():
            continue
        fragments[token_id] = fragment
        grouped.setdefault(first, []).append(token_id)
    result = (fragments, {key: tuple(values) for key, values in grouped.items()})
    _TOKEN_FRAGMENT_CACHE[cache_key] = result
    return result


class CCFDFormulaLogitsProcessor:
    """Mask tokens until the formula field reaches a representable newline.

    Rich fields after the first newline are deliberately unconstrained.  The
    processor changes neither model weights nor tokenizer vocabulary.  If a
    trajectory reaches a prefix with no legal token, only EOS remains; the
    downstream strict parser records that request as failed instead of repairing
    or replacing it.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        start_length: int,
        eos_token_id: int,
        max_atoms: int = 20,
        validator: FormulaValidator = phase0_assignment_validator,
    ) -> None:
        self.tokenizer = tokenizer
        self.start_length = int(start_length)
        self.eos_token_id = int(eos_token_id)
        self.max_atoms = int(max_atoms)
        self.validator = validator
        self.fragments, self.by_first_character = _token_fragment_index(
            tokenizer, self.eos_token_id
        )
        self._allowed_cache: dict[str, tuple[int, ...] | None] = {}

    def _terminal_valid(self, formula: str) -> bool:
        status = analyze_formula_prefix(formula, max_atoms=self.max_atoms)
        return bool(
            status.terminal
            and self.validator(status.elements, status.counts)
        )

    def _fragment_legal(self, current: str, fragment: str) -> bool:
        combined = current + fragment
        normalized = _formula_value(combined)
        if normalized is None:
            return False
        if "\n" in normalized:
            formula = normalized.split("\n", 1)[0]
            return self._terminal_valid(formula)
        return analyze_formula_prefix(normalized, max_atoms=self.max_atoms).valid_prefix

    def allowed_token_ids(self, generated_ids: Sequence[int]) -> tuple[int, ...] | None:
        """Return ``None`` after formula completion, otherwise legal token ids."""

        current = str(
            self.tokenizer.decode(
                list(int(value) for value in generated_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
        if "\n" in current:
            return None
        cached = self._allowed_cache.get(current)
        if cached is not None:
            return cached

        possible_first: list[str] = []
        for character in self.by_first_character:
            if character == " " and current:
                continue
            if character == "\n":
                normalized = _formula_value(current)
                if normalized is not None and self._terminal_valid(normalized):
                    possible_first.append(character)
                continue
            normalized = _formula_value(current + character)
            if normalized is not None and analyze_formula_prefix(
                normalized, max_atoms=self.max_atoms
            ).valid_prefix:
                possible_first.append(character)

        allowed: list[int] = []
        for character in possible_first:
            for token_id in self.by_first_character[character]:
                if self._fragment_legal(current, self.fragments[token_id]):
                    allowed.append(token_id)
        if not allowed:
            allowed = [self.eos_token_id]
        result = tuple(sorted(set(allowed)))
        self._allowed_cache[current] = result
        return result

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        masked = scores.clone()
        for row_index in range(int(input_ids.shape[0])):
            generated = input_ids[row_index, self.start_length :].tolist()
            allowed = self.allowed_token_ids(generated)
            if allowed is None:
                continue
            row = torch.full_like(masked[row_index], -math.inf)
            indexes = torch.tensor(allowed, dtype=torch.long, device=masked.device)
            row[indexes] = masked[row_index, indexes]
            masked[row_index] = row
        return masked


__all__ = [
    "CCFDFormulaLogitsProcessor",
    "FormulaValidator",
    "phase0_assignment_validator",
]
