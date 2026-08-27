"""Text-level helpers for attaching CCFD to an unchanged AR tokenizer.

The semantic conservation state lives in :mod:`crystal_dlm.ccfd`.  This module
only answers a narrower interface question: can a decoded text prefix still be
completed into a flat MP-20 formula without repairing or replacing a sample?
It deliberately does not assign oxidation states or judge stability.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from crystal_dlm.fixed_slot import SYMBOL_TO_Z


_TERM_RE = re.compile(r"([A-Z][a-z]?)([1-9][0-9]*)?")


@dataclass(frozen=True)
class FormulaPrefixStatus:
    """Syntactic status of one formula-value prefix."""

    valid_prefix: bool
    terminal: bool
    elements: tuple[str, ...] = ()
    counts: tuple[int, ...] = ()
    total_atoms: int = 0
    reason: str = "ok"


def _complete_terms(text: str) -> tuple[list[tuple[str, int]], str | None]:
    """Return complete terms and a possible trailing partial element symbol."""

    terms: list[tuple[str, int]] = []
    offset = 0
    while offset < len(text):
        match = _TERM_RE.match(text, offset)
        if match is None:
            # A final uppercase character may be the start of a two-letter
            # symbol.  It is not terminal, but it remains a legal prefix.
            suffix = text[offset:]
            if (
                len(suffix) == 1
                and suffix.isupper()
                and any(symbol.startswith(suffix) for symbol in SYMBOL_TO_Z)
            ):
                return terms, suffix
            raise ValueError("invalid_formula_character_or_term")
        symbol = match.group(1)
        count_text = match.group(2)
        end = match.end()
        if symbol not in SYMBOL_TO_Z:
            # The regex greedily consumes a trailing lower-case character.  If
            # the unsupported symbol is the last term, it can still be a
            # partial prefix only when a supported symbol begins with it.
            if end == len(text) and any(value.startswith(symbol) for value in SYMBOL_TO_Z):
                return terms, symbol
            raise ValueError("unsupported_element")
        count = 1 if count_text is None else int(count_text)
        terms.append((symbol, count))
        offset = end
    return terms, None


def analyze_formula_prefix(prefix: str, *, max_atoms: int = 20) -> FormulaPrefixStatus:
    """Classify a flat formula prefix under a fixed maximum atom count.

    The function accepts a complete formula as a valid prefix as well.  It
    rejects whitespace, zero/leading-zero counts, repeated elements, malformed
    symbols, and prefixes whose already committed atom count exceeds the
    budget.  Chemical charge legality is intentionally handled by CCFD rather
    than this lexical layer.
    """

    text = str(prefix)
    if not text:
        return FormulaPrefixStatus(valid_prefix=True, terminal=False, reason="empty_prefix")
    if any(character.isspace() for character in text):
        return FormulaPrefixStatus(False, False, reason="whitespace_in_formula")
    if "0" in text:
        # Zero is legal inside 10 or 20, but never as the first digit of a
        # count.  Let the parser below distinguish those cases.
        if re.search(r"[A-Za-z]0", text):
            return FormulaPrefixStatus(False, False, reason="zero_or_leading_zero_count")
    # A trailing uppercase character is ambiguous under ordinary BPE: it may
    # be a complete one-letter element or the first character of a two-letter
    # element.  Preserve the latter branch even when greedily treating it as a
    # one-letter element would look like a duplicate (for example ``P6P`` is a
    # legal prefix of ``P6Pd6``).
    partial_option: FormulaPrefixStatus | None = None
    if text[-1].isupper():
        base_text = text[:-1]
        base = analyze_formula_prefix(base_text, max_atoms=max_atoms) if base_text else FormulaPrefixStatus(True, True)
        candidates = [
            symbol
            for symbol in SYMBOL_TO_Z
            if len(symbol) == 2
            and symbol.startswith(text[-1])
            and symbol not in set(base.elements)
        ]
        if base.terminal and candidates and base.total_atoms + 1 <= int(max_atoms):
            partial_option = FormulaPrefixStatus(
                valid_prefix=True,
                terminal=False,
                elements=base.elements,
                counts=base.counts,
                total_atoms=base.total_atoms,
                reason="partial_element",
            )
    try:
        terms, partial = _complete_terms(text)
    except ValueError as exc:
        return partial_option or FormulaPrefixStatus(False, False, reason=str(exc))
    symbols = [symbol for symbol, _count in terms]
    if len(symbols) != len(set(symbols)):
        return partial_option or FormulaPrefixStatus(False, False, reason="repeated_element")
    counts = [int(count) for _symbol, count in terms]
    total = sum(counts)
    if any(count <= 0 for count in counts):
        return FormulaPrefixStatus(False, False, reason="nonpositive_count")
    if total > int(max_atoms):
        return FormulaPrefixStatus(False, False, reason="atom_budget_exceeded")
    terminal = partial is None and bool(terms)
    return FormulaPrefixStatus(
        valid_prefix=True,
        terminal=terminal,
        elements=tuple(symbols),
        counts=tuple(counts),
        total_atoms=int(total),
        reason="ok" if terminal else "partial_element",
    )


def formula_term_boundaries(formula: str) -> tuple[int, ...]:
    """Return character offsets after each complete element/count term."""

    status = analyze_formula_prefix(formula)
    if not status.terminal:
        raise ValueError(f"formula is not terminal: {status.reason}")
    offsets: list[int] = []
    cursor = 0
    while cursor < len(formula):
        match = _TERM_RE.match(formula, cursor)
        if match is None:
            raise ValueError("formula term boundary parse failed")
        cursor = match.end()
        offsets.append(cursor)
    return tuple(offsets)


def token_prefix_alignment(
    tokenizer: object,
    token_ids: Sequence[int],
    target_text: str,
    *,
    max_atoms: int = 20,
) -> dict[str, object]:
    """Audit incremental decode alignment up to the first formula newline."""

    prefix_lengths: list[int] = []
    formula_prefix_valid = True
    exact_prefix_decode = True
    newline_step: int | None = None
    first_invalid_formula_prefix: str | None = None
    first_invalid_reason: str | None = None
    first_nonprefix_decode: str | None = None
    for step in range(1, len(token_ids) + 1):
        decoded = tokenizer.decode(  # type: ignore[attr-defined]
            list(token_ids[:step]),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not str(target_text).startswith(decoded):
            exact_prefix_decode = False
            if first_nonprefix_decode is None:
                first_nonprefix_decode = decoded
        prefix_lengths.append(len(decoded))
        formula_text = decoded.split("\n", 1)[0]
        if formula_text.startswith(" "):
            formula_text = formula_text[1:]
        status = analyze_formula_prefix(formula_text, max_atoms=max_atoms)
        if not status.valid_prefix:
            formula_prefix_valid = False
            if first_invalid_formula_prefix is None:
                first_invalid_formula_prefix = formula_text
                first_invalid_reason = status.reason
        if "\n" in decoded:
            newline_step = step
            break
    full_decoded = tokenizer.decode(  # type: ignore[attr-defined]
        list(token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return {
        "roundtrip_exact": full_decoded == str(target_text),
        "incremental_prefix_exact": exact_prefix_decode,
        "formula_prefix_valid": formula_prefix_valid,
        "newline_step": newline_step,
        "prefix_lengths": prefix_lengths,
        "first_invalid_formula_prefix": first_invalid_formula_prefix,
        "first_invalid_reason": first_invalid_reason,
        "first_nonprefix_decode": first_nonprefix_decode,
    }


__all__ = [
    "FormulaPrefixStatus",
    "analyze_formula_prefix",
    "formula_term_boundaries",
    "token_prefix_alignment",
]
