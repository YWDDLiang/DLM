"""Latest C3FD witness support on an unchanged H1A2 text-token decoder.

Display order is free.  Completed chemical terms are a mandatory element SET,
not a prefix of C3FD's canonical element traversal.  N/family/arity are latent
members of the declared support, never sampled by a second composer.
"""
from __future__ import annotations

from functools import lru_cache
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from crystal_dlm.family_reachability import FAMILY_FORBIDDEN, FAMILY_REQUIRED
from crystal_dlm.r5_plan_state import anion_framework_from_symbols


_TERM = re.compile(r"([A-Z][a-z]?)([1-9][0-9]*)?")
_LABEL = "formula:"


class C3FDFormulaOracle:
    """Exact finite-support prefix viability with single-valence witnesses.

    The charge-bitset recurrence is the C3FD-v2.5 mechanism, generalized to
    remaining element sets so an arbitrary original formula order is retained.
    A bounded cache changes runtime only; no approximate acceptance is used.
    """

    def __init__(
        self,
        nodes: Mapping[str, Sequence[int]],
        electronegativities: Mapping[str, float | None],
        metal_symbols: Iterable[str],
        max_atoms: int = 20,
        max_species: int = 7,
        allowed_strata: Iterable[tuple[int, int, str]] | None = None,
        *,
        cache_size: int = 250_000,
    ) -> None:
        if not nodes or max_atoms < 1 or max_species < 1:
            raise ValueError("nonempty vocabulary and positive bounds are required")
        self.max_atoms, self.max_species = int(max_atoms), int(max_species)
        self.symbols = tuple(sorted(str(s) for s in nodes))
        if any(re.fullmatch(r"[A-Z][a-z]?", s) is None for s in self.symbols):
            raise ValueError("node keys must be element symbols")
        self.indices = {s: i for i, s in enumerate(self.symbols)}
        self.states = tuple(tuple(sorted(set(map(int, nodes[s])))) for s in self.symbols)
        if any(not q for q in self.states):
            raise ValueError("every element requires a declared oxidation state")
        self.metals = frozenset(str(s) for s in metal_symbols)
        self.eneg = {s: electronegativities.get(s) for s in self.symbols}
        if any(v is not None and not math.isfinite(float(v)) for v in self.eneg.values()):
            raise ValueError("electronegativity must be finite or explicitly unavailable")
        levels = sorted({float(v) for v in self.eneg.values() if v is not None})
        ranks = {v: i for i, v in enumerate(levels)}
        self.ranks = {s: None if self.eneg[s] is None else ranks[float(self.eneg[s])] for s in self.symbols}
        self.boundaries = tuple(range(max(0, len(levels) - 1)))
        self.choices = tuple(
            tuple(
                tuple(q for q in self.states[i] if q != 0 and self.ranks[s] is not None
                      and ((q > 0 and self.ranks[s] <= b) or (q < 0 and self.ranks[s] > b)))
                for i, s in enumerate(self.symbols)
            ) for b in self.boundaries
        )
        self.boundary_masks = tuple(sum(1 << i for i, qs in enumerate(row) if qs) for row in self.choices)
        self.zero_mask = sum(1 << i for i, qs in enumerate(self.states) if 0 in qs)
        self.metal_mask = sum(1 << i for i, s in enumerate(self.symbols) if s in self.metals)
        self.all_mask = (1 << len(self.symbols)) - 1
        self.family_masks = {f: sum(1 << i for i, s in enumerate(self.symbols) if s not in forbidden)
                             for f, forbidden in FAMILY_FORBIDDEN.items()}
        self.required_masks = {f: sum(1 << i for i, s in enumerate(self.symbols) if s in required)
                               for f, required in FAMILY_REQUIRED.items()}
        self.offset = max(1, max(abs(q) for states in self.states for q in states) * self.max_atoms)
        self.bit_mask = (1 << (2 * self.offset + 1)) - 1
        if allowed_strata is None:
            self.strata = tuple((n, k, f) for n in range(1, self.max_atoms + 1)
                                for k in range(1, min(n, self.max_species, len(self.symbols)) + 1)
                                for f in FAMILY_FORBIDDEN)
        else:
            values = {(int(n), int(k), str(f)) for n, k, f in allowed_strata}
            if any(not 1 <= n <= self.max_atoms or not 1 <= k <= min(n, self.max_species)
                   or f not in FAMILY_FORBIDDEN for n, k, f in values):
                raise ValueError("declared stratum is outside the finite support")
            self.strata = tuple(sorted(values))
        self.stratum_set = frozenset(self.strata)
        self._suffix_bits = lru_cache(maxsize=cache_size)(self._suffix_bits_impl)
        self._can_complete_fixed = lru_cache(maxsize=cache_size)(self._can_complete_fixed_impl)
        self._prefix_cached = lru_cache(maxsize=cache_size)(self._prefix_impl)
        self._terminal_cached = lru_cache(maxsize=cache_size)(self._terminal_impl)

    def stats(self) -> dict[str, Any]:
        return {name: getattr(self, name).cache_info()._asdict()
                for name in ("_suffix_bits", "_can_complete_fixed", "_prefix_cached", "_terminal_cached")}

    def is_prefix_viable(self, text: str) -> bool:
        return self._prefix_cached(str(text))

    def is_terminal_valid(self, text: str) -> bool:
        return self._terminal_cached(str(text))

    def _closed_terms(self, text: str) -> tuple[tuple[str, int], ...] | None:
        if not text or not text.isascii():
            return None
        at, terms = 0, []
        while at < len(text):
            match = _TERM.match(text, at)
            if match is None:
                return None
            symbol, digits = match.groups()
            if symbol not in self.indices:
                return None
            terms.append((symbol, 1 if digits is None else int(digits)))
            at = match.end()
        if len({s for s, _ in terms}) != len(terms) or len(terms) > self.max_species:
            return None
        if sum(n for _, n in terms) > self.max_atoms:
            return None
        return tuple(sorted(terms))

    def _terminal_impl(self, text: str) -> bool:
        terms = self._closed_terms(text)
        if terms is None:
            return False
        n, arity = sum(c for _, c in terms), len(terms)
        family = anion_framework_from_symbols([s for s, _ in terms])
        if (n, arity, family) not in self.stratum_set:
            return False
        if all(0 in self.states[self.indices[s]] for s, _ in terms) and (arity == 1 or all(s in self.metals for s, _ in terms)):
            return True
        for boundary in self.boundaries:
            charges = self._fixed_charges(terms, boundary)
            if 0 in charges:
                return True
        return False

    def _prefix_closures(self, text: str):
        """Finish only the last lexical term; prior terms are immutable.

        A final 'F' can become F, Fe or Fr; a final count '1' can become 1,
        10..19.  This is deliberately not a greedy completed-formula parser.
        """
        if not text:
            yield ()
            return
        if not text.isascii() or any(not (c.isalpha() or c.isdigit()) for c in text):
            return
        fixed: list[tuple[str, int]] = []
        seen: set[str] = set()
        at = 0
        while at < len(text):
            if not ("A" <= text[at] <= "Z"):
                return
            start = at
            at += 1
            if at < len(text) and "a" <= text[at] <= "z":
                at += 1
            symbol_text = text[start:at]
            digits_start = at
            while at < len(text) and text[at].isdigit():
                at += 1
            digits = text[digits_start:at]
            if digits.startswith("0"):
                return
            remaining = self.max_atoms - sum(n for _, n in fixed)
            if at == len(text):
                choices = ([s for s in self.symbols if s.startswith(symbol_text)]
                           if not digits else ([symbol_text] if symbol_text in self.indices else []))
                for symbol in choices:
                    if symbol in seen:
                        continue
                    for count in range(1, remaining + 1):
                        if digits and not str(count).startswith(digits):
                            continue
                        if len(fixed) + 1 <= self.max_species:
                            yield tuple(sorted([*fixed, (symbol, count)]))
                return
            if symbol_text not in self.indices or symbol_text in seen:
                return
            count = 1 if not digits else int(digits)
            if count > remaining:
                return
            fixed.append((symbol_text, count));seen.add(symbol_text)
            if len(fixed) >= self.max_species:
                return

    def _prefix_impl(self, text: str) -> bool:
        if not self.strata:
            return False
        return any(self._can_complete_fixed(terms) for terms in self._prefix_closures(text))

    def _fixed_charges(self, terms: tuple[tuple[str, int], ...], boundary: int) -> set[int]:
        charges = {0}
        choices = self.choices[boundary]
        for symbol, count in terms:
            states = choices[self.indices[symbol]]
            if not states:
                return set()
            charges = {value + count * q for value in charges for q in states}
        return charges

    def _can_complete_fixed_impl(self, terms: tuple[tuple[str, int], ...]) -> bool:
        used = sum(1 << self.indices[s] for s, _ in terms)
        atoms, count = sum(n for _, n in terms), len(terms)
        if count > self.max_species or atoms > self.max_atoms:
            return False
        all_zero = all(0 in self.states[self.indices[s]] for s, _ in terms)
        all_metal = all(s in self.metals for s, _ in terms)
        charges_by_boundary: dict[int, set[int]] = {}
        for target_atoms, target_count, family in self.strata:
            slots, remaining = target_count - count, target_atoms - atoms
            if slots < 0 or remaining < slots or (slots == 0 and remaining != 0):
                continue
            family_mask = self.family_masks[family]
            if used & ~family_mask:
                continue
            required = self.required_masks[family]
            need_required = bool(FAMILY_REQUIRED[family]) and not bool(used & required)
            available = family_mask & ~used
            if available.bit_count() < slots:
                continue
            if all_zero and (target_count == 1 or all_metal):
                zero_available = available & self.zero_mask
                if target_count > 1:
                    zero_available &= self.metal_mask
                if (slots == 0 and not need_required) or (
                    slots > 0 and zero_available.bit_count() >= slots
                    and (not need_required or bool(zero_available & required))
                ):
                    return True
            for boundary in self.boundaries:
                if boundary not in charges_by_boundary:
                    charges_by_boundary[boundary] = self._fixed_charges(terms, boundary)
                charges = charges_by_boundary[boundary]
                if not charges:
                    continue
                eligible = available & self.boundary_masks[boundary]
                if eligible.bit_count() < slots or (need_required and not eligible & required):
                    continue
                bits = self._suffix_bits(family, boundary, eligible, remaining, slots, need_required)
                if any(((bits >> (self.offset - q)) & 1) for q in charges if -self.offset <= q <= self.offset):
                    return True
        return False

    def _suffix_bits_impl(self, family: str, boundary: int, available: int, atoms: int, slots: int, need_required: bool) -> int:
        if slots == 0:
            return (1 << self.offset) if atoms == 0 and not need_required else 0
        if atoms < slots or available.bit_count() < slots:
            return 0
        required = self.required_masks[family]
        if need_required and not available & required:
            return 0
        if slots == 1:
            bits, remaining_elements = 0, available & required if need_required else available
            while remaining_elements:
                bit = remaining_elements & -remaining_elements
                index = bit.bit_length() - 1
                for q in self.choices[boundary][index]:
                    location = self.offset + atoms * q
                    if 0 <= location <= 2 * self.offset:
                        bits |= 1 << location
                remaining_elements ^= bit
            return bits
        bit = available & -available
        index, tail = bit.bit_length() - 1, available ^ bit
        result = self._suffix_bits(family, boundary, tail, atoms, slots, need_required)
        next_need = bool(need_required and not bit & required)
        for n in range(1, atoms - slots + 2):
            suffix = self._suffix_bits(family, boundary, tail, atoms - n, slots - 1, next_need)
            if not suffix:
                continue
            for q in self.choices[boundary][index]:
                delta = q * n
                result |= ((suffix << delta) & self.bit_mask) if delta >= 0 else (suffix >> -delta)
        return result


def formula_phase(text: str, oracle: C3FDFormulaOracle) -> tuple[str, str]:
    """Recognize only the actual formula field, not earlier newlines."""
    position = text.lower().find(_LABEL)
    if position < 0:
        return "seek", ""
    value = text[position + len(_LABEL):]
    if value and not value.startswith(" "):
        return "invalid", value
    if "\n" in value:
        formula = value.split("\n", 1)[0].strip(" \t\r")
        return ("done" if oracle.is_terminal_valid(formula) else "invalid"), formula
    chemical = value.lstrip(" \t")
    if chemical.endswith((" ", "\t", "\r")):
        return ("active" if oracle.is_terminal_valid(chemical.rstrip(" \t\r")) else "invalid"), chemical
    return "active", chemical


class R03C3FDFormulaLogitsProcessor:
    """Formula-only legality, preserving the native no-prefill rich prompt.

    Unrelated pre-formula generation and all completed rich-field rows pass
    through. Candidate tokens crossing the formula label or newline are checked
    before committing. Empty support produces explicit EOS failure, never a
    replacement formula or fabricated scientific field.
    """

    def __init__(self, tokenizer: Any, *, oracle: C3FDFormulaOracle, start_length: int, eos_token_id: int) -> None:
        self.tokenizer, self.oracle = tokenizer, oracle
        self.start_length, self.eos_token_id = int(start_length), int(eos_token_id)
        self.fragments = {i: str(tokenizer.decode([i], skip_special_tokens=True, clean_up_tokenization_spaces=False))
                          for i in range(len(tokenizer)) if i != self.eos_token_id}
        self.active_ids = tuple(i for i, text in self.fragments.items()
                                if text and text[0] in " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\t\r\n")
        self.embedded_label_ids = tuple(i for i, text in self.fragments.items() if _LABEL in text.lower())
        self.crossing_ids = {k: tuple(i for i, text in self.fragments.items() if text.lower().startswith(_LABEL[k:]))
                             for k in range(1, len(_LABEL))}
        self.cache: dict[str, tuple[int, ...] | None] = {}
        self.failures: dict[str, str] = {}

    def _candidate_valid(self, text: str) -> bool:
        phase, chemical = formula_phase(text, self.oracle)
        if phase in ("seek", "done"):
            return True
        if phase == "invalid":
            return False
        if chemical.endswith((" ", "\t", "\r")):
            return self.oracle.is_terminal_valid(chemical.rstrip(" \t\r"))
        return self.oracle.is_prefix_viable(chemical)

    def allowed_token_ids(self, generated_ids: Sequence[int]) -> tuple[int, ...] | None:
        current = str(self.tokenizer.decode(list(map(int, generated_ids)), skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False))
        if current in self.cache:
            return self.cache[current]
        phase, _ = formula_phase(current, self.oracle)
        if phase == "done":
            return None
        if phase == "invalid":
            self.failures[current] = "invalid_committed_formula"
            return (self.eos_token_id,)
        if phase == "seek":
            candidates = set(self.embedded_label_ids)
            lowered = current.lower()
            for k in range(1, len(_LABEL)):
                if lowered.endswith(_LABEL[:k]):
                    candidates.update(self.crossing_ids[k])
            invalid = {i for i in candidates if not self._candidate_valid(current + self.fragments[i])}
            allowed = None if not invalid else tuple(i for i in range(len(self.tokenizer)) if i not in invalid)
        else:
            allowed = tuple(i for i in self.active_ids if self._candidate_valid(current + self.fragments[i]))
            if not allowed:
                self.failures[current] = "no_legal_formula_token"
                allowed = (self.eos_token_id,)
        self.cache[current] = allowed
        return allowed

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch
        output = scores.clone()
        for row in range(int(input_ids.shape[0])):
            allowed = self.allowed_token_ids(input_ids[row, self.start_length:].tolist())
            if allowed is None:
                continue
            masked = torch.full_like(output[row], -math.inf)
            indexes = torch.tensor(allowed, dtype=torch.long, device=output.device)
            masked[indexes] = output[row, indexes]
            output[row] = masked
        return output
