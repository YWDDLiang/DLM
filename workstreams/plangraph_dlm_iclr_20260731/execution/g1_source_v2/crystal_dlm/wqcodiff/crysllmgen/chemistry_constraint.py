"""Chemistry-aware termination support for constrained WQ proposals.

This module changes only whether ``STOP`` is currently legal.  It never
repairs, retries, reranks, or replaces a proposal.  A proposal whose current
composition has no legacy charge-neutral assignment must continue with another
legal Wyckoff orbit while atom-count support remains.  Pauling-only,
oxidation-state-missing, all-metal, and single-element cases remain soft.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections import Counter
from functools import reduce
from typing import Any, Callable, Mapping, Sequence

from crystal_dlm.composition_validity import classify_smact_validity

from ..bridge import ChartCatalog
from ..vocabulary import target_to_atomic_number
from .constrained import (
    GrammarViolation,
    ProposalTextCursor,
    ProposalTokenConstraint,
)


CompositionClassifier = Callable[
    [Sequence[int], Sequence[int]],
    Mapping[str, Any],
]

_COMPLETE_ORBIT = re.compile(
    r"O=[0-9]+,W=([0-9]+),E=([1-9][0-9]*),"
    r"U=(?:-|[0-9A-F]{2}(?:,[0-9A-F]{2})*);"
)


@dataclasses.dataclass(frozen=True, slots=True)
class ChargeStopDecision:
    raw_counts: tuple[tuple[int, int], ...]
    reduced_counts: tuple[tuple[int, int], ...]
    atom_count: int
    orbit_count: int
    reason: str
    stop_deferred: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_counts": [
                {"atomic_number": atomic_number, "count": count}
                for atomic_number, count in self.raw_counts
            ],
            "reduced_counts": [
                {"atomic_number": atomic_number, "count": count}
                for atomic_number, count in self.reduced_counts
            ],
            "atom_count": self.atom_count,
            "orbit_count": self.orbit_count,
            "reason": self.reason,
            "stop_deferred": self.stop_deferred,
        }


def _reduce_counts(
    raw_counts: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    if not raw_counts:
        return ()
    divisor = reduce(math.gcd, (int(count) for _, count in raw_counts))
    divisor = max(divisor, 1)
    return tuple(
        (int(atomic_number), int(count) // divisor)
        for atomic_number, count in raw_counts
    )


def completed_orbit_composition(
    proposal_prefix: str,
    *,
    space_group: int,
    catalog: ChartCatalog,
) -> tuple[tuple[int, int], ...]:
    """Return exact primitive composition from completed orbit macros."""

    counts: Counter[int] = Counter()
    for match in _COMPLETE_ORBIT.finditer(str(proposal_prefix)):
        wyckoff_type = int(match.group(1))
        species_code = int(match.group(2))
        if not 1 <= species_code <= 89:
            raise GrammarViolation("completed orbit has an invalid species code")
        spec = catalog.get(int(space_group), wyckoff_type)
        counts[target_to_atomic_number(species_code - 1)] += int(
            spec.primitive_multiplicity
        )
    return tuple((value, int(counts[value])) for value in sorted(counts))


class ChargeAwareStopConstraint(ProposalTokenConstraint):
    """Defer only charge-neutrality-invalid STOP decisions.

    The base finite grammar remains unchanged.  The policy is evaluated before
    the first byte of a ``STOP`` macro, including when a tokenizer token spans
    an orbit terminator and the subsequent STOP macro.
    """

    def __init__(
        self,
        tokenizer: Any,
        catalog: ChartCatalog,
        *,
        prompt_width: int,
        token_fragments: Sequence[tuple[int, str]] | None = None,
        classifier: CompositionClassifier = classify_smact_validity,
    ) -> None:
        super().__init__(
            tokenizer,
            catalog,
            prompt_width=prompt_width,
            token_fragments=token_fragments,
        )
        self.classifier = classifier
        self._classification_cache: dict[
            tuple[tuple[int, int], ...], Mapping[str, Any]
        ] = {}
        self._decisions: dict[
            tuple[tuple[tuple[int, int], ...], int, int],
            ChargeStopDecision,
        ] = {}
        self._classifier_evaluations = 0

    def _decision(
        self,
        proposal_prefix: str,
        cursor: ProposalTextCursor,
    ) -> ChargeStopDecision:
        if (
            cursor.stage != "orbit_or_stop"
            or cursor.option_prefix
            or cursor.semantic.space_group is None
            or cursor.semantic.orbit_count < 1
        ):
            raise ValueError("charge-aware STOP decision requested outside STOP support")
        raw = completed_orbit_composition(
            proposal_prefix,
            space_group=cursor.semantic.space_group,
            catalog=self.catalog,
        )
        if sum(count for _, count in raw) != cursor.semantic.atom_count:
            raise GrammarViolation(
                "completed orbit composition disagrees with grammar atom count"
            )
        reduced = _reduce_counts(raw)
        classification = self._classification_cache.get(reduced)
        if classification is None:
            classification = self.classifier(
                tuple(value for value, _ in reduced),
                tuple(count for _, count in reduced),
            )
            if (
                not isinstance(classification, Mapping)
                or "reason" not in classification
                or "valid" not in classification
            ):
                raise TypeError(
                    "composition classifier must return valid and reason fields"
                )
            self._classification_cache[reduced] = classification
            self._classifier_evaluations += 1
        reason = str(classification["reason"])
        has_future_orbit = bool(
            cursor.semantic.legal_wyckoff_types(self.catalog)
        )
        decision = ChargeStopDecision(
            raw_counts=raw,
            reduced_counts=reduced,
            atom_count=int(cursor.semantic.atom_count),
            orbit_count=int(cursor.semantic.orbit_count),
            reason=reason,
            stop_deferred=bool(
                reason == "charge_neutrality_fail" and has_future_orbit
            ),
        )
        decision_key = (
            raw,
            int(cursor.semantic.atom_count),
            int(cursor.semantic.orbit_count),
        )
        self._decisions.setdefault(decision_key, decision)
        return decision

    def _fragment_allowed(
        self,
        *,
        proposal_prefix: str,
        cursor: ProposalTextCursor,
        fragment: str,
    ) -> bool:
        text = str(proposal_prefix)
        probe = cursor
        for character in fragment:
            if (
                probe.stage == "orbit_or_stop"
                and not probe.option_prefix
                and character == "S"
                and self._decision(text, probe).stop_deferred
            ):
                return False
            try:
                probe = probe.feed(character, self.catalog)
            except GrammarViolation:
                return False
            text += character
        return True

    def __call__(self, batch_id: int, input_ids: Any) -> list[int]:
        del batch_id
        generated = tuple(
            int(value) for value in input_ids.tolist()[self.prompt_width :]
        )
        cursor = self._cursor(generated)
        if cursor.terminal:
            return [self.eos_token_id]
        proposal_prefix = self.tokenizer.decode(
            list(generated),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        first_characters = {
            option[len(cursor.option_prefix)]
            for option in cursor.options(self.catalog)
            if len(option) > len(cursor.option_prefix)
        }
        allowed: list[int] = []
        for character in sorted(first_characters):
            for token_id, fragment in self.tokens_by_first.get(character, ()):
                if self._fragment_allowed(
                    proposal_prefix=proposal_prefix,
                    cursor=cursor,
                    fragment=fragment,
                ):
                    allowed.append(token_id)
        if not allowed:
            raise RuntimeError(
                "charge-aware constraint has no legal tokenizer continuation"
            )
        return sorted(set(allowed))

    def diagnostics(self) -> dict[str, Any]:
        decisions = tuple(
            sorted(
                self._decisions.values(),
                key=lambda value: (
                    value.atom_count,
                    value.orbit_count,
                    value.reduced_counts,
                ),
            )
        )
        return {
            "schema": "wq_charge_aware_stop_constraint_diagnostics_v1",
            "policy": (
                "defer_STOP_only_for_charge_neutrality_fail_while_"
                "another_wyckoff_orbit_fits"
            ),
            "classifier_evaluations": self._classifier_evaluations,
            "unique_stop_decisions": len(decisions),
            "unique_stop_deferrals": sum(
                int(value.stop_deferred) for value in decisions
            ),
            "decisions": [value.to_dict() for value in decisions],
            "retry_or_replacement_used": False,
            "pauling_is_hard_constraint": False,
        }


__all__ = [
    "ChargeAwareStopConstraint",
    "ChargeStopDecision",
    "completed_orbit_composition",
]
