"""Chemistry-first formula planning for constrained Wyckoff generation.

The planner emits one compact primitive-cell composition before any geometric
tokens.  The body decoder then consumes that plan exactly through Wyckoff
primitive multiplicities.  Neither stage repairs, retries, or reranks output.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache, reduce
from typing import Any, Callable, Literal, Sequence

from crystal_dlm.composition_validity import classify_smact_validity

from ..bridge import ChartCatalog
from ..charts import LatticeChartCodec
from ..state import StratifiedState
from ..vocabulary import atomic_number_to_input_id, target_to_atomic_number
from .constrained import grammar_token_fragments
from .wq_text import (
    GrammarViolation,
    ProposalGrammarState,
    _decode_hex,
    _encode_hex,
    crystal_system_for_space_group,
    decode_free_coordinate,
)


FORMULA_PLAN_SYSTEM_PROMPT = (
    "Return one MP20 primitive-cell formula plan in the registered grammar."
)
FORMULA_PLAN_USER_PROMPT = (
    "Plan the complete composition before geometry. Return only the plan."
)
FORMULA_BODY_SYSTEM_PROMPT = (
    "Return one crystal record that exactly matches the supplied formula plan."
)
FORMULA_PLAN_PATTERN = re.compile(
    r"^F=E=([1-9][0-9]*),N=([1-9][0-9]*);"
    r"(?:(?:E=[1-9][0-9]*,N=[1-9][0-9]*;))*END$"
)


@dataclasses.dataclass(frozen=True, slots=True)
class FormulaPlan:
    """Canonical MP20 species-code/count plan for one primitive cell."""

    entries: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise GrammarViolation("formula plan must contain at least one species")
        previous = 0
        total = 0
        for raw_code, raw_count in self.entries:
            code = int(raw_code)
            count = int(raw_count)
            if not 1 <= code <= 89:
                raise GrammarViolation("formula-plan species code must be in [1,89]")
            if code <= previous:
                raise GrammarViolation(
                    "formula-plan species codes must be unique and increasing"
                )
            if count <= 0:
                raise GrammarViolation("formula-plan counts must be positive")
            total += count
            if total > 20:
                raise GrammarViolation("formula plan exceeds the MP20 atom limit")
            previous = code

    @property
    def total_atoms(self) -> int:
        return sum(count for _, count in self.entries)

    @property
    def species_codes(self) -> tuple[int, ...]:
        return tuple(code for code, _ in self.entries)

    @property
    def atomic_numbers(self) -> tuple[int, ...]:
        return tuple(target_to_atomic_number(code - 1) for code in self.species_codes)

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple(count for _, count in self.entries)

    @property
    def composition_valid(self) -> bool:
        return _entries_are_composition_valid(self.entries)

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(serialize_formula_plan(self).encode("utf-8")).hexdigest()

    def as_dict(self, *, include_composition_valid: bool = True) -> dict[str, Any]:
        payload = {
            "text": serialize_formula_plan(self),
            "species_codes": list(self.species_codes),
            "atomic_numbers": list(self.atomic_numbers),
            "counts": list(self.counts),
            "total_atoms": self.total_atoms,
        }
        if include_composition_valid:
            payload["composition_valid"] = self.composition_valid
        return payload


def serialize_formula_plan(plan: FormulaPlan) -> str:
    fields = [f"E={code},N={count}" for code, count in plan.entries]
    return "F=" + ";".join(fields) + ";END"


def parse_formula_plan(text: str) -> FormulaPlan:
    if not text or text != text.strip() or any(value.isspace() for value in text):
        raise GrammarViolation("formula plan must be one whitespace-free byte string")
    if FORMULA_PLAN_PATTERN.fullmatch(text) is None:
        raise GrammarViolation("formula plan does not match F=E=<code>,N=<count>;...;END")
    fields = text[2:-4].split(";")
    entries: list[tuple[int, int]] = []
    for field in fields:
        match = re.fullmatch(r"E=([1-9][0-9]*),N=([1-9][0-9]*)", field)
        if match is None:
            raise GrammarViolation("formula plan contains a malformed entry")
        entries.append((int(match.group(1)), int(match.group(2))))
    plan = FormulaPlan(tuple(entries))
    if serialize_formula_plan(plan) != text:
        raise GrammarViolation("formula plan is not canonical")
    return plan


def formula_plan_from_state(state: StratifiedState) -> FormulaPlan:
    counts: Counter[int] = Counter()
    for orbit in state.orbits:
        code = atomic_number_to_input_id(int(orbit.species))
        counts[int(code)] += int(orbit.primitive_multiplicity)
    return FormulaPlan(tuple(sorted((code, int(count)) for code, count in counts.items())))


def formula_plan_matches_state(plan: FormulaPlan, state: StratifiedState) -> bool:
    return formula_plan_from_state(state) == plan


def formula_body_user_prompt(plan: FormulaPlan | str) -> str:
    value = parse_formula_plan(plan) if isinstance(plan, str) else plan
    return f"FORMULA={serialize_formula_plan(value)};Return only the record."


@lru_cache(maxsize=200_000)
def _entries_are_composition_valid(entries: tuple[tuple[int, int], ...]) -> bool:
    atomic_numbers = tuple(target_to_atomic_number(code - 1) for code, _ in entries)
    raw_counts = tuple(count for _, count in entries)
    divisor = reduce(math.gcd, raw_counts)
    counts = tuple(count // divisor for count in raw_counts)
    classification = classify_smact_validity(atomic_numbers, counts)
    return bool(classification["valid"])


PlanValidity = Callable[[tuple[tuple[int, int], ...]], bool]
PlanCursorStage = Literal["prefix", "species", "count", "next_or_end", "terminal"]


def _plan_count_keeps_support(
    entries: tuple[tuple[int, int], ...],
    pending_species: int,
    count: int,
    validity: PlanValidity,
) -> bool:
    """Reject an invalid entry that would immediately exhaust plan support."""

    candidate = (*entries, (int(pending_species), int(count)))
    if validity(candidate):
        return True
    total = sum(value for _, value in candidate)
    return total < 20 and int(pending_species) < 89


@dataclasses.dataclass(frozen=True, slots=True)
class FormulaPlanTextCursor:
    """Semantic prefix cursor for the compact formula-plan grammar."""

    stage: PlanCursorStage = "prefix"
    option_prefix: str = ""
    entries: tuple[tuple[int, int], ...] = ()
    pending_species: int | None = None

    @property
    def terminal(self) -> bool:
        return self.stage == "terminal"

    @property
    def total_atoms(self) -> int:
        return sum(count for _, count in self.entries)

    def options(self, validity: PlanValidity = _entries_are_composition_valid) -> tuple[str, ...]:
        if self.stage == "prefix":
            return ("F=E=",)
        if self.stage == "species":
            lower = 1 if not self.entries else self.entries[-1][0] + 1
            return tuple(f"{code},N=" for code in range(lower, 90))
        if self.stage == "count":
            if self.pending_species is None:
                raise GrammarViolation("formula-plan count cursor lacks a species")
            return tuple(
                f"{count};"
                for count in range(1, 21 - self.total_atoms)
                if _plan_count_keeps_support(
                    self.entries,
                    self.pending_species,
                    count,
                    validity,
                )
            )
        if self.stage == "next_or_end":
            result: list[str] = []
            last_code = self.entries[-1][0]
            remaining = 20 - self.total_atoms
            # Invalid formulas are never accepted merely because MP20 or
            # species-code support is exhausted.
            if validity(self.entries):
                result.append("END")
            if remaining > 0 and last_code < 89:
                result.append("E=")
            return tuple(result)
        return ()

    def _commit(
        self,
        option: str,
        validity: PlanValidity,
    ) -> "FormulaPlanTextCursor":
        del validity
        if self.stage == "prefix":
            if option != "F=E=":
                raise GrammarViolation("invalid formula-plan prefix")
            return FormulaPlanTextCursor(stage="species")
        if self.stage == "species":
            match = re.fullmatch(r"([1-9][0-9]*),N=", option)
            if match is None:
                raise GrammarViolation("invalid formula-plan species macro")
            return dataclasses.replace(
                self,
                stage="count",
                option_prefix="",
                pending_species=int(match.group(1)),
            )
        if self.stage == "count":
            match = re.fullmatch(r"([1-9][0-9]*);", option)
            if match is None or self.pending_species is None:
                raise GrammarViolation("invalid formula-plan count macro")
            count = int(match.group(1))
            entries = (*self.entries, (self.pending_species, count))
            FormulaPlan(entries)
            return FormulaPlanTextCursor(stage="next_or_end", entries=entries)
        if self.stage == "next_or_end":
            if option == "END":
                return FormulaPlanTextCursor(stage="terminal", entries=self.entries)
            if option == "E=":
                return FormulaPlanTextCursor(stage="species", entries=self.entries)
            raise GrammarViolation("invalid formula-plan continuation")
        raise GrammarViolation("cannot commit bytes after formula-plan END")

    def feed(
        self,
        fragment: str,
        validity: PlanValidity = _entries_are_composition_valid,
    ) -> "FormulaPlanTextCursor":
        cursor = self
        for character in fragment:
            if cursor.terminal:
                raise GrammarViolation("token contains bytes after formula-plan END")
            candidate = cursor.option_prefix + character
            options = cursor.options(validity)
            matching = tuple(value for value in options if value.startswith(candidate))
            if not matching:
                raise GrammarViolation(
                    f"formula-plan prefix left support at {cursor.stage}: {candidate!r}"
                )
            complete = next((value for value in matching if value == candidate), None)
            if complete is None:
                cursor = dataclasses.replace(cursor, option_prefix=candidate)
            else:
                cursor = cursor._commit(complete, validity)
        return cursor

    def signature(self) -> tuple[Any, ...]:
        return (
            self.stage,
            self.option_prefix,
            self.entries,
            self.pending_species,
        )


class FormulaPlanTokenConstraint:
    """Tokenizer-level exact support for a chemistry-aware formula plan."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        prompt_width: int,
        token_fragments: Sequence[tuple[int, str]] | None = None,
        validity: PlanValidity = _entries_are_composition_valid,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_width = int(prompt_width)
        if self.prompt_width <= 0 or tokenizer.eos_token_id is None:
            raise ValueError("positive prompt width and EOS token are required")
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.validity = validity
        fragments = tuple(
            grammar_token_fragments(tokenizer)
            if token_fragments is None
            else token_fragments
        )
        by_first: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for token_id, text in fragments:
            by_first[text[0]].append((int(token_id), str(text)))
        self.tokens_by_first = {key: tuple(value) for key, value in by_first.items()}
        self._cursor_by_tokens: dict[tuple[int, ...], FormulaPlanTextCursor] = {
            (): FormulaPlanTextCursor()
        }
        self._allowed_by_signature: dict[tuple[Any, ...], tuple[int, ...]] = {}

    def _cursor(self, generated: tuple[int, ...]) -> FormulaPlanTextCursor:
        cached = self._cursor_by_tokens.get(generated)
        if cached is not None:
            return cached
        parent = self._cursor(generated[:-1])
        fragment = self.tokenizer.decode(
            [generated[-1]],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        cursor = parent.feed(fragment, self.validity)
        self._cursor_by_tokens[generated] = cursor
        return cursor

    def __call__(self, batch_id: int, input_ids: Any) -> list[int]:
        del batch_id
        generated = tuple(int(value) for value in input_ids.tolist()[self.prompt_width :])
        cursor = self._cursor(generated)
        if cursor.terminal:
            return [self.eos_token_id]
        signature = cursor.signature()
        cached = self._allowed_by_signature.get(signature)
        if cached is None:
            first = {
                option[len(cursor.option_prefix)]
                for option in cursor.options(self.validity)
                if len(option) > len(cursor.option_prefix)
            }
            allowed: list[int] = []
            for character in sorted(first):
                for token_id, fragment in self.tokens_by_first.get(character, ()):
                    try:
                        cursor.feed(fragment, self.validity)
                    except GrammarViolation:
                        continue
                    allowed.append(token_id)
            if not allowed:
                raise RuntimeError("tokenizer has no legal formula-plan continuation")
            cached = tuple(sorted(set(allowed)))
            self._allowed_by_signature[signature] = cached
        return list(cached)


PlannedCursorStage = Literal[
    "space_group",
    "lattice_code",
    "wyckoff",
    "species",
    "free_code",
    "orbit_or_stop",
    "terminal",
]


@lru_cache(maxsize=32_768)
def _count_reachable_by_multiplicities(
    count: int,
    multiplicities: tuple[int, ...],
) -> bool:
    """Unbounded exact coin-change reachability for one species count."""

    target = int(count)
    values = tuple(sorted({int(value) for value in multiplicities if int(value) > 0}))
    if target < 0 or not values:
        return False
    reachable = [False] * (target + 1)
    reachable[0] = True
    for subtotal in range(1, target + 1):
        reachable[subtotal] = any(
            value <= subtotal and reachable[subtotal - value] for value in values
        )
    return bool(reachable[target])


def _space_group_multiplicities(
    catalog: ChartCatalog,
    space_group: int,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(catalog.get(int(space_group), int(value)).primitive_multiplicity)
                for value in catalog.types(int(space_group))
            }
        )
    )


def _remaining_is_reachable(
    remaining: tuple[tuple[int, int], ...],
    multiplicities: tuple[int, ...],
) -> bool:
    return all(
        _count_reachable_by_multiplicities(int(count), multiplicities)
        for _, count in remaining
    )


def _remaining_after(
    remaining: tuple[tuple[int, int], ...],
    *,
    species_code: int,
    multiplicity: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            int(code),
            int(count) - int(multiplicity)
            if int(code) == int(species_code)
            else int(count),
        )
        for code, count in remaining
    )


def _reachable_space_groups(
    plan: FormulaPlan,
    catalog: ChartCatalog,
) -> tuple[int, ...]:
    return tuple(
        space_group
        for space_group in range(1, 231)
        if _remaining_is_reachable(
            plan.entries,
            _space_group_multiplicities(catalog, space_group),
        )
    )


@dataclasses.dataclass(frozen=True, slots=True)
class PlannedProposalTextCursor:
    """Proposal cursor that consumes one frozen formula plan exactly."""

    plan: FormulaPlan
    remaining: tuple[tuple[int, int], ...]
    semantic: ProposalGrammarState = dataclasses.field(default_factory=ProposalGrammarState)
    stage: PlannedCursorStage = "space_group"
    option_prefix: str = ""
    lattice_codes: tuple[str, ...] = ()
    lattice_index: int = 0
    pending_wyckoff: int | None = None
    pending_species: int | None = None
    free_codes: tuple[str, ...] = ()
    free_index: int = 0
    reachable_space_groups: tuple[int, ...] | None = None
    space_group_multiplicities: tuple[int, ...] = ()

    @classmethod
    def start(
        cls,
        plan: FormulaPlan,
        catalog: ChartCatalog | None = None,
    ) -> "PlannedProposalTextCursor":
        reachable = None if catalog is None else _reachable_space_groups(plan, catalog)
        return cls(
            plan=plan,
            remaining=plan.entries,
            reachable_space_groups=reachable,
        )

    @property
    def terminal(self) -> bool:
        return self.stage == "terminal"

    @property
    def exhausted(self) -> bool:
        return all(count == 0 for _, count in self.remaining)

    def legal_space_groups(self, catalog: ChartCatalog) -> tuple[int, ...]:
        if self.reachable_space_groups is not None:
            return self.reachable_space_groups
        return _reachable_space_groups(self.plan, catalog)

    def _multiplicities(self, catalog: ChartCatalog) -> tuple[int, ...]:
        if self.space_group_multiplicities:
            return self.space_group_multiplicities
        if self.semantic.space_group is None:
            return ()
        return _space_group_multiplicities(catalog, self.semantic.space_group)

    def legal_wyckoff_types(self, catalog: ChartCatalog) -> tuple[int, ...]:
        legal = self.semantic.legal_wyckoff_types(catalog)
        multiplicities = self._multiplicities(catalog)
        return tuple(
            value
            for value in legal
            if any(
                count >= primitive_multiplicity
                and _remaining_is_reachable(
                    _remaining_after(
                        self.remaining,
                        species_code=code,
                        multiplicity=primitive_multiplicity,
                    ),
                    multiplicities,
                )
                for code, count in self.remaining
                for primitive_multiplicity in (
                    int(
                        catalog.get(
                            self.semantic.space_group,
                            value,
                        ).primitive_multiplicity
                    ),
                )
            )
        )

    def options(self, catalog: ChartCatalog) -> tuple[str, ...]:
        if self.stage == "space_group":
            return tuple(
                f"SG={value};Q=" for value in self.legal_space_groups(catalog)
            )
        if self.stage == "lattice_code":
            assert self.semantic.space_group is not None
            dimension = LatticeChartCodec.dimension(
                crystal_system_for_space_group(self.semantic.space_group)
            )
            suffix = "," if self.lattice_index < dimension - 1 else ";O=0,W="
            return tuple(_encode_hex(value) + suffix for value in range(256))
        if self.stage == "wyckoff":
            return tuple(f"{value},E=" for value in self.legal_wyckoff_types(catalog))
        if self.stage == "species":
            if self.semantic.space_group is None or self.pending_wyckoff is None:
                raise GrammarViolation("planned species cursor lacks a Wyckoff type")
            multiplicity = int(
                catalog.get(
                    self.semantic.space_group, self.pending_wyckoff
                ).primitive_multiplicity
            )
            multiplicities = self._multiplicities(catalog)
            return tuple(
                f"{code},U="
                for code, count in self.remaining
                if count >= multiplicity
                and _remaining_is_reachable(
                    _remaining_after(
                        self.remaining,
                        species_code=code,
                        multiplicity=multiplicity,
                    ),
                    multiplicities,
                )
            )
        if self.stage == "free_code":
            if self.semantic.space_group is None or self.pending_wyckoff is None:
                raise GrammarViolation("planned free-coordinate cursor is incomplete")
            dimension = catalog.get(
                self.semantic.space_group, self.pending_wyckoff
            ).dimension
            if dimension == 0:
                return ("-;",)
            suffix = "," if self.free_index < dimension - 1 else ";"
            return tuple(_encode_hex(value) + suffix for value in range(256))
        if self.stage == "orbit_or_stop":
            result: list[str] = []
            if self.exhausted:
                result.append("STOP")
            elif self.legal_wyckoff_types(catalog):
                result.append(f"O={self.semantic.orbit_count},W=")
            return tuple(result)
        return ()

    def _commit(
        self,
        option: str,
        catalog: ChartCatalog,
    ) -> "PlannedProposalTextCursor":
        if self.stage == "space_group":
            match = re.fullmatch(r"SG=([1-9][0-9]*);Q=", option)
            if match is None:
                raise GrammarViolation("invalid planned SG macro")
            space_group = int(match.group(1))
            if space_group not in self.legal_space_groups(catalog):
                raise GrammarViolation(
                    "space group cannot realize every planned species count"
                )
            return dataclasses.replace(
                self,
                semantic=self.semantic.accept_space_group(space_group),
                stage="lattice_code",
                option_prefix="",
                space_group_multiplicities=_space_group_multiplicities(
                    catalog,
                    space_group,
                ),
            )
        if self.stage == "lattice_code":
            code = option[:2]
            _decode_hex(code)
            codes = (*self.lattice_codes, code)
            assert self.semantic.space_group is not None
            dimension = LatticeChartCodec.dimension(
                crystal_system_for_space_group(self.semantic.space_group)
            )
            if self.lattice_index + 1 == dimension:
                return dataclasses.replace(
                    self,
                    semantic=self.semantic.accept_lattice(codes),
                    stage="wyckoff",
                    option_prefix="",
                    lattice_codes=codes,
                )
            return dataclasses.replace(
                self,
                option_prefix="",
                lattice_codes=codes,
                lattice_index=self.lattice_index + 1,
            )
        if self.stage == "wyckoff":
            match = re.fullmatch(r"([0-9]+),E=", option)
            if match is None:
                raise GrammarViolation("invalid planned Wyckoff macro")
            return dataclasses.replace(
                self,
                stage="species",
                option_prefix="",
                pending_wyckoff=int(match.group(1)),
            )
        if self.stage == "species":
            match = re.fullmatch(r"([1-9][0-9]*),U=", option)
            if match is None:
                raise GrammarViolation("invalid planned species macro")
            return dataclasses.replace(
                self,
                stage="free_code",
                option_prefix="",
                pending_species=int(match.group(1)),
                free_codes=(),
                free_index=0,
            )
        if self.stage == "free_code":
            if (
                self.semantic.space_group is None
                or self.pending_wyckoff is None
                or self.pending_species is None
            ):
                raise GrammarViolation("incomplete planned orbit payload")
            spec = catalog.get(self.semantic.space_group, self.pending_wyckoff)
            if int(spec.dimension) == 0:
                codes: tuple[str, ...] = ()
            else:
                code = option[:2]
                _decode_hex(code)
                codes = (*self.free_codes, code)
                if self.free_index + 1 < int(spec.dimension):
                    return dataclasses.replace(
                        self,
                        option_prefix="",
                        free_codes=codes,
                        free_index=self.free_index + 1,
                    )
            semantic = self.semantic.accept_orbit(
                wyckoff_type=self.pending_wyckoff,
                species=target_to_atomic_number(self.pending_species - 1),
                free_codes=codes,
                catalog=catalog,
            )
            multiplicity = int(spec.primitive_multiplicity)
            remaining = _remaining_after(
                self.remaining,
                species_code=self.pending_species,
                multiplicity=multiplicity,
            )
            if any(count < 0 for _, count in remaining):
                raise GrammarViolation("planned orbit exceeded a remaining count")
            if not _remaining_is_reachable(
                remaining,
                self._multiplicities(catalog),
            ):
                raise GrammarViolation(
                    "planned orbit left an unreachable species-count remainder"
                )
            return dataclasses.replace(
                self,
                semantic=semantic,
                remaining=remaining,
                stage="orbit_or_stop",
                option_prefix="",
            )
        if self.stage == "orbit_or_stop":
            if option == "STOP":
                if not self.exhausted:
                    raise GrammarViolation("planned proposal stopped before exhausting plan")
                return dataclasses.replace(
                    self,
                    semantic=self.semantic.accept_stop(),
                    stage="terminal",
                    option_prefix="",
                )
            if option != f"O={self.semantic.orbit_count},W=":
                raise GrammarViolation("invalid planned next-orbit macro")
            return dataclasses.replace(
                self,
                stage="wyckoff",
                option_prefix="",
                pending_wyckoff=None,
                pending_species=None,
                free_codes=(),
                free_index=0,
            )
        raise GrammarViolation("cannot commit bytes after planned STOP")

    def feed(
        self,
        fragment: str,
        catalog: ChartCatalog,
    ) -> "PlannedProposalTextCursor":
        cursor = self
        for character in fragment:
            if cursor.terminal:
                raise GrammarViolation("token contains bytes after planned STOP")
            candidate = cursor.option_prefix + character
            options = cursor.options(catalog)
            matching = tuple(value for value in options if value.startswith(candidate))
            if not matching:
                raise GrammarViolation(
                    f"planned proposal left support at {cursor.stage}: {candidate!r}"
                )
            complete = next((value for value in matching if value == candidate), None)
            if complete is None:
                cursor = dataclasses.replace(cursor, option_prefix=candidate)
            else:
                cursor = cursor._commit(complete, catalog)
        return cursor

    def signature(self) -> tuple[Any, ...]:
        return (
            self.stage,
            self.option_prefix,
            self.semantic.phase,
            self.semantic.space_group,
            self.semantic.atom_count,
            self.semantic.orbit_count,
            self.remaining,
            self.lattice_index,
            self.pending_wyckoff,
            self.pending_species,
            self.free_index,
        )


class PlannedProposalTokenConstraint:
    """Tokenizer-level exact WQ support conditioned on a formula plan."""

    def __init__(
        self,
        tokenizer: Any,
        catalog: ChartCatalog,
        plan: FormulaPlan,
        *,
        prompt_width: int,
        token_fragments: Sequence[tuple[int, str]] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.catalog = catalog
        self.plan = plan
        self.prompt_width = int(prompt_width)
        if self.prompt_width <= 0 or tokenizer.eos_token_id is None:
            raise ValueError("positive prompt width and EOS token are required")
        self.eos_token_id = int(tokenizer.eos_token_id)
        fragments = tuple(
            grammar_token_fragments(tokenizer)
            if token_fragments is None
            else token_fragments
        )
        by_first: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for token_id, text in fragments:
            by_first[text[0]].append((int(token_id), str(text)))
        self.tokens_by_first = {key: tuple(value) for key, value in by_first.items()}
        self._cursor_by_tokens: dict[tuple[int, ...], PlannedProposalTextCursor] = {
            (): PlannedProposalTextCursor.start(plan, catalog)
        }
        self._allowed_by_signature: dict[tuple[Any, ...], tuple[int, ...]] = {}

    def _cursor(self, generated: tuple[int, ...]) -> PlannedProposalTextCursor:
        cached = self._cursor_by_tokens.get(generated)
        if cached is not None:
            return cached
        parent = self._cursor(generated[:-1])
        fragment = self.tokenizer.decode(
            [generated[-1]],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        cursor = parent.feed(fragment, self.catalog)
        self._cursor_by_tokens[generated] = cursor
        return cursor

    def __call__(self, batch_id: int, input_ids: Any) -> list[int]:
        del batch_id
        generated = tuple(int(value) for value in input_ids.tolist()[self.prompt_width :])
        cursor = self._cursor(generated)
        if cursor.terminal:
            return [self.eos_token_id]
        signature = cursor.signature()
        cached = self._allowed_by_signature.get(signature)
        if cached is None:
            first = {
                option[len(cursor.option_prefix)]
                for option in cursor.options(self.catalog)
                if len(option) > len(cursor.option_prefix)
            }
            allowed: list[int] = []
            for character in sorted(first):
                for token_id, fragment in self.tokens_by_first.get(character, ()):
                    try:
                        cursor.feed(fragment, self.catalog)
                    except GrammarViolation:
                        continue
                    allowed.append(token_id)
            if not allowed:
                raise RuntimeError("tokenizer has no legal planned-proposal continuation")
            cached = tuple(sorted(set(allowed)))
            self._allowed_by_signature[signature] = cached
        return list(cached)
