"""Tokenizer-aware finite-state constrained decoding for Wyckoff proposals."""

from __future__ import annotations

import dataclasses
import re
from collections import defaultdict
from typing import Any, Literal, Sequence

from ..bridge import ChartCatalog
from ..charts import LatticeChartCodec
from .wq_text import (
    HEX_BINS,
    GrammarViolation,
    ProposalGrammarState,
    _decode_hex,
    _encode_hex,
    crystal_system_for_space_group,
    serialize_topology_edit,
    TopologyEdit,
)
from ..state import StratifiedState
from ..vocabulary import MP20_ATOMIC_NUMBERS


CursorStage = Literal[
    "space_group",
    "lattice_code",
    "wyckoff",
    "species",
    "free_code",
    "orbit_or_stop",
    "terminal",
]


GRAMMAR_CHARACTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789=,;-")


def grammar_token_fragments(tokenizer: Any) -> tuple[tuple[int, str], ...]:
    """Decode the frozen ASCII grammar portion of a tokenizer vocabulary once."""

    eos = None if tokenizer.eos_token_id is None else int(tokenizer.eos_token_id)
    return tuple(
        (token_id, text)
        for token_id in range(len(tokenizer))
        if token_id != eos
        for text in (
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        )
        if text and all(character in GRAMMAR_CHARACTERS for character in text)
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ProposalTextCursor:
    """A prefix cursor whose options are prefix-free semantic macro strings."""

    semantic: ProposalGrammarState = dataclasses.field(default_factory=ProposalGrammarState)
    stage: CursorStage = "space_group"
    option_prefix: str = ""
    lattice_codes: tuple[str, ...] = ()
    lattice_index: int = 0
    pending_wyckoff: int | None = None
    pending_species: int | None = None
    free_codes: tuple[str, ...] = ()
    free_index: int = 0

    @property
    def terminal(self) -> bool:
        return self.stage == "terminal"

    def options(self, catalog: ChartCatalog) -> tuple[str, ...]:
        if self.stage == "space_group":
            return tuple(f"SG={value};Q=" for value in range(1, 231))
        if self.stage == "lattice_code":
            if self.semantic.space_group is None:
                raise GrammarViolation("lattice cursor has no space group")
            dimension = LatticeChartCodec.dimension(
                crystal_system_for_space_group(self.semantic.space_group)
            )
            suffix = "," if self.lattice_index < dimension - 1 else ";O=0,W="
            return tuple(_encode_hex(value) + suffix for value in range(HEX_BINS))
        if self.stage == "wyckoff":
            return tuple(
                f"{value},E=" for value in self.semantic.legal_wyckoff_types(catalog)
            )
        if self.stage == "species":
            return tuple(f"{value},U=" for value in range(1, 90))
        if self.stage == "free_code":
            if self.semantic.space_group is None or self.pending_wyckoff is None:
                raise GrammarViolation("free-coordinate cursor has no target orbit")
            dimension = catalog.get(
                self.semantic.space_group, self.pending_wyckoff
            ).dimension
            if dimension == 0:
                return ("-;",)
            suffix = "," if self.free_index < dimension - 1 else ";"
            return tuple(_encode_hex(value) + suffix for value in range(HEX_BINS))
        if self.stage == "orbit_or_stop":
            result = ["STOP"]
            if self.semantic.legal_wyckoff_types(catalog):
                result.append(f"O={self.semantic.orbit_count},W=")
            return tuple(result)
        return ()

    def _commit(self, option: str, catalog: ChartCatalog) -> "ProposalTextCursor":
        if self.stage == "space_group":
            match = re.fullmatch(r"SG=([1-9][0-9]*);Q=", option)
            if match is None:
                raise GrammarViolation("invalid committed SG macro")
            semantic = self.semantic.accept_space_group(int(match.group(1)))
            return ProposalTextCursor(semantic=semantic, stage="lattice_code")
        if self.stage == "lattice_code":
            code = option[:2]
            _decode_hex(code)
            codes = (*self.lattice_codes, code)
            assert self.semantic.space_group is not None
            dimension = LatticeChartCodec.dimension(
                crystal_system_for_space_group(self.semantic.space_group)
            )
            if self.lattice_index + 1 == dimension:
                semantic = self.semantic.accept_lattice(codes)
                return ProposalTextCursor(semantic=semantic, stage="wyckoff")
            return dataclasses.replace(
                self,
                option_prefix="",
                lattice_codes=codes,
                lattice_index=self.lattice_index + 1,
            )
        if self.stage == "wyckoff":
            match = re.fullmatch(r"([0-9]+),E=", option)
            if match is None:
                raise GrammarViolation("invalid committed Wyckoff macro")
            return dataclasses.replace(
                self,
                stage="species",
                option_prefix="",
                pending_wyckoff=int(match.group(1)),
            )
        if self.stage == "species":
            match = re.fullmatch(r"([1-9][0-9]*),U=", option)
            if match is None:
                raise GrammarViolation("invalid committed species macro")
            return dataclasses.replace(
                self,
                stage="free_code",
                option_prefix="",
                pending_species=int(match.group(1)),
                free_codes=(),
                free_index=0,
            )
        if self.stage == "free_code":
            if self.semantic.space_group is None or self.pending_wyckoff is None or self.pending_species is None:
                raise GrammarViolation("incomplete orbit payload")
            dimension = catalog.get(
                self.semantic.space_group, self.pending_wyckoff
            ).dimension
            if dimension == 0:
                codes: tuple[str, ...] = ()
            else:
                code = option[:2]
                _decode_hex(code)
                codes = (*self.free_codes, code)
                if self.free_index + 1 < dimension:
                    return dataclasses.replace(
                        self,
                        option_prefix="",
                        free_codes=codes,
                        free_index=self.free_index + 1,
                    )
            semantic = self.semantic.accept_orbit(
                wyckoff_type=self.pending_wyckoff,
                species=self.pending_species,
                free_codes=codes,
                catalog=catalog,
            )
            return ProposalTextCursor(semantic=semantic, stage="orbit_or_stop")
        if self.stage == "orbit_or_stop":
            if option == "STOP":
                return ProposalTextCursor(
                    semantic=self.semantic.accept_stop(),
                    stage="terminal",
                )
            expected = f"O={self.semantic.orbit_count},W="
            if option != expected:
                raise GrammarViolation("invalid committed next-orbit macro")
            return dataclasses.replace(
                self,
                stage="wyckoff",
                option_prefix="",
                pending_wyckoff=None,
                pending_species=None,
                free_codes=(),
                free_index=0,
            )
        raise GrammarViolation("cannot commit text after STOP")

    def feed(self, fragment: str, catalog: ChartCatalog) -> "ProposalTextCursor":
        cursor = self
        for character in fragment:
            if cursor.terminal:
                raise GrammarViolation("token contains bytes after STOP")
            candidate = cursor.option_prefix + character
            options = cursor.options(catalog)
            matching = tuple(value for value in options if value.startswith(candidate))
            if not matching:
                raise GrammarViolation(
                    f"text prefix is outside grammar at {cursor.stage}: {candidate!r}"
                )
            complete = next((value for value in matching if value == candidate), None)
            if complete is not None:
                cursor = cursor._commit(complete, catalog)
            else:
                cursor = dataclasses.replace(cursor, option_prefix=candidate)
        return cursor

    def signature(self) -> tuple[Any, ...]:
        """Allowed-token support identity; continuous byte history is irrelevant."""

        return (
            self.stage,
            self.option_prefix,
            self.semantic.phase,
            self.semantic.space_group,
            self.semantic.atom_count,
            self.semantic.orbit_count,
            self.lattice_index,
            self.pending_wyckoff,
            self.pending_species,
            self.free_index,
        )


class ProposalTokenConstraint:
    """Callable for Transformers ``prefix_allowed_tokens_fn``."""

    _GRAMMAR_CHARACTERS = GRAMMAR_CHARACTERS

    def __init__(
        self,
        tokenizer: Any,
        catalog: ChartCatalog,
        *,
        prompt_width: int,
        token_fragments: Sequence[tuple[int, str]] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.catalog = catalog
        self.prompt_width = int(prompt_width)
        if self.prompt_width <= 0 or tokenizer.eos_token_id is None:
            raise ValueError("positive prompt width and EOS token are required")
        self.eos_token_id = int(tokenizer.eos_token_id)
        by_first: dict[str, list[tuple[int, str]]] = defaultdict(list)
        fragments = (
            grammar_token_fragments(tokenizer)
            if token_fragments is None
            else tuple(token_fragments)
        )
        for token_id, text in fragments:
            by_first[text[0]].append((int(token_id), str(text)))
        self.tokens_by_first = {key: tuple(value) for key, value in by_first.items()}
        self._cursor_by_tokens: dict[tuple[int, ...], ProposalTextCursor] = {
            (): ProposalTextCursor()
        }
        self._allowed_by_signature: dict[tuple[Any, ...], tuple[int, ...]] = {}

    def _cursor(self, generated: tuple[int, ...]) -> ProposalTextCursor:
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
        values = tuple(int(value) for value in input_ids.tolist()[self.prompt_width :])
        cursor = self._cursor(values)
        if cursor.terminal:
            return [self.eos_token_id]
        signature = cursor.signature()
        cached = self._allowed_by_signature.get(signature)
        if cached is None:
            first_characters = {
                option[len(cursor.option_prefix)]
                for option in cursor.options(self.catalog)
                if len(option) > len(cursor.option_prefix)
            }
            allowed: list[int] = []
            for character in sorted(first_characters):
                for token_id, fragment in self.tokens_by_first.get(character, ()):
                    try:
                        cursor.feed(fragment, self.catalog)
                    except GrammarViolation:
                        continue
                    allowed.append(token_id)
            if not allowed:
                raise RuntimeError(f"tokenizer has no legal continuation for {signature}")
            cached = tuple(sorted(set(allowed)))
            self._allowed_by_signature[signature] = cached
        return list(cached)


def legal_topology_edit_commands(
    state: StratifiedState,
    catalog: ChartCatalog,
) -> tuple[str, ...]:
    """Enumerate the exact finite support consumed by one Llama edit call."""

    commands = {"NOOP"}
    for wyckoff_type in catalog.types(state.space_group):
        spec = catalog.get(state.space_group, int(wyckoff_type))
        if state.atom_count + int(spec.primitive_multiplicity) <= 20:
            for species in MP20_ATOMIC_NUMBERS:
                commands.add(
                    serialize_topology_edit(
                        TopologyEdit(
                            "birth",
                            wyckoff_type=int(wyckoff_type),
                            species=int(species),
                        )
                    )
                )
    for index, orbit in enumerate(state.orbits):
        if len(state.orbits) > 1:
            commands.add(serialize_topology_edit(TopologyEdit("death", orbit_index=index)))
        remaining = state.atom_count - int(orbit.primitive_multiplicity)
        for wyckoff_type in catalog.types(state.space_group):
            spec = catalog.get(state.space_group, int(wyckoff_type))
            if (
                int(wyckoff_type) != orbit.wyckoff_type
                and 1
                <= remaining + int(spec.primitive_multiplicity)
                <= 20
            ):
                commands.add(
                    serialize_topology_edit(
                        TopologyEdit(
                            "type_change",
                            orbit_index=index,
                            wyckoff_type=int(wyckoff_type),
                        )
                    )
                )
        for species in MP20_ATOMIC_NUMBERS:
            if int(species) != orbit.species:
                commands.add(
                    serialize_topology_edit(
                        TopologyEdit(
                            "species_change",
                            orbit_index=index,
                            species=int(species),
                        )
                    )
                )
    return tuple(sorted(commands))


class TopologyEditTokenConstraint:
    """Tokenizer-level exact support mask for a state-dependent edit command."""

    _GRAMMAR_CHARACTERS = ProposalTokenConstraint._GRAMMAR_CHARACTERS

    def __init__(
        self,
        tokenizer: Any,
        state: StratifiedState,
        catalog: ChartCatalog,
        *,
        prompt_width: int,
        token_fragments: Sequence[tuple[int, str]] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.commands = legal_topology_edit_commands(state, catalog)
        self.prompt_width = int(prompt_width)
        if self.prompt_width <= 0 or tokenizer.eos_token_id is None:
            raise ValueError("positive prompt width and EOS token are required")
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.token_fragments = tuple(
            grammar_token_fragments(tokenizer)
            if token_fragments is None
            else token_fragments
        )
        self._prefix_by_tokens: dict[tuple[int, ...], str] = {(): ""}
        self._allowed_by_prefix: dict[str, tuple[int, ...]] = {}

    def _prefix(self, generated: tuple[int, ...]) -> str:
        cached = self._prefix_by_tokens.get(generated)
        if cached is not None:
            return cached
        parent = self._prefix(generated[:-1])
        fragment = self.tokenizer.decode(
            [generated[-1]],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        prefix = parent + fragment
        if not any(command.startswith(prefix) for command in self.commands):
            raise GrammarViolation("generated edit prefix left the legal support")
        self._prefix_by_tokens[generated] = prefix
        return prefix

    def __call__(self, batch_id: int, input_ids: Any) -> list[int]:
        del batch_id
        generated = tuple(int(value) for value in input_ids.tolist()[self.prompt_width :])
        prefix = self._prefix(generated)
        cached = self._allowed_by_prefix.get(prefix)
        if cached is None:
            allowed: list[int] = []
            if prefix in self.commands:
                allowed.append(self.eos_token_id)
            for token_id, fragment in self.token_fragments:
                candidate = prefix + fragment
                if any(command.startswith(candidate) for command in self.commands):
                    allowed.append(token_id)
            if not allowed:
                raise RuntimeError("tokenizer has no legal edit continuation")
            cached = tuple(sorted(set(allowed)))
            self._allowed_by_prefix[prefix] = cached
        return list(cached)
