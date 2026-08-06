from __future__ import annotations

import unittest

from crystal_dlm.wqcodiff.bridge import ChartCatalog, ChartSpec
from crystal_dlm.wqcodiff.crysllmgen.constrained import (
    ProposalTextCursor,
    ProposalTokenConstraint,
    TopologyEditTokenConstraint,
    legal_topology_edit_commands,
)
from crystal_dlm.wqcodiff.crysllmgen.wq_text import GrammarViolation
from crystal_dlm.wqcodiff.state import OrbitState, StratifiedState


class FakeCatalog(ChartCatalog):
    def types(self, space_group: int) -> tuple[int, ...]:
        if not 1 <= int(space_group) <= 230:
            raise KeyError(space_group)
        return (0, 1)

    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        if int(wyckoff_type) == 0:
            return ChartSpec(int(space_group), 0, "a", 1, 0, 1)
        if int(wyckoff_type) == 1:
            return ChartSpec(int(space_group), 1, "b", 2, 1, 2)
        raise KeyError((space_group, wyckoff_type))


class CharacterTokenizer:
    """Small deterministic tokenizer exposing every grammar byte directly."""

    def __init__(self) -> None:
        characters = sorted(ProposalTokenConstraint._GRAMMAR_CHARACTERS)
        self._text_by_id = {index: value for index, value in enumerate(characters)}
        self._id_by_text = {value: index for index, value in self._text_by_id.items()}
        self.eos_token_id = len(characters)

    def __len__(self) -> int:
        return self.eos_token_id + 1

    def decode(self, token_ids, **_kwargs) -> str:
        return "".join(self._text_by_id[int(value)] for value in token_ids)

    def encode_character(self, value: str) -> int:
        return self._id_by_text[value]


class FakeInputIds(list):
    def tolist(self):
        return list(self)


VALID = "SG=225;Q=80;O=0,W=1,E=8,U=60;O=1,W=0,E=14,U=-;STOP"


class CrysLLMGenConstrainedDecodingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = FakeCatalog()

    def test_cursor_accepts_valid_text_characterwise_and_in_chunks(self) -> None:
        characterwise = ProposalTextCursor()
        for value in VALID:
            characterwise = characterwise.feed(value, self.catalog)
        self.assertTrue(characterwise.terminal)
        self.assertEqual(characterwise.semantic.atom_count, 3)
        self.assertEqual(characterwise.semantic.orbit_count, 2)

        chunked = ProposalTextCursor()
        offsets = (4, 11, 23, 38, 51, len(VALID))
        start = 0
        for stop in offsets:
            chunked = chunked.feed(VALID[start:stop], self.catalog)
            start = stop
        self.assertEqual(chunked, characterwise)

    def test_cursor_rejects_whitespace_and_illegal_wyckoff_support(self) -> None:
        with self.assertRaises(GrammarViolation):
            ProposalTextCursor().feed(VALID.replace("SG=", "SG= ", 1), self.catalog)
        with self.assertRaises(GrammarViolation):
            ProposalTextCursor().feed(VALID.replace("W=1", "W=9", 1), self.catalog)

    def test_stop_only_becomes_available_after_a_complete_orbit(self) -> None:
        cursor = ProposalTextCursor().feed("SG=225;Q=80;O=0,W=", self.catalog)
        self.assertEqual(cursor.stage, "wyckoff")
        self.assertNotIn("STOP", cursor.options(self.catalog))
        cursor = cursor.feed("0,E=1,U=-;", self.catalog)
        self.assertEqual(cursor.stage, "orbit_or_stop")
        self.assertIn("STOP", cursor.options(self.catalog))

    def test_token_constraint_allows_exact_valid_path_then_only_eos(self) -> None:
        tokenizer = CharacterTokenizer()
        constraint = ProposalTokenConstraint(tokenizer, self.catalog, prompt_width=2)
        generated: list[int] = []
        for character in VALID:
            token_id = tokenizer.encode_character(character)
            allowed = constraint(0, FakeInputIds([999, 998, *generated]))
            self.assertIn(token_id, allowed)
            generated.append(token_id)
        self.assertEqual(
            constraint(0, FakeInputIds([999, 998, *generated])),
            [tokenizer.eos_token_id],
        )

    def test_edit_constraint_masks_state_illegal_commands(self) -> None:
        state = StratifiedState(
            space_group=225,
            lattice_system="cubic",
            lattice_chart=(1.5,),
            orbits=(OrbitState("o0", 0, 14, 1, 0, (), 1),),
        )
        commands = legal_topology_edit_commands(state, self.catalog)
        self.assertNotIn("DEATH;O=0", commands)
        self.assertIn("SPECIES;O=0;E=89", commands)
        tokenizer = CharacterTokenizer()
        constraint = TopologyEditTokenConstraint(
            tokenizer, state, self.catalog, prompt_width=1
        )
        generated: list[int] = []
        for character in "SPECIES;O=0;E=89":
            token_id = tokenizer.encode_character(character)
            self.assertIn(token_id, constraint(0, FakeInputIds([999, *generated])))
            generated.append(token_id)
        self.assertEqual(
            constraint(0, FakeInputIds([999, *generated])),
            [tokenizer.eos_token_id],
        )


if __name__ == "__main__":
    unittest.main()
