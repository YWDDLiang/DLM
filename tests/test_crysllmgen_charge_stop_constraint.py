from __future__ import annotations

import unittest

from crystal_dlm.wqcodiff.bridge import ChartCatalog, ChartSpec
from crystal_dlm.wqcodiff.crysllmgen.chemistry_constraint import (
    ChargeAwareStopConstraint,
    completed_orbit_composition,
)
from crystal_dlm.wqcodiff.crysllmgen.constrained import (
    ProposalTextCursor,
    ProposalTokenConstraint,
)


class FakeCatalog(ChartCatalog):
    def types(self, space_group: int) -> tuple[int, ...]:
        if int(space_group) != 225:
            raise KeyError(space_group)
        return (0, 1)

    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        if int(space_group) != 225:
            raise KeyError(space_group)
        if int(wyckoff_type) == 0:
            return ChartSpec(225, 0, "a", 1, 0, 1)
        if int(wyckoff_type) == 1:
            return ChartSpec(225, 1, "b", 2, 0, 2)
        raise KeyError((space_group, wyckoff_type))


class CharacterTokenizer:
    def __init__(self) -> None:
        characters = sorted(ProposalTokenConstraint._GRAMMAR_CHARACTERS)
        self._text_by_id = {index: value for index, value in enumerate(characters)}
        self._id_by_text = {value: index for index, value in self._text_by_id.items()}
        self.eos_token_id = len(characters)

    def __len__(self) -> int:
        return self.eos_token_id + 1

    def decode(self, token_ids, **_kwargs) -> str:
        return "".join(self._text_by_id[int(value)] for value in token_ids)

    def encode(self, text: str) -> list[int]:
        return [self._id_by_text[value] for value in text]

    def token(self, value: str) -> int:
        return self._id_by_text[value]


class FakeInputIds(list):
    def tolist(self):
        return list(self)


def li2o_classifier(elems, counts):
    values = dict(zip((int(value) for value in elems), (int(value) for value in counts)))
    if values == {3: 2, 8: 1}:
        return {"valid": True, "reason": "charge_neutral_pauling_valid"}
    return {"valid": False, "reason": "charge_neutrality_fail"}


class ChargeAwareStopConstraintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = FakeCatalog()
        self.tokenizer = CharacterTokenizer()

    def _allowed_after(self, text: str, *, classifier=li2o_classifier) -> list[int]:
        constraint = ChargeAwareStopConstraint(
            self.tokenizer,
            self.catalog,
            prompt_width=1,
            classifier=classifier,
        )
        return constraint(
            0,
            FakeInputIds([999, *self.tokenizer.encode(text)]),
        )

    def test_completed_orbit_composition_uses_primitive_multiplicity(self) -> None:
        text = (
            "SG=225;Q=80;O=0,W=1,E=3,U=-;"
            "O=1,W=0,E=8,U=-;"
        )
        self.assertEqual(
            completed_orbit_composition(
                text,
                space_group=225,
                catalog=self.catalog,
            ),
            ((3, 2), (8, 1)),
        )

    def test_charge_invalid_stop_is_deferred_but_next_orbit_remains(self) -> None:
        prefix = "SG=225;Q=80;O=0,W=0,E=3,U=-;"
        allowed = self._allowed_after(prefix)
        self.assertNotIn(self.tokenizer.token("S"), allowed)
        self.assertIn(self.tokenizer.token("O"), allowed)

    def test_charge_valid_stop_is_allowed(self) -> None:
        prefix = (
            "SG=225;Q=80;O=0,W=1,E=3,U=-;"
            "O=1,W=0,E=8,U=-;"
        )
        self.assertIn(self.tokenizer.token("S"), self._allowed_after(prefix))

    def test_pauling_only_is_soft_and_does_not_defer_stop(self) -> None:
        classifier = lambda elems, counts: {
            "valid": False,
            "reason": "pauling_fail_or_ratio_rejected",
        }
        prefix = "SG=225;Q=80;O=0,W=0,E=3,U=-;"
        self.assertIn(
            self.tokenizer.token("S"),
            self._allowed_after(prefix, classifier=classifier),
        )

    def test_invalid_stop_is_allowed_when_atom_support_is_exhausted(self) -> None:
        prefix = "SG=225;Q=80;" + "".join(
            f"O={index},W=0,E=3,U=-;" for index in range(20)
        )
        self.assertIn(self.tokenizer.token("S"), self._allowed_after(prefix))

    def test_token_spanning_orbit_terminator_and_stop_cannot_bypass_mask(self) -> None:
        constraint = ChargeAwareStopConstraint(
            self.tokenizer,
            self.catalog,
            prompt_width=1,
            classifier=li2o_classifier,
        )
        prefix = "SG=225;Q=80;O=0,W=0,E=3,U="
        cursor = ProposalTextCursor().feed(prefix, self.catalog)
        self.assertFalse(
            constraint._fragment_allowed(
                proposal_prefix=prefix,
                cursor=cursor,
                fragment="-;STOP",
            )
        )

    def test_diagnostics_count_unique_deferral_without_retry(self) -> None:
        constraint = ChargeAwareStopConstraint(
            self.tokenizer,
            self.catalog,
            prompt_width=1,
            classifier=li2o_classifier,
        )
        prefix = "SG=225;Q=80;O=0,W=0,E=3,U=-;"
        input_ids = FakeInputIds([999, *self.tokenizer.encode(prefix)])
        constraint(0, input_ids)
        constraint(0, input_ids)
        diagnostics = constraint.diagnostics()
        self.assertEqual(diagnostics["unique_stop_deferrals"], 1)
        self.assertEqual(diagnostics["classifier_evaluations"], 1)
        self.assertFalse(diagnostics["retry_or_replacement_used"])
        self.assertFalse(diagnostics["pauling_is_hard_constraint"])


if __name__ == "__main__":
    unittest.main()
