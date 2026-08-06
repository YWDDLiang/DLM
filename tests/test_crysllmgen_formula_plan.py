from __future__ import annotations

import unittest
from unittest.mock import patch

from crystal_dlm.wqcodiff.bridge import ChartCatalog, ChartSpec
from crystal_dlm.wqcodiff.crysllmgen.formula_plan import (
    FormulaPlan,
    FormulaPlanTextCursor,
    FormulaPlanTokenConstraint,
    PlannedProposalTextCursor,
    _count_reachable_by_multiplicities,
    formula_body_user_prompt,
    formula_plan_from_state,
    formula_plan_matches_state,
    parse_formula_plan,
    serialize_formula_plan,
)
from crystal_dlm.wqcodiff.crysllmgen.wq_text import GrammarViolation
from crystal_dlm.wqcodiff.state import OrbitState, StratifiedState
from scripts.a800.build_wq_formula_plan_sft_data import (
    formula_plan_training_examples,
)
from scripts.a800.run_wq_formula_plan_paired64_v1 import _formula_attempt, summarize


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


class ReachabilityCatalog(ChartCatalog):
    """SG 225 supports multiplicities 2/3; other groups support only 2."""

    def types(self, space_group: int) -> tuple[int, ...]:
        if not 1 <= int(space_group) <= 230:
            raise KeyError(space_group)
        return (0, 1) if int(space_group) == 225 else (0,)

    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        if int(wyckoff_type) == 0:
            return ChartSpec(int(space_group), 0, "a", 2, 0, 2)
        if int(space_group) == 225 and int(wyckoff_type) == 1:
            return ChartSpec(225, 1, "b", 3, 0, 3)
        raise KeyError((space_group, wyckoff_type))


class CharacterTokenizer:
    def __init__(self) -> None:
        characters = sorted(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789=,;-"))
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


PLAN = "F=E=8,N=2;E=14,N=1;END"
BODY = "SG=225;Q=80;O=0,W=1,E=8,U=60;O=1,W=0,E=14,U=-;STOP"


class FormulaPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = FakeCatalog()

    def test_plan_roundtrip_is_canonical_and_bounded(self) -> None:
        plan = parse_formula_plan(PLAN)
        self.assertEqual(plan.entries, ((8, 2), (14, 1)))
        self.assertEqual(plan.total_atoms, 3)
        self.assertEqual(serialize_formula_plan(plan), PLAN)
        self.assertIn(PLAN, formula_body_user_prompt(plan))
        with self.assertRaises(GrammarViolation):
            parse_formula_plan("F=E=14,N=1;E=8,N=2;END")
        with self.assertRaises(GrammarViolation):
            FormulaPlan(((8, 21),))

    def test_planner_defers_invalid_end_until_a_second_species(self) -> None:
        validity = lambda entries: len(entries) >= 2
        cursor = FormulaPlanTextCursor().feed("F=E=8,N=2;", validity)
        self.assertEqual(cursor.stage, "next_or_end")
        self.assertNotIn("END", cursor.options(validity))
        self.assertIn("E=", cursor.options(validity))
        cursor = cursor.feed("E=14,N=1;END", validity)
        self.assertTrue(cursor.terminal)
        self.assertEqual(cursor.entries, ((8, 2), (14, 1)))

    def test_planner_never_accepts_invalid_end_at_atom_limit(self) -> None:
        validity = lambda _entries: False
        cursor = FormulaPlanTextCursor().feed("F=E=8,N=", validity)
        with self.assertRaises(GrammarViolation):
            cursor.feed("20;", validity)

    def test_formula_plan_token_constraint_accepts_exact_path(self) -> None:
        tokenizer = CharacterTokenizer()
        validity = lambda entries: len(entries) >= 2
        constraint = FormulaPlanTokenConstraint(
            tokenizer,
            prompt_width=2,
            validity=validity,
        )
        generated: list[int] = []
        for character in PLAN:
            token = tokenizer.encode_character(character)
            self.assertIn(token, constraint(0, FakeInputIds([999, 998, *generated])))
            generated.append(token)
        self.assertEqual(
            constraint(0, FakeInputIds([999, 998, *generated])),
            [tokenizer.eos_token_id],
        )

    def test_body_cursor_consumes_plan_multiplicities_before_stop(self) -> None:
        plan = parse_formula_plan(PLAN)
        cursor = PlannedProposalTextCursor.start(plan)
        prefix = "SG=225;Q=80;O=0,W=1,E=8,U=60;"
        cursor = cursor.feed(prefix, self.catalog)
        self.assertEqual(cursor.remaining, ((8, 0), (14, 1)))
        self.assertNotIn("STOP", cursor.options(self.catalog))
        cursor = cursor.feed("O=1,W=0,E=14,U=-;STOP", self.catalog)
        self.assertTrue(cursor.terminal)
        self.assertTrue(cursor.exhausted)
        self.assertEqual(cursor.semantic.atom_count, 3)

    def test_exact_count_reachability_masks_space_group_and_trap_species(self) -> None:
        catalog = ReachabilityCatalog()
        plan = FormulaPlan(((8, 4), (14, 3)))
        self.assertTrue(_count_reachable_by_multiplicities(7, (2, 3)))
        self.assertFalse(_count_reachable_by_multiplicities(1, (2, 3)))

        cursor = PlannedProposalTextCursor.start(plan, catalog)
        space_groups = cursor.options(catalog)
        self.assertIn("SG=225;Q=", space_groups)
        self.assertNotIn("SG=224;Q=", space_groups)

        cursor = cursor.feed("SG=225;Q=00;O=0,W=0,E=", catalog)
        # Multiplicity 2 can be assigned to species 8 (4 -> 2), but assigning
        # it to species 14 (3 -> 1) would leave an unreachable remainder.
        self.assertEqual(cursor.options(catalog), ("8,U=",))

    def test_reachability_mask_guarantees_complete_planned_body(self) -> None:
        catalog = ReachabilityCatalog()
        plan = FormulaPlan(((8, 4), (14, 3)))
        cursor = PlannedProposalTextCursor.start(plan, catalog)
        body = (
            "SG=225;Q=00;O=0,W=0,E=8,U=-;"
            "O=1,W=0,E=8,U=-;"
            "O=2,W=1,E=14,U=-;STOP"
        )
        cursor = cursor.feed(body, catalog)
        self.assertTrue(cursor.terminal)
        self.assertTrue(cursor.exhausted)
        self.assertEqual(cursor.semantic.atom_count, 7)

    def test_evaluator_preserves_successful_plan_when_body_fails(self) -> None:
        class BodyFailureEngine:
            def generate_formula_plan(self, *, plan_seed):
                self.plan_seed = plan_seed
                return parse_formula_plan(PLAN), PLAN, {"prompt_tokens": 1}

            def generate_formula_body(self, **_kwargs):
                raise RuntimeError("synthetic body failure")

        with patch(
            "crystal_dlm.wqcodiff.crysllmgen.formula_plan."
            "_entries_are_composition_valid",
            return_value=True,
        ):
            row = _formula_attempt(
                engine=BodyFailureEngine(),
                catalog=self.catalog,
                baseline={"proposal_seed": 7, "ordinal": 9, "pair_id": "pair-9"},
                attempt_id="attempt-9",
                execution_patch_sha256="a" * 64,
                contract_sha256="b" * 64,
                identity="one-epoch-test",
                row_schema="one-epoch-row-test",
            )
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["plan_status"], "succeeded")
        self.assertEqual(row["body_status"], "failed")
        self.assertEqual(row["failure_stage"], "formula_body")
        self.assertEqual(row["formula_plan_text"], PLAN)
        self.assertIn("synthetic body failure", row["reason"])

    def test_plan_matches_state_independent_of_orbit_order(self) -> None:
        state = StratifiedState(
            space_group=225,
            lattice_system="cubic",
            lattice_chart=(1.5,),
            orbits=(
                OrbitState("o0", 0, 14, 1, 0, (), 1),
                OrbitState("o1", 1, 8, 2, 1, (0.25,), 2),
            ),
        )
        plan = formula_plan_from_state(state)
        self.assertEqual(plan.entries, ((8, 2), (14, 1)))
        self.assertTrue(formula_plan_matches_state(plan, state))
        mismatch = FormulaPlan(((8, 1), (14, 1)))
        self.assertFalse(formula_plan_matches_state(mismatch, state))

    def test_body_cursor_rejects_species_or_stop_outside_plan(self) -> None:
        cursor = PlannedProposalTextCursor.start(parse_formula_plan(PLAN))
        with self.assertRaises(GrammarViolation):
            cursor.feed(BODY.replace("E=8", "E=9", 1), self.catalog)
        with self.assertRaises(GrammarViolation):
            cursor.feed(
                "SG=225;Q=80;O=0,W=1,E=8,U=60;STOP",
                self.catalog,
            )

    def test_sft_transform_emits_plan_body_and_half_edit_replay_tasks(self) -> None:
        source = {
            "schema": "crysllmgen_sft_example_v1",
            "example_id": "sft-source",
            "stage": "coarse_proposal",
            "material_id": "mp-test",
            "training_seed": 11,
            "source_topology_hash": "a" * 64,
            "answer": BODY,
            "answer_sha256": "b" * 64,
        }
        examples = formula_plan_training_examples(
            source,
            catalog=self.catalog,
            replay_direct_edit=False,
            require_composition_valid=False,
        )
        self.assertEqual(
            [value["stage"] for value in examples],
            ["formula_plan", "formula_conditioned_body"],
        )
        self.assertEqual(examples[0]["answer"], PLAN)
        self.assertEqual(examples[1]["answer"], BODY)
        self.assertIn(PLAN, examples[1]["user_prompt"])

        edit = {
            **source,
            "example_id": "edit-source",
            "stage": "direct_edit",
            "system_prompt": "Return one edit.",
            "user_prompt": "P=x;G=-",
            "answer": "NOOP",
            "answer_sha256": "c" * 64,
        }
        self.assertEqual(
            formula_plan_training_examples(
                edit,
                catalog=self.catalog,
                replay_direct_edit=False,
                require_composition_valid=False,
            ),
            (),
        )
        replay = formula_plan_training_examples(
            edit,
            catalog=self.catalog,
            replay_direct_edit=True,
            require_composition_valid=False,
        )
        self.assertEqual(replay[0]["stage"], "direct_edit_replay")

    def test_paired64_gate_rewards_real_gain_and_blocks_diversity_collapse(self) -> None:
        baseline = []
        formula = []
        for ordinal in range(64):
            pair_id = f"pair-{ordinal}"
            baseline_valid = ordinal < 61
            formula_valid = ordinal < 63
            baseline.append(
                {
                    "pair_id": pair_id,
                    "status": "succeeded",
                    "composition_valid": baseline_valid,
                    "composition_reason": (
                        "charge_neutral_pauling_valid"
                        if baseline_valid
                        else "charge_neutrality_fail"
                    ),
                    "formula": f"B{ordinal}",
                    "atom_count": 10,
                    "orbit_count": 5,
                }
            )
            formula.append(
                {
                    "pair_id": pair_id,
                    "status": "succeeded",
                    "composition_valid": formula_valid,
                    "composition_reason": (
                        "charge_neutral_pauling_valid"
                        if formula_valid
                        else "charge_neutrality_fail"
                    ),
                    "formula": f"F{ordinal}",
                    "atom_count": 11,
                    "orbit_count": 5,
                    "plan_body_exact_match": True,
                }
            )
        gate = {
            "minimum_formula_generation_success_count": 63,
            "minimum_formula_composition_valid_count": 63,
            "minimum_composition_valid_gain_count": 2,
            "maximum_valid_to_invalid_pairs": 1,
            "minimum_unique_formula_all_attempt_rate": 0.90,
            "maximum_absolute_mean_atom_count_shift": 4.0,
        }
        result = summarize(baseline, formula, denominator=64, gate=gate)
        self.assertTrue(result["promotion_pass"])
        self.assertEqual(result["paired"]["composition_valid_gain_count"], 2)

        for row in formula:
            row["formula"] = "collapsed"
        collapsed = summarize(baseline, formula, denominator=64, gate=gate)
        self.assertFalse(collapsed["promotion_pass"])
        self.assertFalse(
            collapsed["promotion_checks"]["formula_diversity_noncollapse"]
        )


if __name__ == "__main__":
    unittest.main()
