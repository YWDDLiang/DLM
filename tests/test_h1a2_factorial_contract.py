import copy
import hashlib
import unittest

from crystal_dlm.h1a2_factorial_contract import (
    STRUCTURE_DERIVED_TEACHER_PROVENANCE,
    assert_planner_input_identity,
    build_factorial_body_inputs,
    build_factorial_ordinal_record,
    build_planner_input_contract,
    ordered_factorial_attempts,
    persist_model_sampled_plan,
    persist_parser_accepted_model_sampled_plan,
)


P0_SHA = "0" * 64
PSTAR_SHA = "1" * 64
P0_PLAN = (
    "formula: Li2O\n"
    "anion: oxide\n"
    "charge: neutral_plausible\n"
    "lattice: hexagonal\n"
    "spacegroup: sg_168_194\n"
    "volume: volpa_005_009\n"
    "end: plan"
)
PSTAR_PLAN = (
    "formula: NaCl\n"
    "anion: halide\n"
    "charge: neutral_plausible\n"
    "lattice: cubic\n"
    "spacegroup: sg_195_230\n"
    "volume: volpa_010_014\n"
    "end: plan"
)


class DummyTokenizer:
    name_or_path = "frozen-h1a2-tokenizer"
    chat_template = "dummy_h1a2_chat_template_v1"

    def __len__(self):
        return 128

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
    ):
        if tokenize:
            raise AssertionError("contract requests exact text before tokenization")
        rendered = "\n".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "\n<assistant>"
        return rendered

    def __call__(self, text, *, add_special_tokens=False):
        self.assert_no_special_tokens(add_special_tokens)
        return {"input_ids": [ord(character) % 127 for character in text]}

    @staticmethod
    def assert_no_special_tokens(add_special_tokens):
        if add_special_tokens:
            raise AssertionError("unexpected tokenizer special-token insertion")


class H1A2FactorialContractTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()
        self.p0_input = build_planner_input_contract(
            self.tokenizer,
            planner_arm="P0",
            checkpoint_sha256=P0_SHA,
        )
        self.pstar_input = build_planner_input_contract(
            self.tokenizer,
            planner_arm="Pstar",
            checkpoint_sha256=PSTAR_SHA,
        )
        self.ordinal = build_factorial_ordinal_record(17, sample_idx=3)

    def sampled_plan(self, text, arm, contract):
        return persist_model_sampled_plan(
            text,
            planner_arm=arm,
            sample_idx=3,
            planner_sampling_seed=self.ordinal["planner_sampling_seed"],
            planner_input_contract=contract,
        )

    def test_p0_and_pstar_have_identical_prompt_bytes_and_token_ids(self):
        assert_planner_input_identity(self.p0_input, self.pstar_input)
        self.assertNotEqual(
            self.p0_input["checkpoint_sha256"],
            self.pstar_input["checkpoint_sha256"],
        )
        self.assertEqual(
            self.p0_input["prompt_sha256"],
            self.pstar_input["prompt_sha256"],
        )
        self.assertEqual(
            self.p0_input["input_ids_sha256"],
            self.pstar_input["input_ids_sha256"],
        )
        self.assertNotIn("sample_id:", self.p0_input["prompt_text"])

    def test_planner_input_mismatch_fails_closed(self):
        changed = copy.deepcopy(self.pstar_input)
        changed["input_ids"] = changed["input_ids"] + [7]
        with self.assertRaisesRegex(ValueError, "input mismatch"):
            assert_planner_input_identity(self.p0_input, changed)

    def test_model_must_supply_exactly_all_seven_visible_lines(self):
        record = self.sampled_plan(P0_PLAN, "P0", self.p0_input)
        self.assertTrue(record["model_proposed_plan"])
        self.assertEqual(record["plan_provenance"], "model_sampled_h1a2_planner")
        self.assertEqual(
            set(record["plan_state"]["generated_rich_fields"]),
            {"anion", "charge", "lattice", "spacegroup", "volume"},
        )
        for invalid in (
            P0_PLAN.replace("volume: volpa_005_009\n", ""),
            P0_PLAN + "\nexplanation: chosen by renderer",
            P0_PLAN.replace("lattice: hexagonal", "lattice:"),
        ):
            with self.assertRaisesRegex(ValueError, "exactly the seven-line"):
                self.sampled_plan(invalid, "P0", self.p0_input)

    def test_within_planner_body_pairs_share_exact_plan_and_prompt(self):
        p0 = self.sampled_plan(P0_PLAN, "P0", self.p0_input)
        pstar = self.sampled_plan(PSTAR_PLAN, "Pstar", self.pstar_input)
        arms = build_factorial_body_inputs(
            p0,
            pstar,
            ordinal_record=self.ordinal,
        )
        for left, right in (("M00", "M01"), ("M10", "M11")):
            self.assertEqual(
                arms[left]["plan_text_sha256"],
                arms[right]["plan_text_sha256"],
            )
            self.assertEqual(
                arms[left]["body_prompt_sha256"],
                arms[right]["body_prompt_sha256"],
            )
            self.assertEqual(
                arms[left]["body_sampling_seed"],
                arms[right]["body_sampling_seed"],
            )
            self.assertEqual(
                arms[left]["refiner_sampling_seed"],
                arms[right]["refiner_sampling_seed"],
            )
        self.assertNotEqual(
            arms["M00"]["plan_text_sha256"],
            arms["M10"]["plan_text_sha256"],
        )

    def test_parser_accepted_raw_and_canonical_hashes_remain_distinct(self):
        raw = P0_PLAN.replace(
            "spacegroup: sg_168_194\n",
            "spacegroup: sg_168_194\nprimate: hexagonal\n",
        )
        p0 = persist_parser_accepted_model_sampled_plan(
            raw,
            P0_PLAN,
            planner_arm="P0",
            sample_idx=3,
            planner_sampling_seed=self.ordinal["planner_sampling_seed"],
            planner_input_contract=self.p0_input,
        )
        pstar = self.sampled_plan(PSTAR_PLAN, "Pstar", self.pstar_input)

        self.assertEqual(p0["raw_model_sampled_plan_text"], raw)
        self.assertEqual(
            p0["raw_plan_text_sha256"],
            hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            p0["plan_text_sha256"],
            hashlib.sha256(P0_PLAN.encode("utf-8")).hexdigest(),
        )
        self.assertNotEqual(
            p0["raw_plan_text_sha256"],
            p0["plan_text_sha256"],
        )
        self.assertFalse(p0["raw_plan_contract_conforming"])
        self.assertEqual(p0["raw_plan_format_gate"], "advisory_nonblocking")
        self.assertTrue(p0["canonicalization_used"])

        arms = build_factorial_body_inputs(
            p0,
            pstar,
            ordinal_record=self.ordinal,
        )
        self.assertEqual(arms["M00"]["plan_text"], P0_PLAN)
        self.assertEqual(
            arms["M00"]["raw_plan_text_sha256"],
            p0["raw_plan_text_sha256"],
        )

    def test_teacher_plan_substitution_fails_closed(self):
        p0 = self.sampled_plan(P0_PLAN, "P0", self.p0_input)
        pstar = self.sampled_plan(PSTAR_PLAN, "Pstar", self.pstar_input)
        p0["plan_provenance"] = STRUCTURE_DERIVED_TEACHER_PROVENANCE
        p0["source_plan_provenance"] = STRUCTURE_DERIVED_TEACHER_PROVENANCE
        with self.assertRaisesRegex(ValueError, "model-sampled"):
            build_factorial_body_inputs(
                p0,
                pstar,
                ordinal_record=self.ordinal,
            )

    def test_stage_seeds_are_rank_and_order_independent(self):
        same = build_factorial_ordinal_record(17, sample_idx=3)
        other = build_factorial_ordinal_record(17, sample_idx=4)
        self.assertEqual(self.ordinal, same)
        self.assertNotEqual(
            self.ordinal["planner_sampling_seed"],
            other["planner_sampling_seed"],
        )
        self.assertNotEqual(
            self.ordinal["body_sampling_seed"],
            self.ordinal["refiner_sampling_seed"],
        )

    def test_four_arm_merge_is_ordered_and_complete(self):
        records = [
            {"sample_idx": sample_idx, "factorial_arm": arm}
            for sample_idx in reversed(range(2))
            for arm in reversed(("M00", "M10", "M01", "M11"))
        ]
        ordered = ordered_factorial_attempts(records, expected_count=2)
        self.assertEqual(
            [(row["sample_idx"], row["factorial_arm"]) for row in ordered],
            [
                (0, "M00"),
                (0, "M10"),
                (0, "M01"),
                (0, "M11"),
                (1, "M00"),
                (1, "M10"),
                (1, "M01"),
                (1, "M11"),
            ],
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ordered_factorial_attempts(records + [records[0]], expected_count=2)
        with self.assertRaisesRegex(ValueError, "missing"):
            ordered_factorial_attempts(records[:-1], expected_count=2)


if __name__ == "__main__":
    unittest.main()
