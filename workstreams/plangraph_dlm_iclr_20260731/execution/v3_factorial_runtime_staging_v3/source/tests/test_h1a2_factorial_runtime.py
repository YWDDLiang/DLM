import copy
import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.h1a2_factorial_contract import (
    build_factorial_arm_input,
    build_factorial_ordinal_record,
    build_planner_input_contract,
    persist_model_sampled_plan,
)
from crystal_dlm.h1a2_factorial_runtime import (
    assert_additive_body_tokenization,
    assert_body_tokenizer_identity,
    assert_factorial_pairing,
    compile_body_condition,
    load_planner_attempts,
    ordered_single_arm_attempts,
    propagated_planner_failure,
    tokenizer_vocab_sha256,
)
from crystal_dlm.r5_dynamic_length import exact_dynamic_generation_schedule


P0_PLAN = (
    "formula: Li2O\n"
    "anion: oxide\n"
    "charge: neutral_plausible\n"
    "lattice: hexagonal\n"
    "spacegroup: sg_168_194\n"
    "volume: volpa_005_009\n"
    "end: plan"
)


class CharacterTokenizer:
    name_or_path = "frozen-r5c-tokenizer"
    chat_template = "h1a2-template"

    def __init__(self):
        self.vocab = {character: index for index, character in enumerate(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<>_:;,.|=+- \n{}[]\""
        )}

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
    ):
        if tokenize:
            raise AssertionError("unexpected tokenized chat template")
        text = "\n".join(message["content"] for message in messages)
        return text + ("\nassistant:" if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("unexpected special tokens")
        return {"input_ids": [self.vocab[character] for character in text]}


class H1A2FactorialRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.ordinal = build_factorial_ordinal_record(17, sample_idx=0)
        planner_input = build_planner_input_contract(
            self.tokenizer,
            planner_arm="P0",
            checkpoint_sha256="0" * 64,
        )
        self.sampled = persist_model_sampled_plan(
            P0_PLAN,
            planner_arm="P0",
            sample_idx=0,
            planner_sampling_seed=self.ordinal["planner_sampling_seed"],
            planner_input_contract=planner_input,
        )

    def body_input(self, arm):
        return build_factorial_arm_input(
            self.sampled,
            factorial_arm=arm,
            ordinal_record=self.ordinal,
        )

    def test_b0_uses_historical_order_and_bstar_compiles_sampled_plangraph(self):
        b0 = compile_body_condition(self.body_input("M00"))
        bstar = compile_body_condition(self.body_input("M01"))
        self.assertEqual(b0["generation_policy"], "d1")
        self.assertEqual(
            b0["generation_schedule"],
            exact_dynamic_generation_schedule(self.sampled["plan_state"]["N"]),
        )
        self.assertIsNone(b0["compiled_plangraph"])
        self.assertEqual(bstar["generation_policy"], "d2")
        self.assertIsNotNone(bstar["compiled_plangraph"])
        self.assertIsNotNone(bstar["compiled_plangraph_sha256"])
        self.assertEqual(
            bstar["compiled_plangraph"]["composition"]["formula"],
            self.sampled["plan_state"]["formula"],
        )

    def test_teacher_or_prompt_substitution_fails_before_schedule(self):
        changed = self.body_input("M01")
        changed["plan_provenance"] = "structure_derived_teacher_plan_state"
        with self.assertRaisesRegex(ValueError, "model-sampled"):
            compile_body_condition(changed)
        changed = self.body_input("M01")
        changed["body_prompt"] += "mutation"
        with self.assertRaisesRegex(ValueError, "prompt SHA"):
            compile_body_condition(changed)

    def test_tokenizer_identity_and_additive_boundary(self):
        vocab_sha = tokenizer_vocab_sha256(self.tokenizer)
        identity = assert_body_tokenizer_identity(
            self.tokenizer,
            expected_vocab_sha256=vocab_sha,
        )
        self.assertEqual(identity["vocab_sha256"], vocab_sha)
        prompt = "prompt\n"
        answer = "<N_003>"
        generated = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        report = assert_additive_body_tokenization(
            self.tokenizer,
            prompt=prompt,
            answer=answer,
            generated_token_ids=generated,
            expected_answer_token_count=len(generated),
        )
        self.assertTrue(report["additive_tokenization"])
        with self.assertRaisesRegex(ValueError, "round-trip"):
            assert_additive_body_tokenization(
                self.tokenizer,
                prompt=prompt,
                answer=answer,
                generated_token_ids=generated[:-1],
                expected_answer_token_count=len(generated),
            )

    def test_complete_planner_attempt_ledger_is_strict(self):
        rows = []
        for sample_idx in range(2):
            ordinal = build_factorial_ordinal_record(17, sample_idx=sample_idx)
            planner_input = build_planner_input_contract(
                self.tokenizer,
                planner_arm="P0",
                checkpoint_sha256="0" * 64,
            )
            plan = persist_model_sampled_plan(
                P0_PLAN,
                planner_arm="P0",
                sample_idx=sample_idx,
                planner_sampling_seed=ordinal["planner_sampling_seed"],
                planner_input_contract=planner_input,
            )
            plan["attempt_status"] = "complete"
            rows.append(plan)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in reversed(rows)),
                encoding="utf-8",
            )
            loaded = load_planner_attempts(
                path,
                expected_count=2,
                expected_planner_arm="P0",
            )
        self.assertEqual([row["sample_idx"] for row in loaded], [0, 1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ordered_single_arm_attempts(
                [
                    {
                        "sample_idx": 0,
                        "factorial_arm": "M00",
                        "attempt_status": "failed",
                        "evaluation_order": 0,
                    },
                    {
                        "sample_idx": 0,
                        "factorial_arm": "M00",
                        "attempt_status": "failed",
                        "evaluation_order": 0,
                    },
                ],
                expected_count=1,
                expected_factorial_arm="M00",
            )

    def test_planner_failure_propagates_without_body_attempt(self):
        failure = {
            "sample_idx": 0,
            "planner_arm": "P0",
            "attempt_status": "failed",
            "earliest_failure_stage": "planner",
            "failure_reason": "ValueError",
            "failure_message": "not seven lines",
        }
        result = propagated_planner_failure(
            failure,
            factorial_arm="M00",
            ordinal_record=self.ordinal,
        )
        self.assertEqual(result["earliest_failure_stage"], "planner")
        self.assertEqual(result["attempt_status"], "failed")
        self.assertFalse(result["retry_used"])
        changed = copy.deepcopy(failure)
        changed["planner_arm"] = "Pstar"
        with self.assertRaisesRegex(ValueError, "arm mismatch"):
            propagated_planner_failure(
                changed,
                factorial_arm="M00",
                ordinal_record=self.ordinal,
            )

    def test_four_arm_runtime_pairing_rejects_plan_or_noise_drift(self):
        pstar_input = build_planner_input_contract(
            self.tokenizer,
            planner_arm="Pstar",
            checkpoint_sha256="1" * 64,
        )
        pstar = persist_model_sampled_plan(
            P0_PLAN,
            planner_arm="Pstar",
            sample_idx=0,
            planner_sampling_seed=self.ordinal["planner_sampling_seed"],
            planner_input_contract=pstar_input,
        )
        by_arm = {}
        for arm, plan in (
            ("M00", self.sampled),
            ("M10", pstar),
            ("M01", self.sampled),
            ("M11", pstar),
        ):
            row = build_factorial_arm_input(
                plan,
                factorial_arm=arm,
                ordinal_record=self.ordinal,
            )
            row["attempt_status"] = "complete"
            row["evaluation_order"] = 0
            by_arm[arm] = row
        report = assert_factorial_pairing(
            list(reversed(list(by_arm.values()))),
            expected_count=1,
        )
        self.assertEqual(report["total_attempts"], 4)
        changed = copy.deepcopy(list(by_arm.values()))
        changed[2]["body_sampling_seed"] += 1
        with self.assertRaisesRegex(ValueError, "body_sampling_seed"):
            assert_factorial_pairing(changed, expected_count=1)
        changed = copy.deepcopy(list(by_arm.values()))
        changed[2]["body_prompt_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "body_prompt_sha256"):
            assert_factorial_pairing(changed, expected_count=1)


if __name__ == "__main__":
    unittest.main()
