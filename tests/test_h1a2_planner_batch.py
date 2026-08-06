import unittest

from crystal_dlm.h1a2_planner_batch import prepare_planner_example
from crystal_dlm.h1a2_planner_objective import (
    FIELD_GROUP_IDS,
    build_lookahead_vocabs,
)


PLAN = (
    "formula: Li2O\n"
    "anion: oxide\n"
    "charge: neutral_plausible\n"
    "lattice: hexagonal\n"
    "spacegroup: sg_168_194\n"
    "volume: volpa_005_009\n"
    "end: plan"
)


class CharacterTokenizer:
    is_fast = True
    eos_token = "<eos>"
    eos_token_id = 10000
    chat_template = None

    def __init__(self, *, break_eos_contract: bool = False):
        self.break_eos_contract = break_eos_contract

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        del add_special_tokens
        text = str(text)
        has_eos = text.endswith(self.eos_token)
        plain = text[: -len(self.eos_token)] if has_eos else text
        ids = [ord(char) for char in plain]
        if has_eos:
            ids.append(
                self.eos_token_id + (1 if self.break_eos_contract else 0)
            )
        output = {"input_ids": ids}
        if return_offsets_mapping:
            if has_eos:
                raise AssertionError("fixture does not request EOS offsets")
            output["offset_mapping"] = [
                (index, index + 1) for index in range(len(plain))
            ]
        return output


class H1A2PlannerBatchTests(unittest.TestCase):
    def setUp(self):
        self.vocabs = build_lookahead_vocabs([PLAN])
        self.row = {
            "prompt": "fixed prompt",
            "answer": PLAN,
            "v3_planner_stream": {"source_line_sha256": "a" * 64},
        }

    def test_additive_tokenization_fields_boundaries_and_labels(self):
        example = prepare_planner_example(
            self.row,
            CharacterTokenizer(),
            max_length=256,
            lookahead_vocabs=self.vocabs,
        )
        self.assertEqual(len(example.input_ids), len(example.labels))
        self.assertEqual(len(example.input_ids), len(example.field_group_ids))
        self.assertTrue(all(value == -100 for value in example.labels[:12]))
        self.assertTrue(
            all(value == -1 for value in example.field_group_ids[:12])
        )
        self.assertEqual(
            example.field_group_ids[example.formula_boundary],
            FIELD_GROUP_IDS["formula"],
        )
        self.assertEqual(
            example.field_group_ids[example.lattice_boundary],
            FIELD_GROUP_IDS["geometry"],
        )
        self.assertLess(example.formula_boundary, example.lattice_boundary)
        self.assertEqual(example.input_ids[-1], CharacterTokenizer.eos_token_id)
        self.assertEqual(example.lookahead_labels["anion"], 0)
        self.assertEqual(example.source_line_sha256, "a" * 64)

    def test_left_truncates_prompt_without_touching_answer_or_boundaries(self):
        tokenizer = CharacterTokenizer()
        full = prepare_planner_example(
            self.row,
            tokenizer,
            max_length=256,
            lookahead_vocabs=self.vocabs,
        )
        short = prepare_planner_example(
            {**self.row, "prompt": "x" * 500},
            tokenizer,
            max_length=180,
            lookahead_vocabs=self.vocabs,
        )
        self.assertEqual(short.answer_tokens, full.answer_tokens)
        self.assertEqual(len(short.input_ids), 180)
        self.assertEqual(
            short.lattice_boundary - short.formula_boundary,
            full.lattice_boundary - full.formula_boundary,
        )

    def test_eos_mismatch_and_nonfast_tokenizer_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "historical additive"):
            prepare_planner_example(
                self.row,
                CharacterTokenizer(break_eos_contract=True),
                max_length=256,
                lookahead_vocabs=self.vocabs,
            )
        tokenizer = CharacterTokenizer()
        tokenizer.is_fast = False
        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            prepare_planner_example(
                self.row,
                tokenizer,
                max_length=256,
                lookahead_vocabs=self.vocabs,
            )


if __name__ == "__main__":
    unittest.main()
