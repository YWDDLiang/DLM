import unittest

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer, build_special_tokens, parse_dynamic_answer
from crystal_dlm.r5_dynamic_length import (
    build_exact_length_record,
    exact_body_token_count,
    exact_dynamic_generation_schedule,
    exact_dynamic_schema_constraints,
    validate_answer_matches_plan,
)
from crystal_dlm.r5_plan_state import plan_state_from_arrays


class FakeTokenizer:
    def __init__(self):
        self.vocab = {token: idx for idx, token in enumerate(build_special_tokens())}

    def get_vocab(self):
        return self.vocab


class R5DynamicLengthTests(unittest.TestCase):
    def make_arrays_and_plan(self):
        answer, _ = arrays_to_dynamic_answer(
            lengths=[4.0, 4.1, 4.2],
            angles=[90, 91, 92],
            species=["Na", "Cl"],
            frac_coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        arrays = parse_dynamic_answer(answer, strict=True)
        return arrays, plan_state_from_arrays(arrays)

    def test_exact_record_has_no_tail_canvas(self):
        arrays, plan = self.make_arrays_and_plan()
        record = build_exact_length_record(plan_state=plan, arrays=arrays)
        self.assertEqual(record["answer_semantic_length"], 15)
        self.assertEqual(exact_body_token_count(plan), 15)
        self.assertNotIn("<EMPTY>", record["answer"])
        parsed = validate_answer_matches_plan(plan, record["answer"])
        self.assertEqual(parsed["species"], ["Na", "Cl"])

    def test_schema_constraints_and_schedule_are_exact_length(self):
        arrays, plan = self.make_arrays_and_plan()
        tokenizer = FakeTokenizer()
        allowed = exact_dynamic_schema_constraints(tokenizer, plan["N"])
        schedule = exact_dynamic_generation_schedule(plan["N"])
        self.assertEqual(len(allowed), 15)
        scheduled = sorted(position for group in schedule for position in group)
        self.assertEqual(scheduled, list(range(15)))
        count_allowed = allowed[0]
        count_token_id = tokenizer.get_vocab()["<N_002>"]
        self.assertEqual(count_allowed, [count_token_id])


if __name__ == "__main__":
    unittest.main()
