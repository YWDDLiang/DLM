import unittest

from crystal_dlm.fixed_slot import ANSWER_TOKEN_COUNT, arrays_to_answer, parse_fixed_slot_answer
from crystal_dlm.physical_header import (
    PHYSICAL_HEADER_ANSWER_TOKEN_COUNT,
    PHYSICAL_HEADER_BODY_OFFSET,
    build_physical_header_special_tokens,
    parse_physical_header_answer,
    physical_header_allowed_token_strings,
    prepend_physical_header_to_answer,
    tokenize_physical_header_text,
)


class FixedSlotPhysicalHeaderTests(unittest.TestCase):
    def test_header_prepended_to_fixed_slot_body_round_trips(self):
        fixed_answer, _ = arrays_to_answer(
            lengths=[3.0, 3.0, 5.2],
            angles=[90, 90, 120],
            species=["Li", "O", "O"],
            frac_coords=[[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.75, 0.5]],
        )
        answer, labels = prepend_physical_header_to_answer(fixed_answer)
        header_tokens = tokenize_physical_header_text(answer)

        self.assertEqual(len(header_tokens), PHYSICAL_HEADER_BODY_OFFSET)
        self.assertEqual(len(build_physical_header_special_tokens()), len(set(build_physical_header_special_tokens())))
        self.assertEqual(labels["tokens"][0], "<H_START>")
        self.assertEqual(labels["tokens"][-1], "<H_END>")
        parsed = parse_physical_header_answer(answer)
        fixed_parsed = parse_fixed_slot_answer(fixed_answer, strict=True)
        self.assertEqual(parsed["species"], fixed_parsed["species"])
        self.assertEqual(parsed["num_atoms"], 3)

    def test_header_schema_length(self):
        allowed = physical_header_allowed_token_strings()
        self.assertEqual(len(allowed), PHYSICAL_HEADER_BODY_OFFSET)
        self.assertEqual(PHYSICAL_HEADER_ANSWER_TOKEN_COUNT, PHYSICAL_HEADER_BODY_OFFSET + ANSWER_TOKEN_COUNT)
        self.assertEqual(allowed[0], ["<H_START>"])
        self.assertEqual(allowed[-1], ["<H_END>"])
        self.assertIn("<LAT_SYS_CUBIC>", allowed[6])
        self.assertIn("<VOLPA_020>", allowed[7])


if __name__ == "__main__":
    unittest.main()
