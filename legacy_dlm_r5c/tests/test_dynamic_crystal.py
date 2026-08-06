import unittest

from crystal_dlm.dynamic_crystal import (
    dynamic_answer_token_count,
    arrays_to_dynamic_answer,
    arrays_to_dynamic_tokens,
    parse_dynamic_answer,
)
from crystal_dlm.fixed_slot import FixedSlotError


class DynamicCrystalTests(unittest.TestCase):
    def test_round_trip_two_atoms_has_dynamic_length(self):
        answer, diagnostics = arrays_to_dynamic_answer(
            lengths=[4.13, 4.13, 18.42],
            angles=[90.0, 90.0, 120.0],
            species=["Ga", "Te"],
            frac_coords=[[1.0, 0.666, -0.1], [0.25, 0.75, 0.5]],
        )
        self.assertNotIn("<S00>", answer)
        self.assertNotIn("<EMPTY>", answer)
        self.assertNotIn("<X_PAD>", answer)
        parsed = parse_dynamic_answer(answer, strict=True)
        self.assertEqual(parsed["num_atoms"], 2)
        self.assertEqual(parsed["species"], ["Ga", "Te"])
        self.assertEqual(len(parsed["tokens"]), dynamic_answer_token_count(2))
        self.assertGreaterEqual(diagnostics.coord_wraps, 1)

    def test_compact_answer_extracts_tokens_from_text(self):
        tokens, _ = arrays_to_dynamic_tokens(
            lengths=[3, 3, 3],
            angles=[90, 90, 90],
            species=["Na"],
            frac_coords=[[0, 0, 0]],
        )
        parsed = parse_dynamic_answer("prefix" + "".join(tokens) + "<|endoftext|>trailing")
        self.assertEqual(parsed["num_atoms"], 1)
        self.assertEqual(parsed["species"], ["Na"])
        self.assertEqual(len(parsed["tokens"]), 11)

    def test_rejects_fixed_slot_only_tokens(self):
        answer, _ = arrays_to_dynamic_answer(
            lengths=[3, 3, 3],
            angles=[90, 90, 90],
            species=["Na"],
            frac_coords=[[0, 0, 0]],
        )
        with self.assertRaises(FixedSlotError):
            parse_dynamic_answer(answer + "<S00>", strict=True)


if __name__ == "__main__":
    unittest.main()
