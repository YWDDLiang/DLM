import unittest

from crystal_dlm.ccfd_text import (
    analyze_formula_prefix,
    formula_term_boundaries,
    token_prefix_alignment,
)


class CharacterTokenizer:
    unk_token_id = 999

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(value) for value in text]

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(value) for value in token_ids)


class CCFDTextTest(unittest.TestCase):
    def test_formula_prefix_accepts_partial_and_terminal_formula(self):
        self.assertTrue(analyze_formula_prefix("").valid_prefix)
        self.assertTrue(analyze_formula_prefix("F").valid_prefix)
        self.assertTrue(analyze_formula_prefix("Fe").terminal)
        status = analyze_formula_prefix("Fe3O4")
        self.assertTrue(status.terminal)
        self.assertEqual(status.elements, ("Fe", "O"))
        self.assertEqual(status.counts, (3, 4))
        self.assertEqual(status.total_atoms, 7)
        ambiguous = analyze_formula_prefix("P6P")
        self.assertTrue(ambiguous.valid_prefix)
        self.assertFalse(ambiguous.terminal)

    def test_formula_prefix_rejects_malformed_or_over_budget_formula(self):
        self.assertFalse(analyze_formula_prefix("Fe0O").valid_prefix)
        self.assertFalse(analyze_formula_prefix("Fe2Fe3").valid_prefix)
        self.assertFalse(analyze_formula_prefix("Fe21").valid_prefix)
        self.assertFalse(analyze_formula_prefix("fe2O3").valid_prefix)
        self.assertFalse(analyze_formula_prefix("Fe 2O3").valid_prefix)

    def test_formula_boundaries_are_semantic_term_ends(self):
        self.assertEqual(formula_term_boundaries("Li2FeO3"), (3, 5, 7))

    def test_token_prefix_alignment_for_character_tokenizer(self):
        tokenizer = CharacterTokenizer()
        text = " Li2FeO3\nanion: oxide"
        ids = tokenizer.encode(text)
        result = token_prefix_alignment(tokenizer, ids, text)
        self.assertTrue(result["roundtrip_exact"])
        self.assertTrue(result["incremental_prefix_exact"])
        self.assertTrue(result["formula_prefix_valid"])
        self.assertEqual(result["newline_step"], len(" Li2FeO3\n"))
        self.assertIsNone(result["first_invalid_formula_prefix"])


if __name__ == "__main__":
    unittest.main()
