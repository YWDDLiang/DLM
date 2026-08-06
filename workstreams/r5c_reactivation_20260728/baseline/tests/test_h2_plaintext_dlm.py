import unittest

from crystal_dlm.crysllmgen_text import CRYSLLMGEN_TEXT_PROMPT, format_crysllmgen_text, parse_crysllmgen_text
from crystal_dlm.h2_plaintext_dlm import (
    build_h2_plaintext_prompt,
    composition_signature_from_arrays,
    composition_signature_from_plan,
    proposal_matches_plan_composition,
)


class H2PlaintextDlmTests(unittest.TestCase):
    def make_plan(self):
        return {
            "N": 3,
            "elements": ["Li", "O"],
            "counts": [2, 1],
            "formula": "Li2O",
            "anion_framework": "oxide",
            "charge_bucket": "neutral_plausible",
            "lattice_system": "hexagonal",
            "spacegroup_bucket": "sg_168_194",
            "volume_per_atom_bin": "volpa_005_009",
        }

    def test_prompt_contains_rich_plan_but_no_structure_answer(self):
        prompt = build_h2_plaintext_prompt(self.make_plan())
        self.assertIn(CRYSLLMGEN_TEXT_PROMPT.rstrip(), prompt)
        self.assertIn("plan:\nformula: Li2O", prompt)
        self.assertIn("anion: oxide", prompt)
        self.assertIn("structure:", prompt)
        self.assertIn("Return only compact CrysLLMGen plain text", prompt)
        self.assertIn("Use exactly the formula counts", prompt)
        self.assertIn("Never put an element symbol and coordinates on the same line", prompt)
        self.assertIn("Do not write labels, prose", prompt)
        self.assertNotIn("0.25 0.25 0.25", prompt)
        self.assertNotIn("<LA_", prompt)

    def test_plaintext_answer_parse_and_composition_match(self):
        answer = format_crysllmgen_text(
            species=["Li", "Li", "O"],
            frac_coords=[[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
            lengths=[3.1, 3.1, 5.2],
            angles=[90, 90, 120],
        )
        arrays = parse_crysllmgen_text(answer)
        self.assertEqual(composition_signature_from_arrays(arrays), {"Li": 2, "O": 1})
        self.assertEqual(composition_signature_from_plan(self.make_plan()), {"Li": 2, "O": 1})
        self.assertTrue(proposal_matches_plan_composition(arrays, self.make_plan()))

    def test_composition_mismatch_is_detected(self):
        answer = format_crysllmgen_text(
            species=["Li", "O", "O"],
            frac_coords=[[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
            lengths=[3.1, 3.1, 5.2],
            angles=[90, 90, 120],
        )
        arrays = parse_crysllmgen_text(answer)
        self.assertFalse(proposal_matches_plan_composition(arrays, self.make_plan()))


if __name__ == "__main__":
    unittest.main()
