"""Independent acceptance tests for C3FD on the unchanged H1A2 text path.

Run with: python -m unittest discover -s tests -p test_r03_formula_bridge_contract.py
The exhaustive reference is test-only and dependency-free. Tensor tests use
CPU torch when available; no model, GPU, network or chemistry service is used.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from r03_formula_bridge_reference import (
    BruteForceFormulaLanguage,
    CORE_ELEMENTS,
    PieceTokenizer,
    SyntheticElement,
)


class BruteForceReferenceSanityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.language = BruteForceFormulaLanguage(CORE_ELEMENTS, max_atoms=5, max_species=3)

    def test_terminal_language_has_all_display_orders_and_count_one_spellings(self):
        for text in ("FeO", "OFe", "Fe1O", "O1Fe1", "Fe2O3", "O3Fe2", "Li2O", "OLi2"):
            with self.subTest(text=text):
                self.assertTrue(self.language.is_terminal_valid(text))

    def test_invalid_terminals_do_not_gain_a_mixed_valence_witness(self):
        larger = BruteForceFormulaLanguage(CORE_ELEMENTS, max_atoms=7, max_species=3)
        for text in ("Fe2O4", "Fe3O4", "LiO", "OF", "Li0", "Fe2Fe", "feO", "Fe 2O3"):
            with self.subTest(text=text):
                self.assertFalse(larger.is_terminal_valid(text))

    def test_complete_one_letter_symbol_keeps_two_letter_branch(self):
        self.assertTrue(self.language.is_terminal_valid("F"))
        self.assertTrue(self.language.is_prefix_viable("F"))
        self.assertTrue(self.language.is_prefix_viable("F2F"))
        self.assertFalse(self.language.is_terminal_valid("F2F"))
        self.assertTrue(self.language.is_terminal_valid("F2Fe"))

    def test_high_atomic_number_prefix_can_later_append_lower_atomic_number(self):
        self.assertTrue(self.language.is_prefix_viable("Fe2O"))
        self.assertTrue(self.language.is_prefix_viable("FeLi"))
        self.assertTrue(self.language.is_terminal_valid("FeLi"))

    def test_prefix_preserves_many_latent_target_strata(self):
        targets = self.language.compatible_targets("F")
        self.assertIn((1, "halide", 1), targets)
        self.assertIn((5, "oxide", 2), targets)  # Fe2O3, not an F-containing oxide.
        self.assertIn((3, "halide", 2), targets)  # F2Fe.
        self.assertGreater(len(targets), 3)

    def test_partial_count_one_can_be_ten_or_eleven(self):
        language = BruteForceFormulaLanguage(
            (SyntheticElement("Li", 3, (0, 1), 0.98, True),), max_atoms=20, max_species=1
        )
        for text in ("L", "Li", "Li1", "Li10", "Li11", "Li2", "Li20"):
            with self.subTest(text=text):
                self.assertTrue(language.is_prefix_viable(text))
        for text in ("Li0", "Li01", "Li21", "Li200"):
            with self.subTest(text=text):
                self.assertFalse(language.is_prefix_viable(text))

    def test_newline_tests_only_formula_before_rich_suffix(self):
        self.assertTrue(self.language.fragment_legal("Fe", "2O3\nanion: impossible rich text"))
        self.assertFalse(self.language.fragment_legal("Fe", "2O4\nanion: oxide"))
        self.assertFalse(self.language.fragment_legal("F2F", "\n"))
        self.assertTrue(self.language.fragment_legal("", " Fe2O3\nanion:"))
        self.assertFalse(self.language.fragment_legal("", "  Fe2O3\n"))

    def test_all_returned_prefixes_have_a_complete_witness(self):
        self.assertGreater(len(self.language.terminals), 100)
        for prefix in self.language.prefixes:
            self.assertTrue(any(text.startswith(prefix) for text in self.language.terminals))

    def test_missing_electronegativity_is_not_an_ionic_witness(self):
        language = BruteForceFormulaLanguage(
            (
                SyntheticElement("Fe", 26, (0, 2), None, True),
                SyntheticElement("O", 8, (-2, 0), 3.44),
            ), max_atoms=4, max_species=2,
        )
        self.assertFalse(language.is_terminal_valid("FeO"))
        self.assertTrue(language.is_terminal_valid("O"))
        self.assertTrue(language.is_terminal_valid("Fe"))

    def test_zero_valence_nonmetal_does_not_enter_an_alloy(self):
        self.assertTrue(self.language.is_terminal_valid("O2"))
        self.assertTrue(self.language.is_terminal_valid("LiFe"))
        self.assertFalse(self.language.is_terminal_valid("OF"))

    def test_fragment_tokenizer_does_not_hide_boundary_crossings(self):
        tokenizer = PieceTokenizer()
        ids = [tokenizer.token_id(piece) for piece in ("for", "mula: Fe", "2O3\nanion: oxide")]
        self.assertEqual(tokenizer.decode(ids), "formula: Fe2O3\nanion: oxide")


def production_oracle(elements, *, max_atoms, max_species, allowed_strata=None):
    """One API adapter; the expected language never calls production helpers."""
    try:
        from crystal_dlm.r03_formula_bridge import C3FDFormulaOracle
    except ModuleNotFoundError as exc:
        if exc.name == "crystal_dlm.r03_formula_bridge":
            raise unittest.SkipTest("new bridge prototype has not been written yet") from exc
        raise
    kwargs = {
        "nodes": {element.symbol: element.oxidation_states for element in elements},
        "electronegativities": {
            element.symbol: element.electronegativity for element in elements
        },
        "metal_symbols": {element.symbol for element in elements if element.metal},
        "max_atoms": max_atoms,
        "max_species": max_species,
    }
    if allowed_strata is not None:
        kwargs["allowed_strata"] = allowed_strata
    return C3FDFormulaOracle(**kwargs)


class SemanticBridgeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.language = BruteForceFormulaLanguage(CORE_ELEMENTS, max_atoms=5, max_species=3)
        cls.oracle = production_oracle(CORE_ELEMENTS, max_atoms=5, max_species=3)

    def test_every_reference_prefix_remains_viable(self):
        for prefix in sorted(self.language.prefixes):
            with self.subTest(prefix=prefix):
                self.assertTrue(self.oracle.is_prefix_viable(prefix))

    def test_terminal_decisions_match_exhaustive_reference(self):
        for text in sorted(self.language.prefixes | {"Xx", "Fe2Fe", "Fe0", "Fe2O4", ""}):
            with self.subTest(text=text):
                self.assertEqual(self.oracle.is_terminal_valid(text), self.language.is_terminal_valid(text))

    def test_one_character_transitions_match_exhaustive_reference(self):
        # Explore every accepted text prefix, including mutations that never
        # appear in an accepted terminal. This checks both recall and precision.
        next_characters = "FeLiO0123456Xxo- ()"
        for prefix in sorted(self.language.prefixes):
            for character in next_characters:
                text = prefix + character
                with self.subTest(text=text):
                    self.assertEqual(self.oracle.is_prefix_viable(text), self.language.is_prefix_viable(text))

    def test_same_composition_has_same_support_in_every_display_order(self):
        for text in ("Fe2O3", "O3Fe2", "FeO", "OFe", "F2Fe", "FeF2", "LiFe", "FeLi"):
            with self.subTest(text=text):
                self.assertTrue(self.oracle.is_terminal_valid(text))

    def test_closed_count_is_not_reinterpreted_after_next_element(self):
        strata = frozenset({(20, 2, "halide")})
        elements = (
            SyntheticElement("Na", 11, (0, 1), 0.93, True),
            SyntheticElement("Cl", 17, (-1, 0), 3.16),
        )
        oracle = production_oracle(elements, max_atoms=20, max_species=2, allowed_strata=strata)
        expected = BruteForceFormulaLanguage(
            elements, max_atoms=20, max_species=2, allowed_strata=strata
        )
        for text in ("Na", "Na1", "Na10", "Na10Cl", "Na10Cl1", "Na10Cl10", "Na1Cl", "Na11", "Cl10Na10"):
            with self.subTest(text=text):
                self.assertEqual(oracle.is_prefix_viable(text), expected.is_prefix_viable(text))
                self.assertEqual(oracle.is_terminal_valid(text), expected.is_terminal_valid(text))

    def test_joint_target_family_charge_support_rejects_split_witness(self):
        strata = frozenset({(2, 2, "oxide")})
        oracle = production_oracle(CORE_ELEMENTS, max_atoms=5, max_species=3, allowed_strata=strata)
        expected = BruteForceFormulaLanguage(
            CORE_ELEMENTS, max_atoms=5, max_species=3, allowed_strata=strata
        )
        for text in ("", "Li", "Li1", "Fe", "FeO", "OFe", "F", "F2", "O", "Fe2", "Fe2O3"):
            with self.subTest(text=text):
                self.assertEqual(oracle.is_prefix_viable(text), expected.is_prefix_viable(text))
                self.assertEqual(oracle.is_terminal_valid(text), expected.is_terminal_valid(text))

    def test_alternative_oxidation_assignments_are_not_prematurely_collapsed(self):
        oracle = production_oracle(CORE_ELEMENTS, max_atoms=7, max_species=3)
        for text in ("FeO", "Fe2O3", "OFe", "O3Fe2", "FeF2", "FeF3"):
            with self.subTest(text=text):
                self.assertTrue(oracle.is_terminal_valid(text))
        for text in ("Fe3O4", "Fe2O4"):
            with self.subTest(text=text):
                self.assertFalse(oracle.is_terminal_valid(text))

    def test_invalid_pauling_order_is_not_a_chemical_witness(self):
        elements = (
            SyntheticElement("Fe", 26, (0, 2), 4.0, True),
            SyntheticElement("O", 8, (-2, 0), 1.0),
        )
        oracle = production_oracle(elements, max_atoms=4, max_species=2)
        self.assertFalse(oracle.is_prefix_viable("FeO"))
        self.assertFalse(oracle.is_terminal_valid("FeO"))
        self.assertTrue(oracle.is_terminal_valid("O"))

    def test_nonmetal_unary_branch_does_not_require_any_metal_vocabulary(self):
        elements = tuple(element for element in CORE_ELEMENTS if element.symbol in {"O", "F"})
        oracle = production_oracle(elements, max_atoms=4, max_species=2)
        for text in ("O", "O4", "F", "F2"):
            with self.subTest(text=text):
                self.assertTrue(oracle.is_terminal_valid(text))
        for text in ("OF", "O2F2", "FO"):
            with self.subTest(text=text):
                self.assertFalse(oracle.is_prefix_viable(text))

    def test_unknown_electronegativity_is_not_promoted_to_ionic_support(self):
        elements = (
            SyntheticElement("Fe", 26, (0, 2), None, True),
            SyntheticElement("O", 8, (-2, 0), 3.44),
        )
        oracle = production_oracle(elements, max_atoms=4, max_species=2)
        self.assertFalse(oracle.is_prefix_viable("FeO"))
        self.assertTrue(oracle.is_terminal_valid("Fe"))

    def test_max_arity_and_atom_count_are_hard_limits(self):
        oracle = production_oracle(CORE_ELEMENTS, max_atoms=2, max_species=1)
        for text in ("Li", "Li1", "Li2", "F", "Fe2", "O2"):
            with self.subTest(text=text):
                self.assertTrue(oracle.is_terminal_valid(text))
        for text in ("Li3", "FeO", "OFe", "LiFe", "F2Fe", "Li20"):
            with self.subTest(text=text):
                self.assertFalse(oracle.is_prefix_viable(text))

    def test_empty_target_support_does_not_fall_back_to_syntax_only(self):
        oracle = production_oracle(
            CORE_ELEMENTS, max_atoms=5, max_species=3, allowed_strata=frozenset()
        )
        for text in ("", "Fe", "Fe2O3", "Li", "O"):
            with self.subTest(text=text):
                self.assertFalse(oracle.is_prefix_viable(text))
                self.assertFalse(oracle.is_terminal_valid(text))


def production_processor(oracle, *, start_length=2, tokenizer=None):
    try:
        from crystal_dlm.r03_formula_bridge import R03C3FDFormulaLogitsProcessor
    except ModuleNotFoundError as exc:
        if exc.name == "crystal_dlm.r03_formula_bridge":
            raise unittest.SkipTest("new bridge prototype has not been written yet") from exc
        raise
    tokenizer = tokenizer or PieceTokenizer()
    return R03C3FDFormulaLogitsProcessor(
        tokenizer, oracle=oracle, start_length=start_length,
        eos_token_id=tokenizer.eos_token_id,
    )


class NativeTokenBridgeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.language = BruteForceFormulaLanguage(CORE_ELEMENTS, max_atoms=5, max_species=3)
        cls.oracle = production_oracle(CORE_ELEMENTS, max_atoms=5, max_species=3)

    def setUp(self):
        self.tokenizer = PieceTokenizer()
        self.processor = production_processor(self.oracle, tokenizer=self.tokenizer)

    def ids(self, *pieces):
        return [self.tokenizer.token_id(piece) for piece in pieces]

    def test_masks_only_chemical_support_in_active_formula_value(self):
        prefix = self.ids("formula", ":", " Fe")
        observed = set(self.processor.allowed_token_ids(prefix))
        expected = {
            token_id for token_id, piece in enumerate(self.tokenizer.pieces)
            if token_id != self.tokenizer.eos_token_id
            and self.language.native_field_fragment_legal(" Fe", piece)
        }
        self.assertEqual(observed, expected)

    def test_candidate_crossing_label_into_invalid_chemistry_is_rejected(self):
        allowed = self.processor.allowed_token_ids(self.ids("for"))
        self.assertIn(self.tokenizer.token_id("mula: Fe"), allowed)
        self.assertNotIn(self.tokenizer.token_id("mula: Xx\n"), allowed)
        # Outside the recognized formula field the original model can still
        # make formatting mistakes; this bridge does not rewrite its prompt.
        self.assertIn(self.tokenizer.token_id("Xx"), allowed)

    def test_whole_formula_and_rich_label_can_share_one_token(self):
        prefix = self.ids("formula", ":", " Fe")
        allowed = self.processor.allowed_token_ids(prefix)
        self.assertIn(self.tokenizer.token_id("2O3\nanion: oxide"), allowed)
        self.assertIsNone(self.processor.allowed_token_ids(
            [*prefix, self.tokenizer.token_id("2O3\nanion: oxide")]
        ))

    def test_complete_label_formula_and_newline_in_one_candidate(self):
        allowed = self.processor.allowed_token_ids([])
        token_id = self.tokenizer.token_id("formula: Fe2O3\nanion:")
        # None denotes a no-op when every possible crossing candidate is valid,
        # as well as after completion; either represents unmasked inclusion.
        if allowed is not None:
            self.assertIn(token_id, allowed)
        self.assertIsNone(self.processor.allowed_token_ids([token_id]))

    def test_native_field_padding_does_not_permit_ignored_chemical_suffixes(self):
        prefix = self.ids("formula", ":", " Fe", " ")
        allowed = self.processor.allowed_token_ids(prefix)
        self.assertIn(self.tokenizer.token_id("\n"), allowed)
        self.assertNotIn(self.tokenizer.token_id("2O3"), allowed)
        self.assertNotIn(self.tokenizer.token_id("O"), allowed)

    def test_leading_newline_does_not_disable_future_formula_constraint(self):
        prefix = self.ids("\n", "formula", ":", " Fe")
        allowed = self.processor.allowed_token_ids(prefix)
        self.assertIsNotNone(allowed)
        self.assertNotIn(self.tokenizer.token_id("21"), allowed)
        self.assertIn(self.tokenizer.token_id("2O3\nanion: oxide"), allowed)

    def test_one_two_letter_element_ambiguity_survives_tokenization(self):
        prefix = self.ids("formula", ":", " ", "F2F")
        allowed = self.processor.allowed_token_ids(prefix)
        self.assertIn(self.tokenizer.token_id("e"), allowed)
        self.assertNotIn(self.tokenizer.token_id("\n"), allowed)

    def test_newline_is_not_an_early_termination_escape(self):
        prefix = self.ids("formula", ":", " Fe", "2", "O")
        allowed = self.processor.allowed_token_ids(prefix)
        self.assertNotIn(self.tokenizer.token_id("\n"), allowed)
        self.assertNotIn(self.tokenizer.eos_token_id, allowed)
        self.assertIn(self.tokenizer.token_id("3"), allowed)

    def test_rich_fields_are_unconstrained_after_first_valid_formula_newline(self):
        prefix = self.ids("formula: Fe2O3\nanion:", "\nvolume: nonsense")
        self.assertIsNone(self.processor.allowed_token_ids(prefix))

    def test_past_newline_does_not_silently_certify_an_invalid_formula(self):
        prefix = self.ids("formula", ":", " Fe2O4\nanion:")
        allowed = self.processor.allowed_token_ids(prefix)
        self.assertIsNotNone(allowed)
        self.assertEqual(tuple(allowed), (self.tokenizer.eos_token_id,))

    def test_empty_support_is_explicit_eos_failure_not_unmasked_sampling(self):
        empty = production_oracle(
            CORE_ELEMENTS, max_atoms=5, max_species=3, allowed_strata=frozenset()
        )
        processor = production_processor(empty, tokenizer=self.tokenizer)
        allowed = processor.allowed_token_ids(self.ids("formula", ":", " "))
        self.assertEqual(tuple(allowed), (self.tokenizer.eos_token_id,))

    def test_tokens_that_decode_to_no_text_do_not_become_chemical_actions(self):
        allowed = self.processor.allowed_token_ids(self.ids("formula", ":", " Fe"))
        self.assertNotIn(self.tokenizer.eos_token_id, allowed)

    def test_tensor_mask_preserves_input_and_unmasked_rows_exactly(self):
        try:
            import torch
        except ImportError as exc:
            raise unittest.SkipTest("CPU torch is not installed in this interpreter") from exc
        # Both continuations have three tokens; two ignored prompt tokens are
        # deliberately text that would be invalid inside a formula.
        prompt = self.ids("Xx", "21")
        active = self.ids("formula", ":", " Fe")
        done = self.ids("formula: Fe2O3\nanion:", "\n", "\nvolume: nonsense")
        input_ids = torch.tensor([prompt + active, prompt + done], device="cpu")
        scores = torch.arange(2 * len(self.tokenizer), dtype=torch.float32, device="cpu").reshape(2, -1)
        before = scores.clone()
        result = self.processor(input_ids, scores)
        self.assertTrue(torch.equal(scores, before), "the caller's logits were mutated")
        self.assertTrue(torch.equal(result[1], before[1]), "rich-field logits changed")
        allowed = set(self.processor.allowed_token_ids(active))
        for token_id in range(len(self.tokenizer)):
            if token_id in allowed:
                self.assertEqual(float(result[0, token_id]), float(before[0, token_id]))
            else:
                self.assertTrue(bool(torch.isneginf(result[0, token_id])))

if __name__ == "__main__":
    unittest.main()
