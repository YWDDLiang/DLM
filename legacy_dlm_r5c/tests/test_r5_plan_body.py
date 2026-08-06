import unittest

from crystal_dlm.crysllmgen_text import CRYSLLMGEN_TEXT_PROMPT
from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer, parse_dynamic_answer
from crystal_dlm.r5_dynamic_length import exact_body_token_count, validate_answer_matches_plan
from crystal_dlm.r5_plan_body import (
    H1_RICH_PLAN_BODY_REPRESENTATION,
    H1_RICH_PLAN_FORMAT,
    R5C_FORMULA_END_PLAN_BODY_REPRESENTATION,
    R5C_FORMULA_END_PLAN_FORMAT,
    R5C_PLAN_BODY_REPRESENTATION,
    R5C_SEMANTIC_FORMULA_PLAN_BODY_REPRESENTATION,
    R5C_SEMANTIC_PLAN_FORMAT,
    build_plan_body_record,
    format_composition_plan,
    has_plan_end_marker,
    has_plan_tail_after_end_marker,
    parse_composition_plan,
    split_plan_body_answer,
)
from crystal_dlm.r5_plan_state import plan_state_from_arrays, validate_plan_state


class R5PlanBodyTests(unittest.TestCase):
    def make_arrays_and_plan(self):
        answer, _ = arrays_to_dynamic_answer(
            lengths=[3.1, 3.1, 5.2],
            angles=[90.0, 90.0, 120.0],
            species=["Li", "Li", "O"],
            frac_coords=[[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        )
        arrays = parse_dynamic_answer(answer, strict=True)
        return arrays, plan_state_from_arrays(arrays, metadata={"material_id": "toy-Li2O", "spacegroup.number": "194"})

    def test_record_uses_crysllmgen_prompt_and_single_answer_trajectory(self):
        arrays, plan = self.make_arrays_and_plan()
        record = build_plan_body_record(plan_state=plan, arrays=arrays)

        self.assertEqual(record["prompt"], CRYSLLMGEN_TEXT_PROMPT.rstrip())
        self.assertEqual(record["representation"], "dynamic_v1")
        self.assertEqual(record["r5_representation"], R5C_PLAN_BODY_REPRESENTATION)
        self.assertEqual(record["loss_profile"], "text")
        self.assertTrue(record["answer"].startswith("plan:\n"))
        self.assertIn("\nbody:\n", record["answer"])

        plan_text, body_answer = split_plan_body_answer(record["answer"])
        parsed_plan = parse_composition_plan(plan_text)
        self.assertTrue(validate_plan_state(parsed_plan).valid)
        self.assertEqual(parsed_plan["formula"], "Li2O")
        parsed_body = validate_answer_matches_plan(parsed_plan, body_answer)
        self.assertEqual(parsed_body["species"], ["Li", "Li", "O"])
        self.assertEqual(record["block_spans"]["body"]["semantic_length"], 19)

    def test_composition_text_plan_round_trip(self):
        _arrays, plan = self.make_arrays_and_plan()
        text = format_composition_plan(plan)
        self.assertEqual(text, "formula: Li2O")
        parsed = parse_composition_plan("plan:\n" + text + "\nbody:\nignored")
        self.assertEqual(parsed["elements"], ["Li", "O"])
        self.assertEqual(parsed["counts"], [2, 1])
        self.assertEqual(parsed["N"], 3)
        self.assertEqual(parsed["formula"], "Li2O")
        self.assertTrue(parsed["derived_counts_from_formula"])
        self.assertTrue(parsed["derived_n_from_formula"])
        self.assertEqual(exact_body_token_count(parsed), 7 + 4 * 3)

    def test_formula_text_plan_derives_counts_and_accepts_noncanonical_order(self):
        parsed = parse_composition_plan("formula: As4Te2Ir4Ho\ncounts: 999\nN: 999")
        self.assertEqual(parsed["elements"], ["As", "Te", "Ho", "Ir"])
        self.assertEqual(parsed["counts"], [4, 2, 1, 4])
        self.assertEqual(parsed["N"], 11)
        self.assertEqual(parsed["formula"], "As4Te2HoIr4")

    def test_formula_text_plan_rejects_invalid_formula(self):
        with self.assertRaises(ValueError):
            parse_composition_plan("formula: O8SrWb2")
        with self.assertRaises(ValueError):
            parse_composition_plan("formula: Li2O0")

    def test_formula_end_plan_round_trip_and_exact_length(self):
        _arrays, plan = self.make_arrays_and_plan()
        text = format_composition_plan(plan, plan_style=R5C_FORMULA_END_PLAN_FORMAT)
        self.assertEqual(text, "formula: Li2O\nend: plan")
        self.assertTrue(has_plan_end_marker(text))
        parsed = parse_composition_plan("plan:\n" + text + "\nbody:\nignored", plan_style=R5C_FORMULA_END_PLAN_FORMAT)
        self.assertEqual(parsed["elements"], ["Li", "O"])
        self.assertEqual(parsed["counts"], [2, 1])
        self.assertEqual(parsed["N"], 3)
        self.assertEqual(parsed["formula"], "Li2O")
        self.assertTrue(parsed["plan_end_marker_present"])
        self.assertEqual(parsed["plan_format"], R5C_FORMULA_END_PLAN_FORMAT)
        self.assertEqual(exact_body_token_count(parsed), 7 + 4 * 3)

    def test_formula_end_plan_rejects_missing_marker(self):
        with self.assertRaises(ValueError):
            parse_composition_plan("formula: Li2O", plan_style=R5C_FORMULA_END_PLAN_FORMAT)
        parsed = parse_composition_plan("formula: Li2O")
        self.assertEqual(parsed["formula"], "Li2O")

    def test_formula_end_plan_truncates_leaked_body_tail(self):
        text = "formula: Li2O\nend: plan\nbody:\n<N_003><LA_031><E_Li>"
        self.assertTrue(has_plan_end_marker(text))
        self.assertTrue(has_plan_tail_after_end_marker(text))
        parsed = parse_composition_plan(text, plan_style=R5C_FORMULA_END_PLAN_FORMAT)
        self.assertEqual(parsed["formula"], "Li2O")
        self.assertEqual(parsed["N"], 3)

    def test_semantic_formula_plan_round_trip(self):
        _arrays, plan = self.make_arrays_and_plan()
        text = format_composition_plan(plan, plan_style=R5C_SEMANTIC_PLAN_FORMAT)
        self.assertEqual(
            text,
            "family: oxide\narity: binary\nsize: tiny\nformula: Li2O",
        )
        parsed = parse_composition_plan(text, plan_style=R5C_SEMANTIC_PLAN_FORMAT)
        self.assertEqual(parsed["elements"], ["Li", "O"])
        self.assertEqual(parsed["counts"], [2, 1])
        self.assertEqual(parsed["N"], 3)
        self.assertEqual(parsed["family"], "oxide")
        self.assertEqual(parsed["arity"], "binary")
        self.assertEqual(parsed["size"], "tiny")
        self.assertEqual(parsed["generated_family"], "oxide")
        self.assertTrue(parsed["semantic_consistency"]["family_match_formula"])
        self.assertTrue(parsed["semantic_consistency"]["arity_match_formula"])
        self.assertTrue(parsed["semantic_consistency"]["size_match_formula"])

    def test_semantic_fields_are_diagnostics_not_execution_source(self):
        parsed = parse_composition_plan(
            "family: intermetallic\narity: ternary\nsize: large\nformula: Li2O",
            plan_style=R5C_SEMANTIC_PLAN_FORMAT,
        )
        self.assertEqual(parsed["elements"], ["Li", "O"])
        self.assertEqual(parsed["counts"], [2, 1])
        self.assertEqual(parsed["N"], 3)
        self.assertEqual(parsed["family"], "oxide")
        self.assertFalse(parsed["semantic_consistency"]["family_match_formula"])
        self.assertFalse(parsed["semantic_consistency"]["arity_match_formula"])
        self.assertFalse(parsed["semantic_consistency"]["size_match_formula"])

    def test_semantic_plan_body_record_uses_exact_length_body(self):
        arrays, plan = self.make_arrays_and_plan()
        record = build_plan_body_record(plan_state=plan, arrays=arrays, plan_style=R5C_SEMANTIC_PLAN_FORMAT)
        self.assertEqual(record["r5_representation"], R5C_SEMANTIC_FORMULA_PLAN_BODY_REPRESENTATION)
        self.assertEqual(record["plan_format"], R5C_SEMANTIC_PLAN_FORMAT)
        self.assertIn("family: oxide\narity: binary\nsize: tiny\nformula: Li2O", record["answer"])
        plan_text, body_answer = split_plan_body_answer(record["answer"])
        parsed_plan = parse_composition_plan(plan_text, plan_style=R5C_SEMANTIC_PLAN_FORMAT)
        parsed_body = validate_answer_matches_plan(parsed_plan, body_answer)
        self.assertEqual(parsed_body["num_atoms"], 3)
        self.assertEqual(record["block_spans"]["body"]["semantic_length"], 7 + 4 * 3)

    def test_h1_rich_plan_round_trip_uses_generated_fields(self):
        _arrays, plan = self.make_arrays_and_plan()
        text = format_composition_plan(plan, plan_style=H1_RICH_PLAN_FORMAT)
        self.assertIn("formula: Li2O", text)
        self.assertIn("anion: oxide", text)
        self.assertIn("charge:", text)
        self.assertIn("lattice: hexagonal", text)
        self.assertIn("spacegroup: sg_168_194", text)
        self.assertIn("volume: volpa_", text)
        self.assertIn("end: plan", text)
        parsed = parse_composition_plan(text, plan_style=H1_RICH_PLAN_FORMAT)
        self.assertEqual(parsed["elements"], ["Li", "O"])
        self.assertEqual(parsed["counts"], [2, 1])
        self.assertEqual(parsed["N"], 3)
        self.assertEqual(parsed["anion_framework"], "oxide")
        self.assertEqual(parsed["lattice_system"], "hexagonal")
        self.assertEqual(parsed["spacegroup_bucket"], "sg_168_194")
        self.assertTrue(parsed["rich_field_valid"])
        self.assertTrue(parsed["anion_match_formula"])
        self.assertIn("charge=", parsed["prototype_key"])

    def test_h1_rich_fields_do_not_override_formula_composition(self):
        parsed = parse_composition_plan(
            "\n".join(
                [
                    "formula: Li2O",
                    "anion: other",
                    "charge: all_metal",
                    "lattice: cubic",
                    "spacegroup: sg_195_230",
                    "volume: volpa_100_104",
                    "end: plan",
                ]
            ),
            plan_style=H1_RICH_PLAN_FORMAT,
        )
        self.assertEqual(parsed["elements"], ["Li", "O"])
        self.assertEqual(parsed["counts"], [2, 1])
        self.assertEqual(parsed["N"], 3)
        self.assertEqual(parsed["anion_framework"], "other")
        self.assertFalse(parsed["anion_match_formula"])
        self.assertEqual(parsed["charge_bucket"], "all_metal")
        self.assertEqual(parsed["lattice_system"], "cubic")

    def test_h1_rich_plan_rejects_missing_or_invalid_fields(self):
        with self.assertRaises(ValueError):
            parse_composition_plan("formula: Li2O\nend: plan", plan_style=H1_RICH_PLAN_FORMAT)
        with self.assertRaises(ValueError):
            parse_composition_plan(
                "formula: Li2O\nanion: oxide\ncharge: neutral_plausible\n"
                "lattice: magic\nspacegroup: sg_001_002\nvolume: volpa_005_009\nend: plan",
                plan_style=H1_RICH_PLAN_FORMAT,
            )

    def test_h1_rich_plan_body_record_uses_exact_length_body(self):
        arrays, plan = self.make_arrays_and_plan()
        record = build_plan_body_record(plan_state=plan, arrays=arrays, plan_style=H1_RICH_PLAN_FORMAT)
        self.assertEqual(record["r5_representation"], H1_RICH_PLAN_BODY_REPRESENTATION)
        self.assertEqual(record["plan_format"], H1_RICH_PLAN_FORMAT)
        self.assertIn("anion: oxide", record["answer"])
        plan_text, body_answer = split_plan_body_answer(record["answer"])
        parsed_plan = parse_composition_plan(plan_text, plan_style=H1_RICH_PLAN_FORMAT)
        parsed_body = validate_answer_matches_plan(parsed_plan, body_answer)
        self.assertEqual(parsed_body["num_atoms"], 3)
        self.assertEqual(record["block_spans"]["body"]["semantic_length"], 7 + 4 * 3)

    def test_formula_end_plan_body_record_uses_exact_length_body(self):
        arrays, plan = self.make_arrays_and_plan()
        record = build_plan_body_record(plan_state=plan, arrays=arrays, plan_style=R5C_FORMULA_END_PLAN_FORMAT)
        self.assertEqual(record["r5_representation"], R5C_FORMULA_END_PLAN_BODY_REPRESENTATION)
        self.assertEqual(record["plan_format"], R5C_FORMULA_END_PLAN_FORMAT)
        self.assertIn("formula: Li2O\nend: plan", record["answer"])
        plan_text, body_answer = split_plan_body_answer(record["answer"])
        parsed_plan = parse_composition_plan(plan_text, plan_style=R5C_FORMULA_END_PLAN_FORMAT)
        parsed_body = validate_answer_matches_plan(parsed_plan, body_answer)
        self.assertEqual(parsed_body["num_atoms"], 3)
        self.assertEqual(record["block_spans"]["body"]["semantic_length"], 7 + 4 * 3)

    def test_split_rejects_missing_blocks(self):
        with self.assertRaises(ValueError):
            split_plan_body_answer("plan:\nformula: Li2O")
        with self.assertRaises(ValueError):
            split_plan_body_answer("body:\n<N_001><LA_010>")


if __name__ == "__main__":
    unittest.main()
