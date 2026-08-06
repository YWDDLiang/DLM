import unittest

from crystal_dlm.crysllmgen_text import CRYSLLMGEN_TEXT_PROMPT
from crystal_dlm.h1_llm_planner import (
    H1_PLANNER_PROMPT_VERSION,
    H1_PLANNER_PROMPT_STYLE_FORMULA_PREFILL,
    H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
    build_planner_messages,
    build_planner_user_prompt,
    canonical_plan_record,
    canonical_plan_record_for_style,
    clean_generated_plan_text,
    format_planner_prompt,
    teacher_formula_answer,
)
from scripts.build_h1_llm_formula_sft_data import (
    build_records_for_plan,
    corrupted_rich_plan,
    formula_count_answer,
)


class H1LlmPlannerTests(unittest.TestCase):
    def test_prompt_is_crysllmgen_formula_only(self):
        prompt = build_planner_user_prompt(sample_idx=7)
        self.assertIn(CRYSLLMGEN_TEXT_PROMPT.rstrip(), prompt)
        self.assertIn("formula:", prompt)
        self.assertIn("end: plan", prompt)
        self.assertIn("sample_id: 7", prompt)
        self.assertNotIn("coordinates:", prompt.lower())
        messages = build_planner_messages(sample_idx=7)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

    def test_canonical_plan_record_derives_body_prompt(self):
        record = canonical_plan_record("formula: Li2O\nend: plan", sample_idx=3)
        self.assertEqual(record["sample_idx"], 3)
        self.assertEqual(record["plan_text"], "formula: Li2O\nend: plan")
        self.assertEqual(record["plan_state"]["elements"], ["Li", "O"])
        self.assertEqual(record["plan_state"]["counts"], [2, 1])
        self.assertEqual(record["plan_state"]["N"], 3)
        self.assertTrue(record["plan_end_marker_present"])
        self.assertIn('"formula":"Li2O"', record["prompt"])
        self.assertTrue(record["prompt"].endswith("\n"))

    def test_canonical_plan_record_rejects_missing_marker(self):
        with self.assertRaises(ValueError):
            canonical_plan_record("formula: Li2O")

    def test_canonical_plan_record_reports_leaked_body_tail(self):
        record = canonical_plan_record("formula: Li2O\nend: plan\nbody:\n<N_003><LA_031>", sample_idx=1)
        self.assertTrue(record["plan_tail_after_end_marker"])
        self.assertEqual(record["plan_state"]["formula"], "Li2O")

    def test_clean_generated_plan_text_keeps_only_generated_plan_boundary(self):
        text = (
            "System: copied prompt\n"
            "Formula: Na2Cl2 end: plan. Is it correct?\n"
            "User: Thank you"
        )
        cleaned = clean_generated_plan_text(text)
        self.assertEqual(cleaned, "Formula: Na2Cl2\nend: plan")
        record = canonical_plan_record(cleaned)
        self.assertEqual(record["plan_state"]["formula"], "Na2Cl2")

    def test_clean_generated_plan_text_does_not_invent_missing_formula(self):
        text = "Assistant: Here is a plan.\nend: plan"
        cleaned = clean_generated_plan_text(text)
        self.assertEqual(cleaned, "Assistant: Here is a plan.\nend: plan")
        with self.assertRaises(ValueError):
            canonical_plan_record(cleaned)

    def test_formula_prefill_style_supervises_only_formula_value_suffix(self):
        answer = teacher_formula_answer(
            {"elements": ["Li", "O"], "counts": [2, 1], "N": 3},
            prompt_style=H1_PLANNER_PROMPT_STYLE_FORMULA_PREFILL,
        )
        self.assertEqual(answer, "Li2O\nend: plan")
        prompt = format_planner_prompt(None, prompt_style=H1_PLANNER_PROMPT_STYLE_FORMULA_PREFILL)
        self.assertTrue(prompt.endswith("Assistant: formula: "))
        cleaned = clean_generated_plan_text(
            "Li2O\nend: plan\nAssistant: extra",
            prompt_style=H1_PLANNER_PROMPT_STYLE_FORMULA_PREFILL,
        )
        self.assertEqual(cleaned, "formula: Li2O\nend: plan")
        self.assertEqual(canonical_plan_record(cleaned)["plan_state"]["formula"], "Li2O")

    def test_teacher_formula_answer_is_formula_end(self):
        answer = teacher_formula_answer({"elements": ["Li", "O"], "counts": [2, 1], "N": 3})
        self.assertEqual(answer, "formula: Li2O\nend: plan")

    def test_rich_prompt_requests_teacher_fields(self):
        prompt = build_planner_user_prompt(sample_idx=4, prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_PLAN)
        self.assertIn("formula:", prompt)
        self.assertIn("anion:", prompt)
        self.assertIn("charge:", prompt)
        self.assertIn("lattice:", prompt)
        self.assertIn("spacegroup:", prompt)
        self.assertIn("volume:", prompt)
        self.assertIn("end: plan", prompt)
        self.assertIn("sample_id: 4", prompt)
        self.assertNotIn("coordinates:", prompt.lower())

    def test_rich_prompt_can_omit_sample_id_for_de_novo_prior(self):
        prompt = build_planner_user_prompt(sample_idx=None, prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_PLAN)
        self.assertIn("formula:", prompt)
        self.assertIn("end: plan", prompt)
        self.assertNotIn("sample_id:", prompt)

    def test_rich_teacher_answer_and_canonical_record_feed_full_plan_state(self):
        plan = {
            "elements": ["Li", "O"],
            "counts": [2, 1],
            "N": 3,
            "anion_framework": "oxide",
            "charge_bucket": "neutral_plausible",
            "lattice_system": "hexagonal",
            "spacegroup_bucket": "sg_168_194",
            "volume_per_atom_bin": "volpa_005_009",
        }
        answer = teacher_formula_answer(plan, prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_PLAN)
        self.assertEqual(
            answer,
            "formula: Li2O\n"
            "anion: oxide\n"
            "charge: neutral_plausible\n"
            "lattice: hexagonal\n"
            "spacegroup: sg_168_194\n"
            "volume: volpa_005_009\n"
            "end: plan",
        )
        record = canonical_plan_record_for_style(
            answer,
            sample_idx=2,
            prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
        )
        self.assertEqual(record["plan_state"]["formula"], "Li2O")
        self.assertEqual(record["plan_state"]["elements"], ["Li", "O"])
        self.assertEqual(record["plan_state"]["lattice_system"], "hexagonal")
        self.assertEqual(record["plan_state"]["spacegroup_bucket"], "sg_168_194")
        self.assertIn('"lattice_system":"hexagonal"', record["prompt"])
        self.assertIn('"spacegroup_bucket":"sg_168_194"', record["prompt"])

    def test_prompt_version_constant(self):
        self.assertEqual(H1_PLANNER_PROMPT_VERSION, "h1_llm_formula_planner_v1")

    def test_h1a3_corrupted_plan_and_formula_count_answer(self):
        plan = {
            "elements": ["Li", "O"],
            "counts": [2, 1],
            "N": 3,
            "formula": "Li2O",
            "anion_framework": "oxide",
            "charge_bucket": "neutral_plausible",
            "lattice_system": "hexagonal",
            "spacegroup_bucket": "sg_168_194",
            "volume_per_atom_bin": "volpa_005_009",
        }
        corrupted, labels = corrupted_rich_plan(plan, row_idx=1)
        self.assertIn("formula: Li2O", corrupted)
        self.assertIn("end: plan", corrupted)
        self.assertIn("charge_formula_mismatch", labels)
        self.assertEqual(
            formula_count_answer(plan),
            "formula: Li2O\nelements: Li, O\ncounts: 2, 1\nN: 3\nend: check",
        )

    def test_h1a3_records_have_expected_sample_types_and_no_sample_id(self):
        plan = {
            "elements": ["Li", "O"],
            "counts": [2, 1],
            "N": 3,
            "formula": "Li2O",
            "anion_framework": "oxide",
            "charge_bucket": "neutral_plausible",
            "lattice_system": "hexagonal",
            "spacegroup_bucket": "sg_168_194",
            "volume_per_atom_bin": "volpa_005_009",
        }
        records = build_records_for_plan(
            split="train",
            row_idx=7,
            plan_state=plan,
            metadata={},
            tokenizer=None,
            prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
            include_sample_id=False,
            sample_types=["direct_plan", "correct_plan", "consistency_explain", "formula_count_check"],
            weights={
                "direct_plan": 1.0,
                "correct_plan": 0.5,
                "consistency_explain": 0.25,
                "formula_count_check": 0.25,
            },
        )
        self.assertEqual([row["h1a3_sample_type"] for row in records], [
            "direct_plan",
            "correct_plan",
            "consistency_explain",
            "formula_count_check",
        ])
        self.assertTrue(all(not row["include_sample_id"] for row in records))
        self.assertEqual(records[0]["sample_weight"], 1.0)
        self.assertEqual(records[1]["sample_weight"], 0.5)
        self.assertIn("violation_labels:", records[2]["answer"])
        self.assertIn("end: check", records[3]["answer"])
        self.assertNotIn("sample_id:", records[0]["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
