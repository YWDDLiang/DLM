import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer, parse_dynamic_answer
from crystal_dlm.h1_formula_only_body import (
    H1G1_CONDITION_VIEWS,
    H1G1_ROBUST_BODY_REPRESENTATION,
    build_condition_view_body_prompt,
    build_condition_view_body_record,
    H1_FORMULA_ONLY_BODY_REPRESENTATION,
    build_formula_only_body_prompt,
    build_formula_only_body_record,
)
from crystal_dlm.r5_plan_state import plan_state_from_arrays


class H1FormulaOnlyBodyTests(unittest.TestCase):
    def make_arrays_and_plan(self):
        answer, _ = arrays_to_dynamic_answer(
            lengths=[3.1, 3.1, 5.2],
            angles=[90.0, 90.0, 120.0],
            species=["Li", "Li", "O"],
            frac_coords=[[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        )
        arrays = parse_dynamic_answer(answer, strict=True)
        plan = plan_state_from_arrays(arrays, metadata={"spacegroup.number": "194"})
        return arrays, plan

    def test_formula_only_prompt_contains_only_formula_derived_execution_fields(self):
        _arrays, plan = self.make_arrays_and_plan()
        prompt = build_formula_only_body_prompt(plan)
        self.assertIn("formula: Li2O", prompt)
        self.assertIn("elements: Li, O", prompt)
        self.assertIn("counts: 2, 1", prompt)
        self.assertIn("N: 3", prompt)
        self.assertIn("dynamic_crystal_body:", prompt)
        self.assertNotIn("charge_bucket", prompt)
        self.assertNotIn("lattice_system", prompt)
        self.assertNotIn("spacegroup_bucket", prompt)
        self.assertNotIn("volume_per_atom_bin", prompt)
        self.assertNotIn("prototype_key", prompt)

    def test_formula_only_record_is_exact_dynamic_body(self):
        arrays, plan = self.make_arrays_and_plan()
        record = build_formula_only_body_record(plan_state=plan, arrays=arrays)
        self.assertEqual(record["r5_representation"], H1_FORMULA_ONLY_BODY_REPRESENTATION)
        self.assertEqual(record["loss_profile"], "fixed_slot")
        self.assertEqual(record["answer_semantic_length"], 19)
        self.assertTrue(record["answer"].startswith("<N_003>"))
        self.assertNotIn("spacegroup_bucket", record["prompt"])

    def test_sampler_read_records_can_rebuild_formula_only_prompt(self):
        try:
            from scripts.sample_llada_r5_exact_length import read_plan_records
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                self.skipTest("local environment lacks torch; sampler import is covered in A800 preflight")
            raise
        _arrays, plan = self.make_arrays_and_plan()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plans.jsonl"
            path.write_text(
                '{"plan_state":' + json.dumps(plan) + ',"prompt":"full plan_state prompt should be ignored"}\n',
                encoding="utf-8",
            )
            records = read_plan_records(path, "prompt", body_prompt_style="formula_only")
        self.assertEqual(len(records), 1)
        self.assertIn("formula: Li2O", records[0]["prompt"])
        self.assertNotIn("full plan_state prompt", records[0]["prompt"])
        self.assertNotIn("spacegroup_bucket", records[0]["prompt"])

    def test_h1g1_condition_views_execute_formula_counts(self):
        _arrays, plan = self.make_arrays_and_plan()
        for view in H1G1_CONDITION_VIEWS:
            prompt = build_condition_view_body_prompt(plan, condition_view=view)
            self.assertIn(f"condition_view: {view}", prompt)
            self.assertIn("formula: Li2O", prompt)
            self.assertIn("elements: Li, O", prompt)
            self.assertIn("counts: 2, 1", prompt)
            self.assertIn("N: 3", prompt)
            self.assertIn("dynamic_crystal_body:", prompt)

    def test_h1g1_condition_view_record_is_dynamic_body(self):
        arrays, plan = self.make_arrays_and_plan()
        record = build_condition_view_body_record(
            plan_state=plan,
            arrays=arrays,
            condition_view="formula-volume-sg",
            sample_weight=0.5,
        )
        self.assertEqual(record["r5_representation"], H1G1_ROBUST_BODY_REPRESENTATION)
        self.assertEqual(record["condition_view"], "formula-volume-sg")
        self.assertEqual(record["sample_weight"], 0.5)
        self.assertTrue(record["answer"].startswith("<N_003>"))
        self.assertIn("spacegroup:", record["prompt"])
        self.assertNotIn("0.25 0.25 0.25", record["prompt"])

    def test_h1g1_condition_view_prompt_jsonl_conversion(self):
        from scripts.build_h1g1_condition_view_prompts import main as convert_main
        import sys

        _arrays, plan = self.make_arrays_and_plan()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "plans.jsonl"
            dst = root / "condition.jsonl"
            src.write_text(json.dumps({"sample_idx": 5, "plan_state": plan, "prompt": "old prompt"}) + "\n", encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = [
                    "build_h1g1_condition_view_prompts.py",
                    "--input-jsonl",
                    str(src),
                    "--output-jsonl",
                    str(dst),
                    "--condition-view",
                    "formula-volume-only",
                ]
                convert_main()
            finally:
                sys.argv = old_argv
            row = json.loads(dst.read_text(encoding="utf-8"))
        self.assertEqual(row["condition_view"], "formula-volume-only")
        self.assertIn("condition_view: formula-volume-only", row["prompt"])
        self.assertIn("formula: Li2O", row["prompt"])
        self.assertNotIn("old prompt", row["prompt"])


if __name__ == "__main__":
    unittest.main()
