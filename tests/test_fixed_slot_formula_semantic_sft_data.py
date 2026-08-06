import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.fixed_slot import arrays_to_answer, build_special_tokens
from scripts.build_fixed_slot_formula_semantic_sft_data import (
    build_formula_payload,
    build_formula_prompt,
    main as build_main,
)


class FixedSlotFormulaSemanticDataTest(unittest.TestCase):
    def make_row(self):
        answer, _ = arrays_to_answer(
            lengths=[3.1, 3.1, 5.2],
            angles=[90.0, 90.0, 120.0],
            species=["Li", "Li", "O"],
            frac_coords=[
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [0.25, 0.25, 0.25],
            ],
        )
        return {
            "task": "unconditional",
            "prompt": "Generate fixed-slot crystal tokens:",
            "answer": answer,
            "text": "Generate fixed-slot crystal tokens:\n" + answer,
            "sample_weight": 0.7,
            "composition_reason": "charge_neutral_pauling_valid",
            "metadata": {"material_id": "toy-Li2O", "e_above_hull": 0.0},
        }

    def test_prompt_binds_formula_special_tokens_and_n(self):
        row = self.make_row()
        arrays = {
            "num_atoms": 3,
            "atom_types": [3, 3, 8],
        }
        payload = build_formula_payload(row, arrays)
        prompt = build_formula_prompt(payload)
        self.assertEqual(payload["full_formula"], "Li2O")
        self.assertEqual(payload["reduced_formula"], "Li2O")
        self.assertEqual(payload["atom_count_token"], "<N_003>")
        self.assertIn("<E_Li> x2", payload["special_formula"])
        self.assertIn("<E_O> x1", payload["special_formula"])
        self.assertIn("Li == <E_Li>", prompt)
        self.assertIn("O == <E_O>", prompt)
        self.assertIn("fixed_slot_crystal_tokens:", prompt)

    def test_builder_keeps_val_plain_and_adds_train_semantic_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            row = self.make_row()
            for split in ("train", "val", "test"):
                (input_dir / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (input_dir / "vocab_tokens.txt").write_text("\n".join(build_special_tokens()) + "\n", encoding="utf-8")

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "build_fixed_slot_formula_semantic_sft_data.py",
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--semantic-fraction",
                    "1.0",
                ]
                build_main()
            finally:
                sys.argv = old_argv

            train_rows = [
                json.loads(line)
                for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            val_rows = [
                json.loads(line)
                for line in (output_dir / "val.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(train_rows), 2)
            self.assertEqual(len(val_rows), 1)
            self.assertEqual(train_rows[0]["composition_reason"], "charge_neutral_pauling_valid")
            self.assertEqual(train_rows[0]["composition_bucket"], "strict")
            semantic = train_rows[1]
            self.assertEqual(semantic["task"], "formula_semantic_fixed_slot")
            self.assertEqual(semantic["answer"], row["answer"])
            self.assertEqual(semantic["sample_weight"], 0.7)
            self.assertEqual(semantic["composition_reason"], "charge_neutral_pauling_valid")
            self.assertEqual(semantic["composition_bucket"], "strict")
            self.assertEqual(semantic["formula_semantic"]["atom_count_token"], "<N_003>")
            summary = json.loads((output_dir / "formula_semantic_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["splits"]["train"]["semantic_rows"], 1)
            self.assertEqual(summary["splits"]["val"]["semantic_rows"], 0)

    def test_builder_can_filter_semantic_rows_by_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            row = self.make_row()
            row["composition_reason"] = "single_element_shortcut"
            for split in ("train", "val", "test"):
                (input_dir / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (input_dir / "vocab_tokens.txt").write_text("\n".join(build_special_tokens()) + "\n", encoding="utf-8")

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "build_fixed_slot_formula_semantic_sft_data.py",
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--semantic-fraction",
                    "1.0",
                    "--semantic-include-reasons",
                    "strict,charge_neutral_pauling_valid",
                ]
                build_main()
            finally:
                sys.argv = old_argv

            train_rows = [
                json.loads(line)
                for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(train_rows), 1)
            summary = json.loads((output_dir / "formula_semantic_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["splits"]["train"]["semantic_rows"], 0)
            self.assertEqual(summary["splits"]["train"]["semantic_include_reasons"], ["charge_neutral_pauling_valid", "strict"])


if __name__ == "__main__":
    unittest.main()
