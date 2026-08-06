import csv
import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.fixed_slot import arrays_to_answer, build_special_tokens
from crystal_dlm.r5_conditioning import (
    build_r5_prompt,
    build_z_payload_from_answer,
    ehull_tier,
    validate_z_matches_answer,
)
from scripts.build_r5_z_prompt_sft_data import main as build_main


class R5ConditioningTest(unittest.TestCase):
    def make_answer(self):
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
        return answer

    def test_z_prompt_keeps_fixed_slot_body_as_answer(self):
        answer = self.make_answer()
        z = build_z_payload_from_answer(
            answer,
            metadata={
                "material_id": "toy-Li2O",
                "e_above_hull": 0.0,
                "spacegroup.number": 194,
                "spacegroup.number.conv": 194,
            },
        )
        validate_z_matches_answer(z, answer)
        prompt = build_r5_prompt(z)
        self.assertEqual(z["full_formula"], "Li2O")
        self.assertEqual(z["atom_count_token"], "<N_003>")
        self.assertEqual(z["ehull_tier"], "strict_anchor")
        self.assertIn("<E_Li> x2", prompt)
        self.assertIn("<E_O> x1", prompt)
        self.assertIn("prototype_key:", prompt)
        self.assertIn("Do not output a physical header", prompt)
        self.assertNotIn("<H_START>", answer)

    def test_ehull_tiers(self):
        self.assertEqual(ehull_tier(0.0), "strict_anchor")
        self.assertEqual(ehull_tier(-0.01), "strict_anchor")
        self.assertEqual(ehull_tier(0.05), "meta_anchor")
        self.assertEqual(ehull_tier(0.1), "higher_ehull")
        self.assertEqual(ehull_tier(None), "unknown_ehull")

    def test_builder_outputs_conditioned_rows_and_prompt_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            csv_dir = root / "csv"
            output_dir = root / "output"
            prototype_jsonl = root / "prototypes" / "mp20_stable_prototype_library.jsonl"
            input_dir.mkdir()
            csv_dir.mkdir()
            answer = self.make_answer()
            row = {
                "task": "unconditional",
                "prompt": "Generate fixed-slot crystal tokens:",
                "answer": answer,
                "text": "Generate fixed-slot crystal tokens:\n" + answer,
                "metadata": {"material_id": "toy-Li2O"},
            }
            for split in ("train", "val", "test"):
                (input_dir / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
                with (csv_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "material_id",
                            "pretty_formula",
                            "e_above_hull",
                            "spacegroup.number",
                            "spacegroup.number.conv",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "material_id": "toy-Li2O",
                            "pretty_formula": "Li2O",
                            "e_above_hull": "0.0",
                            "spacegroup.number": "194",
                            "spacegroup.number.conv": "194",
                        }
                    )
            (input_dir / "vocab_tokens.txt").write_text("\n".join(build_special_tokens()) + "\n", encoding="utf-8")

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "build_r5_z_prompt_sft_data.py",
                    "--input-dir",
                    str(input_dir),
                    "--csv-dir",
                    str(csv_dir),
                    "--output-dir",
                    str(output_dir),
                    "--prototype-jsonl",
                    str(prototype_jsonl),
                    "--replay-fraction",
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
            self.assertEqual(len(train_rows), 2)
            conditioned = [row for row in train_rows if row["selection_role"] == "r5_z_conditioned"][0]
            replay = [row for row in train_rows if row["selection_role"] == "r5_unconditional_replay"][0]
            self.assertEqual(conditioned["answer"], answer)
            self.assertEqual(conditioned["answer_token_count"], 107)
            self.assertIn("physical plan z", conditioned["prompt"])
            self.assertEqual(replay["answer"], answer)
            prototypes = [
                json.loads(line)
                for line in prototype_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(len(prototypes), 3)
            self.assertEqual(prototypes[0]["ehull_tier"], "strict_anchor")
            summary = json.loads((output_dir / "r5_z_prompt_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["answer_token_count"], 107)
            self.assertGreaterEqual(summary["prompt_pool_rows"], 1)


if __name__ == "__main__":
    unittest.main()

