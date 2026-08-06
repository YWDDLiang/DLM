import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.chemical_plan import build_plan_conditioned_prompt, chemical_plan_from_symbols
from crystal_dlm.fixed_slot import arrays_to_answer
from scripts.build_fixed_slot_chemical_plan_sft_data import build_split


class FixedSlotChemicalPlanDataTest(unittest.TestCase):
    def test_plan_mentions_charge_balanced_oxidation_states(self):
        payload = chemical_plan_from_symbols(["Li", "Li", "O"], metadata={"e_above_hull": "0.0"})

        self.assertIn("formula: Li2O", payload["plan"])
        self.assertIn("Li+1", payload["plan"])
        self.assertIn("O-2", payload["plan"])
        self.assertIn("crystal tokens:", payload["plan"])

    def test_conditioned_prompt_keeps_boundary(self):
        prompt = build_plan_conditioned_prompt("formula: Li2O")

        self.assertIn("formula: Li2O", prompt)
        self.assertTrue(prompt.rstrip().endswith("crystal tokens:"))

    def test_build_split_writes_text_and_fixed_slot_profiles(self):
        answer, _ = arrays_to_answer(
            lengths=[3.0, 3.0, 3.0],
            angles=[90.0, 90.0, 90.0],
            species=["Li", "Li", "O"],
            frac_coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
            separator="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "train.jsonl"
            output_path = tmp_path / "out.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "prompt": "Generate:",
                        "answer": answer,
                        "metadata": {"e_above_hull": "0.0"},
                        "sample_weight": 0.7,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            stats = build_split(
                split="train",
                input_path=input_path,
                output_path=output_path,
                tokenizer=None,
                rng=__import__("random").Random(1),
                plan_row_fraction=1.0,
                train_only_plan_rows=True,
                limit=None,
            )
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(stats["fixed_slot_rows"], 1)
        self.assertEqual(stats["chemical_plan_rows"], 1)
        self.assertEqual(rows[0]["loss_profile"], "fixed_slot")
        self.assertEqual(rows[1]["loss_profile"], "text")
        self.assertIn("crystal tokens:", rows[0]["prompt"])
        self.assertIn("formula: Li2O", rows[1]["answer"])


if __name__ == "__main__":
    unittest.main()
