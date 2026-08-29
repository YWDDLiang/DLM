import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_d3po_pairs", ROOT / "scripts/build_d3po_pairs.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildD3POPairsTest(unittest.TestCase):
    def test_minimal_prompt_is_canonical_and_certified(self):
        plan = {
            "N": 3,
            "elements": ["O", "Li"],
            "counts": [1, 2],
            "anion_framework": "oxide",
            "charge_bucket": "neutral_plausible",
        }
        prompt, reason = MODULE.minimal_prompt_from_plan(plan)
        self.assertEqual(reason, "ok")
        self.assertEqual(MODULE.composition_identity(plan), "Li:2|O:1")
        self.assertEqual(MODULE.chemsys(plan), "Li-O")
        self.assertIn('"elements":["Li","O"]', prompt)
        self.assertTrue(prompt.endswith("\ndynamic_crystal_body:"))

    def test_unsupported_historical_charge_is_not_relabelled(self):
        plan = {
            "N": 2,
            "elements": ["Li", "O"],
            "counts": [1, 1],
            "anion_framework": "oxide",
            "charge_bucket": "charge_fail",
        }
        prompt, reason = MODULE.minimal_prompt_from_plan(plan)
        self.assertIsNone(prompt)
        self.assertEqual(reason, "unsupported_charge:charge_fail")

    def test_pair_weights_sum_to_one_per_composition(self):
        outcomes = [
            {
                "answer": "a",
                "energy_per_atom": -2.00,
                "source": "s0",
            },
            {
                "answer": "b",
                "energy_per_atom": -1.97,
                "source": "s1",
            },
            {
                "answer": "c",
                "energy_per_atom": -1.90,
                "source": "s2",
            },
        ]
        pairs = MODULE.build_pairs_for_composition(outcomes)
        self.assertEqual(len(pairs), 3)
        self.assertAlmostEqual(sum(row["pair_weight"] for row in pairs), 1.0)
        self.assertTrue(all(row["soft_target"] > 0.5 for row in pairs))
        self.assertTrue(
            all(
                row["winner"]["energy_per_atom"]
                < row["loser"]["energy_per_atom"]
                for row in pairs
            )
        )

    def test_duplicate_body_text_does_not_create_false_pair(self):
        outcomes = [
            {"answer": "same", "energy_per_atom": -2.0, "source": "s0"},
            {"answer": "same", "energy_per_atom": -1.9, "source": "s1"},
            {"answer": "other", "energy_per_atom": -1.8, "source": "s2"},
        ]
        pairs = MODULE.build_pairs_for_composition(outcomes)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["winner"]["answer"], "same")
        self.assertAlmostEqual(pairs[0]["winner"]["energy_per_atom"], -1.95)
        self.assertEqual(pairs[0]["winner"]["replicate_count"], 2)

    def test_chemsys_split_is_deterministic(self):
        self.assertEqual(
            MODULE.chemsys_split("Li-O"), MODULE.chemsys_split("Li-O")
        )

    def test_preference_identity_preserves_exact_multiplicity_and_n(self):
        primitive = {
            "N": 2,
            "elements": ["Ca", "O"],
            "counts": [1, 1],
        }
        supercell = {
            "N": 4,
            "elements": ["Ca", "O"],
            "counts": [2, 2],
        }
        self.assertEqual(MODULE.composition_identity(primitive), "Ca:1|O:1")
        self.assertEqual(MODULE.composition_identity(supercell), "Ca:2|O:2")
        self.assertNotEqual(
            MODULE.composition_identity(primitive),
            MODULE.composition_identity(supercell),
        )


if __name__ == "__main__":
    unittest.main()
