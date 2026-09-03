import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "label_potential_closure_actions",
    ROOT / "scripts" / "label_potential_closure_actions.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import potential closure labeler")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def prediction(energy):
    return {
        "e": energy,
        "f": np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        "s": np.eye(3),
    }


class PotentialClosureLabelerTest(unittest.TestCase):
    def test_finite_prediction_reports_energy_force_and_stress(self):
        value = MODULE.finite_prediction(prediction(-2.0))
        self.assertIsNotNone(value)
        self.assertEqual(value["raw_chgnet_energy_eV_per_atom"], -2.0)
        self.assertEqual(value["raw_force_max_eV_per_A"], 1.0)
        self.assertAlmostEqual(value["raw_stress_frobenius_GPa"], np.sqrt(3.0))

    def test_attach_labels_uses_one_mev_informative_threshold(self):
        group = {
            "group_idx": 0,
            "stratum": "mp20_clean_cell",
            "candidates": [
                {"candidate_idx": 0, "action_tokens": ["a"] * 6},
                {"candidate_idx": 1, "action_tokens": ["b"] * 6},
            ],
        }
        informative = MODULE.attach_labels(
            [group],
            [prediction(-2.0), prediction(-1.998)],
        )[0]
        flat = MODULE.attach_labels(
            [group],
            [prediction(-2.0), prediction(-1.9995)],
        )[0]
        self.assertTrue(informative["informative"])
        self.assertFalse(flat["informative"])

    def test_unknown_prediction_blocks_group_without_becoming_negative(self):
        group = {
            "group_idx": 0,
            "stratum": "on_policy_site",
            "candidates": [
                {"candidate_idx": 0, "action_tokens": ["a"] * 3},
                {"candidate_idx": 1, "action_tokens": ["b"] * 3},
            ],
        }
        result = MODULE.attach_labels([group], [prediction(-2.0), None])[0]
        self.assertFalse(result["informative"])
        self.assertFalse(result["candidates"][1]["raw_chgnet_known"])


if __name__ == "__main__":
    unittest.main()
