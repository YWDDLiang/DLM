import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "decide_spad_basin_preflight", ROOT / "scripts/decide_spad_basin_preflight.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def report(median=20.0, above10=70, coverage=0.98, known=100):
    return {
        "schema": "spad_basin_preflight_value_final_v1",
        "groups": 128,
        "coverage": {"K10": {"coverage": coverage}},
        "headroom": {
            "K10": {
                "groups_with_known_no_op_and_candidate": known,
                "headroom_meV_per_atom": {"median": median},
                "groups_above_headroom_threshold_meV": {"10": above10},
            }
        },
        "kendall_tau_b": {"K10_vs_K20": {"pooled_tau_b": 0.8}},
    }


def groups():
    return [
        {
            "candidates": [
                {"source": "no_op", "terminal_legal": True, "E0_energy_eV_per_atom": -1.0, "K10_energy_eV_per_atom": -1.0},
                {"source": "physics_downhill", "terminal_legal": True, "E0_energy_eV_per_atom": -0.99, "K10_energy_eV_per_atom": -1.1},
            ]
        }
        for _ in range(128)
    ]


class DecisionTest(unittest.TestCase):
    def test_material_headroom_authorizes_primary_k10(self):
        value = MODULE.decide(report(), groups())
        self.assertTrue(value["authorized"])
        self.assertEqual(value["primary_route"], "k10_basin_consistent")
        self.assertEqual(value["K10_selected_action_E0_delta_eV_per_atom"]["count"], 128)

    def test_missing_headroom_rejects_without_changing_candidates(self):
        value = MODULE.decide(report(median=0.0, above10=0), groups())
        self.assertFalse(value["authorized"])
        self.assertTrue(value["no_parameter_or_candidate_change"])


if __name__ == "__main__":
    unittest.main()
