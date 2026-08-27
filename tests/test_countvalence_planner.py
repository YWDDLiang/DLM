import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.r5_plan_state import parse_countvalence_plan_state, plan_state_to_countvalencefields
from crystal_dlm.valence_assignment import annotate_plan_with_valence, assign_crysvcd_valences

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "h1a2_audit_countvalence_plan",
    PROJECT_ROOT / "scripts" / "audit_countvalence_plan.py",
)
if _AUDIT_SPEC is None or _AUDIT_SPEC.loader is None:
    raise RuntimeError("cannot load audit_countvalence_plan.py")
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
training_gate = _AUDIT_MODULE.training_gate


class CountValencePlannerTest(unittest.TestCase):
    def test_roundtrip_preserves_composition_and_soft_properties(self) -> None:
        plan = {
            "formula": "Li2O",
            "N": 3,
            "elements": ["H", "Li", "O"],
            "counts": [0, 2, 1],
            "oxidation_candidates": [0, 1, -2],
            "charge_bucket": "neutral_plausible",
            "lattice_system": "cubic",
            "spacegroup_bucket": "sg_195_230",
            "volume_per_atom_bin": "volpa_010_014",
        }
        text = plan_state_to_countvalencefields(plan)
        rebuilt = parse_countvalence_plan_state(text)
        self.assertEqual(rebuilt["formula"], "Li2O")
        self.assertEqual(rebuilt["N"], 3)
        self.assertEqual(rebuilt["elements"], ["Li", "O"])
        self.assertEqual(rebuilt["counts"], [2, 1])
        self.assertEqual(rebuilt["generated_charge_sum"], 0)
        self.assertEqual(rebuilt["lattice_system"], "cubic")
        self.assertEqual(rebuilt["spacegroup_bucket"], "sg_195_230")
        self.assertEqual(rebuilt["volume_per_atom_bin"], "volpa_010_014")

    def test_mixed_valence_assignment_balances_magnetite(self) -> None:
        assignment = assign_crysvcd_valences(["O", "Fe"], [4, 3])
        self.assertTrue(assignment["assigned"])
        self.assertEqual(assignment["mode"], "ionic_mixed")
        self.assertEqual(assignment["charge_sum"], 0)
        fe_species = [
            (value["oxidation_state"], value["count"])
            for value in assignment["species"]
            if value["element"] == "Fe"
        ]
        self.assertEqual(fe_species, [(2, 1), (3, 2)])

    def test_mixed_valence_roundtrip_preserves_species_and_composition(self) -> None:
        plan = {
            "formula": "O4Fe3",
            "N": 7,
            "elements": ["O", "Fe"],
            "counts": [4, 3],
            "charge_bucket": "neutral_plausible",
            "lattice_system": "cubic",
            "spacegroup_bucket": "sg_195_230",
            "volume_per_atom_bin": "volpa_010_014",
        }
        annotated = annotate_plan_with_valence(plan)
        text = plan_state_to_countvalencefields(annotated)
        rebuilt = parse_countvalence_plan_state(text)
        self.assertEqual(rebuilt["elements"], ["O", "Fe"])
        self.assertEqual(rebuilt["counts"], [4, 3])
        self.assertEqual(rebuilt["generated_charge_sum"], 0)
        self.assertTrue(rebuilt["count_valence_validator"]["mixed_valence"])
        self.assertEqual(len(rebuilt["valence_species"]), 3)

    def test_unsupported_element_is_disclosed(self) -> None:
        assignment = assign_crysvcd_valences(["O", "La"], [3, 2])
        self.assertFalse(assignment["assigned"])
        self.assertEqual(assignment["reason"], "unsupported_elements")
        self.assertEqual(assignment["unsupported_elements"], ["La"])

    def test_training_gate_requires_valence_coverage_and_raw_match(self) -> None:
        def row(name: str, coverage: float) -> dict:
            return {
                "name": name,
                "rates": {
                    "serialization": 1.0,
                    "parse": 1.0,
                    "exact_composition_roundtrip": 1.0,
                    "exact_soft_field_roundtrip": 1.0,
                    "valence_assignment_known": coverage,
                    "generated_charge_neutral": coverage,
                },
            }

        rejected = training_gate(
            [row("train", 0.26), row("val", 0.25), row("raw1000", 0.0)]
        )
        self.assertFalse(rejected["candidate_training_authorized"])
        accepted = training_gate(
            [row("train", 0.98), row("val", 0.97), row("raw1000", 0.96)]
        )
        self.assertTrue(accepted["candidate_training_authorized"])


if __name__ == "__main__":
    unittest.main()
