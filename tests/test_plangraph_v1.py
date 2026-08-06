from copy import deepcopy
import json
from pathlib import Path
import unittest

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer, parse_dynamic_answer
from crystal_dlm.plangraph_v1 import (
    PLANGRAPH_SITE_GROUP_STRATEGY,
    PLANGRAPH_VERSION,
    PlanGraphError,
    build_plangraph_sft_records,
    plangraph_from_plan_state,
    plangraph_from_record,
    plangraph_to_json,
    validate_plangraph,
)
from crystal_dlm.r5_plan_state import plan_state_from_arrays


class PlanGraphV1Tests(unittest.TestCase):
    def make_arrays_plan_answer(self):
        answer, _diagnostics = arrays_to_dynamic_answer(
            lengths=[3.1, 3.1, 5.2],
            angles=[90.0, 90.0, 120.0],
            species=["Li", "O", "Li"],
            frac_coords=[
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5],
            ],
        )
        arrays = parse_dynamic_answer(answer, strict=True)
        plan = plan_state_from_arrays(
            arrays,
            metadata={
                "material_id": "mp-secret-id",
                "spacegroup.number": 194,
                "e_above_hull": 9.87654321,
                "formation_energy_per_atom": -123.456789,
            },
        )
        return arrays, plan, answer

    def test_converter_is_deterministic_and_uses_actual_site_order(self):
        arrays, plan, _answer = self.make_arrays_plan_answer()
        graph = plangraph_from_plan_state(
            plan,
            site_species=arrays["species"],
        )
        reordered_plan = dict(reversed(list(plan.items())))
        repeated = plangraph_from_plan_state(
            reordered_plan,
            site_species=arrays["species"],
        )

        self.assertEqual(graph, repeated)
        self.assertEqual(graph["schema_version"], PLANGRAPH_VERSION)
        self.assertEqual(
            graph["site_group_strategy"],
            PLANGRAPH_SITE_GROUP_STRATEGY,
        )
        self.assertEqual(graph["composition"]["formula"], "Li2O")
        self.assertEqual(graph["site_groups"][0]["element"], "Li")
        self.assertEqual(graph["site_groups"][0]["slot_indices"], [0, 2])
        self.assertEqual(graph["site_groups"][1]["element"], "O")
        self.assertEqual(graph["site_groups"][1]["slot_indices"], [1])
        self.assertTrue(validate_plangraph(graph).valid)

    def test_converter_does_not_serialize_metadata_or_energy_fields(self):
        arrays, plan, answer = self.make_arrays_plan_answer()
        record = {
            "representation": "dynamic_v1",
            "plan_state": plan,
            "answer": answer,
            "prompt": "formation_energy=-999; S.U.N.=true; MP_API_KEY=secret",
            "metadata": {
                "e_above_hull": -999,
                "chgnet_score": 1.0,
                "stable": True,
            },
        }
        graph = plangraph_from_record(record)
        encoded = plangraph_to_json(graph)
        lowered = encoded.lower()

        for forbidden in (
            "metadata",
            "e_above_hull",
            "formation_energy",
            "chgnet",
            "mp_api",
            "stable",
            "9.87654321",
            "-123.456789",
        ):
            self.assertNotIn(forbidden, lowered)
        self.assertEqual(graph["site_groups"][0]["slot_indices"], [0, 2])
        self.assertEqual(graph["site_groups"][1]["slot_indices"], [1])

    def test_sft_records_are_leakage_safe_and_reproducible(self):
        _arrays, plan, answer = self.make_arrays_plan_answer()
        source = {
            "representation": "dynamic_v1",
            "plan_state": plan,
            "answer": answer,
            "prompt": "e_above_hull=0.0; stable=true",
            "metadata": {
                "formation_energy": -999,
                "material_id": "mp-secret-id",
            },
            "sample_weight": 99.0,
        }
        records = build_plangraph_sft_records(source)
        repeated = build_plangraph_sft_records(dict(reversed(list(source.items()))))
        self.assertEqual(records, repeated)
        self.assertEqual(records["planner"]["sample_weight"], 1.0)
        self.assertEqual(records["body"]["sample_weight"], 1.0)
        self.assertEqual(records["body"]["answer_semantic_length"], 19)
        self.assertEqual(
            records["planner"]["training_pair_sha256"],
            records["body"]["training_pair_sha256"],
        )
        encoded = json.dumps(records, sort_keys=True).lower()
        for forbidden in (
            "e_above_hull",
            "formation_energy",
            "energy",
            "mp-secret-id",
            "stable=true",
            "stability",
            "chgnet",
            "materials project",
            "s.u.n.",
            "mp_api",
            '"sample_weight": 99',
        ):
            self.assertNotIn(forbidden, encoded)

    def test_validator_rejects_forbidden_and_extra_field(self):
        arrays, plan, _answer = self.make_arrays_plan_answer()
        graph = plangraph_from_plan_state(plan, site_species=arrays["species"])
        tampered = deepcopy(graph)
        tampered["composition"]["e_above_hull"] = 0.0
        validation = validate_plangraph(tampered)
        self.assertFalse(validation.valid)
        self.assertIn("$.composition.e_above_hull", validation.forbidden_key_paths)
        self.assertTrue(any("unsupported keys" in error for error in validation.errors))

    def test_validator_rejects_duplicate_or_missing_site_slots(self):
        arrays, plan, _answer = self.make_arrays_plan_answer()
        graph = plangraph_from_plan_state(plan, site_species=arrays["species"])
        tampered = deepcopy(graph)
        tampered["site_groups"][0]["slot_indices"] = [0, 1]
        tampered["site_groups"][1]["slot_indices"] = [1]
        validation = validate_plangraph(tampered)
        self.assertFalse(validation.valid)
        self.assertTrue(
            any("cover each atom slot" in error for error in validation.errors)
        )

    def test_record_converter_rejects_composition_mismatch(self):
        _arrays, plan, _answer = self.make_arrays_plan_answer()
        mismatch_answer, _diagnostics = arrays_to_dynamic_answer(
            lengths=[3.1, 3.1, 5.2],
            angles=[90.0, 90.0, 120.0],
            species=["Li", "O", "O"],
            frac_coords=[
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5],
            ],
        )
        with self.assertRaises(PlanGraphError):
            plangraph_from_record(
                {
                    "representation": "dynamic_v1",
                    "plan_state": plan,
                    "answer": mismatch_answer,
                }
            )

    def test_json_schema_identity_matches_python_contract(self):
        root = Path(__file__).resolve().parents[1]
        schema = json.loads(
            (root / "schemas/plangraph_v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            PLANGRAPH_VERSION,
        )
        self.assertEqual(
            schema["properties"]["site_group_strategy"]["const"],
            PLANGRAPH_SITE_GROUP_STRATEGY,
        )
        self.assertFalse(schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
