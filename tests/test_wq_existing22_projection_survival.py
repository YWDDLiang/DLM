from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from diagnostics.run_wq_existing22_projection_survival import (
    FrozenProjection,
    RenderedStructure,
    SurvivalAuditError,
    _write_json_exclusive,
    evaluate_projections,
    select_frozen_projections,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs"
    / "experiments"
    / "wyckoff_codiffusion"
    / "wq_existing22_projection_survival_v1.json"
)
SCRIPT = ROOT / "diagnostics" / "run_wq_existing22_projection_survival.py"


def state(attempt_id: str, species: tuple[int, ...]) -> dict:
    return {
        "attempt_id": attempt_id,
        "space_group": 1,
        "lattice_system": "triclinic",
        "lattice_chart": [1.0, 1.1, 1.2, 0.0, 0.0, 0.0],
        "orbits": [
            {
                "orbit_id": f"o{index}",
                "wyckoff_type": index,
                "species": value,
                "multiplicity": 1,
                "primitive_multiplicity": 1,
                "chart_dimension": 0,
                "free_coordinate": [],
            }
            for index, value in enumerate(species)
        ],
        "timestep": 1.0,
        "space_group_committed": True,
    }


def paired_rows(
    attempt_id: str,
    ordinal: int,
    *,
    original_species: tuple[int, ...] = (3, 8, 8),
    projected_species: tuple[int, ...] = (3, 3, 8),
) -> tuple[dict, dict]:
    original = state(attempt_id, original_species)
    projected = state(attempt_id, projected_species)
    panel = {
        "schema": "wqcodiff_composition_mechanism_panel_input_row_v1",
        "attempt_id": attempt_id,
        "ordinal": ordinal,
        "panel_group": "no_neutral",
        "state": original,
    }
    projection = {
        "schema": "wqcodiff_composition_mechanism_panel_row_v1",
        "attempt_id": attempt_id,
        "panel_group": "no_neutral",
        "input_line_number": ordinal,
        "projection": {
            "schema": "wqcodiff_fixed_topology_composition_projection_v1",
            "attempt_id": attempt_id,
            "source_reason": "charge_neutrality_fail",
            "status": "projected",
            "changed_orbit_ids": ["o1"],
            "original_formula": "LiO2",
            "projected_formula": "Li2O",
            "state": projected,
        },
    }
    return panel, projection


def frozen(attempt_id: str, ordinal: int) -> FrozenProjection:
    _, row = paired_rows(attempt_id, ordinal)
    payload = row["projection"]
    return FrozenProjection(
        attempt_id=attempt_id,
        ordinal=ordinal,
        input_line_number=ordinal,
        original_formula=payload["original_formula"],
        projected_formula=payload["projected_formula"],
        changed_orbit_ids=("o1",),
        original_state=state(attempt_id, (3, 8, 8)),
        projected_state=payload["state"],
        projected_state_sha256="state-sha",
    )


class FrozenSelectionTests(unittest.TestCase):
    def test_selects_only_exact_projected_denominator(self) -> None:
        p1, r1 = paired_rows("a1", 10)
        p2, r2 = paired_rows("a2", 20)
        selected = select_frozen_projections(
            [p1, p2],
            [r1, r2],
            expected_panel_rows=2,
            expected_projection_rows=2,
            expected_ordinals=[10, 20],
        )
        self.assertEqual([value.attempt_id for value in selected], ["a1", "a2"])
        self.assertEqual([value.ordinal for value in selected], [10, 20])

    def test_rejects_topology_or_geometry_change(self) -> None:
        panel, projection = paired_rows("a1", 10)
        projection["projection"]["state"]["orbits"][0]["free_coordinate"] = [0.1]
        projection["projection"]["state"]["orbits"][0]["chart_dimension"] = 1
        with self.assertRaisesRegex(SurvivalAuditError, "topology/geometry"):
            select_frozen_projections(
                [panel],
                [projection],
                expected_panel_rows=1,
                expected_projection_rows=1,
                expected_ordinals=[10],
            )

    def test_rejects_element_set_change(self) -> None:
        panel, projection = paired_rows(
            "a1",
            10,
            projected_species=(3, 3, 14),
        )
        projection["projection"]["changed_orbit_ids"] = ["o1", "o2"]
        with self.assertRaisesRegex(SurvivalAuditError, "element set"):
            select_frozen_projections(
                [panel],
                [projection],
                expected_panel_rows=1,
                expected_projection_rows=1,
                expected_ordinals=[10],
            )

    def test_rejects_survivor_identity_change(self) -> None:
        panel, projection = paired_rows("a1", 10)
        with self.assertRaisesRegex(SurvivalAuditError, "ordinal identity"):
            select_frozen_projections(
                [panel],
                [projection],
                expected_panel_rows=1,
                expected_projection_rows=1,
                expected_ordinals=[11],
            )


class DirectMetricGateTests(unittest.TestCase):
    @staticmethod
    def renderer(payload: dict) -> RenderedStructure:
        return RenderedStructure(
            structure=payload["attempt_id"],
            structure_dict={"attempt_id": payload["attempt_id"]},
            atom_count=3,
            volume=64.0,
            redetected_space_group=1,
        )

    @staticmethod
    def passing_metric(_: object) -> dict:
        return {
            "constructed": True,
            "comp_valid": True,
            "struct_valid": True,
            "valid": True,
            "fingerprint_valid": True,
            "reason": "",
        }

    def test_all_attempt_gate_passes_without_survivor_filter(self) -> None:
        structures, metrics, report = evaluate_projections(
            [frozen("a1", 10), frozen("a2", 20)],
            renderer=self.renderer,
            metric=self.passing_metric,
            minimum_structural_valid=2,
            minimum_joint_valid=2,
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["attempts"], 2)
        self.assertEqual(report["joint_valid_count"], 2)
        self.assertEqual(len(structures), 2)
        self.assertEqual(len(metrics), 2)
        self.assertFalse(any(row["retry_or_replacement_used"] for row in metrics))

    def test_render_failure_stays_in_denominator_and_fails_gate(self) -> None:
        def renderer(payload: dict) -> RenderedStructure:
            if payload["attempt_id"] == "a2":
                raise ValueError("render failed")
            return self.renderer(payload)

        structures, metrics, report = evaluate_projections(
            [frozen("a1", 10), frozen("a2", 20)],
            renderer=renderer,
            metric=self.passing_metric,
            minimum_structural_valid=2,
            minimum_joint_valid=2,
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["attempts"], 2)
        self.assertEqual(report["rendered_count"], 1)
        self.assertEqual(report["joint_valid_count"], 1)
        self.assertEqual(structures[1]["status"], "failed")
        self.assertFalse(metrics[1]["valid"])

    def test_fingerprint_failure_counts_as_joint_failure(self) -> None:
        def metric(structure: str) -> dict:
            passed = structure == "a1"
            return {
                "constructed": True,
                "comp_valid": True,
                "struct_valid": True,
                "valid": passed,
                "fingerprint_valid": passed,
                "reason": "" if passed else "fingerprint_invalid",
            }

        _, _, report = evaluate_projections(
            [frozen("a1", 10), frozen("a2", 20)],
            renderer=self.renderer,
            metric=metric,
            minimum_structural_valid=2,
            minimum_joint_valid=2,
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["structural_valid_count"], 2)
        self.assertEqual(report["joint_valid_count"], 1)

    def test_output_identity_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "claim.json"
            _write_json_exclusive(target, {"ok": True})
            with self.assertRaises(FileExistsError):
                _write_json_exclusive(target, {"ok": False})


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        cls.script = SCRIPT.read_text(encoding="utf-8")

    def test_contract_freezes_existing_22_and_thresholds(self) -> None:
        self.assertEqual(
            self.contract["schema"],
            "wqcodiff_existing22_projection_survival_contract_v1",
        )
        self.assertEqual(self.contract["denominator"]["attempts"], 22)
        self.assertEqual(len(self.contract["expected_projected_ordinals"]), 22)
        self.assertEqual(
            self.contract["acceptance"]["composition_valid_count_exact"], 22
        )
        self.assertEqual(
            self.contract["acceptance"]["minimum_structural_valid_count"], 20
        )
        self.assertEqual(
            self.contract["acceptance"]["minimum_joint_valid_count"], 20
        )

    def test_contract_forbids_generation_training_mlip_and_sun(self) -> None:
        scope = self.contract["scope"]
        for key in (
            "new_generation",
            "training",
            "projector_rerun",
            "candidate_reselection",
            "retry_or_replacement",
            "chgnet_or_other_mlip",
            "mp_api_or_external_api",
            "sun",
            "slurm",
            "gpu",
        ):
            self.assertFalse(scope[key], key)

    def test_runner_has_fail_closed_runtime_and_exclusive_claim(self) -> None:
        self.assertIn("SLURM_JOB_ID", self.script)
        self.assertIn("CUDA_VISIBLE_DEVICES", self.script)
        self.assertIn("exist_ok=False", self.script)
        self.assertIn("scientific_call_index", self.script)
        self.assertIn("retry_or_replacement_used", self.script)


if __name__ == "__main__":
    unittest.main()
