import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from crystal_dlm.h1_nocharge_ion_aux import (
    H1_NOCHARGE_ION_AUX_TASK_COUNTS,
    H1_NOCHARGE_ION_AUX_VALIDATION_TASK_COUNTS,
    assert_task_contract,
    deterministic_ranked_indices,
    deterministic_task_schedule,
    format_atom_sequence,
    format_ion_sequence,
    formula_from_ion_sequence,
    formula_from_atom_sequence,
    formula_weight_span,
    ion_charge_sum,
    oxidation_from_code,
    oxidation_to_code,
    parse_ion_sequence,
    parse_atom_sequence,
    payload_weight_span,
    raked_select_indices,
    validation_anchor_nll_gate,
)
from scripts.build_h1_nocharge_ion_aux_sft_data import (
    audit_paired_records,
    load_legacy_snapshot,
    make_paired_records,
    scaled_task_counts,
)


class H1NochargeIonAuxTests(unittest.TestCase):
    @staticmethod
    def fixture_row():
        return {
            "row_idx": 7,
            "material_id": "mp-fixture",
            "legacy": {"valid": True, "reason": "charge_neutral_pauling_valid", "oxidation_states": [1, -2]},
            "smact4": {
                "valid": True,
                "stratum": "uniform_primary",
                "witness": [("Li", 1), ("Li", 1), ("O", -2)],
            },
            "plan": {
                "N": 3,
                "elements": ["Li", "O"],
                "counts": [2, 1],
                "formula": "Li2O",
                "reduced_formula": "Li2O",
                "anion_framework": "oxide",
                "charge_bucket": "neutral_plausible",
                "lattice_system": "cubic",
                "spacegroup_bucket": "sg_195_230",
                "volume_per_atom_bin": "volpa_005_009",
                "family": "oxide",
                "arity": "binary",
                "size": "tiny",
            },
        }

    def test_oxidation_codes_round_trip_and_distinguish_neutral_placeholder(self):
        for value in (-8, -2, -1, 0, 1, 3, 8):
            self.assertEqual(oxidation_from_code(oxidation_to_code(value)), value)
        self.assertEqual(oxidation_to_code(None), "QU00")
        self.assertEqual(oxidation_to_code(None, neutral_placeholder=True), "QX00")
        self.assertIsNone(oxidation_from_code("QX00"))

    def test_validation_anchor_nll_gate_is_relative_to_frozen_p0(self):
        self.assertTrue(validation_anchor_nll_gate(2.0, 2.02)["passed"])
        self.assertFalse(validation_anchor_nll_gate(2.0, 2.021)["passed"])
        with self.assertRaises(ValueError):
            validation_anchor_nll_gate(0.0, 0.0)

    def test_legacy_snapshot_loader_verifies_contract_and_jsonl_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            row = {
                "row_idx": 0,
                "material_id": "mp-test",
                "plan": self.fixture_row()["plan"],
                "legacy": self.fixture_row()["legacy"],
            }
            payload = json.dumps(row, sort_keys=True) + "\n"
            path = root / "val.jsonl"
            path.write_text(payload, encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            report = {
                "schema": "h1_nocharge_mp20_legacy_snapshot_v1",
                "status": "pass",
                "fixture_only": True,
                "legacy_smact_version": "3.1.0",
                "legacy_evaluator_sha256": "ca1c94f583e0c97a172b5c9b7ba96505257fd74dedfc618b584c34486ac1f178",
                "splits": {
                    "val": {
                        "row_count": 1,
                        "snapshot_jsonl_sha256": digest,
                        "source_csv_sha256": "source",
                    }
                },
            }
            from crystal_dlm.h1_nocharge_ion_aux import canonical_json_sha256

            report["contract_sha256"] = canonical_json_sha256(report)
            (root / "report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            (root / "_SUCCESS").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "contract_sha256": report["contract_sha256"],
                    }
                ),
                encoding="utf-8",
            )
            rows, loaded_report = load_legacy_snapshot(
                root,
                splits=("val",),
                require_frozen=False,
            )
            self.assertEqual(rows["val"][0]["material_id"], "mp-test")
            self.assertEqual(
                loaded_report["contract_sha256"], report["contract_sha256"]
            )

    def test_ion_sequence_is_formula_round_trippable_and_charge_neutral(self):
        sequence = format_ion_sequence(
            ["Li", "O"],
            [2, 1],
            [("Li", 1), ("Li", 1), ("O", -2)],
        )
        self.assertEqual(sequence, "I=Li:QP01,Li:QP01,O:QM02")
        self.assertEqual(formula_from_ion_sequence(sequence), "Li2O")
        self.assertEqual(ion_charge_sum(sequence), 0)
        self.assertEqual(parse_ion_sequence(sequence), [("Li", 1), ("Li", 1), ("O", -2)])

    def test_mixed_valence_sequence_preserves_multiplicity(self):
        sequence = format_ion_sequence(
            ["O", "Fe"],
            [4, 3],
            [("Fe", 2), ("Fe", 3), ("Fe", 3), ("O", -2), ("O", -2), ("O", -2), ("O", -2)],
        )
        self.assertEqual(formula_from_ion_sequence(sequence), "O4Fe3")
        self.assertEqual(ion_charge_sum(sequence), 0)

    def test_c0_placeholder_has_same_atoms_but_no_claimed_charge(self):
        sequence = format_ion_sequence(["Li", "O"], [2, 1], neutral_placeholder=True)
        self.assertEqual(sequence, "I=Li:QX00,Li:QX00,O:QX00")
        self.assertEqual(formula_from_ion_sequence(sequence), "Li2O")
        self.assertIsNone(ion_charge_sum(sequence))

    def test_c0_repeated_atom_sequence_round_trips_without_oxidation_labels(self):
        sequence = format_atom_sequence(["Li", "O"], [2, 1])
        self.assertEqual(sequence, "A=Li:C001,Li:C001,O:C001")
        self.assertEqual(parse_atom_sequence(sequence), ["Li", "Li", "O"])
        self.assertEqual(formula_from_atom_sequence(sequence), "Li2O")

    def test_task_ledgers_are_exact_and_deterministic(self):
        assert_task_contract(H1_NOCHARGE_ION_AUX_TASK_COUNTS, expected_total=3200)
        assert_task_contract(H1_NOCHARGE_ION_AUX_VALIDATION_TASK_COUNTS, expected_total=640)
        first = deterministic_task_schedule(H1_NOCHARGE_ION_AUX_TASK_COUNTS, seed=17)
        second = deterministic_task_schedule(H1_NOCHARGE_ION_AUX_TASK_COUNTS, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3200)
        self.assertNotEqual(first, deterministic_task_schedule(H1_NOCHARGE_ION_AUX_TASK_COUNTS, seed=18))

    def test_fixture_task_scaling_preserves_all_tasks_and_total(self):
        scaled = scaled_task_counts(H1_NOCHARGE_ION_AUX_TASK_COUNTS, 64)
        self.assertEqual(sum(scaled.values()), 64)
        self.assertTrue(all(value > 0 for value in scaled.values()))

    def test_ranked_indices_are_stateless_and_without_replacement(self):
        first = deterministic_ranked_indices(range(20), 5, seed=7, role="direct")
        second = deterministic_ranked_indices(list(reversed(range(20))), 5, seed=7, role="direct")
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 5)

    def test_weight_spans_cover_only_named_payload(self):
        answer = "formula: Li2O\nend: plan"
        self.assertEqual(formula_weight_span(answer, "Li2O"), [
            {"start": 9, "end": 13, "weight": 2.0, "label": "formula"}
        ])
        self.assertEqual(payload_weight_span("state: QP01", "QP01"), [
            {"start": 7, "end": 11, "weight": 2.0, "label": "chemistry_payload"}
        ])

    def test_raking_is_deterministic_without_row_or_formula_reuse(self):
        rows = [
            {
                "plan": {
                    "formula": f"F{idx}",
                    "reduced_formula": f"R{idx}",
                    "N": 2 if idx < 6 else 4,
                    "arity": "binary" if idx % 2 == 0 else "ternary",
                }
            }
            for idx in range(12)
        ]
        selected, report = raked_select_indices(
            rows,
            list(reversed(range(12))),
            rows,
            6,
            fields=("N", "arity"),
            seed=19,
            role="fixture",
            exact_formula_cap=1,
            reduced_formula_cap=1,
        )
        selected_again, report_again = raked_select_indices(
            rows,
            range(12),
            rows,
            6,
            fields=("N", "arity"),
            seed=19,
            role="fixture",
            exact_formula_cap=1,
            reduced_formula_cap=1,
        )
        self.assertEqual(selected, selected_again)
        self.assertEqual(report, report_again)
        self.assertEqual(len(set(selected)), 6)
        self.assertEqual(report["max_exact_formula_exposure"], 1)
        self.assertEqual(report["selected_histograms"]["N"], {"2": 3, "4": 3})

    def test_builder_pairs_only_change_the_registered_auxiliary_content(self):
        tasks = [
            "direct_nocharge_plan",
            "sequence_to_formula",
            "oxidation_infill",
            "conditional_mp20_anchor",
            "p0_kl_anchor",
        ]
        c0_records = []
        c1_records = []
        for ordinal, task in enumerate(tasks):
            c0, c1 = make_paired_records(
                self.fixture_row(),
                split="train",
                ordinal=ordinal,
                task=task,
                seed=11,
            )
            c0_records.append(c0)
            c1_records.append(c1)
        report = audit_paired_records(
            c0_records,
            c1_records,
            expected_task_counts={task: 1 for task in tasks},
        )
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["nonaux_content_identity_count"], 3)
        self.assertEqual(c0_records[0]["answer"], c1_records[0]["answer"])
        self.assertNotIn("charge:", c1_records[0]["answer"])
        self.assertIn("C001", c0_records[1]["aux_sequence"])
        self.assertIn("QP01", c1_records[1]["aux_sequence"])
        self.assertTrue(c0_records[3]["formula_is_input_only"])
        self.assertEqual(c0_records[4]["loss_mode"], "kl_only")
        self.assertTrue(c0_records[4]["formula_is_input_only"])
        self.assertNotIn("formula:", c0_records[4]["answer"].lower())
        self.assertNotIn("charge:", c0_records[4]["answer"].lower())


if __name__ == "__main__":
    unittest.main()
