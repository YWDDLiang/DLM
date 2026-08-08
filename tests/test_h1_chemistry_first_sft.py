from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path
import tempfile
import unittest

from crystal_dlm.h1_chemistry_first_sft import (
    H1_CHEMISTRY_FIRST_AUX_TASKS,
    H1_CHEMISTRY_FIRST_INFERENCE_MESSAGES_SHA256,
    accumulation_group_size,
    accumulation_is_update_boundary,
    assign_auxiliary_tasks,
    canonical_json_sha256,
    curriculum_order,
    hash_shuffle,
    optimizer_update_count,
    order_pair_audit,
    record_multiset_sha256,
    record_order_sha256,
    warmup_step_count,
)
from crystal_dlm.h1_llm_planner import (
    H1_PLANNER_PROMPT_STYLE_RICH_NOCHARGE,
    build_planner_messages,
)
from crystal_dlm.h1_local_smact4_ledger import (
    EXPECTED_SMACT4_CONTRACT_SHA256,
    EXPECTED_SMACT4_VERSION,
    EXPECTED_SMACT4_WHEEL_SHA256,
    build_witness_ledger_payload,
    load_and_attach_witness_bundle,
    write_witness_bundle,
)
from crystal_dlm.h1_nocharge_ion_aux import (
    formula_from_atom_sequence,
    formula_from_ion_sequence,
    ion_charge_sum,
)
from scripts.build_h1_chemistry_first_sft_data import (
    EXPECTED_WEIGHT_LABELS,
    audit_records,
    build_common_train_records,
    build_validation_records,
    make_auxiliary_record,
    make_direct_record,
    tokenizer_audit,
)
from scripts.evaluate_h1_chemistry_first_planner_gate import (
    candidate_scientific_gates,
    distribution_deltas,
    distribution_summary,
)


class FakeTokenizer:
    eos_token = "~"
    chat_template = None

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        value = str(text)
        result = {"input_ids": [ord(character) for character in value]}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(value))
            ]
        return result


def fixture_row(row_idx: int, *, positive: bool = True) -> dict:
    formula = "Li2O" if positive else "LiO"
    counts = [2, 1] if positive else [1, 1]
    witness = [("Li", 1), ("Li", 1), ("O", -2)] if positive else None
    return {
        "row_idx": row_idx,
        "material_id": f"mp-fixture-{row_idx}",
        "legacy": {
            "valid": positive,
            "reason": "charge_neutral_pauling_valid" if positive else "charge_fail",
        },
        "smact4": (
            {
                "valid": True,
                "stratum": "uniform_primary",
                "witness": witness,
                "official_witness_parity": True,
            }
            if positive
            else None
        ),
        "plan": {
            "N": sum(counts),
            "elements": ["Li", "O"],
            "counts": counts,
            "formula": formula,
            "reduced_formula": formula,
            "anion_framework": "oxide",
            "charge_bucket": "neutral_plausible" if positive else "charge_fail",
            "lattice_system": "cubic",
            "spacegroup_bucket": "sg_195_230",
            "volume_per_atom_bin": "volpa_005_009",
            "family": "oxide",
            "arity": "binary",
            "size": "tiny",
        },
    }


class ChemistryFirstSFTTests(unittest.TestCase):
    @staticmethod
    def planner_summary(**overrides):
        value = {
            "denominator": 64,
            "parse_count": 64,
            "completion_count": 64,
            "legacy_comp_valid_count": 40,
            "legacy_primary_nonshortcut_count": 35,
            "single_element_shortcut_count": 2,
            "all_metal_shortcut_count": 1,
            "smact4_valid_count": 41,
            "smact4_uniform_primary_count": 36,
            "smact4_stratum_counts": {"uniform_primary": 36, "charge_or_pauling_fail": 28},
            "unique_formula_rate": 0.90,
            "top1_formula_rate": 0.05,
            "element_coverage_rate": 0.80,
            "mean_N_when_parsed": 10.0,
            "failure_class_counts": {},
        }
        value.update(overrides)
        return value

    def test_direct_plan_preserves_exact_nocharge_inference_messages(self):
        record = make_direct_record(fixture_row(0), split="train")
        expected = build_planner_messages(
            sample_idx=None,
            prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_NOCHARGE,
        )
        self.assertEqual(record["messages"], expected)
        self.assertEqual(
            canonical_json_sha256(record["messages"]),
            H1_CHEMISTRY_FIRST_INFERENCE_MESSAGES_SHA256,
        )
        self.assertEqual(
            record["answer"].splitlines(),
            [
                "formula: Li2O",
                "anion: oxide",
                "lattice: cubic",
                "spacegroup: sg_195_230",
                "volume: volpa_005_009",
                "end: plan",
            ],
        )
        self.assertFalse(
            any(line.lower().startswith("charge:") for line in record["answer"].splitlines())
        )

    def test_all_rows_anchor_but_only_pos_rows_are_unconditional_targets(self):
        rows = [fixture_row(0, positive=True), fixture_row(1, positive=False)]
        records, census = build_common_train_records(rows, [0], seed=17)
        self.assertEqual(census["all_count"], 2)
        self.assertEqual(census["pos_count"], 1)
        self.assertEqual(census["record_count"], 4)
        anchors = [row for row in records if row["task"] == "conditional_structural_anchor"]
        unconditional = [row for row in records if row["formula_is_unconditional_target"]]
        self.assertEqual(len(anchors), 2)
        self.assertTrue(all("formula:" not in row["answer"].lower() for row in anchors))
        self.assertTrue(all(row["source_is_pos"] for row in unconditional))
        invalid = [row for row in records if row["source_row_idx"] == 1]
        self.assertEqual([row["task"] for row in invalid], ["conditional_structural_anchor"])
        audit = audit_records(records, expected_all_count=2, expected_pos_count=1)
        self.assertTrue(audit["passed"], audit["failures"])
        self.assertEqual(audit["invalid_unconditional_formula_target_count"], 0)

    def test_local_smact4_witness_bundle_is_complete_and_fail_closed(self):
        source_rows = {}
        legacy_report = {"contract_sha256": "legacy-parent", "splits": {}}
        witness_reports = {}
        for split in ("train", "val"):
            positive = fixture_row(0, positive=True)
            positive["split"] = split
            negative = fixture_row(1, positive=False)
            negative["split"] = split
            source_rows[split] = [positive, negative]
            legacy_report["splits"][split] = {
                "snapshot_jsonl_sha256": f"{split}-snapshot",
                "source_csv_sha256": f"{split}-csv",
            }
            witness_reports[split] = {
                "official_witness_parity": True,
                "stable_primary_indices": [0],
            }
        contract = {
            "smact_version": EXPECTED_SMACT4_VERSION,
            "release_wheel_sha256": EXPECTED_SMACT4_WHEEL_SHA256,
            "contract_sha256": EXPECTED_SMACT4_CONTRACT_SHA256,
        }
        payload = build_witness_ledger_payload(
            source_rows,
            source_inventory_sha256="a" * 64,
            legacy_report=legacy_report,
            smact4_contract=contract,
            witness_reports=witness_reports,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            manifest = write_witness_bundle(root, payload)
            clean_rows = {
                split: [
                    {key: copy.deepcopy(value) for key, value in row.items() if key != "smact4"}
                    for row in rows
                ]
                for split, rows in source_rows.items()
            }
            imported, reports = load_and_attach_witness_bundle(
                root,
                clean_rows,
                legacy_report=legacy_report,
                expected_source_inventory_sha256="a" * 64,
                expected_manifest_sha256=manifest["manifest_sha256"],
            )
            self.assertEqual(
                imported["smact4_contract"]["contract_sha256"],
                EXPECTED_SMACT4_CONTRACT_SHA256,
            )
            self.assertEqual(reports["train"]["stable_primary_indices"], [0])
            self.assertNotIn("smact4", clean_rows["train"][1])
            clean_rows["train"][0]["material_id"] = "tampered"
            with self.assertRaisesRegex(ValueError, "join mismatch"):
                load_and_attach_witness_bundle(
                    root,
                    clean_rows,
                    legacy_report=legacy_report,
                    expected_source_inventory_sha256="a" * 64,
                    expected_manifest_sha256=manifest["manifest_sha256"],
                )

    def test_all_four_auxiliary_tasks_are_exact_and_weighted(self):
        row = fixture_row(0)
        records = {
            task: make_auxiliary_record(
                row,
                split="train",
                task=task,
                seed=19,
            )
            for task in H1_CHEMISTRY_FIRST_AUX_TASKS
        }
        self.assertEqual(
            formula_from_atom_sequence(records["atoms_to_formula"]["aux_sequence"]),
            "Li2O",
        )
        ions = records["ions_to_charge_sum_formula"]["aux_sequence"]
        self.assertEqual(formula_from_ion_sequence(ions), "Li2O")
        self.assertEqual(ion_charge_sum(ions), 0)
        self.assertIn("<MASK_OXIDATION>", records["masked_oxidation"]["aux_sequence"])
        self.assertIn(
            records["masked_oxidation"]["infill_target"],
            {"QP01", "QM02"},
        )
        for task, record in records.items():
            labels = tuple(span["label"] for span in record["weighted_answer_spans"])
            self.assertEqual(labels, EXPECTED_WEIGHT_LABELS[task])
            self.assertTrue(all(span["weight"] == 2.0 for span in record["weighted_answer_spans"]))
            self.assertEqual(record["sample_weight"], 1.0)
        self.assertNotIn(
            "N",
            {
                span["label"]
                for span in records["formula_to_elements_counts_n"][
                    "weighted_answer_spans"
                ]
            },
        )

    def test_auxiliary_cycle_is_deterministic_and_balanced(self):
        rows = [fixture_row(index) for index in range(19)]
        first = assign_auxiliary_tasks(rows, seed=23)
        second = assign_auxiliary_tasks(list(reversed(rows)), seed=23)
        self.assertEqual(first, second)
        counts = Counter(first.values())
        self.assertEqual(set(counts), set(H1_CHEMISTRY_FIRST_AUX_TASKS))
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_base_and_curriculum_share_exact_multiset_but_not_order(self):
        rows = [fixture_row(index) for index in range(24)]
        common, _ = build_common_train_records(rows, list(range(24)), seed=31)
        base = hash_shuffle(common, seed=31, role="base")
        curriculum = curriculum_order(common, seed=31)
        report = order_pair_audit(base, curriculum)
        self.assertTrue(report["passed"], report["failures"])
        self.assertTrue(report["orders_differ"])
        self.assertEqual(record_multiset_sha256(base), record_multiset_sha256(curriculum))
        self.assertNotEqual(record_order_sha256(base), record_order_sha256(curriculum))
        prefix = curriculum[: report["curriculum_prefix_count"]]
        roles = [
            "direct" if row["task"] == "direct_nocharge_plan" else "auxiliary"
            for row in prefix
        ]
        self.assertTrue(all(left != right for left, right in zip(roles, roles[1:])))

    def test_hashes_and_orders_are_repeatable(self):
        rows = [fixture_row(index) for index in range(8)]
        common, _ = build_common_train_records(rows, list(range(8)), seed=41)
        first = hash_shuffle(common, seed=41, role="base")
        second = hash_shuffle(copy.deepcopy(common), seed=41, role="base")
        self.assertEqual(first, second)
        self.assertEqual(record_order_sha256(first), record_order_sha256(second))
        self.assertNotEqual(
            record_order_sha256(first),
            record_order_sha256(hash_shuffle(common, seed=42, role="base")),
        )

    def test_partial_accumulation_uses_actual_final_microbatch_count(self):
        total = 10
        self.assertEqual(optimizer_update_count(total, 8), 2)
        divisors = [
            accumulation_group_size(index, total_microbatches=total, grad_accum=8)
            for index in range(total)
        ]
        boundaries = [
            index
            for index in range(total)
            if accumulation_is_update_boundary(
                index,
                total_microbatches=total,
                grad_accum=8,
            )
        ]
        self.assertEqual(divisors, [8] * 8 + [2] * 2)
        self.assertEqual(boundaries, [7, 9])
        self.assertAlmostEqual(sum(1.0 / value for value in divisors[:8]), 1.0)
        self.assertAlmostEqual(sum(1.0 / value for value in divisors[8:]), 1.0)
        self.assertEqual(warmup_step_count(400), 25)
        self.assertEqual(warmup_step_count(5162), 155)

    def test_validation_is_all_anchor_and_tokenizer_audit_fails_closed_on_truncation(self):
        records = build_validation_records(
            [fixture_row(0, positive=True), fixture_row(1, positive=False)]
        )
        self.assertEqual(
            [record["task"] for record in records],
            ["conditional_structural_anchor", "conditional_structural_anchor"],
        )
        self.assertTrue(tokenizer_audit(records, FakeTokenizer(), max_length=4096)["passed"])
        failed = tokenizer_audit(records, FakeTokenizer(), max_length=8)
        self.assertFalse(failed["passed"])
        self.assertTrue(
            all("would_truncate" in item["reasons"] for item in failed["failures"])
        )

    def test_raw64_and_raw256_use_literal_positive_candidate_gates(self):
        baseline = self.planner_summary()
        candidate = self.planner_summary(
            legacy_comp_valid_count=41,
            legacy_primary_nonshortcut_count=36,
            smact4_valid_count=42,
            smact4_uniform_primary_count=37,
            smact4_stratum_counts={"uniform_primary": 37, "charge_or_pauling_fail": 27},
            unique_formula_rate=0.89,
            top1_formula_rate=0.06,
            element_coverage_rate=0.79,
            mean_N_when_parsed=10.4,
        )
        gates64, report64 = candidate_scientific_gates(
            baseline,
            candidate,
            stage=64,
            anchor_nll_passed=True,
            distribution_complete=True,
        )
        self.assertTrue(all(gates64.values()), gates64)
        self.assertEqual(report64["candidate_minus_p0"]["legacy_comp_valid_count"], 1.0)
        baseline256 = {**baseline, "denominator": 256, "smact4_stratum_counts": {"uniform_primary": 36, "charge_or_pauling_fail": 220}}
        candidate256 = {**candidate, "denominator": 256, "smact4_stratum_counts": {"uniform_primary": 37, "charge_or_pauling_fail": 219}}
        gates256, _ = candidate_scientific_gates(
            baseline256,
            candidate256,
            stage=256,
            anchor_nll_passed=True,
            distribution_complete=True,
        )
        self.assertTrue(all(gates256.values()), gates256)
        failed, _ = candidate_scientific_gates(
            baseline,
            self.planner_summary(
                legacy_comp_valid_count=40,
                legacy_primary_nonshortcut_count=36,
            ),
            stage=64,
            anchor_nll_passed=True,
            distribution_complete=True,
        )
        self.assertFalse(failed["legacy_comp_valid_literal_gain_ge_1"])

    def test_distribution_audit_reports_arity_and_all_coarse_fields(self):
        rows = []
        for ordinal, lattice in enumerate(("cubic", "hexagonal")):
            rows.append(
                {
                    "parsed": True,
                    "plan_state": {
                        "elements": ["Li", "O"],
                        "anion_framework": "oxide",
                        "lattice_system": lattice,
                        "spacegroup_bucket": "sg_195_230",
                        "volume_per_atom_bin": "volpa_005_009",
                    },
                }
            )
        summary = distribution_summary(rows)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["arity_counts"], {"2": 2})
        delta = distribution_deltas(summary, summary)
        self.assertEqual(delta["arity_total_variation"], 0.0)
        self.assertTrue(
            all(value == 0.0 for value in delta["coarse_field_total_variation"].values())
        )


if __name__ == "__main__":
    unittest.main()
