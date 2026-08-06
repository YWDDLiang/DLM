from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.audit_h1_nocharge_planner_smact4 import audit_rows
from scripts.evaluate_h1_nocharge_sft_planner_gate import (
    classify_arm,
    exact_mcnemar_p,
    paired_binary_report,
    paired_bootstrap_ci,
    scientific_gates,
)


def legacy_stub(elems, counts):
    if list(elems) == [3, 8] and list(counts) == [2, 1]:
        return {
            "valid": True,
            "reason": "charge_neutral_pauling_valid",
            "oxidation_states": (1, -2),
        }
    return {"valid": False, "reason": "charge_neutrality_fail"}


def gate_summary(
    denominator: int,
    *,
    comp: int,
    primary: int,
    smact_valid: int,
    smact_uniform: int,
    parse: int | None = None,
    completion: int | None = None,
    single: int = 0,
    all_metal: int = 0,
    unique_rate: float = 0.90,
    top1_rate: float = 0.02,
    coverage_rate: float = 0.60,
    mean_n: float = 10.0,
    failures: dict[str, int] | None = None,
):
    return {
        "parse_count": denominator if parse is None else parse,
        "completion_count": denominator if completion is None else completion,
        "legacy_comp_valid_count": comp,
        "legacy_primary_nonshortcut_count": primary,
        "single_element_shortcut_count": single,
        "all_metal_shortcut_count": all_metal,
        "smact4_valid_count": smact_valid,
        "smact4_uniform_primary_count": smact_uniform,
        "unique_formula_rate": unique_rate,
        "top1_formula_rate": top1_rate,
        "element_coverage_rate": coverage_rate,
        "mean_N_when_parsed": mean_n,
        "failure_class_counts": failures or {},
    }


class Smact4AuditTests(unittest.TestCase):
    def test_audit_keeps_parse_failure_in_raw_denominator(self) -> None:
        rows = [
            {
                "sample_idx": 0,
                "parsed": True,
                "plan_state": {
                    "formula": "Li2O",
                    "elements": ["Li", "O"],
                    "counts": [2, 1],
                },
            },
            {"sample_idx": 1, "parsed": False},
        ]
        result = {
            "valid": True,
            "witness_valid": True,
            "official_witness_parity": True,
            "stratum": "uniform_primary",
            "witness": [("Li", 1), ("Li", 1), ("O", -2)],
            "charge_sum": 0,
        }
        with patch(
            "scripts.audit_h1_nocharge_planner_smact4.smact4_validity_with_witness",
            return_value=result,
        ):
            attempts, summary = audit_rows(rows, {"Li": [1], "O": [-2]})
        self.assertEqual(len(attempts), 2)
        self.assertEqual(summary["parsed_count"], 1)
        self.assertEqual(summary["valid_count"], 1)
        self.assertEqual(attempts[1]["stratum"], "planner_parse_failure")


class PlannerArmTests(unittest.TestCase):
    def raw_row(self, *, with_charge: bool = False):
        validator = legacy_stub([3, 8], [2, 1])
        generated = {
            "anion": "oxide",
            "lattice": "cubic",
            "spacegroup": "sg_195_230",
            "volume": "volpa_016_020",
        }
        text = (
            "formula: Li2O\nanion: oxide\nlattice: cubic\n"
            "spacegroup: sg_195_230\nvolume: volpa_016_020\nend: plan"
        )
        if with_charge:
            generated["charge"] = "neutral_plausible"
            text = text.replace("anion: oxide\n", "anion: oxide\ncharge: neutral_plausible\n")
        return {
            "sample_idx": 0,
            "parsed": True,
            "plan_end_marker_present": True,
            "prompt_style": "h1_rich_nocharge_plan_v1",
            "seed_mode": "stateless_ordinal_v1",
            "formula_constraint_mode": "off",
            "crplan_diagnostics": None,
            "planner_generation_latency_sec": 1.0,
            "planner_sampling_seed": 123,
            "planner_input_prompt_sha256": "prompt",
            "planner_input_ids_sha256": "ids",
            "raw_plan_text": text,
            "plan_state": {
                "formula": "Li2O",
                "elements": ["Li", "O"],
                "counts": [2, 1],
                "N": 3,
                "anion_framework": "oxide",
                "generated_rich_fields": generated,
                "validator": validator,
            },
        }

    def smact_row(self):
        return {
            0: {
                "ordinal": 0,
                "parsed": True,
                "formula": "Li2O",
                "valid": True,
                "witness_valid": True,
                "official_witness_parity": True,
                "stratum": "uniform_primary",
            }
        }

    def test_nocharge_arm_classifies_primary_and_has_no_identity_failures(self) -> None:
        summary, records = classify_arm(
            [self.raw_row()],
            self.smact_row(),
            arm="c1",
            denominator=1,
            legacy_classifier=legacy_stub,
        )
        self.assertEqual(summary["legacy_comp_valid_count"], 1)
        self.assertEqual(summary["legacy_primary_nonshortcut_count"], 1)
        self.assertEqual(summary["smact4_uniform_primary_count"], 1)
        self.assertTrue(all(value == 0 for value in summary["identity_failures"].values()))
        self.assertTrue(records[0]["legacy_comp_valid"])

    def test_generated_charge_is_an_engineering_failure(self) -> None:
        summary, _records = classify_arm(
            [self.raw_row(with_charge=True)],
            self.smact_row(),
            arm="c1",
            denominator=1,
            legacy_classifier=legacy_stub,
        )
        self.assertEqual(summary["generated_charge_field_count"], 1)
        self.assertEqual(summary["identity_failures"]["nocharge_schema_failure"], 1)


class StatisticalGateTests(unittest.TestCase):
    def test_exact_mcnemar(self) -> None:
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_p(0, 4), 0.125)

    def test_bootstrap_is_deterministic(self) -> None:
        first = paired_bootstrap_ci([1, 0, -1, 1], draws=200, seed=7)
        second = paired_bootstrap_ci([1, 0, -1, 1], draws=200, seed=7)
        self.assertEqual(first, second)

    def test_paired_report_counts_discordance(self) -> None:
        baseline = {0: {"ok": True}, 1: {"ok": False}, 2: {"ok": True}}
        candidate = {0: {"ok": False}, 1: {"ok": True}, 2: {"ok": True}}
        report = paired_binary_report(baseline, candidate, field="ok", denominator=3)
        self.assertEqual(report["baseline_only"], 1)
        self.assertEqual(report["candidate_only"], 1)
        self.assertEqual(report["candidate_minus_baseline_count"], 0)

    def test_64_gate_passes_only_for_three_count_primary_gain(self) -> None:
        summaries = {
            "p0": gate_summary(64, comp=22, primary=19, smact_valid=25, smact_uniform=20),
            "c0": gate_summary(64, comp=20, primary=18, smact_valid=23, smact_uniform=19),
            "c1": gate_summary(64, comp=23, primary=21, smact_valid=26, smact_uniform=22),
        }
        gates, _ = scientific_gates(summaries, stage=64)
        self.assertTrue(all(gates.values()))
        summaries["c1"] = gate_summary(64, comp=22, primary=20, smact_valid=26, smact_uniform=22)
        gates, _ = scientific_gates(summaries, stage=64)
        self.assertFalse(gates["c1_minus_c0_legacy_comp_valid_gain_ge_3"])
        self.assertFalse(gates["c1_minus_c0_legacy_primary_gain_ge_3"])

    def test_256_gate_requires_noninferiority_to_p0(self) -> None:
        summaries = {
            "p0": gate_summary(256, comp=108, primary=98, smact_valid=120, smact_uniform=104),
            "c0": gate_summary(256, comp=100, primary=90, smact_valid=112, smact_uniform=96),
            "c1": gate_summary(256, comp=108, primary=98, smact_valid=120, smact_uniform=104),
        }
        gates, _ = scientific_gates(summaries, stage=256)
        self.assertTrue(all(gates.values()))
        summaries["p0"] = gate_summary(256, comp=109, primary=99, smact_valid=120, smact_uniform=104)
        gates, _ = scientific_gates(summaries, stage=256)
        self.assertFalse(gates["c1_legacy_comp_valid_not_below_p0"])
        self.assertFalse(gates["c1_legacy_primary_not_below_p0"])


if __name__ == "__main__":
    unittest.main()
