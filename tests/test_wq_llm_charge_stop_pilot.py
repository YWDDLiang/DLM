from __future__ import annotations

import unittest

from scripts.a800.run_wq_llm_charge_stop_paired64_v1 import summarize_rows


GATE = {
    "maximum_generation_success_loss": 0,
    "minimum_composition_valid_gain_count": 3,
    "minimum_charge_failure_reduction_count": 3,
    "maximum_valid_to_invalid_pairs": 1,
    "maximum_unique_formula_rate_loss": 0.05,
    "maximum_mean_atom_count_increase": 4.0,
}


def row(
    pair: int,
    arm: str,
    *,
    valid: bool,
    formula: str,
    atom_count: int = 8,
    status: str = "succeeded",
):
    return {
        "pair_id": f"pair-{pair}",
        "arm": arm,
        "status": status,
        "composition_valid": valid,
        "composition_reason": (
            "charge_neutral_pauling_valid"
            if valid
            else "charge_neutrality_fail"
        ),
        "formula": formula,
        "proposal_text_sha256": f"{arm}-{pair}",
        "atom_count": atom_count,
        "orbit_count": 3,
        "usage": {"generated_tokens": 20},
    }


class WQLLMChargeStopPilotSummaryTests(unittest.TestCase):
    def test_promotes_directional_gain_without_diversity_collapse(self) -> None:
        rows = []
        for pair in range(64):
            baseline_valid = pair >= 16
            masked_valid = pair >= 12
            rows.extend(
                (
                    row(
                        pair,
                        "baseline",
                        valid=baseline_valid,
                        formula=f"B{pair}",
                    ),
                    row(
                        pair,
                        "charge_stop",
                        valid=masked_valid,
                        formula=f"M{pair}",
                        atom_count=9,
                    ),
                )
            )
        result = summarize_rows(rows, attempts_per_arm=64, gate=GATE)
        self.assertTrue(result["promotion_pass"])
        self.assertEqual(result["paired"]["composition_valid_gain_count"], 4)
        self.assertEqual(
            result["paired"]["baseline_invalid_to_mask_valid"],
            4,
        )

    def test_fails_when_mask_converts_too_many_valid_pairs_to_invalid(self) -> None:
        rows = []
        for pair in range(64):
            baseline_valid = pair >= 16
            masked_valid = pair >= 12 and pair not in {20, 21}
            rows.extend(
                (
                    row(
                        pair,
                        "baseline",
                        valid=baseline_valid,
                        formula=f"B{pair}",
                    ),
                    row(
                        pair,
                        "charge_stop",
                        valid=masked_valid,
                        formula=f"M{pair}",
                    ),
                )
            )
        result = summarize_rows(rows, attempts_per_arm=64, gate=GATE)
        self.assertFalse(result["promotion_pass"])
        self.assertFalse(
            result["promotion_checks"]["valid_to_invalid_pairs_bounded"]
        )

    def test_generation_failure_stays_in_denominator(self) -> None:
        rows = []
        for pair in range(64):
            rows.append(
                row(
                    pair,
                    "baseline",
                    valid=pair >= 16,
                    formula=f"B{pair}",
                )
            )
            rows.append(
                row(
                    pair,
                    "charge_stop",
                    valid=pair >= 12,
                    formula=f"M{pair}",
                    status="failed" if pair == 0 else "succeeded",
                )
            )
        result = summarize_rows(rows, attempts_per_arm=64, gate=GATE)
        self.assertEqual(result["arms"]["charge_stop"]["terminal"], 64)
        self.assertEqual(result["arms"]["charge_stop"]["failed"], 1)
        self.assertFalse(
            result["promotion_checks"]["masked_generation_success_noninferior"]
        )


if __name__ == "__main__":
    unittest.main()
