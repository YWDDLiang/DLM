import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from finalize_g2_full_ab_sun import collect_cache_omissions, promotion_decision


def cells(a_body=255, b_body=254, a_direct=128, b_direct=130):
    return [
        {"stage": "raw", "route": "A", "reconstructed": a_body, "direct_joint": a_direct},
        {"stage": "raw", "route": "B", "reconstructed": b_body, "direct_joint": b_direct},
    ]


def energy(ci95):
    return {"raw_chgnet_energy_per_atom": {"ci95": ci95}}


class PromotionDecisionTest(unittest.TestCase):
    def test_simpler_a_wins_when_registered_b_rules_fail(self) -> None:
        result = promotion_decision(cells(), energy([-0.02, 0.01]))
        self.assertEqual(result["promoted_route"], "A")
        self.assertFalse(result["direct_rule_met"])
        self.assertFalse(result["raw_energy_rule_met"])

    def test_b_can_win_by_clear_raw_energy(self) -> None:
        result = promotion_decision(cells(), energy([-0.03, -0.01]))
        self.assertEqual(result["promoted_route"], "B")
        self.assertTrue(result["raw_energy_rule_met"])

    def test_b_can_win_by_body_and_direct_rule(self) -> None:
        result = promotion_decision(
            cells(a_body=255, b_body=255, a_direct=128, b_direct=136),
            energy([-0.01, 0.01]),
        )
        self.assertEqual(result["promoted_route"], "B")
        self.assertTrue(result["direct_rule_met"])


class CacheOmissionTest(unittest.TestCase):
    def test_only_reconstructed_finite_energy_missing_chemsys_is_unknown(self) -> None:
        class Protocol:
            @staticmethod
            def read_jsonl(_path):
                return [
                    {
                        "reconstructed": True,
                        "chgnet_energy_per_atom": -1.0,
                        "chemsys": "Ag-Ca-Pb",
                    },
                    {
                        "reconstructed": False,
                        "chgnet_energy_per_atom": -1.0,
                        "chemsys": "Not-Reconstructed",
                    },
                    {
                        "reconstructed": True,
                        "chgnet_energy_per_atom": None,
                        "chemsys": "No-Energy",
                    },
                    {
                        "reconstructed": True,
                        "chgnet_energy_per_atom": -1.0,
                        "chemsys": "Known-System",
                    },
                ]

        omitted = collect_cache_omissions(
            Protocol(), Path("unused"), {"Known-System": object()}, set()
        )
        self.assertEqual(omitted, {"Ag-Ca-Pb"})


if __name__ == "__main__":
    unittest.main()
