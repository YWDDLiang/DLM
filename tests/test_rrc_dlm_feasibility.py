import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_rrc_dlm_feasibility",
    ROOT / "scripts/analyze_rrc_dlm_feasibility.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(ordinal, energy, hull, strict=False, meta=False):
    return {
        "ordinal": ordinal,
        "chgnet_energy_per_atom": energy,
        "official_e_above_hull": hull,
        "official_hull_status": "known",
        "strict_stable": strict,
        "strict_sun": strict,
        "meta_stable": meta,
        "meta_sun": meta,
    }


class RRCDLMFeasibilityTest(unittest.TestCase):
    def setUp(self):
        self.cells = {
            "a": {
                0: row(0, -2.0, 0.00, strict=True, meta=True),
                1: row(1, -1.0, 0.20),
            },
            "b": {
                0: row(0, -1.9, 0.10, meta=True),
                1: row(1, -1.2, 0.00, strict=True, meta=True),
            },
            "c": {
                0: row(0, -1.8, 0.20),
                1: row(1, -1.1, 0.10, meta=True),
            },
        }

    def test_rank_agreement_uses_within_composition_order(self):
        summary = MODULE.rank_agreement(self.cells, [0, 1])
        self.assertEqual(summary["discordant"], 0)
        self.assertEqual(summary["concordant"], 6)
        self.assertEqual(summary["accuracy_non_tie"], 1.0)

    def test_oracle_curve_improves_low_energy_selection(self):
        curve = MODULE.oracle_curve(self.cells, [0, 1])
        self.assertEqual(curve["k3"]["mean"]["strict_sun"], 2.0)
        self.assertGreater(
            curve["k1"]["mean"]["official_e_hull_q50"],
            curve["k3"]["mean"]["official_e_hull_q50"],
        )

    def test_threshold_pair_summary_counts_variable_compositions(self):
        strict = MODULE.threshold_pair_summary(
            self.cells, [0, 1], "strict_stable"
        )
        self.assertEqual(strict["variable_compositions"], 2)
        self.assertEqual(strict["crossing_pairs"], 4)


if __name__ == "__main__":
    unittest.main()
