import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "critic_audit", ROOT / "scripts" / "finalize_noisy_critic_feasibility.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load critic feasibility finalizer")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NoisyCriticFeasibilityTest(unittest.TestCase):
    def test_rankdata_averages_ties(self):
        self.assertEqual(MODULE.rankdata([2.0, 1.0, 1.0]), [2.0, 0.5, 0.5])

    def test_perfect_same_plan_ranking(self):
        rows = [
            {
                "sample_idx": 0,
                "stream": index,
                "chgnet_relaxed_energy_per_atom": float(index),
                "mattersim_energy_per_atom": float(index) * 2.0,
            }
            for index in range(4)
        ]
        summary = MODULE.summarize_groups({0: rows})
        self.assertAlmostEqual(summary["pooled_within_plan_spearman"], 1.0)
        self.assertAlmostEqual(summary["pairwise_concordance_auc"], 1.0)
        self.assertAlmostEqual(summary["extreme_pair_direction_agreement"], 1.0)

    def test_reversed_same_plan_ranking(self):
        rows = [
            {
                "sample_idx": 0,
                "stream": index,
                "chgnet_relaxed_energy_per_atom": float(index),
                "mattersim_energy_per_atom": float(-index),
            }
            for index in range(4)
        ]
        summary = MODULE.summarize_groups({0: rows})
        self.assertAlmostEqual(summary["pooled_within_plan_spearman"], -1.0)
        self.assertAlmostEqual(summary["pairwise_concordance_auc"], 0.0)


if __name__ == "__main__":
    unittest.main()
