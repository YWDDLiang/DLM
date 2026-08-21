import math
import unittest

from h1a2_repro.story_panel_analysis import (
    cluster_with_matcher,
    effective_multiplicity,
    multiplicity_gate,
    summarize_story_records,
)


class StoryPanelAnalysisTests(unittest.TestCase):
    def test_cluster_with_matcher(self) -> None:
        class Matcher:
            def fit(self, left, right):
                return left == right

        self.assertEqual(cluster_with_matcher(["a", "a", "b"], Matcher()), [0, 0, 1])

    def test_effective_multiplicity(self) -> None:
        self.assertEqual(effective_multiplicity([]), 0.0)
        self.assertAlmostEqual(effective_multiplicity([0, 0]), 1.0)
        self.assertAlmostEqual(effective_multiplicity([0, 1]), 2.0)
        self.assertTrue(math.isfinite(effective_multiplicity([0, 0, 1])))

    def test_failure_preserving_summary(self) -> None:
        rows = [
            {"plan_source": "learned", "arm": "full", "parsed": False},
            {
                "plan_source": "learned",
                "arm": "full",
                "parsed": True,
                "graph_success": True,
                "lattice_legal": True,
                "model_forward_calls": 15,
            },
        ]
        result = summarize_story_records(rows)["learned/full"]
        self.assertEqual(result["requested"], 2)
        self.assertEqual(result["parsed"]["true"], 1)
        self.assertEqual(result["model_forward_calls"], 15)

    def test_multiplicity_gate(self) -> None:
        gate = multiplicity_gate(
            {
                "a/full": {"clustered": 8, "structure_clusters": 1},
                "b/full": {"clustered": 8, "structure_clusters": 2},
                "a/formula": {"clustered": 4, "structure_clusters": 1},
            }
        )
        self.assertEqual(gate["eligible_full_plans"], 2)
        self.assertEqual(gate["single_cluster_rate"], 0.5)
        self.assertFalse(gate["remove_multiple_realizations_claim"])


if __name__ == "__main__":
    unittest.main()
