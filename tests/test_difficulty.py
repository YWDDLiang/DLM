import unittest

from h1a2_repro.difficulty import (
    cross_fitted_difficulty,
    deduplicate,
    difficulty_weights,
    kitagawa_decomposition,
    normalize_attempt,
    summarize,
)


def row(ordinal, formula, elements, n, strict, meta, *, charge="neutral_plausible", known=True):
    return {
        "ordinal": ordinal,
        "planner_plan_features": {
            "formula": formula,
            "elements": elements,
            "N": n,
            "charge_bucket": charge,
        },
        "strict_sun_intersection": strict,
        "meta_sun_intersection": meta,
        "official_hull_known": known,
    }


class DifficultyTest(unittest.TestCase):
    def attempts(self, method="base"):
        rows = [
            row(0, "Li2O", ["Li", "O"], 3, True, True),
            row(1, "NaCl", ["Na", "Cl"], 2, False, True),
            row(2, "Fe2O3", ["Fe", "O"], 5, False, False),
            row(3, "FeNi", ["Fe", "Ni"], 2, False, True, charge="all_metal"),
            row(4, "KBr", ["K", "Br"], 2, False, False),
            row(5, "MgO", ["Mg", "O"], 2, False, True),
            row(6, "CaS", ["Ca", "S"], 2, False, False),
            row(7, "AlN", ["Al", "N"], 2, True, True),
            row(8, "SiP", ["Si", "P"], 2, False, True),
            row(9, "CsI", ["Cs", "I"], 2, False, False),
        ]
        return [normalize_attempt(value, cohort_id=f"{method}-cohort", method=method) for value in rows]

    def test_family_and_unknown_contract(self):
        oxide = normalize_attempt(row(0, "Li2O", ["Li", "O"], 3, False, True), cohort_id="c", method="m")
        alloy = normalize_attempt(row(1, "FeNi", ["Fe", "Ni"], 2, False, True, charge="all_metal"), cohort_id="c", method="m")
        unknown = normalize_attempt(row(2, "NaCl", ["Na", "Cl"], 2, False, False, known=False), cohort_id="c", method="m")
        self.assertEqual(oxide.family, "oxide")
        self.assertEqual(alloy.family, "all_metal")
        self.assertEqual(oxide.reward, 1.0)
        self.assertIsNone(unknown.reward)

    def test_dedup_and_summary(self):
        attempts = self.attempts()
        unique, duplicates = deduplicate(attempts + [attempts[0]])
        self.assertEqual(duplicates, 1)
        self.assertEqual(len(unique), 10)
        families = {item["value"]: item for item in summarize(unique, "family")}
        self.assertEqual(families["halide"]["attempts"], 3)

    def test_cross_fit_and_ess_control(self):
        attempts = self.attempts()
        baselines = cross_fitted_difficulty(attempts, folds=2, prior_strength=2.0)
        weights, report = difficulty_weights(attempts, baselines, max_weight=3.0, min_ess_ratio=0.7)
        self.assertEqual(len(weights), len(attempts))
        self.assertGreaterEqual(report["ess_ratio"], 0.7 - 1e-6)
        self.assertLessEqual(report["max_weight"], 3.0)

    def test_decomposition_is_finite(self):
        baseline = self.attempts("base")
        candidate = self.attempts("candidate")
        result = kitagawa_decomposition(baseline, candidate, endpoint="meta_sun")
        self.assertAlmostEqual(result["proposal_mix_effect"], 0.0)
        self.assertAlmostEqual(result["conditional_realization_effect"], 0.0)


if __name__ == "__main__":
    unittest.main()

