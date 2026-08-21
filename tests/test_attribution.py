import unittest

from h1a2_repro.attribution import (
    STAGES,
    analyze_cohort,
    atom_weighted_element_distribution,
    composition_family,
    discovery_pareto,
    element_presence_distribution,
    jensen_shannon,
    normalize_attempt,
    paired_mcnemar,
    parse_formula,
    standardized_rate,
    symmetric_decomposition,
    total_variation,
)


def attempt(formula: str, *, strict: bool = False, meta: bool = False, **extra):
    row = {
        "attempt_id": f"{formula}-{strict}-{meta}",
        "formula": formula,
        "stages": {stage: True for stage in STAGES},
        "outcomes": {"novel": True, "unique": True, "strict_sun": strict, "meta_sun": meta},
    }
    row.update(extra)
    return row


class AttributionTests(unittest.TestCase):
    def test_formula_and_family(self) -> None:
        self.assertEqual(parse_formula("Li2O"), [("Li", 2), ("O", 1)])
        self.assertEqual(composition_family(["Fe", "Ni"]), "all-metal/intermetallic")
        self.assertEqual(composition_family(["Li", "O"]), "oxide")
        self.assertEqual(composition_family(["Li", "O", "Cl"]), "mixed oxide+halide")
        with self.assertRaises(ValueError):
            parse_formula("Ca(OH)2")

    def test_probability_distances(self) -> None:
        self.assertAlmostEqual(total_variation({"a": 1.0}, {"b": 1.0}), 1.0)
        self.assertAlmostEqual(jensen_shannon({"a": 1.0}, {"a": 1.0}), 0.0)

    def test_funnel_keeps_requested_denominator(self) -> None:
        good = attempt("Li2O", strict=True, meta=True)
        failed = attempt("NaCl")
        failed["stages"].update(
            {
                "body_success": False,
                "refined": False,
                "reconstructed": False,
                "hull_known": False,
            }
        )
        report = analyze_cohort([good, failed])
        self.assertEqual(report["funnel"]["requested"], 2)
        self.assertEqual(report["funnel"]["body_success"], 1)
        self.assertEqual(report["funnel"]["novel_unique"], 2)
        self.assertEqual(report["strict_rate"], 0.5)
        self.assertEqual(report["hull_known_rates"]["denominator"], 1)
        self.assertEqual(report["hull_known_rates"]["unknown"], 0)
        self.assertEqual(report["hull_known_rates"]["not_reconstructed"], 1)
        self.assertEqual(report["hull_known_rates"]["strict_rate"], 1.0)

    def test_element_distributions_and_pareto(self) -> None:
        rows = [normalize_attempt(attempt("Li2O", strict=True)), normalize_attempt(attempt("Fe"))]
        self.assertEqual(element_presence_distribution(rows)["Li"], 0.5)
        self.assertAlmostEqual(atom_weighted_element_distribution(rows)["Li"], 0.5)
        pareto = discovery_pareto(rows)
        self.assertEqual(pareto["novel_unique"], 2)
        self.assertAlmostEqual(pareto["strict_conversion_within_novel_unique"], 0.5)

    def test_symmetric_decomposition_identity(self) -> None:
        rows_a = [attempt("Li2O", strict=True), attempt("Li2O"), attempt("Fe", strict=True)]
        rows_b = [attempt("Li2O"), attempt("Fe"), attempt("Fe")]
        normalized_a = [normalize_attempt(row) for row in rows_a]
        normalized_b = [normalize_attempt(row) for row in rows_b]
        result = symmetric_decomposition(normalized_a, normalized_b, "strict_sun")
        self.assertTrue(result["estimable"])
        self.assertAlmostEqual(
            result["gap_common"],
            result["mix_effect"] + result["conditional_effect"],
            places=12,
        )
        self.assertAlmostEqual(result["identity_residual"], 0.0, places=12)

    def test_standardization_reports_overlap_and_ess(self) -> None:
        target = [normalize_attempt(attempt("Li2O", meta=True)), normalize_attempt(attempt("Fe"))]
        reference = [
            normalize_attempt(attempt("Li2O")),
            normalize_attempt(attempt("Li2O")),
            normalize_attempt(attempt("Fe")),
        ]
        result = standardized_rate(target, reference, "meta_sun")
        self.assertTrue(result["estimable"])
        self.assertEqual(result["reference_coverage"], 1.0)
        self.assertGreater(result["effective_sample_size"], 0.0)

    def test_non_monotone_stage_rejected(self) -> None:
        row = attempt("Li2O")
        row["stages"]["plan_eligible"] = False
        with self.assertRaises(ValueError):
            normalize_attempt(row)

    def test_exact_paired_mcnemar(self) -> None:
        rows_a = [normalize_attempt(attempt("Li2O", strict=True)), normalize_attempt(attempt("Fe"))]
        rows_b = [normalize_attempt(attempt("Li2O")), normalize_attempt(attempt("Fe", strict=True))]
        for index, row in enumerate(rows_a):
            row["ordinal"] = index
        for index, row in enumerate(rows_b):
            row["ordinal"] = index
        report = paired_mcnemar(rows_a, rows_b, "strict_sun", key="ordinal")
        self.assertEqual(report["known_both_pairs"], 2)
        self.assertEqual(report["discordant_a_only"], 1)
        self.assertEqual(report["discordant_b_only"], 1)
        self.assertEqual(report["exact_two_sided_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
