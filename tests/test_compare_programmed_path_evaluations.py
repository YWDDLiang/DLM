import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("compare_programmed_path_evaluations", Path(__file__).resolve().parents[1] / "scripts/compare_programmed_path_evaluations.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PairedEvaluationTest(unittest.TestCase):
    def test_missing_values_do_not_become_zero_or_hide_coverage(self):
        pairs = [({"gap": 3., "terminal_verified": True}, {"gap": 2., "terminal_verified": True}),
                 ({"gap": 10., "terminal_verified": True}, {"gap": None, "terminal_verified": False}),
                 ({"gap": 1., "terminal_verified": False}, {"gap": 9., "terminal_verified": True})]
        verified = MODULE.paired_metric(pairs, "gap", verified=True)
        self.assertEqual(verified["pairs"], 1)
        self.assertEqual(verified["excluded_requests"], 2)
        self.assertEqual(verified["mean_method_minus_reference"], -1.)
        self.assertEqual(MODULE.paired_metric(pairs, "gap")["mean_method_minus_reference"], 3.5)

    def test_empty_physical_intersection_is_unavailable(self):
        row = MODULE.paired_metric([({"gap": None}, {"gap": 3.})], "gap")
        self.assertEqual(row["pairs"], 0)
        self.assertIsNone(row["mean_method_minus_reference"])

    def test_pairing_checks_frozen_condition_and_rng(self):
        row = dict(sample_idx=0, group_id="eval:0", source_split="evaluation", prompt="p\n", plan_state={"n": 1},
                   species_program=["H"], species_program_source="pointer", num_atoms=1,
                   sampling_seed=12, sampling_batch_size=4, candidate_index=0)
        MODULE.validate_pairing([row], [dict(row)])
        for field, value in (("sampling_seed", 13), ("species_program", ["He"]), ("source_split", "train")):
            with self.assertRaises(ValueError):
                MODULE.validate_pairing([row], [dict(row, **{field: value})])

    def test_duplicate_or_missing_request_cannot_be_paired(self):
        for rows in ([{"sample_idx": 0}, {"sample_idx": 0}], [{"sample_idx": 1}]):
            with self.assertRaises(ValueError):
                MODULE.index_requests(rows)

    def test_binary_comparison_retains_all_failures_and_pairing_order(self):
        fields = ("reconstructed", "native_execution_success", "endpoint_execution_success", "novel", "unique_representative",
                  "novel_unique", "terminal_verified", "strict_stable", "meta_stable", "strict_sun", "meta_sun",
                  "verified_strict_sun", "verified_meta_sun")
        reference = [dict.fromkeys(fields, False) for _ in range(3)]
        for i, row in enumerate(reference):
            row.update(sample_idx=i, group_id=f"eval:{i}", terminal_status="generation_failure",
                       official_hull_status="input_not_reconstructed")
        reference[0]["strict_sun"] = True
        method = [dict(row) for row in reference]
        method[0]["strict_sun"] = False
        method[1]["strict_sun"] = True
        report = MODULE.compare_results(reference, list(reversed(method)))
        counts = report["binary_all_requests"]["strict_sun"]
        self.assertEqual(report["requests"], 3)
        self.assertEqual(counts["reference_only"], 1)
        self.assertEqual(counts["method_only"], 1)
        self.assertEqual(counts["neither"], 1)
        self.assertEqual(counts["change_percentage_points"], 0)


if __name__ == "__main__":
    unittest.main()
