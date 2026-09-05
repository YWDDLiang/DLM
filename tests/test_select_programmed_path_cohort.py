import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("select_programmed_path_cohort", Path(__file__).resolve().parents[1] / "scripts/select_programmed_path_cohort.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ParserPrefixTest(unittest.TestCase):
    def test_only_parser_and_original_order_determine_selection(self):
        ledger = [{"sample_idx": i, "planner_valid": i != 2} for i in range(6)]
        paths = [{"sample_idx": i, "source_split": "evaluation", "success": i != 1,
                  "body": "bad" if i == 3 else "good", "energy": -100. if i == 3 else 100.}
                 for i in (0, 1, 3, 4, 5)]
        rows = MODULE.complete_requests(paths, ledger)
        seen = []
        def parser(row):
            seen.append(row["sample_idx"])
            if row["body"] == "bad":
                raise ValueError("bad CIF")
        selected, accounting, report = MODULE.select_prefix(rows, target=3, parser=parser)
        self.assertEqual([r["sample_idx"] for r in selected], [0, 1, 4])
        self.assertEqual([r["evaluation_ordinal"] for r in selected], [0, 1, 2])
        self.assertEqual(seen, [0, 1, 3, 4])
        self.assertEqual(accounting[-1]["status"], "not_examined_after_target")
        self.assertEqual(report["source_requests_examined"], 5)

    def test_missing_valid_planner_body_is_not_imputed_as_failure(self):
        with self.assertRaises(ValueError):
            MODULE.complete_requests([], [{"sample_idx": 0, "planner_valid": True}])

    def test_insufficient_parseable_stream_is_explicit(self):
        with self.assertRaises(ValueError):
            MODULE.select_prefix([{"source_request_ordinal": 0}], target=2, parser=lambda row: None)


if __name__ == "__main__":
    unittest.main()
