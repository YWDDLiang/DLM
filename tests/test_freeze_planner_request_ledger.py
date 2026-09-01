import tempfile
from pathlib import Path
import unittest

from scripts.freeze_planner_request_ledger import build_rows


class PlannerRequestLedgerTest(unittest.TestCase):
    def test_fixed_ordinals_are_complete_and_outcome_blind(self) -> None:
        rows = build_rows(requested=1200, seed=23, purpose="official_available_1200")
        self.assertEqual(len(rows), 1200)
        self.assertEqual([row["sample_idx"] for row in rows], list(range(1200)))
        self.assertTrue(all(row["outcomes_read"] is False for row in rows))
        self.assertTrue(all(row["stability_goal"] == "meta_or_better" for row in rows))

    def test_invalid_contract_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            build_rows(requested=0, seed=23, purpose="x")
        with self.assertRaises(ValueError):
            build_rows(requested=1, seed=-1, purpose="x")
        with self.assertRaises(ValueError):
            build_rows(requested=1, seed=23, purpose="")


if __name__ == "__main__":
    unittest.main()
