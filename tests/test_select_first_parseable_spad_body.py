from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_first_parseable_spad_body",
    ROOT / "scripts" / "select_first_parseable_spad_body.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FirstParseableBodyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = [
            {"sample_idx": idx, "planner_valid": idx != 1} for idx in range(7)
        ]
        self.bodies = [
            {"sample_idx": 0, "parsed": True, "cif": "ok-0"},
            {"sample_idx": 2, "parsed": False, "cif": None},
            {"sample_idx": 3, "parsed": True, "cif": "bad-independent"},
            {"sample_idx": 4, "parsed": True, "cif": "ok-4"},
            {"sample_idx": 5, "parsed": True, "cif": "ok-5"},
            {"sample_idx": 6, "parsed": True, "cif": "ok-6"},
        ]
        self.graphs = [
            {"graph": source_idx, "sample_idx": source_idx}
            for source_idx in (0, 3, 4, 5, 6)
        ]

    @staticmethod
    def parse_cif(value: str) -> None:
        if value == "bad-independent":
            raise ValueError("not a CIF")

    def test_first_parseable_only_and_reindex(self) -> None:
        rows, graphs, accounting, report = MODULE.select_first_parseable(
            self.bodies,
            self.ledger,
            self.graphs,
            target=3,
            parse_cif=self.parse_cif,
        )
        self.assertEqual([row["source_attempt_idx"] for row in rows], [0, 4, 5])
        self.assertEqual([row["sample_idx"] for row in rows], [0, 1, 2])
        self.assertEqual([row["graph"] for row in graphs], [0, 4, 5])
        self.assertEqual([row["sample_idx"] for row in graphs], [0, 1, 2])
        self.assertEqual(report["source_requests_examined"], 6)
        self.assertEqual(report["discarded_before_target"], 3)
        self.assertEqual(report["planner_invalid_before_target"], 1)
        self.assertEqual(report["body_parser_failures_before_target"], 1)
        self.assertEqual(report["independent_parser_failures_before_target"], 1)
        self.assertFalse(report["energy_hull_novelty_sun_read"])
        self.assertEqual(accounting[6]["status"], "not_examined_after_target")

    def test_insufficient_fixed_stream_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 3 parseable CIFs"):
            MODULE.select_first_parseable(
                self.bodies[:5],
                self.ledger[:6],
                self.graphs[:4],
                target=4,
                parse_cif=self.parse_cif,
            )

    def test_graph_count_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            MODULE.select_first_parseable(
                self.bodies,
                self.ledger,
                self.graphs[:-1],
                target=3,
                parse_cif=self.parse_cif,
            )

    def test_valid_planner_requires_body_attempt(self) -> None:
        bodies = [row for row in self.bodies if row["sample_idx"] != 2]
        graphs = self.graphs
        with self.assertRaisesRegex(ValueError, "misses body"):
            MODULE.select_first_parseable(
                bodies,
                self.ledger,
                graphs,
                target=3,
                parse_cif=self.parse_cif,
            )


if __name__ == "__main__":
    unittest.main()
