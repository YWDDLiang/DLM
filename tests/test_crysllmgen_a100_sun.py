from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.crysllmgen.a100_sun import parse_a100_summary


class R5CA100SunTests(unittest.TestCase):
    def test_parses_exact_resumable_summary_without_using_adjusted_as_lower_bound(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "RESULTS_SUMMARY.md"
            path.write_text(
                "\n".join(
                    (
                        "# Resumable Full S.U.N. Summary",
                        "generated: fixture.pt",
                        "Total generated: 256",
                        "Reconstructed: 250",
                        "Novel: 210/250 (84.00%)",
                        "Unique: 200/250 (80.00%)",
                        "Novel + Unique: 180/250 (72.00%)",
                        "E_hull evaluated: 170/180",
                        "E_hull unknown: 10/180",
                        "Stable: 30/170 evaluated (17.65%)",
                        "Full S.U.N. lower-bound: 12.00%",
                        "Coverage-adjusted S.U.N. estimate: 12.71%",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            result = parse_a100_summary(path)
            self.assertEqual(result["total_generated"], 256)
            self.assertEqual(result["reconstructed"], 250)
            self.assertEqual(result["stable"], {"numerator": 30, "denominator": 170})
            self.assertEqual(result["full_sun_lower_bound_percent"], 12.0)
            self.assertEqual(result["coverage_adjusted_percent"], 12.71)

    def test_rejects_incomplete_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "RESULTS_SUMMARY.md"
            path.write_text("Total generated: 256\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing reconstructed"):
                parse_a100_summary(path)


if __name__ == "__main__":
    unittest.main()

