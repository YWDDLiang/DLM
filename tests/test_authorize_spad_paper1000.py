from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "authorize_spad_paper1000", ROOT / "scripts" / "authorize_spad_paper1000.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def cell(endpoint: str, strict: int, meta: int, denominator: int = 256) -> dict:
    return {
        "endpoint": endpoint,
        "fixed_denominator": denominator,
        "strict_sun": {"count": strict},
        "meta_sun": {"count": meta},
    }


class Paper1000AuthorizationTests(unittest.TestCase):
    def test_exact_gate(self) -> None:
        report = MODULE.authorize({"cells": [cell("raw", 26, 128)]})
        self.assertTrue(report["paper1000_authorized"])
        self.assertTrue(report["qualifying_endpoints"][0]["exact_10_50"])

    def test_preregistered_near_miss_gate(self) -> None:
        report = MODULE.authorize(
            {"cells": [cell("raw", 22, 140), cell("refined", 23, 125)]}
        )
        self.assertTrue(report["paper1000_authorized"])
        self.assertEqual(report["qualifying_endpoints"][0]["endpoint"], "refined")

    def test_near_on_only_one_metric_does_not_pass(self) -> None:
        report = MODULE.authorize({"cells": [cell("refined", 26, 124)]})
        self.assertFalse(report["paper1000_authorized"])

    def test_denominator_change_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "denominator 256"):
            MODULE.authorize({"cells": [cell("refined", 30, 140, 255)]})


if __name__ == "__main__":
    unittest.main()
