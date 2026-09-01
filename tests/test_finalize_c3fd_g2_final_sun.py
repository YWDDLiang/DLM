import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "finalize_c3fd_g2_final_sun",
    ROOT / "scripts/finalize_c3fd_g2_final_sun.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import G2 SUN finalizer")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RuntimeStub:
    @staticmethod
    def _exact_mcnemar(control, candidate):
        return {
            "control_only": sum(left and not right for left, right in zip(control, candidate)),
            "candidate_only": sum(right and not left for left, right in zip(control, candidate)),
            "two_sided_exact_p": 1.0,
        }


class G2SUNFinalizerTest(unittest.TestCase):
    def test_distribution_keeps_official_thresholds(self):
        result = MODULE.distribution([-0.1, 0.0, 0.1, 0.4])
        self.assertEqual(result["known"], 4)
        self.assertEqual(result["ecdf"]["0.0"], 0.5)
        self.assertEqual(result["ecdf"]["0.1"], 0.75)

    def test_paired_binary_preserves_fixed256_identity(self):
        base = []
        g2 = []
        for index in range(MODULE.ATTEMPTS):
            identity = f"X:{index + 1}"
            base.append(
                {
                    "ordinal": index,
                    "requested_exact_composition_identity": identity,
                    "strict_sun": index == 0,
                }
            )
            g2.append(
                {
                    "ordinal": index,
                    "requested_exact_composition_identity": identity,
                    "strict_sun": index in (0, 1),
                }
            )
        result = MODULE.paired_boolean(RuntimeStub, base, g2, "strict_sun")
        self.assertEqual(result["control_only"], 0)
        self.assertEqual(result["candidate_only"], 1)

    def test_render_states_single_stream_and_fixed_denominator(self):
        cell = {
            "stage": "raw",
            "route": "G2",
            "direct_joint": 121,
            "novel": 100,
            "unique": 90,
            "novel_unique": 80,
            "hull_known": 79,
            "strict_sun": 26,
            "meta_sun": 128,
            "strict_sun_rate": 26 / 256,
            "meta_sun_rate": 128 / 256,
        }
        report = {
            "cells": [cell],
            "target_evaluation": {
                "raw_G2": {
                    "strict_rate": 26 / 256,
                    "meta_rate": 128 / 256,
                    "strict_met": True,
                    "meta_met": True,
                }
            },
            "paired_continuous": {},
            "paired_binary": {},
            "official_cache": {
                "query_resolved": 241,
                "query_count": 246,
                "query_unresolved": 5,
            },
        }
        text = MODULE.render(report)
        self.assertIn("26/256", text)
        self.assertIn("one pre-registered stream", text)


if __name__ == "__main__":
    unittest.main()
