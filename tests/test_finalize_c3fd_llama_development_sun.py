import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_c3fd_llama_development_sun",
    ROOT / "scripts/finalize_c3fd_llama_development_sun.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import development SUN finalizer")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DevelopmentSUNFinalizerTest(unittest.TestCase):
    def test_aggregate_means_stream_rates(self):
        cells = []
        for group, arms in MODULE.GROUPS.items():
            for stage in MODULE.STAGES:
                for arm in arms:
                    for stream, strict, meta in ((17, 1, 10), (18, 3, 14)):
                        cells.append(
                            {
                                "group": group,
                                "stage": stage,
                                "arm": arm,
                                "stream": stream,
                                "strict_sun": strict,
                                "meta_sun": meta,
                                "strict_sun_attempt_rate": strict / 256,
                                "meta_sun_attempt_rate": meta / 256,
                            }
                        )
        result = MODULE.aggregate(cells)
        self.assertEqual(len(result), 8)
        self.assertEqual(result[0]["strict_sun_mean_stream_rate"], 2 / 256)
        self.assertEqual(result[0]["meta_sun_mean_stream_rate"], 12 / 256)

    def test_excluded_runs_are_named(self):
        source = (
            ROOT / "scripts/finalize_c3fd_llama_development_sun.py"
        ).read_text(encoding="utf-8")
        for value in ("alignment_pool38881", "malformed_canary38420", "cancelled38914"):
            self.assertIn(value, source)


if __name__ == "__main__":
    unittest.main()
