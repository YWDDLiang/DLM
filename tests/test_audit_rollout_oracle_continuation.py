import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_rollout_oracle_continuation",
    ROOT / "scripts" / "audit_rollout_oracle_continuation.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import audit_rollout_oracle_continuation.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
counts = MODULE.counts
paired_summary = MODULE.paired_summary


def metric(valid: bool) -> dict:
    return {
        "parsed": True,
        "composition_valid": True,
        "structure_valid": valid,
        "direct": valid,
        "reason": "" if valid else "crysllmgen_invalid",
    }


class OracleContinuationSummaryTests(unittest.TestCase):
    def test_counts_and_paired_flips(self) -> None:
        base = {0: metric(False), 1: metric(True), 2: metric(False)}
        candidate = {0: metric(True), 1: metric(False), 2: metric(True)}
        summary = paired_summary(base, candidate)
        self.assertEqual(summary["invalid_to_valid"], 2)
        self.assertEqual(summary["valid_to_invalid"], 1)
        self.assertEqual(summary["net_invalid_to_valid"], 1)
        self.assertEqual(summary["direct_delta"], 1)
        self.assertEqual(counts(candidate.values())["direct"], 2)

    def test_sample_set_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample sets differ"):
            paired_summary({0: metric(True)}, {1: metric(True)})


if __name__ == "__main__":
    unittest.main()
