from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


THIS_DIR = Path(__file__).resolve().parent
INNOVATION_ROOT = THIS_DIR.parent
CODE_ROOT = INNOVATION_ROOT / "code"
REACTIVATION_ROOT = INNOVATION_ROOT.parent
PROJECT_ROOT = INNOVATION_ROOT.parents[2]
RESTORED_BASELINE_ROOT = REACTIVATION_ROOT / "baseline"
RUNTIME_ROOT = (
    RESTORED_BASELINE_ROOT
    if (RESTORED_BASELINE_ROOT / "crystal_dlm").is_dir()
    else PROJECT_ROOT
)
for value in (str(CODE_ROOT), str(RUNTIME_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

import evaluate_jointchem_plans_v2 as evaluator  # noqa: E402


class JointChemPlanEvaluationV2Tests(unittest.TestCase):
    def test_unit_cell_counts_are_gcd_reduced(self):
        elems, counts = evaluator.normalized_plan_composition(
            ["Se", "Ta"],
            [8, 4],
        )
        self.assertEqual(counts, (2, 1))
        self.assertEqual(len(elems), 2)

    def test_audit_passes_reduced_counts_to_classifier(self):
        record = {
            "sample_idx": 0,
            "parsed_plan": {
                "formula": "Se8Ta4",
                "elements": ["Se", "Ta"],
                "counts": [8, 4],
                "N": 12,
                "anion_framework": "chalcogenide",
                "charge_bucket": "neutral_plausible",
            },
        }
        observed = []

        def classifier(elems, counts):
            observed.append((tuple(elems), tuple(counts)))
            return {"valid": True, "reason": "charge_neutral_pauling_valid"}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with mock.patch.object(evaluator, "classify_smact_validity", side_effect=classifier):
                result = evaluator.composition_audit(path, denominator=1)

        self.assertEqual(observed[0][1], (2, 1))
        self.assertEqual(result["composition_valid_count"], 1)
        self.assertEqual(result["count_normalization"], evaluator.NORMALIZATION)

    def test_invalid_counts_fail_closed(self):
        with self.assertRaises(ValueError):
            evaluator.normalized_plan_composition(["O", "Mg"], [8, 0])


if __name__ == "__main__":
    unittest.main()
