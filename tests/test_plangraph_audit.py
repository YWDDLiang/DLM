import json
from pathlib import Path
import tempfile
import unittest

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer, parse_dynamic_answer
from crystal_dlm.plangraph_audit import audit_plangraph_jsonl
from crystal_dlm.r5_plan_state import plan_state_from_arrays


class PlanGraphAuditTests(unittest.TestCase):
    def make_record(self):
        answer, _diagnostics = arrays_to_dynamic_answer(
            lengths=[3.1, 3.1, 5.2],
            angles=[90.0, 90.0, 120.0],
            species=["Li", "O", "Li"],
            frac_coords=[
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5],
            ],
        )
        arrays = parse_dynamic_answer(answer, strict=True)
        plan = plan_state_from_arrays(
            arrays,
            metadata={
                "spacegroup.number": 194,
                "e_above_hull": 0.0,
            },
        )
        return {
            "representation": "dynamic_v1",
            "plan_state": plan,
            "answer": answer,
            "metadata": {"formation_energy": -1.0},
        }

    def test_audit_retains_denominator_and_counts_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            good = self.make_record()
            bad = {**good, "answer": "not dynamic tokens"}
            path.write_text(
                "\n".join(
                    [
                        json.dumps(good),
                        json.dumps(bad),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            report = audit_plangraph_jsonl(path)
            self.assertEqual(report["total_rows"], 2)
            self.assertEqual(report["converted_rows"], 1)
            self.assertEqual(report["failed_rows"], 1)
            self.assertEqual(report["conversion_rate"], 0.5)
            self.assertEqual(
                report["failure_categories"]["answer_plan_mismatch"],
                1,
            )
            self.assertEqual(
                report["field_coverage"]["spacegroup_known"]["count"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
