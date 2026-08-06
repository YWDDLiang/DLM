from __future__ import annotations

from collections import Counter
import unittest

from workstreams.plangraph_dlm_iclr_20260731.execution.h1_crplan_e1_physical_performance_probe_v1.evaluate_crplan_e1_probe import (
    decision_from_components,
)
from workstreams.plangraph_dlm_iclr_20260731.execution.h1_crplan_e1_physical_performance_probe_v1.run_crplan_e1_probe import (
    ATTEMPTS,
    MODES,
    REFERENCE_ORDINALS,
    build_schedule,
)


class CRPlanE1ContractTests(unittest.TestCase):
    def test_schedule_is_complete_and_position_balanced(self) -> None:
        schedule = build_schedule()
        self.assertEqual(len(schedule), ATTEMPTS * len(MODES))
        per_ordinal = Counter(int(row["ordinal"]) for row in schedule)
        self.assertEqual(per_ordinal, Counter({value: 3 for value in range(18)}))
        mode_position = Counter(
            (str(row["mode"]), int(row["position_within_ordinal"]))
            for row in schedule
        )
        self.assertEqual(
            mode_position,
            Counter(
                {
                    (mode, position): 6
                    for mode in MODES
                    for position in range(3)
                }
            ),
        )
        self.assertEqual(REFERENCE_ORDINALS, (2, 11))

    def test_physical_gate_passes_at_registered_boundaries(self) -> None:
        result = decision_from_components(
            p0_latency={"median_sec": 3.0, "p95_sec": 4.0},
            full_latency={"median_sec": 4.5, "p95_sec": 8.0},
            identity_ok=True,
            trace_parity_ok=True,
            scalar_parity_ok=True,
            no_engineering_failures=True,
            applicable_count=10,
            affected_count=1,
        )
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["full_over_p0_median_ratio"], 1.5)
        self.assertEqual(result["full_over_p0_p95_ratio"], 2.0)
        self.assertEqual(result["preterminal_affected_rate"], 0.1)

    def test_missing_mechanism_activity_fails(self) -> None:
        result = decision_from_components(
            p0_latency={"median_sec": 3.0, "p95_sec": 4.0},
            full_latency={"median_sec": 3.1, "p95_sec": 4.1},
            identity_ok=True,
            trace_parity_ok=True,
            scalar_parity_ok=True,
            no_engineering_failures=True,
            applicable_count=12,
            affected_count=0,
        )
        self.assertFalse(result["gate_passed"])
        self.assertFalse(
            result["gates"]["preterminal_affected_rate_ge_5pct"]
        )

    def test_large_logical_state_count_is_report_only_in_e1(self) -> None:
        result = decision_from_components(
            p0_latency={"median_sec": 3.0, "p95_sec": 4.0},
            full_latency={"median_sec": 3.5, "p95_sec": 5.0},
            identity_ok=True,
            trace_parity_ok=True,
            scalar_parity_ok=True,
            no_engineering_failures=True,
            applicable_count=18,
            affected_count=2,
        )
        self.assertTrue(result["gate_passed"])
        self.assertNotIn("states", " ".join(result["gates"]))

    def test_any_exactness_failure_kills_probe(self) -> None:
        result = decision_from_components(
            p0_latency={"median_sec": 3.0, "p95_sec": 4.0},
            full_latency={"median_sec": 3.1, "p95_sec": 4.1},
            identity_ok=True,
            trace_parity_ok=False,
            scalar_parity_ok=True,
            no_engineering_failures=True,
            applicable_count=10,
            affected_count=2,
        )
        self.assertFalse(result["gate_passed"])


if __name__ == "__main__":
    unittest.main()
