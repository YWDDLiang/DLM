import unittest

from h1a2_repro.refiner_attribution import summarize_refiner_rows


class RefinerAttributionTests(unittest.TestCase):
    def test_requested_failures_and_conversion(self) -> None:
        rows = [
            {"task_id": "a", "body_success": False, "refined": False},
            {
                "task_id": "b",
                "body_success": True,
                "refined": True,
                "n_invariant": True,
                "composition_invariant": True,
                "coordinate_periodic_rms": 0.1,
                "energy_pre": -1.0,
                "energy_post": -1.5,
                "min_distance_pre": 0.4,
                "min_distance_post": 0.8,
                "body_good": False,
                "final_good": True,
                "arm": "full",
                "plan_source": "learned",
            },
        ]
        report = summarize_refiner_rows(rows)
        self.assertEqual(report["requested_attempts"], 2)
        self.assertEqual(report["body_failures"], 1)
        self.assertEqual(report["identity"]["n_invariant"]["rate"], 1.0)
        self.assertEqual(report["paired_deltas"]["energy"]["mean_delta"], -0.5)
        self.assertEqual(report["paired_deltas"]["minimum_distance"]["improved"], 1)
        self.assertEqual(report["body_to_final"]["counts"]["bad_body->good_final"], 1)


if __name__ == "__main__":
    unittest.main()
