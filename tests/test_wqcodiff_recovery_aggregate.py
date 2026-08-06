from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.recovery_aggregate import _edit, _net, _tangent, aggregate_recovery


class RecoveryAggregateTests(unittest.TestCase):
    def test_failed_attempts_receive_registered_penalties_instead_of_disappearing(self) -> None:
        failed = {"status": "failed"}
        self.assertEqual(_edit(failed), 20.0)
        self.assertEqual(_tangent(failed), 1.0)
        self.assertEqual(_net(failed), 0.0)

    def test_missing_mechanism_cells_cannot_silently_promote_dlm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cells.jsonl"
            rows = []
            for level in (0.7, 0.9):
                for method, exact, edit in (
                    ("B-WQ-AR", False, 2),
                    ("B-WQ-DLM-MONO", True, 0),
                ):
                    rows.append(
                        {
                            "schema": "wqcodiff_recovery_attempt_v1",
                            "attempt_id": f"{method}-{level}",
                            "material_id": "m0",
                            "method": method,
                            "corruption_seed": 101,
                            "corruption_level": level,
                            "operator": "wrong-species",
                            "geometry_condition": "noisy",
                            "schedule": "fixed",
                            "control": "none",
                            "subset_hash": "fixed",
                            "status": "succeeded",
                            "exact_full_protostructure_recovery": exact,
                            "topology_edit_distance_after": edit,
                            "tangent_coordinate_error": 0.1,
                            "mechanism": {"net_correction": int(exact)},
                        }
                    )
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            result = aggregate_recovery(
                [source],
                output_path=Path(directory) / "gate.json",
            )
            self.assertFalse(result["dlm_promoted"])
            self.assertEqual(
                result["required_claim_action"],
                "delete_dlm_superiority_claim_and_use_best_ar_or_d3pm",
            )
            self.assertTrue(result["gates"]["high_corruption_exact_ci_lower_positive"])


if __name__ == "__main__":
    unittest.main()
