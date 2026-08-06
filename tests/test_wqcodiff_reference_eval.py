from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.reference_eval import _queue


class ReferenceEvaluationQueueTests(unittest.TestCase):
    def _write_queue(
        self,
        path: Path,
        *,
        evaluator: str = "mattersim",
        contract_hash: str = "contract-mattersim",
        closed: bool = False,
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema": "wqcodiff_evaluator_hull_v1",
                    "evaluator": evaluator,
                    "contract_hash": contract_hash,
                    "closed": closed,
                    "pending_relaxation_ids": ["mp-1", "mp-2"],
                }
            ),
            encoding="utf-8",
        )

    def test_queue_is_bound_to_evaluator_and_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.json"
            self._write_queue(path)
            self.assertEqual(
                _queue(
                    str(path),
                    evaluator="mattersim",
                    contract_hash="contract-mattersim",
                ),
                {"mp-1", "mp-2"},
            )
            with self.assertRaisesRegex(ValueError, "evaluator mismatch"):
                _queue(
                    str(path),
                    evaluator="mace",
                    contract_hash="contract-mattersim",
                )
            with self.assertRaisesRegex(ValueError, "contract hash mismatch"):
                _queue(
                    str(path),
                    evaluator="mattersim",
                    contract_hash="other-contract",
                )

    def test_closed_queue_still_requires_matching_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.json"
            self._write_queue(path, closed=True)
            self.assertEqual(
                _queue(
                    str(path),
                    evaluator="mattersim",
                    contract_hash="contract-mattersim",
                ),
                set(),
            )
            with self.assertRaisesRegex(ValueError, "evaluator mismatch"):
                _queue(
                    str(path),
                    evaluator="chgnet",
                    contract_hash="contract-mattersim",
                )


if __name__ == "__main__":
    unittest.main()
