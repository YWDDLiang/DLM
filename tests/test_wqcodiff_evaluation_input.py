from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.evaluation import _read_generation


def _row(attempt: str, pair: str, *, method: str = "M-WQ-STRAT-GEO") -> dict[str, object]:
    return {
        "schema": "wqcodiff_generation_attempt_v1",
        "attempt_id": attempt,
        "pair_id": pair,
        "method": method,
        "training_seed": 11,
        "sampling_seed": 101,
        "status": "succeeded",
    }


class EvaluationInputTests(unittest.TestCase):
    def _write(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_single_method_terminal_pool_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation.jsonl"
            self._write(path, [_row("a-1", "p-1"), _row("a-2", "p-2")])
            self.assertEqual(len(_read_generation(path)), 2)

    def test_mixed_methods_and_duplicate_pairs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation.jsonl"
            self._write(
                path,
                [_row("a-1", "p-1"), _row("a-2", "p-2", method="B-WQ-D3PM")],
            )
            with self.assertRaisesRegex(ValueError, "mixes methods"):
                _read_generation(path)
            self._write(path, [_row("a-1", "p-1"), _row("a-2", "p-1")])
            with self.assertRaisesRegex(ValueError, "duplicate generation pair"):
                _read_generation(path)

    def test_nonterminal_rows_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation.jsonl"
            row = _row("a-1", "p-1")
            row["status"] = "submitted"
            self._write(path, [row])
            with self.assertRaisesRegex(ValueError, "not terminal"):
                _read_generation(path)


if __name__ == "__main__":
    unittest.main()
