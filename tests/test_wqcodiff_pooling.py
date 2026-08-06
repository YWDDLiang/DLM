from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.pooling import (
    GenerationPoolConfig,
    parse_seed_count,
    pool_generation_artifacts,
)


def _row(
    attempt: str,
    pair: str,
    *,
    training_seed: int,
    sampling_seed: int,
    ordinal: int,
    status: str = "succeeded",
) -> dict[str, object]:
    return {
        "schema": "wqcodiff_generation_attempt_v1",
        "attempt_id": attempt,
        "pair_id": pair,
        "method": "M-WQ-STRAT-GEO",
        "training_seed": training_seed,
        "sampling_seed": sampling_seed,
        "ordinal": ordinal,
        "status": status,
        "checkpoint_sha256": str(training_seed) * 64,
        "source_bundle_sha256": "a" * 64,
        "revision_lock_sha256": "b" * 64,
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class GenerationPoolingTests(unittest.TestCase):
    def test_pool_is_deterministic_and_locks_seed_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "seed-23.jsonl"
            second = root / "seed-11.jsonl"
            _write(
                first,
                [
                    _row(
                        "a-23-1",
                        "p-23-1",
                        training_seed=23,
                        sampling_seed=202,
                        ordinal=1,
                        status="projection_failure",
                    ),
                    _row(
                        "a-23-0",
                        "p-23-0",
                        training_seed=23,
                        sampling_seed=202,
                        ordinal=0,
                    ),
                ],
            )
            _write(
                second,
                [
                    _row(
                        "a-11-0",
                        "p-11-0",
                        training_seed=11,
                        sampling_seed=101,
                        ordinal=0,
                    )
                ],
            )
            output = root / "pooled.jsonl"
            manifest = root / "pooled.manifest.json"
            result = pool_generation_artifacts(
                GenerationPoolConfig(
                    inputs=(str(first), str(second)),
                    output_jsonl=str(output),
                    manifest_json=str(manifest),
                    expected_method="M-WQ-STRAT-GEO",
                    expected_total=3,
                    expected_training_seed_counts=((11, 1), (23, 2)),
                )
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(
                [row["attempt_id"] for row in rows],
                ["a-11-0", "a-23-0", "a-23-1"],
            )
            self.assertEqual(result["records"], 3)
            self.assertEqual(result["training_seed_counts"], {"11": 1, "23": 2})
            self.assertEqual(
                result["status_counts"],
                {"projection_failure": 1, "succeeded": 2},
            )
            self.assertEqual(
                result,
                json.loads(manifest.read_text(encoding="utf-8")),
            )

    def test_duplicate_pair_is_rejected_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            _write(
                first,
                [_row("a-1", "p-shared", training_seed=11, sampling_seed=101, ordinal=0)],
            )
            _write(
                second,
                [_row("a-2", "p-shared", training_seed=23, sampling_seed=202, ordinal=0)],
            )
            output = root / "pooled.jsonl"
            manifest = root / "pooled.manifest.json"
            with self.assertRaisesRegex(ValueError, "duplicate pooled pair_id"):
                pool_generation_artifacts(
                    GenerationPoolConfig(
                        inputs=(str(first), str(second)),
                        output_jsonl=str(output),
                        manifest_json=str(manifest),
                    )
                )
            self.assertFalse(output.exists())
            self.assertFalse(manifest.exists())

    def test_nonterminal_and_count_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            _write(
                source,
                [_row("a-1", "p-1", training_seed=11, sampling_seed=101, ordinal=0)],
            )
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                pool_generation_artifacts(
                    GenerationPoolConfig(
                        inputs=(str(source),),
                        output_jsonl=str(root / "wrong.jsonl"),
                        manifest_json=str(root / "wrong.manifest.json"),
                        expected_total=2,
                    )
                )
            _write(
                source,
                [
                    _row(
                        "a-1",
                        "p-1",
                        training_seed=11,
                        sampling_seed=101,
                        ordinal=0,
                        status="submitted",
                    )
                ],
            )
            with self.assertRaisesRegex(ValueError, "not terminal"):
                pool_generation_artifacts(
                    GenerationPoolConfig(
                        inputs=(str(source),),
                        output_jsonl=str(root / "terminal.jsonl"),
                        manifest_json=str(root / "terminal.manifest.json"),
                    )
                )

    def test_seed_count_parser(self) -> None:
        self.assertEqual(parse_seed_count("47=3333"), (47, 3333))
        with self.assertRaises(ValueError):
            parse_seed_count("47")


if __name__ == "__main__":
    unittest.main()
