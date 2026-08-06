from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.cli import _recovery_config_kwargs


class RecoveryCliContractTests(unittest.TestCase):
    def test_runtime_source_bundle_sha_reaches_recovery_config(self) -> None:
        source_sha = "a" * 64
        variant = object()
        args = argparse.Namespace(
            checkpoint=Path("checkpoint.pt"),
            dataset=[Path("validation.jsonl")],
            output=Path("recovery.jsonl"),
            ledger=Path("attempts.jsonl"),
            experiment_id="cli-regression",
            pairing_id="matched-pairing",
            runtime_source_bundle_sha256=source_sha,
            training_seed=11,
            corruption_seed=202,
            structures=160,
            corruption_level=0.9,
            operator="joint",
            geometry_condition="noisy",
            schedule="geometry-adaptive",
            control="none",
            calls=16,
            revision_threshold=0.6,
            temperature=1.0,
            inference_batch_size=64,
            runtime_workers=12,
            device="cuda",
        )

        kwargs = _recovery_config_kwargs(args, variant=variant)

        self.assertEqual(kwargs["runtime_source_bundle_sha256"], source_sha)
        self.assertIs(kwargs["variant"], variant)
        self.assertEqual(kwargs["dataset_paths"], ("validation.jsonl",))
        self.assertEqual(kwargs["checkpoint"], "checkpoint.pt")
        self.assertEqual(kwargs["runtime_workers"], 12)


if __name__ == "__main__":
    unittest.main()
