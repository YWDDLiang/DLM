from __future__ import annotations

import dataclasses
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required")
class DeterministicPrefetchTests(unittest.TestCase):
    @staticmethod
    def _record(index: int):
        from crystal_dlm.wqcodiff.dataset import tolerance_tag

        primary = {
            "state": {
                "space_group": 1,
                "lattice_system": "triclinic",
                "lattice_chart": [1.6, 1.6, 1.6, 0.0, 0.0, 0.0],
            },
            "primitive_lattice_transform": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "orbits": [
                {
                    "orbit": {
                        "orbit_id": f"o{index}",
                        "wyckoff_type": 0,
                        "species": 6,
                        "multiplicity": 1,
                        "primitive_multiplicity": 1,
                        "chart_dimension": 3,
                        "free_coordinate": [0.2, 0.3, 0.4],
                    },
                    "primitive_fractional_coordinates": [[0.2, 0.3, 0.4]],
                    "primitive_chart_jacobians": [
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                    ],
                }
            ],
        }
        return {
            "material_id": f"toy-{index}",
            "selected": True,
            "decompositions": {tolerance_tag(1.0e-2): primary},
        }

    @staticmethod
    def _assert_batch_equal(first, second) -> None:
        import torch

        for field in dataclasses.fields(first.inputs):
            torch.testing.assert_close(
                getattr(first.inputs, field.name),
                getattr(second.inputs, field.name),
                rtol=0.0,
                atol=0.0,
            )
        for left, right in zip(first.targets, second.targets):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        for left, right in zip(first.prior_targets, second.prior_targets):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        if first.metadata != second.metadata:
            raise AssertionError("prefetch metadata differs")

    def test_spawned_prefetch_is_bitwise_equal_and_ordered(self) -> None:
        from crystal_dlm.wqcodiff.model import WQVariant
        from crystal_dlm.wqcodiff.training import EpochSampler
        from crystal_dlm.wqcodiff.training_data import (
            JsonlRecordIndex,
            build_corrupted_batch,
        )
        from crystal_dlm.wqcodiff.training_prefetch import (
            DeterministicCorruptionPrefetcher,
        )

        def seed_for(training_seed: int, update: int, microbatch: int) -> int:
            return training_seed * 10_000 + update * 10 + microbatch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for index in range(8):
                    handle.write(json.dumps(self._record(index)) + "\n")
            dataset = JsonlRecordIndex((path,))
            direct_sampler = EpochSampler(len(dataset), 11)
            direct = []
            for update in range(3):
                indices = direct_sampler.take(2)
                direct.append(
                    build_corrupted_batch(
                        [dataset[index] for index in indices],
                        seed=seed_for(11, update, 0),
                        variant=WQVariant.STRAT_GEO,
                        representation_variant=WQVariant.STRAT_GEO,
                        enable_revision_training=False,
                        mask_discrete_fields=False,
                        enable_topology_corruption=True,
                    )
                )
            sampler = EpochSampler(len(dataset), 11)
            prefetch = DeterministicCorruptionPrefetcher(
                dataset=dataset,
                sampler=sampler,
                training_seed=11,
                total_updates=3,
                microbatch_size=2,
                accumulation_steps=1,
                seed_for=seed_for,
                revision_start_update=3,
                variant=WQVariant.STRAT_GEO,
                representation_variant=WQVariant.STRAT_GEO,
                mask_discrete_fields=False,
                enable_topology_corruption=True,
                workers=1,
                depth=1,
            )
            try:
                for update, expected in enumerate(direct):
                    self._assert_batch_equal(expected, prefetch.take(update, 0).batch)
            finally:
                prefetch.close()
                dataset.close()


if __name__ == "__main__":
    unittest.main()
