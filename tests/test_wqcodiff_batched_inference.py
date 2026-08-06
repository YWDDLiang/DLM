from __future__ import annotations

import importlib.util
import dataclasses
import os
import random
import unittest

import numpy as np

from crystal_dlm.wqcodiff.bridge import ChartCatalog, ChartSpec
from crystal_dlm.wqcodiff.charts import LatticeChartCodec
from crystal_dlm.wqcodiff.state import OrbitState, StratifiedState


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
SPGLIB_AVAILABLE = importlib.util.find_spec("spglib") is not None
RUN_SLOW = os.environ.get("WQCODIFF_RUN_SLOW_TESTS") == "1"


class _P1Catalog(ChartCatalog):
    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        if (space_group, wyckoff_type) != (1, 0):
            raise KeyError((space_group, wyckoff_type))
        return ChartSpec(1, 0, "a", 1, 3)

    def types(self, space_group: int):
        if space_group != 1:
            raise KeyError(space_group)
        return (0,)

    def expand(self, space_group: int, wyckoff_type: int, free_coordinate):
        self.get(space_group, wyckoff_type)
        return (tuple(float(value) % 1.0 for value in free_coordinate),)


@unittest.skipUnless(
    TORCH_AVAILABLE and SPGLIB_AVAILABLE and RUN_SLOW,
    "set WQCODIFF_RUN_SLOW_TESTS=1 for the full batched reverse-process smoke",
)
class BatchedInferenceTests(unittest.TestCase):
    def _context(self, attempt_id: str, seed: int, *, prior: bool):
        import torch

        from crystal_dlm.wqcodiff.sampling import _AttemptContext

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        calls = {"joint": 0, "bridge": 0, "projection": 0}
        if prior:
            calls["prior"] = 0
        return _AttemptContext(
            attempt_id,
            random.Random(seed),
            generator,
            calls,
            [],
        )

    def _state(self, attempt_id: str, species: int = 6) -> StratifiedState:
        return StratifiedState(
            space_group=1,
            lattice_system="triclinic",
            lattice_chart=LatticeChartCodec.encode_matrix(
                np.eye(3) * 5.0, "triclinic"
            ),
            orbits=(
                OrbitState(
                    "o0",
                    0,
                    species,
                    1,
                    3,
                    (0.2, 0.3, 0.4),
                ),
            ),
            attempt_id=attempt_id,
            timestep=0.7,
        )

    def test_unconditional_batch_accounts_every_attempt_terminally(self) -> None:
        import torch

        from crystal_dlm.wqcodiff.model import WQCoDenoiser, WQVariant
        from crystal_dlm.wqcodiff.sampling import SamplingConfig, _sample_batch

        torch.manual_seed(3)
        model = WQCoDenoiser().eval()
        contexts = (
            self._context("a-batch-0", 101, prior=True),
            self._context("a-batch-1", 202, prior=True),
        )
        config = SamplingConfig(
            checkpoint="unused.pt",
            output_jsonl="unused.jsonl",
            attempt_ledger="unused-ledger.jsonl",
            experiment_id="unit-batch",
            variant=WQVariant.ATOM_JOINT,
            training_seed=11,
            sampling_seed=101,
            attempts=2,
            backbone_calls=16,
            revision_control="none",
            inference_batch_size=16,
            device="cpu",
        )
        works = _sample_batch(model, _P1Catalog(), contexts, config, torch.device("cpu"))
        self.assertEqual(len(works), 2)
        for work in works:
            self.assertNotEqual(work.artifact is None, work.error is None)
            if work.artifact is not None:
                self.assertEqual(work.context.calls["joint"], 16)

    def test_recovery_batch_keeps_per_attempt_mechanism_accounting(self) -> None:
        import torch

        from crystal_dlm.wqcodiff.model import WQCoDenoiser, WQVariant
        from crystal_dlm.wqcodiff.recovery import (
            RecoveryConfig,
            _RecoveryWork,
            _run_recovery_batch,
        )

        torch.manual_seed(5)
        model = WQCoDenoiser().eval()
        works = []
        for index, seed in enumerate((303, 404)):
            source = self._state(f"r-batch-{index}", species=6)
            corrupt = self._state(f"r-batch-{index}", species=8)
            work = _RecoveryWork(
                context=self._context(f"r-batch-{index}", seed, prior=False)
            )
            work.initialize(source, corrupt)
            works.append(work)
        config = RecoveryConfig(
            checkpoint="unused.pt",
            dataset_paths=("unused.jsonl",),
            output_jsonl="unused-output.jsonl",
            attempt_ledger="unused-ledger.jsonl",
            experiment_id="unit-recovery-batch",
            runtime_source_bundle_sha256="a" * 64,
            variant=WQVariant.DLM_MONO,
            training_seed=11,
            corruption_seed=101,
            structures=2,
            corruption_level=0.7,
            operator="wrong-species",
            geometry_condition="clean",
            schedule="fixed",
            calls=16,
            inference_batch_size=16,
            device="cpu",
        )
        completed = _run_recovery_batch(
            works, model, _P1Catalog(), config, torch.device("cpu")
        )
        self.assertEqual(len(completed), 2)
        for work in completed:
            self.assertIsNone(work.error)
            self.assertIsNotNone(work.mechanism)
            self.assertEqual(work.context.calls["joint"], 16)

    def test_ordered_runtime_workers_are_attempt_exact(self) -> None:
        import torch

        from crystal_dlm.wqcodiff.model import WQCoDenoiser, WQVariant
        from crystal_dlm.wqcodiff.recovery import (
            RecoveryConfig,
            _RecoveryWork,
            _run_recovery_batch,
        )

        torch.manual_seed(17)
        model = WQCoDenoiser().eval()

        def make_works():
            result = []
            for index, seed in enumerate((303, 404, 505, 606)):
                source = self._state(f"parallel-{index}", species=6)
                corrupt = self._state(f"parallel-{index}", species=8)
                work = _RecoveryWork(
                    context=self._context(f"parallel-{index}", seed, prior=False)
                )
                work.initialize(source, corrupt)
                result.append(work)
            return result

        sequential_config = RecoveryConfig(
            checkpoint="unused.pt",
            dataset_paths=("unused.jsonl",),
            output_jsonl="unused-output.jsonl",
            attempt_ledger="unused-ledger.jsonl",
            experiment_id="unit-runtime-workers",
            runtime_source_bundle_sha256="b" * 64,
            variant=WQVariant.D3PM,
            training_seed=11,
            corruption_seed=101,
            structures=4,
            corruption_level=0.9,
            operator="wrong-species",
            geometry_condition="clean",
            schedule="fixed",
            calls=16,
            inference_batch_size=16,
            runtime_workers=1,
            device="cpu",
        )
        sequential = _run_recovery_batch(
            make_works(), model, _P1Catalog(), sequential_config, torch.device("cpu")
        )
        parallel = _run_recovery_batch(
            make_works(),
            model,
            _P1Catalog(),
            dataclasses.replace(sequential_config, runtime_workers=2),
            torch.device("cpu"),
        )

        for old, new in zip(sequential, parallel):
            self.assertEqual(type(old.error), type(new.error))
            self.assertEqual(old.context.calls, new.context.calls)
            self.assertEqual(old.context.trace, new.context.trace)
            self.assertEqual(old.mechanism, new.mechanism)
            self.assertEqual(old.state, new.state)


if __name__ == "__main__":
    unittest.main()
