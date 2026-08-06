from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/a800/train_crysllmgen_lora.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("train_crysllmgen_lora", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load training script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CrysLLMGenDDPTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_script()

    def test_torchrun_environment_is_strict(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"},
            clear=False,
        ):
            self.assertEqual(self.module._distributed_environment(), (2, 1, 1))
        with mock.patch.dict(
            os.environ,
            {"WORLD_SIZE": "2", "RANK": "2", "LOCAL_RANK": "0"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid torchrun"):
                self.module._distributed_environment()

    def test_two_ranks_require_eight_allocated_cpus(self) -> None:
        environment = {
            "SLURM_JOB_ID": "1",
            "SLURM_CPUS_PER_TASK": "8",
            "OPENBLAS_NUM_THREADS": "4",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            self.assertEqual(
                self.module._require_environment(mixed_edit=True, world_size=2),
                4,
            )
        environment["SLURM_CPUS_PER_TASK"] = "4"
        with mock.patch.dict(os.environ, environment, clear=False):
            with self.assertRaisesRegex(RuntimeError, "allocated Slurm CPUs"):
                self.module._require_environment(mixed_edit=True, world_size=2)

    def test_rank_zero_exclusively_creates_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-output"
            self.module._prepare_distributed_output(output, rank=0)
            self.assertTrue(output.is_dir())
            self.module._prepare_distributed_output(output, rank=1)
            with self.assertRaises(FileExistsError):
                self.module._prepare_distributed_output(output, rank=0)


if __name__ == "__main__":
    unittest.main()
