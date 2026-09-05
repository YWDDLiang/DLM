from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.manifold_repair_head import ManifoldRepairConfig, ManifoldRepairHead
from crystal_dlm.pmtr_checkpoint import (
    PMTR_CHECKPOINT_SCHEMA,
    load_pmtr_checkpoint,
    save_pmtr_checkpoint,
)
from crystal_dlm.pmtr_runtime import PMTRRuntimeConfig


class FakeTokenizer:
    def __init__(self):
        tokens = ["<N_001>", "<E_Li>"]
        for axis in "ABC":
            tokens.extend(f"<L{axis}_{value:03d}>" for value in (20, 40, 60))
        for axis in "ABG":
            tokens.extend(f"<A{axis}_{value:03d}>" for value in (60, 90, 120))
        for axis in "XYZ":
            tokens.extend(f"<{axis}_{value:03d}>" for value in (0, 50, 100))
        self.vocab = {token: index for index, token in enumerate(tokens)}

    def get_vocab(self):
        return dict(self.vocab)


class PMTRCheckpointTest(unittest.TestCase):
    def test_round_trip_restores_strict_frozen_head_and_runtime_config(self):
        config = ManifoldRepairConfig(hidden_size=12, width=16, radial_basis_count=4)
        head = ManifoldRepairHead(config)
        with torch.no_grad():
            head.metric_output.weight.fill_(0.125)
        runtime = PMTRRuntimeConfig(transport_gain=4.5, image_radius=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pmtr_final.pt"
            save_pmtr_checkpoint(path, repair_head=head, runtime_config=runtime)
            loaded = load_pmtr_checkpoint(
                path,
                tokenizer=FakeTokenizer(),
                device="cpu",
                dtype=torch.float32,
                expected_hidden_size=12,
            )
        self.assertEqual(loaded.metadata["schema"], PMTR_CHECKPOINT_SCHEMA)
        self.assertEqual(loaded.metadata["runtime_config"]["transport_gain"], 4.5)
        loaded_head = loaded.transform.repair_head
        self.assertFalse(loaded_head.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in loaded_head.parameters()))
        self.assertTrue(
            torch.equal(
                loaded_head.metric_output.weight,
                torch.full_like(loaded_head.metric_output.weight, 0.125),
            )
        )

    def test_hidden_size_and_schema_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "pmtr.pt"
            save_pmtr_checkpoint(
                path,
                repair_head=ManifoldRepairHead(
                    ManifoldRepairConfig(hidden_size=8, width=8, radial_basis_count=4)
                ),
            )
            with self.assertRaisesRegex(ValueError, "hidden size"):
                load_pmtr_checkpoint(
                    path,
                    tokenizer=FakeTokenizer(),
                    device="cpu",
                    dtype=torch.float32,
                    expected_hidden_size=9,
                )
            payload = torch.load(path, weights_only=True)
            payload["schema"] = "wrong"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "schema"):
                load_pmtr_checkpoint(
                    path,
                    tokenizer=FakeTokenizer(),
                    device="cpu",
                    dtype=torch.float32,
                    expected_hidden_size=8,
                )


if __name__ == "__main__":
    unittest.main()
