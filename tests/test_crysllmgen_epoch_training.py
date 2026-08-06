from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.crysllmgen.epoch_training import (
    audit_epoch_checkpoints,
    expected_epoch_steps,
    load_mixed_edit_epoch_contract,
    updates_per_effective_epoch,
)
from crystal_dlm.wqcodiff.crysllmgen.protocol import load_protocol_v4


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiments/wyckoff_codiffusion/protocol_v4.yaml"
CONTRACT = (
    ROOT
    / "configs/experiments/wyckoff_codiffusion/mixed_edit_three_epoch_v1.json"
)


class CrysLLMGenMixedEditEpochTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol_v4(PROTOCOL)

    def test_contract_locks_short_warmup_without_decay(self) -> None:
        contract = load_mixed_edit_epoch_contract(
            CONTRACT,
            base_protocol_name=self.protocol.name,
            base_protocol_sha256=self.protocol.sha256,
        )
        self.assertEqual(contract.maximum_effective_epochs, 3)
        self.assertEqual(contract.effective_batch_sequences, 64)
        self.assertEqual(contract.world_size, 2)
        self.assertEqual(contract.per_device_microbatch, 8)
        self.assertEqual(contract.gradient_accumulation, 4)
        self.assertEqual(
            contract.world_size
            * contract.per_device_microbatch
            * contract.gradient_accumulation,
            contract.effective_batch_sequences,
        )
        self.assertEqual(contract.required_checkpoint_epochs, (1, 2, 3))
        self.assertEqual(contract.learning_rate, 1.0e-4)
        self.assertEqual(contract.warmup_fraction, 0.03)
        self.assertEqual(contract.scheduler, "constant_with_warmup")
        self.assertFalse(contract.gradient_checkpointing)
        self.assertIsNone(contract.gradient_checkpointing_use_reentrant)
        self.assertEqual(contract.data_mode, "pretokenized_memmap")
        self.assertEqual(contract.dataloader_num_workers_per_rank, 4)
        self.assertEqual(
            contract.allowed_attention_implementations,
            ("sdpa", "flash_attention_2"),
        )

    def test_current_denominator_has_exact_epoch_boundaries(self) -> None:
        self.assertEqual(
            updates_per_effective_epoch(examples=54_270, effective_batch=64),
            848,
        )
        self.assertEqual(
            expected_epoch_steps(examples=54_270, effective_batch=64, epochs=3),
            (848, 1696, 2544),
        )

    def test_three_checkpoint_set_is_audited_by_bytes_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for epoch, step in enumerate((848, 1696, 2544), start=1):
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps({"global_step": step, "epoch": float(epoch)}),
                    encoding="utf-8",
                )
                (checkpoint / "adapter_model.safetensors").write_bytes(
                    f"epoch-{epoch}".encode("ascii")
                )
                (checkpoint / "adapter_config.json").write_text(
                    json.dumps({"epoch": epoch}), encoding="utf-8"
                )
            audited = audit_epoch_checkpoints(
                root,
                examples=54_270,
                effective_batch=64,
            )
            self.assertEqual(
                [item["global_step"] for item in audited],
                [848, 1696, 2544],
            )
            self.assertTrue(all(len(item["adapter_model_sha256"]) == 64 for item in audited))

            (root / "checkpoint-1696" / "adapter_model.safetensors").unlink()
            with self.assertRaises(FileNotFoundError):
                audit_epoch_checkpoints(
                    root,
                    examples=54_270,
                    effective_batch=64,
                )

    def test_cosine_or_base_protocol_drift_is_rejected(self) -> None:
        payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
        changed = copy.deepcopy(payload)
        changed["training"]["optimization"]["scheduler"] = "cosine"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint contract"):
                load_mixed_edit_epoch_contract(
                    path,
                    base_protocol_name=self.protocol.name,
                    base_protocol_sha256=self.protocol.sha256,
                )
        with self.assertRaisesRegex(ValueError, "base protocol"):
            load_mixed_edit_epoch_contract(
                CONTRACT,
                base_protocol_name=self.protocol.name,
                base_protocol_sha256="0" * 64,
            )

    def test_world_size_or_accumulation_drift_is_rejected(self) -> None:
        payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
        for key, value in (("world_size", 1), ("gradient_accumulation", 16)):
            changed = copy.deepcopy(payload)
            changed["training"]["distributed_batch"][key] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"changed-{key}.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "checkpoint contract"):
                    load_mixed_edit_epoch_contract(
                        path,
                        base_protocol_name=self.protocol.name,
                        base_protocol_sha256=self.protocol.sha256,
                    )


if __name__ == "__main__":
    unittest.main()
