from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from crystal_dlm.peft_adapter_identity import (
    adapter_pair_identity_report,
    adapter_source_identity_report,
    copy_adapter_state_exact,
)


class _FakeAdapterModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(2, 2, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora_A = torch.nn.ModuleDict(
            {
                "candidate": torch.nn.Linear(2, 2, bias=False),
                "reference": torch.nn.Linear(2, 2, bias=False),
            }
        )
        self.lora_A.reference.weight.requires_grad_(False)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PeftAdapterIdentityTest(unittest.TestCase):
    def test_exact_copy_uses_distinct_storage_and_freezes_reference(self) -> None:
        model = _FakeAdapterModel()
        with torch.no_grad():
            model.lora_A.candidate.weight.copy_(
                torch.tensor([[0.1234567, -0.2345678], [0.3456789, -0.4567891]])
            )
            rounded = model.lora_A.candidate.weight.bfloat16().float()
            model.lora_A.reference.weight.copy_(rounded)
        report = copy_adapter_state_exact(model)
        self.assertTrue(report["passed"])
        self.assertGreater(report["pre_copy_mismatched_count"], 0)
        pair = adapter_pair_identity_report(model, expected_active_adapter=None)
        self.assertTrue(pair["passed"])
        self.assertEqual(pair["storage_overlap_count"], 0)
        reference_before = model.lora_A.reference.weight.detach().clone()
        with torch.no_grad():
            model.lora_A.candidate.weight.add_(1.0)
        self.assertTrue(torch.equal(model.lora_A.reference.weight, reference_before))
        self.assertFalse(model.lora_A.reference.weight.requires_grad)

    def test_pair_gate_rejects_nonfinite_and_dtype_mismatch(self) -> None:
        model = _FakeAdapterModel()
        with torch.no_grad():
            model.lora_A.reference.weight.copy_(model.lora_A.candidate.weight)
            model.lora_A.candidate.weight[0, 0] = float("nan")
            model.lora_A.reference.weight[0, 0] = float("nan")
        report = adapter_pair_identity_report(model, expected_active_adapter=None)
        self.assertFalse(report["passed"])
        self.assertEqual(report["nonfinite_count"], 1)

        model = _FakeAdapterModel().to(dtype=torch.bfloat16)
        model.lora_A.candidate.weight.requires_grad_(True)
        model.lora_A.reference.weight.requires_grad_(False)
        report = adapter_pair_identity_report(model, expected_active_adapter=None)
        self.assertFalse(report["passed"])
        self.assertGreater(report["dtype_mismatch_count"], 0)

    def test_pair_gate_rejects_aliasing(self) -> None:
        model = _FakeAdapterModel()
        model.lora_A.reference.weight = model.lora_A.candidate.weight
        report = adapter_pair_identity_report(model, expected_active_adapter=None)
        self.assertFalse(report["passed"])
        self.assertEqual(report["storage_overlap_count"], 1)

    def test_source_attestation_rejects_jointly_rounded_pair(self) -> None:
        model = _FakeAdapterModel()
        source = torch.tensor(
            [[0.1234567, -0.2345678], [0.3456789, -0.4567891]],
            dtype=torch.float32,
        )
        rounded = source.bfloat16().float()
        with torch.no_grad():
            model.lora_A.candidate.weight.copy_(rounded)
            model.lora_A.reference.weight.copy_(rounded)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_file({"lora_A.weight": source}, root / "adapter_model.safetensors")
            (root / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA"}) + "\n",
                encoding="utf-8",
            )
            report = adapter_source_identity_report(
                model,
                "candidate",
                root,
                expected_weight_sha256=_sha(root / "adapter_model.safetensors"),
                expected_config_sha256=_sha(root / "adapter_config.json"),
            )
        self.assertFalse(report["passed"])
        self.assertEqual(report["mismatched_count"], 1)


if __name__ == "__main__":
    unittest.main()
