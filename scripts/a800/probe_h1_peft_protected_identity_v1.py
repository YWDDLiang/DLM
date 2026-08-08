#!/usr/bin/env python3
"""No-forward A800 probe for protected-P0 dual-adapter identity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM

from crystal_dlm.h1_llm_planner import (
    disable_peft_bnb_autodetect,
    ensure_peft_cache_compat,
    load_llama3_compatible_config,
)
from crystal_dlm.peft_adapter_identity import (
    PROTECTED_P0_ADAPTER_CONFIG_SHA256,
    PROTECTED_P0_ADAPTER_WEIGHT_SHA256,
    adapter_pair_identity_report,
    adapter_source_identity_report,
    copy_adapter_state_exact,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--p0-adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-inventory-sha256", required=True)
    parser.add_argument(
        "--expected-p0-adapter-weight-sha256",
        default=PROTECTED_P0_ADAPTER_WEIGHT_SHA256,
    )
    parser.add_argument(
        "--expected-p0-adapter-config-sha256",
        default=PROTECTED_P0_ADAPTER_CONFIG_SHA256,
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("identity probe requires exactly one visible CUDA device")
    if "A800" not in torch.cuda.get_device_name(0) or not torch.cuda.is_bf16_supported():
        raise RuntimeError("identity probe requires one BF16 A800")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError(f"immutable probe output already exists: {args.output}")

    config = load_llama3_compatible_config(args.model_path)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False
    ensure_peft_cache_compat()
    disable_peft_bnb_autodetect()
    from peft import PeftModel
    import peft
    import transformers

    model = PeftModel.from_pretrained(
        base,
        args.p0_adapter_path,
        adapter_name="candidate",
        is_trainable=True,
        autocast_adapter_dtype=True,
    )
    candidate_after_load = adapter_source_identity_report(
        model,
        "candidate",
        args.p0_adapter_path,
        expected_weight_sha256=args.expected_p0_adapter_weight_sha256,
        expected_config_sha256=args.expected_p0_adapter_config_sha256,
    )
    if not candidate_after_load["passed"]:
        raise RuntimeError("candidate differs from protected source immediately after load")

    model.load_adapter(
        args.p0_adapter_path,
        adapter_name="reference",
        is_trainable=True,
        autocast_adapter_dtype=True,
    )
    pair_before_copy = adapter_pair_identity_report(
        model,
        expected_active_adapter="candidate",
    )
    copy_report = copy_adapter_state_exact(model)
    model.set_adapter("candidate")
    pair_after_copy_cpu = adapter_pair_identity_report(
        model,
        expected_active_adapter="candidate",
    )

    model.to(torch.device("cuda"))
    pair_after_device = adapter_pair_identity_report(
        model,
        expected_active_adapter="candidate",
    )
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()
    pair_after_setup = adapter_pair_identity_report(
        model,
        expected_active_adapter="candidate",
    )
    candidate_after_setup = adapter_source_identity_report(
        model,
        "candidate",
        args.p0_adapter_path,
        expected_weight_sha256=args.expected_p0_adapter_weight_sha256,
        expected_config_sha256=args.expected_p0_adapter_config_sha256,
    )
    reference_after_setup = adapter_source_identity_report(
        model,
        "reference",
        args.p0_adapter_path,
        expected_weight_sha256=args.expected_p0_adapter_weight_sha256,
        expected_config_sha256=args.expected_p0_adapter_config_sha256,
    )

    expected_rounding_signature = bool(
        not pair_before_copy["passed"]
        and pair_before_copy["candidate_tensor_count"]
        == pair_before_copy["reference_tensor_count"]
        and pair_before_copy["mismatched_count"] > 0
        and pair_before_copy["max_abs_diff"] > 0.0
    )
    passed = bool(
        candidate_after_load["passed"]
        and expected_rounding_signature
        and copy_report["passed"]
        and pair_after_copy_cpu["passed"]
        and pair_after_device["passed"]
        and pair_after_setup["passed"]
        and candidate_after_setup["passed"]
        and reference_after_setup["passed"]
    )
    report = {
        "schema": "h1_peft_protected_p0_real_adapter_probe_v1",
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "peft": peft.__version__,
            "transformers": transformers.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "model_path": str(args.model_path),
        "p0_adapter_path": str(args.p0_adapter_path),
        "source_inventory_sha256": str(args.expected_source_inventory_sha256),
        "candidate_after_load": candidate_after_load,
        "pair_before_copy": pair_before_copy,
        "expected_rounding_signature": expected_rounding_signature,
        "copy_report": copy_report,
        "pair_after_copy_cpu": pair_after_copy_cpu,
        "pair_after_device": pair_after_device,
        "pair_after_setup": pair_after_setup,
        "candidate_after_setup": candidate_after_setup,
        "reference_after_setup": reference_after_setup,
        "forward_performed": False,
        "optimizer_constructed": False,
        "training_performed": False,
        "smact4_executed": False,
        "passed": passed,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass" if passed else "fail", **report}, indent=2))
    if not passed:
        raise RuntimeError("protected-P0 identity probe failed")


if __name__ == "__main__":
    main()
