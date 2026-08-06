#!/usr/bin/env python3
"""Gate-B answer-only NLL comparison against the frozen Llama backbone."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path


def _require_runtime() -> int:
    values = []
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        value = int(os.environ.get(name, "0"))
        if value not in (4, 8, 16):
            raise RuntimeError(f"{name} must be one of 4, 8, or 16")
        values.append(value)
    if len(set(values)) != 1:
        raise RuntimeError("NLL numerical thread settings must agree")
    if values[0] > int(os.environ.get("SLURM_CPUS_PER_TASK", "0")):
        raise RuntimeError("NLL threads exceed allocated Slurm CPUs")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be exactly 1")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("LoRA NLL evaluation must run through Slurm")
    return values[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--gate-a-lock", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--token-audit", type=Path, required=True)
    parser.add_argument("--representation", choices=("atom", "wyckoff"), required=True)
    parser.add_argument("--training-stage", choices=("coarse", "mixed_edit"), required=True)
    parser.add_argument("--adapter-training-stage", choices=("coarse", "mixed_edit"))
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--execution-patch-sha256", required=True)
    parser.add_argument("--adapter-training-execution-patch-sha256")
    args = parser.parse_args()
    runtime_threads = _require_runtime()

    import torch
    import torch.nn.functional as functional
    from peft import PeftModel
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from crystal_dlm.wqcodiff.contracts import write_json_exclusive
    from crystal_dlm.wqcodiff.crysllmgen.gate import GateALock
    from crystal_dlm.wqcodiff.crysllmgen.lora import (
        FixedLengthSFTCollator,
        JsonlSFTDataset,
        validate_sft_artifacts,
        validate_trained_adapter,
    )
    from crystal_dlm.wqcodiff.crysllmgen.protocol import load_protocol_v4

    torch.set_num_threads(runtime_threads)
    try:
        torch.set_num_interop_threads(min(4, runtime_threads))
    except RuntimeError:
        pass
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA NLL evaluation requires CUDA")
    protocol = load_protocol_v4(args.protocol.resolve())
    adapter_training_stage = args.adapter_training_stage or args.training_stage
    adapter_training_execution_patch_sha256 = (
        args.adapter_training_execution_patch_sha256
        or args.execution_patch_sha256
    )
    project_root = args.protocol.resolve().parents[3]
    gate = GateALock.load(
        args.gate_a_lock.resolve(),
        project_root=project_root,
        protocol_path=args.protocol.resolve(),
        execution_patch_manifest_sha256=args.execution_patch_sha256,
    )
    max_length = int(protocol.data["llama_training"]["max_sequence_length"])
    data_seed = args.training_seed if args.data_seed is None else args.data_seed
    artifact = validate_sft_artifacts(
        data_path=args.data.resolve(),
        manifest_path=args.data_manifest.resolve(),
        token_audit_path=args.token_audit.resolve(),
        representation=args.representation,
        training_seed=data_seed,
        max_length=max_length,
        dataset_stage=args.training_stage,
    )
    adapter_identity = validate_trained_adapter(
        adapter_root=args.adapter.resolve(),
        gate_a_lock_sha256=gate.sha256,
        source_bundle_sha256=gate.source_bundle_sha256,
        representation=args.representation,
        training_stage=adapter_training_stage,
        training_seed=args.training_seed,
        execution_patch_sha256=adapter_training_execution_patch_sha256,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.llama_root.resolve(),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
        model_max_length=max_length,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = JsonlSFTDataset(args.data, tokenizer, max_length=max_length)
    collator = FixedLengthSFTCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=max_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        drop_last=False,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.llama_root.resolve(),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(
        base,
        args.adapter.resolve(),
        local_files_only=True,
        is_trainable=False,
    )
    model.eval()
    model.config.use_cache = False

    def evaluate(*, disable_adapter: bool) -> tuple[float, int, int]:
        total_loss = 0.0
        total_tokens = 0
        examples = 0
        context = model.disable_adapter() if disable_adapter else __import__(
            "contextlib"
        ).nullcontext()
        with context, torch.inference_mode():
            for batch in loader:
                input_ids = batch["input_ids"].cuda(non_blocking=True)
                attention = batch["attention_mask"].cuda(non_blocking=True)
                labels = batch["labels"].cuda(non_blocking=True)
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention,
                ).logits
                shifted_logits = logits[:, :-1].float().contiguous()
                shifted_labels = labels[:, 1:].contiguous()
                loss = functional.cross_entropy(
                    shifted_logits.view(-1, shifted_logits.shape[-1]),
                    shifted_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                tokens = int((shifted_labels != -100).sum())
                if tokens <= 0 or not bool(torch.isfinite(loss)):
                    raise RuntimeError("invalid answer-only validation loss")
                total_loss += float(loss)
                total_tokens += tokens
                examples += int(input_ids.shape[0])
        if examples != len(dataset):
            raise RuntimeError("validation NLL example denominator changed")
        return total_loss / total_tokens, total_tokens, examples

    started = time.monotonic()
    frozen_nll, frozen_tokens, frozen_examples = evaluate(disable_adapter=True)
    adapter_nll, adapter_tokens, adapter_examples = evaluate(disable_adapter=False)
    if (frozen_tokens, frozen_examples) != (adapter_tokens, adapter_examples):
        raise RuntimeError("base/adapter validation denominators differ")
    report = {
        "schema": "crysllmgen_lora_validation_nll_v1",
        "ok": math.isfinite(adapter_nll) and adapter_nll < frozen_nll,
        "representation": args.representation,
        "training_stage": args.training_stage,
        "adapter_training_stage": adapter_training_stage,
        "training_seed": args.training_seed,
        "validation_data_seed": data_seed,
        "examples": adapter_examples,
        "answer_tokens": adapter_tokens,
        "frozen_backbone_nll": frozen_nll,
        "adapter_nll": adapter_nll,
        "nll_improvement": frozen_nll - adapter_nll,
        "perplexity_ratio_adapter_over_frozen": math.exp(
            min(50.0, adapter_nll - frozen_nll)
        ),
        "data": artifact,
        "adapter": adapter_identity,
        "protocol_sha256": protocol.sha256,
        "gate_a_lock_sha256": gate.sha256,
        "adapter_training_execution_patch_sha256": (
            adapter_training_execution_patch_sha256
        ),
        "evaluation_execution_patch_sha256": args.execution_patch_sha256,
        "walltime_s": time.monotonic() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(0),
        "threads": runtime_threads,
        "retry_or_replacement_used": False,
    }
    write_json_exclusive(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
