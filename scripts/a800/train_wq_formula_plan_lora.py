#!/usr/bin/env python3
"""Short continuation LoRA for chemistry-first WQ formula planning."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crystal_dlm.wqcodiff.contracts import write_json_exclusive  # noqa: E402
from crystal_dlm.wqcodiff.crysllmgen.lora import (  # noqa: E402
    FixedLengthSFTCollator,
    JsonlSFTDataset,
    sha256_file,
)


IDENTITY = "wq_formula_plan_sft_pilot_v1"
REPORT_SCHEMA = "wq_formula_plan_sft_training_report_v1"


def _require_sha(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _load_data(
    data_dir: Path,
    *,
    expected_source_sha256: str,
    expected_identity: str,
    expected_data_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_token_audit_sha256: str | None = None,
) -> dict[str, Any]:
    data = data_dir / "formula_plan_sft.jsonl"
    manifest_path = data_dir / "manifest.json"
    token_audit_path = data_dir / "token_audit.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(token_audit_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "crysllmgen_formula_plan_sft_manifest_v1"
        or manifest.get("identity") != expected_identity
        or manifest.get("source_sha256") != expected_source_sha256
        or manifest.get("heldout_metrics_or_generations_used") is not False
        or sha256_file(data) != manifest.get("jsonl_sha256")
    ):
        raise ValueError("formula-plan training data identity changed")
    stages = manifest.get("stage_counts") or {}
    planned = int(stages.get("formula_plan", -1))
    if (
        not 20_000 <= planned <= 27_135
        or int(stages.get("formula_conditioned_body", -1)) != planned
        or int(stages.get("direct_edit_replay", -1)) != (planned + 1) // 2
        or int(manifest.get("examples", -1)) != sum(int(value) for value in stages.values())
        or manifest.get("chemistry_filter")
        != "training_only_same_reduced_SMACT_Pauling_validity"
    ):
        raise ValueError("formula-plan chemistry-filtered mixture changed")
    if (
        audit.get("schema") != "crysllmgen_formula_plan_sft_token_audit_v1"
        or audit.get("identity") != expected_identity
        or audit.get("ok") is not True
        or audit.get("data_sha256") != manifest["jsonl_sha256"]
        or int(audit.get("examples_tokenized", -1)) != manifest["examples"]
        or int(audit.get("max_length", -1)) != 640
        or int(audit.get("overflow_count", -1)) != 0
    ):
        raise ValueError("formula-plan token audit changed or failed")
    observed = {
        "data": sha256_file(data),
        "manifest": sha256_file(manifest_path),
        "token_audit": sha256_file(token_audit_path),
    }
    expected = {
        "data": expected_data_sha256,
        "manifest": expected_manifest_sha256,
        "token_audit": expected_token_audit_sha256,
    }
    if any(
        expected_value is not None and observed[key] != expected_value
        for key, expected_value in expected.items()
    ):
        raise ValueError("formula-plan prebuilt training-data hashes changed")
    return {
        "data_path": data,
        "data_sha256": observed["data"],
        "manifest_path": manifest_path,
        "manifest_sha256": observed["manifest"],
        "token_audit_path": token_audit_path,
        "token_audit_sha256": observed["token_audit"],
        "manifest": manifest,
        "token_audit": audit,
    }


def _require_runtime() -> tuple[Any, str]:
    import torch

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("formula-plan continuation must run through Slurm")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 8:
        raise RuntimeError("formula-plan continuation requires exactly 8 CPU")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be exactly 1")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("formula-plan continuation requires one visible GPU")
    name = torch.cuda.get_device_name(0)
    if "A800" not in name:
        raise RuntimeError(f"formula-plan continuation requires A800, observed {name}")
    return torch, name


def train(args: argparse.Namespace) -> dict[str, Any]:
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    torch_module, device_name = _require_runtime()
    execution_patch = _require_sha(args.execution_patch_sha256, "execution patch")
    contract_sha = _require_sha(args.contract_sha256, "contract")
    source_sha = _require_sha(args.source_sha256, "source SFT data")
    initial_model = args.initial_adapter / "adapter_model.safetensors"
    initial_config = args.initial_adapter / "adapter_config.json"
    if (
        sha256_file(initial_model) != args.initial_adapter_model_sha256
        or sha256_file(initial_config) != args.initial_adapter_config_sha256
    ):
        raise ValueError("initial epoch-3 adapter identity changed")
    data = _load_data(
        args.data_dir,
        expected_source_sha256=source_sha,
        expected_identity=args.data_identity,
        expected_data_sha256=args.expected_data_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_token_audit_sha256=args.expected_token_audit_sha256,
    )
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.llama_root,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
        model_max_length=640,
        padding_side="right",
    )
    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise RuntimeError("registered tokenizer lacks EOS or chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = JsonlSFTDataset(data["data_path"], tokenizer, max_length=640)
    if len(dataset) != int(data["manifest"]["examples"]):
        raise ValueError("formula-plan indexed dataset denominator changed")
    collator = FixedLengthSFTCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=640,
    )

    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.llama_root,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(
        model,
        args.initial_adapter,
        local_files_only=True,
        is_trainable=True,
    )
    model.config.use_cache = False
    trainable = sum(value.numel() for value in model.parameters() if value.requires_grad)
    if trainable <= 0:
        raise RuntimeError("loaded continuation adapter has no trainable parameters")

    trainer_dir = args.output_dir / "trainer"
    training_args = TrainingArguments(
        output_dir=str(trainer_dir),
        overwrite_output_dir=False,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        warmup_ratio=0.05,
        lr_scheduler_type="constant_with_warmup",
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=False,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        load_best_model_at_end=False,
        report_to=[],
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        optim="adamw_torch",
        seed=11,
        data_seed=11,
        full_determinism=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    result = trainer.train()
    completed = int(trainer.state.global_step)
    train_loss = float(result.metrics.get("train_loss", float("nan")))
    completed_epoch = float(trainer.state.epoch or 0.0)
    if args.max_steps > 0:
        schedule_complete = completed == args.max_steps
    else:
        schedule_complete = (
            completed >= args.minimum_completed_updates
            and completed <= args.maximum_completed_updates
            and completed_epoch >= args.num_train_epochs - 1.0e-6
        )
    if not schedule_complete or not math.isfinite(train_loss):
        raise RuntimeError("formula-plan continuation did not complete its finite schedule")

    adapter_dir = args.output_dir / "adapter_final"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir / "tokenizer_identity")
    trainer.save_state()
    adapter_model = adapter_dir / "adapter_model.safetensors"
    adapter_config = adapter_dir / "adapter_config.json"
    report = {
        "schema": args.report_schema,
        "identity": args.identity,
        "ok": True,
        "training_seed": 11,
        "execution_patch_sha256": execution_patch,
        "contract_sha256": contract_sha,
        "data": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in data.items()
            if key not in {"manifest", "token_audit"}
        },
        "mixture": data["manifest"]["stage_counts"],
        "initial_adapter": {
            "path": str(args.initial_adapter),
            "adapter_model_sha256": sha256_file(initial_model),
            "adapter_config_sha256": sha256_file(initial_config),
        },
        "optimizer": {
            "completed_updates": completed,
            "completed_epochs": completed_epoch,
            "requested_maximum_updates": args.max_steps,
            "requested_epochs": args.num_train_epochs,
            "per_device_microbatch": 8,
            "gradient_accumulation": 8,
            "global_effective_batch": 64,
            "maximum_sequence_length": 640,
            "learning_rate": args.learning_rate,
            "warmup_fraction": 0.05,
            "scheduler": "constant_with_warmup",
            "gradient_checkpointing": False,
            "attention_implementation": "sdpa",
        },
        "metrics": dict(result.metrics),
        "adapter_final": {
            "path": str(adapter_dir),
            "adapter_model_sha256": sha256_file(adapter_model),
            "adapter_config_sha256": sha256_file(adapter_config),
        },
        "runtime": {
            "walltime_s": time.monotonic() - started,
            "gpu_name": device_name,
            "peak_memory_bytes": int(torch_module.cuda.max_memory_allocated(0)),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "offline": True,
        },
        "heldout_metrics_or_generations_used_for_training": False,
        "retry_or_replacement_used": False,
        "automatic_downstream_authorized": False,
    }
    write_json_exclusive(args.output_dir / "training_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--initial-adapter-model-sha256", required=True)
    parser.add_argument("--initial-adapter-config-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-patch-sha256", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--identity", default=IDENTITY)
    parser.add_argument("--report-schema", default=REPORT_SCHEMA)
    parser.add_argument("--data-identity", default=IDENTITY)
    parser.add_argument("--expected-data-sha256")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-token-audit-sha256")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--minimum-completed-updates", type=int, default=200)
    parser.add_argument("--maximum-completed-updates", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    args = parser.parse_args()
    for name in ("data_dir", "llama_root", "initial_adapter", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    report = train(args)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
