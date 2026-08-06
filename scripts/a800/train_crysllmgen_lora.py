#!/usr/bin/env python3
"""Train one registered atom or shared-Wyckoff rank-16 LoRA lane."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

from crystal_dlm.wqcodiff.crysllmgen.epoch_training import (
    audit_epoch_checkpoints,
    load_mixed_edit_epoch_contract,
)
from crystal_dlm.wqcodiff.crysllmgen.formal_execution import (
    load_formal_execution_selection,
)
from crystal_dlm.wqcodiff.crysllmgen.lora import (
    FixedLengthSFTCollator,
    JsonlSFTDataset,
    PretokenizedMemmapSFTDataset,
    sha256_file,
    validate_sft_artifacts,
)
from crystal_dlm.wqcodiff.crysllmgen.performance_profile import (
    load_lora_ddp_profile_matrix,
)
from crystal_dlm.wqcodiff.crysllmgen.gate import GateALock
from crystal_dlm.wqcodiff.crysllmgen.protocol import load_protocol_v4


def _distributed_environment() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0 or not 0 <= rank < world_size or not 0 <= local_rank < world_size:
        raise RuntimeError("invalid torchrun WORLD_SIZE/RANK/LOCAL_RANK environment")
    return world_size, rank, local_rank


def _require_environment(*, mixed_edit: bool, world_size: int) -> int:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("LoRA training must run through Slurm")
    thread_values = []
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        value = int(os.environ.get(name, "0"))
        permitted = (4, 8, 16) if mixed_edit else (1, 4, 8, 16)
        if value not in permitted:
            raise RuntimeError(f"{name} must be one of {permitted}")
        thread_values.append(value)
    if len(set(thread_values)) != 1:
        raise RuntimeError("numerical thread settings must agree")
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    required_threads = thread_values[0] * (world_size if mixed_edit else 1)
    if required_threads > allocated:
        raise RuntimeError(
            "rank-local numerical threads exceed total allocated Slurm CPUs"
        )
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be exactly 1")
    return thread_values[0]


def _prepare_distributed_output(path: Path, *, rank: int) -> None:
    if rank == 0:
        path.mkdir(parents=True, exist_ok=False)
        return
    deadline = time.monotonic() + 60.0
    while not path.is_dir():
        if time.monotonic() >= deadline:
            raise TimeoutError("rank 0 did not create the distributed output directory")
        time.sleep(0.1)


def _write_exclusive(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-amendment", type=Path)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--token-audit", type=Path, required=True)
    parser.add_argument("--representation", choices=("atom", "wyckoff"), required=True)
    parser.add_argument("--training-stage", choices=("coarse", "mixed_edit"), default="coarse")
    parser.add_argument("--initial-adapter", type=Path)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--execution-patch-sha256")
    parser.add_argument("--gate-a-lock", type=Path)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--performance-profile-matrix", type=Path)
    parser.add_argument("--performance-profile-id")
    parser.add_argument("--formal-execution-selection", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--pretokenized-cache", type=Path)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--run-role",
        choices=("smoke", "performance_profile", "main", "canonical_ablation"),
        required=True,
    )
    args = parser.parse_args()
    world_size, rank, local_rank = _distributed_environment()
    runtime_threads = _require_environment(
        mixed_edit=args.training_stage == "mixed_edit",
        world_size=world_size,
    )
    if re.fullmatch(r"[0-9a-f]{64}", args.source_bundle_sha256) is None:
        raise ValueError("source-bundle-sha256 must be one lowercase SHA256")
    if args.execution_patch_sha256 is not None and re.fullmatch(
        r"[0-9a-f]{64}", args.execution_patch_sha256
    ) is None:
        raise ValueError("execution-patch-sha256 must be one lowercase SHA256")

    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    torch.set_num_threads(runtime_threads)
    try:
        torch.set_num_interop_threads(min(2, runtime_threads))
    except RuntimeError:
        pass
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the registered Slurm job")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError("torchrun world size exceeds visible CUDA devices")
    torch.cuda.set_device(local_rank)
    protocol = load_protocol_v4(args.protocol.resolve())
    gate_lock = None
    if args.run_role == "smoke":
        if args.gate_a_lock is not None:
            raise ValueError("Gate A smoke runs cannot depend on their own future lock")
    else:
        if args.gate_a_lock is None:
            raise ValueError("main/ablation training requires the frozen Gate A lock")
        project_root = Path(__file__).resolve().parents[2]
        gate_lock = GateALock.load(
            args.gate_a_lock,
            project_root=project_root,
            protocol_path=args.protocol.resolve(),
            execution_patch_manifest_sha256=args.execution_patch_sha256,
        )
        if gate_lock.source_bundle_sha256 != args.source_bundle_sha256:
            raise ValueError("training source bundle differs from Gate A")
    profile_matrix = None
    profile_variant = None
    formal_selection = None
    if args.run_role == "performance_profile":
        if (
            args.performance_profile_matrix is None
            or args.performance_profile_id is None
        ):
            raise ValueError("performance profiles require their frozen matrix and ID")
        profile_matrix = load_lora_ddp_profile_matrix(
            args.performance_profile_matrix,
            base_source_bundle_sha256=args.source_bundle_sha256,
            protocol_v4_sha256=protocol.sha256,
        )
        profile_variant = profile_matrix.select(args.performance_profile_id)
        if world_size != profile_variant.world_size:
            raise ValueError("performance profile/world-size mismatch")
        if args.max_steps != profile_variant.optimizer_updates:
            raise ValueError("performance profile optimizer-update count changed")
        if args.formal_execution_selection is not None:
            raise ValueError("performance profile cannot consume a formal selection")
        microbatch = profile_variant.per_device_microbatch
        gradient_accumulation = profile_variant.gradient_accumulation
    else:
        if (
            args.performance_profile_matrix is not None
            or args.performance_profile_id is not None
        ):
            raise ValueError("profile-only execution inputs cannot enter scientific training")
        if args.training_stage == "mixed_edit" and args.run_role == "main":
            if (
                args.formal_execution_selection is None
                or args.pretokenized_cache is None
                or not args.run_id
                or gate_lock is None
                or args.execution_patch_sha256 is None
            ):
                raise ValueError(
                    "formal mixed-edit training requires its execution selection, "
                    "immutable token cache, run ID, and Gate-A lock"
                )
            formal_selection = load_formal_execution_selection(
                args.formal_execution_selection,
                run_id=args.run_id,
                base_source_bundle_sha256=args.source_bundle_sha256,
                protocol_v4_sha256=protocol.sha256,
                gate_a_lock_sha256=gate_lock.sha256,
                execution_patch_sha256=str(args.execution_patch_sha256),
            )
            microbatch = formal_selection.per_device_microbatch
            gradient_accumulation = formal_selection.gradient_accumulation
        else:
            if (
                args.formal_execution_selection is not None
                or args.pretokenized_cache is not None
            ):
                raise ValueError(
                    "formal execution inputs are restricted to main mixed-edit training"
                )
            microbatch = args.microbatch
            gradient_accumulation = args.gradient_accumulation
    llama_contract = protocol.data["llama_training"]
    max_length = int(llama_contract["max_sequence_length"])
    effective_batch = microbatch * gradient_accumulation * world_size
    if effective_batch != int(llama_contract["effective_batch_sequences"]):
        raise ValueError(
            "world size x microbatch x accumulation must equal the frozen effective batch"
        )
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max_steps must be -1 for main data or a positive smoke limit")
    if args.run_role == "smoke" and (
        args.max_steps <= 0 or args.training_stage != "coarse" or world_size != 1
    ):
        raise ValueError("smoke runs require a positive max_steps")
    epoch_contract = None
    if args.training_stage == "coarse":
        if args.initial_adapter is not None:
            raise ValueError("coarse LoRA stage must initialize from the frozen backbone")
        if args.training_amendment is not None:
            raise ValueError("coarse LoRA stage cannot consume the mixed-edit amendment")
        if args.run_role != "smoke" and args.max_steps != -1:
            raise ValueError("coarse main/ablation consumes its materialized epochs once")
        if world_size != 1:
            raise ValueError("historical coarse LoRA lanes remain single-GPU")
    else:
        if (
            args.representation != "wyckoff"
            or args.run_role not in {"main", "performance_profile"}
            or args.initial_adapter is None
            or args.training_amendment is None
            or args.execution_patch_sha256 is None
        ):
            raise ValueError(
                "mixed edit stage requires WQ main/profile, a coarse adapter, the active "
                "three-epoch amendment, and an authorized execution patch"
            )
        epoch_contract = load_mixed_edit_epoch_contract(
            args.training_amendment,
            base_protocol_name=protocol.name,
            base_protocol_sha256=protocol.sha256,
        )
        if effective_batch != epoch_contract.effective_batch_sequences:
            raise ValueError("mixed-edit amendment/effective batch mismatch")
        if args.run_role == "main":
            if (
                world_size != epoch_contract.world_size
                or microbatch != epoch_contract.per_device_microbatch
                or gradient_accumulation != epoch_contract.gradient_accumulation
                or formal_selection is None
                or formal_selection.gradient_checkpointing
                != epoch_contract.gradient_checkpointing
                or formal_selection.gradient_checkpointing_use_reentrant
                != epoch_contract.gradient_checkpointing_use_reentrant
                or formal_selection.data_mode != epoch_contract.data_mode
                or formal_selection.dataloader_num_workers
                != epoch_contract.dataloader_num_workers_per_rank
                or formal_selection.pretokenized_cache_manifest_sha256
                != epoch_contract.pretokenized_cache_manifest_sha256
                or formal_selection.attention_implementation
                not in epoch_contract.allowed_attention_implementations
            ):
                raise ValueError("mixed-edit distributed batch contract mismatch")
        elif (
            profile_variant is None
            or world_size != profile_variant.world_size
            or effective_batch != profile_variant.global_effective_batch
        ):
            raise ValueError("mixed-edit performance profile batch mismatch")
        if args.run_role == "main" and args.max_steps != -1:
            raise ValueError("full mixed-edit training forbids max-step overrides")
        if args.run_role == "performance_profile" and not 1 <= args.max_steps <= 100:
            raise ValueError("mixed-edit performance profile requires 1-100 updates")
    artifact_identity = validate_sft_artifacts(
        data_path=args.data,
        manifest_path=args.data_manifest,
        token_audit_path=args.token_audit,
        representation=args.representation,
        training_seed=args.training_seed,
        max_length=max_length,
        dataset_stage=args.training_stage,
    )
    if args.run_role == "canonical_ablation" and not artifact_identity["canonical_orbit_order"]:
        raise ValueError("canonical ablation requires canonical-order data")
    if args.run_role in {"main", "performance_profile"} and artifact_identity[
        "canonical_orbit_order"
    ]:
        raise ValueError("headline main run cannot consume canonical-order ablation data")
    output_dir = args.output_dir.resolve()
    _prepare_distributed_output(output_dir, rank=rank)
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.llama_root.resolve(),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
        model_max_length=max_length,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("registered tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_size_before = len(tokenizer)
    data_loading_identity: dict[str, object]
    selected_execution = profile_variant or formal_selection
    if (
        selected_execution is not None
        and selected_execution.data_mode == "pretokenized_memmap"
    ):
        if args.pretokenized_cache is None:
            raise ValueError("pretokenized execution requires its cache")
        dataset = PretokenizedMemmapSFTDataset(
            args.pretokenized_cache,
            data_path=args.data,
            tokenizer=tokenizer,
            max_length=max_length,
            verify_hashes=True,
        )
        data_loading_identity = dataset.identity()
        if (
            formal_selection is not None
            and data_loading_identity["cache_manifest_sha256"]
            != formal_selection.pretokenized_cache_manifest_sha256
        ):
            raise ValueError("formal pretokenized cache manifest changed")
    else:
        if args.pretokenized_cache is not None:
            raise ValueError("lazy JSONL profile forbids a pretokenized cache")
        dataset = JsonlSFTDataset(args.data, tokenizer, max_length=max_length)
        data_loading_identity = {
            "mode": "lazy_jsonl",
            "data_path": str(args.data.resolve()),
            "data_sha256": artifact_identity["data_sha256"],
        }
    if len(dataset) != int(artifact_identity["examples"]):
        raise ValueError("indexed SFT dataset denominator mismatch")
    formal_selection_identity = (
        None
        if formal_selection is None
        else {
            "path": str(formal_selection.path),
            "sha256": formal_selection.sha256,
            "selected_profile_id": formal_selection.selected_profile_id,
            "selected_attention_implementation": (
                formal_selection.attention_implementation
            ),
        }
    )
    collator = FixedLengthSFTCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=max_length,
    )
    attention_implementation = (
        "sdpa" if selected_execution is None else selected_execution.attention_implementation
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.llama_root.resolve(),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation=attention_implementation,
        low_cpu_mem_usage=True,
        device_map={"": local_rank},
    )
    model.config.use_cache = False
    gradient_checkpointing = (
        True if selected_execution is None else selected_execution.gradient_checkpointing
    )
    gradient_checkpointing_use_reentrant = (
        False
        if selected_execution is None
        else selected_execution.gradient_checkpointing_use_reentrant
    )
    if gradient_checkpointing:
        model.enable_input_require_grads()
    lora_contract = llama_contract["lora"]
    lora_config = LoraConfig(
        r=int(lora_contract["rank"]),
        lora_alpha=int(lora_contract["alpha"]),
        lora_dropout=float(lora_contract["dropout"]),
        target_modules=list(lora_contract["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    initial_adapter_identity = None
    if args.training_stage == "mixed_edit":
        initial_adapter = args.initial_adapter.resolve()
        initial_model = initial_adapter / "adapter_model.safetensors"
        initial_config = initial_adapter / "adapter_config.json"
        if not initial_model.is_file() or not initial_config.is_file():
            raise FileNotFoundError("coarse WQ adapter is incomplete")
        registered = json.loads(initial_config.read_text(encoding="utf-8"))
        if (
            int(registered.get("r", -1)) != int(lora_contract["rank"])
            or int(registered.get("lora_alpha", -1)) != int(lora_contract["alpha"])
            or set(registered.get("target_modules", ())) != set(lora_contract["target_modules"])
        ):
            raise ValueError("coarse adapter capacity differs from mixed-edit contract")
        model = peft.PeftModel.from_pretrained(
            model,
            initial_adapter,
            local_files_only=True,
            is_trainable=True,
        )
        initial_adapter_identity = {
            "path": str(initial_adapter),
            "model_sha256": sha256_file(initial_model),
            "config_sha256": sha256_file(initial_config),
        }
    else:
        model = get_peft_model(model, lora_config)
    trainable = sum(value.numel() for value in model.parameters() if value.requires_grad)
    total = sum(value.numel() for value in model.parameters())
    if trainable <= 0 or tokenizer_size_before != len(tokenizer):
        raise RuntimeError("LoRA trainability or no-vocabulary-resize contract failed")

    training_dir = output_dir / "trainer"
    full_mixed_edit = epoch_contract is not None and args.run_role == "main"
    mixed_edit_epochs = (
        None if epoch_contract is None else epoch_contract.maximum_effective_epochs
    )
    learning_rate = (
        float(llama_contract["learning_rate"])
        if epoch_contract is None
        else epoch_contract.learning_rate
    )
    warmup_fraction = (
        float(llama_contract["warmup_fraction"])
        if epoch_contract is None
        else epoch_contract.warmup_fraction
    )
    scheduler = (
        str(llama_contract["scheduler"])
        if epoch_contract is None
        else epoch_contract.scheduler
    )
    dataloader_num_workers = (
        0 if selected_execution is None else selected_execution.dataloader_num_workers
    )
    training_args = TrainingArguments(
        output_dir=str(training_dir),
        overwrite_output_dir=False,
        per_device_train_batch_size=microbatch,
        gradient_accumulation_steps=gradient_accumulation,
        num_train_epochs=float(mixed_edit_epochs or 1),
        max_steps=args.max_steps,
        learning_rate=learning_rate,
        weight_decay=float(llama_contract["weight_decay"]),
        warmup_ratio=warmup_fraction,
        lr_scheduler_type=scheduler,
        max_grad_norm=float(llama_contract["gradient_clip"]),
        bf16=True,
        fp16=False,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": gradient_checkpointing_use_reentrant}
            if gradient_checkpointing
            else None
        ),
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch" if full_mixed_edit else "no",
        save_total_limit=(
            None if not full_mixed_edit else epoch_contract.save_total_limit
        ),
        load_best_model_at_end=False,
        report_to=[],
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        optim="adamw_torch",
        seed=args.training_seed,
        data_seed=args.training_seed,
        full_determinism=False,
        ddp_backend="nccl" if world_size > 1 else None,
        ddp_find_unused_parameters=False if world_size > 1 else None,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    result = trainer.train()
    trainer.accelerator.wait_for_everyone()
    local_runtime = {
        "rank": rank,
        "local_rank": local_rank,
        "cuda_device": torch.cuda.get_device_name(local_rank),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(local_rank),
    }
    rank_runtimes: list[dict[str, object] | None] = [None] * world_size
    if world_size > 1:
        if not torch.distributed.is_initialized():
            raise RuntimeError("Trainer did not initialize the registered DDP process group")
        torch.distributed.all_gather_object(rank_runtimes, local_runtime)
    else:
        rank_runtimes[0] = local_runtime
    if args.run_role == "performance_profile":
        completed = int(trainer.state.global_step)
        train_loss = float(result.metrics.get("train_loss", float("nan")))
        train_runtime = float(result.metrics.get("train_runtime", float("nan")))
        profile_report = {
            "schema": "crysllmgen_lora_ddp_performance_profile_v1",
            "scientific_attempt": False,
            "pass": completed == args.max_steps and math.isfinite(train_loss),
            "training_seed": args.training_seed,
            "training_stage": args.training_stage,
            "representation": args.representation,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "protocol_sha256": protocol.sha256,
            "training_amendment": {
                "path": str(epoch_contract.path),
                "sha256": epoch_contract.sha256,
            },
            "source_bundle_sha256": args.source_bundle_sha256,
            "execution_patch_sha256": args.execution_patch_sha256,
            "profile_matrix": {
                "path": str(profile_matrix.path),
                "sha256": profile_matrix.sha256,
                "authorization_record": profile_matrix.authorization_record,
                "authorization_sha256": profile_matrix.authorization_sha256,
                "profile_id": profile_variant.profile_id,
            },
            "gate_a_lock": {
                "path": str(gate_lock.path),
                "sha256": gate_lock.sha256,
            },
            "data": artifact_identity,
            "optimizer": {
                "profile_updates": args.max_steps,
                "completed_global_step": completed,
                "world_size": world_size,
                "per_device_microbatch": microbatch,
                "gradient_accumulation": gradient_accumulation,
                "global_effective_batch": effective_batch,
                "learning_rate": learning_rate,
                "warmup_fraction": warmup_fraction,
                "scheduler": scheduler,
            },
            "distributed": {
                "backend": str(torch.distributed.get_backend()),
                "rank_runtimes": rank_runtimes,
            },
            "execution": {
                "attention_implementation": attention_implementation,
                "gradient_checkpointing": gradient_checkpointing,
                "gradient_checkpointing_use_reentrant": (
                    gradient_checkpointing_use_reentrant
                ),
                "data_loading": data_loading_identity,
                "dataloader_num_workers_per_rank": dataloader_num_workers,
            },
            "metrics": dict(result.metrics),
            "runtime": {
                "walltime_s": time.time() - started,
                "train_runtime_s": train_runtime,
                "train_seconds_per_optimizer_update": (
                    train_runtime / completed
                    if completed > 0 and math.isfinite(train_runtime)
                    else float("nan")
                ),
                "threads_per_rank": runtime_threads,
                "offline": True,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
                "python": sys.version,
            },
        }
        if rank == 0:
            _write_exclusive(output_dir / "performance_profile.json", profile_report)
            print(json.dumps(profile_report, indent=2, sort_keys=True))
        trainer.accelerator.wait_for_everyone()
        return
    if rank != 0:
        trainer.accelerator.wait_for_everyone()
        return
    epoch_checkpoints: list[dict[str, object]] = []
    if epoch_contract is not None:
        epoch_checkpoints = audit_epoch_checkpoints(
            training_dir,
            examples=int(artifact_identity["examples"]),
            effective_batch=effective_batch,
            required_epochs=epoch_contract.required_checkpoint_epochs,
        )
    adapter_dir = output_dir / "adapter_final"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir / "tokenizer_identity")
    trainer.save_state()
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise RuntimeError("final LoRA adapter was not written")
    if epoch_checkpoints and sha256_file(adapter_file) != str(
        epoch_checkpoints[-1]["adapter_model_sha256"]
    ):
        raise RuntimeError("final adapter differs from the immutable epoch-3 checkpoint")
    epoch_exports: list[dict[str, object]] = []
    for checkpoint in epoch_checkpoints:
        logical_epoch = int(checkpoint["logical_epoch"])
        epoch_root = output_dir / f"epoch_{logical_epoch:02d}"
        epoch_adapter = epoch_root / "adapter_final"
        epoch_adapter.mkdir(parents=True, exist_ok=False)
        source_model = Path(str(checkpoint["adapter_model_path"]))
        source_config = Path(str(checkpoint["adapter_config_path"]))
        exported_model = epoch_adapter / "adapter_model.safetensors"
        exported_config = epoch_adapter / "adapter_config.json"
        shutil.copy2(source_model, exported_model)
        shutil.copy2(source_config, exported_config)
        if (
            sha256_file(exported_model) != checkpoint["adapter_model_sha256"]
            or sha256_file(exported_config) != checkpoint["adapter_config_sha256"]
        ):
            raise RuntimeError("exported epoch adapter differs from Trainer checkpoint")
        epoch_report = {
            "schema": "crysllmgen_lora_training_report_v1",
            "run_role": args.run_role,
            "representation": args.representation,
            "training_stage": args.training_stage,
            "training_seed": args.training_seed,
            "logical_epoch": logical_epoch,
            "global_step": int(checkpoint["global_step"]),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "protocol_path": str(protocol.path),
            "protocol_sha256": protocol.sha256,
            "training_amendment": {
                "path": str(epoch_contract.path),
                "sha256": epoch_contract.sha256,
            },
            "source_bundle_sha256": args.source_bundle_sha256,
            "execution_patch_sha256": args.execution_patch_sha256,
            "gate_a_lock": (
                None
                if gate_lock is None
                else {"path": str(gate_lock.path), "sha256": gate_lock.sha256}
            ),
            "initial_adapter": initial_adapter_identity,
            "formal_execution_selection": formal_selection_identity,
            "llama_root": str(args.llama_root.resolve()),
            "data": artifact_identity,
            "optimizer": {
                "effective_batch": effective_batch,
                "world_size": world_size,
                "per_device_microbatch": microbatch,
                "gradient_accumulation": gradient_accumulation,
                "learning_rate": learning_rate,
                "warmup_fraction": warmup_fraction,
                "scheduler": scheduler,
                "maximum_effective_epochs": mixed_edit_epochs,
                "completed_global_step": int(checkpoint["global_step"]),
                "fixed_sequence_length": max_length,
            },
            "model": {
                "total_parameters": total,
                "trainable_parameters": trainable,
                "lora_rank": int(lora_contract["rank"]),
                "lora_alpha": int(lora_contract["alpha"]),
                "target_modules": list(lora_contract["target_modules"]),
                "adapter_path": str(epoch_adapter),
                "adapter_bytes": exported_model.stat().st_size,
                "adapter_sha256": sha256_file(exported_model),
                "adapter_config_sha256": sha256_file(exported_config),
                "tokenizer_size": tokenizer_size_before,
                "vocabulary_resized": False,
            },
            "checkpoint": checkpoint,
            "execution": {
                "attention_implementation": attention_implementation,
                "gradient_checkpointing": gradient_checkpointing,
                "gradient_checkpointing_use_reentrant": (
                    gradient_checkpointing_use_reentrant
                ),
                "data_loading": data_loading_identity,
                "dataloader_num_workers_per_rank": dataloader_num_workers,
            },
            "runtime": {
                "checkpoint_created_by_job": os.environ["SLURM_JOB_ID"],
                "world_size": world_size,
                "threads_per_rank": runtime_threads,
                "rank_runtimes": rank_runtimes,
                "offline": True,
            },
        }
        epoch_report_path = epoch_root / "training_report.json"
        _write_exclusive(epoch_report_path, epoch_report)
        epoch_exports.append(
            {
                **checkpoint,
                "exported_adapter": str(epoch_adapter),
                "exported_adapter_sha256": sha256_file(exported_model),
                "epoch_report": str(epoch_report_path),
                "epoch_report_sha256": sha256_file(epoch_report_path),
                "evaluation_status": "required_pending",
            }
        )
    report = {
        "schema": "crysllmgen_lora_training_report_v1",
        "run_role": args.run_role,
        "representation": args.representation,
        "training_stage": args.training_stage,
        "training_seed": args.training_seed,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "protocol_path": str(protocol.path),
        "protocol_sha256": protocol.sha256,
        "training_amendment": (
            None
            if epoch_contract is None
            else {"path": str(epoch_contract.path), "sha256": epoch_contract.sha256}
        ),
        "source_bundle_sha256": args.source_bundle_sha256,
        "execution_patch_sha256": args.execution_patch_sha256,
        "gate_a_lock": (
            None
            if gate_lock is None
            else {"path": str(gate_lock.path), "sha256": gate_lock.sha256}
        ),
        "initial_adapter": initial_adapter_identity,
        "formal_execution_selection": formal_selection_identity,
        "llama_root": str(args.llama_root.resolve()),
        "data": artifact_identity,
        "optimizer": {
            "effective_batch": effective_batch,
            "world_size": world_size,
            "per_device_microbatch": microbatch,
            "gradient_accumulation": gradient_accumulation,
            "learning_rate": learning_rate,
            "warmup_fraction": warmup_fraction,
            "scheduler": scheduler,
            "max_steps": args.max_steps,
            "maximum_effective_epochs": mixed_edit_epochs,
            "completed_global_step": int(trainer.state.global_step),
            "fixed_sequence_length": max_length,
        },
        "model": {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "lora_rank": int(lora_contract["rank"]),
            "lora_alpha": int(lora_contract["alpha"]),
            "target_modules": list(lora_contract["target_modules"]),
            "adapter_path": str(adapter_dir),
            "adapter_bytes": adapter_file.stat().st_size,
            "adapter_sha256": sha256_file(adapter_file),
            "tokenizer_size": tokenizer_size_before,
            "vocabulary_resized": False,
        },
        "epoch_checkpoints": epoch_exports,
        "per_epoch_evaluation_required": bool(epoch_exports),
        "execution": {
            "attention_implementation": attention_implementation,
            "gradient_checkpointing": gradient_checkpointing,
            "gradient_checkpointing_use_reentrant": (
                gradient_checkpointing_use_reentrant
            ),
            "data_loading": data_loading_identity,
            "dataloader_num_workers_per_rank": dataloader_num_workers,
        },
        "metrics": dict(result.metrics),
        "runtime": {
            "walltime_s": time.time() - started,
            "peak_memory_bytes_per_rank": [
                int(value["peak_memory_bytes"])
                for value in rank_runtimes
                if value is not None
            ],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "cuda_devices": [
                str(value["cuda_device"])
                for value in rank_runtimes
                if value is not None
            ],
            "world_size": world_size,
            "ddp_backend": (
                str(torch.distributed.get_backend())
                if torch.distributed.is_initialized()
                else "none"
            ),
            "threads_per_rank": runtime_threads,
            "offline": True,
            "python": sys.version,
        },
    }
    _write_exclusive(output_dir / "training_report.json", report)
    trainer.accelerator.wait_for_everyone()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
