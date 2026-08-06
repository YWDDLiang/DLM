"""Three-epoch mixed-edit training and checkpoint audit helpers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class MixedEditEpochContract:
    path: Path
    sha256: str
    data: Mapping[str, Any]
    maximum_effective_epochs: int
    effective_batch_sequences: int
    world_size: int
    per_device_microbatch: int
    gradient_accumulation: int
    learning_rate: float
    warmup_fraction: float
    scheduler: str
    save_total_limit: int
    required_checkpoint_epochs: tuple[int, ...]
    gradient_checkpointing: bool
    gradient_checkpointing_use_reentrant: bool | None
    data_mode: str
    dataloader_num_workers_per_rank: int
    pretokenized_cache_manifest_sha256: str
    allowed_attention_implementations: tuple[str, ...]


def load_mixed_edit_epoch_contract(
    path: str | Path,
    *,
    base_protocol_name: str,
    base_protocol_sha256: str,
) -> MixedEditEpochContract:
    contract_path = Path(path).resolve()
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "crysllmgen_mixed_edit_three_epoch_v1":
        raise ValueError("unsupported mixed-edit epoch contract")
    if payload.get("status") != "active_user_authorized_amendment":
        raise ValueError("mixed-edit epoch contract is not active")
    base = payload.get("base_protocol") or {}
    if (
        base.get("name") != base_protocol_name
        or base.get("sha256") != base_protocol_sha256
    ):
        raise ValueError("mixed-edit epoch contract/base protocol mismatch")
    scope = payload.get("scope") or {}
    if scope != {
        "representation": "wyckoff",
        "training_stage": "mixed_edit",
        "run_role": "main",
    }:
        raise ValueError("mixed-edit epoch contract scope changed")
    training = payload.get("training") or {}
    epochs = int(training.get("maximum_effective_epochs", -1))
    effective_batch = int(training.get("effective_batch_sequences", -1))
    distributed_batch = training.get("distributed_batch") or {}
    world_size = int(distributed_batch.get("world_size", -1))
    per_device_microbatch = int(distributed_batch.get("per_device_microbatch", -1))
    gradient_accumulation = int(distributed_batch.get("gradient_accumulation", -1))
    optimization = training.get("optimization") or {}
    learning_rate = float(optimization.get("learning_rate", -1.0))
    warmup_fraction = float(optimization.get("warmup_fraction", -1.0))
    scheduler = str(optimization.get("scheduler", ""))
    save_total_limit = int(training.get("save_total_limit", -1))
    required = tuple(int(value) for value in training.get("required_checkpoint_epochs", ()))
    if (
        epochs != 3
        or effective_batch <= 0
        or world_size != 2
        or per_device_microbatch != 8
        or gradient_accumulation != 4
        or world_size * per_device_microbatch * gradient_accumulation
        != effective_batch
        or distributed_batch.get("formula")
        != "world_size * per_device_microbatch * gradient_accumulation = effective_batch_sequences"
        or not math.isclose(learning_rate, 1.0e-4, rel_tol=0.0, abs_tol=1.0e-12)
        or not math.isclose(warmup_fraction, 0.03, rel_tol=0.0, abs_tol=1.0e-12)
        or scheduler != "constant_with_warmup"
        or training.get("step_override") != "forbidden"
        or training.get("save_strategy") != "epoch"
        or save_total_limit < 3
        or training.get("load_best_model_at_end") is not False
        or training.get("final_adapter_source") != "epoch_3"
        or required != (1, 2, 3)
    ):
        raise ValueError("three-epoch checkpoint contract changed")
    evaluation = payload.get("evaluation") or {}
    selection = evaluation.get("selection") or {}
    sun_selection = selection.get("r5c_a100_sun") or {}
    direct_metrics = selection.get("crysllmgen_direct_metrics") or {}
    if (
        evaluation.get("evaluate_every_checkpoint") is not True
        or evaluation.get("same_attempt_ids_across_epochs") is not True
        or int(evaluation.get("smoke_attempts_per_configuration", -1)) != 256
        or selection.get("final_test_or_final_mlip_panel_selection") != "forbidden"
        or selection.get("scope")
        != "validation_smoke_panel_only_never_final_test_attempts"
        or selection.get("policy")
        != "r5c_a100_strict_then_meta_with_crysllmgen_noninferiority_v1"
        or sun_selection.get("participates_in_checkpoint_selection") is not True
        or (sun_selection.get("selection_metrics") or {}).get("primary")
        != "attempt_strict_full_sun_lower_bound"
        or (sun_selection.get("selection_metrics") or {}).get("secondary")
        != "attempt_meta_full_sun_lower_bound"
        or (sun_selection.get("selection_metrics") or {}).get(
            "coverage_adjusted_estimates"
        )
        != "report_only_never_checkpoint_selection"
        or int(sun_selection.get("attempt_denominator", -1)) != 256
        or int(sun_selection.get("paired_bootstrap_draws", -1)) != 10000
        or float((sun_selection.get("thresholds_ev_per_atom") or {}).get("strict", -1))
        != 0.0
        or float(
            (sun_selection.get("thresholds_ev_per_atom") or {}).get(
                "meta_like", -1
            )
        )
        != 0.1
        or (sun_selection.get("exact_executor") or {}).get(
            "eval_sun_py_sha256"
        )
        != "564b4490f01464012277653951f8a55b5c1575bc78091f5a06db25ca9339852b"
        or (sun_selection.get("exact_executor") or {}).get(
            "eval_sun_resumable_py_sha256"
        )
        != "44c7d9adf01de29d5bdd0eb6a0e6e5d77f1b47b47f0e4bcd9b35cd6c51e19baa"
        or (sun_selection.get("frozen_references") or {}).get(
            "mp20_training_index_cache_sha256"
        )
        != "f26ea30d6f529cca2d743401049e0328227faa156c95af3e0641d35fe03ffc62"
        or direct_metrics.get("source") != "unchanged_upstream_compute_metrics.py"
        or set(direct_metrics.get("higher_is_better", ()))
        != {"comp_valid", "struct_valid", "valid", "cov_recall", "cov_precision"}
        or set(direct_metrics.get("lower_is_better", ()))
        != {"wdist_density", "wdist_num_elems"}
    ):
        raise ValueError("per-epoch evaluation contract changed")
    execution = payload.get("execution") or {}
    profile = execution.get("profile") or {}
    data_loading = execution.get("data_loading") or {}
    attention_selection = execution.get("attention_selection") or {}
    if (
        execution.get("partition") != "gpu"
        or execution.get("launcher") != "torchrun_standalone_single_node"
        or int(execution.get("training_world_size", -1)) != world_size
        or int(execution.get("gpus_per_training_job", -1)) != world_size
        or int(execution.get("cpus_per_training_job", -1)) != 16
        or int(execution.get("numerical_threads_per_rank", -1)) != 4
        or execution.get("ddp_backend") != "nccl"
        or execution.get("ddp_find_unused_parameters") is not False
        or execution.get("gradient_checkpointing") is not False
        or execution.get("gradient_checkpointing_use_reentrant") is not None
        or data_loading.get("mode") != "pretokenized_memmap"
        or int(data_loading.get("dataloader_num_workers_per_rank", -1)) != 4
        or data_loading.get("cache_manifest_sha256")
        != "afd954868f1f9c431068e92b444061079d741d6b23e8cdddccb1e9bedd9ddb23"
        or int(data_loading.get("examples", -1)) != 54270
        or int(data_loading.get("max_length", -1)) != 512
        or data_loading.get("example_id_order_sha256")
        != "961a40e9bf76cca8df2b09bd3be932f813caa1a47fa0be2358b82c71449a735f"
        or attention_selection.get("record_schema")
        != "crysllmgen_formal_execution_selection_v1"
        or tuple(attention_selection.get("allowed_implementations", ()))
        != ("sdpa", "flash_attention_2")
        or attention_selection.get("flash_profile")
        != "configs/experiments/wyckoff_codiffusion/ddp_flash_followup_v1.json"
        or attention_selection.get("flash_profile_id")
        != "cache4_mb8_acc4_gc_off_flash2"
        or attention_selection.get("sdpa_reference_profile_id")
        != "cache4_mb8_acc4_gc_off_sdpa"
        or float(attention_selection.get("sdpa_seconds_per_update", -1.0))
        != 4.16302
        or float(attention_selection.get("flash_minimum_relative_speedup", -1.0))
        != 0.02
        or attention_selection.get("failed_or_ineligible_flash_fallback") != "sdpa"
        or profile.get("required_before_full_training") is not True
        or int(profile.get("maximum_optimizer_updates", -1)) != 100
        or profile.get("scientific_attempt") is not False
        or int(execution.get("long_job_user_approval_threshold_hours", -1)) != 6
        or execution.get("two_gpu_user_choice_record")
        != "runs/remote_audit/20260721_mixed_edit_two_gpu_user_choice_v1.json"
        or execution.get("batch_submission")
        != "one_manifest_one_dependency_dag"
    ):
        raise ValueError("three-epoch execution contract changed")
    return MixedEditEpochContract(
        path=contract_path,
        sha256=sha256_file(contract_path),
        data=payload,
        maximum_effective_epochs=epochs,
        effective_batch_sequences=effective_batch,
        world_size=world_size,
        per_device_microbatch=per_device_microbatch,
        gradient_accumulation=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_fraction=warmup_fraction,
        scheduler=scheduler,
        save_total_limit=save_total_limit,
        required_checkpoint_epochs=required,
        gradient_checkpointing=False,
        gradient_checkpointing_use_reentrant=None,
        data_mode="pretokenized_memmap",
        dataloader_num_workers_per_rank=4,
        pretokenized_cache_manifest_sha256=str(
            data_loading["cache_manifest_sha256"]
        ),
        allowed_attention_implementations=("sdpa", "flash_attention_2"),
    )


def updates_per_effective_epoch(*, examples: int, effective_batch: int) -> int:
    if int(examples) <= 0 or int(effective_batch) <= 0:
        raise ValueError("examples and effective batch must be positive")
    return int(math.ceil(int(examples) / int(effective_batch)))


def expected_epoch_steps(
    *, examples: int, effective_batch: int, epochs: int
) -> tuple[int, ...]:
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    per_epoch = updates_per_effective_epoch(
        examples=examples,
        effective_batch=effective_batch,
    )
    return tuple(per_epoch * epoch for epoch in range(1, int(epochs) + 1))


def audit_epoch_checkpoints(
    training_dir: str | Path,
    *,
    examples: int,
    effective_batch: int,
    required_epochs: tuple[int, ...] = (1, 2, 3),
) -> list[dict[str, Any]]:
    root = Path(training_dir).resolve()
    expected_steps = expected_epoch_steps(
        examples=examples,
        effective_batch=effective_batch,
        epochs=len(required_epochs),
    )
    expected = dict(zip(required_epochs, expected_steps, strict=True))
    observed: dict[int, dict[str, Any]] = {}
    for checkpoint in sorted(root.glob("checkpoint-*")):
        if not checkpoint.is_dir():
            continue
        try:
            step_from_name = int(checkpoint.name.removeprefix("checkpoint-"))
        except ValueError as exc:
            raise ValueError(f"invalid checkpoint directory: {checkpoint}") from exc
        state_path = checkpoint / "trainer_state.json"
        model_path = checkpoint / "adapter_model.safetensors"
        config_path = checkpoint / "adapter_config.json"
        for required in (state_path, model_path, config_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        global_step = int(state.get("global_step", -1))
        if global_step != step_from_name:
            raise ValueError("checkpoint name/trainer global step mismatch")
        matches = [epoch for epoch, step in expected.items() if step == global_step]
        if len(matches) != 1:
            raise ValueError(f"unregistered epoch checkpoint step: {global_step}")
        logical_epoch = matches[0]
        if logical_epoch in observed:
            raise ValueError(f"duplicate checkpoint for logical epoch {logical_epoch}")
        trainer_epoch = float(state.get("epoch", float("nan")))
        if not math.isfinite(trainer_epoch) or abs(trainer_epoch - logical_epoch) > 0.05:
            raise ValueError("trainer epoch/checkpoint boundary mismatch")
        observed[logical_epoch] = {
            "logical_epoch": logical_epoch,
            "global_step": global_step,
            "trainer_epoch": trainer_epoch,
            "checkpoint_path": str(checkpoint),
            "adapter_model_path": str(model_path),
            "adapter_model_sha256": sha256_file(model_path),
            "adapter_config_path": str(config_path),
            "adapter_config_sha256": sha256_file(config_path),
            "trainer_state_path": str(state_path),
            "trainer_state_sha256": sha256_file(state_path),
        }
    if set(observed) != set(required_epochs):
        raise ValueError(
            "epoch checkpoint set mismatch: "
            f"expected={list(required_epochs)}, observed={sorted(observed)}"
        )
    return [observed[epoch] for epoch in required_epochs]
