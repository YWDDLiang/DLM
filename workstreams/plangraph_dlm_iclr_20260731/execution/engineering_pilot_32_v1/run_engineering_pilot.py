#!/usr/bin/env python3
"""Run one frozen PlanGraph-DLM 32-row engineering arm.

The runner deliberately leaves the frozen training entry point byte-for-byte
unchanged.  It instruments that entry point in-process to capture gradient
norms, CUDA peak memory, and phase timings, then enforces the registered
engineering-only acceptance gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


FROZEN_MANIFEST_SHA256 = (
    "097943f6947ca166a07cb8dcbce0b68b938e52ba1e2d850ec6cca45e0aa69777"
)
FROZEN_ENTRYPOINT_SHA256 = (
    "edaf8a500447ef57edfd3048049acb4250adae2c8dd41ba7d7da71474eb6d977"
)
FROZEN_ADAPTER_SHA256 = (
    "5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d"
)
FROZEN_TOKENIZER_VOCAB_SHA256 = (
    "dd3968132cdd9f75f92363a842d26a8d083a4a05f0411ee856ba5e473090815d"
)
FROZEN_DATA_MANIFEST_SHA256 = (
    "2aac3b91a9cc5f0b18bfa0084886f7cb7e5eca689d0fdce3235f41ccfc0b2e2e"
)
EXPECTED_ARMS = ("D0", "D1", "D2", "D2-shuffle")
ARM_POLICIES = {
    "D0": ("none", 1.0, 0.0),
    "D1": ("d1", 2.0, 1.0),
    "D2": ("d2", 2.0, 1.0),
    "D2-shuffle": ("d2_shuffle", 2.0, 1.0),
}
ERROR_MARKERS = (
    "out of memory",
    "nan",
    " inf",
    "cuda error",
    "nccl",
    "traceback",
    "brokenprocesspool",
)


class PilotGateError(RuntimeError):
    """Raised when a frozen engineering-pilot gate changes or fails."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_first_nonempty_raw_lines(path: str | Path, count: int) -> str:
    """Hash exactly the first ``count`` non-empty raw lines, including newlines."""

    digest = hashlib.sha256()
    observed = 0
    with Path(path).open("rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            digest.update(raw_line)
            observed += 1
            if observed == int(count):
                break
    if observed != int(count):
        raise PilotGateError(
            f"{path} contains only {observed} non-empty rows; expected {count}"
        )
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PilotGateError(f"expected JSON object: {path}")
    return payload


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise PilotGateError(
            f"{label} changed: observed={observed!r} expected={expected!r}"
        )


def validate_authorization(
    authorization: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> None:
    require_equal(
        authorization.get("schema_version"),
        "plangraph-dlm-engineering-pilot-authorization@1",
        "authorization schema",
    )
    require_equal(
        authorization.get("status"),
        "authorized_for_one_submission",
        "authorization status",
    )
    require_equal(
        (authorization.get("frozen_manifest") or {}).get("sha256"),
        manifest_sha256,
        "authorization manifest SHA-256",
    )
    scope = authorization.get("authorized_scope") or {}
    require_equal(scope.get("job_submission"), True, "job submission authorization")
    require_equal(scope.get("submission_count"), 1, "submission count")
    require_equal(scope.get("array_indices"), [0, 1, 2, 3], "array indices")
    require_equal(scope.get("arms_in_order"), list(EXPECTED_ARMS), "arm order")
    require_equal(scope.get("maximum_concurrent_elements"), 1, "array concurrency")
    require_equal(scope.get("a800_per_element"), 1, "A800 count")
    require_equal(scope.get("cpus_per_element"), 8, "CPU count")
    require_equal(scope.get("memory_gib_per_element"), 64, "memory GiB")
    require_equal(scope.get("time_limit_per_element"), "01:00:00", "time limit")
    locks = authorization.get("continuing_locks") or {}
    for key in (
        "automatic_downstream",
        "automatic_promotion",
        "checkpoint_selection",
        "scientific_result",
        "crystal_generation",
        "refinement",
        "sun_evaluation",
        "chgnet",
        "mlip",
        "mp_api",
        "modify_h1_artifacts",
        "modify_continuous_refiner",
    ):
        require_equal(locks.get(key), False, f"continuing lock {key}")


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    require_equal(
        manifest.get("schema_version"),
        "plangraph-dlm-engineering-pilot@1",
        "pilot manifest schema",
    )
    require_equal(
        manifest.get("identity"),
        "plangraph_dlm_engineering32_v1",
        "pilot identity",
    )
    require_equal(
        manifest.get("status"),
        "frozen_review_point_not_authorized_not_submitted",
        "frozen pilot status",
    )
    original_auth = manifest.get("authorization") or {}
    require_equal(
        original_auth.get("job_submission"),
        False,
        "frozen manifest must remain unmodified",
    )
    resources = manifest.get("resources_if_later_authorized") or {}
    expected_resources = {
        "partition": "gpu",
        "array_elements": 4,
        "maximum_concurrent_elements": 1,
        "a800_per_element": 1,
        "cpus_per_element": 8,
        "memory_gib_per_element": 64,
        "time_limit_per_element": "01:00:00",
        "cpus_per_a800_hard_maximum": 8,
        "maximum_reserved_a800_hours": 4,
    }
    for key, expected in expected_resources.items():
        require_equal(resources.get(key), expected, f"manifest resource {key}")
    arms = manifest.get("pilot_protocol", {}).get("arms") or []
    require_equal(
        [entry.get("arm") for entry in arms],
        list(EXPECTED_ARMS),
        "manifest arm order",
    )


def validate_slurm_environment(arm_index: int) -> None:
    require_equal(os.environ.get("SLURM_JOB_PARTITION"), "gpu", "Slurm partition")
    require_equal(int(os.environ.get("SLURM_CPUS_PER_TASK", "0")), 8, "Slurm CPUs")
    require_equal(
        int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")),
        int(arm_index),
        "Slurm array index",
    )
    optional_exact = {
        "SLURM_ARRAY_TASK_MIN": 0,
        "SLURM_ARRAY_TASK_MAX": 3,
        "SLURM_ARRAY_TASK_STEP": 1,
        "SLURM_ARRAY_TASK_COUNT": 4,
    }
    for variable, expected in optional_exact.items():
        if variable in os.environ:
            require_equal(int(os.environ[variable]), expected, variable)


def verify_published_dataset(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = manifest["frozen_data"]
    root = Path(frozen["dataset_root"]).resolve()
    manifest_path = root / "manifest.json"
    success_path = root / "_SUCCESS"
    require_equal(
        sha256_file(manifest_path),
        FROZEN_DATA_MANIFEST_SHA256,
        "dataset manifest SHA-256",
    )
    published = read_json(manifest_path)
    success = read_json(success_path)
    require_equal(
        success.get("manifest_sha256"),
        FROZEN_DATA_MANIFEST_SHA256,
        "dataset success marker",
    )
    require_equal(published.get("published"), True, "dataset publication status")
    require_equal(published.get("total_rows"), 36183, "dataset rows")
    require_equal(published.get("failed_rows"), 0, "dataset failed rows")
    require_equal(
        published.get("fixed_validation_panel", {}).get(
            "ordered_training_pair_sha256"
        ),
        frozen["fixed_validation_panel_ordered_sha256"],
        "fixed validation panel identity",
    )
    for relative, expected_sha in (
        published.get("output_file_sha256") or {}
    ).items():
        require_equal(
            sha256_file(root / relative),
            expected_sha,
            f"dataset output {relative}",
        )
    selection = frozen["selection"]
    for split in ("train", "validation"):
        published_split = "val" if split == "validation" else split
        body_path = root / "body" / f"{published_split}.jsonl"
        ledger_path = root / "row_ledger" / f"{published_split}.jsonl"
        require_equal(
            sha256_first_nonempty_raw_lines(body_path, 32),
            selection[split]["first32_body_raw_sha256"],
            f"{split} first-32 body identity",
        )
        require_equal(
            sha256_first_nonempty_raw_lines(ledger_path, 32),
            selection[split]["first32_row_ledger_raw_sha256"],
            f"{split} first-32 row-ledger identity",
        )
    return {
        "dataset_root": str(root),
        "manifest_sha256": FROZEN_DATA_MANIFEST_SHA256,
        "verified_output_files": len(published.get("output_file_sha256") or {}),
        "first32_train_verified": True,
        "first32_validation_verified": True,
    }


def build_training_argv(
    *,
    manifest: Mapping[str, Any],
    arm: str,
    output_dir: str | Path,
) -> list[str]:
    if arm not in ARM_POLICIES:
        raise PilotGateError(f"unknown arm: {arm}")
    policy, iid_fraction, planned_fraction = ARM_POLICIES[arm]
    shared = manifest["pilot_protocol"]["shared_training_args"]
    model = manifest["frozen_model"]
    data_root = Path(manifest["frozen_data"]["dataset_root"])
    return [
        str(manifest["execution_identity"]["entrypoint"]["path"]),
        "--model-path",
        str(model["base_model_path"]),
        "--checkpoint-path",
        str(model["initial_adapter_path"]),
        "--data-dir",
        str(data_root / "body"),
        "--representation",
        str(shared["representation"]),
        "--planned-corruption-policy",
        policy,
        "--iid-fraction",
        str(iid_fraction),
        "--planned-fraction",
        str(planned_fraction),
        "--corruption-seed",
        str(shared["corruption_seed"]),
        "--output-dir",
        str(output_dir),
        "--max-length",
        str(shared["max_length"]),
        "--answer-token-count",
        str(shared["answer_token_count"]),
        "--epochs",
        str(shared["epochs"]),
        "--max-train-steps",
        str(shared["max_train_steps"]),
        "--batch-size",
        str(shared["batch_size"]),
        "--grad-accum",
        str(shared["grad_accum"]),
        "--lr",
        str(shared["learning_rate"]),
        "--lr-scheduler",
        str(shared["lr_scheduler"]),
        "--warmup-steps",
        str(shared["warmup_steps"]),
        "--weight-decay",
        str(shared["weight_decay"]),
        "--logging-steps",
        str(shared["logging_steps"]),
        "--eval-steps",
        str(shared["eval_steps"]),
        "--eval-max-batches",
        str(shared["eval_max_batches"]),
        "--save-steps",
        str(shared["save_steps"]),
        "--dataloader-num-workers",
        str(shared["dataloader_num_workers"]),
        "--limit-train",
        str(shared["limit_train"]),
        "--limit-val",
        str(shared["limit_val"]),
        "--use-lora",
        "--lora-rank",
        str(shared["lora_rank"]),
        "--lora-alpha",
        str(shared["lora_alpha"]),
        "--lora-dropout",
        str(shared["lora_dropout"]),
        "--lora-target-modules",
        str(shared["lora_target_modules"]),
        "--modules-to-save",
        str(shared["modules_to_save"]),
    ]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise PilotGateError(f"non-object JSONL row: {path}")
                rows.append(payload)
    return rows


def validate_completed_training(
    *,
    training_log: str | Path,
    gradient_log: str | Path,
    instrumentation: Mapping[str, Any],
) -> dict[str, Any]:
    events = load_jsonl(training_log)
    gradients = load_jsonl(gradient_log)
    train_events = [entry for entry in events if entry.get("event") == "train"]
    eval_events = [entry for entry in events if entry.get("event") == "eval"]
    require_equal(
        [entry.get("step") for entry in train_events],
        [1, 2, 3, 4],
        "optimizer update log",
    )
    require_equal(len(eval_events), 1, "validation event count")
    require_equal(eval_events[0].get("step"), 4, "validation step")
    require_equal(len(gradients), 4, "gradient norm count")
    require_equal(
        int(instrumentation.get("train_compute_calls", -1)),
        32,
        "train microbatch count",
    )
    require_equal(
        int(instrumentation.get("optimizer_steps", -1)),
        4,
        "instrumented optimizer steps",
    )
    losses = [entry.get("loss") for entry in train_events]
    task_losses = [entry.get("task_loss") for entry in train_events]
    validation_losses = [entry.get("val_loss") for entry in eval_events]
    gradient_norms = [entry.get("pre_clip_grad_norm") for entry in gradients]
    for label, values in (
        ("loss", losses),
        ("task loss", task_losses),
        ("validation loss", validation_losses),
        ("gradient norm", gradient_norms),
    ):
        if not values or any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in values
        ):
            raise PilotGateError(f"{label} values are not all finite and positive")
    return {
        "train_microbatches": 32,
        "optimizer_updates": 4,
        "train_losses": losses,
        "task_losses": task_losses,
        "validation_losses": validation_losses,
        "pre_clip_gradient_norms": gradient_norms,
    }


def import_frozen_entrypoint(path: Path):
    module_name = "plangraph_engineering32_frozen_llada_sft"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PilotGateError(f"cannot import frozen entry point: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_instrumented_training(
    *,
    manifest: Mapping[str, Any],
    arm: str,
    arm_root: Path,
) -> dict[str, Any]:
    source_root = Path(
        manifest["execution_identity"]["source_install_root"]
    ).resolve()
    entrypoint = source_root / manifest["execution_identity"]["entrypoint"]["path"]
    require_equal(
        sha256_file(entrypoint),
        FROZEN_ENTRYPOINT_SHA256,
        "frozen training entrypoint SHA-256",
    )
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    import torch

    if torch.cuda.device_count() != 1:
        raise PilotGateError(
            f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}"
        )
    device_name = torch.cuda.get_device_name(0)
    if "A800" not in device_name:
        raise PilotGateError(f"engineering pilot requires A800, got {device_name!r}")
    data_seed = int(
        manifest["pilot_protocol"]["shared_training_args"]["data_seed"]
    )
    import random

    random.seed(data_seed)
    torch.manual_seed(data_seed)
    torch.cuda.manual_seed_all(data_seed)
    torch.cuda.reset_peak_memory_stats(0)

    sft = import_frozen_entrypoint(entrypoint)
    training_dir = arm_root / "training"
    gradient_log = arm_root / "gradient_norms.jsonl"
    instrumentation: dict[str, Any] = {
        "model_load_seconds": 0.0,
        "validation_seconds": 0.0,
        "checkpoint_save_seconds": 0.0,
        "final_model_save_seconds": 0.0,
        "train_compute_calls": 0,
        "optimizer_steps": 0,
        "data_seed": data_seed,
        "training_phase_start_monotonic": None,
        "last_optimizer_step_end_monotonic": None,
        "tokenizer_vocab_sha256": None,
    }

    original_load = sft.load_tokenizer_and_model
    original_evaluate = sft.evaluate
    original_save_checkpoint = sft.save_checkpoint
    original_save_model = sft.save_model_pretrained
    original_compute_loss = sft.compute_loss
    original_clip = torch.nn.utils.clip_grad_norm_
    original_adamw_step = torch.optim.AdamW.step
    inside_checkpoint_save = False

    def timed_load(*args: Any, **kwargs: Any):
        started = time.monotonic()
        result = original_load(*args, **kwargs)
        instrumentation["model_load_seconds"] += time.monotonic() - started
        tokenizer = result[0]
        vocab_sha = canonical_json_sha256(
            sorted(
                (str(token), int(token_id))
                for token, token_id in tokenizer.get_vocab().items()
            )
        )
        instrumentation["tokenizer_vocab_sha256"] = vocab_sha
        require_equal(
            vocab_sha,
            FROZEN_TOKENIZER_VOCAB_SHA256,
            "loaded tokenizer vocabulary SHA-256",
        )
        return result

    def timed_evaluate(*args: Any, **kwargs: Any):
        started = time.monotonic()
        try:
            return original_evaluate(*args, **kwargs)
        finally:
            instrumentation["validation_seconds"] += time.monotonic() - started

    def timed_compute_loss(*args: Any, **kwargs: Any):
        if torch.is_grad_enabled():
            if instrumentation["training_phase_start_monotonic"] is None:
                instrumentation["training_phase_start_monotonic"] = time.monotonic()
            instrumentation["train_compute_calls"] += 1
        return original_compute_loss(*args, **kwargs)

    def observed_clip(parameters: Any, max_norm: float, *args: Any, **kwargs: Any):
        norm = original_clip(parameters, max_norm, *args, **kwargs)
        scalar = float(norm.detach().cpu()) if hasattr(norm, "detach") else float(norm)
        with gradient_log.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "optimizer_step": len(load_jsonl(gradient_log)) + 1
                        if gradient_log.exists()
                        else 1,
                        "pre_clip_grad_norm": scalar,
                        "finite": math.isfinite(scalar),
                        "nonzero": scalar > 0.0,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        return norm

    def observed_adamw_step(optimizer: Any, *args: Any, **kwargs: Any):
        result = original_adamw_step(optimizer, *args, **kwargs)
        instrumentation["optimizer_steps"] += 1
        instrumentation["last_optimizer_step_end_monotonic"] = time.monotonic()
        return result

    def timed_checkpoint(*args: Any, **kwargs: Any):
        nonlocal inside_checkpoint_save
        started = time.monotonic()
        inside_checkpoint_save = True
        try:
            return original_save_checkpoint(*args, **kwargs)
        finally:
            inside_checkpoint_save = False
            instrumentation["checkpoint_save_seconds"] += time.monotonic() - started

    def timed_model_save(*args: Any, **kwargs: Any):
        if inside_checkpoint_save:
            return original_save_model(*args, **kwargs)
        started = time.monotonic()
        try:
            return original_save_model(*args, **kwargs)
        finally:
            instrumentation["final_model_save_seconds"] += time.monotonic() - started

    sft.load_tokenizer_and_model = timed_load
    sft.evaluate = timed_evaluate
    sft.compute_loss = timed_compute_loss
    sft.save_checkpoint = timed_checkpoint
    sft.save_model_pretrained = timed_model_save
    torch.nn.utils.clip_grad_norm_ = observed_clip
    torch.optim.AdamW.step = observed_adamw_step

    argv = build_training_argv(
        manifest=manifest,
        arm=arm,
        output_dir=training_dir,
    )
    argv[0] = str(entrypoint)
    previous_argv = sys.argv[:]
    main_started = time.monotonic()
    try:
        sys.argv = argv
        sft.main()
        torch.cuda.synchronize(0)
    finally:
        main_seconds = time.monotonic() - main_started
        sys.argv = previous_argv
        sft.load_tokenizer_and_model = original_load
        sft.evaluate = original_evaluate
        sft.compute_loss = original_compute_loss
        sft.save_checkpoint = original_save_checkpoint
        sft.save_model_pretrained = original_save_model
        torch.nn.utils.clip_grad_norm_ = original_clip
        torch.optim.AdamW.step = original_adamw_step

    phase_start = instrumentation["training_phase_start_monotonic"]
    phase_end = instrumentation["last_optimizer_step_end_monotonic"]
    if not isinstance(phase_start, float) or not isinstance(phase_end, float):
        raise PilotGateError("could not observe the training phase boundaries")
    training_seconds = phase_end - phase_start
    result = {
        **instrumentation,
        "training_seconds": training_seconds,
        "total_main_seconds": main_seconds,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "cuda_device_name": device_name,
        "torch_version": str(torch.__version__),
        "training_argv": argv[1:],
    }
    for key in (
        "model_load_seconds",
        "training_seconds",
        "validation_seconds",
        "total_main_seconds",
    ):
        value = float(result[key])
        if not math.isfinite(value) or value <= 0.0:
            raise PilotGateError(f"invalid timing {key}={value!r}")
    if result["peak_allocated_bytes"] <= 0 or result["peak_reserved_bytes"] <= 0:
        raise PilotGateError("CUDA peak-memory counters were not populated")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--arm-index", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require_equal(
        sha256_file(Path(__file__).resolve()),
        args.runner_sha256,
        "runner SHA-256",
    )
    require_equal(args.manifest_sha256, FROZEN_MANIFEST_SHA256, "manifest argument")
    require_equal(
        sha256_file(args.manifest),
        FROZEN_MANIFEST_SHA256,
        "manifest file SHA-256",
    )
    if args.arm_index < 0 or args.arm_index >= len(EXPECTED_ARMS):
        raise PilotGateError(f"arm index out of range: {args.arm_index}")
    arm = EXPECTED_ARMS[args.arm_index]

    manifest = read_json(args.manifest)
    authorization = read_json(args.authorization)
    validate_manifest(manifest)
    require_equal(
        sha256_file(args.authorization),
        args.authorization_sha256,
        "authorization SHA-256",
    )
    validate_authorization(
        authorization,
        manifest_sha256=FROZEN_MANIFEST_SHA256,
    )
    validate_slurm_environment(args.arm_index)

    execution = manifest["execution_identity"]
    run_root = Path(execution["run_root"]).resolve()
    project_root = Path(execution["project_root"]).resolve()
    try:
        run_root.relative_to(project_root / "runs")
    except ValueError as exc:
        raise PilotGateError(f"unsafe run root: {run_root}") from exc
    require_equal(
        run_root,
        Path(authorization["authorized_scope"]["run_root"]).resolve(),
        "authorized run root",
    )
    if not run_root.is_dir():
        raise PilotGateError(f"prepared run root is absent: {run_root}")
    arm_root = run_root / "arms" / arm
    arm_root.parent.mkdir(parents=True, exist_ok=True)
    arm_root.mkdir(exist_ok=False)

    report_path = arm_root / "engineering_report.json"
    report: dict[str, Any] = {
        "schema_version": "plangraph-dlm-engineering-arm-report@1",
        "identity": execution.get("identity", manifest["identity"]),
        "arm": arm,
        "arm_index": args.arm_index,
        "status": "running",
        "scientific_result": False,
        "checkpoint_selection": False,
        "automatic_downstream": False,
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "authorization_sha256": args.authorization_sha256,
        "runner_sha256": args.runner_sha256,
        "slurm": {
            "job_id": os.environ.get("SLURM_ARRAY_JOB_ID")
            or os.environ.get("SLURM_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "cpus_per_task": int(os.environ.get("SLURM_CPUS_PER_TASK", "0")),
        },
    }
    atomic_write_json(report_path, report)
    try:
        source_root = Path(execution["source_install_root"]).resolve()
        entrypoint = source_root / execution["entrypoint"]["path"]
        require_equal(
            sha256_file(entrypoint),
            FROZEN_ENTRYPOINT_SHA256,
            "entrypoint SHA-256",
        )
        adapter_file = (
            Path(manifest["frozen_model"]["initial_adapter_path"])
            / "adapter_model.safetensors"
        )
        require_equal(
            sha256_file(adapter_file),
            FROZEN_ADAPTER_SHA256,
            "initial adapter SHA-256",
        )
        dataset_evidence = verify_published_dataset(manifest)
        instrumentation = run_instrumented_training(
            manifest=manifest,
            arm=arm,
            arm_root=arm_root,
        )
        acceptance = validate_completed_training(
            training_log=arm_root / "training" / "training_log.jsonl",
            gradient_log=arm_root / "gradient_norms.jsonl",
            instrumentation=instrumentation,
        )
        report.update(
            {
                "status": "complete",
                "initial_adapter": {
                    "path": str(adapter_file),
                    "sha256": FROZEN_ADAPTER_SHA256,
                },
                "tokenizer_vocab_sha256": instrumentation[
                    "tokenizer_vocab_sha256"
                ],
                "dataset": dataset_evidence,
                "instrumentation": instrumentation,
                "acceptance": acceptance,
                "engineering_gate_passed": True,
                "eligible_for_scientific_metrics": False,
                "eligible_for_checkpoint_selection": False,
                "eligible_for_later_initialization": False,
                "automatic_downstream": False,
            }
        )
        atomic_write_json(report_path, report)
        print(f"plangraph_engineering32_arm={arm}")
        print("plangraph_engineering32_gate=PASS")
        print("plangraph_engineering32_automatic_downstream=false")
        return 0
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "engineering_gate_passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "eligible_for_scientific_metrics": False,
                "eligible_for_checkpoint_selection": False,
                "eligible_for_later_initialization": False,
                "automatic_downstream": False,
            }
        )
        atomic_write_json(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
