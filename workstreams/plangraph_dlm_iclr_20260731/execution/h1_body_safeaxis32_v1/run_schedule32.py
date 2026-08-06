#!/usr/bin/env python3
"""Run R03B: frozen H1 D1 versus the H1-preserving safe-axis schedule."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
for location in (RUNTIME, HERE):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from crystal_dlm.fixed_slot import MASK_TOKEN_ID  # noqa: E402
from crystal_dlm.h1a2_factorial_runtime import (  # noqa: E402
    assert_body_tokenizer_identity,
)
from crystal_dlm.r5_dynamic_length import (  # noqa: E402
    count_prefill_for_batch,
    exact_body_token_count,
    exact_dynamic_generation_schedule,
    exact_dynamic_schema_constraints,
    validate_answer_matches_plan,
)
from crystal_dlm.r5_plan_state import build_body_prompt  # noqa: E402
from paired_llada import generate_paired_exact_plan  # noqa: E402
from safe_axis_schedule import (  # noqa: E402
    analyze_axis_schedule,
    h1a2_safe_axis_generation_schedule,
    require_safe_axis_schedule,
)
from scripts.sample_llada_dynamic_crystals import (  # noqa: E402
    build_dynamic_lightweight_constraints,
    graph_from_arrays,
    import_process_one,
    load_model_and_tokenizer,
)
from scripts.sample_llada_r5_exact_length import (  # noqa: E402
    element_prefill_for_batch,
    merge_prefill_maps,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def require_sha(path: Path, expected: str, label: str) -> Path:
    location = path.resolve()
    if not location.is_file():
        raise FileNotFoundError(f"{label} is missing: {location}")
    observed = sha256_file(location)
    if observed != str(expected):
        raise ValueError(f"{label} SHA changed: {observed}")
    return location


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_exclusive(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def validate_runtime() -> torch.device:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("schedule screen must run through Slurm")
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu":
        raise RuntimeError("schedule screen requires the gpu partition")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 8:
        raise RuntimeError("schedule screen requires exactly eight CPUs")
    if os.environ.get("CONDA_DEFAULT_ENV") != "diff_meets_diff":
        raise RuntimeError("schedule screen requires diff_meets_diff")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("schedule screen requires exactly one CUDA device")
    name = torch.cuda.get_device_name(0)
    if "A800" not in name:
        raise RuntimeError(f"schedule screen requires A800, observed {name}")
    return torch.device("cuda", 0)


def load_tasks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    frozen = config["frozen_h1"]
    ledger_path = require_sha(
        Path(frozen["attempt_ledger"]),
        frozen["attempt_ledger_sha256"],
        "frozen H1 attempt ledger",
    )
    rows = read_jsonl(ledger_path)
    if (
        len(rows) != 256
        or [int(row.get("ordinal", -1)) for row in rows] != list(range(256))
    ):
        raise ValueError("frozen H1 attempt ledger order changed")

    denominator = int(config["denominator"])
    tasks: list[dict[str, Any]] = []
    eligible = 0
    for cell in rows[:denominator]:
        ordinal = int(cell["ordinal"])
        entry = cell["arms"]["P0"]
        if not bool(entry["body_eligible"]):
            tasks.append(
                {
                    "ordinal": ordinal,
                    "eligible": False,
                    "reason": str(entry["ineligible_reason"]),
                    "attempt_id": entry["attempt_id"],
                    "pair_id": cell["pair_id"],
                    "body_noise_seed": int(cell["body_noise_seed"]),
                    "plan_state_sha256": entry["plan_state_sha256"],
                }
            )
            continue
        eligible += 1
        plan_state = dict(entry["plan_state"])
        if canonical_sha256(plan_state) != entry["plan_state_sha256"]:
            raise ValueError(f"P0 plan-state identity changed at ordinal {ordinal}")
        prompt = build_body_prompt(plan_state).rstrip() + "\n"
        d1 = exact_dynamic_generation_schedule(int(plan_state["N"]))
        safe_axis = h1a2_safe_axis_generation_schedule(plan_state)
        control_invariant = analyze_axis_schedule(
            d1,
            num_atoms=int(plan_state["N"]),
        )
        candidate_invariant = require_safe_axis_schedule(
            safe_axis,
            num_atoms=int(plan_state["N"]),
        )
        tasks.append(
            {
                "ordinal": ordinal,
                "eligible": True,
                "attempt_id": entry["attempt_id"],
                "pair_id": cell["pair_id"],
                "body_noise_seed": int(cell["body_noise_seed"]),
                "plan_state": plan_state,
                "plan_state_sha256": entry["plan_state_sha256"],
                "body_prompt": prompt,
                "body_prompt_sha256": sha256_text(prompt),
                "control_schedule": d1,
                "control_schedule_sha256": canonical_sha256(d1),
                "candidate_schedule": safe_axis,
                "candidate_schedule_sha256": canonical_sha256(safe_axis),
                "control_schedule_invariant": control_invariant,
                "control_schedule_invariant_sha256": canonical_sha256(
                    control_invariant
                ),
                "candidate_schedule_invariant": candidate_invariant,
                "candidate_schedule_invariant_sha256": canonical_sha256(
                    candidate_invariant
                ),
                "schedule_changed": d1 != safe_axis,
            }
        )
    if eligible != int(frozen["expected_first32_body_eligible"]):
        raise ValueError("first-32 H1 body eligibility changed")
    if sum(task.get("schedule_changed") is True for task in tasks) != eligible:
        raise ValueError(
            "D2-safe-axis is not an active schedule treatment on every eligible row"
        )
    return tasks


def base_record(
    task: Mapping[str, Any],
    *,
    schedule_arm: str,
    generation_policy: str,
) -> dict[str, Any]:
    return {
        "schema": "h1_body_safeaxis32_attempt_v1",
        "ordinal": int(task["ordinal"]),
        "evaluation_order": int(task["ordinal"]),
        "sample_idx": int(task["ordinal"]),
        "planner_arm": "P0",
        "body_checkpoint_arm": "B0",
        "schedule_arm": schedule_arm,
        "generation_policy": generation_policy,
        "attempt_id": task["attempt_id"],
        "pair_id": task["pair_id"],
        "body_noise_seed": int(task["body_noise_seed"]),
        "plan_state_sha256": task["plan_state_sha256"],
        "body_prompt_sha256": task.get("body_prompt_sha256"),
        "control_schedule_sha256": task.get("control_schedule_sha256"),
        "candidate_schedule_sha256": task.get("candidate_schedule_sha256"),
        "schedule_invariant_sha256": task.get(
            "control_schedule_invariant_sha256"
            if generation_policy == "d1"
            else "candidate_schedule_invariant_sha256"
        ),
        "schedule_z_before_xy_count": (
            task.get("control_schedule_invariant", {}).get("z_before_xy_count")
            if generation_policy == "d1"
            else task.get("candidate_schedule_invariant", {}).get(
                "z_before_xy_count"
            )
        ),
        "schedule_all_xy_precede_all_z": (
            task.get("control_schedule_invariant", {}).get(
                "all_xy_precede_all_z"
            )
            if generation_policy == "d1"
            else task.get("candidate_schedule_invariant", {}).get(
                "all_xy_precede_all_z"
            )
        ),
        "schedule_changed": task.get("schedule_changed"),
        "status": "failed",
        "reason": "",
        "body_generation_complete": False,
        "body_plan_match": False,
        "body_graph_complete": False,
        "retry_used": False,
        "replacement_used": False,
        "repair_used": False,
        "filter_used": False,
        "rerank_used": False,
    }


def generate_paired_arms(
    *,
    config: Mapping[str, Any],
    tasks: list[dict[str, Any]],
    output_dir: Path,
    device: torch.device,
) -> None:
    body = config["body"]
    invariant_rows = [
        {
            "ordinal": int(task["ordinal"]),
            "control_schedule_sha256": task["control_schedule_sha256"],
            "control_invariant": task["control_schedule_invariant"],
            "control_invariant_sha256": task[
                "control_schedule_invariant_sha256"
            ],
            "candidate_schedule_sha256": task["candidate_schedule_sha256"],
            "candidate_invariant": task["candidate_schedule_invariant"],
            "candidate_invariant_sha256": task[
                "candidate_schedule_invariant_sha256"
            ],
        }
        for task in tasks
        if task["eligible"]
    ]
    if (
        len(invariant_rows) != int(config["denominator"])
        or any(
            row["candidate_invariant"]["gate_passed"] is not True
            or row["candidate_invariant"]["z_before_xy_count"] != 0
            or row["candidate_invariant"]["all_xy_precede_all_z"] is not True
            or row["candidate_invariant"]["mixed_axis_coordinate_groups"] != 0
            for row in invariant_rows
        )
    ):
        raise ValueError("safe-axis schedule invariant failed before model loading")
    invariant_report = {
        "schema": "h1_body_safeaxis32_invariant_report_v1",
        "status": "complete",
        "attempts": len(invariant_rows),
        "all_candidate_invariants_passed": True,
        "required_z_before_xy_count": 0,
        "rows": invariant_rows,
    }
    invariant_report_path = output_dir / "schedule_invariants.json"
    write_json_exclusive(invariant_report_path, invariant_report)
    checkpoint = Path(body["checkpoint"]).resolve()
    adapter = checkpoint / body["adapter_file"]
    if not adapter.is_file() or adapter.stat().st_size != int(
        body["adapter_expected_bytes"]
    ):
        raise ValueError("B0 adapter path or byte size changed")
    require_sha(
        checkpoint / "tokenizer.json",
        body["tokenizer_json_sha256"],
        "B0 tokenizer.json",
    )
    require_sha(
        checkpoint / "tokenizer_config.json",
        body["tokenizer_config_sha256"],
        "B0 tokenizer_config.json",
    )

    model, tokenizer = load_model_and_tokenizer(
        str(Path(body["base_model"]).resolve()),
        str(checkpoint),
        device,
    )
    if tokenizer.pad_token_id == MASK_TOKEN_ID:
        raise RuntimeError("tokenizer pad token collides with LLaDA mask token")
    tokenizer_identity = assert_body_tokenizer_identity(
        tokenizer,
        expected_vocab_sha256=body["tokenizer_vocab_sha256"],
    )
    if int(tokenizer_identity["vocab_size"]) != int(body["tokenizer_size"]):
        raise ValueError("B0 tokenizer size changed")
    write_json_exclusive(output_dir / "body_tokenizer_identity.json", tokenizer_identity)

    crysllmgen_dir = Path(
        "/public/home/jiaosz/ywliang/ai4s/"
        "diffsion_language_model_meets_diffusion/reference/crysllmgen"
    )
    process_one = import_process_one(crysllmgen_dir)
    lightweight = build_dynamic_lightweight_constraints(
        tokenizer,
        duplicate_coordinate_mask=bool(body["duplicate_coordinate_mask"]),
        lattice_volume_mask=bool(body["lattice_volume_mask"]),
        min_lattice_rad=float(body["min_lattice_rad"]),
    )

    buckets: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        if not task["eligible"]:
            continue
        key = (int(task["plan_state"]["N"]), str(task["candidate_schedule_sha256"]))
        buckets[key].append(task)

    max_batch = int(body["max_batch_size"])
    started = time.monotonic()
    shared_batches: list[list[dict[str, Any]]] = []
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda item: int(item["ordinal"]))
        for offset in range(0, len(bucket), max_batch):
            shared_batches.append(bucket[offset : offset + max_batch])
    shared_partition = [
        [int(item["ordinal"]) for item in batch] for batch in shared_batches
    ]
    write_json_exclusive(
        output_dir / "shared_batch_partition.json",
        {
            "schema": "h1_body_safeaxis32_shared_batch_partition_v1",
            "partition": shared_partition,
            "sha256": canonical_sha256(shared_partition),
            "applied_identically_to_arms": ["D1", "D2_SAFE_AXIS"],
        },
    )

    arm_reports: dict[str, dict[str, Any]] = {}
    for label, generation_policy, schedule_key in (
        ("control", "d1", "control_schedule"),
        ("candidate", "d2_safe_axis", "candidate_schedule"),
    ):
        schedule_arm = generation_policy.upper()
        attempts: dict[int, dict[str, Any]] = {}
        proposal_graphs: list[dict[str, Any]] = []
        for task in tasks:
            if task["eligible"]:
                continue
            record = base_record(
                task,
                schedule_arm=schedule_arm,
                generation_policy=generation_policy,
            )
            record.update(
                {
                    "status": "failed",
                    "reason": f"planner:{task['reason']}",
                    "earliest_failure_stage": "planner",
                }
            )
            attempts[int(task["ordinal"])] = record

        progress = tqdm(
            total=sum(len(batch) for batch in shared_batches),
            desc=f"H1 P0+B0 {schedule_arm} schedule32",
        )
        for batch in shared_batches:
            schedule = batch[0][schedule_key]
            if any(item[schedule_key] != schedule for item in batch):
                raise ValueError(f"{schedule_arm} schedule batch is not homogeneous")
            num_atoms = int(batch[0]["plan_state"]["N"])
            prompts = [str(item["body_prompt"]) for item in batch]
            encoded = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(model_device(model))
            attention_mask = encoded["attention_mask"].to(model_device(model))
            gen_length = exact_body_token_count(num_atoms)
            allowed = exact_dynamic_schema_constraints(tokenizer, num_atoms)
            prefill = merge_prefill_maps(
                count_prefill_for_batch(tokenizer, num_atoms, len(batch)),
                element_prefill_for_batch(
                    tokenizer,
                    [item["plan_state"] for item in batch],
                ),
            )
            outputs = generate_paired_exact_plan(
                model,
                input_ids,
                base_seeds=[int(item["body_noise_seed"]) for item in batch],
                attention_mask=attention_mask,
                gen_length=gen_length,
                temperature=float(body["temperature"]),
                cfg_scale=float(body["cfg_scale"]),
                remasking=str(body["remasking"]),
                mask_id=MASK_TOKEN_ID,
                allowed_token_ids_by_generation_pos=allowed,
                prefill_token_ids_by_generation_pos=prefill,
                generation_position_groups=schedule,
                lightweight_decoding_constraints=lightweight,
            )
            generated = outputs[:, input_ids.shape[1] :]
            texts = tokenizer.batch_decode(
                generated,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for task, token_ids, text in zip(batch, generated, texts, strict=True):
                ordinal = int(task["ordinal"])
                record = base_record(
                    task,
                    schedule_arm=schedule_arm,
                    generation_policy=generation_policy,
                )
                stage = "body_parse"
                record.update(
                    {
                        "body_generation_complete": True,
                        "raw_body_token_ids": [
                            int(value) for value in token_ids.detach().cpu().tolist()
                        ],
                        "text": text,
                        "raw_body_text_sha256": sha256_text(text),
                    }
                )
                try:
                    arrays = validate_answer_matches_plan(task["plan_state"], text)
                    record["body_plan_match"] = True
                    stage = "body_graph"
                    graph, cif = graph_from_arrays(arrays, process_one)
                    graph["h1_safeaxis32_metadata"] = {
                        "ordinal": ordinal,
                        "schedule_arm": schedule_arm,
                        "generation_policy": generation_policy,
                        "body_noise_seed": int(task["body_noise_seed"]),
                        "plan_state_sha256": task["plan_state_sha256"],
                        "body_prompt_sha256": task["body_prompt_sha256"],
                        "candidate_schedule_sha256": task[
                            "candidate_schedule_sha256"
                        ],
                    }
                    proposal_graphs.append(
                        {
                            "ordinal": ordinal,
                            "attempt_id": task["attempt_id"],
                            "graph": graph,
                        }
                    )
                    record.update(
                        {
                            "status": "succeeded",
                            "reason": "",
                            "earliest_failure_stage": None,
                            "body_graph_complete": True,
                            "arrays": arrays,
                            "proposal_cif_sha256": canonical_sha256(cif),
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    record.update(
                        {
                            "status": "failed",
                            "reason": f"body:{type(exc).__name__}:{exc}",
                            "earliest_failure_stage": stage,
                            "arrays": None,
                        }
                    )
                attempts[ordinal] = record
                progress.update(1)
        progress.close()

        denominator = int(config["denominator"])
        if sorted(attempts) != list(range(denominator)):
            raise ValueError(f"{schedule_arm} output lost or duplicated an ordinal")
        ordered = [attempts[index] for index in range(denominator)]
        attempts_path = output_dir / f"{label}_body_attempts.jsonl"
        graphs_path = output_dir / f"{label}_proposal_graphs.pt"
        write_jsonl_exclusive(attempts_path, ordered)
        with graphs_path.open("xb") as handle:
            torch.save(
                sorted(proposal_graphs, key=lambda row: int(row["ordinal"])),
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())

        failures: dict[str, int] = {}
        for record in ordered:
            if record["status"] == "succeeded":
                continue
            reason = str(record.get("reason") or "unknown")
            failure = ":".join(reason.split(":")[:2])
            failures[failure] = failures.get(failure, 0) + 1
        arm_reports[label] = {
            "schedule_arm": schedule_arm,
            "generation_policy": generation_policy,
            "attempts": denominator,
            "succeeded": sum(row["status"] == "succeeded" for row in ordered),
            "failed": sum(row["status"] != "succeeded" for row in ordered),
            "failure_classes": failures,
            "schedule_treatment_applied": (
                sum(row.get("schedule_changed") is True for row in ordered)
                if label == "candidate"
                else 0
            ),
            "body_attempts_sha256": sha256_file(attempts_path),
            "proposal_graphs_sha256": sha256_file(graphs_path),
        }

    report = {
        "schema": "h1_body_safeaxis32_generation_report_v1",
        "status": "complete",
        "attempts_per_arm": int(config["denominator"]),
        "arms": arm_reports,
        "shared_batch_partition_sha256": canonical_sha256(shared_partition),
        "shared_batch_partition_applied_identically": True,
        "schedule_invariants_sha256": sha256_file(invariant_report_path),
        "all_candidate_invariants_passed": True,
        "walltime_s": time.monotonic() - started,
        "body_checkpoint": str(checkpoint),
        "body_adapter_sha256_recorded": body["adapter_sha256"],
        "generation_policies": ["d1", "d2_safe_axis"],
        "max_batch_size": max_batch,
        "effective_batching": (
            "shared_homogeneous_candidate_schedule_signature_up_to_8"
        ),
        "retry_replacement_repair_filter_rerank": False,
        "refinement_run": False,
        "direct_metrics_run": False,
        "sun_run": False,
        "automatic_downstream": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json_exclusive(output_dir / "generation_report.json", report)
    del model, tokenizer, process_one
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    device = validate_runtime()
    config = read_json(args.config.resolve())
    if (
        config.get("schema") != "h1_body_safeaxis32_config_v1"
        or int(config.get("denominator", -1)) != 32
        or config.get("automatic_downstream") is not False
        or config["treatment"].get("candidate_policy") != "d2_safe_axis"
        or config["body"].get("exact_length_generation") is not True
        or config["body"].get("answer_token_count_formula") != "7+4N"
    ):
        raise ValueError("schedule-screen configuration changed")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    tasks = load_tasks(config)
    generate_paired_arms(config=config, tasks=tasks, output_dir=output, device=device)


if __name__ == "__main__":
    main()
