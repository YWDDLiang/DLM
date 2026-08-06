#!/usr/bin/env python3
"""Sample R5 plan-state JSON proposals from a LLaDA checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from tqdm import tqdm

from crystal_dlm.composition_validity import classify_smact_validity  # noqa: E402
from crystal_dlm.fixed_slot import MASK_TOKEN_ID, SYMBOL_TO_Z, write_json  # noqa: E402
from crystal_dlm.llada_generation import generate  # noqa: E402
from crystal_dlm.r5_plan_state import (  # noqa: E402
    build_plan_prompt,
    parse_plan_state_json,
    plan_state_to_json,
    validate_plan_state,
)
from scripts.sample_llada_dynamic_crystals import (  # noqa: E402
    init_distributed,
    load_model_and_tokenizer,
    rank_path,
)


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(1.0, float(denominator))


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def load_generation_length(stats_json: Path | None, explicit: int | None, padding: int) -> int:
    if explicit is not None:
        return int(explicit)
    if stats_json is None or not stats_json.exists():
        return 256
    stats = json.loads(stats_json.read_text(encoding="utf-8"))
    if stats.get("max_answer_model_length"):
        return max(32, int(stats["max_answer_model_length"]) + int(padding))
    if stats.get("answer_token_count"):
        return max(32, int(stats["answer_token_count"]))
    return 256


def smact_record(plan: Mapping[str, Any]) -> Dict[str, Any]:
    elements = plan.get("elements")
    counts = plan.get("counts")
    if not isinstance(elements, list) or not isinstance(counts, list) or len(elements) != len(counts):
        return {"valid": False, "reason": "invalid_elements_or_counts"}
    try:
        symbols = [str(symbol) for symbol in elements]
        elem_ids = tuple(SYMBOL_TO_Z[symbol] for symbol in symbols)
        count_values = tuple(int(value) for value in counts)
    except Exception:
        return {"valid": False, "reason": "unsupported_element_or_count"}
    try:
        payload = classify_smact_validity(elem_ids, count_values)
        return dict(payload)
    except Exception as exc:  # noqa: BLE001 - SMACT can be absent in light local envs.
        return {"valid": None, "reason": "validator_unavailable", "validator_error": type(exc).__name__}


def update_plan_metrics(metrics: Dict[str, Any], plan: Mapping[str, Any], validation: Mapping[str, Any], smact: Mapping[str, Any]) -> None:
    metrics["valid_N"] += int(bool(validation.get("valid_N")))
    metrics["valid_formula"] += int(bool(validation.get("valid_formula")))
    metrics["valid_counts"] += int(bool(validation.get("valid_counts")))
    metrics["valid_elements"] += int(bool(validation.get("valid_elements")))
    metrics["valid_plan"] += int(bool(validation.get("valid")))
    metrics["smact_plausible"] += int(smact.get("valid") is True)
    metrics["smact_checked"] += int(smact.get("valid") is not None)

    elements = plan.get("elements")
    counts = plan.get("counts")
    if isinstance(elements, list):
        metrics["single_element"] += int(len(elements) == 1)
        if smact.get("reason") == "all_metal_shortcut":
            metrics["all_metal"] += 1
    if isinstance(counts, list):
        n_from_counts = sum(int(value) for value in counts if isinstance(value, int))
        if n_from_counts > 0:
            metrics["counts_n_histogram"][str(n_from_counts)] += 1

    n_value = int_or_none(plan.get("N"))
    if n_value is not None:
        metrics["n_histogram"][str(n_value)] += 1
    formula = plan.get("formula")
    if isinstance(formula, str) and formula:
        metrics["formula_histogram"][formula] += 1
    prototype = plan.get("prototype_key")
    if isinstance(prototype, str) and prototype:
        metrics["prototype_histogram"][prototype] += 1
    for field_name in ("charge_bucket", "anion_framework", "lattice_system", "spacegroup_bucket", "volume_per_atom_bin"):
        field_value = plan.get(field_name)
        if isinstance(field_value, str) and field_value:
            metrics[f"{field_name}_histogram"][field_value] += 1


def add_failure(metrics: Dict[str, Any], failure_handle, sample_idx: int, stage: str, exc: Exception, text: str) -> None:
    reason = f"{stage}:{type(exc).__name__}"
    metrics["failures"][reason] = metrics["failures"].get(reason, 0) + 1
    failure_handle.write(
        json.dumps(
            {
                "sample_idx": sample_idx,
                "stage": stage,
                "reason": type(exc).__name__,
                "message": str(exc),
                "text_prefix": text[:512],
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def histogram_top(counter: Counter[str], limit: int = 100) -> Dict[str, int]:
    return {key: int(value) for key, value in counter.most_common(limit)}


def finalize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    decoded = int(metrics.get("decoded_samples") or 0)
    parsed = int(metrics.get("parse_success") or 0)
    smact_checked = int(metrics.get("smact_checked") or 0)
    formula_counter = metrics.get("formula_histogram") or Counter()
    prototype_counter = metrics.get("prototype_histogram") or Counter()
    n_counter = metrics.get("n_histogram") or Counter()
    formula_values = list(formula_counter.values())
    prototype_values = list(prototype_counter.values())
    n_values = list(n_counter.values())

    for key in (
        "n_histogram",
        "counts_n_histogram",
        "formula_histogram",
        "prototype_histogram",
        "charge_bucket_histogram",
        "anion_framework_histogram",
        "lattice_system_histogram",
        "spacegroup_bucket_histogram",
        "volume_per_atom_bin_histogram",
    ):
        counter = metrics.get(key)
        if isinstance(counter, Counter):
            metrics[key] = histogram_top(counter)

    metrics.update(
        {
            "parse_rate": rate(parsed, decoded),
            "valid_N_rate": rate(metrics.get("valid_N", 0), parsed),
            "valid_formula_rate": rate(metrics.get("valid_formula", 0), parsed),
            "valid_counts_rate": rate(metrics.get("valid_counts", 0), parsed),
            "valid_elements_rate": rate(metrics.get("valid_elements", 0), parsed),
            "valid_plan_rate": rate(metrics.get("valid_plan", 0), parsed),
            "smact_plausible_rate": rate(metrics.get("smact_plausible", 0), smact_checked),
            "single_element_rate": rate(metrics.get("single_element", 0), parsed),
            "all_metal_rate": rate(metrics.get("all_metal", 0), parsed),
            "unique_formula_count": len(formula_counter),
            "unique_prototype_count": len(prototype_counter),
            "top_formula_fraction": rate(max(formula_values or [0]), parsed),
            "top_prototype_fraction": rate(max(prototype_values or [0]), parsed),
            "n_max_fraction": rate(max(n_values or [0]), parsed),
        }
    )
    return metrics


def empty_metrics(args, rank: int, world_size: int) -> Dict[str, Any]:
    return {
        "requested_samples": 0,
        "decoded_samples": 0,
        "parse_success": 0,
        "valid_N": 0,
        "valid_formula": 0,
        "valid_counts": 0,
        "valid_elements": 0,
        "valid_plan": 0,
        "smact_plausible": 0,
        "smact_checked": 0,
        "single_element": 0,
        "all_metal": 0,
        "failures": {},
        "time_sec": 0.0,
        "rank": rank,
        "world_size": world_size,
        "generation_length": int(args.generation_length),
        "n_histogram": Counter(),
        "counts_n_histogram": Counter(),
        "formula_histogram": Counter(),
        "prototype_histogram": Counter(),
        "charge_bucket_histogram": Counter(),
        "anion_framework_histogram": Counter(),
        "lattice_system_histogram": Counter(),
        "spacegroup_bucket_histogram": Counter(),
        "volume_per_atom_bin_histogram": Counter(),
    }


def merge_counter_payload(target: Counter[str], payload: Mapping[str, Any]) -> None:
    for key, value in payload.items():
        target[str(key)] += int(value)


def merge_distributed_outputs(output_dir: Path, world_size: int, args) -> None:
    merged = empty_metrics(args, rank=0, world_size=world_size)
    merged["distributed"] = True
    merged["world_size"] = world_size
    with (output_dir / "raw_generations.jsonl").open("w", encoding="utf-8") as raw_out, (
        output_dir / "failure_cases.jsonl"
    ).open("w", encoding="utf-8") as failure_out:
        for rank in range(world_size):
            metrics_path = rank_path(output_dir, "sample_metrics.json", rank, True)
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                for key in (
                    "requested_samples",
                    "decoded_samples",
                    "parse_success",
                    "valid_N",
                    "valid_formula",
                    "valid_counts",
                    "valid_elements",
                    "valid_plan",
                    "smact_plausible",
                    "smact_checked",
                    "single_element",
                    "all_metal",
                ):
                    merged[key] += int(metrics.get(key, 0))
                merged["time_sec"] = max(float(merged["time_sec"]), float(metrics.get("time_sec") or 0.0))
                for reason, count in (metrics.get("failures") or {}).items():
                    merged["failures"][reason] = int(merged["failures"].get(reason, 0)) + int(count)
                for hist_key in (
                    "n_histogram",
                    "counts_n_histogram",
                    "formula_histogram",
                    "prototype_histogram",
                    "charge_bucket_histogram",
                    "anion_framework_histogram",
                    "lattice_system_histogram",
                    "spacegroup_bucket_histogram",
                    "volume_per_atom_bin_histogram",
                ):
                    merge_counter_payload(merged[hist_key], metrics.get(hist_key) or {})
            for filename, handle in (("raw_generations.jsonl", raw_out), ("failure_cases.jsonl", failure_out)):
                path = rank_path(output_dir, filename, rank, True)
                if path.exists():
                    handle.write(path.read_text(encoding="utf-8"))
    write_json(str(output_dir / "sample_metrics.json"), finalize_metrics(merged))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--gen-length", type=int, default=None)
    parser.add_argument("--gen-length-padding", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--block-length", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260529)
    args = parser.parse_args()
    args.generation_length = load_generation_length(args.stats_json, args.gen_length, args.gen_length_padding)
    args.steps = int(args.steps or args.generation_length)

    dist_info = init_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    distributed = dist_info["distributed"]
    is_main = dist_info["is_main"]
    random.seed(int(args.seed) + rank)
    torch.manual_seed(int(args.seed) + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed) + rank)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.checkpoint_path, dist_info["device"])
    prompt = build_plan_prompt().rstrip() + "\n"
    tasks = [sample_idx for sample_idx in range(args.num_samples) if sample_idx % world_size == rank]

    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config.update({"representation": "r5_plan_state", "distributed": distributed, "world_size": world_size})
    if is_main:
        write_json(str(args.output_dir / "run_config.json"), run_config)
        write_json(
            str(args.output_dir / "tokenizer_report.json"),
            {
                "vocab_size": len(tokenizer),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "mask_token_id": MASK_TOKEN_ID,
            },
        )

    raw_path = rank_path(args.output_dir, "raw_generations.jsonl", rank, distributed)
    failure_path = rank_path(args.output_dir, "failure_cases.jsonl", rank, distributed)
    metrics = empty_metrics(args, rank=rank, world_size=world_size)
    metrics["requested_samples"] = len(tasks)

    start = time.time()
    with raw_path.open("w", encoding="utf-8") as raw_handle, failure_path.open("w", encoding="utf-8") as failure_handle:
        progress = tqdm(total=len(tasks), desc=f"R5-D plan sampling rank{rank}", disable=distributed and not is_main)
        offset = 0
        while offset < len(tasks):
            batch_indices = tasks[offset : offset + int(args.batch_size)]
            offset += len(batch_indices)
            prompts = [prompt for _ in batch_indices]
            encoded = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(model_device(model))
            attention_mask = encoded["attention_mask"].to(model_device(model))
            outputs = generate(
                model,
                input_ids,
                attention_mask=attention_mask,
                steps=int(args.steps),
                gen_length=int(args.generation_length),
                block_length=int(args.block_length),
                temperature=float(args.temperature),
                cfg_scale=float(args.cfg_scale),
                remasking=args.remasking,
                mask_id=MASK_TOKEN_ID,
            )
            generated_ids = outputs[:, input_ids.shape[1] :]
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            for sample_idx, text in zip(batch_indices, decoded):
                metrics["decoded_samples"] += 1
                record: Dict[str, Any] = {
                    "sample_idx": int(sample_idx),
                    "text": text,
                    "representation": "r5_plan_state",
                    "conditioning_prompt": prompt.rstrip(),
                }
                try:
                    plan = parse_plan_state_json(text)
                    validation = validate_plan_state(plan).to_dict()
                    smact = smact_record(plan)
                    metrics["parse_success"] += 1
                    update_plan_metrics(metrics, plan, validation, smact)
                    record.update(
                        {
                            "parsed": True,
                            "plan_state": plan,
                            "canonical_plan_json": plan_state_to_json(plan) if validation.get("valid") else None,
                            "plan_validation": validation,
                            "smact": smact,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    add_failure(metrics, failure_handle, int(sample_idx), "parse_or_validate", exc, text)
                    record.update({"parsed": False, "reason": type(exc).__name__, "message": str(exc)})
                raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress.update(1)
        progress.close()

    metrics["time_sec"] = time.time() - start
    write_json(str(rank_path(args.output_dir, "sample_metrics.json", rank, distributed)), finalize_metrics(metrics))
    if distributed:
        dist.barrier()
        if is_main:
            merge_distributed_outputs(args.output_dir, world_size, args)
        dist.barrier()


if __name__ == "__main__":
    main()
