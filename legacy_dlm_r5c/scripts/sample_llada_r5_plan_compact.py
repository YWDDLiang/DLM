#!/usr/bin/env python3
"""Sample compact/canonical R5 plan-state proposals from a LLaDA checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from tqdm import tqdm

from crystal_dlm.fixed_slot import MASK_TOKEN_ID, write_json  # noqa: E402
from crystal_dlm.llada_generation import generate  # noqa: E402
from crystal_dlm.r5_plan_state import (  # noqa: E402
    build_atomfields_plan_prompt,
    build_atomseq_plan_prompt,
    build_atomslots_plan_prompt,
    build_compact_plan_prompt,
    build_countfields_plan_prompt,
    build_countvalence_plan_prompt,
    parse_atomfields_plan_state,
    parse_atomseq_plan_state,
    parse_atomslots_plan_state,
    parse_compact_plan_state,
    parse_countfields_plan_state,
    parse_countvalence_plan_state,
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


def load_generation_length(stats_json: Path | None, explicit: int | None, padding: int) -> int:
    if explicit is not None:
        return int(explicit)
    if stats_json is None or not stats_json.exists():
        return 96
    stats = json.loads(stats_json.read_text(encoding="utf-8"))
    if stats.get("max_answer_model_length"):
        return max(32, int(stats["max_answer_model_length"]) + int(padding))
    if stats.get("answer_token_count"):
        return max(32, int(stats["answer_token_count"]))
    return 96


def empty_metrics(args, rank: int, world_size: int) -> Dict[str, Any]:
    return {
        "requested_samples": 0,
        "decoded_samples": 0,
        "attempted_decodes": 0,
        "accepted_samples": 0,
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
        "max_attempt_factor": int(args.max_attempt_factor),
        "formula_cap": int(args.formula_cap),
        "n_cap_fraction": float(args.n_cap_fraction),
        "n_histogram": Counter(),
        "formula_histogram": Counter(),
        "prototype_histogram": Counter(),
        "charge_bucket_histogram": Counter(),
        "anion_framework_histogram": Counter(),
        "lattice_system_histogram": Counter(),
        "spacegroup_bucket_histogram": Counter(),
        "volume_per_atom_bin_histogram": Counter(),
    }


def add_failure(metrics: Dict[str, Any], reason: str, count: int = 1) -> None:
    metrics["failures"][reason] = int(metrics["failures"].get(reason, 0)) + int(count)


def plan_is_strict_plausible(plan: Mapping[str, Any]) -> bool:
    validator = plan.get("validator") or {}
    return validator.get("valid") is True and str(validator.get("reason")) == "charge_neutral_pauling_valid"


def accept_reasons(
    *,
    plan: Mapping[str, Any],
    validation: Mapping[str, Any],
    formula_counts: Counter[str],
    n_counts: Counter[str],
    rank_target: int,
    args,
) -> list[str]:
    reasons: list[str] = []
    if not validation.get("valid"):
        reasons.append("invalid_canonical_plan")
    if args.require_strict_smact and not plan_is_strict_plausible(plan):
        reasons.append("non_strict_smact")
    if len(plan.get("elements") or []) <= 1:
        reasons.append("single_element")
    if str((plan.get("validator") or {}).get("reason")) == "all_metal_shortcut":
        reasons.append("all_metal")
    formula = str(plan.get("formula") or "")
    if args.formula_cap > 0 and formula_counts[formula] >= int(args.formula_cap):
        reasons.append("formula_cap")
    n_key = str(plan.get("N"))
    n_cap = max(1, int(float(args.n_cap_fraction) * float(max(1, rank_target))))
    if n_counts[n_key] >= n_cap:
        reasons.append("n_cap")
    return reasons


def update_accepted_metrics(metrics: Dict[str, Any], plan: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    metrics["decoded_samples"] += 1
    metrics["accepted_samples"] += 1
    metrics["parse_success"] += 1
    metrics["valid_N"] += int(bool(validation.get("valid_N")))
    metrics["valid_formula"] += int(bool(validation.get("valid_formula")))
    metrics["valid_counts"] += int(bool(validation.get("valid_counts")))
    metrics["valid_elements"] += int(bool(validation.get("valid_elements")))
    metrics["valid_plan"] += int(bool(validation.get("valid")))
    metrics["smact_checked"] += 1
    metrics["smact_plausible"] += int((plan.get("validator") or {}).get("valid") is True)
    metrics["single_element"] += int(len(plan.get("elements") or []) <= 1)
    metrics["all_metal"] += int(str((plan.get("validator") or {}).get("reason")) == "all_metal_shortcut")

    metrics["n_histogram"][str(plan.get("N"))] += 1
    metrics["formula_histogram"][str(plan.get("formula"))] += 1
    metrics["prototype_histogram"][str(plan.get("prototype_key"))] += 1
    for field_name in ("charge_bucket", "anion_framework", "lattice_system", "spacegroup_bucket", "volume_per_atom_bin"):
        metrics[f"{field_name}_histogram"][str(plan.get(field_name))] += 1


def update_direct_metrics(metrics: Dict[str, Any], plan: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    metrics["parse_success"] += 1
    metrics["valid_N"] += int(bool(validation.get("valid_N")))
    metrics["valid_formula"] += int(bool(validation.get("valid_formula")))
    metrics["valid_counts"] += int(bool(validation.get("valid_counts")))
    metrics["valid_elements"] += int(bool(validation.get("valid_elements")))
    metrics["valid_plan"] += int(bool(validation.get("valid")))
    metrics["smact_checked"] += 1
    metrics["smact_plausible"] += int((plan.get("validator") or {}).get("valid") is True)
    metrics["single_element"] += int(len(plan.get("elements") or []) <= 1)
    metrics["all_metal"] += int(str((plan.get("validator") or {}).get("reason")) == "all_metal_shortcut")

    metrics["n_histogram"][str(plan.get("N"))] += 1
    metrics["formula_histogram"][str(plan.get("formula"))] += 1
    metrics["prototype_histogram"][str(plan.get("prototype_key"))] += 1
    for field_name in ("charge_bucket", "anion_framework", "lattice_system", "spacegroup_bucket", "volume_per_atom_bin"):
        metrics[f"{field_name}_histogram"][str(plan.get(field_name))] += 1


def histogram_top(counter: Counter[str], limit: int = 100) -> Dict[str, int]:
    return {key: int(value) for key, value in counter.most_common(limit)}


def finalize_metrics(metrics: Dict[str, Any], histogram_limit: int | None = 100) -> Dict[str, Any]:
    decoded = int(metrics.get("decoded_samples") or 0)
    parsed = int(metrics.get("parse_success") or 0)
    attempted = int(metrics.get("attempted_decodes") or 0)
    formula_counter = metrics.get("formula_histogram") or Counter()
    prototype_counter = metrics.get("prototype_histogram") or Counter()
    n_counter = metrics.get("n_histogram") or Counter()
    formula_values = list(formula_counter.values())
    prototype_values = list(prototype_counter.values())
    n_values = list(n_counter.values())

    if histogram_limit is not None:
        for key in (
            "n_histogram",
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
                metrics[key] = histogram_top(counter, limit=int(histogram_limit))

    metrics.update(
        {
            "parse_rate": rate(parsed, decoded),
            "attempt_acceptance_rate": rate(decoded, attempted),
            "valid_N_rate": rate(metrics.get("valid_N", 0), parsed),
            "valid_formula_rate": rate(metrics.get("valid_formula", 0), parsed),
            "valid_counts_rate": rate(metrics.get("valid_counts", 0), parsed),
            "valid_elements_rate": rate(metrics.get("valid_elements", 0), parsed),
            "valid_plan_rate": rate(metrics.get("valid_plan", 0), parsed),
            "smact_plausible_rate": rate(metrics.get("smact_plausible", 0), metrics.get("smact_checked", 0)),
            "single_element_rate": rate(metrics.get("single_element", 0), parsed),
            "all_metal_rate": rate(metrics.get("all_metal", 0), parsed),
            "unique_formula_count": len(formula_counter),
            "unique_prototype_count": len(prototype_counter),
            "top_formula_fraction": rate(max(formula_values or [0]), parsed),
            "top_prototype_fraction": rate(max(prototype_values or [0]), parsed),
            "n_max_fraction": rate(max(n_values or [0]), parsed),
            "target_reached": decoded >= int(metrics.get("requested_samples") or 0),
        }
    )
    return metrics


def merge_counter_payload(target: Counter[str], payload: Mapping[str, Any]) -> None:
    for key, value in payload.items():
        target[str(key)] += int(value)


def merge_distributed_outputs(output_dir: Path, world_size: int, args) -> None:
    merged = empty_metrics(args, rank=0, world_size=world_size)
    merged["distributed"] = True
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
                    "attempted_decodes",
                    "accepted_samples",
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


def plan_prompt_and_parser(plan_format: str):
    if plan_format == "countvalence":
        return (
            build_countvalence_plan_prompt().rstrip() + "\n",
            parse_countvalence_plan_state,
            "r5_plan_state_countvalence",
        )
    if plan_format == "countfields":
        return build_countfields_plan_prompt().rstrip() + "\n", parse_countfields_plan_state, "r5_plan_state_countfields"
    if plan_format == "atomfields":
        return build_atomfields_plan_prompt().rstrip() + "\n", parse_atomfields_plan_state, "r5_plan_state_atomfields"
    if plan_format == "atomslots":
        return build_atomslots_plan_prompt().rstrip() + "\n", parse_atomslots_plan_state, "r5_plan_state_atomslots"
    if plan_format == "atomseq":
        return build_atomseq_plan_prompt().rstrip() + "\n", parse_atomseq_plan_state, "r5_plan_state_atomseq"
    if plan_format == "compact":
        return build_compact_plan_prompt().rstrip() + "\n", parse_compact_plan_state, "r5_plan_state_compact"
    raise ValueError(f"Unsupported plan format {plan_format!r}")


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
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument(
        "--plan-format",
        choices=["compact", "atomseq", "atomslots", "atomfields", "countfields", "countvalence"],
        default="compact",
    )
    parser.add_argument("--max-attempt-factor", type=int, default=12)
    parser.add_argument("--formula-cap", type=int, default=0)
    parser.add_argument("--n-cap-fraction", type=float, default=1.0)
    parser.add_argument("--require-strict-smact", action="store_true")
    parser.add_argument(
        "--direct-samples",
        action="store_true",
        help="Generate exactly num-samples once each and count invalid outputs instead of reject-until-accepted sampling.",
    )
    args = parser.parse_args()
    if not args.direct_samples:
        raise SystemExit(
            "Reject-until-accepted plan sampling is disabled for R5 de novo experiments; "
            "use --direct-samples so every requested sample is generated exactly once."
        )
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
    prompt, parse_plan_state, representation = plan_prompt_and_parser(args.plan_format)
    tasks = [sample_idx for sample_idx in range(args.num_samples) if sample_idx % world_size == rank]
    target = len(tasks)
    max_attempts = max(1, target * int(args.max_attempt_factor))

    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config.update({"representation": representation, "distributed": distributed, "world_size": world_size})
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
    metrics["requested_samples"] = target
    formula_counts: Counter[str] = Counter()
    n_counts: Counter[str] = Counter()

    start = time.time()
    accepted = 0
    with raw_path.open("w", encoding="utf-8") as raw_handle, failure_path.open("w", encoding="utf-8") as failure_handle:
        progress = tqdm(total=target, desc=f"R5 plan {args.plan_format} sampling rank{rank}", disable=distributed and not is_main)
        if args.direct_samples:
            for offset in range(0, len(tasks), int(args.batch_size)):
                batch_tasks = tasks[offset : offset + int(args.batch_size)]
                prompts = [prompt for _ in batch_tasks]
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
                for sample_idx, text in zip(batch_tasks, decoded):
                    metrics["attempted_decodes"] += 1
                    metrics["decoded_samples"] += 1
                    record: Dict[str, Any] = {
                        "sample_idx": int(sample_idx),
                        "attempt_idx": int(metrics["attempted_decodes"]),
                        "text": text,
                        "parsed": False,
                        "representation": representation,
                        "conditioning_prompt": prompt.rstrip(),
                    }
                    try:
                        plan = parse_plan_state(text)
                        validation = validate_plan_state(plan).to_dict()
                        update_direct_metrics(metrics, plan, validation)
                        record.update(
                            {
                                "parsed": True,
                                "plan_state": plan,
                                "canonical_plan_json": plan_state_to_json(plan),
                                "plan_validation": validation,
                                "smact": plan.get("validator"),
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        add_failure(metrics, f"parse:{type(exc).__name__}")
                        record.update(
                            {
                                "reason": type(exc).__name__,
                                "message": str(exc),
                            }
                        )
                        failure_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    progress.update(1)
            accepted = target
        while (not args.direct_samples) and accepted < target and metrics["attempted_decodes"] < max_attempts:
            batch_size = min(int(args.batch_size), max_attempts - metrics["attempted_decodes"])
            prompts = [prompt for _ in range(batch_size)]
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
            for text in decoded:
                metrics["attempted_decodes"] += 1
                try:
                    plan = parse_plan_state(text)
                    validation = validate_plan_state(plan).to_dict()
                    reasons = accept_reasons(
                        plan=plan,
                        validation=validation,
                        formula_counts=formula_counts,
                        n_counts=n_counts,
                        rank_target=target,
                        args=args,
                    )
                    if reasons:
                        for reason in reasons:
                            add_failure(metrics, f"reject:{reason}")
                        failure_handle.write(
                            json.dumps(
                                {
                                    "attempt_idx": int(metrics["attempted_decodes"]),
                                    "stage": "reject",
                                    "reasons": reasons,
                                    "text_prefix": text[:240],
                                    "plan_state": plan,
                                    "plan_validation": validation,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue
                    sample_idx = int(tasks[accepted])
                    update_accepted_metrics(metrics, plan, validation)
                    formula_counts[str(plan["formula"])] += 1
                    n_counts[str(plan["N"])] += 1
                    raw_handle.write(
                        json.dumps(
                            {
                                "sample_idx": sample_idx,
                                "attempt_idx": int(metrics["attempted_decodes"]),
                                "text": text,
                                "parsed": True,
                                "representation": representation,
                                "conditioning_prompt": prompt.rstrip(),
                                "plan_state": plan,
                                "canonical_plan_json": plan_state_to_json(plan),
                                "plan_validation": validation,
                                "smact": plan.get("validator"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    accepted += 1
                    progress.update(1)
                    if accepted >= target:
                        break
                except Exception as exc:  # noqa: BLE001
                    add_failure(metrics, f"parse:{type(exc).__name__}")
                    failure_handle.write(
                        json.dumps(
                            {
                                "attempt_idx": int(metrics["attempted_decodes"]),
                                "stage": "parse",
                                "reason": type(exc).__name__,
                                "message": str(exc),
                                "text_prefix": text[:240],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        progress.close()

    metrics["time_sec"] = time.time() - start
    rank_histogram_limit = None if distributed else 100
    write_json(str(rank_path(args.output_dir, "sample_metrics.json", rank, distributed)), finalize_metrics(metrics, histogram_limit=rank_histogram_limit))
    if distributed:
        dist.barrier()
        if is_main:
            merge_distributed_outputs(args.output_dir, world_size, args)
        dist.barrier()


if __name__ == "__main__":
    main()
