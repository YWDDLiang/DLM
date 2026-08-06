#!/usr/bin/env python3
"""Apply one learned R5 compact-plan repair pass without candidate selection."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from tqdm import tqdm

from crystal_dlm.fixed_slot import MASK_TOKEN_ID, write_json  # noqa: E402
from crystal_dlm.llada_generation import generate  # noqa: E402
from crystal_dlm.r5_plan_state import (  # noqa: E402
    build_compact_plan_repair_prompt,
    parse_compact_plan_state,
    plan_state_to_json,
    validate_plan_state,
)
from scripts.sample_llada_dynamic_crystals import init_distributed, load_model_and_tokenizer, rank_path  # noqa: E402
from scripts.sample_llada_r5_plan_compact import add_failure, finalize_metrics, update_direct_metrics  # noqa: E402


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def first_line(text: Any) -> str:
    return str(text or "").strip().splitlines()[0] if str(text or "").strip() else ""


def element_count_sum(text: str) -> tuple[int | None, int | None]:
    n_match = re.search(r"\bN\s*=\s*(\d{1,3})", text)
    generated_n = int(n_match.group(1)) if n_match else None
    e_match = re.search(r"\bE\s*=\s*([^;]+)", text)
    if not e_match:
        return generated_n, None
    total = 0
    found = False
    for match in re.finditer(r"([A-Z][a-z]?)[\s:=xX*_-]*(\d{1,2})", e_match.group(1)):
        found = True
        total += int(match.group(2))
    return generated_n, total if found else None


def labels_for_parse_failure(text: str, message: str) -> list[str]:
    labels = ["parse_fail"]
    generated_n, count_sum = element_count_sum(text)
    if "atom count" in message or (count_sum is not None and (count_sum < 1 or count_sum > 20)):
        labels.append("atom_count_out_of_range")
    if generated_n is None:
        labels.append("missing_N")
    if count_sum is None:
        labels.append("missing_element_counts")
    elif generated_n is not None and generated_n != count_sum:
        labels.append("generated_N_count_mismatch")
    if "SG=" not in text or re.search(r"\bSG\s*=\s*[^;,\s]+,", text) or re.search(r"\bSG\s*=\s*sg_0{4,}", text):
        labels.append("spacegroup_bucket_malformed")
    if "VP=" not in text:
        labels.append("volume_bin_malformed")
    return labels


def labels_for_parsed_plan(plan: Mapping[str, Any], validation: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    if not validation.get("valid_generated_N", True):
        labels.append("generated_N_count_mismatch")
    if not validation.get("valid_counts", True):
        labels.append("invalid_counts")
    if not validation.get("valid_formula", True):
        labels.append("formula_mismatch")
    if len(plan.get("elements") or []) <= 1:
        labels.append("single_element")
    validator = plan.get("validator") or {}
    reason = str(validator.get("reason") or "")
    if reason == "all_metal_shortcut":
        labels.append("all_metal")
    elif reason == "charge_neutrality_fail":
        labels.append("charge_fail")
    elif reason == "pauling_fail_or_ratio_rejected":
        labels.append("pauling_fail")
    elif validator.get("valid") is not True:
        labels.append("non_smact")
    return list(dict.fromkeys(labels))


def diagnose_record(record: Mapping[str, Any]) -> dict[str, Any]:
    text = first_line(record.get("text"))
    try:
        plan = parse_compact_plan_state(text)
        validation = validate_plan_state(plan).to_dict()
        labels = labels_for_parsed_plan(plan, validation)
        return {
            "parsed": True,
            "plan_state": plan,
            "plan_validation": validation,
            "labels": labels,
            "message": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "parsed": False,
            "plan_state": None,
            "plan_validation": None,
            "labels": labels_for_parse_failure(text, str(exc)),
            "reason": type(exc).__name__,
            "message": str(exc),
        }


def empty_metrics(args, rank: int, world_size: int) -> dict[str, Any]:
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
        "repair_requested": 0,
        "repair_attempted": 0,
        "repair_parse_success": 0,
        "passthrough": 0,
        "failures": {},
        "time_sec": 0.0,
        "rank": rank,
        "world_size": world_size,
        "generation_length": int(args.generation_length),
        "max_attempt_factor": 1,
        "formula_cap": 0,
        "n_cap_fraction": 1.0,
        "n_histogram": Counter(),
        "formula_histogram": Counter(),
        "prototype_histogram": Counter(),
        "charge_bucket_histogram": Counter(),
        "anion_framework_histogram": Counter(),
        "lattice_system_histogram": Counter(),
        "spacegroup_bucket_histogram": Counter(),
        "volume_per_atom_bin_histogram": Counter(),
        "repair_label_histogram": Counter(),
    }


def record_success_metrics(metrics: dict[str, Any], plan: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    update_direct_metrics(metrics, plan, validation)


def parse_repaired_output(text: str, output_record: dict[str, Any], metrics: dict[str, Any]) -> None:
    try:
        plan = parse_compact_plan_state(text)
        validation = validate_plan_state(plan).to_dict()
        record_success_metrics(metrics, plan, validation)
        output_record.update(
            {
                "parsed": True,
                "plan_state": plan,
                "canonical_plan_json": plan_state_to_json(plan),
                "plan_validation": validation,
                "smact": plan.get("validator"),
            }
        )
        if output_record.get("repair_applied"):
            metrics["repair_parse_success"] += 1
    except Exception as exc:  # noqa: BLE001
        add_failure(metrics, f"parse:{type(exc).__name__}")
        output_record.update({"parsed": False, "reason": type(exc).__name__, "message": str(exc)})


def merge_counter_payload(counter: Counter[str], payload: Mapping[str, Any]) -> None:
    for key, value in payload.items():
        counter[str(key)] += int(value)


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
                    "repair_requested",
                    "repair_attempted",
                    "repair_parse_success",
                    "passthrough",
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
                    "repair_label_histogram",
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
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--block-length", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260530)
    args = parser.parse_args()
    args.generation_length = int(args.gen_length)
    args.steps = int(args.steps or args.generation_length)

    dist_info = init_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    distributed = dist_info["distributed"]
    is_main = dist_info["is_main"]
    torch.manual_seed(int(args.seed) + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed) + rank)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.input_jsonl)
    tasks = [(idx, row) for idx, row in enumerate(records) if idx % world_size == rank]
    metrics = empty_metrics(args, rank=rank, world_size=world_size)
    metrics["requested_samples"] = len(tasks)

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.checkpoint_path, dist_info["device"])
    if is_main:
        run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        run_config.update(
            {
                "representation": "r5_plan_state_compact_repair",
                "distributed": distributed,
                "world_size": world_size,
                "repair_policy": "one_learned_repair_pass_for_each_flagged_sample_no_candidate_selection",
            }
        )
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
    start = time.time()
    output_records: dict[int, dict[str, Any]] = {}
    repair_items: list[dict[str, Any]] = []

    for input_idx, record in tasks:
        diagnosis = diagnose_record(record)
        labels = list(diagnosis.get("labels") or [])
        for label in labels:
            metrics["repair_label_histogram"][str(label)] += 1
        if labels:
            metrics["repair_requested"] += 1
            visible = first_line(record.get("text"))
            repair_items.append(
                {
                    "input_idx": input_idx,
                    "source_record": record,
                    "diagnosis": diagnosis,
                    "prompt": build_compact_plan_repair_prompt(visible_plan=visible, violation_labels=labels).rstrip() + "\n",
                    "labels": labels,
                    "visible_plan": visible,
                }
            )
            continue
        plan = diagnosis["plan_state"]
        validation = diagnosis["plan_validation"]
        metrics["decoded_samples"] += 1
        metrics["attempted_decodes"] += 1
        metrics["passthrough"] += 1
        record_success_metrics(metrics, plan, validation)
        output_records[input_idx] = {
            "sample_idx": int(record.get("sample_idx", input_idx)),
            "input_idx": input_idx,
            "text": first_line(record.get("text")),
            "parsed": True,
            "representation": "r5_plan_state_compact",
            "repair_applied": False,
            "repair_labels": [],
            "source_record": record,
            "plan_state": plan,
            "canonical_plan_json": plan_state_to_json(plan),
            "plan_validation": validation,
            "smact": plan.get("validator"),
        }

    progress = tqdm(total=len(repair_items), desc=f"R5 compact repair rank{rank}", disable=distributed and not is_main)
    for offset in range(0, len(repair_items), int(args.batch_size)):
        batch = repair_items[offset : offset + int(args.batch_size)]
        prompts = [item["prompt"] for item in batch]
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
        for item, text in zip(batch, decoded):
            metrics["attempted_decodes"] += 1
            metrics["repair_attempted"] += 1
            metrics["decoded_samples"] += 1
            output_record: dict[str, Any] = {
                "sample_idx": int(item["source_record"].get("sample_idx", item["input_idx"])),
                "input_idx": int(item["input_idx"]),
                "text": first_line(text),
                "parsed": False,
                "representation": "r5_plan_state_compact",
                "repair_applied": True,
                "repair_labels": item["labels"],
                "visible_compact_plan": item["visible_plan"],
                "source_record": item["source_record"],
                "conditioning_prompt": item["prompt"].rstrip(),
            }
            parse_repaired_output(first_line(text), output_record, metrics)
            output_records[int(item["input_idx"])] = output_record
            progress.update(1)
    progress.close()

    with raw_path.open("w", encoding="utf-8") as raw_handle, failure_path.open("w", encoding="utf-8") as failure_handle:
        for input_idx in sorted(output_records):
            record = output_records[input_idx]
            raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if not record.get("parsed"):
                failure_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics["time_sec"] = time.time() - start
    rank_histogram_limit = None if distributed else 100
    write_json(
        str(rank_path(args.output_dir, "sample_metrics.json", rank, distributed)),
        finalize_metrics(metrics, histogram_limit=rank_histogram_limit),
    )
    if distributed:
        dist.barrier()
        if is_main:
            merge_distributed_outputs(args.output_dir, world_size, args)
        dist.barrier()


if __name__ == "__main__":
    main()
