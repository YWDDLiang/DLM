#!/usr/bin/env python3
"""Sample de novo R5-C composition-plan-to-body crystal proposals."""

from __future__ import annotations

import argparse
import json
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

from crystal_dlm.crysllmgen_text import CRYSLLMGEN_TEXT_PROMPT  # noqa: E402
from crystal_dlm.dynamic_crystal import arrays_to_torch_payload, write_json  # noqa: E402
from crystal_dlm.fixed_slot import MASK_TOKEN_ID  # noqa: E402
from crystal_dlm.llada_generation import generate  # noqa: E402
from crystal_dlm.r5_dynamic_length import (  # noqa: E402
    count_prefill_for_batch,
    exact_body_token_count,
    exact_dynamic_generation_schedule,
    exact_dynamic_schema_constraints,
    validate_answer_matches_plan,
)
from crystal_dlm.r5_plan_body import (  # noqa: E402
    R5C_FORMULA_END_PLAN_FORMAT,
    R5C_PLAN_BODY_BODY_LABEL,
    R5C_PLAN_BODY_PLAN_LABEL,
    R5C_PLAN_BODY_REPRESENTATION,
    R5C_PLAN_FORMAT,
    R5C_PLAN_STYLES,
    R5C_SEMANTIC_PLAN_FORMAT,
    format_composition_plan,
    has_plan_end_marker,
    has_plan_tail_after_end_marker,
    normalize_plan_style,
    parse_composition_plan,
    representation_for_plan_style,
)
from scripts.sample_llada_dynamic_crystals import (  # noqa: E402
    build_dynamic_lightweight_constraints,
    graph_from_arrays,
    import_process_one,
    init_distributed,
    load_model_and_tokenizer,
    rank_path,
    read_valid_arrays,
    write_valid_arrays,
)
from scripts.sample_llada_r5_exact_length import element_prefill_for_batch, merge_prefill_maps  # noqa: E402


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def clean_decoded_text(text: str, tokenizer: Any) -> str:
    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n")
    for marker in (getattr(tokenizer, "eos_token", None), "<|endoftext|>", "</s>"):
        if marker and marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0]
    return cleaned.strip()


def build_plan_prompt(base_prompt: str) -> str:
    return f"{base_prompt.rstrip()}\n"


def build_body_prompt(base_prompt: str, plan_text: str) -> str:
    return (
        f"{base_prompt.rstrip()}\n"
        f"{R5C_PLAN_BODY_PLAN_LABEL}\n"
        f"{plan_text.strip()}\n"
        f"{R5C_PLAN_BODY_BODY_LABEL}\n"
    )


def add_failure(
    metrics: Dict[str, Any],
    failure_handle,
    sample_idx: int,
    stage: str,
    exc: Exception,
    record: Mapping[str, Any],
) -> None:
    reason = f"{stage}:{type(exc).__name__}"
    metrics["failures"][reason] = int(metrics["failures"].get(reason, 0)) + 1
    failure_handle.write(
        json.dumps(
            {
                "sample_idx": int(sample_idx),
                "stage": stage,
                "reason": type(exc).__name__,
                "message": str(exc),
                "raw_plan_text": record.get("raw_plan_text"),
                "plan_text": record.get("plan_text"),
                "body_text": record.get("body_text"),
                "parsed_plan": record.get("parsed_plan"),
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def merge_distributed_outputs(output_dir: Path, world_size: int) -> None:
    metric_keys = [
        "requested_samples",
        "decoded_samples",
        "plan_parse_success",
        "formula_parse_success",
        "plan_end_marker_success",
        "plan_tail_after_end_marker_count",
        "valid_n_success",
        "valid_formula_success",
        "body_parse_success",
        "plan_match_success",
        "pymatgen_success",
        "graph_success",
        "single_element_plans",
        "family_match_formula_success",
        "arity_match_formula_success",
        "size_match_formula_success",
    ]
    merged_metrics: Dict[str, Any] = {
        **{key: 0 for key in metric_keys},
        "failures": {},
        "time_sec": 0.0,
        "distributed": True,
        "world_size": int(world_size),
    }
    valid_arrays: List[Dict[str, Any]] = []
    proposal_graphs: List[Dict[str, Any]] = []
    with (output_dir / "raw_generations.jsonl").open("w", encoding="utf-8") as raw_out, (
        output_dir / "failure_cases.jsonl"
    ).open("w", encoding="utf-8") as failure_out:
        for rank in range(world_size):
            metrics_path = rank_path(output_dir, "sample_metrics.json", rank, True)
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                for key in metric_keys:
                    merged_metrics[key] += int(metrics.get(key, 0))
                merged_metrics["time_sec"] = max(
                    float(merged_metrics["time_sec"]),
                    float(metrics.get("time_sec") or 0.0),
                )
                for reason, count in metrics.get("failures", {}).items():
                    merged_metrics["failures"][reason] = int(merged_metrics["failures"].get(reason, 0)) + int(count)
            for filename, handle in (("raw_generations.jsonl", raw_out), ("failure_cases.jsonl", failure_out)):
                path = rank_path(output_dir, filename, rank, True)
                if path.exists():
                    handle.write(path.read_text(encoding="utf-8"))
            valid_arrays.extend(read_valid_arrays(rank_path(output_dir, "valid_arrays.jsonl", rank, True)))
            graph_path = rank_path(output_dir, "proposal_graphs.pt", rank, True)
            if graph_path.exists():
                proposal_graphs.extend(torch.load(graph_path, map_location="cpu"))
    finalize_metrics(merged_metrics, len(valid_arrays))
    write_json(str(output_dir / "sample_metrics.json"), merged_metrics)
    if valid_arrays:
        payload = arrays_to_torch_payload(valid_arrays)
        payload["time"] = merged_metrics["time_sec"]
        torch.save(payload, output_dir / "raw_dlm_samples.pt")
        torch.save(proposal_graphs, output_dir / "proposal_graphs.pt")


def finalize_metrics(metrics: Dict[str, Any], valid_array_count: int) -> None:
    decoded = max(1, int(metrics.get("decoded_samples", 0)))
    metrics["plan_parse_rate"] = int(metrics.get("plan_parse_success", 0)) / decoded
    metrics["formula_parse_rate"] = int(metrics.get("formula_parse_success", 0)) / decoded
    metrics["plan_end_marker_rate"] = int(metrics.get("plan_end_marker_success", 0)) / decoded
    metrics["plan_tail_after_end_marker_rate"] = int(metrics.get("plan_tail_after_end_marker_count", 0)) / decoded
    metrics["valid_n_rate"] = int(metrics.get("valid_n_success", 0)) / decoded
    metrics["valid_formula_rate"] = int(metrics.get("valid_formula_success", 0)) / decoded
    metrics["body_parse_rate"] = int(metrics.get("body_parse_success", 0)) / decoded
    metrics["plan_match_rate"] = int(metrics.get("plan_match_success", 0)) / decoded
    metrics["graph_acceptance_rate"] = int(metrics.get("graph_success", 0)) / decoded
    metrics["single_element_rate"] = int(metrics.get("single_element_plans", 0)) / decoded
    metrics["family_match_formula_rate"] = int(metrics.get("family_match_formula_success", 0)) / decoded
    metrics["arity_match_formula_rate"] = int(metrics.get("arity_match_formula_success", 0)) / decoded
    metrics["size_match_formula_rate"] = int(metrics.get("size_match_formula_success", 0)) / decoded
    metrics["valid_array_count"] = int(valid_array_count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--plan-gen-length", type=int, default=96)
    parser.add_argument("--plan-steps", type=int, default=96)
    parser.add_argument("--plan-block-length", type=int, default=1)
    parser.add_argument("--plan-style", choices=list(R5C_PLAN_STYLES), default=R5C_PLAN_FORMAT)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--prompt", default=CRYSLLMGEN_TEXT_PROMPT)
    parser.add_argument("--schema-logit-mask", action="store_true", default=True)
    parser.add_argument("--no-schema-logit-mask", dest="schema_logit_mask", action="store_false")
    parser.add_argument("--prefill-count-token", action="store_true", default=True)
    parser.add_argument("--no-prefill-count-token", dest="prefill_count_token", action="store_false")
    parser.add_argument("--freeze-plan-composition", action="store_true", default=True)
    parser.add_argument("--no-freeze-plan-composition", dest="freeze_plan_composition", action="store_false")
    parser.add_argument("--duplicate-coordinate-mask", action="store_true", default=True)
    parser.add_argument("--no-duplicate-coordinate-mask", dest="duplicate_coordinate_mask", action="store_false")
    parser.add_argument("--lattice-volume-mask", action="store_true", default=True)
    parser.add_argument("--no-lattice-volume-mask", dest="lattice_volume_mask", action="store_false")
    parser.add_argument("--min-lattice-rad", type=float, default=1e-4)
    parser.add_argument("--generation-schedule", choices=["exact-plan", "default"], default="exact-plan")
    parser.add_argument("--skip-graph-validation", action="store_true")
    args = parser.parse_args()

    if args.plan_gen_length % args.plan_block_length != 0:
        raise ValueError("--plan-gen-length must be divisible by --plan-block-length")
    if args.plan_steps % (args.plan_gen_length // args.plan_block_length) != 0:
        raise ValueError("--plan-steps must divide evenly across plan blocks")
    plan_style = normalize_plan_style(args.plan_style)
    r5_representation = representation_for_plan_style(plan_style)

    dist_info = init_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    distributed = dist_info["distributed"]
    is_main = dist_info["is_main"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    process_one = None if args.skip_graph_validation else import_process_one(args.crysllmgen_dir)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.checkpoint_path, dist_info["device"])

    sample_indices = list(range(rank, int(args.num_samples), world_size))
    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config.update(
        {
            "representation": "dynamic_v1",
            "r5_representation": r5_representation,
            "plan_style": plan_style,
            "distributed": distributed,
            "world_size": world_size,
            "de_novo": True,
            "external_plan_source": None,
        }
    )
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
    valid_arrays_path = rank_path(args.output_dir, "valid_arrays.jsonl", rank, distributed)
    valid_arrays: List[Dict[str, Any]] = []
    proposal_graphs: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {
        "requested_samples": len(sample_indices),
        "decoded_samples": 0,
        "plan_parse_success": 0,
        "formula_parse_success": 0,
        "plan_end_marker_success": 0,
        "plan_tail_after_end_marker_count": 0,
        "valid_n_success": 0,
        "valid_formula_success": 0,
        "body_parse_success": 0,
        "plan_match_success": 0,
        "pymatgen_success": 0,
        "graph_success": 0,
        "single_element_plans": 0,
        "family_match_formula_success": 0,
        "arity_match_formula_success": 0,
        "size_match_formula_success": 0,
        "failures": {},
        "time_sec": None,
        "rank": rank,
        "world_size": world_size,
    }
    base_prompt = str(args.prompt)
    plan_prompt = build_plan_prompt(base_prompt)

    start = time.time()
    with raw_path.open("w", encoding="utf-8") as raw_handle, failure_path.open("w", encoding="utf-8") as failure_handle:
        progress = tqdm(total=len(sample_indices), desc=f"R5-C de novo rank{rank}", disable=distributed and not is_main)
        for batch_start in range(0, len(sample_indices), args.batch_size):
            current_indices = sample_indices[batch_start : batch_start + args.batch_size]
            encoded = tokenizer(
                [plan_prompt] * len(current_indices),
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(model_device(model))
            attention_mask = encoded["attention_mask"].to(model_device(model))
            plan_outputs = generate(
                model,
                input_ids,
                attention_mask=attention_mask,
                steps=args.plan_steps,
                gen_length=args.plan_gen_length,
                block_length=args.plan_block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
                mask_id=MASK_TOKEN_ID,
            )
            plan_ids = plan_outputs[:, input_ids.shape[1] :]
            raw_plan_texts = tokenizer.batch_decode(
                plan_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            plan_records: List[Dict[str, Any]] = []
            for sample_idx, raw_plan_text in zip(current_indices, raw_plan_texts):
                metrics["decoded_samples"] += 1
                cleaned_plan = clean_decoded_text(raw_plan_text, tokenizer)
                raw_record: Dict[str, Any] = {
                    "sample_idx": int(sample_idx),
                    "representation": "dynamic_v1",
                    "r5_representation": r5_representation,
                    "plan_style": plan_style,
                    "raw_plan_text": raw_plan_text,
                    "plan_text": cleaned_plan,
                    "de_novo": True,
                }
                end_marker_present = has_plan_end_marker(cleaned_plan)
                tail_after_end_marker = has_plan_tail_after_end_marker(cleaned_plan)
                raw_record["plan_end_marker_present"] = end_marker_present
                raw_record["plan_tail_after_end_marker"] = tail_after_end_marker
                if end_marker_present:
                    metrics["plan_end_marker_success"] += 1
                if tail_after_end_marker:
                    metrics["plan_tail_after_end_marker_count"] += 1
                try:
                    formula_parse_plan = parse_composition_plan(cleaned_plan, plan_style=R5C_PLAN_FORMAT)
                    metrics["formula_parse_success"] += 1
                    raw_record["formula_parse"] = True
                    raw_record["formula_parse_formula"] = formula_parse_plan.get("formula")
                    raw_record["formula_parse_N"] = formula_parse_plan.get("N")
                except Exception as formula_exc:  # noqa: BLE001
                    raw_record["formula_parse"] = False
                    raw_record["formula_parse_error"] = f"{type(formula_exc).__name__}: {formula_exc}"
                try:
                    plan = parse_composition_plan(cleaned_plan, plan_style=plan_style)
                    canonical_plan_text = format_composition_plan(plan, plan_style=plan_style)
                    metrics["plan_parse_success"] += 1
                    metrics["valid_n_success"] += 1
                    metrics["valid_formula_success"] += 1
                    consistency = plan.get("semantic_consistency") or {}
                    if consistency.get("family_match_formula") is True:
                        metrics["family_match_formula_success"] += 1
                    if consistency.get("arity_match_formula") is True:
                        metrics["arity_match_formula_success"] += 1
                    if consistency.get("size_match_formula") is True:
                        metrics["size_match_formula_success"] += 1
                    if len(plan.get("elements") or []) == 1:
                        metrics["single_element_plans"] += 1
                    raw_record["generated_plan_text"] = cleaned_plan
                    raw_record["plan_text"] = canonical_plan_text
                    raw_record["parsed_plan"] = plan
                    raw_record["semantic_consistency"] = consistency
                    raw_record["generated_semantic_fields"] = plan.get("generated_semantic_fields") or {}
                    plan_records.append(
                        {
                            "sample_idx": int(sample_idx),
                            "plan": plan,
                            "plan_text": canonical_plan_text,
                            "raw_record": raw_record,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    add_failure(metrics, failure_handle, int(sample_idx), "plan_parse", exc, raw_record)
                    raw_record.update({"parsed": False, "stage": "plan_parse", "reason": type(exc).__name__, "message": str(exc)})
                    raw_handle.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                    progress.update(1)

            plan_records.sort(key=lambda item: (int(item["plan"]["N"]), int(item["sample_idx"])))
            offset = 0
            while offset < len(plan_records):
                num_atoms = int(plan_records[offset]["plan"]["N"])
                body_batch: List[Dict[str, Any]] = []
                while offset < len(plan_records) and len(body_batch) < args.batch_size and int(plan_records[offset]["plan"]["N"]) == num_atoms:
                    body_batch.append(plan_records[offset])
                    offset += 1
                body_prompts = [build_body_prompt(base_prompt, item["plan_text"]) for item in body_batch]
                gen_length = exact_body_token_count(num_atoms)
                allowed = exact_dynamic_schema_constraints(tokenizer, num_atoms) if args.schema_logit_mask else None
                prefill_maps: List[Mapping[int, List[int]]] = []
                if args.prefill_count_token:
                    prefill_maps.append(count_prefill_for_batch(tokenizer, num_atoms, len(body_batch)))
                if args.freeze_plan_composition:
                    prefill_maps.append(element_prefill_for_batch(tokenizer, [item["plan"] for item in body_batch]))
                prefill = merge_prefill_maps(*prefill_maps) if prefill_maps else None
                schedule = exact_dynamic_generation_schedule(num_atoms) if args.generation_schedule == "exact-plan" else None
                lightweight_constraints = build_dynamic_lightweight_constraints(
                    tokenizer,
                    duplicate_coordinate_mask=args.duplicate_coordinate_mask,
                    lattice_volume_mask=args.lattice_volume_mask,
                    min_lattice_rad=args.min_lattice_rad,
                )
                body_encoded = tokenizer(body_prompts, add_special_tokens=False, padding=True, return_tensors="pt")
                body_input_ids = body_encoded["input_ids"].to(model_device(model))
                body_attention_mask = body_encoded["attention_mask"].to(model_device(model))
                body_outputs = generate(
                    model,
                    body_input_ids,
                    attention_mask=body_attention_mask,
                    steps=gen_length,
                    gen_length=gen_length,
                    block_length=1,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                    mask_id=MASK_TOKEN_ID,
                    allowed_token_ids_by_generation_pos=allowed,
                    prefill_token_ids_by_generation_pos=prefill,
                    generation_position_groups=schedule,
                    lightweight_decoding_constraints=lightweight_constraints,
                )
                body_ids = body_outputs[:, body_input_ids.shape[1] :]
                body_texts = tokenizer.batch_decode(body_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                for item, body_text in zip(body_batch, body_texts):
                    raw_record = dict(item["raw_record"])
                    raw_record["body_text"] = body_text
                    raw_record["text"] = (
                        f"{R5C_PLAN_BODY_PLAN_LABEL}\n"
                        f"{item['plan_text']}\n"
                        f"{R5C_PLAN_BODY_BODY_LABEL}\n"
                        f"{body_text}"
                    )
                    try:
                        arrays = validate_answer_matches_plan(item["plan"], body_text)
                        metrics["body_parse_success"] += 1
                        metrics["plan_match_success"] += 1
                        if process_one is not None:
                            graph, cif = graph_from_arrays(arrays, process_one)
                            metrics["graph_success"] += 1
                            proposal_graphs.append(graph)
                            raw_record["cif"] = cif
                        else:
                            metrics["graph_success"] += 1
                        metrics["pymatgen_success"] += 1
                        valid_arrays.append(arrays)
                        raw_record.update({"parsed": True, "num_atoms": arrays["num_atoms"]})
                    except Exception as exc:  # noqa: BLE001
                        add_failure(metrics, failure_handle, int(item["sample_idx"]), "body_or_graph", exc, raw_record)
                        raw_record.update({"parsed": False, "stage": "body_or_graph", "reason": type(exc).__name__, "message": str(exc)})
                    raw_handle.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                    progress.update(1)

    metrics["time_sec"] = time.time() - start
    finalize_metrics(metrics, len(valid_arrays))
    write_json(str(rank_path(args.output_dir, "sample_metrics.json", rank, distributed)), metrics)
    write_valid_arrays(valid_arrays_path, valid_arrays)
    if proposal_graphs:
        torch.save(proposal_graphs, rank_path(args.output_dir, "proposal_graphs.pt", rank, distributed))
    if valid_arrays:
        torch.save(arrays_to_torch_payload(valid_arrays), rank_path(args.output_dir, "raw_dlm_samples.pt", rank, distributed))
    if distributed:
        dist.barrier()
        if is_main:
            merge_distributed_outputs(args.output_dir, world_size)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
