#!/usr/bin/env python3
"""Sample R5 exact-length dynamic crystal proposals from a LLaDA checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from tqdm import tqdm

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
from crystal_dlm.h1_formula_only_body import (  # noqa: E402
    H1_FORMULA_ONLY_BODY_REPRESENTATION,
    build_formula_only_body_prompt,
)
from crystal_dlm.r5_plan_state import build_body_prompt, parse_plan_state_json  # noqa: E402
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


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def build_prompt_for_style(plan: Mapping[str, Any], *, prompt_style: str, row: Mapping[str, Any] | None = None, prompt_field: str = "prompt") -> str:
    if prompt_style == "formula_only":
        return build_formula_only_body_prompt(plan).rstrip() + "\n"
    if row is not None and row.get(prompt_field):
        return str(row[prompt_field]).rstrip() + "\n"
    return build_body_prompt(plan).rstrip() + "\n"


def read_plan_records(path: Path, prompt_field: str, *, body_prompt_style: str = "full_plan_state") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            plan = row.get("plan_state") or row.get("r5_plan_state")
            if plan is None and row.get(prompt_field):
                plan = parse_plan_state_json(str(row[prompt_field]))
            if not isinstance(plan, dict) or "N" not in plan:
                raise ValueError(f"Prompt JSONL row {idx} has no plan_state with N")
            prompt = build_prompt_for_style(plan, prompt_style=body_prompt_style, row=row, prompt_field=prompt_field)
            rows.append({"plan_state": plan, "prompt": prompt, "source_row": row, "source_idx": idx})
    if not rows:
        raise ValueError(f"No plan records found in {path}")
    return rows


def fallback_plan(num_atoms: int) -> Dict[str, Any]:
    return {
        "N": int(num_atoms),
        "elements": [],
        "counts": [],
        "formula": "",
        "reduced_formula": "",
        "charge_bucket": "unknown",
        "oxidation_candidates": "unknown",
        "anion_framework": "unknown",
        "lattice_system": "unknown",
        "spacegroup_bucket": "sg_unknown",
        "volume_per_atom_bin": "volpa_unknown",
        "prototype_key": f"N={int(num_atoms)}",
    }


def element_prefill_for_batch(tokenizer: Any, plans: List[Mapping[str, Any]]) -> Dict[int, List[int]]:
    vocab = tokenizer.get_vocab()
    position_to_ids: Dict[int, List[int]] = {}
    expanded_by_plan: List[List[int]] = []
    for plan in plans:
        elements = [str(value) for value in plan.get("elements") or []]
        counts = [int(value) for value in plan.get("counts") or []]
        if not elements or len(elements) != len(counts):
            raise ValueError("freeze-plan-composition requires plan elements/counts")
        expanded: List[int] = []
        for element, count in zip(elements, counts):
            token = f"<E_{element}>"
            if token not in vocab:
                raise RuntimeError(f"Tokenizer is missing element token {token}")
            expanded.extend([int(vocab[token])] * int(count))
        expected_n = int(plan["N"])
        if len(expanded) != expected_n:
            raise ValueError(f"Expanded composition length {len(expanded)} does not match plan N {expected_n}")
        expanded_by_plan.append(expanded)
    max_n = max((len(item) for item in expanded_by_plan), default=0)
    for slot_idx in range(max_n):
        position = 7 + 4 * slot_idx
        ids: List[int] = []
        for expanded in expanded_by_plan:
            if slot_idx >= len(expanded):
                raise ValueError("Batched plans must have equal N for element prefill")
            ids.append(expanded[slot_idx])
        position_to_ids[position] = ids
    return position_to_ids


def merge_prefill_maps(*maps: Mapping[int, List[int]]) -> Dict[int, List[int]]:
    merged: Dict[int, List[int]] = {}
    for item in maps:
        for position, values in item.items():
            if position in merged and merged[position] != list(values):
                raise ValueError(f"Conflicting prefill values at generation position {position}")
            merged[int(position)] = list(values)
    return merged


def build_tasks(args) -> List[Dict[str, Any]]:
    if args.prompt_jsonl is not None:
        records = read_plan_records(args.prompt_jsonl, args.prompt_field, body_prompt_style=args.body_prompt_style)
        tasks = []
        for sample_idx in range(args.num_samples):
            record = records[sample_idx % len(records)]
            tasks.append(
                {
                    "sample_idx": sample_idx,
                    "plan_state": record["plan_state"],
                    "prompt": record["prompt"],
                    "prompt_record_idx": record["source_idx"],
                    "prompt_record": record["source_row"],
                }
            )
        return tasks
    plan = fallback_plan(args.num_atoms)
    if args.prompt is not None and args.body_prompt_style != "formula_only":
        prompt = str(args.prompt).rstrip() + "\n"
    else:
        prompt = build_prompt_for_style(plan, prompt_style=args.body_prompt_style)
    return [
        {
            "sample_idx": sample_idx,
            "plan_state": plan,
            "prompt": prompt,
            "prompt_record_idx": None,
            "prompt_record": None,
        }
        for sample_idx in range(args.num_samples)
    ]


def merge_distributed_outputs(output_dir: Path, world_size: int) -> None:
    merged_metrics = {
        "requested_samples": 0,
        "decoded_samples": 0,
        "parse_success": 0,
        "plan_match_success": 0,
        "pymatgen_success": 0,
        "graph_success": 0,
        "failures": {},
        "time_sec": 0.0,
        "distributed": True,
        "world_size": world_size,
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
                for key in ("requested_samples", "decoded_samples", "parse_success", "plan_match_success", "pymatgen_success", "graph_success"):
                    merged_metrics[key] += int(metrics.get(key, 0))
                merged_metrics["time_sec"] = max(merged_metrics["time_sec"], float(metrics.get("time_sec") or 0.0))
                for reason, count in metrics.get("failures", {}).items():
                    merged_metrics["failures"][reason] = merged_metrics["failures"].get(reason, 0) + int(count)
            for filename, handle in (("raw_generations.jsonl", raw_out), ("failure_cases.jsonl", failure_out)):
                path = rank_path(output_dir, filename, rank, True)
                if path.exists():
                    handle.write(path.read_text(encoding="utf-8"))
            valid_arrays.extend(read_valid_arrays(rank_path(output_dir, "valid_arrays.jsonl", rank, True)))
            graph_path = rank_path(output_dir, "proposal_graphs.pt", rank, True)
            if graph_path.exists():
                proposal_graphs.extend(torch.load(graph_path, map_location="cpu"))
    merged_metrics["parse_rate"] = merged_metrics["parse_success"] / max(1, merged_metrics["decoded_samples"])
    merged_metrics["plan_match_rate"] = merged_metrics["plan_match_success"] / max(1, merged_metrics["decoded_samples"])
    merged_metrics["graph_acceptance_rate"] = merged_metrics["graph_success"] / max(1, merged_metrics["decoded_samples"])
    merged_metrics["valid_array_count"] = len(valid_arrays)
    write_json(str(output_dir / "sample_metrics.json"), merged_metrics)
    if valid_arrays:
        payload = arrays_to_torch_payload(valid_arrays)
        payload["time"] = merged_metrics["time_sec"]
        torch.save(payload, output_dir / "raw_dlm_samples.pt")
        torch.save(proposal_graphs, output_dir / "proposal_graphs.pt")


def wait_for_rank_metrics(output_dir: Path, world_size: int, *, timeout_sec: float = 120.0) -> list[str]:
    """Wait briefly for rank metrics so merge can survive a final NCCL hiccup."""

    deadline = time.time() + float(timeout_sec)
    missing: list[str] = []
    while True:
        missing = [
            str(rank_path(output_dir, "sample_metrics.json", rank, True))
            for rank in range(world_size)
            if not rank_path(output_dir, "sample_metrics.json", rank, True).exists()
        ]
        if not missing or time.time() >= deadline:
            return missing
        time.sleep(2.0)


def write_merge_warning(output_dir: Path, payload: Dict[str, Any]) -> None:
    warning_path = output_dir / "distributed_merge_warning.json"
    existing: list[Dict[str, Any]] = []
    if warning_path.exists():
        try:
            existing = json.loads(warning_path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []
    existing.append(payload)
    write_json(str(warning_path), existing)


def add_failure(metrics: Dict[str, Any], failure_handle, sample_idx: int, stage: str, exc: Exception, record: Dict[str, Any]) -> None:
    reason = f"{stage}:{type(exc).__name__}"
    metrics["failures"][reason] = metrics["failures"].get(reason, 0) + 1
    failure_handle.write(
        json.dumps(
            {
                "sample_idx": sample_idx,
                "stage": stage,
                "reason": type(exc).__name__,
                "message": str(exc),
                "text": record.get("text", ""),
                "plan_state": record.get("plan_state"),
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, required=True)
    parser.add_argument("--prompt-jsonl", type=Path, default=None)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--body-prompt-style", choices=["full_plan_state", "formula_only"], default="full_plan_state")
    parser.add_argument("--num-atoms", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
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

    if not 1 <= int(args.num_atoms) <= 20:
        raise ValueError("--num-atoms must be in 1..20")

    dist_info = init_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    distributed = dist_info["distributed"]
    is_main = dist_info["is_main"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    process_one = None if args.skip_graph_validation else import_process_one(args.crysllmgen_dir)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.checkpoint_path, dist_info["device"])
    tasks = build_tasks(args)
    tasks = [task for idx, task in enumerate(tasks) if idx % world_size == rank]
    tasks.sort(key=lambda item: (int(item["plan_state"]["N"]), int(item["sample_idx"])))

    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    r5_representation = (
        H1_FORMULA_ONLY_BODY_REPRESENTATION
        if args.body_prompt_style == "formula_only"
        else "r5_exact_dynamic_v1"
    )
    run_config.update({"representation": "dynamic_v1", "r5_representation": r5_representation, "distributed": distributed, "world_size": world_size})
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
    metrics = {
        "requested_samples": len(tasks),
        "decoded_samples": 0,
        "parse_success": 0,
        "plan_match_success": 0,
        "pymatgen_success": 0,
        "graph_success": 0,
        "failures": {},
        "time_sec": None,
        "rank": rank,
        "world_size": world_size,
    }

    start = time.time()
    with raw_path.open("w", encoding="utf-8") as raw_handle, failure_path.open("w", encoding="utf-8") as failure_handle:
        progress = tqdm(total=len(tasks), desc=f"R5 exact sampling rank{rank}", disable=distributed and not is_main)
        offset = 0
        while offset < len(tasks):
            num_atoms = int(tasks[offset]["plan_state"]["N"])
            batch: List[Dict[str, Any]] = []
            while offset < len(tasks) and len(batch) < args.batch_size and int(tasks[offset]["plan_state"]["N"]) == num_atoms:
                batch.append(tasks[offset])
                offset += 1
            prompts = [item["prompt"] for item in batch]
            gen_length = exact_body_token_count(num_atoms)
            allowed = exact_dynamic_schema_constraints(tokenizer, num_atoms) if args.schema_logit_mask else None
            prefill_maps: List[Mapping[int, List[int]]] = []
            if args.prefill_count_token:
                prefill_maps.append(count_prefill_for_batch(tokenizer, num_atoms, len(batch)))
            if args.freeze_plan_composition:
                prefill_maps.append(element_prefill_for_batch(tokenizer, [item["plan_state"] for item in batch]))
            prefill = merge_prefill_maps(*prefill_maps) if prefill_maps else None
            schedule = exact_dynamic_generation_schedule(num_atoms) if args.generation_schedule == "exact-plan" else None
            lightweight_constraints = build_dynamic_lightweight_constraints(
                tokenizer,
                duplicate_coordinate_mask=args.duplicate_coordinate_mask,
                lattice_volume_mask=args.lattice_volume_mask,
                min_lattice_rad=args.min_lattice_rad,
            )
            encoded = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(model_device(model))
            attention_mask = encoded["attention_mask"].to(model_device(model))
            outputs = generate(
                model,
                input_ids,
                attention_mask=attention_mask,
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
            generated_ids = outputs[:, input_ids.shape[1] :]
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            for item, text in zip(batch, decoded):
                sample_idx = int(item["sample_idx"])
                raw_record: Dict[str, Any] = {
                    "sample_idx": sample_idx,
                    "text": text,
                    "representation": "dynamic_v1",
                    "r5_representation": r5_representation,
                    "plan_state": item["plan_state"],
                    "conditioning_prompt": item["prompt"].rstrip(),
                    "prompt_record_idx": item["prompt_record_idx"],
                }
                if item.get("prompt_record") is not None:
                    source = dict(item["prompt_record"])
                    source.pop(args.prompt_field, None)
                    raw_record["prompt_record"] = source
                metrics["decoded_samples"] += 1
                try:
                    arrays = validate_answer_matches_plan(item["plan_state"], text)
                    metrics["parse_success"] += 1
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
                    add_failure(metrics, failure_handle, sample_idx, "decode_or_graph", exc, {**raw_record, "text": text})
                    raw_record.update({"parsed": False, "reason": type(exc).__name__, "message": str(exc)})
                raw_handle.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                progress.update(1)

    metrics["time_sec"] = time.time() - start
    metrics["parse_rate"] = metrics["parse_success"] / max(1, metrics["decoded_samples"])
    metrics["plan_match_rate"] = metrics["plan_match_success"] / max(1, metrics["decoded_samples"])
    metrics["graph_acceptance_rate"] = metrics["graph_success"] / max(1, metrics["decoded_samples"])
    metrics["valid_array_count"] = len(valid_arrays)
    write_json(str(rank_path(args.output_dir, "sample_metrics.json", rank, distributed)), metrics)
    write_valid_arrays(valid_arrays_path, valid_arrays)
    if proposal_graphs:
        torch.save(proposal_graphs, rank_path(args.output_dir, "proposal_graphs.pt", rank, distributed))
    if valid_arrays:
        torch.save(arrays_to_torch_payload(valid_arrays), rank_path(args.output_dir, "raw_dlm_samples.pt", rank, distributed))
    if distributed:
        first_barrier_error = None
        try:
            dist.barrier()
        except Exception as exc:  # noqa: BLE001
            first_barrier_error = f"{type(exc).__name__}: {exc}"
        if is_main:
            if first_barrier_error is not None:
                write_merge_warning(
                    args.output_dir,
                    {
                        "stage": "pre_merge_barrier",
                        "message": first_barrier_error,
                        "note": "Attempting best-effort merge from rank files.",
                    },
                )
            missing = wait_for_rank_metrics(args.output_dir, world_size)
            if missing:
                write_merge_warning(
                    args.output_dir,
                    {
                        "stage": "wait_for_rank_metrics",
                        "missing": missing,
                        "note": "Merged available rank files only.",
                    },
                )
            merge_distributed_outputs(args.output_dir, world_size)
        try:
            dist.barrier()
        except Exception as exc:  # noqa: BLE001
            if is_main:
                write_merge_warning(
                    args.output_dir,
                    {
                        "stage": "post_merge_barrier",
                        "message": f"{type(exc).__name__}: {exc}",
                        "note": "Merged outputs were already written by rank 0.",
                    },
                )


if __name__ == "__main__":
    main()
