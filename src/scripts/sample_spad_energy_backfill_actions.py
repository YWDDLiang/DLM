#!/usr/bin/env python3
"""Sample no-op plus three legal suffix-visible SPAD backfill actions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import torch
import torch.distributed as dist
from tqdm import tqdm

from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.r5_dynamic_length import (
    exact_body_token_count,
    exact_dynamic_schema_constraints,
    validate_answer_matches_plan,
    validate_dynamic_tokenizer_contract,
)
from crystal_dlm.spad_generation import revise_spad_anchors
from crystal_dlm.spad_program import (
    anchor_revision_slots,
    program_from_element_order,
)
from scripts.sample_llada_dynamic_crystals import (
    build_dynamic_lightweight_constraints,
    graph_from_arrays,
    import_process_one,
    init_distributed,
    load_model_and_tokenizer,
    rank_path,
)


SCHEMA = "spad_energy_backfill_action_v1"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(path)
                yield value


def model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def prepare_task(row: Mapping[str, Any]) -> dict[str, Any] | None:
    group_idx = int(row["sample_idx"])
    if row.get("parsed") is not True or not isinstance(row.get("text"), str):
        return None
    plan = row.get("plan_state")
    prompt_record = row.get("prompt_record")
    if not isinstance(plan, dict) or not isinstance(prompt_record, dict):
        raise ValueError(f"predictor row {group_idx} lacks Plan/program metadata")
    order = prompt_record.get("species_program")
    if not isinstance(order, list) or not order:
        raise ValueError(f"predictor row {group_idx} lacks species program")
    arrays = validate_answer_matches_plan(plan, str(row["text"]))
    program = program_from_element_order(
        plan,
        [str(value) for value in order],
        order_source=str(
            prompt_record.get("species_program_source")
            or "frozen_planner_llama_pointer"
        ),
    )
    revisions = anchor_revision_slots(program)
    if not revisions:
        raise ValueError("SPAD energy state has no anchor to backfill")
    prompt = str(row.get("conditioning_prompt") or "").rstrip() + "\n"
    if not prompt.strip():
        raise ValueError("predictor row lacks conditioning prompt")
    return {
        "group_idx": group_idx,
        "source_row": dict(row),
        "plan_state": plan,
        "source_arrays": arrays,
        "prompt": prompt,
        "answer": str(row["text"]),
        "species_program": list(program.element_order),
        "backfill_slot": int(revisions[0]),
    }


def candidate_record(
    task: Mapping[str, Any],
    *,
    candidate_idx: int,
    answer: str,
    revision_log: list[dict[str, Any]],
    process_one: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    group_idx = int(task["group_idx"])
    flat_idx = group_idx * 4 + int(candidate_idx)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "sample_idx": flat_idx,
        "group_idx": group_idx,
        "candidate_idx": int(candidate_idx),
        "mandatory_noop": int(candidate_idx) == 0,
        "plan_state": task["plan_state"],
        "prompt": task["prompt"],
        "source_answer": task["answer"],
        "answer": answer,
        "species_program": task["species_program"],
        "backfill_slot": int(task["backfill_slot"]),
        "suffix_visible": True,
        "revision_log": revision_log,
        "validity_before_energy": True,
        "outcomes_read": False,
        "selection_rerank_replacement": False,
    }
    try:
        arrays = validate_answer_matches_plan(task["plan_state"], answer)
        source = task["source_arrays"]
        if arrays["species"] != source["species"]:
            raise RuntimeError("backfill action changed atom-type sequence")
        if arrays["lengths"] != source["lengths"] or arrays["angles"] != source["angles"]:
            raise RuntimeError("backfill action changed frozen lattice")
        graph, cif = graph_from_arrays(arrays, process_one)
        graph["sample_idx"] = flat_idx
        graph["group_idx"] = group_idx
        graph["candidate_idx"] = int(candidate_idx)
        record.update(
            {
                "valid_action": True,
                "parsed": True,
                "graphable": True,
                "cif": cif,
                "failure": None,
            }
        )
        return record, graph
    except RuntimeError:
        raise
    except Exception as exc:
        record.update(
            {
                "valid_action": False,
                "parsed": False,
                "graphable": False,
                "cif": None,
                "failure": f"{type(exc).__name__}:{exc}",
            }
        )
        return record, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--predictor-body", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=94017)
    args = parser.parse_args()
    if int(args.batch_size) <= 0:
        raise ValueError("batch-size must be positive")

    dist_info = init_distributed()
    rank = int(dist_info["rank"])
    world_size = int(dist_info["world_size"])
    distributed = bool(dist_info["distributed"])
    is_main = bool(dist_info["is_main"])
    if is_main:
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=False)
    if distributed:
        dist.barrier()

    all_rows = list(iter_jsonl(args.predictor_body))
    if len({int(row["sample_idx"]) for row in all_rows}) != len(all_rows):
        raise ValueError("predictor body contains duplicate sample_idx")
    rows = all_rows[rank::world_size] if distributed else all_rows
    tasks = [task for row in rows if (task := prepare_task(row)) is not None]
    task_by_group = {int(task["group_idx"]): task for task in tasks}
    process_one = import_process_one(args.crysllmgen_dir)
    device = dist_info["device"]
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.checkpoint_path, device
    )
    tokenizer_report = validate_dynamic_tokenizer_contract(
        tokenizer, mask_token_id=MASK_TOKEN_ID
    )
    tasks.sort(key=lambda item: (int(item["plan_state"]["N"]), int(item["group_idx"])))
    results: dict[int, dict[str, Any]] = {}
    graphs: list[dict[str, Any]] = []
    started = time.time()
    offset = 0
    progress = tqdm(total=len(tasks), desc="SPAD-E K4 actions")
    while offset < len(tasks):
        num_atoms = int(tasks[offset]["plan_state"]["N"])
        batch: list[dict[str, Any]] = []
        while (
            offset < len(tasks)
            and len(batch) < int(args.batch_size)
            and int(tasks[offset]["plan_state"]["N"]) == num_atoms
        ):
            batch.append(tasks[offset])
            offset += 1
        gen_length = exact_body_token_count(num_atoms)
        encoded = tokenizer(
            [task["prompt"] for task in batch],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(model_device(model))
        attention_mask = encoded["attention_mask"].to(model_device(model))
        answer_ids = []
        for task in batch:
            ids = tokenizer(task["answer"], add_special_tokens=False)["input_ids"]
            if len(ids) != gen_length:
                raise ValueError("predictor answer retokenization changed exact 7+4N")
            answer_ids.append([int(value) for value in ids])
        complete = torch.cat(
            (
                input_ids,
                torch.tensor(answer_ids, dtype=torch.long, device=input_ids.device),
            ),
            dim=1,
        )
        allowed = exact_dynamic_schema_constraints(tokenizer, num_atoms)
        constraints = build_dynamic_lightweight_constraints(
            tokenizer,
            duplicate_coordinate_mask=True,
            lattice_volume_mask=True,
            min_lattice_rad=1.0e-4,
            canonicalize_periodic_alias=True,
            pbc_min_distance_mask=True,
            pbc_min_distance_A=0.5,
            pbc_image_radius=2,
        )

        # Candidate zero is the mandatory source no-op.
        for task in batch:
            record, graph = candidate_record(
                task,
                candidate_idx=0,
                answer=task["answer"],
                revision_log=[],
                process_one=process_one,
            )
            results[int(record["sample_idx"])] = record
            if graph is not None:
                graphs.append(graph)

        for candidate_idx in (1, 2, 3):
            batch_key = min(int(task["group_idx"]) for task in batch)
            sample_seed = int(args.seed) + candidate_idx * 1_000_003 + batch_key
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)
            revised, logs = revise_spad_anchors(
                model,
                complete,
                prompt_length=int(input_ids.shape[1]),
                gen_length=gen_length,
                revision_slots_by_batch=[
                    [int(task["backfill_slot"])] for task in batch
                ],
                attention_mask=attention_mask,
                temperature=0.7,
                cfg_scale=0.0,
                remasking="low_confidence",
                mask_id=MASK_TOKEN_ID,
                allowed_token_ids_by_generation_pos=allowed,
                atom_count_grammar=None,
                lightweight_decoding_constraints=constraints,
                suffix_visible=True,
                strict_pbc_no_legal_fallback=True,
            )
            decoded = tokenizer.batch_decode(
                revised[:, input_ids.shape[1] :],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for task, answer, revision_log in zip(batch, decoded, logs, strict=True):
                if len(revision_log) != 1:
                    raise RuntimeError("SPAD-E action did not execute one backfill transaction")
                record, graph = candidate_record(
                    task,
                    candidate_idx=candidate_idx,
                    answer=answer,
                    revision_log=revision_log,
                    process_one=process_one,
                )
                results[int(record["sample_idx"])] = record
                if graph is not None:
                    graphs.append(graph)
        progress.update(len(batch))
    progress.close()

    # Predictor failures remain four invalid action rows with no energy support.
    for source in rows:
        group_idx = int(source["sample_idx"])
        if group_idx in task_by_group:
            continue
        for candidate_idx in range(4):
            flat_idx = group_idx * 4 + candidate_idx
            results[flat_idx] = {
                "schema": SCHEMA,
                "sample_idx": flat_idx,
                "group_idx": group_idx,
                "candidate_idx": candidate_idx,
                "mandatory_noop": candidate_idx == 0,
                "plan_state": source.get("plan_state"),
                "prompt": source.get("conditioning_prompt"),
                "source_answer": source.get("text"),
                "answer": None,
                "species_program": (source.get("prompt_record") or {}).get("species_program"),
                "backfill_slot": None,
                "suffix_visible": True,
                "revision_log": [],
                "validity_before_energy": True,
                "valid_action": False,
                "parsed": False,
                "graphable": False,
                "cif": None,
                "failure": "predictor_body_invalid",
                "outcomes_read": False,
                "selection_rerank_replacement": False,
            }
    group_indices = sorted(int(row["sample_idx"]) for row in rows)
    expected = {
        group_idx * 4 + candidate_idx
        for group_idx in group_indices
        for candidate_idx in range(4)
    }
    if set(results) != expected:
        raise RuntimeError("SPAD-E action accounting changed")
    ordered = [results[index] for index in sorted(results)]
    candidate_path = rank_path(
        args.output_dir, "candidate_actions.jsonl", rank, distributed
    )
    graph_path = rank_path(args.output_dir, "proposal_graphs.pt", rank, distributed)
    metrics_path = rank_path(args.output_dir, "sample_metrics.json", rank, distributed)
    with candidate_path.open("x", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    torch.save(sorted(graphs, key=lambda graph: int(graph["sample_idx"])), graph_path)
    failures = Counter(str(row.get("failure")) for row in ordered if not row["valid_action"])
    duplicate_groups = 0
    for group_idx in group_indices:
        answers = [results[group_idx * 4 + index].get("answer") for index in range(4)]
        duplicate_groups += int(len(set(answers)) < 4)
    rank_report = {
        "schema": SCHEMA,
        "rank": rank,
        "world_size": world_size,
        "assigned_groups": len(rows),
        "K": 4,
        "assigned_candidates": len(ordered),
        "valid_actions": sum(row["valid_action"] is True for row in ordered),
        "valid_noops": sum(
            row["candidate_idx"] == 0 and row["valid_action"] is True
            for row in ordered
        ),
        "groups_with_duplicate_draws": duplicate_groups,
        "failures": dict(failures),
        "mandatory_noop": True,
        "suffix_visible": True,
        "strict_validity_before_energy": True,
        "selection_rerank_replacement": False,
        "temperature": 0.7,
        "seed": int(args.seed),
        "elapsed_sec": time.time() - started,
    }
    metrics_path.write_text(
        json.dumps(rank_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if distributed:
        dist.barrier()
    if is_main:
        if distributed:
            merged_rows: list[dict[str, Any]] = []
            merged_graphs: list[dict[str, Any]] = []
            rank_reports = []
            for source_rank in range(world_size):
                merged_rows.extend(
                    iter_jsonl(
                        rank_path(
                            args.output_dir,
                            "candidate_actions.jsonl",
                            source_rank,
                            True,
                        )
                    )
                )
                merged_graphs.extend(
                    torch.load(
                        rank_path(
                            args.output_dir,
                            "proposal_graphs.pt",
                            source_rank,
                            True,
                        ),
                        map_location="cpu",
                    )
                )
                rank_reports.append(
                    json.loads(
                        rank_path(
                            args.output_dir,
                            "sample_metrics.json",
                            source_rank,
                            True,
                        ).read_text(encoding="utf-8")
                    )
                )
            merged_rows.sort(key=lambda row: int(row["sample_idx"]))
            merged_graphs.sort(key=lambda graph: int(graph["sample_idx"]))
            expected_all = set(range(len(all_rows) * 4))
            if {int(row["sample_idx"]) for row in merged_rows} != expected_all:
                raise RuntimeError("distributed SPAD-E merge changed action accounting")
            with (args.output_dir / "candidate_actions.jsonl").open(
                "x", encoding="utf-8"
            ) as handle:
                for row in merged_rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            torch.save(merged_graphs, args.output_dir / "proposal_graphs.pt")
        else:
            merged_rows = ordered
            merged_graphs = graphs
            rank_reports = [rank_report]
        report = {
            "schema": SCHEMA,
            "groups": len(all_rows),
            "K": 4,
            "candidates": len(merged_rows),
            "valid_actions": sum(row["valid_action"] is True for row in merged_rows),
            "valid_noops": sum(
                row["candidate_idx"] == 0 and row["valid_action"] is True
                for row in merged_rows
            ),
            "groups_with_duplicate_draws": sum(
                int(item["groups_with_duplicate_draws"]) for item in rank_reports
            ),
            "failures": dict(
                Counter(
                    str(row.get("failure"))
                    for row in merged_rows
                    if not row["valid_action"]
                )
            ),
            "mandatory_noop": True,
            "suffix_visible": True,
            "strict_validity_before_energy": True,
            "selection_rerank_replacement": False,
            "temperature": 0.7,
            "seed": int(args.seed),
            "distributed": distributed,
            "world_size": world_size,
            "elapsed_sec": max(float(item["elapsed_sec"]) for item in rank_reports),
        }
        (args.output_dir / "sample_metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output_dir / "tokenizer_report.json").write_text(
            json.dumps(tokenizer_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "_SUCCESS").touch()
        print(json.dumps(report, sort_keys=True))
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
