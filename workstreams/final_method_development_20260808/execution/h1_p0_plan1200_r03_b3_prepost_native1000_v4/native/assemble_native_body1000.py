#!/usr/bin/env python3
"""Select the first 1,000 body successes from prefix plus frozen reserve."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from native_protocol import (
    NATIVE_DENOMINATOR,
    PREFIX_COUNT,
    candidate_seed,
    first_success_ranks,
    identity,
    ordered_candidate_rows,
    read_json,
    read_jsonl,
    sha256_file,
    validate_arm,
    validate_repeat,
    write_json_exclusive,
    write_jsonl_exclusive,
)


def _load_graphs(path: Path, *, arm: str, repeat: int) -> dict[int, Mapping[str, Any]]:
    payload = torch.load(path.resolve(), map_location="cpu")
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected graph list")
    graphs: dict[int, Mapping[str, Any]] = {}
    for wrapper in payload:
        if not isinstance(wrapper, Mapping):
            raise ValueError(f"{path}: malformed graph wrapper")
        rank = int(wrapper.get("candidate_rank", wrapper.get("ordinal", -1)))
        graph = wrapper.get("graph")
        if (
            rank in graphs
            or str(wrapper.get("arm")) != arm
            or int(wrapper.get("repeat", -1)) != repeat
            or not isinstance(graph, Mapping)
        ):
            raise ValueError(f"{path}: graph identity changed at rank {rank}")
        graphs[rank] = graph
    return graphs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--main-body-dir", type=Path, required=True)
    parser.add_argument("--reserve-body-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    arm = validate_arm(args.arm)
    repeat = validate_repeat(args.repeat)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)

    candidates = ordered_candidate_rows(read_jsonl(args.candidate_pool.resolve()))
    main_root = args.main_body_dir.resolve()
    reserve_root = args.reserve_body_dir.resolve()
    main_report = read_json(main_root / "generation_report.json")
    reserve_report = read_json(reserve_root / "reserve_generation_report.json")
    main_attempts = sorted(
        read_jsonl(main_root / "body_attempts.jsonl"),
        key=lambda row: int(row["ordinal"]),
    )
    reserve_attempts = sorted(
        read_jsonl(reserve_root / "reserve_body_attempts.jsonl"),
        key=lambda row: int(row["candidate_rank"]),
    )
    if (
        len(main_attempts) != PREFIX_COUNT
        or [int(row.get("ordinal", -1)) for row in main_attempts]
        != list(range(PREFIX_COUNT))
        or int(main_report.get("attempts", -1)) != PREFIX_COUNT
        or str(main_report.get("arm")) != arm
        or int(main_report.get("repeat", -1)) != repeat
        or int(reserve_report.get("attempts", -1)) != len(candidates) - PREFIX_COUNT
        or str(reserve_report.get("arm")) != arm
        or int(reserve_report.get("repeat", -1)) != repeat
        or [int(row.get("candidate_rank", -1)) for row in reserve_attempts]
        != list(range(PREFIX_COUNT, len(candidates)))
    ):
        raise ValueError("prefix or reserve body denominator changed")

    attempts: dict[int, dict[str, Any]] = {}
    for rank, row in enumerate(main_attempts):
        record = dict(row)
        if (
            int(record.get("planner_candidate_ordinal", -1))
            != int(candidates[rank]["planner_candidate_ordinal"])
            or record.get("plan_state_sha256")
            != candidates[rank].get("plan_state_sha256")
            or int(record.get("body_noise_seed", -1))
            != candidate_seed(repeat, rank, "body")
            or int(record.get("refiner_noise_seed", -1))
            != candidate_seed(repeat, rank, "refiner")
        ):
            raise ValueError(f"main body candidate identity changed at rank {rank}")
        record["candidate_rank"] = rank
        record["candidate_partition"] = "v3_prefix"
        attempts[rank] = record
    for row in reserve_attempts:
        rank = int(row["candidate_rank"])
        if (
            rank in attempts
            or int(row.get("planner_candidate_ordinal", -1))
            != int(candidates[rank]["planner_candidate_ordinal"])
            or int(row.get("body_noise_seed", -1))
            != candidate_seed(repeat, rank, "body")
            or int(row.get("refiner_noise_seed", -1))
            != candidate_seed(repeat, rank, "refiner")
        ):
            raise ValueError(f"reserve body candidate identity changed at rank {rank}")
        record = dict(row)
        record["candidate_partition"] = "frozen_reserve"
        attempts[rank] = record
    if sorted(attempts) != list(range(len(candidates))):
        raise ValueError("combined body ledger lost or duplicated a candidate")

    main_graphs = _load_graphs(
        main_root / "proposal_graphs.pt", arm=arm, repeat=repeat
    )
    reserve_graphs = _load_graphs(
        reserve_root / "reserve_proposal_graphs.pt", arm=arm, repeat=repeat
    )
    graphs = {**main_graphs, **reserve_graphs}
    all_success_ranks = [
        rank for rank in range(len(candidates)) if attempts[rank].get("status") == "succeeded"
    ]
    if len(all_success_ranks) < NATIVE_DENOMINATOR:
        evidence = {
            "schema": "h1_plan1200_native_body1000_selection_failure_v1",
            "status": "failed_closed",
            "arm": arm,
            "repeat": repeat,
            "candidate_count": len(candidates),
            "body_successes": len(all_success_ranks),
            "required": NATIVE_DENOMINATOR,
            "same_plan_retry": False,
            "additional_planner_draw": False,
        }
        output.mkdir(parents=True)
        write_json_exclusive(output / "selection_failure.json", evidence)
        raise RuntimeError("frozen raw-1200 planner draw cannot supply 1,000 body successes")
    selected_ranks = first_success_ranks(attempts, len(candidates))
    stop_rank = selected_ranks[-1]
    if set(graphs) != set(all_success_ranks):
        raise ValueError("proposal graphs do not exactly match all body successes")

    selected_rows: list[dict[str, Any]] = []
    selected_graphs: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for native_ordinal, rank in enumerate(selected_ranks):
        source = dict(attempts[rank])
        if source.get("status") != "succeeded" or source.get("body_graph_complete") is not True:
            raise ValueError(f"selected candidate {rank} is not a body success")
        selected = {
            **source,
            "schema": "h1_plan1200_crysllmgen_native_body_success_v1",
            "ordinal": native_ordinal,
            "sample_idx": native_ordinal,
            "evaluation_order": native_ordinal,
            "pair_id": f"h1-plan1200-native-{arm.lower()}-r{repeat}:{native_ordinal:04d}",
            "candidate_rank": rank,
            "source_candidate_rank": rank,
            "source_candidate_partition": source["candidate_partition"],
            "source_body_noise_seed": int(source["body_noise_seed"]),
            "source_refiner_noise_seed": int(source["refiner_noise_seed"]),
            "crysllmgen_native_selected": True,
            "crysllmgen_native_selection_ordinal": native_ordinal,
            "body_success_conditioned_denominator": NATIVE_DENOMINATOR,
            "retry_used": False,
            "replacement_used": False,
            "repair_used": False,
            "filter_used": False,
            "rerank_used": False,
        }
        selected_rows.append(selected)
        graph = dict(graphs[rank])
        metadata = dict(graph.get("h1_plan1200_prepost_metadata") or {})
        metadata.update(
            {
                "arm": arm,
                "repeat": repeat,
                "ordinal": native_ordinal,
                "candidate_rank": rank,
                "source_candidate_partition": source["candidate_partition"],
                "crysllmgen_native_selected": True,
            }
        )
        graph["h1_plan1200_prepost_metadata"] = metadata
        selected_graphs.append(
            {
                "ordinal": native_ordinal,
                "candidate_rank": rank,
                "arm": arm,
                "repeat": repeat,
                "graph": graph,
            }
        )
        mapping_rows.append(
            {
                "native_ordinal": native_ordinal,
                "candidate_rank": rank,
                "planner_candidate_ordinal": int(source["planner_candidate_ordinal"]),
                "candidate_partition": source["candidate_partition"],
                "plan_state_sha256": source.get("plan_state_sha256"),
                "body_noise_seed": int(source["body_noise_seed"]),
                "refiner_noise_seed": int(source["refiner_noise_seed"]),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    body_path = output / "body_attempts.jsonl"
    graph_path = output / "proposal_graphs.pt"
    mapping_path = output / "selection_mapping.jsonl"
    write_jsonl_exclusive(body_path, selected_rows)
    write_jsonl_exclusive(mapping_path, mapping_rows)
    with graph_path.open("xb") as handle:
        torch.save(selected_graphs, handle)
        handle.flush()
        os.fsync(handle.fileno())
    failure_classes = Counter(
        ":".join(str(attempts[rank].get("reason") or "unknown").split(":")[:2])
        for rank in range(stop_rank + 1)
        if attempts[rank].get("status") != "succeeded"
    )
    report = {
        "schema": "h1_plan1200_crysllmgen_native_body1000_report_v1",
        "status": "complete",
        "arm": arm,
        "repeat": repeat,
        "candidate_pool_count": len(candidates),
        "candidate_attempts_through_completion": stop_rank + 1,
        "last_selected_candidate_rank": stop_rank,
        "body_failures_before_completion": stop_rank + 1 - NATIVE_DENOMINATOR,
        "failure_classes_before_completion": dict(sorted(failure_classes.items())),
        "selected_body_successes": NATIVE_DENOMINATOR,
        "selected_prefix_count": sum(rank < PREFIX_COUNT for rank in selected_ranks),
        "selected_reserve_count": sum(rank >= PREFIX_COUNT for rank in selected_ranks),
        "unused_candidates_after_completion": len(candidates) - stop_rank - 1,
        "selection": "first_1000_body_successes_by_frozen_candidate_order",
        "upstream_crysllmgen_semantics": True,
        "outcome_conditioned_sampling_contract": True,
        "same_plan_retry": False,
        "stochastic_replacement": False,
        "additional_planner_draw": False,
        "all_selected_rows_body_success": True,
        "artifacts": {
            "candidate_pool": identity(args.candidate_pool),
            "main_body_attempts": identity(main_root / "body_attempts.jsonl"),
            "reserve_body_attempts": identity(
                reserve_root / "reserve_body_attempts.jsonl"
            ),
            "selected_body_attempts": identity(body_path),
            "selected_proposal_graphs": identity(graph_path),
            "selection_mapping": identity(mapping_path),
        },
    }
    write_json_exclusive(output / "selection_report.json", report)
    with (output / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
