#!/usr/bin/env python3
"""Run the frozen 8-Plan/256-completion CTV-DLM resource canary."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist

from crystal_dlm.ctv_branching import validate_canary_layout  # noqa: E402
from crystal_dlm.ctv_protocol import branch_record_id, counter_seed  # noqa: E402
from crystal_dlm.ctv_rollout import (  # noqa: E402
    collect_ctv_branch_states,
    complete_ctv_forced_branches,
)
from crystal_dlm.fixed_slot import MASK_TOKEN_ID  # noqa: E402
from crystal_dlm.r5_dynamic_length import (  # noqa: E402
    count_prefill_for_batch,
    exact_body_token_count,
    exact_dynamic_generation_schedule,
    exact_dynamic_schema_constraints,
    validate_answer_matches_plan,
)
from scripts.sample_llada_dynamic_crystals import (  # noqa: E402
    build_dynamic_lightweight_constraints,
    graph_from_arrays,
    import_process_one,
    init_distributed,
    load_model_and_tokenizer,
    rank_path,
)
from scripts.sample_llada_r5_exact_length import element_prefill_for_batch  # noqa: E402


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 8:
        raise ValueError(f"CTV resource canary requires exactly 8 Plans, got {len(rows)}")
    identities = [str(row.get("reduced_composition_identity") or "") for row in rows]
    if any(not value for value in identities) or len(set(identities)) != 8:
        raise ValueError("CTV canary reduced composition identities changed")
    for index, row in enumerate(rows):
        if row.get("minimal_spec_schema") != "h1a2_ctv_minimal_spec_v1":
            raise ValueError(f"CTV canary row {index} lacks the frozen minimal spec")
        if not isinstance(row.get("plan_state"), Mapping):
            raise ValueError(f"CTV canary row {index} lacks plan_state")
        prompt = str(row.get("prompt") or "")
        if not prompt.endswith("\ndynamic_crystal_body:"):
            raise ValueError(f"CTV canary row {index} prompt serialization changed")
    return rows


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def state_id(*, composition_id: str, sample_idx: int, milestone: float) -> str:
    payload = f"{composition_id}|{int(sample_idx)}|{float(milestone):.2f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def merge_outputs(output_dir: Path, world_size: int) -> None:
    branches: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    rank_metrics: list[dict[str, Any]] = []
    for rank in range(int(world_size)):
        branches.extend(
            json.loads(line)
            for line in rank_path(output_dir, "branches.jsonl", rank, True)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        states.extend(
            json.loads(line)
            for line in rank_path(output_dir, "states.jsonl", rank, True)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        rank_metrics.append(
            json.loads(
                rank_path(output_dir, "resource_metrics.json", rank, True).read_text(
                    encoding="utf-8"
                )
            )
        )
        graph_path = rank_path(output_dir, "proposal_graphs.pt", rank, True)
        if graph_path.is_file():
            graphs.extend(torch.load(graph_path, map_location="cpu"))
    branches.sort(key=lambda row: int(row["branch_ordinal"]))
    states.sort(key=lambda row: (int(row["canary_plan_idx"]), float(row["milestone"])))
    graphs.sort(key=lambda row: int(row["sample_idx"]))
    if [int(row["branch_ordinal"]) for row in branches] != list(range(256)):
        raise ValueError("CTV merged branch ordinals are not exactly 0..255")
    layout = validate_canary_layout(branches, expected_plans=8)
    if len(states) != 16 or len({str(row["state_id"]) for row in states}) != 16:
        raise ValueError("CTV merged state ledger must contain 16 unique states")
    write_jsonl(output_dir / "branches.jsonl", branches)
    write_jsonl(output_dir / "states.jsonl", states)
    torch.save(graphs, output_dir / "proposal_graphs.pt")
    total_seconds = sum(float(row["elapsed_seconds"]) for row in rank_metrics)
    maximum_seconds = max(float(row["elapsed_seconds"]) for row in rank_metrics)
    manifest = {
        "schema": "h1a2_ctv_resource_canary_v1",
        "engineering_only": True,
        "method_adaptation_from_outcomes_forbidden": True,
        "layout": layout,
        "states": len(states),
        "decoded_branches": len(branches),
        "parse_success": sum(row.get("parsed") is True for row in branches),
        "graph_success": len(graphs),
        "rank_metrics": rank_metrics,
        "summed_rank_seconds": total_seconds,
        "wall_seconds_max_rank": maximum_seconds,
        "branches_per_wall_second": 256.0 / max(maximum_seconds, 1e-12),
        "max_rss_kib": max(int(row["max_rss_kib"]) for row in rank_metrics),
        "max_cuda_allocated_bytes": max(
            int(row["max_cuda_allocated_bytes"]) for row in rank_metrics
        ),
        "max_cuda_reserved_bytes": max(
            int(row["max_cuda_reserved_bytes"]) for row in rank_metrics
        ),
        "branch_file_sha256": hashlib.sha256(
            (output_dir / "branches.jsonl").read_bytes()
        ).hexdigest(),
        "state_file_sha256": hashlib.sha256(
            (output_dir / "states.jsonl").read_bytes()
        ).hexdigest(),
    }
    (output_dir / "CTV_RESOURCE_CANARY_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# CTV-DLM resource canary",
        "",
        "This run is engineering-only; energy, rank and success outcomes cannot alter the method.",
        "",
        f"- Branch ledger: `{len(branches)}/256`",
        f"- State ledger: `{len(states)}/16`",
        f"- Parse / graph: `{manifest['parse_success']}/{manifest['graph_success']}`",
        f"- Peak host RSS: `{manifest['max_rss_kib']} KiB`",
        f"- Peak CUDA allocated/reserved: `{manifest['max_cuda_allocated_bytes']}/"
        f"{manifest['max_cuda_reserved_bytes']} bytes`",
        f"- Wall time (slowest rank): `{maximum_seconds:.3f} s`",
    ]
    (output_dir / "CTV_RESOURCE_CANARY_MANIFEST.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output_dir / "_DLM_SUCCESS").touch()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, required=True)
    parser.add_argument("--canary-jsonl", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=61017)
    args = parser.parse_args()

    distributed = init_distributed()
    rank = int(distributed["rank"])
    world_size = int(distributed["world_size"])
    if world_size != 4:
        raise ValueError("CTV resource canary requires exactly four GPU ranks")
    is_main = bool(distributed["is_main"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.checkpoint_path,
        distributed["device"],
    )
    process_one = import_process_one(args.crysllmgen_dir)
    rows = read_rows(args.canary_jsonl)
    assigned = [(index, row) for index, row in enumerate(rows) if index % world_size == rank]
    if len(assigned) != 2:
        raise RuntimeError("CTV canary must assign exactly two Plans to each rank")
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    branch_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    proposal_graphs: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()

    for canary_plan_idx, row in assigned:
        plan = dict(row["plan_state"])
        sample_idx = int(row.get("sample_idx", canary_plan_idx))
        composition_id = str(row["reduced_composition_identity"])
        num_atoms = int(plan["N"])
        prompt_text = str(row["prompt"]).rstrip() + "\n"
        encoded = tokenizer(
            [prompt_text], add_special_tokens=False, padding=True, return_tensors="pt"
        )
        prompt = encoded["input_ids"].to(distributed["device"])
        attention = encoded["attention_mask"].to(distributed["device"])
        gen_length = exact_body_token_count(num_atoms)
        allowed = exact_dynamic_schema_constraints(tokenizer, num_atoms)
        prefill = count_prefill_for_batch(tokenizer, num_atoms, 1)
        prefill.update(element_prefill_for_batch(tokenizer, [plan]))
        schedule = exact_dynamic_generation_schedule(num_atoms)
        lightweight = build_dynamic_lightweight_constraints(
            tokenizer,
            duplicate_coordinate_mask=True,
            lattice_volume_mask=True,
            min_lattice_rad=1e-4,
        )
        base_noise_group = counter_seed(
            "ctv-base-state-v1", int(args.seed), composition_id, sample_idx
        )
        _base_tokens, snapshots = collect_ctv_branch_states(
            model,
            prompt,
            attention_mask=attention,
            num_atoms=num_atoms,
            gen_length=gen_length,
            temperature=float(args.temperature),
            mask_id=MASK_TOKEN_ID,
            allowed_token_ids_by_generation_pos=allowed,
            prefill_token_ids_by_generation_pos=prefill,
            generation_position_groups=schedule,
            lightweight_decoding_constraints=lightweight,
            base_noise_group=base_noise_group,
        )
        for milestone_idx, snapshot in enumerate(snapshots):
            milestone = float(snapshot["milestone"])
            sid = state_id(
                composition_id=composition_id,
                sample_idx=sample_idx,
                milestone=milestone,
            )
            state_rows.append(
                {
                    "schema": "h1a2_ctv_branch_state_v1",
                    "state_id": sid,
                    "canary_plan_idx": canary_plan_idx,
                    "composition_id": composition_id,
                    "sample_idx": sample_idx,
                    "milestone": milestone,
                    "visible_free_geometry_fraction": float(
                        snapshot["visible_free_geometry_fraction"]
                    ),
                    "intervention_position": int(snapshot["position"]),
                    "eligible_schedule_positions": [
                        int(value)
                        for value in snapshot["eligible_schedule_positions"]
                    ],
                    "frozen_base_confidence": float(snapshot["confidence"]),
                    "action_selection": str(snapshot["action_selection"]),
                    "action_token_ids": [int(v) for v in snapshot["action_token_ids"]],
                    "action_probabilities": [
                        float(v) for v in snapshot["action_probabilities"]
                    ],
                    "next_denoise_step": int(snapshot["next_denoise_step"]),
                    "state_token_ids": snapshot["tokens"][0].detach().cpu().tolist(),
                }
            )
            continuation_seeds = (
                counter_seed(
                    "ctv-canary-cont-v1",
                    args.seed,
                    composition_id,
                    sample_idx,
                    milestone,
                    0,
                ),
                counter_seed(
                    "ctv-canary-cont-v1",
                    args.seed,
                    composition_id,
                    sample_idx,
                    milestone,
                    1,
                ),
            )
            completed, layout = complete_ctv_forced_branches(
                model,
                snapshot,
                composition_id=composition_id,
                sample_idx=sample_idx,
                continuation_seeds=continuation_seeds,
                gen_length=gen_length,
                temperature=float(args.temperature),
                mask_id=MASK_TOKEN_ID,
                allowed_token_ids_by_generation_pos=allowed,
                generation_position_groups=schedule,
                lightweight_decoding_constraints=lightweight,
            )
            decoded = tokenizer.batch_decode(
                completed[:, prompt.shape[1] :],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            probability_by_action = dict(
                zip(
                    [int(v) for v in snapshot["action_token_ids"]],
                    [float(v) for v in snapshot["action_probabilities"]],
                    strict=True,
                )
            )
            for local_branch_idx, (metadata, text) in enumerate(zip(layout, decoded, strict=True)):
                branch_ordinal = canary_plan_idx * 32 + milestone_idx * 16 + local_branch_idx
                branch_id = branch_record_id(
                    composition_id=composition_id,
                    sample_idx=sample_idx,
                    milestone=milestone,
                    position=int(snapshot["position"]),
                    action_token=int(metadata["action_token"]),
                    continuation_seed=int(metadata["continuation_seed"]),
                )
                record = {
                    "schema": "h1a2_ctv_branch_completion_v1",
                    **metadata,
                    "state_id": sid,
                    "branch_id": branch_id,
                    "branch_ordinal": branch_ordinal,
                    "canary_plan_idx": canary_plan_idx,
                    "base_action_probability": probability_by_action[
                        int(metadata["action_token"])
                    ],
                    "text": text,
                    "parsed": False,
                }
                try:
                    arrays = validate_answer_matches_plan(plan, text)
                    graph, cif = graph_from_arrays(arrays, process_one)
                    graph["sample_idx"] = branch_ordinal
                    graph["branch_id"] = branch_id
                    proposal_graphs.append(graph)
                    record.update(
                        {
                            "parsed": True,
                            "num_atoms": int(arrays["num_atoms"]),
                            "cif": cif,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - failures stay in denominator.
                    reason = f"{type(exc).__name__}:{str(exc)}"
                    failures[reason] += 1
                    record["failure"] = reason
                branch_rows.append(record)

    branch_rows.sort(key=lambda row: int(row["branch_ordinal"]))
    state_rows.sort(key=lambda row: (int(row["canary_plan_idx"]), float(row["milestone"])))
    write_jsonl(rank_path(args.output_dir, "branches.jsonl", rank, True), branch_rows)
    write_jsonl(rank_path(args.output_dir, "states.jsonl", rank, True), state_rows)
    torch.save(
        proposal_graphs,
        rank_path(args.output_dir, "proposal_graphs.pt", rank, True),
    )
    elapsed = time.time() - started
    metrics = {
        "rank": rank,
        "plans": len(assigned),
        "states": len(state_rows),
        "branches": len(branch_rows),
        "parse_success": sum(row["parsed"] is True for row in branch_rows),
        "graph_success": len(proposal_graphs),
        "failures": dict(failures.most_common()),
        "elapsed_seconds": elapsed,
        "branches_per_second": len(branch_rows) / max(elapsed, 1e-12),
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    rank_path(args.output_dir, "resource_metrics.json", rank, True).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dist.barrier()
    if is_main:
        merge_outputs(args.output_dir, world_size)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
