#!/usr/bin/env python3
"""Generate one H1-A2 factorial body arm from persisted Planner attempts.

This is a fresh runner: it never generates a Plan and never reads a
structure-derived teacher Plan.  B0 uses the historical exact order; B* uses
the PlanGraph schedule compiled from the same persisted model-sampled Plan.
Each ordinal is generated in an isolated batch of one with stateless noise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from tqdm import tqdm

from crystal_dlm.dynamic_crystal import arrays_to_torch_payload, write_json  # noqa: E402
from crystal_dlm.fixed_slot import MASK_TOKEN_ID  # noqa: E402
from crystal_dlm.h1a2_factorial_contract import (  # noqa: E402
    FACTORIAL_ARM_COMPONENTS,
    FACTORIAL_ARMS,
    build_factorial_arm_input,
    build_factorial_ordinal_record,
)
from crystal_dlm.h1a2_factorial_runtime import (  # noqa: E402
    assert_additive_body_tokenization,
    assert_body_tokenizer_identity,
    compile_body_condition,
    load_planner_attempts,
    ordered_single_arm_attempts,
    propagated_planner_failure,
    read_jsonl_objects,
)
from crystal_dlm.llada_generation import generate  # noqa: E402
from crystal_dlm.r5_dynamic_length import (  # noqa: E402
    count_prefill_for_batch,
    exact_dynamic_schema_constraints,
    validate_answer_matches_plan,
)
from crystal_dlm.ordinal_rng import sha256_text  # noqa: E402
from scripts.sample_llada_dynamic_crystals import (  # noqa: E402
    build_dynamic_lightweight_constraints,
    graph_from_arrays,
    import_process_one,
    init_distributed,
    load_model_and_tokenizer,
    rank_path,
)
from scripts.sample_llada_r5_exact_length import (  # noqa: E402
    element_prefill_for_batch,
    merge_prefill_maps,
)


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def _write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _body_metrics(
    attempts: list[Mapping[str, Any]],
    *,
    elapsed: float,
    rank: int,
    world_size: int,
    distributed: bool,
) -> dict[str, Any]:
    failures: dict[str, int] = {}
    planner_complete = 0
    body_generated = 0
    body_complete = 0
    for record in attempts:
        if record.get("earliest_failure_stage") != "planner":
            planner_complete += 1
        if record.get("body_generation_complete") is True:
            body_generated += 1
        if record.get("body_status") == "complete":
            body_complete += 1
        if record.get("attempt_status") == "failed":
            stage = str(record.get("earliest_failure_stage") or "unknown")
            reason = str(record.get("failure_reason") or "unknown")
            key = f"{stage}:{reason}"
            failures[key] = failures.get(key, 0) + 1
    denominator = len(attempts)
    return {
        "schema": "h1a2_factorial_body_metrics_v1",
        "all_attempt_denominator": denominator,
        "planner_complete": planner_complete,
        "body_generated": body_generated,
        "body_complete": body_complete,
        "planner_completion_rate_all_attempt": planner_complete / max(1, denominator),
        "body_generation_rate_all_attempt": body_generated / max(1, denominator),
        "body_completion_rate_all_attempt": body_complete / max(1, denominator),
        "failures": failures,
        "time_sec": float(elapsed),
        "rank": int(rank),
        "world_size": int(world_size),
        "distributed": bool(distributed),
        "effective_generation_batch_size": 1,
        "retry": False,
        "replacement": False,
        "repair": False,
        "filter": False,
        "rerank": False,
    }


def _graph_identity(graph: Mapping[str, Any]) -> tuple[int, str]:
    metadata = graph.get("h1a2_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("proposal graph lacks H1-A2 metadata")
    return int(metadata["sample_idx"]), str(metadata["factorial_arm"])


def _merge_body_outputs(
    output_dir: Path,
    *,
    world_size: int,
    expected_count: int,
    factorial_arm: str,
    elapsed: float,
) -> None:
    attempt_rows: list[Mapping[str, Any]] = []
    arrays: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    for rank in range(int(world_size)):
        attempt_path = rank_path(output_dir, "body_attempts.jsonl", rank, True)
        array_path = rank_path(output_dir, "valid_arrays.jsonl", rank, True)
        graph_path = rank_path(output_dir, "proposal_graphs.pt", rank, True)
        if not attempt_path.exists():
            raise FileNotFoundError(f"missing body rank output: {attempt_path}")
        attempt_rows.extend(read_jsonl_objects(attempt_path))
        if array_path.exists():
            arrays.extend(read_jsonl_objects(array_path))
        if graph_path.exists():
            loaded = torch.load(graph_path, map_location="cpu")
            if not isinstance(loaded, list):
                raise TypeError(f"{graph_path} is not a proposal-graph list")
            graphs.extend(loaded)

    attempts = ordered_single_arm_attempts(
        attempt_rows,
        expected_count=int(expected_count),
        expected_factorial_arm=factorial_arm,
    )
    successful = {
        int(record["sample_idx"])
        for record in attempts
        if record.get("body_status") == "complete"
    }
    arrays_by_idx: dict[int, dict[str, Any]] = {}
    for row in arrays:
        idx = int(row["sample_idx"])
        if row.get("factorial_arm") != factorial_arm:
            raise ValueError("valid array arm mismatch")
        if idx in arrays_by_idx:
            raise ValueError(f"duplicate valid array ordinal {idx}")
        arrays_by_idx[idx] = row
    graphs_by_idx: dict[int, dict[str, Any]] = {}
    for graph in graphs:
        idx, arm = _graph_identity(graph)
        if arm != factorial_arm:
            raise ValueError("proposal graph arm mismatch")
        if idx in graphs_by_idx:
            raise ValueError(f"duplicate proposal graph ordinal {idx}")
        graphs_by_idx[idx] = graph
    if set(arrays_by_idx) != successful:
        raise ValueError("valid arrays do not exactly match successful body attempts")
    if set(graphs_by_idx) != successful:
        raise ValueError("proposal graphs do not exactly match successful body attempts")

    ordered_arrays = [arrays_by_idx[idx] for idx in sorted(successful)]
    ordered_graphs = [graphs_by_idx[idx] for idx in sorted(successful)]
    _write_jsonl(output_dir / "body_attempts.jsonl", list(attempts))
    _write_jsonl(output_dir / "valid_arrays.jsonl", ordered_arrays)
    torch.save(ordered_graphs, output_dir / "proposal_graphs.pt")
    payload = arrays_to_torch_payload(ordered_arrays)
    payload["sample_idx"] = torch.tensor(sorted(successful), dtype=torch.long)
    payload["time"] = float(elapsed)
    torch.save(payload, output_dir / "raw_dlm_samples.pt")
    write_json(
        str(output_dir / "body_metrics.json"),
        _body_metrics(
            list(attempts),
            elapsed=elapsed,
            rank=0,
            world_size=int(world_size),
            distributed=True,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/",
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--planner-attempts", type=Path, required=True)
    parser.add_argument("--factorial-arm", choices=FACTORIAL_ARMS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--crysllmgen-dir",
        type=Path,
        default=PROJECT_ROOT / "reference/crysllmgen",
    )
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--base-seed", type=int, default=17)
    parser.add_argument("--expected-tokenizer-vocab-sha256", required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--min-lattice-rad", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.num_samples) <= 0:
        raise ValueError("--num-samples must be positive")
    planner_arm, body_arm = FACTORIAL_ARM_COMPONENTS[args.factorial_arm]

    dist_info = init_distributed()
    rank = int(dist_info["rank"])
    world_size = int(dist_info["world_size"])
    distributed = bool(dist_info["distributed"])
    is_main = bool(dist_info["is_main"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    planner_attempts = load_planner_attempts(
        args.planner_attempts,
        expected_count=int(args.num_samples),
        expected_planner_arm=planner_arm,
    )
    process_one = import_process_one(args.crysllmgen_dir)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.checkpoint_path,
        dist_info["device"],
    )
    tokenizer_identity = assert_body_tokenizer_identity(
        tokenizer,
        expected_vocab_sha256=args.expected_tokenizer_vocab_sha256,
    )
    if is_main:
        write_json(
            str(args.output_dir / "body_tokenizer_identity.json"),
            tokenizer_identity,
        )
        write_json(
            str(args.output_dir / "run_config.json"),
            {
                "schema": "h1a2_factorial_body_run_v1",
                "model_path": args.model_path,
                "checkpoint_path": args.checkpoint_path,
                "checkpoint_sha256": args.checkpoint_sha256,
                "planner_attempts": str(args.planner_attempts),
                "factorial_arm": args.factorial_arm,
                "planner_arm": planner_arm,
                "body_arm": body_arm,
                "num_samples": int(args.num_samples),
                "base_seed": int(args.base_seed),
                "temperature": float(args.temperature),
                "cfg_scale": float(args.cfg_scale),
                "remasking": args.remasking,
                "generation_policy": "d1" if body_arm == "B0" else "d2",
                "schema_logit_mask": True,
                "prefill_count_token": True,
                "freeze_plan_composition": True,
                "duplicate_coordinate_mask": True,
                "lattice_volume_mask": True,
                "min_lattice_rad": float(args.min_lattice_rad),
                "effective_generation_batch_size": 1,
                "seed_mode": "stateless_ordinal_v1",
                "distributed": distributed,
                "world_size": world_size,
                "retry": False,
                "replacement": False,
                "repair": False,
                "filter": False,
                "rerank": False,
            },
        )

    assigned = [
        sample_idx
        for sample_idx in range(int(args.num_samples))
        if sample_idx % world_size == rank
    ]
    attempts: list[dict[str, Any]] = []
    valid_arrays: list[dict[str, Any]] = []
    proposal_graphs: list[dict[str, Any]] = []
    lightweight_constraints = build_dynamic_lightweight_constraints(
        tokenizer,
        duplicate_coordinate_mask=True,
        lattice_volume_mask=True,
        min_lattice_rad=float(args.min_lattice_rad),
    )

    start = time.time()
    progress = tqdm(
        assigned,
        desc=f"H1-A2 {args.factorial_arm} body rank{rank}",
        disable=distributed and not is_main,
    )
    for sample_idx in progress:
        ordinal = build_factorial_ordinal_record(
            int(args.base_seed),
            sample_idx=int(sample_idx),
        )
        planner_attempt = planner_attempts[sample_idx]
        if planner_attempt.get("attempt_status") == "failed":
            failure = propagated_planner_failure(
                planner_attempt,
                factorial_arm=args.factorial_arm,
                ordinal_record=ordinal,
            )
            failure.update(
                {
                    "body_status": "not_started",
                    "body_generation_complete": False,
                    "body_plan_match": False,
                    "body_graph_complete": False,
                }
            )
            attempts.append(failure)
            continue

        stage = "body_contract"
        base_attempt: dict[str, Any] = {
            "runtime_schema": "h1a2_factorial_runtime_v1",
            "sample_idx": int(sample_idx),
            "evaluation_order": int(ordinal["evaluation_order"]),
            "factorial_arm": args.factorial_arm,
            "planner_arm": planner_arm,
            "body_arm": body_arm,
            "attempt_status": "failed",
            "earliest_failure_stage": stage,
            "planner_sampling_seed": int(ordinal["planner_sampling_seed"]),
            "body_sampling_seed": int(ordinal["body_sampling_seed"]),
            "refiner_sampling_seed": int(ordinal["refiner_sampling_seed"]),
            "body_status": "failed",
            "body_generation_complete": False,
            "body_plan_match": False,
            "body_graph_complete": False,
            "retry_used": False,
            "replacement_used": False,
            "repair_used": False,
            "filter_used": False,
            "rerank_used": False,
        }
        try:
            body_input = build_factorial_arm_input(
                planner_attempt,
                factorial_arm=args.factorial_arm,
                ordinal_record=ordinal,
            )
            condition = compile_body_condition(body_input)
            base_attempt.update(
                {
                    "raw_plan_text_sha256": condition["raw_plan_text_sha256"],
                    "plan_text_sha256": condition["plan_text_sha256"],
                    "body_prompt_sha256": condition["body_prompt_sha256"],
                    "generation_policy": condition["generation_policy"],
                    "generation_schedule_sha256": condition[
                        "generation_schedule_sha256"
                    ],
                    "compiled_plangraph_sha256": condition[
                        "compiled_plangraph_sha256"
                    ],
                    "plan_condition_sha256": condition["plan_condition_sha256"],
                }
            )

            stage = "body_generation"
            body_seed = int(condition["body_sampling_seed"])
            torch.manual_seed(body_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(body_seed)
            prompt = str(condition["body_prompt"])
            encoded = tokenizer(
                [prompt],
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(model_device(model))
            attention_mask = encoded["attention_mask"].to(model_device(model))
            num_atoms = int(condition["plan_state"]["N"])
            gen_length = int(condition["expected_answer_token_count"])
            allowed = exact_dynamic_schema_constraints(tokenizer, num_atoms)
            prefill = merge_prefill_maps(
                count_prefill_for_batch(tokenizer, num_atoms, 1),
                element_prefill_for_batch(tokenizer, [condition["plan_state"]]),
            )
            outputs = generate(
                model,
                input_ids,
                attention_mask=attention_mask,
                steps=gen_length,
                gen_length=gen_length,
                block_length=1,
                temperature=float(args.temperature),
                cfg_scale=float(args.cfg_scale),
                remasking=args.remasking,
                mask_id=MASK_TOKEN_ID,
                allowed_token_ids_by_generation_pos=allowed,
                prefill_token_ids_by_generation_pos=prefill,
                generation_position_groups=condition["generation_schedule"],
                lightweight_decoding_constraints=lightweight_constraints,
            )
            generated_ids = outputs[0, input_ids.shape[1] :]
            body_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            base_attempt["body_generation_complete"] = True
            base_attempt["raw_body_token_ids"] = [
                int(value) for value in generated_ids.tolist()
            ]
            base_attempt["raw_body_text"] = body_text
            base_attempt["raw_body_text_sha256"] = sha256_text(body_text)

            stage = "body_tokenization"
            tokenization = assert_additive_body_tokenization(
                tokenizer,
                prompt=prompt,
                answer=body_text,
                generated_token_ids=base_attempt["raw_body_token_ids"],
                expected_answer_token_count=gen_length,
            )
            base_attempt["body_tokenization"] = tokenization

            stage = "body_parse"
            arrays = validate_answer_matches_plan(
                condition["plan_state"],
                body_text,
            )
            base_attempt["body_plan_match"] = True

            stage = "body_graph"
            graph, cif = graph_from_arrays(arrays, process_one)
            metadata = {
                "sample_idx": int(sample_idx),
                "evaluation_order": int(ordinal["evaluation_order"]),
                "factorial_arm": args.factorial_arm,
                "planner_arm": planner_arm,
                "body_arm": body_arm,
                "raw_plan_text_sha256": condition["raw_plan_text_sha256"],
                "plan_text_sha256": condition["plan_text_sha256"],
                "body_prompt_sha256": condition["body_prompt_sha256"],
                "generation_schedule_sha256": condition[
                    "generation_schedule_sha256"
                ],
                "compiled_plangraph_sha256": condition[
                    "compiled_plangraph_sha256"
                ],
                "body_sampling_seed": body_seed,
                "refiner_sampling_seed": int(condition["refiner_sampling_seed"]),
            }
            graph["h1a2_metadata"] = metadata
            proposal_graphs.append(graph)
            array_record = {
                **arrays,
                "sample_idx": int(sample_idx),
                "evaluation_order": int(ordinal["evaluation_order"]),
                "factorial_arm": args.factorial_arm,
                "planner_arm": planner_arm,
                "body_arm": body_arm,
            }
            valid_arrays.append(array_record)
            base_attempt.update(
                {
                    "attempt_status": "complete",
                    "earliest_failure_stage": None,
                    "failure_reason": None,
                    "failure_message": None,
                    "body_status": "complete",
                    "body_graph_complete": True,
                    "body_cif_sha256": sha256_text(cif),
                    "num_atoms": int(arrays["num_atoms"]),
                }
            )
        except Exception as exc:  # noqa: BLE001
            base_attempt.update(
                {
                    "attempt_status": "failed",
                    "earliest_failure_stage": stage,
                    "failure_reason": type(exc).__name__,
                    "failure_message": str(exc),
                    "body_status": "failed",
                }
            )
        attempts.append(base_attempt)

    elapsed = time.time() - start
    attempt_path = rank_path(
        args.output_dir,
        "body_attempts.jsonl",
        rank,
        distributed,
    )
    arrays_path = rank_path(
        args.output_dir,
        "valid_arrays.jsonl",
        rank,
        distributed,
    )
    graph_path = rank_path(
        args.output_dir,
        "proposal_graphs.pt",
        rank,
        distributed,
    )
    _write_jsonl(attempt_path, attempts)
    _write_jsonl(arrays_path, valid_arrays)
    torch.save(proposal_graphs, graph_path)
    write_json(
        str(rank_path(args.output_dir, "body_metrics.json", rank, distributed)),
        _body_metrics(
            attempts,
            elapsed=elapsed,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
        ),
    )

    if distributed:
        dist.barrier()
        if is_main:
            _merge_body_outputs(
                args.output_dir,
                world_size=world_size,
                expected_count=int(args.num_samples),
                factorial_arm=args.factorial_arm,
                elapsed=elapsed,
            )
        dist.barrier()
        dist.destroy_process_group()
    else:
        ordered = ordered_single_arm_attempts(
            attempts,
            expected_count=int(args.num_samples),
            expected_factorial_arm=args.factorial_arm,
        )
        _write_jsonl(args.output_dir / "body_attempts.jsonl", list(ordered))
        payload = arrays_to_torch_payload(valid_arrays)
        payload["sample_idx"] = torch.tensor(
            [int(row["sample_idx"]) for row in valid_arrays],
            dtype=torch.long,
        )
        payload["time"] = float(elapsed)
        torch.save(payload, args.output_dir / "raw_dlm_samples.pt")


if __name__ == "__main__":
    main()
