#!/usr/bin/env python3
"""Composition-protected fixed-slot geometry inpainting for MP-20 LLaDA samples."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from tqdm import tqdm

from crystal_dlm.diagnostic_remask import (
    anti_high_symmetry_failures,
    build_prefill_token_ids_by_position,
    composition_preserved,
    geometry_degeneracy_record,
    summarize_geometry,
    write_geometry_markdown,
)
from crystal_dlm.fixed_slot import (
    ANSWER_TOKEN_COUNT,
    CANONICAL_PROMPT,
    MASK_TOKEN_ID,
    FixedSlotConfig,
    arrays_to_structure,
    arrays_to_torch_payload,
    parse_fixed_slot_answer,
    write_json,
)
from crystal_dlm.generation_schedule import n_elements_sequential_rest_schedule
from crystal_dlm.llada_generation import generate
from scripts.sample_llada_crystals import (
    build_atom_count_grammar,
    build_lightweight_decoding_constraints,
    build_schema_generation_constraints,
    graph_from_arrays,
    import_process_one,
    init_distributed,
    load_model_and_tokenizer,
    rank_path,
    read_valid_arrays,
    write_valid_arrays,
)


def merge_counter_dict(target: Dict[str, int], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        target[str(key)] = int(target.get(str(key), 0)) + int(value)


def merge_distributed_inpaint_outputs(output_dir: Path, world_size: int) -> None:
    merged_metrics: Dict[str, Any] = {
        "requested_samples": 0,
        "decoded_samples": 0,
        "attempted_generations": 0,
        "parse_success": 0,
        "pymatgen_success": 0,
        "graph_success": 0,
        "composition_preserved_success": 0,
        "accepted_with_anti_high_symmetry_failure": 0,
        "anti_high_symmetry_retry_count": 0,
        "failures": {},
        "time_sec": 0.0,
        "distributed": True,
        "world_size": world_size,
    }
    valid_arrays: List[Dict[str, Any]] = []
    proposal_graphs: List[Dict[str, Any]] = []
    with (output_dir / "raw_generations.jsonl").open("w", encoding="utf-8") as raw_out, (
        output_dir / "raw_attempts.jsonl"
    ).open("w", encoding="utf-8") as attempts_out, (
        output_dir / "failure_cases.jsonl"
    ).open("w", encoding="utf-8") as failure_out:
        for rank in range(world_size):
            metrics_path = rank_path(output_dir, "sample_metrics.json", rank, True)
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                for key in (
                    "requested_samples",
                    "decoded_samples",
                    "attempted_generations",
                    "parse_success",
                    "pymatgen_success",
                    "graph_success",
                    "composition_preserved_success",
                    "accepted_with_anti_high_symmetry_failure",
                    "anti_high_symmetry_retry_count",
                ):
                    merged_metrics[key] += int(metrics.get(key, 0))
                merged_metrics["time_sec"] = max(
                    merged_metrics["time_sec"], float(metrics.get("time_sec") or 0.0)
                )
                merge_counter_dict(merged_metrics["failures"], metrics.get("failures", {}))

            for filename, handle in (
                ("raw_generations.jsonl", raw_out),
                ("raw_attempts.jsonl", attempts_out),
                ("failure_cases.jsonl", failure_out),
            ):
                path = rank_path(output_dir, filename, rank, True)
                if path.exists():
                    handle.write(path.read_text(encoding="utf-8"))

            valid_arrays.extend(read_valid_arrays(rank_path(output_dir, "valid_arrays.jsonl", rank, True)))
            graph_path = rank_path(output_dir, "proposal_graphs.pt", rank, True)
            if graph_path.exists():
                proposal_graphs.extend(torch.load(graph_path, map_location="cpu"))

    merged_metrics["parse_rate"] = merged_metrics["parse_success"] / max(
        1, merged_metrics["decoded_samples"]
    )
    merged_metrics["graph_rate"] = merged_metrics["graph_success"] / max(
        1, merged_metrics["decoded_samples"]
    )
    merged_metrics["graph_acceptance_rate"] = merged_metrics["graph_rate"]
    merged_metrics["valid_array_count"] = len(valid_arrays)
    write_json(str(output_dir / "sample_metrics.json"), merged_metrics)
    if valid_arrays:
        write_valid_arrays(output_dir / "valid_arrays.jsonl", valid_arrays)
        payload = arrays_to_torch_payload(valid_arrays)
        payload["time"] = merged_metrics["time_sec"]
        torch.save(payload, output_dir / "raw_dlm_samples.pt")
        torch.save(proposal_graphs, output_dir / "proposal_graphs.pt")
        geometry_summary = summarize_geometry(valid_arrays)
        write_json(str(output_dir / "geometry_diagnostics.json"), geometry_summary)
        (output_dir / "geometry_diagnostics.md").write_text(
            write_geometry_markdown(geometry_summary),
            encoding="utf-8",
        )


def load_source_arrays(path: Path, num_samples: int) -> List[Dict[str, Any]]:
    rows = read_valid_arrays(path)
    if not rows:
        for rank_path_candidate in sorted(path.parent.glob(f"{path.stem}.rank*{path.suffix}")):
            rows.extend(read_valid_arrays(rank_path_candidate))
    if len(rows) < num_samples:
        raise RuntimeError(
            f"--input-valid-arrays-jsonl has {len(rows)} rows, fewer than requested {num_samples}"
        )
    return rows[:num_samples]


def record_failure(
    handle,
    metrics: Dict[str, Any],
    *,
    sample_idx: int,
    source_idx: int,
    attempt: int,
    reason: str,
    message: str,
    text: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    metrics["failures"][reason] = int(metrics["failures"].get(reason, 0)) + 1
    row: Dict[str, Any] = {
        "sample_idx": sample_idx,
        "source_idx": source_idx,
        "attempt": attempt,
        "reason": reason,
        "message": message,
    }
    if text is not None:
        row["text"] = text
    if extra:
        row.update(dict(extra))
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def accept_candidate(
    raw_handle,
    valid_arrays: List[Dict[str, Any]],
    proposal_graphs: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    *,
    raw_record: Dict[str, Any],
    arrays: Dict[str, Any],
    graph: Dict[str, Any],
    accepted_with_anti_failure: bool,
) -> None:
    if accepted_with_anti_failure:
        raw_record["accepted_with_anti_high_symmetry_failure"] = True
        metrics["accepted_with_anti_high_symmetry_failure"] += 1
    raw_handle.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
    valid_arrays.append(arrays)
    proposal_graphs.append(graph)
    metrics["decoded_samples"] += 1
    metrics["parse_success"] += 1
    metrics["pymatgen_success"] += 1
    metrics["graph_success"] += 1
    metrics["composition_preserved_success"] += 1


def final_failure(raw_handle, metrics: Dict[str, Any], raw_record: Dict[str, Any]) -> None:
    raw_handle.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
    metrics["decoded_samples"] += 1
    if raw_record.get("parsed"):
        metrics["parse_success"] += 1
    if raw_record.get("pymatgen_success"):
        metrics["pymatgen_success"] += 1
    if raw_record.get("graph_success"):
        metrics["graph_success"] += 1
    if raw_record.get("composition_preserved"):
        metrics["composition_preserved_success"] += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--input-valid-arrays-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen")
    parser.add_argument("--mode", choices=["lattice_only", "geometry"], default="geometry")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=ANSWER_TOKEN_COUNT)
    parser.add_argument("--gen-length", type=int, default=ANSWER_TOKEN_COUNT)
    parser.add_argument("--block-length", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--prompt", default=CANONICAL_PROMPT)
    parser.add_argument("--schema-logit-mask", action="store_true", default=True)
    parser.add_argument("--no-schema-logit-mask", dest="schema_logit_mask", action="store_false")
    parser.add_argument("--atom-count-grammar-mask", action="store_true", default=True)
    parser.add_argument("--no-atom-count-grammar-mask", dest="atom_count_grammar_mask", action="store_false")
    parser.add_argument("--duplicate-coordinate-mask", action="store_true", default=True)
    parser.add_argument("--no-duplicate-coordinate-mask", dest="duplicate_coordinate_mask", action="store_false")
    parser.add_argument("--lattice-volume-mask", action="store_true", default=True)
    parser.add_argument("--no-lattice-volume-mask", dest="lattice_volume_mask", action="store_false")
    parser.add_argument("--min-lattice-rad", type=float, default=1e-4)
    parser.add_argument("--anti-high-symmetry", action="store_true", default=False)
    parser.add_argument("--attempts-per-source", type=int, default=4)
    parser.add_argument("--max-high-symmetry-coord-fraction", type=float, default=0.75)
    parser.add_argument("--reject-all-lengths-equal", action="store_true", default=False)
    parser.add_argument("--reject-all-angles-90", action="store_true", default=False)
    args = parser.parse_args()

    if args.block_length != 1:
        raise RuntimeError("E4A inpainting requires --block-length 1")
    if not args.schema_logit_mask or not args.atom_count_grammar_mask:
        raise RuntimeError("E4A inpainting requires schema and atom-count grammar masks")
    if args.attempts_per_source < 1:
        raise ValueError("--attempts-per-source must be >= 1")

    dist_info = init_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    distributed = dist_info["distributed"]
    is_main = dist_info["is_main"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    process_one = import_process_one(args.crysllmgen_dir)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.checkpoint_path, dist_info["device"])
    if tokenizer.pad_token_id == MASK_TOKEN_ID:
        raise RuntimeError("tokenizer.pad_token_id must not equal LLaDA mask token id 126336")

    config = FixedSlotConfig()
    source_arrays = load_source_arrays(args.input_valid_arrays_jsonl, args.num_samples)
    local_sources: List[Tuple[int, Dict[str, Any]]] = [
        (idx, source_arrays[idx]) for idx in range(rank, args.num_samples, world_size)
    ]
    allowed_token_ids_by_generation_pos, _slot_prefill = build_schema_generation_constraints(
        tokenizer,
        config=config,
    )
    if not args.schema_logit_mask:
        allowed_token_ids_by_generation_pos = None
    atom_count_grammar = (
        build_atom_count_grammar(tokenizer, config=config) if args.atom_count_grammar_mask else None
    )
    lightweight_decoding_constraints = build_lightweight_decoding_constraints(
        tokenizer,
        duplicate_coordinate_mask=args.duplicate_coordinate_mask,
        lattice_volume_mask=args.lattice_volume_mask,
        min_lattice_rad=args.min_lattice_rad,
        config=config,
    )
    generation_position_groups = n_elements_sequential_rest_schedule()

    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config.update(
        {
            "distributed": distributed,
            "world_size": world_size,
            "generation_schedule": "n-elements-sequential-rest",
            "freeze_policy": "composition_and_slot_boundary",
            "source_geometry_summary": summarize_geometry(source_arrays),
        }
    )
    if is_main:
        write_json(str(args.output_dir / "run_config.json"), run_config)
        write_json(str(args.output_dir / "source_geometry_diagnostics.json"), run_config["source_geometry_summary"])
        (args.output_dir / "source_geometry_diagnostics.md").write_text(
            write_geometry_markdown(run_config["source_geometry_summary"]),
            encoding="utf-8",
        )

    raw_path = rank_path(args.output_dir, "raw_generations.jsonl", rank, distributed)
    attempts_path = rank_path(args.output_dir, "raw_attempts.jsonl", rank, distributed)
    failure_path = rank_path(args.output_dir, "failure_cases.jsonl", rank, distributed)
    valid_arrays_path = rank_path(args.output_dir, "valid_arrays.jsonl", rank, distributed)
    graph_path = rank_path(args.output_dir, "proposal_graphs.pt", rank, distributed)

    valid_arrays: List[Dict[str, Any]] = []
    proposal_graphs: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {
        "requested_samples": len(local_sources),
        "decoded_samples": 0,
        "attempted_generations": 0,
        "parse_success": 0,
        "pymatgen_success": 0,
        "graph_success": 0,
        "composition_preserved_success": 0,
        "accepted_with_anti_high_symmetry_failure": 0,
        "anti_high_symmetry_retry_count": 0,
        "anti_high_symmetry": bool(args.anti_high_symmetry),
        "mode": args.mode,
        "failures": {},
        "time_sec": None,
        "rank": rank,
        "world_size": world_size,
    }

    pending: List[Tuple[int, Dict[str, Any], int]] = [
        (source_idx, arrays, 1) for source_idx, arrays in local_sources
    ]
    best_degenerate: Dict[int, Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = {}
    start = time.time()
    prompt_text = args.prompt.rstrip() + "\n"
    with raw_path.open("w", encoding="utf-8") as raw_handle, attempts_path.open(
        "w", encoding="utf-8"
    ) as attempt_handle, failure_path.open("w", encoding="utf-8") as failure_handle:
        progress = tqdm(total=len(local_sources), desc=f"E4A inpaint rank{rank}", disable=distributed and not is_main)
        while pending:
            current = pending
            pending = []
            for batch_start in range(0, len(current), args.batch_size):
                batch_records = current[batch_start : batch_start + args.batch_size]
                batch_sources = [record[1] for record in batch_records]
                current_prefill = build_prefill_token_ids_by_position(
                    tokenizer,
                    batch_sources,
                    mode=args.mode,
                    config=config,
                )
                encoded = tokenizer(
                    [prompt_text] * len(batch_records),
                    add_special_tokens=False,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(model.device)
                attention_mask = encoded["attention_mask"].to(model.device)
                outputs = generate(
                    model,
                    input_ids,
                    attention_mask=attention_mask,
                    steps=args.steps,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                    mask_id=MASK_TOKEN_ID,
                    allowed_token_ids_by_generation_pos=allowed_token_ids_by_generation_pos,
                    prefill_token_ids_by_generation_pos=current_prefill,
                    atom_count_grammar=atom_count_grammar,
                    generation_position_groups=generation_position_groups,
                    lightweight_decoding_constraints=lightweight_decoding_constraints,
                )
                generated_ids = outputs[:, input_ids.shape[1] :]
                decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
                for (source_idx, source_arrays_item, attempt), text in zip(batch_records, decoded):
                    metrics["attempted_generations"] += 1
                    raw_record: Dict[str, Any] = {
                        "sample_idx": source_idx,
                        "source_idx": source_idx,
                        "attempt": attempt,
                        "mode": args.mode,
                        "text": text,
                    }
                    attempt_record = dict(raw_record)
                    final_attempt = attempt >= args.attempts_per_source
                    try:
                        arrays = parse_fixed_slot_answer(text)
                        raw_record["parsed"] = True
                        attempt_record["parsed"] = True
                        source_diag = geometry_degeneracy_record(source_arrays_item)
                        candidate_diag = geometry_degeneracy_record(arrays)
                        raw_record["source_geometry"] = source_diag
                        raw_record["candidate_geometry"] = candidate_diag
                        attempt_record["source_geometry"] = source_diag
                        attempt_record["candidate_geometry"] = candidate_diag
                        if not composition_preserved(source_arrays_item, arrays):
                            raw_record.update(
                                {
                                    "graph_success": False,
                                    "composition_preserved": False,
                                    "reason": "CompositionChanged",
                                    "message": "candidate changed frozen composition or atom order",
                                }
                            )
                            attempt_record.update(raw_record)
                            record_failure(
                                failure_handle,
                                metrics,
                                sample_idx=source_idx,
                                source_idx=source_idx,
                                attempt=attempt,
                                reason="CompositionChanged",
                                message="candidate changed frozen composition or atom order",
                                text=text,
                            )
                            if not final_attempt:
                                pending.append((source_idx, source_arrays_item, attempt + 1))
                                continue
                            if source_idx in best_degenerate:
                                stored_raw, stored_arrays, stored_graph = best_degenerate[source_idx]
                                accept_candidate(
                                    raw_handle,
                                    valid_arrays,
                                    proposal_graphs,
                                    metrics,
                                    raw_record=stored_raw,
                                    arrays=stored_arrays,
                                    graph=stored_graph,
                                    accepted_with_anti_failure=True,
                                )
                            else:
                                final_failure(raw_handle, metrics, raw_record)
                            progress.update(1)
                            continue
                        raw_record["composition_preserved"] = True
                        attempt_record["composition_preserved"] = True
                        try:
                            arrays_to_structure(arrays)
                            raw_record["pymatgen_success"] = True
                            attempt_record["pymatgen_success"] = True
                        except Exception:
                            raw_record["pymatgen_success"] = False
                            attempt_record["pymatgen_success"] = False
                        graph, cif = graph_from_arrays(arrays, process_one)
                        raw_record.update(
                            {
                                "graph_success": True,
                                "cif": cif,
                                "num_atoms": arrays["num_atoms"],
                            }
                        )
                        attempt_record.update(raw_record)
                        anti_failures: List[str] = []
                        if args.anti_high_symmetry:
                            anti_failures = anti_high_symmetry_failures(
                                arrays,
                                max_high_symmetry_coord_fraction=args.max_high_symmetry_coord_fraction,
                                reject_all_lengths_equal=args.reject_all_lengths_equal,
                                reject_all_angles_90=args.reject_all_angles_90,
                            )
                        raw_record["anti_high_symmetry_failures"] = anti_failures
                        attempt_record["anti_high_symmetry_failures"] = anti_failures
                        if anti_failures:
                            best_degenerate.setdefault(source_idx, (dict(raw_record), arrays, graph))
                            record_failure(
                                failure_handle,
                                metrics,
                                sample_idx=source_idx,
                                source_idx=source_idx,
                                attempt=attempt,
                                reason="AntiHighSymmetryRetry",
                                message=",".join(anti_failures),
                                text=text,
                                extra={"geometry": candidate_diag},
                            )
                            if not final_attempt:
                                metrics["anti_high_symmetry_retry_count"] += 1
                                pending.append((source_idx, source_arrays_item, attempt + 1))
                                continue
                            stored_raw, stored_arrays, stored_graph = best_degenerate[source_idx]
                            accept_candidate(
                                raw_handle,
                                valid_arrays,
                                proposal_graphs,
                                metrics,
                                raw_record=stored_raw,
                                arrays=stored_arrays,
                                graph=stored_graph,
                                accepted_with_anti_failure=True,
                            )
                            progress.update(1)
                            continue
                        accept_candidate(
                            raw_handle,
                            valid_arrays,
                            proposal_graphs,
                            metrics,
                            raw_record=raw_record,
                            arrays=arrays,
                            graph=graph,
                            accepted_with_anti_failure=False,
                        )
                        progress.update(1)
                    except Exception as exc:
                        reason = type(exc).__name__
                        raw_record.update(
                            {
                                "parsed": bool(raw_record.get("parsed", False)),
                                "graph_success": False,
                                "reason": reason,
                                "message": str(exc),
                            }
                        )
                        attempt_record.update(raw_record)
                        record_failure(
                            failure_handle,
                            metrics,
                            sample_idx=source_idx,
                            source_idx=source_idx,
                            attempt=attempt,
                            reason=reason,
                            message=str(exc),
                            text=text,
                        )
                        if not final_attempt:
                            pending.append((source_idx, source_arrays_item, attempt + 1))
                            continue
                        if source_idx in best_degenerate:
                            stored_raw, stored_arrays, stored_graph = best_degenerate[source_idx]
                            accept_candidate(
                                raw_handle,
                                valid_arrays,
                                proposal_graphs,
                                metrics,
                                raw_record=stored_raw,
                                arrays=stored_arrays,
                                graph=stored_graph,
                                accepted_with_anti_failure=True,
                            )
                        else:
                            final_failure(raw_handle, metrics, raw_record)
                        progress.update(1)
                    finally:
                        attempt_handle.write(json.dumps(attempt_record, ensure_ascii=False) + "\n")
        progress.close()

    metrics["time_sec"] = time.time() - start
    metrics["parse_rate"] = metrics["parse_success"] / max(1, metrics["decoded_samples"])
    metrics["graph_rate"] = metrics["graph_success"] / max(1, metrics["decoded_samples"])
    metrics["graph_acceptance_rate"] = metrics["graph_rate"]
    metrics["valid_array_count"] = len(valid_arrays)
    write_json(str(rank_path(args.output_dir, "sample_metrics.json", rank, distributed)), metrics)
    if valid_arrays:
        write_valid_arrays(valid_arrays_path, valid_arrays)
        torch.save(proposal_graphs, graph_path)
        if not distributed:
            payload = arrays_to_torch_payload(valid_arrays)
            payload["time"] = metrics["time_sec"]
            torch.save(payload, args.output_dir / "raw_dlm_samples.pt")
            torch.save(proposal_graphs, args.output_dir / "proposal_graphs.pt")
            geometry_summary = summarize_geometry(valid_arrays)
            write_json(str(args.output_dir / "geometry_diagnostics.json"), geometry_summary)
            (args.output_dir / "geometry_diagnostics.md").write_text(
                write_geometry_markdown(geometry_summary),
                encoding="utf-8",
            )

    if distributed:
        dist.barrier()
        if is_main:
            merge_distributed_inpaint_outputs(args.output_dir, world_size)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
