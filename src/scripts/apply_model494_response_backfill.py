#!/usr/bin/env python3
"""Apply one deterministic model494-response-aligned SPAD backfill pass.

The script consumes an already sampled BS body and its one-to-one model494
tau800 endpoint.  It does not resample a Planner, create candidates, or select
between outputs.  Each site is visited exactly once in reverse SPAD order;
schema and PBC support are applied before a KL-bounded response bias.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from tqdm import tqdm

from crystal_dlm.dynamic_crystal import arrays_to_torch_payload, write_json
from crystal_dlm.fixed_slot import MASK_TOKEN_ID, SYMBOL_TO_Z
from crystal_dlm.r5_dynamic_length import (
    exact_body_token_count,
    exact_dynamic_schema_constraints,
    validate_answer_matches_plan,
    validate_dynamic_tokenizer_contract,
)
from crystal_dlm.spad_generation import (
    Model494ResponseConfig,
    revise_spad_anchors,
)
from crystal_dlm.spad_program import (
    program_from_element_order,
    response_revision_slots,
)
from scripts.sample_llada_dynamic_crystals import (
    build_dynamic_lightweight_constraints,
    graph_from_arrays,
    import_process_one,
    load_model_and_tokenizer,
    write_valid_arrays,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _single_eval_tensor(value: Any, *, name: str, rank_without_eval: int) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim == rank_without_eval + 1:
        if int(tensor.shape[0]) != 1:
            raise ValueError(f"{name} contains more than one refinement evaluation")
        tensor = tensor[0]
    if tensor.ndim != rank_without_eval:
        raise ValueError(f"unexpected {name} tensor rank")
    return tensor


def load_model494_endpoints(metrics_path: Path) -> dict[int, dict[str, Any]]:
    """Load one model494 endpoint per sample while preserving atom order."""

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    output_path = Path(str(metrics["output_file"]))
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    payload = torch.load(output_path, map_location="cpu")
    sample_indices = torch.as_tensor(payload["sample_indices"]).reshape(-1).tolist()
    num_atoms = _single_eval_tensor(
        payload["num_atoms"], name="num_atoms", rank_without_eval=1
    ).reshape(-1)
    frac_coords = _single_eval_tensor(
        payload["frac_coords"], name="frac_coords", rank_without_eval=2
    )
    atom_types = _single_eval_tensor(
        payload["atom_types"], name="atom_types", rank_without_eval=1
    ).reshape(-1)
    if len(sample_indices) != int(num_atoms.numel()):
        raise ValueError("model494 sample_indices/num_atoms length mismatch")
    if int(frac_coords.shape[0]) != int(atom_types.numel()):
        raise ValueError("model494 coordinate/type length mismatch")
    endpoints: dict[int, dict[str, Any]] = {}
    cursor = 0
    for sample_idx, raw_count in zip(sample_indices, num_atoms.tolist(), strict=True):
        count = int(raw_count)
        if count <= 0 or cursor + count > int(frac_coords.shape[0]):
            raise ValueError("model494 endpoint has invalid atom accounting")
        key = int(sample_idx)
        if key in endpoints:
            raise ValueError(f"duplicate model494 sample_idx {key}")
        endpoints[key] = {
            "frac_coords": frac_coords[cursor : cursor + count].tolist(),
            "atom_types": [
                int(value) for value in atom_types[cursor : cursor + count].tolist()
            ],
            "num_atoms": count,
        }
        cursor += count
    if cursor != int(frac_coords.shape[0]):
        raise ValueError("model494 endpoint left unassigned atoms")
    return endpoints


def _species_program(row: Mapping[str, Any]) -> list[str]:
    source = row.get("prompt_record")
    if not isinstance(source, Mapping):
        raise ValueError("BS body row lacks prompt_record")
    values = source.get("species_program")
    if not isinstance(values, list) or not values:
        raise ValueError("BS body row lacks species_program")
    return [str(value) for value in values]


def _pair_task(
    row: Mapping[str, Any],
    endpoint: Mapping[str, Any] | None,
    source_graph: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    sample_idx = int(row["sample_idx"])
    plan = row.get("plan_state")
    if not isinstance(plan, dict):
        return None, "missing_plan"
    if row.get("parsed") is not True or not isinstance(row.get("text"), str):
        return None, "source_body_not_parsed"
    if endpoint is None:
        return None, "missing_model494_endpoint"
    try:
        arrays = validate_answer_matches_plan(plan, str(row["text"]))
        expected_types = [int(SYMBOL_TO_Z[str(symbol)]) for symbol in arrays["species"]]
        if int(endpoint["num_atoms"]) != int(arrays["num_atoms"]):
            return None, "model494_atom_count_mismatch"
        target_frac_coords = list(endpoint["frac_coords"])
        graph_to_body = list(range(int(arrays["num_atoms"])))
        if source_graph is not None:
            graph_types = [int(value) for value in source_graph["a_type"]]
            graph_coords = torch.as_tensor(source_graph["x_coord"], dtype=torch.float64)
            if list(endpoint["atom_types"]) != graph_types:
                return None, "model494_graph_atom_order_mismatch"
            raw_coords = torch.as_tensor(arrays["frac_coords"], dtype=torch.float64)
            if graph_coords.shape != raw_coords.shape:
                return None, "source_graph_coordinate_shape_mismatch"
            unused = set(range(int(arrays["num_atoms"])))
            graph_to_body = []
            for graph_index, atomic_number in enumerate(graph_types):
                candidates = [
                    index for index in unused if expected_types[index] == atomic_number
                ]
                if not candidates:
                    return None, "source_graph_species_mapping_failed"
                distances: list[tuple[float, int]] = []
                for body_index in candidates:
                    delta = graph_coords[graph_index] - raw_coords[body_index]
                    delta = delta - torch.round(delta)
                    distances.append(
                        (float(torch.linalg.vector_norm(delta).item()), body_index)
                    )
                distance, selected = min(distances)
                if distance > 1.0e-5:
                    return None, "source_graph_coordinate_mapping_failed"
                unused.remove(selected)
                graph_to_body.append(int(selected))
            if unused:
                return None, "source_graph_mapping_not_bijective"
            reordered: list[list[float] | None] = [None] * int(arrays["num_atoms"])
            for graph_index, body_index in enumerate(graph_to_body):
                reordered[body_index] = [
                    float(value) for value in target_frac_coords[graph_index]
                ]
            if any(value is None for value in reordered):
                return None, "model494_target_reorder_incomplete"
            target_frac_coords = [list(value) for value in reordered if value is not None]
        elif list(endpoint["atom_types"]) != expected_types:
            return None, "model494_atom_order_mismatch"
        program = program_from_element_order(
            plan,
            _species_program(row),
            order_source=str(
                (row.get("prompt_record") or {}).get("species_program_source")
                or "frozen_prompt_species_program"
            ),
        )
    except Exception as exc:
        return None, f"pair_validation:{type(exc).__name__}:{exc}"
    prompt = str(row.get("conditioning_prompt") or "").rstrip() + "\n"
    if not prompt.strip():
        return None, "missing_conditioning_prompt"
    return (
        {
            "sample_idx": sample_idx,
            "source_row": dict(row),
            "plan_state": plan,
            "source_arrays": arrays,
            "prompt": prompt,
            "answer": str(row["text"]),
            "target_frac_coords": target_frac_coords,
            "model494_graph_to_body_permutation": graph_to_body,
            "revision_slots": list(response_revision_slots(program)),
        },
        None,
    )


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _guidance_summary(logs: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(item.get("guidance_status")) for item in logs)
    component_reports = [
        report
        for item in logs
        for report in item.get("response_component_reports", [])
    ]
    return {
        "site_visits": len(logs),
        "status_counts": dict(sorted(statuses.items())),
        "changed_sites": sum(int(item.get("changed_components", 0)) > 0 for item in logs),
        "changed_components": sum(int(item.get("changed_components", 0)) for item in logs),
        "mean_decision_kl_nats": (
            sum(float(item["kl_nats"]) for item in component_reports)
            / len(component_reports)
            if component_reports
            else 0.0
        ),
        "max_decision_kl_nats": max(
            (float(item["kl_nats"]) for item in component_reports), default=0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--body-rows", type=Path, required=True)
    parser.add_argument("--body-proposal-graphs", type=Path, required=True)
    parser.add_argument("--refinement-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if int(args.batch_size) <= 0:
        raise ValueError("batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    process_one = import_process_one(args.crysllmgen_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.checkpoint_path, device
    )
    tokenizer_report = validate_dynamic_tokenizer_contract(
        tokenizer, mask_token_id=MASK_TOKEN_ID
    )
    source_rows = _read_jsonl(args.body_rows)
    if len({int(row["sample_idx"]) for row in source_rows}) != len(source_rows):
        raise ValueError("source body contains duplicate sample_idx")
    source_graphs = torch.load(args.body_proposal_graphs, map_location="cpu")
    source_graph_by_idx = {int(graph["sample_idx"]): graph for graph in source_graphs}
    if len(source_graph_by_idx) != len(source_graphs):
        raise ValueError("source proposal graphs contain duplicate sample_idx")
    endpoints = load_model494_endpoints(args.refinement_metrics)

    tasks: list[dict[str, Any]] = []
    unguided: dict[int, str] = {}
    for row in source_rows:
        sample_idx = int(row["sample_idx"])
        task, reason = _pair_task(
            row,
            endpoints.get(sample_idx),
            source_graph_by_idx.get(sample_idx),
        )
        if task is None:
            unguided[sample_idx] = str(reason)
        else:
            tasks.append(task)
    tasks.sort(key=lambda item: (int(item["plan_state"]["N"]), int(item["sample_idx"])))

    config = Model494ResponseConfig()
    results: dict[int, dict[str, Any]] = {}
    result_arrays: dict[int, dict[str, Any]] = {}
    result_graphs: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    start = time.time()
    offset = 0
    progress = tqdm(total=len(tasks), desc="model494-response backfill")
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
            [item["prompt"] for item in batch],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(_model_device(model))
        attention_mask = encoded["attention_mask"].to(_model_device(model))
        answer_ids: list[list[int]] = []
        for item in batch:
            ids = tokenizer(
                item["answer"], add_special_tokens=False
            )["input_ids"]
            if len(ids) != gen_length:
                raise ValueError(
                    f"sample {item['sample_idx']} answer retokenized to {len(ids)}, "
                    f"expected {gen_length}"
                )
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
            duplicate_coordinate_mask=False,
            lattice_volume_mask=False,
            min_lattice_rad=1.0e-4,
            canonicalize_periodic_alias=True,
            pbc_min_distance_mask=True,
            pbc_min_distance_A=0.5,
            pbc_image_radius=2,
        )
        corrected, revision_logs = revise_spad_anchors(
            model,
            complete,
            prompt_length=int(input_ids.shape[1]),
            gen_length=gen_length,
            revision_slots_by_batch=[item["revision_slots"] for item in batch],
            attention_mask=attention_mask,
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=MASK_TOKEN_ID,
            allowed_token_ids_by_generation_pos=allowed,
            atom_count_grammar=None,
            lightweight_decoding_constraints=constraints,
            suffix_visible=True,
            model494_target_frac_coords_by_batch=[
                item["target_frac_coords"] for item in batch
            ],
            model494_response_config=config,
        )
        decoded = tokenizer.batch_decode(
            corrected[:, input_ids.shape[1] :],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for item, text, logs in zip(batch, decoded, revision_logs, strict=True):
            sample_idx = int(item["sample_idx"])
            row = dict(item["source_row"])
            row["text"] = text
            row["spad_response_revision_log"] = logs
            row["model494_response_guidance"] = {
                "schema": "model494_response_aligned_spad_backfill_v1",
                "endpoint_source": "frozen_model494_tau800",
                "lattice_tokens_frozen": True,
                "deterministic": True,
                "one_pass": True,
                "all_sites_reverse_spad_order": True,
                "config": config.__dict__,
                "model494_graph_to_body_permutation": item[
                    "model494_graph_to_body_permutation"
                ],
                **_guidance_summary(logs),
            }
            try:
                arrays = validate_answer_matches_plan(item["plan_state"], text)
                graph, cif = graph_from_arrays(arrays, process_one)
                graph["sample_idx"] = sample_idx
                row["cif"] = cif
                row["parsed"] = True
                row["num_atoms"] = int(arrays["num_atoms"])
                results[sample_idx] = row
                result_arrays[sample_idx] = arrays
                result_graphs[sample_idx] = graph
            except Exception as exc:
                row["parsed"] = False
                row.pop("cif", None)
                row["failure_reason"] = f"corrected_output:{type(exc).__name__}:{exc}"
                results[sample_idx] = row
                failures.append(
                    {
                        "sample_idx": sample_idx,
                        "reason": row["failure_reason"],
                        "text": text,
                    }
                )
        progress.update(len(batch))
    progress.close()

    # A missing or mismatched endpoint is not a candidate-selection event: the
    # frozen BS trajectory is retained unchanged and explicitly marked unguided.
    for source in source_rows:
        sample_idx = int(source["sample_idx"])
        if sample_idx not in unguided:
            continue
        row = dict(source)
        row["model494_response_guidance"] = {
            "schema": "model494_response_aligned_spad_backfill_v1",
            "status": "not_applied",
            "reason": unguided[sample_idx],
        }
        results[sample_idx] = row
        if row.get("parsed") is True:
            arrays = validate_answer_matches_plan(row["plan_state"], str(row["text"]))
            result_arrays[sample_idx] = arrays
            if sample_idx in source_graph_by_idx:
                result_graphs[sample_idx] = source_graph_by_idx[sample_idx]

    expected_indices = [int(row["sample_idx"]) for row in source_rows]
    if set(results) != set(expected_indices):
        raise RuntimeError("response backfill changed source sample accounting")
    ordered_rows = [results[index] for index in expected_indices]
    ordered_arrays = [result_arrays[index] for index in expected_indices if index in result_arrays]
    ordered_graphs = [result_graphs[index] for index in expected_indices if index in result_graphs]
    _write_jsonl(args.output_dir / "raw_generations.jsonl", ordered_rows)
    _write_jsonl(args.output_dir / "failure_cases.jsonl", failures)
    write_valid_arrays(args.output_dir / "valid_arrays.jsonl", ordered_arrays)
    torch.save(ordered_graphs, args.output_dir / "proposal_graphs.pt")
    if ordered_arrays:
        raw_payload = arrays_to_torch_payload(ordered_arrays)
        raw_payload["time"] = time.time() - start
        torch.save(raw_payload, args.output_dir / "raw_dlm_samples.pt")
    guidance = [row["model494_response_guidance"] for row in ordered_rows]
    metrics = {
        "schema": "model494_response_aligned_spad_backfill_v1",
        "requested_samples": len(source_rows),
        "guided_samples": len(tasks),
        "unguided_samples": len(unguided),
        "decoded_samples": len(tasks),
        "parse_success": len(result_arrays),
        "graph_success": len(result_graphs),
        "changed_samples": sum(
            int(item.get("changed_components", 0)) > 0 for item in guidance
        ),
        "changed_components": sum(
            int(item.get("changed_components", 0)) for item in guidance
        ),
        "guidance_skip_reasons": dict(sorted(Counter(unguided.values()).items())),
        "corrected_failures": len(failures),
        "model494_response_config": config.__dict__,
        "lattice_tokens_frozen": True,
        "one_plan_one_trajectory": True,
        "selection_rerank_replacement": False,
        "direct_evaluation": "DEFERRED_COST",
        "time_sec": time.time() - start,
    }
    write_json(str(args.output_dir / "sample_metrics.json"), metrics)
    write_json(str(args.output_dir / "tokenizer_report.json"), tokenizer_report)
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
