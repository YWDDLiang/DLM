#!/usr/bin/env python3
"""Sample R5 exact-length dynamic crystal proposals from a LLaDA checkpoint."""

from __future__ import annotations

import argparse
import hashlib
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
from crystal_dlm.spad_generation import (  # noqa: E402
    SPAD_BASIN_CLOSURE_BLOCK_SALT_LIMIT,
    _spad_basin_closure_block_salt,
    revise_spad_anchors,
    revise_spad_cell,
    revise_spad_species_blocks,
)
from crystal_dlm.spad_program import (  # noqa: E402
    anchor_revision_slots,
    limited_anchor_revision_slots,
    program_from_element_order,
    reverse_species_block_revision_slots,
    spad_predictor_position_groups,
)
from crystal_dlm.r5_dynamic_length import (  # noqa: E402
    count_prefill_for_batch,
    exact_body_token_count,
    exact_dynamic_generation_schedule,
    exact_dynamic_generation_schedule_joint_coordinates,
    exact_dynamic_schema_constraints,
    validate_answer_matches_plan,
    validate_dynamic_tokenizer_contract,
)
from crystal_dlm.h1_formula_only_body import (  # noqa: E402
    H1_FORMULA_ONLY_BODY_REPRESENTATION,
    build_formula_only_body_prompt,
)
from crystal_dlm.r5_plan_state import (  # noqa: E402
    build_body_prompt,
    build_hard_anchor_body_prompt,
    parse_plan_state_json,
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


SPAD_BASIN_CLOSURE_SCHEDULE_VERSION = (
    "cell_then_reverse_llama_species_blocks_v1"
)
SPAD_BASIN_CLOSURE_SAMPLE_SEED_STRIDE = 10_000_000
SPAD_BASIN_CLOSURE_CELL_STAGE_OFFSET = 1_000_000
SPAD_BASIN_CLOSURE_BLOCK_STAGE_OFFSET = 2_000_000
SPAD_BASIN_CLOSURE_CELL_INTERNAL_SALT_LIMIT = 6 * 10_007

if not (
    SPAD_BASIN_CLOSURE_CELL_STAGE_OFFSET
    + SPAD_BASIN_CLOSURE_CELL_INTERNAL_SALT_LIMIT
    < SPAD_BASIN_CLOSURE_BLOCK_STAGE_OFFSET
    and SPAD_BASIN_CLOSURE_BLOCK_STAGE_OFFSET
    + SPAD_BASIN_CLOSURE_BLOCK_SALT_LIMIT
    < SPAD_BASIN_CLOSURE_SAMPLE_SEED_STRIDE
):
    raise RuntimeError("SPAD basin-closure RNG namespaces overlap")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spad_basin_closure_stage_seed(
    base_seed: int,
    sample_idx: int,
    stage_offset: int,
) -> int:
    """Return the base seed for one sample-local closure stage namespace."""

    if int(sample_idx) < 0:
        raise ValueError("sample_idx must be nonnegative")
    return (
        int(base_seed)
        + int(sample_idx) * SPAD_BASIN_CLOSURE_SAMPLE_SEED_STRIDE
        + int(stage_offset)
    )


def _resolve_recorded_checkpoint_path(
    value: Any,
    *,
    manifest_path: Path,
) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def validate_spad_basin_closure_configuration(args: Any) -> Dict[str, Any] | None:
    """Validate the opt-in closure contract before model initialization."""

    enabled = bool(getattr(args, "spad_basin_closure", False))
    capability_path_value = getattr(
        args, "spad_basin_closure_capability_json", None
    )
    if not enabled:
        if capability_path_value is not None:
            raise ValueError(
                "--spad-basin-closure-capability-json requires "
                "--spad-basin-closure"
            )
        return None
    if getattr(args, "generation_schedule", None) != "spad":
        raise ValueError(
            "--spad-basin-closure requires --generation-schedule spad"
        )
    if bool(getattr(args, "spad_backfill", False)):
        raise ValueError(
            "--spad-basin-closure cannot be combined with --spad-backfill"
        )
    if bool(getattr(args, "spad_cell_closure", False)):
        raise ValueError(
            "--spad-basin-closure cannot be combined with --spad-cell-closure"
        )
    if not bool(getattr(args, "pbc_min_distance_mask", False)):
        raise ValueError(
            "--spad-basin-closure requires --pbc-min-distance-mask"
        )
    if capability_path_value is None:
        raise ValueError(
            "--spad-basin-closure requires "
            "--spad-basin-closure-capability-json"
        )
    checkpoint_value = getattr(args, "checkpoint_path", None)
    if checkpoint_value is None:
        raise ValueError("--spad-basin-closure requires --checkpoint-path")

    actual_path = Path(checkpoint_value).expanduser().resolve(strict=True)
    capability_path = Path(capability_path_value).expanduser().resolve()
    expected_capability_path = (
        actual_path / "spad_basin_closure_capability.json"
    ).resolve()
    if capability_path != expected_capability_path:
        raise ValueError(
            "SPAD basin-closure capability JSON must be the checkpoint-local "
            f"manifest: {expected_capability_path}"
        )
    try:
        payload = json.loads(capability_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read SPAD basin-closure capability JSON: {capability_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("SPAD basin-closure capability JSON must be an object")
    if payload.get("schema") != "spad_basin_closure_capability_v1":
        raise ValueError(
            "SPAD basin-closure capability schema must equal "
            "'spad_basin_closure_capability_v1'"
        )

    required_capabilities = {
        "spad_cell_closure_trained": True,
        "spad_species_block_closure_trained": True,
        "closure_schedule_version": SPAD_BASIN_CLOSURE_SCHEDULE_VERSION,
    }
    for key, expected in required_capabilities.items():
        observed = payload.get(key)
        matches = observed is True if expected is True else observed == expected
        if not matches:
            raise ValueError(
                f"SPAD basin-closure capability {key!r} must equal {expected!r}"
            )

    recorded_checkpoint = payload.get("checkpoint_path")
    if not isinstance(recorded_checkpoint, str) or not recorded_checkpoint.strip():
        raise ValueError("capability checkpoint record requires checkpoint_path")

    recorded_path = _resolve_recorded_checkpoint_path(
        recorded_checkpoint,
        manifest_path=capability_path,
    )
    if actual_path != recorded_path:
        raise ValueError(
            "--checkpoint-path does not match the capability manifest record: "
            f"{actual_path} != {recorded_path}"
        )
    adapter_path = actual_path / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise ValueError(
            "SPAD basin-closure checkpoint is missing adapter_model.safetensors"
        )
    expected_adapter_sha256 = payload.get("adapter_model_sha256")
    if not isinstance(expected_adapter_sha256, str) or len(expected_adapter_sha256) != 64:
        raise ValueError("capability manifest requires adapter_model_sha256")
    adapter_sha256 = _sha256_file(adapter_path)
    if adapter_sha256 != expected_adapter_sha256.lower():
        raise ValueError("adapter_model.safetensors SHA256 does not match capability")
    return {
        "capability_json": str(capability_path),
        "checkpoint_path": str(actual_path),
        "adapter_model_sha256": adapter_sha256,
        **required_capabilities,
    }


def apply_spad_basin_closure(
    model: Any,
    complete_tokens: torch.Tensor,
    *,
    programs: List[Any],
    batch: List[Mapping[str, Any]],
    base_seed: int,
    prompt_length: int,
    gen_length: int,
    attention_mask: torch.Tensor | None,
    temperature: float,
    cfg_scale: float,
    remasking: str,
    mask_id: int,
    allowed_token_ids_by_generation_pos: list[list[int]] | None,
    lightweight_decoding_constraints: dict | None,
) -> tuple[
    torch.Tensor,
    list[dict[str, Any]],
    list[list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    """Execute predictor -> cell -> reverse Llama species-block closure."""

    if len(programs) != len(batch):
        raise ValueError("one SPAD program is required per closure batch row")
    revision_blocks = [
        [list(block) for block in reverse_species_block_revision_slots(program)]
        for program in programs
    ]
    cell_sampling_seeds = [
        _spad_basin_closure_stage_seed(
            int(base_seed),
            int(item["sample_idx"]),
            SPAD_BASIN_CLOSURE_CELL_STAGE_OFFSET,
        )
        for item in batch
    ]
    block_sampling_seeds = [
        _spad_basin_closure_stage_seed(
            int(base_seed),
            int(item["sample_idx"]),
            SPAD_BASIN_CLOSURE_BLOCK_STAGE_OFFSET,
        )
        for item in batch
    ]
    closed_tokens, cell_logs = revise_spad_cell(
        model,
        complete_tokens,
        prompt_length=prompt_length,
        gen_length=gen_length,
        attention_mask=attention_mask,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=mask_id,
        allowed_token_ids_by_generation_pos=allowed_token_ids_by_generation_pos,
        atom_count_grammar=None,
        lightweight_decoding_constraints=lightweight_decoding_constraints,
        strict_geometry_fallback=True,
        sampling_seeds_by_batch=cell_sampling_seeds,
    )
    closed_tokens, block_logs = revise_spad_species_blocks(
        model,
        closed_tokens,
        prompt_length=prompt_length,
        gen_length=gen_length,
        revision_blocks_by_batch=revision_blocks,
        attention_mask=attention_mask,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=mask_id,
        allowed_token_ids_by_generation_pos=allowed_token_ids_by_generation_pos,
        atom_count_grammar=None,
        lightweight_decoding_constraints=lightweight_decoding_constraints,
        sampling_seeds_by_batch=block_sampling_seeds,
    )
    metadata = [
        {
            "closure_schedule_version": SPAD_BASIN_CLOSURE_SCHEDULE_VERSION,
            "stage_order": ["predictor", "cell", "reverse_species_blocks"],
            "species_program": list(program.element_order),
            "species_program_source": str(program.order_source),
            "reverse_species_block_slots": blocks,
            "cell_sampling_seed": int(cell_seed),
            "species_block_sampling_seed": int(block_seed),
            "sample_seed_stride": SPAD_BASIN_CLOSURE_SAMPLE_SEED_STRIDE,
            "cell_stage_offset": SPAD_BASIN_CLOSURE_CELL_STAGE_OFFSET,
            "species_block_stage_offset": SPAD_BASIN_CLOSURE_BLOCK_STAGE_OFFSET,
            "final_geometry_supported": bool(
                block_log
                and block_log[-1].get("final_geometry_supported") is True
            ),
        }
        for program, blocks, cell_seed, block_seed, block_log in zip(
            programs,
            revision_blocks,
            cell_sampling_seeds,
            block_sampling_seeds,
            block_logs,
            strict=True,
        )
    ]
    return closed_tokens, list(cell_logs), list(block_logs), metadata


def spad_basin_closure_record_fields(
    *,
    cell_revision_log: Mapping[str, Any],
    species_block_revision_log: List[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build fields that are emitted only by the opt-in closure path."""

    return {
        "spad_basin_closure": True,
        "spad_basin_closure_cell_revision_log": dict(cell_revision_log),
        "spad_basin_closure_species_block_revision_log": [
            dict(entry) for entry in species_block_revision_log
        ],
        "spad_basin_closure_metadata": dict(metadata),
    }


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def build_prompt_for_style(plan: Mapping[str, Any], *, prompt_style: str, row: Mapping[str, Any] | None = None, prompt_field: str = "prompt") -> str:
    if prompt_style == "formula_only":
        return build_formula_only_body_prompt(plan).rstrip() + "\n"
    if prompt_style == "hard_anchor_only":
        return build_hard_anchor_body_prompt(plan).rstrip() + "\n"
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
        task_count = args.num_samples if args.repeat_prompt_records else min(args.num_samples, len(records))
        for task_idx in range(task_count):
            record = records[task_idx % len(records)]
            sample_idx = task_idx
            if args.preserve_prompt_sample_idx:
                source_sample_idx = record["source_row"].get("sample_idx")
                if source_sample_idx is None:
                    raise ValueError("--preserve-prompt-sample-idx requires sample_idx in every prompt row")
                sample_idx = int(source_sample_idx)
            tasks.append(
                {
                    "sample_idx": sample_idx,
                    "plan_state": record["plan_state"],
                    "prompt": record["prompt"],
                    "prompt_record_idx": record["source_idx"],
                    "prompt_record": record["source_row"],
                }
            )
        if len({int(task["sample_idx"]) for task in tasks}) != len(tasks):
            raise ValueError("prompt sample_idx values must be unique")
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
    parser.add_argument("--repeat-prompt-records", action="store_true", default=True)
    parser.add_argument("--no-repeat-prompt-records", dest="repeat_prompt_records", action="store_false")
    parser.add_argument("--preserve-prompt-sample-idx", action="store_true")
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--body-prompt-style",
        choices=["full_plan_state", "hard_anchor_only", "formula_only"],
        default="full_plan_state",
    )
    parser.add_argument("--num-atoms", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17017)
    parser.add_argument("--seed-by-sample-index", action="store_true")
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
    parser.add_argument("--canonicalize-periodic-alias", action="store_true")
    parser.add_argument("--pbc-min-distance-mask", action="store_true")
    parser.add_argument(
        "--generation-schedule",
        choices=["exact-plan", "joint-coordinates", "spad", "default"],
        default="exact-plan",
    )
    parser.add_argument("--spad-backfill", action="store_true")
    parser.add_argument("--spad-cell-closure", action="store_true")
    parser.add_argument("--spad-basin-closure", action="store_true")
    parser.add_argument(
        "--spad-basin-closure-capability-json",
        type=Path,
        default=None,
    )
    parser.add_argument("--spad-max-anchor-revisions", type=int, default=0)
    parser.add_argument("--spad-hide-suffix", action="store_true")
    parser.add_argument("--skip-graph-validation", action="store_true")
    args = parser.parse_args()

    if not 1 <= int(args.num_atoms) <= 20:
        raise ValueError("--num-atoms must be in 1..20")
    if args.spad_backfill and args.generation_schedule != "spad":
        raise ValueError("--spad-backfill requires --generation-schedule spad")
    if args.spad_cell_closure and args.generation_schedule != "spad":
        raise ValueError("--spad-cell-closure requires --generation-schedule spad")
    basin_closure_capability = validate_spad_basin_closure_configuration(args)
    if args.spad_hide_suffix and not args.spad_backfill:
        raise ValueError("--spad-hide-suffix requires --spad-backfill")
    if int(args.spad_max_anchor_revisions) < 0:
        raise ValueError("--spad-max-anchor-revisions must be nonnegative")
    if int(args.spad_max_anchor_revisions) > 0 and not args.spad_backfill:
        raise ValueError("limited anchor revisions require --spad-backfill")

    dist_info = init_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    distributed = dist_info["distributed"]
    is_main = dist_info["is_main"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    process_one = None if args.skip_graph_validation else import_process_one(args.crysllmgen_dir)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.checkpoint_path, dist_info["device"])
    tokenizer_contract = validate_dynamic_tokenizer_contract(
        tokenizer,
        mask_token_id=MASK_TOKEN_ID,
    )
    rank_seed = int(args.seed) + int(rank)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
    tasks = build_tasks(args)
    tasks = [task for idx, task in enumerate(tasks) if idx % world_size == rank]
    tasks.sort(key=lambda item: (int(item["plan_state"]["N"]), int(item["sample_idx"])))

    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.body_prompt_style == "formula_only":
        r5_representation = H1_FORMULA_ONLY_BODY_REPRESENTATION
    elif args.body_prompt_style == "hard_anchor_only":
        r5_representation = "r5_exact_dynamic_hard_anchor_v1"
    else:
        r5_representation = "r5_exact_dynamic_v1"
    run_config.update(
        {
            "representation": "dynamic_v1",
            "r5_representation": r5_representation,
            "distributed": distributed,
            "world_size": world_size,
            "rank_seed_rule": "seed + sample_idx" if args.seed_by_sample_index else "seed + rank",
        }
    )
    if basin_closure_capability is not None:
        run_config["spad_basin_closure_capability"] = basin_closure_capability
    if is_main:
        write_json(str(args.output_dir / "run_config.json"), run_config)
        write_json(
            str(args.output_dir / "tokenizer_report.json"),
            tokenizer_contract,
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
            if args.seed_by_sample_index:
                if len(batch) != 1:
                    raise ValueError("--seed-by-sample-index requires --batch-size 1")
                sample_seed = int(args.seed) + int(batch[0]["sample_idx"])
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
            gen_length = exact_body_token_count(num_atoms)
            allowed = exact_dynamic_schema_constraints(tokenizer, num_atoms) if args.schema_logit_mask else None
            prefill_maps: List[Mapping[int, List[int]]] = []
            if args.prefill_count_token:
                prefill_maps.append(count_prefill_for_batch(tokenizer, num_atoms, len(batch)))
            if args.freeze_plan_composition:
                prefill_maps.append(element_prefill_for_batch(tokenizer, [item["plan_state"] for item in batch]))
            prefill = merge_prefill_maps(*prefill_maps) if prefill_maps else None
            if args.generation_schedule == "exact-plan":
                schedule = exact_dynamic_generation_schedule(num_atoms)
                row_schedules = None
                programs = None
            elif args.generation_schedule == "joint-coordinates":
                schedule = exact_dynamic_generation_schedule_joint_coordinates(num_atoms)
                row_schedules = None
                programs = None
            elif args.generation_schedule == "spad":
                schedule = None
                programs = []
                row_schedules = []
                for item in batch:
                    source = item.get("prompt_record") or {}
                    order = source.get("species_program")
                    if not isinstance(order, list) or not order:
                        raise ValueError(
                            "SPAD sampling requires species_program in every prompt row"
                        )
                    program = program_from_element_order(
                        item["plan_state"],
                        [str(value) for value in order],
                        order_source=str(
                            source.get("species_program_source")
                            or "prompt_species_program"
                        ),
                    )
                    programs.append(program)
                    row_schedules.append(
                        [list(group) for group in spad_predictor_position_groups(program)]
                    )
            else:
                schedule = None
                row_schedules = None
                programs = None
            lightweight_constraints = build_dynamic_lightweight_constraints(
                tokenizer,
                duplicate_coordinate_mask=args.duplicate_coordinate_mask,
                lattice_volume_mask=args.lattice_volume_mask,
                min_lattice_rad=args.min_lattice_rad,
                canonicalize_periodic_alias=args.canonicalize_periodic_alias,
                pbc_min_distance_mask=args.pbc_min_distance_mask,
                pbc_min_distance_A=0.5,
                pbc_image_radius=2,
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
                generation_position_groups_by_batch=row_schedules,
                lightweight_decoding_constraints=lightweight_constraints,
            )
            revision_logs: List[List[Dict[str, Any]]] = [
                [] for _ in range(len(batch))
            ]
            cell_revision_logs: List[Dict[str, Any] | None] = [
                None for _ in range(len(batch))
            ]
            basin_cell_revision_logs: List[Dict[str, Any] | None] = [
                None for _ in range(len(batch))
            ]
            basin_block_revision_logs: List[List[Dict[str, Any]]] = [
                [] for _ in range(len(batch))
            ]
            basin_closure_metadata: List[Dict[str, Any] | None] = [
                None for _ in range(len(batch))
            ]
            if args.spad_basin_closure:
                if programs is None:
                    raise RuntimeError("SPAD programs were not built")
                (
                    outputs,
                    basin_cell_logs,
                    basin_block_logs,
                    basin_metadata,
                ) = apply_spad_basin_closure(
                    model,
                    outputs,
                    programs=programs,
                    batch=batch,
                    base_seed=int(args.seed),
                    prompt_length=input_ids.shape[1],
                    gen_length=gen_length,
                    attention_mask=attention_mask,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                    mask_id=MASK_TOKEN_ID,
                    allowed_token_ids_by_generation_pos=allowed,
                    lightweight_decoding_constraints=lightweight_constraints,
                )
                basin_cell_revision_logs = list(basin_cell_logs)
                basin_block_revision_logs = list(basin_block_logs)
                basin_closure_metadata = list(basin_metadata)
            if args.spad_cell_closure:
                cell_sampling_seeds = [
                    int(args.seed)
                    + int(item["sample_idx"]) * 1_000_003
                    + 70_000_019
                    for item in batch
                ]
                outputs, cell_logs = revise_spad_cell(
                    model,
                    outputs,
                    prompt_length=input_ids.shape[1],
                    gen_length=gen_length,
                    attention_mask=attention_mask,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                    mask_id=MASK_TOKEN_ID,
                    allowed_token_ids_by_generation_pos=allowed,
                    atom_count_grammar=None,
                    lightweight_decoding_constraints=lightweight_constraints,
                    strict_geometry_fallback=True,
                    sampling_seeds_by_batch=cell_sampling_seeds,
                )
                cell_revision_logs = list(cell_logs)
            if args.spad_backfill:
                if programs is None:
                    raise RuntimeError("SPAD programs were not built")
                if int(args.spad_max_anchor_revisions) > 0:
                    revision_slots = [
                        list(
                            limited_anchor_revision_slots(
                                program,
                                max_anchors=int(args.spad_max_anchor_revisions),
                            )
                        )
                        for program in programs
                    ]
                else:
                    revision_slots = [
                        list(anchor_revision_slots(program)) for program in programs
                    ]
                anchor_sampling_seeds = [
                    int(args.seed)
                    + int(item["sample_idx"]) * 1_000_003
                    + 90_000_019
                    for item in batch
                ]
                outputs, revision_logs = revise_spad_anchors(
                    model,
                    outputs,
                    prompt_length=input_ids.shape[1],
                    gen_length=gen_length,
                    revision_slots_by_batch=revision_slots,
                    attention_mask=attention_mask,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                    mask_id=MASK_TOKEN_ID,
                    allowed_token_ids_by_generation_pos=allowed,
                    atom_count_grammar=None,
                    lightweight_decoding_constraints=lightweight_constraints,
                    suffix_visible=not args.spad_hide_suffix,
                    sampling_seeds_by_batch=anchor_sampling_seeds,
                )
            generated_ids = outputs[:, input_ids.shape[1] :]
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            for (
                item,
                text,
                revision_log,
                cell_revision_log,
                basin_cell_revision_log,
                basin_block_revision_log,
                basin_metadata,
            ) in zip(
                batch,
                decoded,
                revision_logs,
                cell_revision_logs,
                basin_cell_revision_logs,
                basin_block_revision_logs,
                basin_closure_metadata,
                strict=True,
            ):
                sample_idx = int(item["sample_idx"])
                raw_record: Dict[str, Any] = {
                    "sample_idx": sample_idx,
                    "text": text,
                    "representation": "dynamic_v1",
                    "r5_representation": r5_representation,
                    "plan_state": item["plan_state"],
                    "conditioning_prompt": item["prompt"].rstrip(),
                    "prompt_record_idx": item["prompt_record_idx"],
                    "generation_schedule": args.generation_schedule,
                    "spad_backfill": bool(args.spad_backfill),
                    "spad_cell_closure": bool(args.spad_cell_closure),
                    "spad_max_anchor_revisions": int(
                        args.spad_max_anchor_revisions
                    ),
                    "spad_suffix_visible": bool(
                        args.spad_backfill and not args.spad_hide_suffix
                    ),
                    "spad_revision_log": revision_log,
                    "spad_cell_revision_log": cell_revision_log,
                }
                if args.spad_basin_closure:
                    raw_record.update(
                        spad_basin_closure_record_fields(
                            cell_revision_log=basin_cell_revision_log or {},
                            species_block_revision_log=(
                                basin_block_revision_log
                            ),
                            metadata=basin_metadata or {},
                        )
                    )
                if item.get("prompt_record") is not None:
                    source = dict(item["prompt_record"])
                    source.pop(args.prompt_field, None)
                    raw_record["prompt_record"] = source
                metrics["decoded_samples"] += 1
                try:
                    if args.spad_basin_closure and not bool(
                        (basin_metadata or {}).get("final_geometry_supported")
                    ):
                        raise ValueError(
                            "SPAD basin closure final geometry is outside support"
                        )
                    arrays = validate_answer_matches_plan(item["plan_state"], text)
                    metrics["parse_success"] += 1
                    metrics["plan_match_success"] += 1
                    if process_one is not None:
                        graph, cif = graph_from_arrays(arrays, process_one)
                        graph["sample_idx"] = sample_idx
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
    if is_main:
        (args.output_dir / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
