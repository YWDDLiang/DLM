#!/usr/bin/env python3
"""Attach frozen H1A2 pointer programs without changing original Plan fields.

All input rows, including upstream failures, remain in the output in the same
order.  A controller failure is explicitly marked and never replaced by a
canonical program.  Consumers must require r03_control.status == 'ok' for the
controller arm.  The input request ledger, not successful exports, determines
the scientific denominator.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crystal_dlm.r03_control_pointer import (
    FEATURE_SOURCE,
    POINTER_STATE_SCHEMA,
    FormulaFeatureSpec,
    R03ControlPointer,
    R03ControlPointerConfig,
    compile_revision_program,
    formula_hidden_batch,
    plan_composition,
    pointer_batch,
    sha256_file,
)
from scripts.train_r03_control_pointer import (
    add_feature_arguments,
    iter_jsonl,
    load_frozen_p0,
    validate_approved_feature_spec,
    write_json,
)


EXPORT_SCHEMA = "r03_control_program_export_v1"


def attach_program(
    row: Mapping[str, Any], indices: Sequence[int], *, pointer_sha256: str,
) -> dict[str, Any]:
    """Add only new top-level metadata; never rewrite rich Plan/body fields."""
    plan = row.get("plan_state")
    if not isinstance(plan, Mapping):
        raise ValueError("program export requires original plan_state")
    candidates = [symbol for symbol, _count in plan_composition(plan)]
    order_indices = [int(value) for value in indices]
    if sorted(order_indices) != list(range(len(candidates))):
        raise ValueError("pointer program is not an exact candidate permutation")
    reserved = {"species_program", "species_program_indices", "species_program_source", "r03_control"}
    if reserved.intersection(row):
        raise ValueError("export would overwrite existing program metadata")
    order = [candidates[index] for index in order_indices]
    program = compile_revision_program(plan, order)
    return {
        **row,
        "species_program": order,
        "species_program_indices": order_indices,
        "species_program_source": "h1a2_frozen_formula_pointer",
        "r03_control": {
            **program, "status": "ok", "pointer_sha256": pointer_sha256,
            "feature_source": FEATURE_SOURCE,
            "feature_formula_rendering": "canonical_atomic_number_counts_control_only",
            "sampled_generation_modified": False,
        },
    }


def failure_record(row: Mapping[str, Any], *, status: str, error: Exception | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"status": status, "feature_source": FEATURE_SOURCE,
                             "sampled_generation_modified": False}
    if error is not None:
        value.update({"error_type": type(error).__name__, "error": str(error)})
    if "r03_control" in row:
        # Do not silently overwrite any previous metadata, including on errors.
        return {**row, "r03_control_export_error": value}
    return {**row, "r03_control": value}


def export_rows(
    rows: Sequence[Mapping[str, Any]], *, model: Any, tokenizer: Any, pointer: Any,
    feature_spec: FormulaFeatureSpec, pointer_sha256: str, batch_size: int,
    exact_prompt_text: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    output: list[dict[str, Any] | None] = [None] * len(rows)
    eligible: list[int] = []
    upstream_failures = 0
    controller_failures = 0
    noncanonical = 0
    for index, row in enumerate(rows):
        if (
            row.get("parsed") is False
            or ("attempt_status" in row and row["attempt_status"] != "complete")
            or not isinstance(row.get("plan_state"), Mapping)
        ):
            output[index] = failure_record(row, status="upstream_failure")
            upstream_failures += 1
            continue
        try:
            plan_composition(row["plan_state"])
            if any(key in row for key in ("species_program", "species_program_indices", "species_program_source", "r03_control")):
                raise ValueError("input already carries controller metadata")
        except (ValueError, TypeError, KeyError) as exc:
            output[index] = failure_record(row, status="controller_failure", error=exc)
            controller_failures += 1
            continue
        eligible.append(index)
    pointer.eval()
    device = next(pointer.parameters()).device
    start = time.monotonic()
    with torch.no_grad():
        for begin in range(0, len(eligible), batch_size):
            indices = eligible[begin:begin + batch_size]
            selected = [rows[index] for index in indices]
            # Infrastructure/model errors fail the run, rather than relabelling
            # an entire affected batch as scientific or composition failures.
            features = formula_hidden_batch(
                model, tokenizer, selected, spec=feature_spec, exact_prompt_text=exact_prompt_text,
            ).to(device)
            batch = {key: value.to(device) for key, value in pointer_batch(selected).items()}
            predicted = pointer.decode(features, batch["atomic_numbers"], batch["counts"], batch["valid_mask"])
            for row_number, output_index in enumerate(indices):
                size = int(batch["valid_mask"][row_number].sum())
                order = predicted[row_number, :size].tolist()
                try:
                    output[output_index] = attach_program(rows[output_index], order, pointer_sha256=pointer_sha256)
                    noncanonical += order != list(range(size))
                except (ValueError, TypeError, KeyError) as exc:
                    output[output_index] = failure_record(rows[output_index], status="controller_failure", error=exc)
                    controller_failures += 1
            print(json.dumps({"event": "control_export_progress", "completed": min(begin + batch_size, len(eligible)),
                              "eligible": len(eligible), "elapsed_seconds": round(time.monotonic() - start, 2)}), flush=True)
    if any(row is None for row in output):
        raise RuntimeError("program export lost an input row")
    successes = len(rows) - upstream_failures - controller_failures
    return output, {
        "input_rows": len(rows), "output_rows": len(output), "successful_programs": successes,
        "upstream_failures_preserved": upstream_failures,
        "controller_failures_preserved": controller_failures,
        "noncanonical_programs": noncanonical,
        "noncanonical_program_fraction": noncanonical / max(1, successes),
        "original_fields_modified": False, "initial_r03_schedule_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-model", type=Path, required=True)
    parser.add_argument("--planner-checkpoint", type=Path, required=True)
    parser.add_argument("--pointer-checkpoint", type=Path, required=True)
    parser.add_argument("--plans-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    add_feature_arguments(parser)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.output_jsonl.exists():
        raise FileExistsError(args.output_jsonl)
    spec = FormulaFeatureSpec(args.prompt_style, bool(args.include_sample_id))
    validate_approved_feature_spec(spec)

    import torch

    checkpoint = torch.load(args.pointer_checkpoint, map_location="cpu")
    if checkpoint.get("schema") != POINTER_STATE_SCHEMA:
        raise ValueError("this is not an R03 head-only controller checkpoint")
    if checkpoint["feature_spec"] != asdict(spec):
        raise ValueError("export feature prompt does not match controller training")
    exact_prompt = checkpoint["exact_prompt_text"]
    if args.prompt_text_file and args.prompt_text_file.read_text(encoding="utf-8") != exact_prompt:
        raise ValueError("export exact historical prompt differs from controller training")
    model, tokenizer, identity = load_frozen_p0(args.llama_model, args.planner_checkpoint, device=args.device)
    original_identity = checkpoint["planner_identity"]
    for key in ("adapter_sha256", "base_config_sha256", "llama_hidden_size", "tokenizer_size", "tokenizer_files"):
        if identity[key] != original_identity[key]:
            raise ValueError(f"export frozen P0 identity changed for {key}")
    pointer = R03ControlPointer(R03ControlPointerConfig(**checkpoint["config"]))
    pointer.load_state_dict(checkpoint["state_dict"], strict=True)
    pointer.to(args.device).eval()
    rows = list(iter_jsonl(args.plans_jsonl))
    if not rows:
        raise ValueError("empty input request/plan ledger")
    pointer_sha = sha256_file(args.pointer_checkpoint)
    output, report = export_rows(
        rows, model=model, tokenizer=tokenizer, pointer=pointer, feature_spec=spec,
        pointer_sha256=pointer_sha, batch_size=args.batch_size, exact_prompt_text=exact_prompt,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("x", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report.update({
        "schema": EXPORT_SCHEMA, "feature_spec": asdict(spec),
        "feature_formula_rendering": "canonical_atomic_number_counts_control_only",
        "input_sha256": sha256_file(args.plans_jsonl), "pointer_sha256": pointer_sha,
        "output_sha256": sha256_file(args.output_jsonl), "planner_identity": identity,
    })
    write_json(args.output_jsonl.with_suffix(".metrics.json"), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if report["controller_failures_preserved"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
