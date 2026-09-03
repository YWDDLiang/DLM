#!/usr/bin/env python3
"""Export frozen Planner-Llama pointer programs for SPAD-E train Plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from crystal_dlm.fixed_slot import Z_TO_SYMBOL
from crystal_dlm.species_program_pointer import (
    PlanConditionedSpeciesPointer,
    SpeciesPointerConfig,
)
from scripts.train_spad_species_pointer import (
    collate_pointer_rows,
    load_frozen_planner,
    metadata_bundle,
    move_batch,
    terminal_hidden,
)


SCHEMA = "spad_energy_pointer_program_v1"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(path)
                yield value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--llama-model", type=Path, required=True)
    parser.add_argument("--planner-final", type=Path, required=True)
    parser.add_argument("--pointer-state", type=Path, required=True)
    parser.add_argument("--c3fd-checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("pointer export requires its allocated GPU")
    manifest = json.loads((args.cohort_dir / "manifest.json").read_text())
    if manifest.get("schema") != "spad_energy_train_cohort_v1" or int(
        manifest.get("selected", 0)
    ) != 2048:
        raise ValueError("SPAD-E cohort contract changed")
    pointer_rows = list(iter_jsonl(args.cohort_dir / "pointer_rows.jsonl"))
    teacher_rows = list(iter_jsonl(args.cohort_dir / "teacher_rows.jsonl"))
    if len(pointer_rows) != 2048 or len(teacher_rows) != 2048:
        raise ValueError("SPAD-E cohort row count changed")
    teacher_by_sample = {int(row["sample_idx"]): row for row in teacher_rows}
    if set(teacher_by_sample) != set(range(2048)):
        raise ValueError("teacher sample_idx coverage changed")

    device = torch.device("cuda")
    bundle = metadata_bundle(args.c3fd_checkpoint, args.vocabulary)
    llama, typed, typed_config = load_frozen_planner(
        llama_model=args.llama_model,
        planner_final=args.planner_final,
        device=device,
    )
    state = torch.load(args.pointer_state, map_location="cpu")
    config = SpeciesPointerConfig(**state["config"])
    if int(config.llama_hidden_size) != int(typed_config.llama_hidden_size):
        raise ValueError("pointer/Planner hidden sizes differ")
    pointer = PlanConditionedSpeciesPointer(config).to(device)
    pointer.load_state_dict(state["state_dict"], strict=True)
    pointer.eval()

    output_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for offset in range(0, len(pointer_rows), int(args.batch_size)):
            chunk = pointer_rows[offset : offset + int(args.batch_size)]
            batch = move_batch(collate_pointer_rows(chunk, bundle=bundle), device)
            hidden = terminal_hidden(llama, typed, batch)
            decoded = pointer.decode(
                hidden,
                batch["pointer_atomic_numbers"],
                batch["pointer_counts"],
                batch["pointer_valid_mask"],
                batch["pointer_soft_field_ids"],
            ).detach().cpu()
            valid = batch["pointer_valid_mask"].detach().cpu()
            atomic = batch["pointer_atomic_numbers"].detach().cpu()
            for local_index, source in enumerate(chunk):
                sample_idx = int(source["sample_idx"])
                teacher = teacher_by_sample[sample_idx]
                size = int(valid[local_index].sum().item())
                indices = [int(value) for value in decoded[local_index, :size].tolist()]
                if sorted(indices) != list(range(size)):
                    raise RuntimeError("pointer output is not an exact permutation")
                canonical = [
                    str(Z_TO_SYMBOL[int(value)])
                    for value in atomic[local_index, :size].tolist()
                ]
                plan_elements = [str(value) for value in teacher["plan_state"]["elements"]]
                if canonical != plan_elements:
                    raise ValueError("pointer and teacher Plan element order differ")
                program = [canonical[index] for index in indices]
                output_rows.append(
                    {
                        "schema": SCHEMA,
                        "sample_idx": sample_idx,
                        "source_row_idx": int(teacher["source_row_idx"]),
                        "plan_state": teacher["plan_state"],
                        "prompt": teacher["prompt"],
                        "teacher_answer": teacher["answer"],
                        "species_program": program,
                        "species_program_indices": indices,
                        "species_program_source": "frozen_planner_llama_pointer",
                        "outcomes_read": False,
                    }
                )
    output_rows.sort(key=lambda row: int(row["sample_idx"]))
    if [int(row["sample_idx"]) for row in output_rows] != list(range(2048)):
        raise RuntimeError("pointer export changed cohort order")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "plans_for_dlm.jsonl").open("x", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "schema": SCHEMA,
        "rows": 2048,
        "program_source": "frozen_planner_llama_pointer",
        "noncanonical_programs": sum(
            row["species_program_indices"]
            != list(range(len(row["species_program_indices"])))
            for row in output_rows
        ),
        "composition_mutation": False,
        "outcomes_read": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
