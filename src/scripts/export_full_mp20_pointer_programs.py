#!/usr/bin/env python3
"""Export learned pointer programs for every MP20 SPAD-SFT training row.

Rows represented in the typed pointer dataset are decoded by the frozen
Planner-Llama pointer.  Rows without typed pointer semantics retain the
canonical program explicitly declared by the SPAD-SFT dataset; no synthetic
typed transcript is constructed for them.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crystal_dlm.fixed_slot import Z_TO_SYMBOL


SCHEMA = "full_mp20_pointer_program_v1"
MANIFEST_SCHEMA = "full_mp20_pointer_program_manifest_v1"
SFT_MANIFEST_SCHEMA = "spad_schedule_sft_manifest_v1"
POINTER_MANIFEST_SCHEMA = "spad_species_pointer_manifest_v1"
POINTER_SOURCE = "frozen_planner_llama_pointer"
FALLBACK_SOURCE = "canonical_missing_pointer_semantics"
EXPECTED_SFT_ROWS = 27_136
EXPECTED_POINTER_ROWS = 24_558
EXPECTED_FALLBACK_ROWS = EXPECTED_SFT_ROWS - EXPECTED_POINTER_ROWS
POINTER_STATE_SCHEMA = "spad_species_pointer_state_v1"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            yield value


def _indexed_rows(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[int, Mapping[str, Any]]:
    indexed: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        source_idx = int(row["source_row_idx"])
        if source_idx in indexed:
            raise ValueError(f"{label} duplicates source_row_idx {source_idx}")
        indexed[source_idx] = row
    return indexed


def _plan_components(plan: Mapping[str, Any]) -> tuple[list[str], list[int]]:
    elements = [str(value) for value in plan.get("elements") or ()]
    counts = [int(value) for value in plan.get("counts") or ()]
    if not elements or len(elements) != len(counts) or len(set(elements)) != len(elements):
        raise ValueError("SFT Plan elements/counts are malformed")
    if any(value <= 0 for value in counts) or sum(counts) != int(plan.get("N", -1)):
        raise ValueError("SFT Plan composition disagrees with N")
    return elements, counts


def _pointer_components(row: Mapping[str, Any]) -> tuple[list[str], list[int]]:
    atomic_numbers = [int(value) for value in row.get("canonical_atomic_numbers") or ()]
    counts = [int(value) for value in row.get("canonical_element_counts") or ()]
    if not atomic_numbers or len(atomic_numbers) != len(counts):
        raise ValueError("typed pointer row lacks canonical Plan composition")
    try:
        elements = [str(Z_TO_SYMBOL[value]) for value in atomic_numbers]
    except KeyError as error:
        raise ValueError("typed pointer row contains an unsupported atomic number") from error
    return elements, counts


def validate_input_manifests(
    sft_manifest: Mapping[str, Any],
    pointer_manifest: Mapping[str, Any],
    *,
    expected_sft_rows: int,
    expected_pointer_rows: int,
) -> None:
    """Validate only the dataset facts needed for the full-train join."""

    if sft_manifest.get("schema") != SFT_MANIFEST_SCHEMA:
        raise ValueError("SPAD-SFT manifest schema changed")
    if pointer_manifest.get("schema") != POINTER_MANIFEST_SCHEMA:
        raise ValueError("typed pointer manifest schema changed")
    sft_train = (sft_manifest.get("splits") or {}).get("train") or {}
    pointer_train = (pointer_manifest.get("splits") or {}).get("train") or {}
    if int(sft_train.get("rows", -1)) != int(expected_sft_rows):
        raise ValueError("SPAD-SFT train row count differs from the export contract")
    if int(pointer_train.get("rows", -1)) != int(expected_pointer_rows):
        raise ValueError("typed pointer train row count differs from the export contract")
    expected_missing = int(expected_sft_rows) - int(expected_pointer_rows)
    program_sources = sft_train.get("program_sources") or {}
    if int(program_sources.get(FALLBACK_SOURCE, 0)) != expected_missing:
        raise ValueError("SPAD-SFT manifest canonical fallback count changed")


def join_full_mp20_programs(
    sft_rows: Sequence[Mapping[str, Any]],
    pointer_rows: Sequence[Mapping[str, Any]],
    decoded_indices: Mapping[int, Sequence[int]],
    *,
    expected_sft_rows: int,
    expected_pointer_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join decoded pointer rows with declared canonical SFT fallbacks.

    This function contains no model loading and is intentionally fixture-testable.
    """

    sft_by_source = _indexed_rows(sft_rows, label="SPAD-SFT")
    pointer_by_source = _indexed_rows(pointer_rows, label="typed pointer")
    expected_sources = set(range(int(expected_sft_rows)))
    if set(sft_by_source) != expected_sources:
        raise ValueError("SPAD-SFT source_row_idx values are not contiguous")
    if len(pointer_by_source) != int(expected_pointer_rows):
        raise ValueError("typed pointer row count differs from the join contract")
    if not set(pointer_by_source) <= expected_sources:
        raise ValueError("typed pointer contains a source outside SPAD-SFT")
    if set(decoded_indices) != set(pointer_by_source):
        raise ValueError("decoded pointer coverage differs from typed pointer coverage")

    missing_sources = sorted(expected_sources - set(pointer_by_source))
    if len(missing_sources) != int(expected_sft_rows) - int(expected_pointer_rows):
        raise ValueError("canonical fallback set has the wrong size")

    output: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    noncanonical = 0
    for sample_idx, source_idx in enumerate(range(int(expected_sft_rows))):
        sft = sft_by_source[source_idx]
        plan = sft.get("plan_state")
        if not isinstance(plan, Mapping):
            raise ValueError(f"SPAD-SFT row {source_idx} lacks plan_state")
        elements, counts = _plan_components(plan)
        if source_idx in pointer_by_source:
            pointer_elements, pointer_counts = _pointer_components(
                pointer_by_source[source_idx]
            )
            if pointer_elements != elements or pointer_counts != counts:
                raise ValueError(
                    f"SPAD-SFT and typed pointer Plan differ at source_row_idx {source_idx}"
                )
            indices = [int(value) for value in decoded_indices[source_idx]]
            if sorted(indices) != list(range(len(elements))):
                raise ValueError(
                    f"pointer output is not an exact permutation at source_row_idx {source_idx}"
                )
            program = [elements[index] for index in indices]
            program_source = POINTER_SOURCE
        else:
            if str(sft.get("species_program_source")) != FALLBACK_SOURCE:
                raise ValueError(
                    f"missing typed pointer row {source_idx} is not a declared canonical fallback"
                )
            program = [str(value) for value in sft.get("species_program") or ()]
            indices = list(range(len(elements)))
            if program != elements:
                raise ValueError(
                    f"canonical fallback program changed at source_row_idx {source_idx}"
                )
            program_source = FALLBACK_SOURCE

        if sorted(program) != sorted(elements) or program != [
            elements[index] for index in indices
        ]:
            raise RuntimeError("exported program changed Plan composition or order mapping")
        source_counts[program_source] += 1
        noncanonical += int(indices != list(range(len(elements))))
        output.append(
            {
                "schema": SCHEMA,
                "sample_idx": sample_idx,
                "source_row_idx": source_idx,
                "plan_state": dict(plan),
                "prompt": str(sft["prompt"]),
                "teacher_answer": str(sft["answer"]),
                "species_program": program,
                "species_program_indices": indices,
                "species_program_source": program_source,
                "outcomes_read": False,
            }
        )

    if [row["sample_idx"] for row in output] != list(range(int(expected_sft_rows))):
        raise RuntimeError("exported sample_idx values are not contiguous")
    if [row["source_row_idx"] for row in output] != list(range(int(expected_sft_rows))):
        raise RuntimeError("exported source_row_idx values are not contiguous")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "rows": len(output),
        "pointer_decoded_rows": int(source_counts[POINTER_SOURCE]),
        "canonical_missing_pointer_rows": int(source_counts[FALLBACK_SOURCE]),
        "canonical_missing_pointer_source_row_indices": missing_sources,
        "program_sources": dict(sorted(source_counts.items())),
        "noncanonical_programs": noncanonical,
        "exact_permutation": True,
        "composition_mutation": False,
        "typed_transcript_fabricated_for_missing_rows": False,
        "outcomes_read": False,
    }
    if manifest["pointer_decoded_rows"] != int(expected_pointer_rows):
        raise RuntimeError("exported learned-pointer count changed")
    return output, manifest


def decode_pointer_indices(
    pointer_rows: Sequence[Mapping[str, Any]],
    *,
    llama_model: Path,
    planner_final: Path,
    pointer_state: Path,
    c3fd_checkpoint: Path,
    vocabulary: Path,
    batch_size: int,
) -> dict[int, list[int]]:
    """Run the existing trained pointer decoder on typed rows only."""

    import torch

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

    if not torch.cuda.is_available():
        raise RuntimeError("full-MP20 pointer export requires its allocated GPU")
    if int(batch_size) <= 0:
        raise ValueError("batch-size must be positive")
    device = torch.device("cuda")
    bundle = metadata_bundle(c3fd_checkpoint, vocabulary)
    llama, typed, typed_config = load_frozen_planner(
        llama_model=llama_model,
        planner_final=planner_final,
        device=device,
    )
    state = torch.load(pointer_state, map_location="cpu")
    if state.get("schema") != POINTER_STATE_SCHEMA:
        raise ValueError("pointer state schema changed")
    config = SpeciesPointerConfig(**state["config"])
    if int(config.llama_hidden_size) != int(typed_config.llama_hidden_size):
        raise ValueError("pointer/Planner hidden sizes differ")
    pointer = PlanConditionedSpeciesPointer(config).to(device)
    pointer.load_state_dict(state["state_dict"], strict=True)
    pointer.eval()

    decoded_by_source: dict[int, list[int]] = {}
    with torch.inference_mode():
        for offset in range(0, len(pointer_rows), int(batch_size)):
            chunk = list(pointer_rows[offset : offset + int(batch_size)])
            batch = move_batch(collate_pointer_rows(chunk, bundle=bundle), device)
            decoded = pointer.decode(
                terminal_hidden(llama, typed, batch),
                batch["pointer_atomic_numbers"],
                batch["pointer_counts"],
                batch["pointer_valid_mask"],
                batch["pointer_soft_field_ids"],
            ).detach().cpu()
            valid = batch["pointer_valid_mask"].detach().cpu()
            for row_index, row in enumerate(chunk):
                source_idx = int(row["source_row_idx"])
                if source_idx in decoded_by_source:
                    raise ValueError(f"typed pointer duplicates source_row_idx {source_idx}")
                size = int(valid[row_index].sum().item())
                decoded_by_source[source_idx] = [
                    int(value) for value in decoded[row_index, :size].tolist()
                ]
    return decoded_by_source


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-data-dir", type=Path, required=True)
    parser.add_argument("--pointer-data-dir", type=Path, required=True)
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

    sft_manifest = json.loads(
        (args.sft_data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    pointer_manifest = json.loads(
        (args.pointer_data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    validate_input_manifests(
        sft_manifest,
        pointer_manifest,
        expected_sft_rows=EXPECTED_SFT_ROWS,
        expected_pointer_rows=EXPECTED_POINTER_ROWS,
    )
    sft_rows = list(iter_jsonl(args.sft_data_dir / "train.jsonl"))
    pointer_rows = list(iter_jsonl(args.pointer_data_dir / "train.jsonl"))
    decoded = decode_pointer_indices(
        pointer_rows,
        llama_model=args.llama_model,
        planner_final=args.planner_final,
        pointer_state=args.pointer_state,
        c3fd_checkpoint=args.c3fd_checkpoint,
        vocabulary=args.vocabulary,
        batch_size=int(args.batch_size),
    )
    output, manifest = join_full_mp20_programs(
        sft_rows,
        pointer_rows,
        decoded,
        expected_sft_rows=EXPECTED_SFT_ROWS,
        expected_pointer_rows=EXPECTED_POINTER_ROWS,
    )
    if manifest["canonical_missing_pointer_rows"] != EXPECTED_FALLBACK_ROWS:
        raise RuntimeError("full-MP20 canonical fallback count changed")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "plans_for_dlm.jsonl", output)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
