#!/usr/bin/env python3
"""Build teacher-only SFT states for SPAD basin closure.

The physical body remains in canonical dynamic ``7+4N`` storage order.  Each
source contributes exactly one deterministic rollout state from either:

* sequential cell closure with all coordinates visible; or
* one component of one species block, visited in reverse Llama-program order.

No outcome, energy, force, relaxation, or generated structure is read here.
``source_answer`` intentionally equals the clean MP20 teacher answer; physical
corruptions are a separate future data extension.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.dynamic_crystal import parse_dynamic_answer  # noqa: E402
from crystal_dlm.spad_program import (  # noqa: E402
    LATTICE_POSITIONS,
    coordinate_positions,
    element_position,
    program_from_element_order,
    reverse_species_block_revision_slots,
)


SCHEMA = "rollout_matched_transition_v1"
CLOSURE_SCHEMA = "spad_basin_closure_sft_v1"
MANIFEST_SCHEMA = "spad_basin_closure_sft_manifest_v1"
FALLBACK_SOURCE = "canonical_missing_pointer_semantics"
LATTICE_COMPONENTS = ("a", "b", "c", "alpha", "beta", "gamma")
COORDINATE_COMPONENTS = ("x", "y", "z")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            yield value


def _plan_components(plan: Mapping[str, Any]) -> tuple[int, list[str], list[int]]:
    num_atoms = int(plan.get("N") or 0)
    elements = [str(value) for value in plan.get("elements") or ()]
    counts = [int(value) for value in plan.get("counts") or ()]
    if not elements or len(elements) != len(counts):
        raise ValueError("Plan elements/counts are malformed")
    if len(elements) != len(set(elements)):
        raise ValueError("Plan elements must be unique")
    if any(value <= 0 for value in counts) or sum(counts) != num_atoms:
        raise ValueError("Plan counts disagree with exact N")
    return num_atoms, elements, counts


def _expanded_species(elements: Sequence[str], counts: Sequence[int]) -> list[str]:
    return [
        str(symbol)
        for symbol, count in zip(elements, counts, strict=True)
        for _ in range(int(count))
    ]


def _validate_teacher_body(answer: str, plan: Mapping[str, Any]) -> None:
    num_atoms, elements, counts = _plan_components(plan)
    parsed = parse_dynamic_answer(str(answer), strict=True)
    if int(parsed["num_atoms"]) != num_atoms:
        raise ValueError("teacher answer N differs from Plan N")
    if list(parsed["species"]) != _expanded_species(elements, counts):
        raise ValueError("teacher answer element slots differ from canonical Plan expansion")
    if len(parsed["tokens"]) != 7 + 4 * num_atoms:
        raise RuntimeError("teacher answer no longer has exact 7+4N length")


def _canonical_program(plan: Mapping[str, Any]) -> list[str]:
    _num_atoms, elements, _counts = _plan_components(plan)
    return elements


def _program_from_row(row: Mapping[str, Any]) -> tuple[list[str] | None, str | None]:
    if row.get("species_program"):
        return (
            [str(value) for value in row["species_program"]],
            str(row.get("species_program_source") or "pointer_program"),
        )
    if row.get("contact_tree_order_symbols"):
        return (
            [str(value) for value in row["contact_tree_order_symbols"]],
            str(row.get("species_program_source") or "contact_tree_teacher"),
        )
    return None, None


def pointer_programs(path: Path) -> dict[int, tuple[list[str], str]]:
    programs: dict[int, tuple[list[str], str]] = {}
    for row in iter_jsonl(path):
        source_idx = int(row["source_row_idx"])
        if source_idx in programs:
            raise ValueError(f"pointer data duplicates source_row_idx {source_idx}")
        order, source = _program_from_row(row)
        if order is None or source is None:
            raise ValueError(f"pointer row {source_idx} lacks a species program")
        programs[source_idx] = (order, source)
    return programs


def _resolve_program(
    source: Mapping[str, Any],
    *,
    pointer: tuple[list[str], str] | None,
) -> tuple[list[str], str, bool]:
    plan = source["plan_state"]
    canonical = _canonical_program(plan)
    if pointer is not None:
        order, program_source = pointer
        return list(order), str(program_source), str(program_source) != FALLBACK_SOURCE

    source_order, source_name = _program_from_row(source)
    if source_order is not None:
        if source_name != FALLBACK_SOURCE or source_order != canonical:
            raise ValueError(
                "a source missing pointer data must declare the canonical fallback"
            )
        return source_order, FALLBACK_SOURCE, False

    # Raw teacher-only rows do not necessarily carry pointer metadata.  Preserve
    # them explicitly as missing-semantics canonical fallbacks; do not fabricate
    # a learned pointer trace.
    return canonical, FALLBACK_SOURCE, False


def closure_states(program: Any) -> list[dict[str, Any]]:
    """Enumerate runtime-matched sequential closure states for one Plan."""

    states: list[dict[str, Any]] = []
    for component_index, position in enumerate(LATTICE_POSITIONS):
        states.append(
            {
                "kind": "cell_sequential_component",
                "forced": list(LATTICE_POSITIONS[component_index:]),
                "loss": [int(position)],
                "metadata": {
                    "cell_component": LATTICE_COMPONENTS[component_index],
                    "cell_component_index": component_index,
                    "active_position": int(position),
                    "committed_lattice_positions": list(
                        LATTICE_POSITIONS[:component_index]
                    ),
                    "remaining_lattice_positions": list(
                        LATTICE_POSITIONS[component_index:]
                    ),
                    "all_coordinates_visible": True,
                },
            }
        )

    runtime_blocks = reverse_species_block_revision_slots(program)
    reverse_entries = tuple(reversed(program.entries))
    if len(runtime_blocks) != len(reverse_entries):
        raise RuntimeError("runtime closure blocks differ from reverse program entries")
    for reverse_block_index, (entry, slots) in enumerate(
        zip(reverse_entries, runtime_blocks, strict=True)
    ):
        symbol = str(entry.symbol)
        slots = tuple(int(slot) for slot in slots)
        if set(slots) != set(int(slot) for slot in entry.slot_indices):
            raise RuntimeError("runtime closure block differs from its program species")
        flattened = [
            int(position)
            for slot in slots
            for position in coordinate_positions(slot)
        ]
        for component_offset, active_position in enumerate(flattened):
            site_offset, component_index = divmod(component_offset, 3)
            states.append(
                {
                    "kind": "reverse_species_block_component",
                    "forced": flattened[component_offset:],
                    "loss": [int(active_position)],
                    "metadata": {
                        "species": symbol,
                        "reverse_block_index": reverse_block_index,
                        "block_slot_indices": list(slots),
                        "site_slot_index": int(slots[site_offset]),
                        "site_index_within_block": site_offset,
                        "coordinate_component": COORDINATE_COMPONENTS[component_index],
                        "coordinate_component_index": component_index,
                        "active_position": int(active_position),
                        "committed_block_coordinate_positions": flattened[:component_offset],
                        "remaining_block_coordinate_positions": flattened[component_offset:],
                        "lattice_visible": True,
                        "other_species_visible": True,
                        "suffix_outside_active_block_visible": True,
                    },
                }
            )
    expected = 6 + 3 * int(program.num_atoms)
    if len(states) != expected:
        raise RuntimeError("closure state coverage differs from 6+3N")
    return states


def deterministic_state_index(
    *, source_split: str, source_idx: int, seed: int, state_count: int
) -> int:
    if int(state_count) <= 0:
        raise ValueError("state_count must be positive")
    payload = f"{int(seed)}:{source_split}:{int(source_idx)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % int(state_count)


def build_closure_row(
    source: Mapping[str, Any],
    *,
    source_idx: int,
    pointer: tuple[list[str], str] | None,
    seed: int,
    state_index: int | None = None,
) -> dict[str, Any]:
    plan = source.get("plan_state")
    if not isinstance(plan, Mapping):
        raise ValueError("teacher row lacks plan_state")
    answer = str(source["answer"])
    _validate_teacher_body(answer, plan)
    order, program_source, pointer_semantics_available = _resolve_program(
        source, pointer=pointer
    )
    program = program_from_element_order(
        plan, order, order_source=program_source
    )
    states = closure_states(program)
    split = str(source.get("source_split") or "")
    selected_index = (
        deterministic_state_index(
            source_split=split,
            source_idx=source_idx,
            seed=seed,
            state_count=len(states),
        )
        if state_index is None
        else int(state_index)
    )
    if not 0 <= selected_index < len(states):
        raise ValueError("state_index lies outside this source's closure states")
    selected = states[selected_index]
    forced = [int(value) for value in selected["forced"]]
    loss = [int(value) for value in selected["loss"]]

    num_atoms = int(plan["N"])
    protected = {0, *(element_position(slot) for slot in range(num_atoms))}
    if not set(loss) <= set(forced):
        raise RuntimeError("loss_positions must be a subset of forced_mask_positions")
    if protected & set(forced):
        raise RuntimeError("closure state attempted to mask N or an element token")

    metadata = dict(selected["metadata"])
    metadata.update(
        {
            "schedule": "cell_then_reverse_llama_species_blocks_v1",
            "state_index": selected_index,
            "state_count": len(states),
            "program_order": list(program.element_order),
            "reverse_block_order": [
                str(entry.symbol) for entry in reversed(program.entries)
            ],
            "pointer_semantics_available": bool(pointer_semantics_available),
            "source_answer_is_clean_teacher": True,
        }
    )
    return {
        "schema": SCHEMA,
        "closure_schema": CLOSURE_SCHEMA,
        "source_row_idx": int(source_idx),
        "source_split": split,
        "prompt": str(source["prompt"]),
        "answer": answer,
        "source_answer": answer,
        "plan_state": dict(plan),
        "num_atoms": num_atoms,
        "sample_weight": 1.0,
        "loss_profile": "fixed_slot",
        "mask_policy": "normal",
        "spad_mask_class": str(selected["kind"]),
        "species_program": list(program.element_order),
        "species_program_source": program.order_source,
        "forced_mask_positions": forced,
        "loss_positions": loss,
        "closure": metadata,
        "outcomes_read": False,
    }


def build_split(
    *,
    source_path: Path,
    pointer_path: Path,
    output_path: Path,
    seed: int,
) -> dict[str, Any]:
    programs = pointer_programs(pointer_path)
    mask_classes: Counter[str] = Counter()
    program_sources: Counter[str] = Counter()
    seen_sources: set[int] = set()
    rows = 0
    with output_path.open("x", encoding="utf-8", newline="\n") as output:
        for source in iter_jsonl(source_path):
            source_idx = int(source["source_row_idx"])
            if source_idx in seen_sources:
                raise ValueError(f"teacher data duplicates source_row_idx {source_idx}")
            seen_sources.add(source_idx)
            row = build_closure_row(
                source,
                source_idx=source_idx,
                pointer=programs.get(source_idx),
                seed=int(seed),
            )
            mask_classes[row["spad_mask_class"]] += 1
            program_sources[row["species_program_source"]] += 1
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            rows += 1
    unexpected_pointer_sources = set(programs) - seen_sources
    if unexpected_pointer_sources:
        raise ValueError(
            "pointer data contains sources absent from teacher split: "
            f"{sorted(unexpected_pointer_sources)[:5]}"
        )
    return {
        "rows": rows,
        "one_row_per_source": True,
        "mask_classes": dict(sorted(mask_classes.items())),
        "program_sources": dict(sorted(program_sources.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-data-dir", type=Path, required=True)
    parser.add_argument("--pointer-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    splits = {
        split: build_split(
            source_path=args.teacher_data_dir / f"{split}.jsonl",
            pointer_path=args.pointer_data_dir / f"{split}.jsonl",
            output_path=args.output_dir / f"{split}.jsonl",
            seed=int(args.seed),
        )
        for split in ("train", "val")
    }
    shutil.copy2(
        args.teacher_data_dir / "vocab_tokens.txt",
        args.output_dir / "vocab_tokens.txt",
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "seed": int(args.seed),
        "teacher": "full MP20 clean teacher",
        "schedule": "cell_then_reverse_llama_species_blocks_v1",
        "one_deterministic_closure_state_per_source": True,
        "source_answer": "clean_teacher_no_physical_corruption",
        "exact_7_plus_4N": True,
        "exact_N_elements_never_masked": True,
        "outcomes_read": False,
        "energy_force_hull_relaxation_read": False,
        "splits": splits,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
