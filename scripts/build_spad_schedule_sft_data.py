#!/usr/bin/env python3
"""Build full-MP20 teacher SFT rows matching SPAD predictor/backfill states."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.spad_program import (  # noqa: E402
    LATTICE_POSITIONS,
    anchor_revision_slots,
    coordinate_positions,
    program_from_element_order,
    spad_predictor_position_groups,
)


SCHEMA = "rollout_matched_transition_v1"
MANIFEST_SCHEMA = "spad_schedule_sft_manifest_v1"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def program_orders(path: Path) -> dict[int, list[str]]:
    return {
        int(row["source_row_idx"]): [
            str(value) for value in row["contact_tree_order_symbols"]
        ]
        for row in iter_jsonl(path)
    }


def geometry_positions(num_atoms: int) -> list[int]:
    values = list(LATTICE_POSITIONS)
    for slot in range(int(num_atoms)):
        values.extend(coordinate_positions(slot))
    return values


def build_schedule_row(
    source: Mapping[str, Any],
    *,
    source_idx: int,
    order: list[str] | None,
    seed: int,
) -> dict[str, Any]:
    plan = source["plan_state"]
    elements = [str(value) for value in plan["elements"]]
    program_source = "contact_tree_teacher"
    if order is None:
        order = elements
        program_source = "canonical_missing_pointer_semantics"
    program = program_from_element_order(
        plan, order, order_source=program_source
    )
    mode_id = int(source_idx) % 3
    if mode_id == 0:
        mode = "deterministic_random_geometry"
        candidates = geometry_positions(program.num_atoms)
        rng = random.Random(int(seed) + int(source_idx) * 1_000_003)
        probability = 0.1 + 0.8 * rng.random()
        forced = [value for value in candidates if rng.random() < probability]
        if not forced:
            forced = [candidates[rng.randrange(len(candidates))]]
        loss = list(forced)
    elif mode_id == 1:
        mode = "program_predictor"
        groups = spad_predictor_position_groups(program)
        stage = 1 + (int(source_idx) // 3) % (len(groups) - 1)
        forced = [
            int(position)
            for group in groups[stage:]
            for position in group
        ]
        loss = [int(position) for position in groups[stage]]
    else:
        mode = "suffix_visible_anchor_correction"
        revisions = anchor_revision_slots(program)
        slot = revisions[(int(source_idx) // 3) % len(revisions)]
        forced = list(coordinate_positions(slot))
        loss = list(forced)
    return {
        "schema": SCHEMA,
        "source_row_idx": int(source_idx),
        "source_split": str(source.get("source_split") or ""),
        "prompt": str(source["prompt"]),
        "answer": str(source["answer"]),
        "source_answer": str(source["answer"]),
        "plan_state": dict(plan),
        "num_atoms": int(plan["N"]),
        "sample_weight": 1.0,
        "loss_profile": "fixed_slot",
        "mask_policy": "normal",
        "spad_mask_class": mode,
        "species_program": list(program.element_order),
        "species_program_source": program.order_source,
        "forced_mask_positions": sorted(set(forced)),
        "loss_positions": sorted(set(loss)),
        "outcomes_read": False,
    }


def build_split(
    *,
    source_path: Path,
    pointer_path: Path,
    output_path: Path,
    seed: int,
) -> dict[str, Any]:
    orders = program_orders(pointer_path)
    modes: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    rows = 0
    with output_path.open("x", encoding="utf-8", newline="\n") as output:
        for source in iter_jsonl(source_path):
            source_idx = int(source["source_row_idx"])
            row = build_schedule_row(
                source,
                source_idx=source_idx,
                order=orders.get(source_idx),
                seed=seed,
            )
            modes[row["spad_mask_class"]] += 1
            sources[row["species_program_source"]] += 1
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            rows += 1
    return {
        "rows": rows,
        "mask_classes": dict(sorted(modes.items())),
        "program_sources": dict(sorted(sources.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-data-dir", type=Path, required=True)
    parser.add_argument("--pointer-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
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
        "teacher": "full MP20 teacher Compact Plan/body",
        "one_row_per_source": True,
        "exact_N_elements_never_masked": True,
        "energy_hull_chgnet_model494_read": False,
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
