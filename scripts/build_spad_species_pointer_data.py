#!/usr/bin/env python3
"""Build MP20-only geometry targets for the SPAD Llama species pointer."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.composition_identity import canonical_symbol_counts  # noqa: E402
from crystal_dlm.fixed_slot import SYMBOL_TO_Z, Z_TO_SYMBOL  # noqa: E402
from crystal_dlm.species_program_pointer import (  # noqa: E402
    maximum_contact_tree_order,
)


SCHEMA = "spad_species_pointer_row_v1"
MANIFEST_SCHEMA = "spad_species_pointer_manifest_v1"
FUSED_SCHEMA = "c3fd_llama_fused_typed_dataset_v1"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            yield value


def _teacher_target(payload: tuple[int, str]) -> tuple[int, list[str]]:
    source_row_idx, cif = payload
    from pymatgen.core import Structure

    structure = Structure.from_str(cif, fmt="cif")
    if not bool(structure.is_ordered):
        raise ValueError(f"MP20 source row {source_row_idx} is not ordered")
    species = [str(site.specie.symbol) for site in structure.sites]
    order = maximum_contact_tree_order(species, structure.distance_matrix.tolist())
    return int(source_row_idx), list(order)


def _mp20_cifs(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "cif" not in rows[0]:
        raise ValueError(f"{path} lacks MP20 CIF rows")
    return [str(row["cif"]) for row in rows]


def _index_teacher_plans(path: Path) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for row in iter_jsonl(path):
        source_idx = int(row["source_row_idx"])
        plan = row.get("plan_state")
        if not isinstance(plan, Mapping):
            raise ValueError("teacher DLM row lacks plan_state")
        if source_idx in result:
            raise ValueError(f"teacher DLM duplicates source_row_idx {source_idx}")
        result[source_idx] = plan
    return result


def _composition_from_fused(
    row: Mapping[str, Any], vocabulary: Mapping[str, Any]
) -> tuple[tuple[str, int], ...]:
    species_rows = {
        int(item["id"]): int(item["atomic_number"])
        for item in vocabulary["species"]
    }
    merged: dict[str, int] = {}
    for species_id, count in zip(
        row["species_ids"], row["count_targets"], strict=True
    ):
        atomic_number = species_rows[int(species_id)]
        symbol = str(Z_TO_SYMBOL[atomic_number])
        merged[symbol] = merged.get(symbol, 0) + int(count)
    return canonical_symbol_counts(list(merged), list(merged.values()))


def build_split(
    *,
    split: str,
    fused_path: Path,
    teacher_path: Path,
    mp20_path: Path,
    vocabulary: Mapping[str, Any],
    workers: int,
    output_path: Path,
) -> dict[str, Any]:
    fused = list(iter_jsonl(fused_path))
    teacher_plans = _index_teacher_plans(teacher_path)
    cifs = _mp20_cifs(mp20_path)
    source_indices: list[int] = []
    seen: set[int] = set()
    for row in fused:
        if row.get("schema") != FUSED_SCHEMA:
            raise ValueError("fused typed row schema changed")
        source_idx = int(row["source_row_idx"])
        if source_idx in seen:
            raise ValueError(f"fused rows duplicate source_row_idx {source_idx}")
        if source_idx not in teacher_plans or not 0 <= source_idx < len(cifs):
            raise ValueError(f"{split} source_row_idx {source_idx} is not aligned")
        seen.add(source_idx)
        source_indices.append(source_idx)

    payloads = [(source_idx, cifs[source_idx]) for source_idx in source_indices]
    if int(workers) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            teacher_results = list(executor.map(_teacher_target, payloads, chunksize=32))
    else:
        teacher_results = [_teacher_target(value) for value in payloads]
    order_by_idx = {source_idx: order for source_idx, order in teacher_results}
    if set(order_by_idx) != set(source_indices):
        raise RuntimeError("contact teacher did not return every fused source row")

    canonical_matches = 0
    changed_positions = 0
    arity_histogram: Counter[str] = Counter()
    root_histogram: Counter[str] = Counter()
    with output_path.open("x", encoding="utf-8", newline="\n") as output:
        for row in fused:
            source_idx = int(row["source_row_idx"])
            plan = teacher_plans[source_idx]
            canonical = canonical_symbol_counts(
                [str(value) for value in plan.get("elements") or ()],
                [int(value) for value in plan.get("counts") or ()],
            )
            if _composition_from_fused(row, vocabulary) != canonical:
                raise ValueError(f"fused/teacher composition mismatch at {split}:{source_idx}")
            elements = [symbol for symbol, _count in canonical]
            counts = [int(count) for _symbol, count in canonical]
            order = order_by_idx[source_idx]
            if set(order) != set(elements) or len(order) != len(elements):
                raise ValueError(f"contact order is not a Plan permutation at {split}:{source_idx}")
            target = [elements.index(symbol) for symbol in order]
            canonical_matches += int(target == list(range(len(elements))))
            changed_positions += sum(index != value for index, value in enumerate(target))
            arity_histogram[str(len(elements))] += 1
            root_histogram[order[0]] += 1
            output.write(
                json.dumps(
                    {
                        **row,
                        "schema": SCHEMA,
                        "source_fused_schema": FUSED_SCHEMA,
                        "canonical_atomic_numbers": [
                            int(SYMBOL_TO_Z[value])
                            for value in elements
                        ],
                        "canonical_element_counts": counts,
                        "contact_tree_order_symbols": order,
                        "contact_tree_order_indices": target,
                        "outcomes_read": False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return {
        "rows": len(fused),
        "mp20_rows": len(cifs),
        "canonical_order_rows": canonical_matches,
        "noncanonical_order_rows": len(fused) - canonical_matches,
        "changed_positions": changed_positions,
        "arity_histogram": dict(sorted(arity_histogram.items())),
        "root_histogram": dict(root_histogram.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused-data-dir", type=Path, required=True)
    parser.add_argument("--teacher-data-dir", type=Path, required=True)
    parser.add_argument("--mp20-dir", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 1 <= int(args.workers) <= 64:
        raise ValueError("workers must be in 1..64")
    vocabulary = json.loads(args.vocabulary.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    splits = {
        split: build_split(
            split=split,
            fused_path=args.fused_data_dir / f"{split}.jsonl",
            teacher_path=args.teacher_data_dir / f"{split}.jsonl",
            mp20_path=args.mp20_dir / f"{split}.csv",
            vocabulary=vocabulary,
            workers=int(args.workers),
            output_path=args.output_dir / f"{split}.jsonl",
        )
        for split in ("train", "val")
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "teacher": "MP20 periodic maximum-contact-tree species order",
        "geometry_only": True,
        "energy_hull_or_generated_outcomes_read": False,
        "composition_mutation": False,
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
