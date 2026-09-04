#!/usr/bin/env python3
"""Populate one deterministic shard of the frozen CHGNet relaxation cache."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            yield value


def load_frozen_resumable(path: Path) -> Any:
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("frozen_eval_sun_resumable", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen evaluator {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache_keys(path: Path) -> set[str]:
    return {
        str(row["structure_key"])
        for row in iter_jsonl(path)
        if row.get("structure_key")
    }


def owner_for_key(key: str, *, shard_count: int) -> int:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    text = str(key)
    if len(text) < 16:
        raise ValueError("structure key is too short")
    return int(text[:16], 16) % int(shard_count)


def validate_generation(
    rows: list[Mapping[str, Any]], *, denominator: int
) -> set[int]:
    if len(rows) != denominator:
        raise ValueError("generation denominator changed")
    if [int(row.get("ordinal", -1)) for row in rows] != list(range(denominator)):
        raise ValueError("generation ordinals changed")
    reconstructed: set[int] = set()
    for ordinal, row in enumerate(rows):
        if int(row.get("sample_idx", -1)) != ordinal:
            raise ValueError(f"sample index mismatch at ordinal {ordinal}")
        status = row.get("status")
        if status == "succeeded":
            if not isinstance(row.get("structure"), dict):
                raise ValueError(f"successful ordinal {ordinal} lacks a structure")
            reconstructed.add(ordinal)
        elif status != "failed":
            raise ValueError(f"ordinal {ordinal} has unsupported status {status!r}")
    return reconstructed


def map_structures_to_ordinals(
    structures: list[Any], manifest: Mapping[str, Any], *, denominator: int,
    expected_ordinals: set[int] | None = None,
) -> dict[int, Any]:
    if int(manifest.get("total_attempts", -1)) != denominator:
        raise ValueError("frozen input manifest attempt count changed")
    if int(manifest.get("reconstructed_structures", -1)) != len(structures):
        raise ValueError("frozen input manifest structure count changed")
    expected = set(range(denominator)) if expected_ordinals is None else set(expected_ordinals)
    mapped: dict[int, Any] = {}
    for record in manifest.get("attempt_records") or ():
        ordinal = int(record.get("generation_ordinal", -1))
        reconstructed_index = record.get("reconstructed_index")
        if reconstructed_index is None:
            if ordinal in expected:
                raise ValueError(f"frozen input did not reconstruct ordinal {ordinal}")
            continue
        index = int(reconstructed_index)
        if not 0 <= ordinal < denominator or not 0 <= index < len(structures):
            raise ValueError("frozen input manifest index is out of range")
        if ordinal not in expected:
            raise ValueError(f"frozen input unexpectedly reconstructed failed ordinal {ordinal}")
        if ordinal in mapped:
            raise ValueError("frozen input manifest repeats an ordinal")
        mapped[ordinal] = structures[index]
    if set(mapped) != expected:
        raise ValueError("frozen input manifest does not cover reconstructed ordinals")
    return mapped


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_jsonl.exists() or args.report_json.exists():
        raise FileExistsError("shard output already exists")
    rows = list(iter_jsonl(args.generation_jsonl.resolve()))
    reconstructed_ordinals = validate_generation(
        rows, denominator=int(args.denominator)
    )
    if int(args.shard_count) <= 0 or not 0 <= int(args.shard_rank) < int(args.shard_count):
        raise ValueError("invalid shard rank/count")
    frozen = load_frozen_resumable(args.frozen_resumable_py.resolve())
    structures, loader_total = frozen.load_generated_structures(args.generated_pt.resolve())
    if int(loader_total) != int(args.denominator):
        raise ValueError("frozen structure loader denominator changed")
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    structures_by_ordinal = map_structures_to_ordinals(
        list(structures), manifest, denominator=int(args.denominator),
        expected_ordinals=reconstructed_ordinals,
    )
    existing_keys = cache_keys(args.base_cache.resolve())
    selected: list[tuple[Mapping[str, Any], str, Any]] = []
    for row in rows:
        ordinal = int(row["ordinal"])
        if ordinal not in structures_by_ordinal:
            continue
        structure = structures_by_ordinal[ordinal]
        key = str(frozen.structure_cache_key(structure))
        if owner_for_key(key, shard_count=int(args.shard_count)) == int(args.shard_rank):
            selected.append((row, key, structure))

    pending: dict[str, Any] = {}
    cache_hits = 0
    for _row, key, structure in selected:
        if key in existing_keys:
            cache_hits += 1
        else:
            pending.setdefault(key, structure)

    output_rows: list[dict[str, Any]] = []
    failures = 0
    if pending:
        from chgnet.model import StructOptimizer

        optimizer = StructOptimizer(use_device=str(args.device))
        for key, structure in pending.items():
            try:
                result = optimizer.relax(structure, verbose=False)
                relaxed = result["final_structure"]
                trajectory = result.get("trajectory")
                if trajectory is None or not getattr(trajectory, "energies", None):
                    raise ValueError("CHGNet relaxation returned no energy trajectory")
                energy = float(trajectory.energies[-1]) / int(relaxed.num_sites)
                composition = relaxed.composition
            except Exception:  # noqa: BLE001
                failures += 1
                energy = None
                composition = structure.composition
            output_rows.append(
                {
                    "structure_key": key,
                    "energy_per_atom": energy,
                    "composition": composition.as_dict(),
                }
            )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("x", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    report = {
        "schema": "spad_basin_closure_relax_cache_shard_v1",
        "shard_rank": int(args.shard_rank),
        "shard_count": int(args.shard_count),
        "fixed_denominator": int(args.denominator),
        "reconstructed_occurrences": len(reconstructed_ordinals),
        "unreconstructed_failures_retained": int(args.denominator)
        - len(reconstructed_ordinals),
        "assigned_occurrences": len(selected),
        "assigned_unique_keys": len({key for _row, key, _structure in selected}),
        "base_cache_hits": cache_hits,
        "unique_missing_relaxed": len(pending),
        "new_cache_entries": len(output_rows),
        "failed_relaxations": failures,
        "device": str(args.device),
        "frozen_structure_key_function": str(args.frozen_resumable_py.resolve()),
        "selection": False,
    }
    args.report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-jsonl", type=Path, required=True)
    parser.add_argument("--generated-pt", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--frozen-resumable-py", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--denominator", type=int, default=256)
    parser.add_argument("--shard-rank", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
