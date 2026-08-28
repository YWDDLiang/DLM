#!/usr/bin/env python3
"""Evaluate critic feasibility structures with exact MatterSim-v1 5M single points."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable


MODEL_ID = "MatterSim-v1.0.0-5M.pth"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def predict(args: argparse.Namespace) -> None:
    from ase.io import iread
    from mattersim.forcefield import MatterSimCalculator

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rank = int(args.rank)
    world_size = int(args.world_size)
    if not (0 <= rank < world_size):
        raise ValueError("invalid rank/world-size")
    final = output / f"predictions.rank{rank}.jsonl"
    if final.exists():
        raise FileExistsError(final)
    preparing = output / f".predictions.rank{rank}.preparing"
    if preparing.exists():
        raise FileExistsError(preparing)
    metadata = read_jsonl(args.metadata)
    calculator = MatterSimCalculator(load_path=str(args.model_path), device=str(args.device))
    expected = [index for index in range(len(metadata)) if index % world_size == rank]
    seen: list[int] = []
    with preparing.open("x", encoding="utf-8", newline="\n") as handle:
        for atoms in iread(args.input_extxyz, index=":"):
            record_index = int(atoms.info["record_index"])
            if record_index % world_size != rank:
                continue
            if record_index >= len(metadata) or int(metadata[record_index]["record_index"]) != record_index:
                raise RuntimeError("extxyz and metadata identity changed")
            row: dict[str, Any] = {
                "schema": "h1a2_mattersim_singlepoint_v1",
                "record_index": record_index,
                "rank": rank,
                "world_size": world_size,
                "model_id": MODEL_ID,
                "known": False,
                "energy_total_eV": None,
                "energy_per_atom_eV": None,
                "error": None,
            }
            try:
                atoms.calc = calculator
                total = float(atoms.get_potential_energy())
                per_atom = total / len(atoms)
                if not math.isfinite(total) or not math.isfinite(per_atom):
                    raise ValueError("nonfinite MatterSim energy")
                row.update(
                    {
                        "known": True,
                        "energy_total_eV": total,
                        "energy_per_atom_eV": per_atom,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}:{exc}"
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            seen.append(record_index)
    if seen != expected:
        raise RuntimeError(f"rank{rank} coverage changed")
    preparing.rename(final)
    report = {
        "rank": rank,
        "world_size": world_size,
        "rows": len(seen),
        "known": sum(row.get("known") is True for row in read_jsonl(final)),
        "model_id": MODEL_ID,
        "model_path": str(args.model_path.resolve()),
        "device": str(args.device),
        "mattersim_version": importlib.metadata.version("mattersim"),
        "python": sys.version,
        "platform": platform.platform(),
        "batch_size": 1,
        "batch_bug_avoidance": "sequential ASE calculator; no batched graph collation",
    }
    (output / f"rank{rank}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / f"rank{rank}_SUCCESS").touch()


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    metadata = read_jsonl(args.metadata)
    rows: list[dict[str, Any]] = []
    rank_reports: list[dict[str, Any]] = []
    for rank in range(int(args.world_size)):
        if not (output / f"rank{rank}_SUCCESS").is_file():
            raise RuntimeError(f"rank{rank} incomplete")
        rows.extend(read_jsonl(output / f"predictions.rank{rank}.jsonl"))
        rank_reports.append(json.loads((output / f"rank{rank}.json").read_text(encoding="utf-8")))
    rows.sort(key=lambda row: int(row["record_index"]))
    if len(rows) != len(metadata) or [int(row["record_index"]) for row in rows] != list(range(len(metadata))):
        raise RuntimeError("merged MatterSim coverage changed")
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema": "h1a2_mattersim_feasibility_predictions_v1",
        "rows": len(rows),
        "known": sum(row.get("known") is True for row in rows),
        "unknown": sum(row.get("known") is not True for row in rows),
        "model_id": MODEL_ID,
        "batch_size": 1,
        "world_size": int(args.world_size),
        "rank_reports": rank_reports,
        "chgnet_or_official_used_at_inference": False,
        "final_rerank": False,
    }
    (output / "MATTERSIM_FEASIBILITY_PREDICTIONS.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "MATTERSIM_FEASIBILITY_PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("record_index", "known", "energy_total_eV", "energy_per_atom_eV", "error"),
        )
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in rows)
    (output / "_SUCCESS").touch()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-extxyz", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge(args)
    else:
        if args.model_path is None:
            parser.error("--model-path is required for prediction")
        predict(args)


if __name__ == "__main__":
    main()
