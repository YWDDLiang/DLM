#!/usr/bin/env python3
"""Finalize the tau800-only Plan1200 scale profile."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import statistics
import sys


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--eval-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(args.eval_runtime.resolve()))
    runtime = importlib.import_module("finalize_official")
    phase_diagrams = runtime._phase_diagrams(
        args.official_run.resolve() / "official_mp_cache/official_slim_cache.jsonl"
    )
    labels = read_jsonl(args.labels.resolve())
    generation = read_jsonl(args.generation.resolve())
    if len(labels) != 1000 or len(generation) != 1000:
        raise ValueError("tau800 main denominator must be 1000")

    hull = []
    strict = meta = 0
    for row in labels:
        value = runtime._e_above_hull(
            phase_diagrams[str(row["chemsys"])],
            row["chgnet_composition"],
            float(row["chgnet_energy_per_atom"]),
        )
        novel_unique = bool(row["novel"]) and bool(row["unique_representative"])
        hull.append(value)
        strict += int(novel_unique and value <= 0.0)
        meta += int(novel_unique and value <= 0.1)

    report = {
        "schema": "plan1200_tau800_main1000_final_v1",
        "status": "complete",
        "denominator": 1000,
        "partition": "first861 prior-main valid CIFs plus all139 remainder valid CIFs",
        "novel": sum(bool(row["novel"]) for row in labels),
        "unique": sum(bool(row["unique_representative"]) for row in labels),
        "novel_unique": sum(
            bool(row["novel"]) and bool(row["unique_representative"])
            for row in labels
        ),
        "strict_sun": strict,
        "meta_sun": meta,
        "hull_mean_eV_per_atom": statistics.fmean(hull),
        "hull_median_eV_per_atom": statistics.median(hull),
        "terminal_diffusion": "tau800",
    }
    output.mkdir(parents=True)
    (output / "PLAN1200_TAU800_FINAL.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
