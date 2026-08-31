#!/usr/bin/env python3
"""Finalize the paired F/M prospective DLM and official S.U.N. experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Mapping, Sequence


ATTEMPTS = 256
STREAMS = (17, 18)
ROUTES = ("F", "M")
STAGES = ("raw", "refined")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260831
STEM = "C3FD_LLAMA_PROSPECTIVE_SUN_FINAL"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runtime(eval_runtime: Path):
    spec = importlib.util.spec_from_file_location(
        "prospective_finalize_runtime", eval_runtime / "finalize_official.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen official finalizer")
    sys.path.insert(0, str(eval_runtime))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap(values: Mapping[int, float], label: str) -> dict[str, Any]:
    ordered = [float(values[index]) for index in sorted(values)]
    if not ordered:
        return {"known": 0, "mean": None, "ci95": [None, None]}
    seed = BOOTSTRAP_SEED ^ int.from_bytes(
        hashlib.sha256(label.encode()).digest()[:8], "big"
    )
    rng = random.Random(seed)
    count = len(ordered)
    means = [
        statistics.fmean(ordered[rng.randrange(count)] for _ in range(count))
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    return {
        "known": count,
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "fraction_lower": sum(value < 0 for value in ordered) / count,
        "ci95": [quantile(means, 0.025), quantile(means, 0.975)],
        "bootstrap_seed": seed,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def indexed(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    result = {int(row["ordinal"]): row for row in rows}
    if set(result) != set(range(ATTEMPTS)):
        raise ValueError("official rows do not cover fixed256")
    return result


def paired_stream_delta(
    f_rows: Sequence[Mapping[str, Any]],
    m_rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    require_hull_known: bool,
) -> dict[int, float]:
    left = indexed(f_rows)
    right = indexed(m_rows)
    result: dict[int, float] = {}
    for index in range(ATTEMPTS):
        f = left[index]
        m = right[index]
        if f.get("chemsys") != m.get("chemsys"):
            raise ValueError(f"paired chemsys changed at {index}")
        if require_hull_known and (
            f.get("official_hull_status") != "known"
            or m.get("official_hull_status") != "known"
        ):
            continue
        if f.get(field) is None or m.get(field) is None:
            continue
        value = float(m[field]) - float(f[field])
        if not math.isfinite(value):
            raise ValueError("paired delta is nonfinite")
        result[index] = value
    return result


def average_streams(values: Sequence[Mapping[int, float]]) -> dict[int, float]:
    common = set(values[0])
    for item in values[1:]:
        common &= set(item)
    return {
        index: statistics.fmean(float(item[index]) for item in values)
        for index in sorted(common)
    }


def summarize_cell(group: str, stage: str, stream: int, route: str, report):
    count = report["counts"]
    direct = report["direct"]
    return {
        "stage": stage,
        "stream": stream,
        "route": route,
        "requested": ATTEMPTS,
        "reconstructed": int(count["reconstructed"]),
        "direct_joint": int(direct["joint_valid"]),
        "novel": int(count["novel"]),
        "unique": int(count["unique_representatives"]),
        "novel_unique": int(count["novel_unique"]),
        "hull_known": int(count["hull_known_reconstructed"]),
        "hull_unknown": int(count["hull_unknown_reconstructed"]),
        "strict_stable": int(count["strict_stable_all_hull_known"]),
        "strict_sun": int(count["strict_sun"]),
        "meta_stable": int(count["meta_stable_all_hull_known"]),
        "meta_sun": int(count["meta_sun"]),
        "strict_sun_rate": int(count["strict_sun"]) / ATTEMPTS,
        "meta_sun_rate": int(count["meta_sun"]) / ATTEMPTS,
        "claim_scope": group,
    }


def aggregate_cells(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for stage in STAGES:
        for route in ROUTES:
            rows = [
                row
                for row in cells
                if row["stage"] == stage and row["route"] == route
            ]
            rows.sort(key=lambda row: int(row["stream"]))
            if [row["stream"] for row in rows] != list(STREAMS):
                raise ValueError("missing stream cell")
            output.append(
                {
                    "stage": stage,
                    "route": route,
                    "strict_sun_mean_stream_rate": statistics.fmean(
                        row["strict_sun_rate"] for row in rows
                    ),
                    "meta_sun_mean_stream_rate": statistics.fmean(
                        row["meta_sun_rate"] for row in rows
                    ),
                    "strict_counts": {
                        str(row["stream"]): row["strict_sun"] for row in rows
                    },
                    "meta_counts": {
                        str(row["stream"]): row["meta_sun"] for row in rows
                    },
                }
            )
    return output


def render(report: Mapping[str, Any]) -> str:
    lines = [
        "# C3FD–Llama prospective DLM S.U.N. final",
        "",
        "Raw DLM realization is the primary mechanism endpoint; refined and official S.U.N. are system endpoints.",
        "",
        "## S.U.N.",
        "",
        "| Stage | Route | Strict | Meta | Strict s17/s18 | Meta s17/s18 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["aggregates"]:
        lines.append(
            f"| {row['stage']} | {row['route']} | {100*row['strict_sun_mean_stream_rate']:.3f}% | "
            f"{100*row['meta_sun_mean_stream_rate']:.3f}% | "
            f"{row['strict_counts']['17']}/{row['strict_counts']['18']} | "
            f"{row['meta_counts']['17']}/{row['meta_counts']['18']} |"
        )
    lines.extend(["", "## Paired M−F continuous effects", ""])
    for name, value in report["paired_effects"].items():
        lines.append(
            f"- {name}: n={value['known']}, mean={value['mean']}, 95% CI={value['ci95']} (negative favors M)."
        )
    lines.extend(
        [
            "",
            "Unknown official hull rows remain missing. No seed, checkpoint, composition, or failed attempt was selected or replaced.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--eval-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    cache = args.official_run.resolve() / "official_mp_cache"
    if not (cache / "completion_SUCCESS").is_file():
        raise RuntimeError("prospective official query is not complete")
    runtime = load_runtime(args.eval_runtime.resolve())
    protocol = __import__("protocol")
    phase_diagrams = runtime._phase_diagrams(cache / "official_slim_cache.jsonl")
    unresolved = {
        str(row["chemsys"])
        for row in protocol.read_jsonl(cache / "unresolved_chemsys.jsonl")
    }
    output.mkdir(parents=True)
    cells = []
    rows_by_key = {}
    input_hashes = {"official_manifest": sha256_file(cache / "completion_manifest.json")}
    for stage in STAGES:
        for stream in STREAMS:
            for route in ROUTES:
                arm = route if stage == "refined" else f"raw_{route}"
                root = args.eval_run.resolve() / f"stream{stream}/{arm}"
                paths = {
                    "labels": root / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl",
                    "generation": root / "generation/generation.jsonl",
                    "direct": root / "evaluation/direct/report.json",
                }
                for name, path in paths.items():
                    input_hashes[f"{stage}.s{stream}.{route}.{name}"] = sha256_file(path)
                rows, cell_report = runtime._evaluate_cell(
                    cell_id=f"{stage}_s{stream}_{route}",
                    labels_path=paths["labels"],
                    generation_path=paths["generation"],
                    direct_path=paths["direct"],
                    phase_diagrams=phase_diagrams,
                    unresolved=unresolved,
                    output_dir=output / f"cells/{stage}/s{stream}/{route}",
                )
                rows_by_key[(stage, stream, route)] = rows
                cells.append(
                    summarize_cell("prospective", stage, stream, route, cell_report)
                )
    paired = {}
    for stage, field, hull in (
        ("raw", "chgnet_energy_per_atom", False),
        ("refined", "chgnet_energy_per_atom", False),
        ("refined", "official_e_above_hull", True),
    ):
        stream_values = [
            paired_stream_delta(
                rows_by_key[(stage, stream, "F")],
                rows_by_key[(stage, stream, "M")],
                field=field,
                require_hull_known=hull,
            )
            for stream in STREAMS
        ]
        name = f"{stage}_{field}"
        paired[name] = bootstrap(average_streams(stream_values), name)
    report = {
        "schema": "c3fd_llama_prospective_sun_final_v1",
        "status": "complete",
        "independent_unit": "C3FD composition",
        "stream_handling": "average streams17/18 within composition before bootstrap",
        "cells": sorted(cells, key=lambda row: (row["stage"], row["stream"], row["route"])),
        "aggregates": aggregate_cells(cells),
        "paired_effects": paired,
        "targets": {"strict": 0.10, "meta": 0.50, "not_result_deletion_gates": True},
        "selection_retry_replacement_rerank": False,
        "inputs": input_hashes,
        "official_cache": read_json(cache / "completion_manifest.json"),
    }
    (output / f"{STEM}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / f"{STEM}.md").write_text(render(report), encoding="utf-8")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "OUTPUTS.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    (output / "_SUCCESS").touch()
    print(json.dumps({"status": "complete", "output": str(output)}))


if __name__ == "__main__":
    main()
