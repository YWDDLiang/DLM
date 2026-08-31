#!/usr/bin/env python3
"""Finalize faithful full-schema and compact-V2 development S.U.N. cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


ATTEMPTS = 256
STREAMS = (17, 18)
STAGES = ("raw", "refined")
GROUPS = {
    "faithful": ("H0", "R0S"),
    "compact_v2": ("82017", "82018"),
}
STEM = "C3FD_LLAMA_DEVELOPMENT_SUN_FINAL"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runtime(eval_runtime: Path):
    path = eval_runtime / "finalize_official.py"
    spec = importlib.util.spec_from_file_location("development_finalize_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen official finalizer")
    sys.path.insert(0, str(eval_runtime))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cell_paths(
    *, group: str, stage: str, stream: int, arm: str, faithful: Path, v2: Path
) -> dict[str, Path]:
    if group == "faithful":
        cell = arm if stage == "refined" else f"raw_{arm}"
        root = faithful / f"stream{stream}/{cell}"
    elif group == "compact_v2":
        cell = f"policy{arm}" if stage == "refined" else f"raw_policy{arm}"
        root = v2 / f"stream{stream}/{cell}"
    else:
        raise ValueError(group)
    return {
        "labels": root / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl",
        "generation": root / "generation/generation.jsonl",
        "direct": root / "evaluation/direct/report.json",
    }


def summarize_report(
    *, group: str, stage: str, stream: int, arm: str, report: Mapping[str, Any]
) -> dict[str, Any]:
    counts = report["counts"]
    direct = report["direct"]
    if int(counts["raw_attempts"]) != ATTEMPTS:
        raise RuntimeError("development denominator changed")
    return {
        "group": group,
        "claim_scope": "development_only",
        "stage": stage,
        "stream": stream,
        "arm": arm,
        "requested": ATTEMPTS,
        "reconstructed": int(counts["reconstructed"]),
        "direct_joint": int(direct["joint_valid"]),
        "novel": int(counts["novel"]),
        "unique": int(counts["unique_representatives"]),
        "novel_unique": int(counts["novel_unique"]),
        "hull_known": int(counts["hull_known_reconstructed"]),
        "hull_unknown": int(counts["hull_unknown_reconstructed"]),
        "strict_stable": int(counts["strict_stable_all_hull_known"]),
        "strict_sun": int(counts["strict_sun"]),
        "meta_stable": int(counts["meta_stable_all_hull_known"]),
        "meta_sun": int(counts["meta_sun"]),
        "strict_sun_attempt_rate": int(counts["strict_sun"]) / ATTEMPTS,
        "meta_sun_attempt_rate": int(counts["meta_sun"]) / ATTEMPTS,
        "unknown_policy": "missing; never mapped to unstable",
    }


def aggregate(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group, arms in GROUPS.items():
        for stage in STAGES:
            for arm in arms:
                rows = [
                    row
                    for row in cells
                    if row["group"] == group
                    and row["stage"] == stage
                    and row["arm"] == arm
                ]
                if [int(row["stream"]) for row in rows] != list(STREAMS):
                    raise RuntimeError(f"missing streams for {group}:{stage}:{arm}")
                output.append(
                    {
                        "group": group,
                        "claim_scope": "development_only",
                        "stage": stage,
                        "arm": arm,
                        "stream_handling": "mean of streams17/18 requested rates",
                        "requested_per_stream": ATTEMPTS,
                        "strict_sun_mean_stream_rate": statistics.fmean(
                            float(row["strict_sun_attempt_rate"]) for row in rows
                        ),
                        "meta_sun_mean_stream_rate": statistics.fmean(
                            float(row["meta_sun_attempt_rate"]) for row in rows
                        ),
                        "strict_sun_counts_by_stream": {
                            str(row["stream"]): int(row["strict_sun"]) for row in rows
                        },
                        "meta_sun_counts_by_stream": {
                            str(row["stream"]): int(row["meta_sun"]) for row in rows
                        },
                    }
                )
    return output


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# C3FD–Llama development S.U.N. final",
        "",
        "These are development results. Faithful H0/R0S are historical-interface diagnostics; compact V2 contains MP20 train/validation compositions and is not prospective.",
        "",
        "## Stream-aggregated S.U.N.",
        "",
        "| Group | Stage | Arm | Strict S.U.N. | Meta S.U.N. | Strict counts s17/s18 | Meta counts s17/s18 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in report["aggregates"]:
        strict = row["strict_sun_counts_by_stream"]
        meta = row["meta_sun_counts_by_stream"]
        lines.append(
            f"| {row['group']} | {row['stage']} | {row['arm']} | "
            f"{100*row['strict_sun_mean_stream_rate']:.3f}% | "
            f"{100*row['meta_sun_mean_stream_rate']:.3f}% | "
            f"{strict['17']}/{strict['18']} | {meta['17']}/{meta['18']} |"
        )
    lines.extend(
        [
            "",
            "## Cell accounting",
            "",
            "| Group | Stage | Stream | Arm | Reconstructed | Direct-J | N/U/NU | Hull K/U | Strict stable/SUN | Meta stable/SUN |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["cells"]:
        lines.append(
            f"| {row['group']} | {row['stage']} | {row['stream']} | {row['arm']} | "
            f"{row['reconstructed']} | {row['direct_joint']} | "
            f"{row['novel']}/{row['unique']}/{row['novel_unique']} | "
            f"{row['hull_known']}/{row['hull_unknown']} | "
            f"{row['strict_stable']}/{row['strict_sun']} | "
            f"{row['meta_stable']}/{row['meta_sun']} |"
        )
    lines.extend(
        [
            "",
            "Unknown official hull rows remain missing and are never converted to unstable. No result is used for seed, checkpoint, or method selection.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, cells: Sequence[Mapping[str, Any]]) -> None:
    fields = list(cells[0])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--faithful-eval", type=Path, required=True)
    parser.add_argument("--v2-eval", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--eval-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    cache = args.official_run.resolve() / "official_mp_cache"
    if not (cache / "completion_SUCCESS").is_file():
        raise RuntimeError("development official query is not terminal")
    runtime = load_runtime(args.eval_runtime.resolve())
    phase_diagrams = runtime._phase_diagrams(cache / "official_slim_cache.jsonl")
    unresolved = {
        str(row["chemsys"])
        for row in __import__("protocol").read_jsonl(cache / "unresolved_chemsys.jsonl")
    }
    output.mkdir(parents=True)
    cells: list[dict[str, Any]] = []
    inputs: dict[str, str] = {
        "official_completion_manifest": sha256_file(cache / "completion_manifest.json")
    }
    for group, arms in GROUPS.items():
        for stage in STAGES:
            for stream in STREAMS:
                for arm in arms:
                    paths = cell_paths(
                        group=group,
                        stage=stage,
                        stream=stream,
                        arm=arm,
                        faithful=args.faithful_eval.resolve(),
                        v2=args.v2_eval.resolve(),
                    )
                    for name, path in paths.items():
                        inputs[f"{group}.{stage}.s{stream}.{arm}.{name}"] = sha256_file(path)
                    _rows, cell_report = runtime._evaluate_cell(
                        cell_id=f"{group}_{stage}_s{stream}_{arm}",
                        labels_path=paths["labels"],
                        generation_path=paths["generation"],
                        direct_path=paths["direct"],
                        phase_diagrams=phase_diagrams,
                        unresolved=unresolved,
                        output_dir=output / f"cells/{group}/{stage}/s{stream}/{arm}",
                    )
                    cells.append(
                        summarize_report(
                            group=group,
                            stage=stage,
                            stream=stream,
                            arm=arm,
                            report=cell_report,
                        )
                    )
    cells.sort(key=lambda row: (row["group"], row["stage"], row["stream"], row["arm"]))
    report = {
        "schema": "c3fd_llama_development_sun_final_v1",
        "status": "complete",
        "claim_scope": "development_only",
        "cells": cells,
        "aggregates": aggregate(cells),
        "exclusions": [
            "alignment_pool38881_train_only",
            "malformed_canary38420",
            "cancelled38914",
            "D3PO_already_official",
        ],
        "inputs": inputs,
        "official_cache": read_json(cache / "completion_manifest.json"),
    }
    (output / f"{STEM}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / f"{STEM}.md").write_text(render_markdown(report), encoding="utf-8")
    write_csv(output / f"{STEM}.csv", cells)
    output_files = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "OUTPUTS.sha256"
    )
    (output / "OUTPUTS.sha256").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
            for path in output_files
        ),
        encoding="utf-8",
    )
    (output / "_SUCCESS").touch(exist_ok=False)
    print(json.dumps({"status": "complete", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
