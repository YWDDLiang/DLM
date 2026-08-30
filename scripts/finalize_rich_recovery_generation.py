#!/usr/bin/env python3
"""Finalize the parent plus refinement-recovery rich canary generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ARMS = ("M0", "RCF", "R0")
STREAMS = (17, 18)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                yield json.loads(raw)


def summarize_indices(
    *,
    parsed_indices: list[int],
    refined_indices: list[int],
    denominator: int,
) -> dict[str, Any]:
    parsed = sorted(int(value) for value in parsed_indices)
    refined = sorted(int(value) for value in refined_indices)
    expected = set(range(int(denominator)))
    return {
        "parsed_count": len(parsed),
        "parsed_unique": len(set(parsed)),
        "parsed_missing": sorted(expected - set(parsed)),
        "refined_count": len(refined),
        "refined_unique": len(set(refined)),
        "refined_missing": sorted(expected - set(refined)),
        "refined_matches_parsed_exactly": refined == parsed,
    }


def cell_paths(parent: Path, recovery: Path, stream: int, arm: str) -> tuple[Path, Path]:
    body = parent / f"stream{stream}" / arm / "body"
    refine = (
        parent / "stream17" / "M0" / "refine"
        if (stream, arm) == (17, "M0")
        else recovery / f"stream{stream}" / arm
    )
    return body, refine


def inspect_cell(parent: Path, recovery: Path, stream: int, arm: str) -> dict[str, Any]:
    import torch

    body, refine = cell_paths(parent, recovery, stream, arm)
    manifest_path = body / "SGTC_BODY_MANIFEST.json"
    attempts_path = body / "raw_generations.jsonl"
    graphs_path = body / "proposal_graphs.pt"
    metrics_path = refine / "refinement_metrics.json"
    for path in (manifest_path, attempts_path, graphs_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    output_file = Path(metrics["output_file"])
    if not output_file.is_file():
        raise FileNotFoundError(output_file)
    payload = torch.load(output_file, map_location="cpu")
    sample_indices = [int(value) for value in payload["sample_indices"].tolist()]
    attempts = list(iter_jsonl(attempts_path))
    parsed_indices = [int(row["sample_idx"]) for row in attempts if row.get("parsed") is True]
    indices = summarize_indices(
        parsed_indices=parsed_indices,
        refined_indices=sample_indices,
        denominator=256,
    )
    row = {
        "stream": int(stream),
        "arm": arm,
        "body_dir": str(body.resolve()),
        "refine_dir": str(refine.resolve()),
        "requested": int(manifest["requested"]),
        "parsed": int(manifest["parsed"]),
        "graphs": int(manifest["graphs"]),
        "num_proposals": int(metrics["num_proposals"]),
        "assigned_proposals": int(metrics["assigned_proposals"]),
        "diff_steps": int(metrics["diff_steps"]),
        "refined_output": str(output_file.resolve()),
        "body_manifest_sha256": sha256_file(manifest_path),
        "body_attempts_sha256": sha256_file(attempts_path),
        "proposal_graphs_sha256": sha256_file(graphs_path),
        "refinement_metrics_sha256": sha256_file(metrics_path),
        "refined_output_sha256": sha256_file(output_file),
        **indices,
    }
    row["gate"] = {
        "requested_256": row["requested"] == 256,
        "parsed_equals_graphs": row["parsed"] == row["graphs"],
        "graphs_equal_num_proposals": row["graphs"] == row["num_proposals"],
        "assigned_equal_num_proposals": row["assigned_proposals"]
        == row["num_proposals"],
        "sample_indices_match_parsed": row["refined_matches_parsed_exactly"],
        "diff_steps_800": row["diff_steps"] == 800,
    }
    row["gate"]["pass"] = all(row["gate"].values())
    return row


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Rich recovery canary generation final",
        "",
        "The fixed denominator is 256 attempts per cell. Missing body/refined",
        "indices remain explicit and are not replaced.",
        "",
        "| stream | arm | requested | body parsed | graphs | refined | missing | pass |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        lines.append(
            f"| {cell['stream']} | {cell['arm']} | {cell['requested']} | "
            f"{cell['parsed']} | {cell['graphs']} | {cell['refined_count']} | "
            f"{len(cell['refined_missing'])} | {cell['gate']['pass']} |"
        )
    resource = report["resources"]
    lines.extend(
        [
            "",
            "## Engineering lineage",
            "",
            "- Parent job 38363 completed all six body cells and stream17/M0",
            "  refinement, then failed because its wrapper expected a refiner",
            "  `_SUCCESS` marker that the refiner never emits.",
            "- Recovery job 38406 reused the five missing immutable body graph",
            "  files with unchanged seeds, model494 and tau800; no body was rerun.",
            "",
            "## Resources",
            "",
            f"- Parent observed: {resource['parent_gpu_hours']:.4f} A800-hours.",
            f"- Recovery observed: {resource['recovery_gpu_hours']:.4f} A800-hours.",
            f"- Combined observed: {resource['combined_gpu_hours']:.4f} A800-hours.",
            f"- Combined scheduler kill ceiling: {resource['combined_kill_ceiling_gpu_hours']:.1f} A800-hours.",
            "",
            f"Overall pass: **{report['gate']['pass']}**.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--recovery-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-elapsed-seconds", type=int, default=3659)
    parser.add_argument("--recovery-elapsed-seconds", type=int, default=2183)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not (args.parent_run / "_FAILED").is_file():
        raise RuntimeError("parent failure marker is absent")
    if not (args.recovery_run / "_SUCCESS").is_file():
        raise RuntimeError("recovery success marker is absent")
    cells = [
        inspect_cell(args.parent_run, args.recovery_run, stream, arm)
        for stream in STREAMS
        for arm in ARMS
    ]
    resources = {
        "parent_elapsed_seconds": int(args.parent_elapsed_seconds),
        "parent_gpus": 6,
        "parent_gpu_hours": 6 * int(args.parent_elapsed_seconds) / 3600,
        "parent_kill_ceiling_gpu_hours": 6 * 6,
        "recovery_elapsed_seconds": int(args.recovery_elapsed_seconds),
        "recovery_gpus": 5,
        "recovery_gpu_hours": 5 * int(args.recovery_elapsed_seconds) / 3600,
        "recovery_kill_ceiling_gpu_hours": 5 * 4,
    }
    resources["combined_gpu_hours"] = (
        resources["parent_gpu_hours"] + resources["recovery_gpu_hours"]
    )
    resources["combined_kill_ceiling_gpu_hours"] = (
        resources["parent_kill_ceiling_gpu_hours"]
        + resources["recovery_kill_ceiling_gpu_hours"]
    )
    report = {
        "schema": "h1a2_rich_recovery_generation_final_v1",
        "parent_run": str(args.parent_run.resolve()),
        "recovery_run": str(args.recovery_run.resolve()),
        "cells": cells,
        "resources": resources,
        "body_rerun_in_recovery": False,
        "retry_replacement_rerank": False,
        "outcomes_evaluated": False,
        "gate": {
            "six_cells": len(cells) == 6,
            "all_cells_pass": all(cell["gate"]["pass"] for cell in cells),
            "parent_failure_preserved": True,
            "recovery_success_preserved": True,
        },
    }
    report["gate"]["pass"] = all(report["gate"].values())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    json_path = args.output_dir / "RICH_RECOVERY_GENERATION_FINAL.json"
    md_path = args.output_dir / "RICH_RECOVERY_GENERATION_FINAL.md"
    csv_path = args.output_dir / "RICH_RECOVERY_GENERATION_FINAL.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "stream",
                "arm",
                "requested",
                "parsed",
                "graphs",
                "num_proposals",
                "assigned_proposals",
                "refined_count",
                "refined_missing",
                "pass",
                "body_dir",
                "refine_dir",
            ),
        )
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    **{key: cell[key] for key in writer.fieldnames if key in cell},
                    "refined_missing": json.dumps(cell["refined_missing"]),
                    "pass": cell["gate"]["pass"],
                }
            )
    outputs = {path.name: sha256_file(path) for path in (json_path, md_path, csv_path)}
    (args.output_dir / "OUTPUTS.sha256.json").write_text(
        json.dumps(outputs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    marker = args.output_dir / ("_SUCCESS" if report["gate"]["pass"] else "_FAILED")
    marker.write_text(sha256_file(json_path) + "\n", encoding="utf-8")
    if not report["gate"]["pass"]:
        raise SystemExit(3)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
