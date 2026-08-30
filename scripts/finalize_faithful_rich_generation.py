#!/usr/bin/env python3
"""Finalize the four-cell faithful H0/R0S generation diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


VIEWS = ("H0", "R0S")
STREAMS = (17, 18)
DENOMINATOR = 256


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise TypeError(path)
                yield value


def summarize_indices(
    *, parsed_indices: list[int], refined_indices: list[int]
) -> dict[str, Any]:
    parsed = sorted(int(value) for value in parsed_indices)
    refined = sorted(int(value) for value in refined_indices)
    expected = set(range(DENOMINATOR))
    return {
        "parsed_count": len(parsed),
        "parsed_unique": len(set(parsed)),
        "parsed_missing": sorted(expected - set(parsed)),
        "refined_count": len(refined),
        "refined_unique": len(set(refined)),
        "refined_missing": sorted(expected - set(refined)),
        "refined_matches_parsed_exactly": refined == parsed,
    }


def inspect_cell(run: Path, stream: int, view: str) -> dict[str, Any]:
    import torch

    root = run / f"stream{stream}" / view
    body = root / "body"
    refine = root / "refine"
    manifest_path = body / "SGTC_BODY_MANIFEST.json"
    attempts_path = body / "raw_generations.jsonl"
    graphs_path = body / "proposal_graphs.pt"
    metrics_path = refine / "refinement_metrics.json"
    for path in (manifest_path, attempts_path, graphs_path, metrics_path, root / "_SUCCESS"):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    output_file = Path(metrics["output_file"])
    if not output_file.is_file():
        raise FileNotFoundError(output_file)
    payload = torch.load(output_file, map_location="cpu")
    refined_indices = [int(value) for value in payload["sample_indices"].tolist()]
    attempts = list(iter_jsonl(attempts_path))
    parsed_indices = [int(row["sample_idx"]) for row in attempts if row.get("parsed") is True]
    indices = summarize_indices(
        parsed_indices=parsed_indices,
        refined_indices=refined_indices,
    )
    row = {
        "stream": int(stream),
        "view": view,
        "requested": int(manifest["requested"]),
        "parsed": int(manifest["parsed"]),
        "graphs": int(manifest["graphs"]),
        "num_proposals": int(metrics["num_proposals"]),
        "assigned_proposals": int(metrics["assigned_proposals"]),
        "diff_steps": int(metrics["diff_steps"]),
        "body_dir": str(body.resolve()),
        "refine_dir": str(refine.resolve()),
        "refined_output": str(output_file.resolve()),
        "body_manifest_sha256": sha256_file(manifest_path),
        "body_attempts_sha256": sha256_file(attempts_path),
        "proposal_graphs_sha256": sha256_file(graphs_path),
        "refinement_metrics_sha256": sha256_file(metrics_path),
        "refined_output_sha256": sha256_file(output_file),
        **indices,
    }
    row["gate"] = {
        "requested_256": row["requested"] == DENOMINATOR,
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
        "# Faithful rich-interface generation final",
        "",
        "Development-only H0/R0S diagnostic; no official or stability outcome is evaluated.",
        "The requested denominator remains 256 and every missing sample index is retained.",
        "",
        "| stream | view | requested | parsed | graphs | refined | missing | pass |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        lines.append(
            f"| {cell['stream']} | {cell['view']} | {cell['requested']} | "
            f"{cell['parsed']} | {cell['graphs']} | {cell['refined_count']} | "
            f"{len(cell['refined_missing'])} | {cell['gate']['pass']} |"
        )
    resource = report["resources"]
    lines.extend(
        [
            "",
            "## Resources",
            "",
            f"- observed: {resource['observed_gpu_hours']:.4f} A800-hours;",
            f"- scheduler kill ceiling: {resource['kill_ceiling_gpu_hours']:.1f} A800-hours;",
            f"- elapsed: {resource['elapsed_seconds']} seconds on {resource['gpus']} A800s.",
            "",
            "## Boundaries",
            "",
            "- H0 checks current-runtime compatibility with historical first256 Plans.",
            "- R0S changes only deterministic prompt schema fields on seed19 C3FD Plans.",
            "- Generation success alone does not establish raw validity, energy, hull, or S.U.N.",
            "",
            f"Overall pass: **{report['gate']['pass']}**.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--elapsed-seconds", type=int, default=3575)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not (args.run / "_SUCCESS").is_file():
        raise RuntimeError("generation run success marker is absent")
    cells = [
        inspect_cell(args.run, stream, view)
        for stream in STREAMS
        for view in VIEWS
    ]
    resources = {
        "elapsed_seconds": int(args.elapsed_seconds),
        "gpus": 4,
        "observed_gpu_hours": 4 * int(args.elapsed_seconds) / 3600,
        "kill_ceiling_gpu_hours": 4 * 5,
    }
    report = {
        "schema": "h1a2_faithful_rich_generation_final_v1",
        "run": str(args.run.resolve()),
        "cells": cells,
        "resources": resources,
        "body_rerun": False,
        "retry_replacement_rerank": False,
        "outcomes_evaluated": False,
        "official_query": False,
        "gate": {
            "four_cells": len(cells) == 4,
            "all_cells_pass": all(cell["gate"]["pass"] for cell in cells),
        },
    }
    report["gate"]["pass"] = all(report["gate"].values())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stem = "FAITHFUL_RICH_GENERATION_FINAL"
    json_path = args.output_dir / f"{stem}.json"
    md_path = args.output_dir / f"{stem}.md"
    csv_path = args.output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "stream",
            "view",
            "requested",
            "parsed",
            "graphs",
            "num_proposals",
            "assigned_proposals",
            "refined_count",
            "refined_missing",
            "pass",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    **{key: cell[key] for key in fieldnames if key in cell},
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
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate"]["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

