#!/usr/bin/env python3
"""Finalize the immutable six-cell D3PO generation/refinement stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STREAMS = (17, 18)
ARMS = ("base", "d3po_seed81017", "d3po_seed81018")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(run: Path, elapsed_seconds: int) -> dict[str, Any]:
    root = run.resolve()
    for name in ("_SUCCESS", "OUTPUTS.sha256", "inputs.sha256", "scientific_contract.tsv"):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    if (root / "D3PO_GENERATION_FINAL.json").exists():
        raise FileExistsError(root / "D3PO_GENERATION_FINAL.json")
    cells = []
    for stream in STREAMS:
        for arm in ARMS:
            cell = root / f"stream{stream}/{arm}"
            for path in (
                cell / "_SUCCESS",
                cell / "body/SGTC_BODY_MANIFEST.json",
                cell / "refine/refinement_metrics.json",
            ):
                if not path.is_file():
                    raise FileNotFoundError(path)
            body = read_json(cell / "body/SGTC_BODY_MANIFEST.json")
            refine = read_json(cell / "refine/refinement_metrics.json")
            cells.append(
                {
                    "stream": stream,
                    "arm": arm,
                    "requested": 256,
                    "parsed": int(body["parsed"]),
                    "graphs": int(body["graphs"]),
                    "body_failures": body.get("failures") or {},
                    "refined": int(refine["num_proposals"]),
                    "attempts_sha256": str(body["attempts_sha256"]),
                    "body_manifest_sha256": sha(cell / "body/SGTC_BODY_MANIFEST.json"),
                    "refinement_manifest_sha256": sha(cell / "refine/refinement_metrics.json"),
                }
            )
    elapsed = int(elapsed_seconds)
    report = {
        "schema": "h1a2_d3po_fixed256_generation_final_v1",
        "status": "success",
        "scientific_stability_result_available": False,
        "cells": cells,
        "totals": {
            "requested": 256 * len(cells),
            "parsed": sum(row["parsed"] for row in cells),
            "graphs": sum(row["graphs"] for row in cells),
            "refined": sum(row["refined"] for row in cells),
        },
        "resources": {
            "gpus": 6,
            "cpus": 48,
            "expected_gpu_hours": 12.0,
            "scheduler_kill_ceiling_gpu_hours": 72.0,
            "elapsed_seconds": elapsed,
            "observed_gpu_hours": 6 * elapsed / 3600.0,
        },
        "hashes": {
            "scientific_contract": sha(root / "scientific_contract.tsv"),
            "inputs": sha(root / "inputs.sha256"),
            "outputs": sha(root / "OUTPUTS.sha256"),
        },
    }
    (root / "D3PO_GENERATION_FINAL.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# D3PO fixed256 generation/refinement terminal",
        "",
        "Status: **engineering success; stability evaluation pending**.",
        "",
        "| Stream | Arm | Requested | Parsed/graphs | Refined | Failures |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in cells:
        lines.append(
            f"| {row['stream']} | {row['arm']} | 256 | {row['parsed']}/{row['graphs']} | "
            f"{row['refined']} | `{row['body_failures']}` |"
        )
    lines.extend(
        [
            "",
            f"Observed GPU-hours: `{report['resources']['observed_gpu_hours']:.4f}`; "
            "scheduler kill ceiling: `72` GPU-hours.",
            "No failed ordinal was replaced or removed from the denominator.",
        ]
    )
    (root / "D3PO_GENERATION_FINAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = {
        name: sha(root / name)
        for name in ("D3PO_GENERATION_FINAL.json", "D3PO_GENERATION_FINAL.md")
    }
    (root / "D3PO_GENERATION_FINAL_OUTPUTS.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(outputs.items())),
        encoding="utf-8",
    )
    (root / "_GENERATION_FINAL_SUCCESS").touch(exist_ok=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--elapsed-seconds", type=int, required=True)
    args = parser.parse_args()
    report = finalize(args.run, args.elapsed_seconds)
    print(json.dumps({"status": report["status"], "totals": report["totals"]}))


if __name__ == "__main__":
    main()
