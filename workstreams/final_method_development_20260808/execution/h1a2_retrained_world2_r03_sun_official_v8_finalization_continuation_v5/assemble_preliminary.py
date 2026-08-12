#!/usr/bin/env python3
"""Assemble nine post-refine preliminary cells and expose official-cache inputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from protocol import (
    ContractError,
    canonical_json,
    cell_specs,
    identity,
    read_json,
    require_source_manifest,
    write_json_exclusive,
)


def rate(count: int, denominator: int) -> str:
    return f"{count}/{denominator} ({100.0 * count / denominator:.2f}%)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(source / "CONFIG.json")
    run_root = args.run_root.resolve()
    reports = []
    for cell in cell_specs(config):
        index = int(cell["cell_index"])
        exit_code = run_root / f"status/preliminary_cell_{index}_exit_code.txt"
        marker = run_root / f"status/preliminary_cell_{index}_SUCCESS"
        failed = run_root / f"status/preliminary_cell_{index}_FAILED"
        root = run_root / f"preliminary/{index:03d}_{cell['cell_id']}"
        if (
            not exit_code.is_file()
            or exit_code.read_text(encoding="ascii").strip() != "0"
            or not marker.is_file()
            or failed.exists()
            or not (root / "_SUCCESS").is_file()
        ):
            raise ContractError(f"preliminary cell failed: {cell['cell_id']}")
        report = read_json(root / "preliminary_report.json")
        if (
            report.get("ok") is not True
            or report.get("cell_id") != cell["cell_id"]
            or report.get("stage") != "post_model494"
            or report.get("pre_refine_evaluated") is not False
        ):
            raise ContractError(f"preliminary report changed: {cell['cell_id']}")
        reports.append(report)
    inputs = read_json(run_root / "inputs/input_manifest.json")
    if (
        int(inputs.get("cell_count", -1)) != 9
        or inputs.get("evaluated_stage") != "post_model494_only"
        or inputs.get("pre_refine_evaluated") is not False
    ):
        raise ContractError("official post-only input cell contract changed")
    terminal = {
        "schema": "h1a2_retrained_postonly_preliminary_terminal_v1",
        "engineering_status": "complete",
        "scientific_status": "official_stability_pending",
        "source_manifest_sha256": args.source_manifest_sha256,
        "evaluated_stage": "post_model494_only",
        "pre_refine_evaluated": False,
        "cells": reports,
        "wanted_chemsys_count": int(inputs["wanted_chemsys_count"]),
        "artifacts": {
            "input_manifest": identity(run_root / "inputs/input_manifest.json"),
            "wanted_chemsys": identity(run_root / "inputs/wanted_chemsys.jsonl"),
        },
        "preliminary_hull_selection_role": "diagnostic_only_never_headline",
    }
    write_json_exclusive(run_root / "preliminary_terminal_report.json", terminal)
    lines = [
        "# Retrained H1-A2 recovery — post-refine preliminary U/N + CHGNet audit",
        "",
        "Only `post_model494` structures are evaluated. The strict/meta columns use the legacy cache only as a diagnostic; they are not the official result. Reconstruction, U/N and CHGNet energies are frozen for the official-MP pass.",
        "",
        "| Cell | Panel | Body | Reconstructed | Novel | Unique | N+U | Legacy strict (diagnostic) | Legacy meta (diagnostic) | Hull unknown |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        counts = report["counts"]
        attempts = int(report["expected_attempts"])
        lines.append(
            f"| {report['cell_id']} | {report['panel']} | {report['body']} | "
            f"{counts['reconstructed']} | {counts['novel']} | {counts['unique']} | "
            f"{counts['novel_unique']} | {rate(counts['strict_full_sun'], attempts)} | "
            f"{rate(counts['meta_full_sun'], attempts)} | "
            f"{counts['relaxation_or_hull_unknown']} |"
        )
    path = run_root / "PRELIMINARY_RESULTS.md"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    (run_root / "status/preliminary_assembly_SUCCESS").touch(exist_ok=False)
    print(
        canonical_json(
            {
                "preliminary_assembly": "complete",
                "cells": len(reports),
                "wanted_chemsys": inputs["wanted_chemsys_count"],
                "evaluated_stage": "post_model494_only",
            }
        )
    )


if __name__ == "__main__":
    main()
