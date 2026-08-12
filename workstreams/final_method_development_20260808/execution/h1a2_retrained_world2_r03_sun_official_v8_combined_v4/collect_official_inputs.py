#!/usr/bin/env python3
"""Freeze post-refine U/N and CHGNet energies for nine official-S cells."""

from __future__ import annotations

import argparse
import math
import os
import shutil
from pathlib import Path
from typing import Any

from pymatgen.core import Composition

from protocol import (
    ContractError,
    DENOMINATOR,
    canonical_sha256,
    identity,
    load_upstream_cells,
    normalized_chemsys,
    read_json,
    read_jsonl,
    require_source_manifest,
    write_json_exclusive,
    write_jsonl_exclusive,
)


STRICT_THRESHOLD = 0.0
META_THRESHOLD = 0.1


def same_optional_float(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def audit_cell(
    run_root: Path, cell: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index = int(cell["cell_index"])
    root = run_root / f"preliminary/{index:03d}_{cell['cell_id']}"
    required = {
        "attempt_results": root / "attempt_results.jsonl",
        "attempt_summary": root / "attempt_summary.json",
        "input_manifest": root / "input_manifest.json",
        "strict_relax": root / "exact_strict/relax_results.jsonl",
        "meta_relax": root / "exact_meta_like/relax_results.jsonl",
        "preliminary_report": root / "preliminary_report.json",
        "success": root / "_SUCCESS",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("preliminary cell is not terminal: " + ", ".join(missing))
    attempts = read_jsonl(required["attempt_results"])
    summary = read_json(required["attempt_summary"])
    manifest = read_json(required["input_manifest"])
    strict_relax = read_jsonl(required["strict_relax"])
    meta_relax = read_jsonl(required["meta_relax"])
    if (
        strict_relax != meta_relax
        or len(attempts) != DENOMINATOR
        or [int(row.get("generation_ordinal", -1)) for row in attempts]
        != list(range(DENOMINATOR))
        or len({str(row.get("attempt_id")) for row in attempts}) != DENOMINATOR
        or int((summary.get("counts") or {}).get("total_attempts", -1))
        != DENOMINATOR
        or summary.get("ok") is not True
        or manifest.get("retry_or_replacement_used") is not False
        or any(row.get("retry_or_replacement_used") is not False for row in attempts)
    ):
        raise ContractError(f"preliminary denominator changed: {cell['cell_id']}")

    manifest_rows = list(manifest["attempt_records"])
    manifest_by_id = {str(row["attempt_id"]): row for row in manifest_rows}
    attempt_ids = [str(row["attempt_id"]) for row in attempts]
    if len(manifest_by_id) != DENOMINATOR or set(manifest_by_id) != set(attempt_ids):
        raise ContractError(f"input mapping changed: {cell['cell_id']}")
    novel_unique_attempts = [
        row for row in attempts if bool((row.get("metrics") or {}).get("novel_unique"))
    ]
    if len(novel_unique_attempts) != int(summary["counts"]["novel_unique"]):
        raise ContractError(f"novel-unique count changed: {cell['cell_id']}")
    novel_unique_attempts.sort(
        key=lambda row: int(
            manifest_by_id[str(row["attempt_id"])]["reconstructed_index"]
        )
    )
    relax_by_local = {int(row["local_index"]): row for row in strict_relax}
    if (
        len(relax_by_local) != len(strict_relax)
        or set(relax_by_local) != set(range(len(novel_unique_attempts)))
    ):
        raise ContractError(f"relax alignment changed: {cell['cell_id']}")

    alignment: list[dict[str, Any]] = []
    for local_index, attempt in enumerate(novel_unique_attempts):
        relax = relax_by_local[local_index]
        attempt_id = str(attempt["attempt_id"])
        manifest_row = manifest_by_id[attempt_id]
        metrics = attempt["metrics"]
        composition = Composition(relax["composition"])
        energy = relax.get("energy_per_atom")
        if not same_optional_float(energy, metrics.get("energy_per_atom")):
            raise ContractError(f"relax energy changed: {cell['cell_id']}:{attempt_id}")
        old_hull = metrics.get("e_above_hull")
        old_strict = bool(metrics.get("strict_full_sun"))
        old_meta = bool(metrics.get("meta_full_sun"))
        if old_hull is None:
            if old_strict or old_meta:
                raise ContractError("unknown preliminary hull marked stable")
        else:
            value = float(old_hull)
            if old_strict != (value <= STRICT_THRESHOLD) or old_meta != (
                value <= META_THRESHOLD
            ):
                raise ContractError("preliminary stability label mismatch")
        alignment.append(
            {
                "schema": "h1a2_retrained_postonly_official_alignment_v1",
                "cell_id": cell["cell_id"],
                "panel": cell["panel"],
                "cohort_id": cell["cohort_id"],
                "cohort_index": cell.get("cohort_index"),
                "process_repeat": cell.get("process_repeat"),
                "stage": cell["stage"],
                "attempt_id": attempt_id,
                "generation_ordinal": int(attempt["generation_ordinal"]),
                "reconstructed_index": int(manifest_row["reconstructed_index"]),
                "local_index": local_index,
                "composition": composition.as_dict(),
                "formula": composition.reduced_formula,
                "chemsys": normalized_chemsys(composition),
                "energy_per_atom": None if energy is None else float(energy),
                "preliminary_e_above_hull": (
                    None if old_hull is None else float(old_hull)
                ),
                "preliminary_strict_full_sun": old_strict,
                "preliminary_meta_full_sun": old_meta,
            }
        )

    counts = summary["counts"]
    report = {
        **{
            key: cell.get(key)
            for key in (
                "cell_index",
                "cell_id",
                "panel",
                "cohort_id",
                "cohort_index",
                "process_repeat",
                "stage",
                "body",
                "schedule",
                "expected_attempts",
            )
        },
        "arm": str(cell["body"]),
        "repeat": int(cell.get("process_repeat") or 0),
        "source_files": {label: identity(path) for label, path in required.items()},
        "attempts": DENOMINATOR,
        "reconstructed": int(counts["reconstructed"]),
        "novel": int(counts["novel"]),
        "unique": int(counts["unique"]),
        "novel_unique": int(counts["novel_unique"]),
        "preliminary_strict_full_sun": int(counts["strict_full_sun"]),
        "preliminary_meta_full_sun": int(counts["meta_full_sun"]),
        "preliminary_hull_unknown": int(counts["relaxation_or_hull_unknown"]),
        "alignment_sha256": canonical_sha256(alignment),
        "pre_refine_evaluated": False,
    }
    return report, alignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(source / "CONFIG.json")
    _, cells = load_upstream_cells(config)
    run_root = args.run_root.resolve()
    reports: list[dict[str, Any]] = []
    alignments: dict[str, list[dict[str, Any]]] = {}
    wanted: set[str] = set()
    for cell in cells:
        report, alignment = audit_cell(run_root, cell)
        reports.append(report)
        alignments[str(cell["cell_id"])] = alignment
        wanted.update(row["chemsys"] for row in alignment)
        print(
            {
                "cell": cell["cell_id"],
                "panel": cell["panel"],
                "reconstructed": report["reconstructed"],
                "novel_unique": report["novel_unique"],
            },
            flush=True,
        )
    if args.audit_only:
        print(
            {
                "audit_only": True,
                "ready_cells": len(reports),
                "wanted_chemsys": len(wanted),
            }
        )
        return

    final = run_root / "inputs"
    if final.exists():
        raise FileExistsError(final)
    preparing = run_root / f".inputs.preparing.{os.getpid()}"
    failed = run_root / f".inputs.FAILED.{os.getpid()}"
    preparing.mkdir(parents=True, exist_ok=False)
    try:
        alignment_dir = preparing / "alignments"
        alignment_dir.mkdir()
        for report in reports:
            cell_id = str(report["cell_id"])
            path = alignment_dir / f"{int(report['cell_index']):03d}_{cell_id}.jsonl"
            write_jsonl_exclusive(path, alignments[cell_id])
            output_identity = identity(path)
            output_identity["path"] = str(
                (final / "alignments" / path.name).resolve()
            )
            report["alignment_file"] = output_identity
        wanted_rows = [
            {
                "query_index": index,
                "chemsys": chemsys,
                "elements": chemsys.split("-"),
            }
            for index, chemsys in enumerate(sorted(wanted))
        ]
        write_jsonl_exclusive(preparing / "wanted_chemsys.jsonl", wanted_rows)
        manifest = {
            "schema": "h1a2_retrained_postonly_official_sun_input_manifest_v1",
            "source_manifest_sha256": args.source_manifest_sha256,
            "cell_count": len(reports),
            "evaluated_stage": "post_model494_only",
            "pre_refine_evaluated": False,
            "wanted_chemsys_count": len(wanted_rows),
            "wanted_chemsys_sha256": canonical_sha256(wanted_rows),
            "cells": reports,
            "generation_or_refinement_rerun": False,
            "frozen_non_stability_components": [
                "reconstruction",
                "novelty",
                "uniqueness",
                "chgnet_relaxed_energy",
            ],
        }
        write_json_exclusive(preparing / "input_manifest.json", manifest)
        (preparing / "inputs_SUCCESS").touch(exist_ok=False)
        preparing.rename(final)
    except Exception:
        if preparing.exists():
            shutil.move(str(preparing), str(failed))
        raise
    print({"input_cells": len(reports), "wanted_chemsys": len(wanted)})


if __name__ == "__main__":
    main()
