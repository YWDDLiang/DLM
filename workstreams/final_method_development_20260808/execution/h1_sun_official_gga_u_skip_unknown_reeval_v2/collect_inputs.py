#!/usr/bin/env python3
"""Freeze stability-only inputs from completed S.U.N. evaluations."""

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
    canonical_sha256,
    cell_specs,
    identity,
    normalized_chemsys,
    read_json,
    read_jsonl,
    require_source_manifest,
    write_json_exclusive,
    write_jsonl_exclusive,
)


STRICT_THRESHOLD = 0.0
META_THRESHOLD = 0.1


def _same_optional_float(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def audit_cell(cell: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluation = Path(cell["evaluation_dir"])
    required = {
        "attempt_results": evaluation / "attempt_results.jsonl",
        "attempt_summary": evaluation / "attempt_summary.json",
        "input_manifest": evaluation / "input_manifest.json",
        "strict_relax": evaluation / "exact_strict/relax_results.jsonl",
        "meta_relax": evaluation / "exact_meta_like/relax_results.jsonl",
        "success": Path(cell["success_marker"]),
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("cell is not terminal: " + ", ".join(missing))

    attempts = read_jsonl(required["attempt_results"])
    summary = read_json(required["attempt_summary"])
    input_manifest = read_json(required["input_manifest"])
    strict_relax = read_jsonl(required["strict_relax"])
    meta_relax = read_jsonl(required["meta_relax"])
    expected = int(cell["expected_attempts"])
    if strict_relax != meta_relax:
        raise ContractError(f"strict/meta relax rows differ: {cell['cell_id']}")
    if len(attempts) != expected:
        raise ContractError(
            f"{cell['cell_id']} has {len(attempts)} attempts, expected {expected}"
        )
    ordinals = [int(row["generation_ordinal"]) for row in attempts]
    if ordinals != list(range(expected)):
        raise ContractError(f"non-contiguous attempt ordinals: {cell['cell_id']}")
    attempt_ids = [str(row["attempt_id"]) for row in attempts]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ContractError(f"duplicate attempt IDs: {cell['cell_id']}")
    if int(summary["counts"]["total_attempts"]) != expected:
        raise ContractError(f"summary denominator changed: {cell['cell_id']}")
    if summary.get("ok") is not True:
        raise ContractError(f"source evaluation is not successful: {cell['cell_id']}")
    if input_manifest.get("retry_or_replacement_used") is not False:
        raise ContractError(f"input manifest used retry/replacement: {cell['cell_id']}")
    if any(row.get("retry_or_replacement_used") is not False for row in attempts):
        raise ContractError(f"attempt ledger used retry/replacement: {cell['cell_id']}")

    manifest_rows = list(input_manifest["attempt_records"])
    manifest_by_id = {str(row["attempt_id"]): row for row in manifest_rows}
    if len(manifest_by_id) != expected or set(manifest_by_id) != set(attempt_ids):
        raise ContractError(f"input-manifest mapping changed: {cell['cell_id']}")

    novel_unique_attempts = [
        row for row in attempts if bool((row.get("metrics") or {}).get("novel_unique"))
    ]
    if len(novel_unique_attempts) != int(summary["counts"]["novel_unique"]):
        raise ContractError(f"novel-unique count changed: {cell['cell_id']}")
    frozen_counts = {
        "novel": sum(bool((row.get("metrics") or {}).get("novel")) for row in attempts),
        "unique": sum(
            bool((row.get("metrics") or {}).get("unique_representative"))
            for row in attempts
        ),
        "strict_full_sun": sum(
            bool((row.get("metrics") or {}).get("strict_full_sun"))
            for row in attempts
        ),
        "meta_full_sun": sum(
            bool((row.get("metrics") or {}).get("meta_full_sun"))
            for row in attempts
        ),
    }
    for name, observed in frozen_counts.items():
        if observed != int(summary["counts"][name]):
            raise ContractError(
                f"attempt/summary {name} mismatch: {cell['cell_id']}"
            )
    try:
        novel_unique_attempts.sort(
            key=lambda row: int(
                manifest_by_id[str(row["attempt_id"])]["reconstructed_index"]
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(
            f"missing reconstructed index for novel-unique attempt: {cell['cell_id']}"
        ) from exc

    relax_by_local: dict[int, dict[str, Any]] = {}
    for row in strict_relax:
        local_index = int(row["local_index"])
        if local_index in relax_by_local:
            raise ContractError(f"duplicate relax local index: {cell['cell_id']}")
        relax_by_local[local_index] = row
    if set(relax_by_local) != set(range(len(novel_unique_attempts))):
        raise ContractError(f"incomplete relax alignment: {cell['cell_id']}")

    alignment: list[dict[str, Any]] = []
    for local_index, attempt in enumerate(novel_unique_attempts):
        relax = relax_by_local[local_index]
        attempt_id = str(attempt["attempt_id"])
        manifest_row = manifest_by_id[attempt_id]
        metrics = attempt["metrics"]
        composition = Composition(relax["composition"])
        energy = relax.get("energy_per_atom")
        old_energy = metrics.get("energy_per_atom")
        if not _same_optional_float(energy, old_energy):
            raise ContractError(f"relax-energy mismatch: {cell['cell_id']}:{attempt_id}")
        old_hull = metrics.get("e_above_hull")
        old_strict = bool(metrics.get("strict_full_sun"))
        old_meta = bool(metrics.get("meta_full_sun"))
        if old_hull is None:
            if old_strict or old_meta:
                raise ContractError(f"unknown hull marked stable: {cell['cell_id']}")
        else:
            value = float(old_hull)
            if old_strict != (value <= STRICT_THRESHOLD):
                raise ContractError(f"old strict label mismatch: {cell['cell_id']}")
            if old_meta != (value <= META_THRESHOLD):
                raise ContractError(f"old meta label mismatch: {cell['cell_id']}")
        alignment.append(
            {
                "schema": "h1_sun_clean_alignment_v1",
                "cell_id": cell["cell_id"],
                "attempt_id": attempt_id,
                "generation_ordinal": int(attempt["generation_ordinal"]),
                "reconstructed_index": int(manifest_row["reconstructed_index"]),
                "local_index": local_index,
                "composition": composition.as_dict(),
                "formula": composition.reduced_formula,
                "chemsys": normalized_chemsys(composition),
                "energy_per_atom": None if energy is None else float(energy),
                "old_e_above_hull": None if old_hull is None else float(old_hull),
                "old_strict_full_sun": old_strict,
                "old_meta_full_sun": old_meta,
                "old_evaluation_status": str(attempt["evaluation_status"]),
            }
        )

    old_unknown = sum(row["old_e_above_hull"] is None for row in alignment)
    if old_unknown != int(summary["counts"]["relaxation_or_hull_unknown"]):
        raise ContractError(
            f"old unknown accounting changed: {cell['cell_id']}"
        )

    report = {
        **cell,
        "source_files": {label: identity(path) for label, path in required.items()},
        "attempts": expected,
        "reconstructed": int(summary["counts"]["reconstructed"]),
        "novel": int(summary["counts"]["novel"]),
        "unique": int(summary["counts"]["unique"]),
        "novel_unique": len(alignment),
        "old_strict_full_sun": int(summary["counts"]["strict_full_sun"]),
        "old_meta_full_sun": int(summary["counts"]["meta_full_sun"]),
        "old_hull_unknown": old_unknown,
        "alignment_sha256": canonical_sha256(alignment),
    }
    return report, alignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    if config.get("schema") != "h1_sun_official_gga_u_skip_unknown_reeval_config_v2":
        raise ContractError("unexpected config schema")
    cells = cell_specs(config)
    if len(cells) != int(config["expected_cell_count"]):
        raise ContractError("configured cell count changed")

    reports: list[dict[str, Any]] = []
    alignments: dict[str, list[dict[str, Any]]] = {}
    wanted: set[str] = set()
    for cell in cells:
        report, alignment = audit_cell(cell)
        reports.append(report)
        alignments[cell["cell_id"]] = alignment
        wanted.update(row["chemsys"] for row in alignment)
        print(
            {
                "cell": cell["cell_id"],
                "attempts": report["attempts"],
                "novel_unique": report["novel_unique"],
                "old_strict": report["old_strict_full_sun"],
                "old_meta": report["old_meta_full_sun"],
            },
            flush=True,
        )

    if args.audit_only:
        print(
            {
                "audit_only": True,
                "ready_cells": len(reports),
                "wanted_chemsys": len(wanted),
            },
            flush=True,
        )
        return
    if args.run_root is None:
        raise ContractError("--run-root is required unless --audit-only is used")
    run_root = args.run_root.resolve()
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
            cell_id = report["cell_id"]
            path = alignment_dir / f"{int(report['cell_index']):03d}_{cell_id}.jsonl"
            write_jsonl_exclusive(path, alignments[cell_id])
            alignment_identity = identity(path)
            alignment_identity["path"] = str(
                (final / "alignments" / path.name).resolve()
            )
            report["alignment_file"] = alignment_identity
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
            "schema": "h1_sun_official_gga_u_clean_input_manifest_v1",
            "source_manifest_sha256": args.source_manifest_sha256,
            "cell_count": len(reports),
            "wanted_chemsys_count": len(wanted_rows),
            "wanted_chemsys_sha256": canonical_sha256(wanted_rows),
            "cells": reports,
            "no_generation_or_relaxation_rerun": True,
        }
        write_json_exclusive(preparing / "input_manifest.json", manifest)
        (preparing / "inputs_SUCCESS").touch(exist_ok=False)
        preparing.rename(final)
    except Exception:
        if preparing.exists():
            shutil.move(str(preparing), str(failed))
        raise
    print(
        {"input_cells": len(reports), "wanted_chemsys": len(wanted)},
        flush=True,
    )


if __name__ == "__main__":
    main()
