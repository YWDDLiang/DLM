#!/usr/bin/env python3
"""Validate one frozen post-refine U/N + CHGNet preliminary S.U.N. cell."""

from __future__ import annotations

import argparse
from pathlib import Path

from protocol import (
    ContractError,
    DENOMINATOR,
    identity,
    load_upstream_cells,
    read_json,
    read_jsonl,
    require_source_manifest,
    write_json_exclusive,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(source / "CONFIG.json")
    _, cells = load_upstream_cells(config)
    index = int(args.cell_index)
    if index < 0 or index >= len(cells):
        raise ContractError("cell index out of range")
    cell = cells[index]
    run_root = args.run_root.resolve()
    root = run_root / f"preliminary/{index:03d}_{cell['cell_id']}"
    attempts_path = root / "attempt_results.jsonl"
    summary_path = root / "attempt_summary.json"
    manifest_path = root / "input_manifest.json"
    strict_relax_path = root / "exact_strict/relax_results.jsonl"
    meta_relax_path = root / "exact_meta_like/relax_results.jsonl"
    attempts = read_jsonl(attempts_path)
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    strict_relax = read_jsonl(strict_relax_path)
    meta_relax = read_jsonl(meta_relax_path)
    generation_path = Path(cell["generation_jsonl"])
    generation = read_jsonl(generation_path)
    expected_ids = [str(row["attempt_id"]) for row in generation]
    if (
        len(attempts) != DENOMINATOR
        or [int(row.get("generation_ordinal", -1)) for row in attempts]
        != list(range(DENOMINATOR))
        or [str(row.get("attempt_id")) for row in attempts] != expected_ids
        or any(row.get("retry_or_replacement_used") is not False for row in attempts)
        or summary.get("ok") is not True
        or int((summary.get("counts") or {}).get("total_attempts", -1))
        != DENOMINATOR
        or summary.get("denominator") != "all_generation_attempts"
        or summary.get("execution_patch_sha256")
        != args.source_manifest_sha256
        or manifest.get("retry_or_replacement_used") is not False
        or int(manifest.get("total_attempts", -1)) != DENOMINATOR
        or strict_relax != meta_relax
    ):
        raise ContractError(f"preliminary cell contract changed: {cell['cell_id']}")
    counts = summary["counts"]
    report = {
        "schema": "h1a2_retrained_postonly_preliminary_cell_v1",
        "status": "complete",
        "ok": True,
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
        "counts": {key: int(value) for key, value in counts.items()},
        "generation": identity(generation_path),
        "attempt_results": identity(attempts_path),
        "attempt_summary": identity(summary_path),
        "input_manifest": identity(manifest_path),
        "strict_relax": identity(strict_relax_path),
        "meta_relax": identity(meta_relax_path),
        "evaluated_stage": "post_model494_only",
        "pre_refine_evaluated": False,
        "preliminary_hull_selection_role": "diagnostic_only_never_headline",
        "frozen_for_official_reevaluation": [
            "reconstruction",
            "novelty",
            "uniqueness",
            "chgnet_relaxed_energy",
        ],
    }
    write_json_exclusive(root / "preliminary_report.json", report)
    (root / "_SUCCESS").touch(exist_ok=False)
    print(
        {
            "cell": cell["cell_id"],
            "panel": cell["panel"],
            "preliminary": "PASS",
            "novel_unique": counts["novel_unique"],
        }
    )


if __name__ == "__main__":
    main()
