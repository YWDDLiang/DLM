#!/usr/bin/env python3
"""Validate one environment-repair V2 replay of a frozen R03 refined256 repeat."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from protocol import (
    ContractError,
    historical_paths,
    identity,
    read_json,
    read_jsonl,
    repeat_spec,
    require_file,
    require_source_manifest,
    sha256_file,
    write_json_exclusive,
)


ENDPOINTS = (
    "novel",
    "unique_representative",
    "novel_unique",
    "strict_full_sun",
    "meta_full_sun",
)


def metric(row: Mapping[str, Any], name: str) -> bool:
    return bool((row.get("metrics") or {}).get(name))


def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    statuses = Counter(str(row.get("evaluation_status")) for row in rows)
    return {
        "total_attempts": len(rows),
        "generation_succeeded": sum(
            row.get("generation_status") == "succeeded" for row in rows
        ),
        "novel": sum(metric(row, "novel") for row in rows),
        "unique_representative": sum(
            metric(row, "unique_representative") for row in rows
        ),
        "novel_unique": sum(metric(row, "novel_unique") for row in rows),
        "strict_full_sun": sum(metric(row, "strict_full_sun") for row in rows),
        "meta_full_sun": sum(metric(row, "meta_full_sun") for row in rows),
        "hull_evaluated": sum(
            (row.get("metrics") or {}).get("e_above_hull") is not None
            for row in rows
        ),
        "relaxation_or_hull_unknown": statuses.get(
            "relaxation_or_hull_unknown", 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repeat", type=int, choices=range(4), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(source / "CONFIG.json")
    run_root = args.run_root.resolve()
    repeat = args.repeat
    output = args.output_dir.resolve()
    expected_output = run_root / f"repeats/{repeat}/current_sun"
    if Path(config["run_root"]).resolve() != run_root or output != expected_output:
        raise ContractError("replay output escaped immutable run root")

    completion_dir = run_root / "mp_cache"
    if not (completion_dir / "completion_SUCCESS").is_file():
        raise FileNotFoundError(completion_dir / "completion_SUCCESS")
    completion = read_json(completion_dir / "completion_manifest.json")
    completed_cache = completion_dir / "completed_mp_hull_cache.jsonl"
    cache_spec = completion.get("completed_mp_hull_cache") or {}
    if (
        completion.get("status")
        != "complete_all_historical_chemsys_resolved"
        or completion.get("source_manifest_sha256")
        != args.source_manifest_sha256
        or int(completion.get("wanted_chemsys_count", -1)) != 224
        or int(completion.get("missing_chemsys_count", -1)) != 92
        or int(cache_spec.get("rows", -1)) != 224
        or cache_spec.get("all_rows_populated") is not True
        or sha256_file(completed_cache) != cache_spec.get("sha256")
    ):
        raise ContractError("cohort-complete current MP cache contract changed")

    generation, relax_cache, old_attempt_path = historical_paths(config, repeat)
    spec = repeat_spec(config, repeat)
    require_file(generation, str(spec["generation_sha256"]), "frozen generation")
    require_file(
        relax_cache,
        str(spec["frozen_relax_cache_sha256"]),
        "frozen CHGNet relax cache",
    )
    require_file(
        old_attempt_path,
        str(spec["old_attempt_results_sha256"]),
        "historical R03G attempt results",
    )
    generation_rows = read_jsonl(generation)
    if (
        len(generation_rows) != 256
        or [int(row.get("ordinal", -1)) for row in generation_rows]
        != list(range(256))
        or len({str(row.get("attempt_id")) for row in generation_rows}) != 256
        or sum(row.get("status") == "succeeded" for row in generation_rows) != 248
    ):
        raise ContractError("frozen refined256 attempt ledger changed")
    expected_ids = [str(row["attempt_id"]) for row in generation_rows]

    summary = read_json(output / "attempt_summary.json")
    current_rows = read_jsonl(output / "attempt_results.jsonl")
    old_rows = read_jsonl(old_attempt_path)
    current_rows.sort(key=lambda row: int(row.get("generation_ordinal", -1)))
    old_rows.sort(key=lambda row: int(row.get("generation_ordinal", -1)))
    if (
        summary.get("ok") is not True
        or summary.get("denominator") != "all_generation_attempts"
        or summary.get("execution_patch_sha256")
        != args.source_manifest_sha256
        or summary.get("retry_or_replacement_used") is not False
        or len(current_rows) != 256
        or len(old_rows) != 256
        or [str(row.get("attempt_id")) for row in current_rows] != expected_ids
        or [str(row.get("attempt_id")) for row in old_rows] != expected_ids
        or [int(row.get("generation_ordinal", -1)) for row in current_rows]
        != list(range(256))
        or any(row.get("retry_or_replacement_used") is not False for row in current_rows)
    ):
        raise ContractError("current/old attempt mapping changed")

    summary_counts = summary.get("counts") or {}
    current_counts = counts(current_rows)
    expected_summary_counts = {
        "total_attempts": current_counts["total_attempts"],
        "reconstructed": 248,
        "novel": current_counts["novel"],
        "unique": current_counts["unique_representative"],
        "novel_unique": current_counts["novel_unique"],
        "strict_full_sun": current_counts["strict_full_sun"],
        "meta_full_sun": current_counts["meta_full_sun"],
        "relaxation_or_hull_unknown": current_counts[
            "relaxation_or_hull_unknown"
        ],
    }
    if any(
        int(summary_counts.get(name, -1)) != value
        for name, value in expected_summary_counts.items()
    ):
        raise ContractError("current S.U.N. summary disagrees with attempts")

    source_cache = ((summary.get("assets") or {}).get("mp_hull_cache") or {})
    working_cache = summary.get("working_mp_hull_cache") or {}
    source_relax = ((summary.get("assets") or {}).get("chgnet_relax_cache") or {})
    working_relax = output / "working_chgnet_relax_cache.jsonl"
    if (
        source_cache.get("sha256") != cache_spec.get("sha256")
        or working_cache.get("sha256") != cache_spec.get("sha256")
        or sha256_file(output / "working_mp_hull_cache.jsonl")
        != cache_spec.get("sha256")
        or source_relax.get("sha256") != spec["frozen_relax_cache_sha256"]
        or sha256_file(working_relax) != spec["frozen_relax_cache_sha256"]
    ):
        raise ContractError("evaluation mutated or rebound a frozen cache")

    novelty_parity = {
        name: sum(metric(old, name) != metric(new, name) for old, new in zip(old_rows, current_rows))
        for name in ("novel", "unique_representative", "novel_unique")
    }
    energy_mismatches = 0
    energy_pairs = 0
    for old, new in zip(old_rows, current_rows):
        old_energy = (old.get("metrics") or {}).get("energy_per_atom")
        new_energy = (new.get("metrics") or {}).get("energy_per_atom")
        if old_energy is None and new_energy is None:
            continue
        energy_pairs += 1
        if (
            old_energy is None
            or new_energy is None
            or not math.isclose(
                float(old_energy), float(new_energy), rel_tol=0.0, abs_tol=1e-12
            )
        ):
            energy_mismatches += 1
    if any(novelty_parity.values()) or energy_mismatches:
        raise ContractError(
            "non-MP components changed; replay no longer isolates hull snapshot"
        )

    old_counts = counts(old_rows)
    transitions: dict[str, dict[str, int]] = {}
    for endpoint in ("strict_full_sun", "meta_full_sun"):
        pairs = [(metric(old, endpoint), metric(new, endpoint)) for old, new in zip(old_rows, current_rows)]
        transitions[endpoint] = {
            "both": sum(before and after for before, after in pairs),
            "old_only": sum(before and not after for before, after in pairs),
            "current_only": sum(not before and after for before, after in pairs),
            "neither": sum(not before and not after for before, after in pairs),
        }

    report = {
        "schema": "h1_r03_refined256_current_sun_repeat_validation_v1",
        "status": "complete",
        "ok": True,
        "repeat": repeat,
        "attempts": 256,
        "reconstructed": 248,
        "old_counts": old_counts,
        "current_counts": current_counts,
        "transitions": transitions,
        "isolation_checks": {
            "generation_byte_frozen": True,
            "chgnet_relax_cache_byte_frozen_after_run": True,
            "mp_cache_cohort_complete_rows": 224,
            "mp_api_used_in_slurm": False,
            "novelty_uniqueness_mismatch_counts": novelty_parity,
            "energy_pairs_compared": energy_pairs,
            "energy_mismatches": energy_mismatches,
            "plan_body_refine_rerun": False,
        },
        "artifacts": {
            "generation": identity(generation),
            "frozen_relax_cache": identity(relax_cache),
            "historical_attempt_results": identity(old_attempt_path),
            "current_attempt_results": identity(output / "attempt_results.jsonl"),
            "current_attempt_summary": identity(output / "attempt_summary.json"),
            "completed_mp_cache": identity(completed_cache),
        },
        "source_manifest_sha256": args.source_manifest_sha256,
    }
    write_json_exclusive(output / "repeat_validation.json", report)
    with (output / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
