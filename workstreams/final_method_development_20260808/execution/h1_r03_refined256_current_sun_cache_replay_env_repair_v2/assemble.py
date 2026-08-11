#!/usr/bin/env python3
"""Assemble the environment-repair V2 historical-vs-current cache replay."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from protocol import (
    ContractError,
    historical_paths,
    identity,
    read_json,
    read_jsonl,
    require_source_manifest,
    write_json_exclusive,
)


def metric(row: Mapping[str, Any], name: str) -> bool:
    return bool((row.get("metrics") or {}).get(name))


def exact_mcnemar(old: list[bool], current: list[bool]) -> dict[str, Any]:
    if len(old) != len(current):
        raise ContractError("paired vectors have different lengths")
    old_only = sum(before and not after for before, after in zip(old, current))
    current_only = sum(not before and after for before, after in zip(old, current))
    discordant = old_only + current_only
    if discordant == 0:
        p = 1.0
    else:
        lower = min(old_only, current_only)
        p = min(
            1.0,
            2.0
            * sum(math.comb(discordant, value) for value in range(lower + 1))
            / (2**discordant),
        )
    return {
        "old_only": old_only,
        "current_only": current_only,
        "discordant": discordant,
        "exact_two_sided_p": p,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "attempts": len(rows),
        "generation_succeeded": sum(
            row.get("generation_status") == "succeeded" for row in rows
        ),
        "novel": sum(metric(row, "novel") for row in rows),
        "unique_representative": sum(
            metric(row, "unique_representative") for row in rows
        ),
        "novel_unique": sum(metric(row, "novel_unique") for row in rows),
        "hull_evaluated": sum(
            (row.get("metrics") or {}).get("e_above_hull") is not None
            for row in rows
        ),
        "hull_unknown": sum(
            row.get("evaluation_status") == "relaxation_or_hull_unknown"
            for row in rows
        ),
        "strict_full_sun": sum(metric(row, "strict_full_sun") for row in rows),
        "meta_full_sun": sum(metric(row, "meta_full_sun") for row in rows),
    }
    reconstructed = 248
    result["rates"] = {
        "strict_exact_legacy_reconstructed": result["strict_full_sun"]
        / reconstructed,
        "meta_exact_legacy_reconstructed": result["meta_full_sun"]
        / reconstructed,
        "strict_all_256": result["strict_full_sun"] / 256,
        "meta_all_256": result["meta_full_sun"] / 256,
        "strict_evaluated_diagnostic": (
            result["strict_full_sun"] / result["hull_evaluated"]
            if result["hull_evaluated"]
            else None
        ),
        "meta_evaluated_diagnostic": (
            result["meta_full_sun"] / result["hull_evaluated"]
            if result["hull_evaluated"]
            else None
        ),
    }
    return result


def rate_cell(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} = {100.0 * numerator / denominator:.2f}%"


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Historical frozen R03 refined256 re-evaluated with current S.U.N./MP cache",
        "",
        "## Interpretation contract",
        "",
        "The active H1A2 and current seven-line h1_rich_plan_v1 prompt branch is byte-identical. The actual sampling change is the random-stream/cohort design: legacy used one global RNG stream at seed 17029, while the current study uses stateless ordinals across three independent base seeds and three disjoint first-1000 parse-success cohorts.",
        "",
        "This replay does not run Plan, body/DLM, diffusion refinement, or CHGNet. It reuses the byte-frozen refined256 generation ledgers and frozen CHGNet relax-energy caches, then reruns the current exact novelty/uniqueness and MP-hull evaluation against a cohort-complete cache. Therefore old-to-current changes isolate the MP snapshot/completion effect under the same S.U.N. implementation.",
        "",
        "Headline S.U.N. uses reconstructed structures (248) exactly as the legacy evaluator. The all-256 rate is a conservative secondary denominator; evaluated-only is diagnostic.",
        "",
        "## Per-repeat S.U.N. and meta-S.U.N.",
        "",
        "| Repeat | Snapshot | Evaluated | Unknown | strict S.U.N. /248 | strict /256 | meta-S.U.N. /248 | meta /256 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for repeat in report["repeats"]:
        for label in ("historical", "current_cohort_complete"):
            row = repeat[label]
            lines.append(
                "| {repeat} | {label} | {evaluated} | {unknown} | {strict248} | {strict256} | {meta248} | {meta256} |".format(
                    repeat=repeat["repeat"],
                    label=label,
                    evaluated=row["hull_evaluated"],
                    unknown=row["hull_unknown"],
                    strict248=rate_cell(row["strict_full_sun"], 248),
                    strict256=rate_cell(row["strict_full_sun"], 256),
                    meta248=rate_cell(row["meta_full_sun"], 248),
                    meta256=rate_cell(row["meta_full_sun"], 256),
                )
            )
    lines += [
        "",
        "## Full S.U.N. components",
        "",
        "| Repeat | Snapshot | Novel | Unique representatives | Novel-unique | Hull evaluated | Hull unknown |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for repeat in report["repeats"]:
        for label in ("historical", "current_cohort_complete"):
            row = repeat[label]
            lines.append(
                f"| {repeat['repeat']} | {label} | {row['novel']} | {row['unique_representative']} | {row['novel_unique']} | {row['hull_evaluated']} | {row['hull_unknown']} |"
            )
    lines += [
        "",
        "## Paired cache-snapshot transitions",
        "",
        "| Repeat | Endpoint | Old only | Current only | Discordant | Exact McNemar p |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for repeat in report["repeats"]:
        for endpoint in ("strict_full_sun", "meta_full_sun"):
            row = repeat["paired"][endpoint]
            lines.append(
                f"| {repeat['repeat']} | {endpoint} | {row['old_only']} | {row['current_only']} | {row['discordant']} | {row['exact_two_sided_p']:.8g} |"
            )
    lines += [
        "",
        "## Cache audit",
        "",
        f"- Historical refined256 union: {report['cache']['wanted_chemsys_count']} chemical systems.",
        f"- Resolved before new queries: {report['cache']['existing_resolved_count']}.",
        f"- Current MP completion queries: {report['cache']['missing_chemsys_count']}; all resolved.",
        f"- Final cohort cache rows: {report['cache']['completed_rows']}; all populated.",
        "- MP API was not available inside Slurm evaluation jobs.",
        "- Four repeats are CUDA process realizations on the same frozen first256 cohort; pooled 1024 values are descriptive only and are not treated as independent samples.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
        handle.flush()
        os.fsync(handle.fileno())


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
    if Path(config["run_root"]).resolve() != run_root:
        raise ContractError("assembly run root changed")
    completion = read_json(run_root / "mp_cache/completion_manifest.json")
    cache_spec = completion.get("completed_mp_hull_cache") or {}

    repeats: list[dict[str, Any]] = []
    old_pool: dict[str, list[bool]] = {
        "strict_full_sun": [], "meta_full_sun": []
    }
    new_pool: dict[str, list[bool]] = {
        "strict_full_sun": [], "meta_full_sun": []
    }
    for repeat in range(4):
        repeat_root = run_root / f"repeats/{repeat}"
        if not (repeat_root / "_SUCCESS").is_file():
            raise FileNotFoundError(repeat_root / "_SUCCESS")
        validation = read_json(
            repeat_root / "current_sun/repeat_validation.json"
        )
        if validation.get("ok") is not True or int(validation.get("repeat", -1)) != repeat:
            raise ContractError(f"repeat {repeat} validation changed")
        _, _, old_path = historical_paths(config, repeat)
        old_rows = read_jsonl(old_path)
        new_rows = read_jsonl(repeat_root / "current_sun/attempt_results.jsonl")
        old_rows.sort(key=lambda row: int(row["generation_ordinal"]))
        new_rows.sort(key=lambda row: int(row["generation_ordinal"]))
        paired: dict[str, Any] = {}
        for endpoint in ("strict_full_sun", "meta_full_sun"):
            old_vector = [metric(row, endpoint) for row in old_rows]
            new_vector = [metric(row, endpoint) for row in new_rows]
            paired[endpoint] = exact_mcnemar(old_vector, new_vector)
            old_pool[endpoint].extend(old_vector)
            new_pool[endpoint].extend(new_vector)
        repeats.append(
            {
                "repeat": repeat,
                "historical": summarize(old_rows),
                "current_cohort_complete": summarize(new_rows),
                "paired": paired,
                "validation": identity(
                    repeat_root / "current_sun/repeat_validation.json"
                ),
            }
        )

    pooled = {
        endpoint: {
            **exact_mcnemar(old_pool[endpoint], new_pool[endpoint]),
            "historical_count": sum(old_pool[endpoint]),
            "current_count": sum(new_pool[endpoint]),
            "descriptive_denominator": 1024,
            "independent_sample_inference": False,
        }
        for endpoint in old_pool
    }
    report = {
        "schema": "h1_r03_refined256_current_sun_cache_replay_terminal_v1",
        "status": "complete",
        "ok": True,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": config["run_id"],
        "prompt_audit": {
            **config["prompt_audit"],
            "conclusion": "active_prompt_unchanged_sampling_stream_and_cohort_changed",
        },
        "replay_interpretation": "current_exact_sun_and_cohort_complete_current_mp_cache_on_byte_frozen_historical_refined256",
        "repeats": repeats,
        "pooled_descriptive_only": pooled,
        "cache": {
            "wanted_chemsys_count": int(completion["wanted_chemsys_count"]),
            "existing_resolved_count": int(completion["existing_resolved_count"]),
            "missing_chemsys_count": int(completion["missing_chemsys_count"]),
            "completed_rows": int(cache_spec["rows"]),
            "completed_sha256": cache_spec["sha256"],
            "transport_retries": int(completion["transport_retries"]),
        },
        "denominators": {
            "headline": "reconstructed_structures_exact_legacy_248",
            "secondary": "all_generation_attempts_256",
            "evaluated_only": "diagnostic_only",
        },
        "isolation": {
            "planner_rerun": False,
            "body_rerun": False,
            "refinement_rerun": False,
            "chgnet_rerun": False,
            "mp_api_in_slurm": False,
            "historical_generation_and_relax_caches_byte_frozen": True,
        },
        "repeat_role": config["historical"]["repeat_role"],
        "pooled_1024_independence_assumed": False,
        "artifacts": {
            "completion_manifest": identity(
                run_root / "mp_cache/completion_manifest.json"
            ),
            "completed_mp_cache": identity(
                run_root / "mp_cache/completed_mp_hull_cache.jsonl"
            ),
        },
        "source_manifest_sha256": args.source_manifest_sha256,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json_exclusive(run_root / "terminal_report.json", report)
    write_markdown(run_root / "RESULTS_COMPLETE.md", report)
    status = run_root / "status"
    status.mkdir(parents=True, exist_ok=True)
    with (status / "assembly_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    with (run_root / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
