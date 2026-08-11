#!/usr/bin/env python3
"""Assemble old-versus-clean S.U.N. results and paired V4 statistics."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from protocol import (
    ContractError,
    canonical_json,
    identity,
    read_json,
    read_jsonl,
    require_source_manifest,
    write_json_exclusive,
)


BOOTSTRAP_DRAWS = 50_000
BOOTSTRAP_SEED = 20260811


def exact_mcnemar(left: list[bool], right: list[bool]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ContractError("paired McNemar vectors differ in length")
    left_only = sum(int(a and not b) for a, b in zip(left, right))
    right_only = sum(int(b and not a) for a, b in zip(left, right))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        # Sum the binomial tail in log space.  Converting 2**n or comb(n, k)
        # directly to float caused the V4 assembly overflow for large n.
        cutoff = min(left_only, right_only)
        log_terms = [
            math.lgamma(discordant + 1)
            - math.lgamma(k + 1)
            - math.lgamma(discordant - k + 1)
            - discordant * math.log(2.0)
            for k in range(cutoff + 1)
        ]
        anchor = max(log_terms)
        tail = math.exp(anchor) * sum(math.exp(value - anchor) for value in log_terms)
        p_value = min(1.0, 2.0 * tail)
    return {
        "left_only": left_only,
        "right_only": right_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def hierarchical_bootstrap(
    differences_by_repeat: list[list[int]], seed_offset: int
) -> dict[str, Any]:
    if not differences_by_repeat or any(not values for values in differences_by_repeat):
        raise ContractError("empty hierarchical bootstrap input")
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    repeats = len(differences_by_repeat)
    selected = rng.integers(0, repeats, size=(BOOTSTRAP_DRAWS, repeats))
    position_means = np.empty((BOOTSTRAP_DRAWS, repeats), dtype=np.float64)
    for position in range(repeats):
        for repeat, values in enumerate(differences_by_repeat):
            mask = selected[:, position] == repeat
            count = int(mask.sum())
            if not count:
                continue
            array = np.asarray(values, dtype=np.int8)
            negatives = int((array == -1).sum())
            zeros = int((array == 0).sum())
            positives = int((array == 1).sum())
            draws = rng.multinomial(
                len(values),
                [negatives / len(values), zeros / len(values), positives / len(values)],
                size=count,
            )
            position_means[mask, position] = (
                draws[:, 2] - draws[:, 0]
            ) / len(values)
    samples = position_means.mean(axis=1)
    observed = float(
        np.mean([np.mean(values) for values in differences_by_repeat])
    )
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED + seed_offset,
        "estimate_rate_difference": observed,
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "probability_difference_gt_zero": float(np.mean(samples > 0.0)),
    }


def rate_cell(value: int, denominator: int) -> str:
    return f"{value}/{denominator} ({100.0 * value / denominator:.2f}%)"


def load_attempt_endpoint(path: Path, key: str) -> dict[int, bool]:
    rows = read_jsonl(path)
    result = {
        int(row["generation_ordinal"]): bool((row.get("metrics") or {}).get(key))
        for row in rows
    }
    if len(result) != len(rows):
        raise ContractError(f"duplicate generation ordinal: {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    run_root = args.run_root.resolve()
    input_manifest = read_json(run_root / "inputs/input_manifest.json")
    cache_manifest = read_json(
        run_root / "official_mp_cache/completion_manifest.json"
    )
    if cache_manifest.get("query_status") != "complete_all_resolved":
        raise ContractError("official MP cache is not complete")

    reports: list[dict[str, Any]] = []
    report_by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    attempts_by_key: dict[tuple[str, str, int, str], Path] = {}
    for cell in input_manifest["cells"]:
        index = int(cell["cell_index"])
        root = run_root / f"cells/{index:03d}_{cell['cell_id']}"
        if not (root / "_SUCCESS").is_file():
            raise ContractError(f"cell is not complete: {cell['cell_id']}")
        report = read_json(root / "cell_report.json")
        key = (
            str(cell["panel"]),
            str(cell["arm"]),
            int(cell["repeat"]),
            str(cell["stage"]),
        )
        if key in report_by_key:
            raise ContractError(f"duplicate cell key: {key}")
        reports.append(report)
        report_by_key[key] = report
        attempts_by_key[key] = root / "attempt_results_clean.jsonl"

    comparisons = [
        (
            "B3_minus_R03_pre_model494",
            lambda repeat: ("V4_ALL_ATTEMPT_1000", "R03", repeat, "pre_model494"),
            lambda repeat: ("V4_ALL_ATTEMPT_1000", "B3", repeat, "pre_model494"),
        ),
        (
            "B3_minus_R03_post_model494",
            lambda repeat: ("V4_ALL_ATTEMPT_1000", "R03", repeat, "post_model494"),
            lambda repeat: ("V4_ALL_ATTEMPT_1000", "B3", repeat, "post_model494"),
        ),
        (
            "post_minus_pre_R03",
            lambda repeat: ("V4_ALL_ATTEMPT_1000", "R03", repeat, "pre_model494"),
            lambda repeat: ("V4_ALL_ATTEMPT_1000", "R03", repeat, "post_model494"),
        ),
        (
            "post_minus_pre_B3",
            lambda repeat: ("V4_ALL_ATTEMPT_1000", "B3", repeat, "pre_model494"),
            lambda repeat: ("V4_ALL_ATTEMPT_1000", "B3", repeat, "post_model494"),
        ),
    ]
    paired_statistics: dict[str, Any] = {}
    seed_offset = 0
    for comparison_name, left_key, right_key in comparisons:
        endpoint_results: dict[str, Any] = {}
        for endpoint in ("strict_full_sun", "meta_full_sun"):
            differences: list[list[int]] = []
            repeat_reports: list[dict[str, Any]] = []
            for repeat in range(3):
                left = load_attempt_endpoint(attempts_by_key[left_key(repeat)], endpoint)
                right = load_attempt_endpoint(attempts_by_key[right_key(repeat)], endpoint)
                if set(left) != set(right):
                    raise ContractError(
                        f"paired ordinal set changed: {comparison_name} repeat {repeat}"
                    )
                ordinals = sorted(left)
                left_values = [left[index] for index in ordinals]
                right_values = [right[index] for index in ordinals]
                differences.append(
                    [int(b) - int(a) for a, b in zip(left_values, right_values)]
                )
                repeat_reports.append(
                    {
                        "repeat": repeat,
                        "left_rate": sum(left_values) / len(left_values),
                        "right_rate": sum(right_values) / len(right_values),
                        "right_minus_left": (
                            sum(right_values) - sum(left_values)
                        )
                        / len(left_values),
                        "exact_mcnemar": exact_mcnemar(left_values, right_values),
                    }
                )
            endpoint_results[endpoint] = {
                "repeat_reports": repeat_reports,
                "hierarchical_paired_bootstrap": hierarchical_bootstrap(
                    differences, seed_offset
                ),
            }
            seed_offset += 1
        paired_statistics[comparison_name] = endpoint_results

    ordered = sorted(
        reports,
        key=lambda row: (
            row["cell"]["panel"],
            row["cell"]["arm"],
            int(row["cell"]["repeat"]),
            row["cell"]["stage"],
        ),
    )
    terminal = {
        "schema": "h1_sun_official_gga_u_clean_terminal_v1",
        "engineering_status": "complete",
        "scientific_status": "report_all_results_without_directional_gate",
        "source_manifest_sha256": args.source_manifest_sha256,
        "thermo_contract": {
            "query_method": "MPRester.get_entries_in_chemsys",
            "compatible_only": True,
            "thermo_type": "GGA_GGA+U",
            "fresh_cache": True,
            "historical_or_august_rows_reused": 0,
            "local_compatibility_reprocessing": False,
        },
        "cell_reports": ordered,
        "paired_statistics": paired_statistics,
        "artifacts": {
            "input_manifest": identity(run_root / "inputs/input_manifest.json"),
            "official_mp_cache_manifest": identity(
                run_root / "official_mp_cache/completion_manifest.json"
            ),
            "official_full_entries": identity(
                run_root / "official_mp_cache/official_full_entries.jsonl.gz"
            ),
            "official_slim_cache": identity(
                run_root / "official_mp_cache/official_slim_cache.jsonl"
            ),
        },
    }
    write_json_exclusive(run_root / "terminal_report.json", terminal)

    lines = [
        "# Official-MP clean S.U.N. re-evaluation V1",
        "",
        "This report changes only MP phase-diagram references and `E_hull`. "
        "Generation, reconstruction, novelty, uniqueness, model-494 refinement, "
        "and CHGNet relaxed energies are byte-frozen inputs.",
        "",
        "Thermo contract: official `MPRester.get_entries_in_chemsys`, "
        "`compatible_only=True`, explicit `GGA_GGA+U`, fresh empty cache, no "
        "local compatibility reprocessing, and complete unary references.",
        "",
        "## Strict S.U.N.",
        "",
        "| Panel | Arm | Repeat | Stage | N+U | Old / all | Clean / all | Old / reconstructed | Clean / reconstructed | 0→1 | 1→0 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in ordered:
        cell = report["cell"]
        den = report["denominators"]
        old = report["old"]
        clean = report["clean"]
        paired = report["paired_old_to_clean"]
        lines.append(
            "| {panel} | {arm} | {repeat} | {stage} | {nu} | {old_all} | "
            "{new_all} | {old_rec} | {new_rec} | {up} | {down} |".format(
                panel=cell["panel"],
                arm=cell["arm"],
                repeat=cell["repeat"],
                stage=cell["stage"],
                nu=den["novel_unique"],
                old_all=rate_cell(old["strict_full_sun"], den["all_attempts"]),
                new_all=rate_cell(clean["strict_full_sun"], den["all_attempts"]),
                old_rec=rate_cell(
                    old["strict_full_sun"], den["reconstructed_exact_legacy"]
                ),
                new_rec=rate_cell(
                    clean["strict_full_sun"], den["reconstructed_exact_legacy"]
                ),
                up=paired["strict_0_to_1"],
                down=paired["strict_1_to_0"],
            )
        )
    lines.extend(
        [
            "",
            "## Meta-S.U.N. (≤0.1 eV/atom)",
            "",
            "| Panel | Arm | Repeat | Stage | N+U | Old / all | Clean / all | Old / reconstructed | Clean / reconstructed | 0→1 | 1→0 |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for report in ordered:
        cell = report["cell"]
        den = report["denominators"]
        old = report["old"]
        clean = report["clean"]
        paired = report["paired_old_to_clean"]
        lines.append(
            "| {panel} | {arm} | {repeat} | {stage} | {nu} | {old_all} | "
            "{new_all} | {old_rec} | {new_rec} | {up} | {down} |".format(
                panel=cell["panel"],
                arm=cell["arm"],
                repeat=cell["repeat"],
                stage=cell["stage"],
                nu=den["novel_unique"],
                old_all=rate_cell(old["meta_full_sun"], den["all_attempts"]),
                new_all=rate_cell(clean["meta_full_sun"], den["all_attempts"]),
                old_rec=rate_cell(
                    old["meta_full_sun"], den["reconstructed_exact_legacy"]
                ),
                new_rec=rate_cell(
                    clean["meta_full_sun"], den["reconstructed_exact_legacy"]
                ),
                up=paired["meta_0_to_1"],
                down=paired["meta_1_to_0"],
            )
        )
    lines.extend(
        [
            "",
            "## Hull coverage and shift",
            "",
            "| Panel | Arm | Repeat | Stage | Old unknown | Clean hull unknown | Relax unknown | Old unknown resolved | Mean ΔE_hull (clean-old) |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for report in ordered:
        cell = report["cell"]
        old = report["old"]
        clean = report["clean"]
        paired = report["paired_old_to_clean"]
        shift = paired["e_hull_clean_minus_old_ev_per_atom"]
        mean = "NA" if shift["mean"] is None else f"{float(shift['mean']):.6f}"
        lines.append(
            f"| {cell['panel']} | {cell['arm']} | {cell['repeat']} | "
            f"{cell['stage']} | {old['hull_unknown']} | {clean['hull_unknown']} | "
            f"{clean['relaxation_unknown']} | {paired['old_unknown_now_hull_resolved']} | {mean} |"
        )
    lines.extend(
        [
            "",
            "## Paired V4 inference",
            "",
            "All four V4 comparisons retain the frozen repeat/ordinal pairing. "
            "The terminal JSON contains per-repeat exact McNemar tests and "
            "50,000-draw hierarchical paired bootstraps for strict and meta endpoints.",
            "",
            "The historical refined-256 panel is descriptive because it has no "
            "paired B3 arm in this repair run.",
            "",
        ]
    )
    markdown = run_root / "RESULTS_COMPLETE.md"
    with markdown.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
        handle.flush()
        os.fsync(handle.fileno())
    status = run_root / "status"
    status.mkdir(exist_ok=True)
    (status / "assembly_SUCCESS").touch(exist_ok=False)
    print(
        canonical_json(
            {
                "engineering_status": "complete",
                "cells": len(ordered),
                "terminal_report": str(run_root / "terminal_report.json"),
                "markdown": str(markdown),
            }
        )
    )


if __name__ == "__main__":
    main()
