#!/usr/bin/env python3
"""Finalize nine post-refine cells with the byte-verified official-S core."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from protocol import (
    ContractError,
    DENOMINATOR,
    canonical_json,
    canonical_sha256,
    cell_specs,
    identity,
    load_upstream_cells,
    read_json,
    read_jsonl,
    require_source_manifest,
    sha256_file,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from run_frozen_official import FROZEN_SOURCE, verify_frozen_source


BASE_FINALIZER_SHA256 = (
    "17ac3657b268fcb690767d9168593cfa6c17141a09612995f66a4f758c102ff4"
)


def load_official_core() -> dict[str, Any]:
    """Load only proven definitions before the frozen finalizer's hardcoded main."""

    verify_frozen_source()
    finalizer = FROZEN_SOURCE / "finalize_official.py"
    if sha256_file(finalizer) != BASE_FINALIZER_SHA256:
        raise ContractError("proven official finalizer changed")
    text = finalizer.read_text(encoding="utf-8")
    prefix, anchor, _ = text.partition("\ndef main() -> None:\n")
    if not anchor:
        raise ContractError("proven finalizer main anchor changed")
    namespace: dict[str, Any] = {
        "__name__": "_h1a2_proven_official_core_",
        "__file__": str(finalizer),
        "__package__": None,
        "__cached__": None,
    }
    exec(compile(prefix, str(finalizer), "exec"), namespace)
    required = (
        "load_resolved",
        "load_unresolved",
        "evaluate_cell",
        "endpoint_map",
        "exact_mcnemar",
        "rate_cell",
        "flatten_values",
    )
    if any(not callable(namespace.get(name)) for name in required):
        raise ContractError("proven official core definitions changed")
    return namespace


def hierarchical_rate_bootstrap(
    vectors: list[list[bool]], draws: int, seed: int
) -> dict[str, Any]:
    if draws <= 0 or not vectors or any(len(values) != DENOMINATOR for values in vectors):
        raise ContractError("invalid post-only hierarchical bootstrap input")
    rng = np.random.default_rng(seed)
    blocks = len(vectors)
    selected = rng.integers(0, blocks, size=(draws, blocks))
    block_rates = np.empty((draws, blocks), dtype=np.float64)
    probabilities = [float(np.mean(np.asarray(values, dtype=np.int8))) for values in vectors]
    for position in range(blocks):
        for block_index, probability in enumerate(probabilities):
            mask = selected[:, position] == block_index
            count = int(mask.sum())
            if count:
                block_rates[mask, position] = rng.binomial(
                    DENOMINATOR, probability, size=count
                ) / DENOMINATOR
    samples = block_rates.mean(axis=1)
    estimate = float(np.mean(probabilities))
    return {
        "draws": draws,
        "seed": seed,
        "cohort_or_process_blocks": blocks,
        "estimate": estimate,
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }


def endpoint_vector(rows: list[dict[str, Any]], metric: str) -> list[bool]:
    ordered = sorted(rows, key=lambda row: int(row["generation_ordinal"]))
    if [int(row["generation_ordinal"]) for row in ordered] != list(range(DENOMINATOR)):
        raise ContractError("post-only endpoint ordinal coverage changed")
    return [bool((row.get("metrics") or {}).get(metric)) for row in ordered]


def controlled_mcnemar(
    attempts_by_cell: dict[str, list[dict[str, Any]]],
    endpoint_map: Callable[..., dict[int, tuple[bool, bool]]],
    exact_mcnemar: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "left": "historical_h1a2_dlm_control_B0_D1",
        "right": "R03_D2_topology_match",
        "shared_left_control_across_four_right_processes": True,
        "pooled_independence_claim": False,
        "metrics": {},
    }
    left_rows = attempts_by_cell["h1a2_b0_d1_once"]
    for metric in ("strict_full_sun", "meta_full_sun"):
        left = endpoint_map(left_rows, metric)
        reports = []
        for repeat in range(4):
            right = endpoint_map(attempts_by_cell[f"topology_repeat_{repeat}"], metric)
            for policy in ("fixed_all256", "paired_mp_known"):
                ordinals = list(range(DENOMINATOR))
                if policy == "paired_mp_known":
                    ordinals = [
                        ordinal
                        for ordinal in ordinals
                        if left[ordinal][1] and right[ordinal][1]
                    ]
                if not ordinals:
                    raise ContractError("no controlled H1-A2/R03 paired ordinals")
                left_values = [left[ordinal][0] for ordinal in ordinals]
                right_values = [right[ordinal][0] for ordinal in ordinals]
                raw = exact_mcnemar(left_values, right_values)
                reports.append(
                    {
                        "topology_repeat": repeat,
                        "denominator_policy": policy,
                        "paired_denominator": len(ordinals),
                        "mp_unknown_pairs_excluded": DENOMINATOR - len(ordinals),
                        "h1a2_numerator": sum(left_values),
                        "r03_numerator": sum(right_values),
                        "r03_minus_h1a2": (
                            sum(right_values) - sum(left_values)
                        )
                        / len(ordinals),
                        "h1a2_only": int(raw["left_pre_only"]),
                        "r03_only": int(raw["right_post_only"]),
                        "discordant": int(raw["discordant"]),
                        "two_sided_exact_p": float(raw["two_sided_exact_p"]),
                    }
                )
        output["metrics"][metric] = reports
    return output


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
    if not (run_root / "status/preliminary_assembly_SUCCESS").is_file():
        raise ContractError("preliminary assembly is incomplete")
    if not (run_root / "official_mp_cache/completion_SUCCESS").is_file():
        raise ContractError("official MP cache is incomplete")
    if (run_root / "official_results").exists() or (
        run_root / "RESULTS_COMPLETE.md"
    ).exists():
        raise FileExistsError("official final result already exists")

    core = load_official_core()
    load_resolved = core["load_resolved"]
    load_unresolved = core["load_unresolved"]
    evaluate_cell = core["evaluate_cell"]
    endpoint_map = core["endpoint_map"]
    exact_mcnemar = core["exact_mcnemar"]
    rate_cell = core["rate_cell"]
    flatten_values = core["flatten_values"]

    input_manifest_path = run_root / "inputs/input_manifest.json"
    input_manifest = read_json(input_manifest_path)
    cells = list(input_manifest["cells"])
    expected_cells = cell_specs(config)
    if (
        input_manifest.get("evaluated_stage") != "post_model494_only"
        or input_manifest.get("pre_refine_evaluated") is not False
        or len(cells) != len(expected_cells)
        or [row["cell_id"] for row in cells]
        != [row["cell_id"] for row in expected_cells]
    ):
        raise ContractError("official post-only input cell map changed")

    cache_root = run_root / "official_mp_cache"
    cache_manifest_path = cache_root / "completion_manifest.json"
    cache_path = cache_root / "official_slim_cache.jsonl"
    unresolved_path = cache_root / "unresolved_chemsys.jsonl"
    cache_manifest = read_json(cache_manifest_path)
    spec = config["official_mp"]
    if (
        cache_manifest.get("query_status")
        != "complete_with_explicit_hull_unknown"
        or cache_manifest.get("query_method") != spec["query_method"]
        or cache_manifest.get("compatible_only") is not True
        or cache_manifest.get("thermo_type") != "GGA_GGA+U"
        or cache_manifest.get("unresolved_policy") != spec["unresolved_policy"]
        or cache_manifest.get("historical_or_august_polluted_cache_rows_reused")
        != 0
        or sha256_file(cache_path)
        != cache_manifest["outputs"]["slim_evaluation_cache"]["sha256"]
        or sha256_file(unresolved_path)
        != cache_manifest["outputs"]["unresolved_chemsys"]["sha256"]
    ):
        raise ContractError("official incremental cache contract changed")
    resolved = load_resolved(cache_path)
    unresolved = load_unresolved(unresolved_path)
    wanted = {
        str(row["chemsys"])
        for row in read_jsonl(run_root / "inputs/wanted_chemsys.jsonl")
    }
    if wanted - set(resolved) - set(unresolved) or set(resolved) & set(unresolved):
        raise ContractError("official cache coverage or disjointness changed")

    upstream_terminal, upstream_cells = load_upstream_cells(config)
    upstream_by_id = {str(row["cell_id"]): row for row in upstream_cells}
    if len(upstream_by_id) != 9:
        raise ContractError("upstream post-only cell inventory changed")
    upstream_root = Path(config["upstream_generation"]["run_root"])
    upstream_terminal_path = (
        upstream_root / config["upstream_generation"]["terminal_report"]
    )
    evaluation_cells_path = (
        upstream_root / config["upstream_generation"]["evaluation_registry"]
    )

    preparing = run_root / f".official_results.preparing.{os.getpid()}"
    failed = run_root / f".official_results.FAILED.{os.getpid()}"
    preparing.mkdir(parents=True, exist_ok=False)
    try:
        output_cells = preparing / "cells"
        output_cells.mkdir()
        reports: list[dict[str, Any]] = []
        attempts_by_cell: dict[str, list[dict[str, Any]]] = {}
        diagrams: dict[str, Any] = {}
        for cell in cells:
            cell_id = str(cell["cell_id"])
            inventory = upstream_by_id[cell_id]
            generation_path = Path(inventory["generation_jsonl"])
            if sha256_file(generation_path) != inventory["generation_sha256"]:
                raise ContractError("upstream generation identity changed")
            generation_rows = read_jsonl(generation_path)
            generation_report_path = generation_path.parent / "generation_report.json"
            generation_report = read_json(generation_report_path)
            direct_path = Path(inventory["direct_report"])
            if sha256_file(direct_path) != inventory["direct_report_sha256"]:
                raise ContractError("upstream Direct report identity changed")
            direct_report = read_json(direct_path)
            succeeded = sum(row.get("status") == "succeeded" for row in generation_rows)
            cell["generation_succeeded"] = succeeded
            cell["generation_failed"] = DENOMINATOR - succeeded
            report, attempts = evaluate_cell(
                cell,
                resolved,
                unresolved,
                diagrams,
                float(spec["strict_threshold_ev_per_atom"]),
                float(spec["meta_threshold_ev_per_atom"]),
            )
            report["cell"]["process_repeat"] = cell.get("process_repeat")
            report["cell"]["body"] = cell.get("body")
            report["cell"]["schedule"] = cell.get("schedule")
            report["generation_report"] = generation_report
            report["generation_report_identity"] = identity(generation_report_path)
            report["direct_report"] = direct_report
            report["direct_report_identity"] = identity(direct_path)
            reports.append(report)
            attempts_by_cell[cell_id] = attempts
            cell_root = output_cells / f"{int(cell['cell_index']):03d}_{cell_id}"
            cell_root.mkdir()
            write_jsonl_exclusive(
                cell_root / "attempt_results_official.jsonl", attempts
            )
            write_json_exclusive(cell_root / "cell_report.json", report)
            (cell_root / "_SUCCESS").touch(exist_ok=False)

        draws = int(config["inference"]["post_only_panel_bootstrap_draws"])
        seed = int(config["inference"]["bootstrap_seed"])
        panel_statistics: dict[str, Any] = {}
        for offset, panel in enumerate(("fresh_cohort", "topology_match")):
            panel_ids = [
                report["cell"]["cell_id"]
                for report in reports
                if report["cell"]["panel"] == panel
            ]
            if len(panel_ids) != 4:
                raise ContractError(f"post-only panel shape changed: {panel}")
            metric_reports: dict[str, Any] = {}
            for metric_offset, metric in enumerate(
                ("strict_full_sun", "meta_full_sun")
            ):
                vectors = [
                    endpoint_vector(attempts_by_cell[cell_id], metric)
                    for cell_id in panel_ids
                ]
                metric_reports[metric] = {
                    "cell_ids": panel_ids,
                    "counts": [sum(vector) for vector in vectors],
                    "fixed_all256_rates": [
                        sum(vector) / DENOMINATOR for vector in vectors
                    ],
                    "hierarchical_rate_bootstrap": hierarchical_rate_bootstrap(
                        vectors,
                        draws,
                        seed + 10 * offset + metric_offset,
                    ),
                }
            panel_statistics[panel] = metric_reports

        controlled = controlled_mcnemar(
            attempts_by_cell, endpoint_map, exact_mcnemar
        )
        benchmark = config["historical_official_refined256_benchmark"]
        topology_strict = panel_statistics["topology_match"]["strict_full_sun"][
            "counts"
        ]
        topology_meta = panel_statistics["topology_match"]["meta_full_sun"][
            "counts"
        ]
        historical_comparison = {
            "role": "descriptive_recovery_target_only",
            "current_topology_strict_counts": topology_strict,
            "historical_strict_counts": benchmark["strict_fixed_all_counts"],
            "strict_count_differences": [
                int(current) - int(reference)
                for current, reference in zip(
                    topology_strict, benchmark["strict_fixed_all_counts"]
                )
            ],
            "current_topology_meta_counts": topology_meta,
            "historical_meta_counts": benchmark["meta_fixed_all_counts"],
            "meta_count_differences": [
                int(current) - int(reference)
                for current, reference in zip(
                    topology_meta, benchmark["meta_fixed_all_counts"]
                )
            ],
        }

        terminal = {
            "schema": "h1a2_retrained_postonly_sun256_official_terminal_v1",
            "engineering_status": "complete",
            "scientific_status": "report_all_results_without_directional_gate",
            "source_manifest_sha256": args.source_manifest_sha256,
            "evaluated_stage": "post_model494_only",
            "pre_refine_evaluated": False,
            "thermo_contract": {
                "query_method": spec["query_method"],
                "compatible_only": True,
                "thermo_type": "GGA_GGA+U",
                "base_source_query_was_fresh_empty_cache": True,
                "historical_or_august_polluted_cache_rows_reused": 0,
                "new_mp_queries": int(cache_manifest["new_mp_queries"]),
                "new_resolved_count": int(cache_manifest["new_resolved_count"]),
                "new_unresolved_count": int(cache_manifest["new_unresolved_count"]),
                "wanted_query_count": int(cache_manifest["wanted_query_count"]),
                "unresolved_policy": spec["unresolved_policy"],
            },
            "cell_reports": reports,
            "post_only_panel_statistics": panel_statistics,
            "controlled_h1a2_dlm_vs_r03_topology_mcnemar": controlled,
            "historical_topology_recovery_comparison": historical_comparison,
            "planner_deep_distribution_audit": upstream_terminal[
                "planner_deep_distribution_audit"
            ],
            "planner_topology_match_audit": upstream_terminal[
                "planner_topology_match_audit"
            ],
            "historical_official_refined256_benchmark": benchmark,
            "artifacts": {
                "input_manifest": identity(input_manifest_path),
                "official_cache_manifest": identity(cache_manifest_path),
                "official_slim_cache": identity(cache_path),
                "official_unresolved_chemsys": identity(unresolved_path),
                "upstream_generation_terminal": identity(upstream_terminal_path),
                "upstream_evaluation_cells": identity(evaluation_cells_path),
                "proven_official_finalizer": identity(
                    FROZEN_SOURCE / "finalize_official.py"
                ),
            },
        }
        write_json_exclusive(preparing / "terminal_report.json", terminal)

        ordered = sorted(reports, key=lambda row: int(row["cell"]["cell_index"]))
        lines = [
            "# Retrained H1-A2 recovery — post-model494 official S.U.N.",
            "",
            "Only `post_model494` / refine800 structures are evaluated. Pre-refine structures are pipeline intermediates and have no Direct, CrysLLMGen or S.U.N. row in this report.",
            "",
            "Reconstruction, novelty, uniqueness, refined structures and CHGNet relaxed energies are frozen. Stability alone is replaced with the byte-verified official `MPRester.get_entries_in_chemsys(..., compatible_only=True, thermo_types=GGA_GGA+U)` phase-diagram core. Unresolved MP systems are explicit `hull_unknown`: they remain failures in fixed-all denominators and are excluded only from columns labelled `skip MP unknown`.",
            "",
            "## Complete CrysLLMGen / generation counts",
            "",
            "| Cell | Panel | Body/schedule | Attempts | Generated | Failed | Reconstructed | Novel | Unique | N+U | Relax unknown | Hull unknown |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for report in ordered:
            cell = report["cell"]
            generation = report["generation"]
            crys = report["frozen_crysllmgen"]
            den = report["denominators"]
            lines.append(
                f"| {cell['cell_id']} | {cell['panel']} | {cell['body']}/{cell['schedule']} | "
                f"{generation['attempts']} | {generation['succeeded']} | {generation['failed']} | "
                f"{crys['reconstructed']} | {crys['novel']} | {crys['unique']} | "
                f"{crys['novel_unique']} | {den['relaxation_unknown']} | {den['hull_unknown']} |"
            )
        lines.extend(
            [
                "",
                "## Complete frozen CrysLLMGen detailed counts",
                "",
                "Every field below comes from the frozen preliminary `attempt_summary.counts`; official-MP replacement does not alter it.",
                "",
                "| Cell | Panel | Count field | Value |",
                "|---|---|---|---:|",
            ]
        )
        for report in ordered:
            cell = report["cell"]
            for metric, value in flatten_values(
                report["frozen_crysllmgen"]["all_preliminary_counts"]
            ):
                lines.append(
                    f"| {cell['cell_id']} | {cell['panel']} | `{metric}` | `{value}` |"
                )

        for metric, title in (
            ("strict_full_sun", "Strict S.U.N. (E_hull = 0 eV/atom)"),
            ("meta_full_sun", "Meta-S.U.N. (E_hull ≤ 0.1 eV/atom)"),
        ):
            lines.extend(
                [
                    "",
                    f"## {title}",
                    "",
                    "| Cell | Panel | Fixed all-256 | Exact legacy reconstructed | All skip MP unknown | Reconstructed skip MP unknown | Evaluated N+U diagnostic | Legacy-cache diagnostic | 0→1 | 1→0 |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for report in ordered:
                cell = report["cell"]
                den = report["denominators"]
                official = report["official"]
                flips = report["preliminary_to_official"]
                numerator = int(official[metric])
                old_key = (
                    "strict_full_sun"
                    if metric == "strict_full_sun"
                    else "meta_full_sun"
                )
                old = int(report["preliminary_legacy_stability_diagnostic"][old_key])
                prefix = "strict" if metric == "strict_full_sun" else "meta"
                lines.append(
                    f"| {cell['cell_id']} | {cell['panel']} | "
                    f"{rate_cell(numerator, den['all_attempts_fixed'])} | "
                    f"{rate_cell(numerator, den['reconstructed_exact_legacy'])} | "
                    f"{rate_cell(numerator, den['all_attempts_skip_mp_unknown'])} | "
                    f"{rate_cell(numerator, den['reconstructed_skip_mp_unknown'])} | "
                    f"{rate_cell(numerator, den['hull_evaluated_novel_unique'])} | "
                    f"{rate_cell(old, den['all_attempts_fixed'])} | "
                    f"{flips[prefix + '_0_to_1']} | {flips[prefix + '_1_to_0']} |"
                )

        lines.extend(
            [
                "",
                "## Hull coverage and official shift",
                "",
                "| Cell | Panel | N+U | Evaluated | Relax unknown | Hull unknown | Preliminary unknown now resolved | Mean ΔE_hull official−legacy |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for report in ordered:
            cell = report["cell"]
            den = report["denominators"]
            flips = report["preliminary_to_official"]
            mean = flips["e_hull_official_minus_preliminary_ev_per_atom"]["mean"]
            lines.append(
                f"| {cell['cell_id']} | {cell['panel']} | {den['novel_unique']} | "
                f"{den['hull_evaluated_novel_unique']} | {den['relaxation_unknown']} | "
                f"{den['hull_unknown']} | {flips['preliminary_unknown_now_officially_resolved']} | "
                f"{'NA' if mean is None else f'{float(mean):.6f}'} |"
            )

        lines.extend(["", "## Post-only panel statistics", ""])
        for panel, metrics in panel_statistics.items():
            lines.append(f"### {panel}")
            lines.append("")
            lines.append("| Endpoint | Counts / 256 | Mean rate | 50k hierarchical 95% CI |")
            lines.append("|---|---|---:|---:|")
            for metric, values in metrics.items():
                boot = values["hierarchical_rate_bootstrap"]
                lines.append(
                    f"| {metric} | {values['counts']} | {boot['estimate']:.6f} | "
                    f"[{boot['ci95'][0]:.6f}, {boot['ci95'][1]:.6f}] |"
                )
            lines.append("")

        lines.extend(
            [
                "### Topology recovery target",
                "",
                "| Endpoint | Current topology counts | Historical clean refined256 counts | Current−historical |",
                "|---|---|---|---|",
                f"| strict | {historical_comparison['current_topology_strict_counts']} | {historical_comparison['historical_strict_counts']} | {historical_comparison['strict_count_differences']} |",
                f"| meta | {historical_comparison['current_topology_meta_counts']} | {historical_comparison['historical_meta_counts']} | {historical_comparison['meta_count_differences']} |",
                "",
                "This is a descriptive recovery target, not a new causal estimate.",
                "",
                "### H1-A2 B0/D1 versus R03 D2 topology (same planner cohort)",
                "",
                "| Endpoint | Policy | R03 repeat | Paired N | MP-unknown pairs excluded | H1-A2 | R03 | R03−H1-A2 | H1-A2 only | R03 only | Exact p |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for metric, metric_rows in controlled["metrics"].items():
            for row in metric_rows:
                lines.append(
                    f"| {metric} | {row['denominator_policy']} | {row['topology_repeat']} | "
                    f"{row['paired_denominator']} | {row['mp_unknown_pairs_excluded']} | "
                    f"{row['h1a2_numerator']} | {row['r03_numerator']} | "
                    f"{row['r03_minus_h1a2']:.6f} | {row['h1a2_only']} | "
                    f"{row['r03_only']} | {row['two_sided_exact_p']:.8g} |"
                )
        lines.extend(
            [
                "",
                "The same single H1-A2 control vector is reused against four R03 topology processes, so no pooled independent-block claim is made.",
                "",
                "## Direct metrics (post-model494 only; descriptive)",
                "",
                "| Cell | Panel | Metric | Value |",
                "|---|---|---|---|",
            ]
        )
        for report in ordered:
            cell = report["cell"]
            for metric, value in flatten_values(report["direct_report"]):
                lines.append(
                    f"| {cell['cell_id']} | {cell['panel']} | `{metric}` | `{value}` |"
                )
        lines.extend(
            [
                "",
                "## Planner-distribution evidence",
                "",
                f"- Deep four-fresh-cohort audit: `{canonical_json(upstream_terminal['planner_deep_distribution_audit'])}`",
                f"- Seed17 topology-match audit: `{canonical_json(upstream_terminal['planner_topology_match_audit'])}`",
                "",
            ]
        )

        markdown_path = preparing / "RESULTS_COMPLETE.md"
        with markdown_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        (preparing / "_SUCCESS").touch(exist_ok=False)
        final = run_root / "official_results"
        preparing.rename(final)
        os.link(final / "terminal_report.json", run_root / "terminal_report.json")
        os.link(final / "RESULTS_COMPLETE.md", run_root / "RESULTS_COMPLETE.md")
        (run_root / "status/finalization_SUCCESS").touch(exist_ok=False)
    except Exception:
        if preparing.exists():
            shutil.move(str(preparing), str(failed))
        raise
    print(
        canonical_json(
            {
                "engineering_status": "complete",
                "cells": len(cells),
                "evaluated_stage": "post_model494_only",
                "strict_fixed": [
                    row["official"]["strict_full_sun"] for row in reports
                ],
                "results": str(run_root / "RESULTS_COMPLETE.md"),
            }
        )
    )


if __name__ == "__main__":
    main()
