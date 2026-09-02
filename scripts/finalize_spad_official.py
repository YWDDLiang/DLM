#!/usr/bin/env python3
"""Join one fresh MP query to the final SPAD prospective comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
EVAL_RUNTIME = ROOT / "eval_runtime"
if str(EVAL_RUNTIME) not in sys.path:
    sys.path.insert(0, str(EVAL_RUNTIME))

import protocol  # noqa: E402


def load_common() -> Any:
    path = EVAL_RUNTIME / "finalize_official.py"
    spec = importlib.util.spec_from_file_location("spad_official_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return protocol.read_jsonl(path)


def find_plan(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("plan_state", "source_plan_state", "r5_plan_state", "identity"):
        value = row.get(key)
        if isinstance(value, Mapping) and value.get("elements"):
            return value
    for key in ("source_row", "record", "candidate"):
        value = row.get(key)
        if isinstance(value, Mapping):
            nested = find_plan(value)
            if nested is not None:
                return nested
    return None


def chemsys_scopes(cohort: Path, mp20_train: Path) -> dict[str, set[int]]:
    train_chemsys: set[str] = set()
    for row in read_jsonl(mp20_train):
        plan = find_plan(row)
        if plan is not None:
            train_chemsys.add("-".join(sorted({str(value) for value in plan["elements"]})))
    ledger = read_jsonl(cohort / "ledger.jsonl")
    if {int(row["sample_idx"]) for row in ledger} != set(range(256)):
        raise ValueError("prospective cohort ledger does not cover fixed256")
    scopes = {"all": set(range(256)), "seen_chemsys": set(), "unseen_chemsys": set()}
    for row in ledger:
        target = "seen_chemsys" if str(row["chemsys"]) in train_chemsys else "unseen_chemsys"
        scopes[target].add(int(row["sample_idx"]))
    return scopes


def quantiles(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {"count": 0, "mean": None, "median": None, "q10": None, "q90": None}

    def at(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "q10": at(0.10),
        "q90": at(0.90),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def composition_average(rows_by_stream: Mapping[int, Sequence[Mapping[str, Any]]], field: str) -> dict[int, float]:
    indexed = {
        stream: {int(row["ordinal"]): row for row in rows}
        for stream, rows in rows_by_stream.items()
    }
    if any(set(rows) != set(range(256)) for rows in indexed.values()):
        raise ValueError("official rows do not cover fixed256")
    return {
        index: statistics.fmean(float(indexed[stream][index][field]) for stream in sorted(indexed))
        for index in range(256)
    }


def paired_effect(
    control: Mapping[int, float],
    candidate: Mapping[int, float],
    *,
    label: str,
) -> dict[str, Any]:
    deltas = {
        index: candidate[index] - control[index]
        for index in sorted(set(control) & set(candidate))
    }
    values = list(deltas.values())
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    replicates = 10_000
    means = sorted(
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(replicates)
    )
    return {
        "compositions_requested": 256,
        "compositions_observed": len(values),
        "mean_delta": statistics.fmean(values),
        "median_delta": statistics.median(values),
        "fraction_higher": sum(value > 0.0 for value in values) / len(values),
        "bootstrap": {
            "unit": "composition",
            "replicates": replicates,
            "seed": seed,
            "ci95_lower": means[249],
            "ci95_upper": means[9749],
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# SPAD prospective official S.U.N.",
        "",
        "The table uses all 256 requested compositions per stream; the two streams are averaged within composition.",
        "",
        "| Arm | Endpoint | Direct | Hull known | N/U/NU | Strict stable | Meta stable | Strict S.U.N. | Meta S.U.N. |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["aggregates"]:
        lines.append(
            f"| {row['arm']} | {row['endpoint']} | {row['direct_joint']}/512 | "
            f"{row['hull_known_reconstructed']}/512 | {row['novel']}/{row['unique']}/{row['novel_unique']} | "
            f"{row['strict_stable']}/512 | {row['meta_stable']}/512 | "
            f"{100*row['strict_sun_rate']:.2f}% | {100*row['meta_sun_rate']:.2f}% |"
        )
    headline = report["headline"]
    lines.extend(
        [
            "",
            "## Headline",
            "",
            f"BS refined Strict S.U.N. = {100*headline['strict_sun_rate']:.2f}% and "
            f"Meta S.U.N. = {100*headline['meta_sun_rate']:.2f}%.",
            f"Targets 10%/50%: Strict={'met' if headline['strict_target_met'] else 'not met'}, "
            f"Meta={'met' if headline['meta_target_met'] else 'not met'}.",
            (
                "Chemsys split: "
                f"seen={headline['chemsys_scopes']['seen_chemsys']['compositions']}, "
                f"unseen={headline['chemsys_scopes']['unseen_chemsys']['compositions']}; "
                "the exact-composition overlap with MP20 train is zero."
            ),
            "",
            "## Paired effects",
            "",
        ]
    )
    for name, effect in report["paired_effects"].items():
        ci = effect["bootstrap"]
        lines.append(
            f"- {name}: {100*effect['mean_delta']:.2f} pp, 95% composition-bootstrap "
            f"[{100*ci['ci95_lower']:.2f}, {100*ci['ci95_upper']:.2f}] pp."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--mp20-train", type=Path, required=True)
    parser.add_argument("--arms", default="B0,BC,BS")
    args = parser.parse_args()
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if not (args.eval_run / "_OFFLINE_SUCCESS").is_file():
        raise FileNotFoundError("prospective offline evaluation is incomplete")
    if not (args.official_run / "official_mp_cache/completion_manifest.json").is_file():
        raise FileNotFoundError("fresh official query is incomplete")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    common = load_common()
    scopes = chemsys_scopes(args.cohort, args.mp20_train)
    cache = args.official_run / "official_mp_cache"
    phase_diagrams = common._phase_diagrams(cache / "official_slim_cache.jsonl")
    unresolved = {
        str(row["chemsys"])
        for row in read_jsonl(cache / "unresolved_chemsys.jsonl")
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)

    rows: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    cell_reports: list[dict[str, Any]] = []
    for arm in arms:
        for endpoint in ("raw", "refined"):
            for stream in (17, 18):
                cell = args.eval_run / arm / f"stream{stream}" / endpoint
                cell_id = f"{arm}_{endpoint}_s{stream}"
                official_rows, report = common._evaluate_cell(
                    cell_id=cell_id,
                    labels_path=cell / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl",
                    generation_path=cell / "generation/generation.jsonl",
                    direct_path=cell / "evaluation/direct/report.json",
                    phase_diagrams=phase_diagrams,
                    unresolved=unresolved,
                    output_dir=args.output_dir / "cells" / cell_id,
                )
                rows[(arm, endpoint, stream)] = official_rows
                cell_reports.append(report)

    aggregates: list[dict[str, Any]] = []
    composition_metrics: dict[tuple[str, str, str], dict[int, float]] = {}
    for arm in arms:
        for endpoint in ("raw", "refined"):
            selected = [
                report for report in cell_reports
                if report["cell_id"].startswith(f"{arm}_{endpoint}_")
            ]
            counts = {
                key: sum(int(report["counts"][key]) for report in selected)
                for key in (
                    "reconstructed", "novel", "unique_representatives", "novel_unique",
                    "chgnet_energy_known", "hull_known_reconstructed",
                    "strict_stable_all_hull_known", "meta_stable_all_hull_known",
                    "strict_sun", "meta_sun",
                )
            }
            all_rows = [row for stream in (17, 18) for row in rows[(arm, endpoint, stream)]]
            hull = [
                float(row["official_e_above_hull"])
                for row in all_rows if row.get("official_e_above_hull") is not None
            ]
            hull_nu = [
                float(row["official_e_above_hull"])
                for row in all_rows
                if row.get("official_e_above_hull") is not None and row.get("novel_unique") is True
            ]
            for metric in ("strict_sun", "meta_sun"):
                composition_metrics[(arm, endpoint, metric)] = composition_average(
                    {stream: rows[(arm, endpoint, stream)] for stream in (17, 18)}, metric
                )
            scope_rates = {}
            for scope, indices in scopes.items():
                scoped_rows = [
                    row for row in all_rows if int(row["ordinal"]) in indices
                ]
                denominator = len(indices) * 2
                scope_rates[scope] = {
                    "compositions": len(indices),
                    "attempts": denominator,
                    "strict_sun": sum(row["strict_sun"] for row in scoped_rows),
                    "meta_sun": sum(row["meta_sun"] for row in scoped_rows),
                    "strict_sun_rate": (
                        sum(row["strict_sun"] for row in scoped_rows) / denominator
                        if denominator else None
                    ),
                    "meta_sun_rate": (
                        sum(row["meta_sun"] for row in scoped_rows) / denominator
                        if denominator else None
                    ),
                }
            aggregates.append(
                {
                    "arm": arm,
                    "endpoint": endpoint,
                    "requested": 512,
                    "direct_joint": sum(int(report["direct"]["joint_valid"]) for report in selected),
                    "novel": counts["novel"],
                    "unique": counts["unique_representatives"],
                    "novel_unique": counts["novel_unique"],
                    "reconstructed": counts["reconstructed"],
                    "chgnet_energy_known": counts["chgnet_energy_known"],
                    "hull_known_reconstructed": counts["hull_known_reconstructed"],
                    "strict_stable": counts["strict_stable_all_hull_known"],
                    "meta_stable": counts["meta_stable_all_hull_known"],
                    "strict_sun": counts["strict_sun"],
                    "meta_sun": counts["meta_sun"],
                    "strict_sun_rate": counts["strict_sun"] / 512,
                    "meta_sun_rate": counts["meta_sun"] / 512,
                    "hull_all": quantiles(hull),
                    "hull_novel_unique": quantiles(hull_nu),
                    "chemsys_scopes": scope_rates,
                }
            )

    paired_effects: dict[str, Any] = {}
    for comparator in ("B0", "BC"):
        if comparator not in arms or "BS" not in arms:
            continue
        for endpoint in ("raw", "refined"):
            for metric in ("strict_sun", "meta_sun"):
                name = f"{endpoint}:{metric}:BS-minus-{comparator}"
                paired_effects[name] = paired_effect(
                    composition_metrics[(comparator, endpoint, metric)],
                    composition_metrics[("BS", endpoint, metric)],
                    label=f"spad:{name}",
                )

    headline = next(
        row for row in aggregates if row["arm"] == "BS" and row["endpoint"] == "refined"
    )
    headline = {
        **headline,
        "strict_target_met": headline["strict_sun_rate"] >= 0.10,
        "meta_target_met": headline["meta_sun_rate"] >= 0.50,
    }
    report = {
        "schema": "spad_prospective_official_final_v1",
        "eval_run": str(args.eval_run.resolve()),
        "official_run": str(args.official_run.resolve()),
        "arms": list(arms),
        "denominator_per_stream": 256,
        "streams_averaged_within_composition": True,
        "prospective_exact_composition_overlap_with_mp20_train": 0,
        "chemsys_scope_sizes": {key: len(value) for key, value in scopes.items()},
        "cell_reports": cell_reports,
        "aggregates": aggregates,
        "paired_effects": paired_effects,
        "headline": headline,
    }
    (args.output_dir / "SPAD_PROSPECTIVE_OFFICIAL_FINAL.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "SPAD_PROSPECTIVE_OFFICIAL_FINAL.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    with (args.output_dir / "SPAD_PROSPECTIVE_AGGREGATES.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "arm", "endpoint", "requested", "direct_joint", "reconstructed",
            "novel", "unique", "novel_unique", "chgnet_energy_known",
            "hull_known_reconstructed", "strict_stable", "meta_stable",
            "strict_sun", "meta_sun", "strict_sun_rate", "meta_sun_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in aggregates:
            writer.writerow({key: row[key] for key in fields})
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps({"headline": headline}, sort_keys=True))


if __name__ == "__main__":
    main()
