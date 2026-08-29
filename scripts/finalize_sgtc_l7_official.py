#!/usr/bin/env python3
"""Finalize the one-time official-hull SGTC requested-1000 L7 gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SEED = 18
ARMS = ("base", "g0_all", "g1_strict")
ATTEMPTS = 1000
RATE_KEYS = (
    "body_rate",
    "direct_joint_rate",
    "novel_rate",
    "unique_rate",
    "strict_attempt_rate",
    "meta_attempt_rate",
    "strict_retention",
    "meta_retention",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rate(value: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else value / denominator


def absolute_gate(summary: Mapping[str, Any]) -> dict[str, bool]:
    gate = {
        "strict_at_least_10pct": float(summary["strict_attempt_rate"]) >= 0.10,
        "meta_at_least_50pct": float(summary["meta_attempt_rate"]) >= 0.50,
    }
    gate["eligible"] = all(gate.values())
    return gate


def secondary_floor_gate(delta: Mapping[str, float]) -> dict[str, bool]:
    gate = {
        "body_ge_minus_3pp": float(delta["body_rate"]) >= -0.03,
        "direct_ge_minus_3pp": float(delta["direct_joint_rate"]) >= -0.03,
        "novel_ge_minus_5pp": float(delta["novel_rate"]) >= -0.05,
        "unique_ge_minus_5pp": float(delta["unique_rate"]) >= -0.05,
        "strict_retention_ge_minus_10pp": float(delta["strict_retention"]) >= -0.10,
        "meta_retention_ge_minus_10pp": float(delta["meta_retention"]) >= -0.10,
    }
    gate["eligible"] = all(gate.values())
    return gate


def direction_gate(delta: Mapping[str, float]) -> dict[str, bool]:
    strict = float(delta["strict_attempt_rate"])
    meta = float(delta["meta_attempt_rate"])
    directional = (strict > 0.0 and meta >= -0.01) or (
        meta > 0.0 and strict >= -0.01
    )
    return {"strict_or_meta_positive_other_ge_minus_1pp": directional}


def paired_delta_summary(
    control: Sequence[bool], candidate: Sequence[bool]
) -> dict[str, Any]:
    if len(control) != len(candidate) or not control:
        raise ValueError("paired binary vectors must be non-empty and aligned")
    differences = [int(right) - int(left) for left, right in zip(control, candidate)]
    mean = sum(differences) / len(differences)
    if len(differences) == 1:
        standard_error = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in differences) / (
            len(differences) - 1
        )
        standard_error = math.sqrt(variance / len(differences))
    radius = 1.959963984540054 * standard_error
    return {
        "known_both": len(differences),
        "candidate_minus_control": mean,
        "wald95_lower": max(-1.0, mean - radius),
        "wald95_upper": min(1.0, mean + radius),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reconstructed = sum(row.get("reconstructed") is True for row in rows)
    strict_stable = sum(row.get("strict_stable") is True for row in rows)
    meta_stable = sum(row.get("meta_stable") is True for row in rows)
    strict_sun = sum(row.get("strict_sun") is True for row in rows)
    meta_sun = sum(row.get("meta_sun") is True for row in rows)
    return {
        "reconstructed": reconstructed,
        "novel": sum(row.get("novel") is True for row in rows),
        "unique": sum(row.get("unique_representative") is True for row in rows),
        "novel_unique": sum(row.get("novel_unique") is True for row in rows),
        "hull_known": sum(row.get("official_hull_status") == "known" for row in rows),
        "hull_unknown": sum(
            row.get("reconstructed") is True
            and row.get("official_hull_status") != "known"
            for row in rows
        ),
        "strict_stable": strict_stable,
        "strict_sun": strict_sun,
        "meta_stable": meta_stable,
        "meta_sun": meta_sun,
        "strict_attempt_rate": rate(strict_sun, len(rows)),
        "meta_attempt_rate": rate(meta_sun, len(rows)),
        "strict_retention": rate(strict_sun, strict_stable),
        "meta_retention": rate(meta_sun, meta_stable),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--generation-run", type=Path, required=True)
    parser.add_argument("--official-cache-run", type=Path, required=True)
    parser.add_argument("--eval-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.eval_runtime.resolve()))
    import protocol  # noqa: PLC0415
    from finalize_official import _evaluate_cell, _exact_mcnemar, _phase_diagrams  # noqa: PLC0415

    if protocol.DENOMINATOR != ATTEMPTS:
        raise RuntimeError("H1_ACTIVE_DENOMINATOR must be 1000")
    cache = args.official_cache_run / "official_mp_cache"
    if not (cache / "completion_SUCCESS").is_file():
        raise RuntimeError("official MP cache is incomplete")
    phase_diagrams = _phase_diagrams(cache / "official_slim_cache.jsonl")
    unresolved = {
        str(row["chemsys"])
        for row in protocol.read_jsonl(cache / "unresolved_chemsys.jsonl")
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    cells: list[dict[str, Any]] = []
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        cell = args.eval_run / f"seed{SEED}/{arm}"
        rows, direct = _evaluate_cell(
            cell_id=f"seed{SEED}_{arm}",
            labels_path=cell
            / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl",
            generation_path=cell / "generation/generation.jsonl",
            direct_path=cell / "evaluation/direct/report.json",
            phase_diagrams=phase_diagrams,
            unresolved=unresolved,
            output_dir=output / f"cells/seed{SEED}/{arm}",
        )
        rows_by_arm[arm] = rows
        body = read_json(
            args.generation_run / f"seed{SEED}/{arm}/body/SGTC_BODY_MANIFEST.json"
        )
        refine = read_json(
            args.generation_run / f"seed{SEED}/{arm}/refine/refinement_metrics.json"
        )
        summary = summarize_rows(rows)
        summary.update(
            {
                "seed": SEED,
                "arm": arm,
                "requested": ATTEMPTS,
                "parsed": int(body["parsed"]),
                "body": int(body["graphs"]),
                "refined": int(refine["num_proposals"]),
                "direct_comp": int(direct["direct"]["composition_valid"]),
                "direct_struct": int(direct["direct"]["structure_valid"]),
                "direct_joint": int(direct["direct"]["joint_valid"]),
            }
        )
        summary.update(
            {
                "body_rate": rate(summary["body"], ATTEMPTS),
                "direct_joint_rate": rate(summary["direct_joint"], ATTEMPTS),
                "novel_rate": rate(summary["novel"], summary["reconstructed"]),
                "unique_rate": rate(summary["unique"], summary["reconstructed"]),
            }
        )
        cells.append(summary)

    by_arm = {row["arm"]: row for row in cells}
    deltas = {
        f"{candidate}-{control}": {
            key: float(by_arm[candidate][key]) - float(by_arm[control][key])
            for key in RATE_KEYS
        }
        for candidate, control in (
            ("g0_all", "base"),
            ("g1_strict", "base"),
            ("g1_strict", "g0_all"),
        )
    }
    gates = {
        "g1_strict_absolute": absolute_gate(by_arm["g1_strict"]),
        "g1_strict_vs_base_floors": secondary_floor_gate(
            deltas["g1_strict-base"]
        ),
        "g1_strict_vs_g0_all_direction": direction_gate(
            deltas["g1_strict-g0_all"]
        ),
    }
    sgtc_l7_pass = (
        gates["g1_strict_absolute"]["eligible"]
        and gates["g1_strict_vs_base_floors"]["eligible"]
        and gates["g1_strict_vs_g0_all_direction"][
            "strict_or_meta_positive_other_ge_minus_1pp"
        ]
    )

    pairwise: dict[str, Any] = {}
    for candidate, control in (
        ("g0_all", "base"),
        ("g1_strict", "base"),
        ("g1_strict", "g0_all"),
    ):
        left = {int(row["ordinal"]): row for row in rows_by_arm[control]}
        right = {int(row["ordinal"]): row for row in rows_by_arm[candidate]}
        if set(left) != set(range(ATTEMPTS)) or set(right) != set(range(ATTEMPTS)):
            raise RuntimeError("official cell ordinals do not cover requested1000")
        known = [
            ordinal
            for ordinal in range(ATTEMPTS)
            if left[ordinal]["official_hull_status"] == "known"
            and right[ordinal]["official_hull_status"] == "known"
        ]
        comparison: dict[str, Any] = {"known_both": len(known)}
        for metric in ("strict_sun", "meta_sun"):
            control_values = [bool(left[index][metric]) for index in known]
            candidate_values = [bool(right[index][metric]) for index in known]
            comparison[metric] = {
                **paired_delta_summary(control_values, candidate_values),
                "mcnemar": _exact_mcnemar(control_values, candidate_values),
            }
        pairwise[f"{candidate}-{control}"] = comparison

    report = {
        "schema": "h1a2_sgtc_l7_official_final_v1",
        "design": {
            "plans": "CTV_DLM_L7_PLANS.jsonl",
            "plans_sha256": "62bf1017b17f696db95b026e7bfe3eed8284a7ea3743332e121c11098e8e46d5",
            "seed": SEED,
            "dlm_seed": 92117,
            "refiner_seed": 102117,
            "temperature": 0.7,
            "refiner_tau": 800,
            "rerank_or_replacement": False,
            "benchmark_attainment_not_confidence_bound_success": True,
        },
        "cells": cells,
        "deltas": deltas,
        "gates": gates,
        "pairwise": pairwise,
        "sgtc_l7_pass": sgtc_l7_pass,
        "public_105_488_changed": False,
        "unknown_policy": "missing; never unstable",
    }
    stem = "SGTC_L7_OFFICIAL_FINAL"
    (output / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / f"{stem}.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    lines = [
        "# SGTC-DLM-v1 official L7",
        "",
        f"SGTC L7 pass: **{sgtc_l7_pass}**",
        "",
        "| Arm | Requested | Body | Direct J | N/U/NU | Hull K/U | Strict stable/SUN | Meta stable/SUN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cells:
        lines.append(
            f"| {row['arm']} | {row['requested']} | {row['body']} | {row['direct_joint']} | "
            f"{row['novel']}/{row['unique']}/{row['novel_unique']} | "
            f"{row['hull_known']}/{row['hull_unknown']} | "
            f"{row['strict_stable']}/{row['strict_sun']} | "
            f"{row['meta_stable']}/{row['meta_sun']} |"
        )
    lines.extend(
        [
            "",
            "The gate uses the requested-1000 denominator. Paired intervals and exact McNemar results are in the JSON artifact.",
            "The existing public 105/488 headline is unchanged by this internal confirmation.",
        ]
    )
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
