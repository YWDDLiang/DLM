#!/usr/bin/env python3
"""Finalize official-hull SGTC L6 attribution and promotion gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


SEEDS = (17, 18)
ARMS = ("base", "g0_all", "g1_strict")
ATTEMPTS = 256


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rate(value: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else value / denominator


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
        raise RuntimeError("H1_ACTIVE_DENOMINATOR must be 256")
    cache = args.official_cache_run / "official_mp_cache"
    phase_diagrams = _phase_diagrams(cache / "official_slim_cache.jsonl")
    unresolved = {
        str(row["chemsys"])
        for row in protocol.read_jsonl(cache / "unresolved_chemsys.jsonl")
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cells = []
    rows_by_cell = {}
    for seed in SEEDS:
        for arm in ARMS:
            cell = args.eval_run / f"seed{seed}/{arm}"
            rows, direct = _evaluate_cell(
                cell_id=f"seed{seed}_{arm}",
                labels_path=cell / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl",
                generation_path=cell / "generation/generation.jsonl",
                direct_path=cell / "evaluation/direct/report.json",
                phase_diagrams=phase_diagrams,
                unresolved=unresolved,
                output_dir=output / f"cells/seed{seed}/{arm}",
            )
            rows_by_cell[(seed, arm)] = rows
            body = read_json(
                args.generation_run / f"seed{seed}/{arm}/body/SGTC_BODY_MANIFEST.json"
            )
            refine = read_json(
                args.generation_run / f"seed{seed}/{arm}/refine/refinement_metrics.json"
            )
            summary = summarize_rows(rows)
            summary.update(
                {
                    "seed": seed,
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
            cells.append(summary)

    count_keys = (
        "requested",
        "parsed",
        "body",
        "refined",
        "reconstructed",
        "direct_comp",
        "direct_struct",
        "direct_joint",
        "novel",
        "unique",
        "novel_unique",
        "hull_known",
        "hull_unknown",
        "strict_stable",
        "strict_sun",
        "meta_stable",
        "meta_sun",
    )
    pooled = []
    for arm in ARMS:
        selected = [row for row in cells if row["arm"] == arm]
        item = {"seed": "pooled-repeat-sum", "arm": arm}
        for key in count_keys:
            item[key] = sum(int(row[key]) for row in selected)
        item.update(
            {
                "body_rate": rate(item["body"], item["requested"]),
                "direct_joint_rate": rate(item["direct_joint"], item["requested"]),
                "novel_rate": rate(item["novel"], item["reconstructed"]),
                "unique_rate": rate(item["unique"], item["reconstructed"]),
                "strict_attempt_rate": rate(item["strict_sun"], item["requested"]),
                "meta_attempt_rate": rate(item["meta_sun"], item["requested"]),
                "strict_retention": rate(item["strict_sun"], item["strict_stable"]),
                "meta_retention": rate(item["meta_sun"], item["meta_stable"]),
            }
        )
        pooled.append(item)
    by_arm = {row["arm"]: row for row in pooled}
    rate_keys = (
        "body_rate",
        "direct_joint_rate",
        "novel_rate",
        "unique_rate",
        "strict_attempt_rate",
        "meta_attempt_rate",
        "strict_retention",
        "meta_retention",
    )
    deltas = {
        f"{candidate}-{control}": {
            key: by_arm[candidate][key] - by_arm[control][key] for key in rate_keys
        }
        for candidate, control in (
            ("g0_all", "base"),
            ("g1_strict", "base"),
            ("g1_strict", "g0_all"),
        )
    }

    def relaxed_gate(delta: dict[str, float]) -> dict[str, bool]:
        strict = delta["strict_attempt_rate"]
        meta = delta["meta_attempt_rate"]
        gate = {
            "strict_or_meta_positive_other_ge_minus_1pp": (
                strict > 0.0 and meta >= -0.01
            )
            or (meta > 0.0 and strict >= -0.01),
            "body_ge_minus_3pp": delta["body_rate"] >= -0.03,
            "direct_ge_minus_3pp": delta["direct_joint_rate"] >= -0.03,
            "novel_ge_minus_5pp": delta["novel_rate"] >= -0.05,
            "unique_ge_minus_5pp": delta["unique_rate"] >= -0.05,
            "strict_retention_ge_minus_10pp": delta["strict_retention"] >= -0.10,
            "meta_retention_ge_minus_10pp": delta["meta_retention"] >= -0.10,
        }
        gate["eligible"] = all(gate.values())
        return gate

    gates = {
        "g0_all_vs_base": relaxed_gate(deltas["g0_all-base"]),
        "g1_strict_vs_base": relaxed_gate(deltas["g1_strict-base"]),
        "g1_strict_vs_g0_all_direction": {
            "strict_or_meta_positive_other_ge_minus_1pp": (
                deltas["g1_strict-g0_all"]["strict_attempt_rate"] > 0.0
                and deltas["g1_strict-g0_all"]["meta_attempt_rate"] >= -0.01
            )
            or (
                deltas["g1_strict-g0_all"]["meta_attempt_rate"] > 0.0
                and deltas["g1_strict-g0_all"]["strict_attempt_rate"] >= -0.01
            )
        },
    }
    sgtc_l6_pass = (
        gates["g1_strict_vs_base"]["eligible"]
        and gates["g1_strict_vs_g0_all_direction"][
            "strict_or_meta_positive_other_ge_minus_1pp"
        ]
    )

    mcnemar = {}
    for arm in ("g0_all", "g1_strict"):
        mcnemar[arm] = {}
        for seed in SEEDS:
            left = {int(row["ordinal"]): row for row in rows_by_cell[(seed, "base")]}
            right = {int(row["ordinal"]): row for row in rows_by_cell[(seed, arm)]}
            known = [
                ordinal
                for ordinal in range(ATTEMPTS)
                if left[ordinal]["official_hull_status"] == "known"
                and right[ordinal]["official_hull_status"] == "known"
            ]
            mcnemar[arm][str(seed)] = {
                "known_both": len(known),
                "strict": _exact_mcnemar(
                    [bool(left[index]["strict_sun"]) for index in known],
                    [bool(right[index]["strict_sun"]) for index in known],
                ),
                "meta": _exact_mcnemar(
                    [bool(left[index]["meta_sun"]) for index in known],
                    [bool(right[index]["meta_sun"]) for index in known],
                ),
            }

    report = {
        "schema": "h1a2_sgtc_l6_official_final_v1",
        "design": {
            "base": "minimal-spec step696",
            "g0": "all-MP20 geometry-only continuation",
            "g1": "strict-stable MP20 geometry-only continuation",
            "seeds": list(SEEDS),
            "rerank_or_replacement": False,
        },
        "cells": cells,
        "pooled_repeat_sum": pooled,
        "deltas": deltas,
        "gates": gates,
        "sgtc_l6_pass": sgtc_l6_pass,
        "l7_authorized": sgtc_l6_pass,
        "mcnemar": mcnemar,
        "unknown_policy": "missing; never unstable",
    }
    stem = "SGTC_L6_OFFICIAL_FINAL"
    (output / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / f"{stem}.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pooled[0]))
        writer.writeheader()
        writer.writerows(pooled)
    lines = [
        "# SGTC-DLM-v1 official L6",
        "",
        f"SGTC L6 pass: **{sgtc_l6_pass}**",
        "",
        "| Arm | Requested | Body | Direct J | N/U/NU | Hull K/U | Strict stable/SUN | Meta stable/SUN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled:
        lines.append(
            f"| {row['arm']} | {row['requested']} | {row['body']} | {row['direct_joint']} | "
            f"{row['novel']}/{row['unique']}/{row['novel_unique']} | "
            f"{row['hull_known']}/{row['hull_unknown']} | "
            f"{row['strict_stable']}/{row['strict_sun']} | "
            f"{row['meta_stable']}/{row['meta_sun']} |"
        )
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
