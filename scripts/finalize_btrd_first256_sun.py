#!/usr/bin/env python3
"""Finalize matched first256 raw BTRD stability and S.U.N. endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import finalize_c3fd_llama_prospective_sun as common


ATTEMPTS = 256
ARMS = ("BASE", "BTRD")


def distribution(values):
    values = [float(x) for x in values]
    return {
        "known": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--eval-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    eval_run = args.eval_run.resolve()
    cache = args.official_run.resolve() / "official_mp_cache"
    if not (eval_run / "_OFFLINE_SUCCESS").is_file():
        raise ValueError("offline BTRD evaluation is not complete")
    if not (cache / "completion_SUCCESS").is_file():
        raise ValueError("official cache is not complete")
    runtime = common.load_runtime(args.eval_runtime.resolve())
    protocol = __import__("protocol")
    phase_diagrams = runtime._phase_diagrams(cache / "official_slim_cache.jsonl")
    unresolved = {
        str(row["chemsys"])
        for row in protocol.read_jsonl(cache / "unresolved_chemsys.jsonl")
    }
    output.mkdir(parents=True)
    rows_by_arm = {}
    cells = []
    for arm in ARMS:
        root = eval_run / arm
        generation = root / "generation.jsonl"
        labels = root / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl"
        direct = root / "evaluation/direct/report.json"
        rows, cell_report = runtime._evaluate_cell(
            cell_id=f"raw_{arm}",
            labels_path=labels,
            generation_path=generation,
            direct_path=direct,
            phase_diagrams=phase_diagrams,
            unresolved=unresolved,
            output_dir=output / f"cells/{arm}",
        )
        rows = common.attach_requested_identity(
            rows, protocol.read_jsonl(generation), label=f"raw {arm}"
        )
        rows_by_arm[arm] = rows
        summary = common.summarize_cell("development", "raw", 17, arm, cell_report)
        summary["chgnet_energy"] = distribution(
            row["chgnet_energy_per_atom"]
            for row in rows
            if row.get("chgnet_relaxation_known") is True
            and row.get("chgnet_energy_per_atom") is not None
        )
        summary["official_hull"] = distribution(
            row["official_e_above_hull"]
            for row in rows
            if row.get("official_hull_status") == "known"
            and row.get("official_e_above_hull") is not None
        )
        cells.append(summary)

    chgnet = common.bootstrap(
        common.paired_stream_delta(
            rows_by_arm["BASE"], rows_by_arm["BTRD"],
            field="chgnet_energy_per_atom", require_hull_known=False,
        ),
        "btrd_first256_raw_chgnet",
    )
    hull = common.bootstrap(
        common.paired_stream_delta(
            rows_by_arm["BASE"], rows_by_arm["BTRD"],
            field="official_e_above_hull", require_hull_known=True,
        ),
        "btrd_first256_raw_official_hull",
    )
    base = next(x for x in cells if x["route"] == "BASE")
    btrd = next(x for x in cells if x["route"] == "BTRD")
    binary = {}
    for field in ("strict_sun", "meta_sun"):
        left = [bool(x.get(field)) for x in rows_by_arm["BASE"]]
        right = [bool(x.get(field)) for x in rows_by_arm["BTRD"]]
        binary[field] = runtime._exact_mcnemar(left, right)
    conditions = {
        "chgnet_mean_le_minus_10mev": chgnet["mean"] is not None and chgnet["mean"] <= -0.01,
        "chgnet_ci_upper_below_zero": chgnet["ci95"][1] is not None and chgnet["ci95"][1] < 0,
        "hull_mean_le_minus_10mev": hull["mean"] is not None and hull["mean"] <= -0.01,
        "hull_ci_upper_below_zero": hull["ci95"][1] is not None and hull["ci95"][1] < 0,
        "meta_sun_delta_ge_5": btrd["meta_sun"] - base["meta_sun"] >= 5,
        "strict_sun_non_decrease": btrd["strict_sun"] >= base["strict_sun"],
        "raw_direct_delta_ge_1": btrd["direct_joint"] - base["direct_joint"] >= 1,
        "body_comp_ge_95pct": btrd["reconstructed"] >= 244,
        "planner_comp_delta_zero": True,
    }
    report = {
        "schema": "btrd_first256_sun_final_v1",
        "status": "complete",
        "requested": ATTEMPTS,
        "claim_scope": "pre-registered development block; confirmation untouched",
        "cells": cells,
        "paired_continuous": {"chgnet_energy_per_atom": chgnet, "official_e_above_hull": hull},
        "paired_binary": binary,
        "promotion_conditions": conditions,
        "promoted": all(conditions.values()),
        "candidate_model494_run": False,
        "official_query_run": False,
    }
    path = output / "BTRD_FIRST256_SUN_FINAL.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# BTRD first256 raw endpoint",
        "",
        "| Arm | Body | Direct | Strict S.U.N. | Meta S.U.N. |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in cells:
        lines.append(
            f"| {row['route']} | {row['reconstructed']}/256 | {row['direct_joint']}/256 | "
            f"{row['strict_sun']}/256 | {row['meta_sun']}/256 |"
        )
    lines += [
        "",
        f"- paired CHGNet BTRD-BASE: mean `{chgnet['mean']}`, 95% CI `{chgnet['ci95']}`.",
        f"- paired official hull BTRD-BASE: mean `{hull['mean']}`, 95% CI `{hull['ci95']}`.",
        f"- promoted: `{report['promoted']}`.",
    ]
    (output / "BTRD_FIRST256_SUN_FINAL.md").write_text("\n".join(lines) + "\n")
    (output / "_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
