#!/usr/bin/env python3
"""Finalize total-epoch 2 versus 3 DLM evaluation on frozen raw1000 Plans."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


ATTEMPTS = 1000
ROUND_SIZE = 500
EPOCH_TO_ARM = {2: "control", 3: "candidate"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rate(value: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else value / denominator


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def evaluated_summary(rows: list[dict[str, Any]], denominator: int) -> dict[str, Any]:
    reconstructed = [row for row in rows if row.get("reconstructed") is True]
    energies = [
        float(row["official_e_above_hull"])
        for row in rows
        if row.get("novel_unique") is True
        and row.get("official_hull_status") == "known"
        and row.get("official_e_above_hull") is not None
    ]
    counts = {
        "reconstructed": len(reconstructed),
        "novel": sum(row.get("novel") is True for row in rows),
        "unique": sum(row.get("unique_representative") is True for row in rows),
        "novel_unique": sum(row.get("novel_unique") is True for row in rows),
        "hull_known": sum(row.get("official_hull_status") == "known" for row in rows),
        "hull_unknown": sum(
            row.get("reconstructed") is True
            and row.get("official_hull_status") != "known"
            for row in rows
        ),
        "strict_stable": sum(row.get("strict_stable") is True for row in rows),
        "meta_stable": sum(row.get("meta_stable") is True for row in rows),
        "strict_sun": sum(row.get("strict_sun") is True for row in rows),
        "meta_sun": sum(row.get("meta_sun") is True for row in rows),
    }
    return {
        **counts,
        "reconstructed_rate": rate(counts["reconstructed"], denominator),
        "novel_rate": rate(counts["novel"], counts["reconstructed"]),
        "unique_rate": rate(counts["unique"], counts["reconstructed"]),
        "strict_attempt_rate": rate(counts["strict_sun"], denominator),
        "meta_attempt_rate": rate(counts["meta_sun"], denominator),
        "strict_known_rate": rate(counts["strict_sun"], counts["hull_known"]),
        "meta_known_rate": rate(counts["meta_sun"], counts["hull_known"]),
        "strict_stable_to_sun_retention": rate(
            counts["strict_sun"], counts["strict_stable"]
        ),
        "meta_stable_to_sun_retention": rate(
            counts["meta_sun"], counts["meta_stable"]
        ),
        "e_hull_quantiles": {
            key: quantile(energies, probability)
            for key, probability in (
                ("q10", 0.10),
                ("q25", 0.25),
                ("q50", 0.50),
                ("q75", 0.75),
                ("q90", 0.90),
            )
        },
        "e_hull_threshold_counts": {
            "eq_0": sum(value <= 0.0 for value in energies),
            "le_0p01": sum(value <= 0.01 for value in energies),
            "le_0p05": sum(value <= 0.05 for value in energies),
            "le_0p10": sum(value <= 0.10 for value in energies),
        },
    }


def training_curve(training_run: Path) -> dict[str, Any]:
    events = read_jsonl(training_run / "train" / "training_log.jsonl")
    evaluations = [
        row
        for row in events
        if row.get("event") == "eval" and row.get("val_loss") is not None
    ]
    return {
        "evaluations": [
            {"step": int(row["step"]), "val_nll": float(row["val_loss"])}
            for row in evaluations
        ],
        "checkpoint_total_epoch2_step": 696,
        "checkpoint_total_epoch3_step": 2392,
        "note": "the training curve is reported in full; checkpoint choice is made by downstream Pareto gates, not minimum CE alone",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--generation-run", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--eval-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.eval_runtime.resolve()))
    import protocol  # noqa: PLC0415
    from finalize_official import _evaluate_cell, _exact_mcnemar, _phase_diagrams  # noqa: PLC0415

    if protocol.DENOMINATOR != ATTEMPTS:
        raise RuntimeError("H1_ACTIVE_DENOMINATOR must be 1000")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = args.eval_run / "official_mp_cache"
    phase_diagrams = _phase_diagrams(cache / "official_slim_cache.jsonl")
    unresolved = {
        str(row["chemsys"])
        for row in protocol.read_jsonl(cache / "unresolved_chemsys.jsonl")
    }

    cells: list[dict[str, Any]] = []
    round_cells: list[dict[str, Any]] = []
    rows_by_epoch: dict[int, list[dict[str, Any]]] = {}
    reports: dict[str, Any] = {}
    for epoch, arm in EPOCH_TO_ARM.items():
        cell = args.eval_run / arm
        rows, report = _evaluate_cell(
            cell_id=f"epoch{epoch}",
            labels_path=cell / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl",
            generation_path=cell / "generation/generation.jsonl",
            direct_path=cell / "evaluation/direct/report.json",
            phase_diagrams=phase_diagrams,
            unresolved=unresolved,
            output_dir=output / "cells" / f"epoch{epoch}",
        )
        rows_by_epoch[epoch] = rows
        reports[f"epoch{epoch}"] = report
        generation_report = read_json(cell / "generation/generation_report.json")
        pooled = evaluated_summary(rows, ATTEMPTS)
        pooled.update(
            {
                "epoch": epoch,
                "scope": "pooled1000",
                "requested": ATTEMPTS,
                "body": int(generation_report["body_success"]),
                "refined": int(generation_report["refined"]),
                "direct_comp": int(report["direct"]["composition_valid"]),
                "direct_struct": int(report["direct"]["structure_valid"]),
                "direct_joint": int(report["direct"]["joint_valid"]),
                "body_rate": rate(int(generation_report["body_success"]), ATTEMPTS),
                "direct_joint_rate": rate(int(report["direct"]["joint_valid"]), ATTEMPTS),
            }
        )
        cells.append(pooled)
        for round_idx, bounds in ((1, (0, ROUND_SIZE)), (2, (ROUND_SIZE, ATTEMPTS))):
            round_rows = [
                row for row in rows if bounds[0] <= int(row["ordinal"]) < bounds[1]
            ]
            direct = read_json(cell / f"evaluation/direct_round{round_idx}/report.json")
            generated = generation_report["rounds"][f"round{round_idx}"]
            summary = evaluated_summary(round_rows, ROUND_SIZE)
            summary.update(
                {
                    "epoch": epoch,
                    "scope": f"round{round_idx}",
                    "requested": ROUND_SIZE,
                    "global_ordinals": list(bounds),
                    "body": int(generated["body_success"]),
                    "refined": int(generated["refined"]),
                    "direct_comp": int(direct["comp_valid_count"]),
                    "direct_struct": int(direct["struct_valid_count"]),
                    "direct_joint": int(direct["valid_count"]),
                    "body_rate": rate(int(generated["body_success"]), ROUND_SIZE),
                    "direct_joint_rate": rate(int(direct["valid_count"]), ROUND_SIZE),
                    "uniqueness_semantics": "pooled1000 representative labels restricted to this global-ordinal half",
                }
            )
            round_cells.append(summary)

    epoch2 = next(row for row in cells if row["epoch"] == 2)
    epoch3 = next(row for row in cells if row["epoch"] == 3)
    rate_keys = [
        key
        for key, value in epoch2.items()
        if (key.endswith("_rate") or key.endswith("_retention"))
        and isinstance(value, (int, float))
    ]
    delta = {key: float(epoch3[key]) - float(epoch2[key]) for key in rate_keys}
    paired2 = {int(row["ordinal"]): row for row in rows_by_epoch[2]}
    paired3 = {int(row["ordinal"]): row for row in rows_by_epoch[3]}
    known_both = [
        idx
        for idx in range(ATTEMPTS)
        if paired2[idx]["official_hull_status"] == "known"
        and paired3[idx]["official_hull_status"] == "known"
    ]
    strict_mcnemar = _exact_mcnemar(
        [bool(paired2[idx]["strict_sun"]) for idx in known_both],
        [bool(paired3[idx]["strict_sun"]) for idx in known_both],
    )
    meta_mcnemar = _exact_mcnemar(
        [bool(paired2[idx]["meta_sun"]) for idx in known_both],
        [bool(paired3[idx]["meta_sun"]) for idx in known_both],
    )

    epoch3_gate = {
        "body_noninferior_1pp": delta["body_rate"] >= -0.01,
        "direct_noninferior_1pp": delta["direct_joint_rate"] >= -0.01,
        "novel_noninferior_1pp": delta["novel_rate"] >= -0.01,
        "unique_noninferior_1pp": delta["unique_rate"] >= -0.01,
        "strict_noninferior": delta["strict_attempt_rate"] >= 0.0,
        "meta_noninferior": delta["meta_attempt_rate"] >= 0.0,
        "strict_retention_noninferior_1pp": (
            delta["strict_stable_to_sun_retention"] >= -0.01
        ),
        "meta_retention_noninferior_1pp": (
            delta["meta_stable_to_sun_retention"] >= -0.01
        ),
    }
    epoch3_gate["select_epoch3"] = all(epoch3_gate.values())
    selected = epoch3 if epoch3_gate["select_epoch3"] else epoch2
    absolute_targets = {
        "strict_at_least_10pct": selected["strict_attempt_rate"] >= 0.10,
        "meta_at_least_50pct": selected["meta_attempt_rate"] >= 0.50,
    }
    absolute_targets["both_met"] = all(absolute_targets.values())

    summary = {
        "schema": "h1a2_dlm_sufficient_raw1000_final_v1",
        "design": {
            "plans": "frozen raw1000 split into first/last valid500 with global ordinals",
            "checkpoints": {"epoch2": "step696", "epoch3": "step2392"},
            "same_seed_stream": True,
            "stability_condition": None,
            "rl": False,
            "rerank": False,
        },
        "unknown_policy": "excluded from hull-known denominators; never mapped to unstable",
        "training": training_curve(args.training_run),
        "pooled": cells,
        "rounds": round_cells,
        "cell_reports": reports,
        "epoch3_minus_epoch2": delta,
        "epoch3_selection_gate": epoch3_gate,
        "selected_epoch": int(selected["epoch"]),
        "selected_metrics": selected,
        "absolute_targets": absolute_targets,
        "known_both": len(known_both),
        "strict_mcnemar": strict_mcnemar,
        "meta_mcnemar": meta_mcnemar,
    }
    stem = "DLM_SUFFICIENT_RAW1000_FINAL"
    (output / f"{stem}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_keys = [
        "epoch",
        "scope",
        "requested",
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
        "strict_attempt_rate",
        "meta_attempt_rate",
        "strict_stable_to_sun_retention",
        "meta_stable_to_sun_retention",
    ]
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_keys)
        writer.writeheader()
        for row in [*cells, *round_cells]:
            writer.writerow({key: row.get(key) for key in csv_keys})

    lines = [
        "# DLM sufficient-training raw1000 result",
        "",
        f"Selected total epoch: **{selected['epoch']}**",
        f"Strict/Meta absolute target met: **{absolute_targets['both_met']}**",
        "",
        "| Epoch | Scope | Body | Direct C/S/J | N/U/NU | Hull K/U | Strict stable/SUN | Meta stable/SUN | Strict retention | Meta retention |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [*cells, *round_cells]:
        lines.append(
            f"| {row['epoch']} | {row['scope']} | {row['body']}/{row['requested']} | "
            f"{row['direct_comp']}/{row['direct_struct']}/{row['direct_joint']} | "
            f"{row['novel']}/{row['unique']}/{row['novel_unique']} | "
            f"{row['hull_known']}/{row['hull_unknown']} | "
            f"{row['strict_stable']}/{row['strict_sun']} | "
            f"{row['meta_stable']}/{row['meta_sun']} | "
            f"{row['strict_stable_to_sun_retention']:.2%} | "
            f"{row['meta_stable_to_sun_retention']:.2%} |"
        )
    lines.extend(["", "## Frozen epoch-3 selection gate", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in epoch3_gate.items())
    lines.extend(["", "## Absolute targets on selected checkpoint", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in absolute_targets.items())
    lines.extend(
        [
            "",
            f"Known-both exact McNemar: Strict epoch3-only/epoch2-only "
            f"`{strict_mcnemar['candidate_only']}/{strict_mcnemar['control_only']}`, "
            f"p=`{strict_mcnemar['two_sided_exact_p']:.4g}`; Meta "
            f"`{meta_mcnemar['candidate_only']}/{meta_mcnemar['control_only']}`, "
            f"p=`{meta_mcnemar['two_sided_exact_p']:.4g}`.",
            "",
            "Round uniqueness uses the pooled-1000 representative definition restricted to each half; pooled1000 is the primary estimand.",
        ]
    )
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
