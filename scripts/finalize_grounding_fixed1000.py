#!/usr/bin/env python3
"""Finalize the paired requested-1000 full-epoch grounding confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ATTEMPTS = 1000
ARMS = ("control", "candidate")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def energy_summary(rows: list[dict]) -> dict:
    reconstructed = [row for row in rows if row.get("reconstructed") is True]
    eligible = [
        row
        for row in rows
        if row.get("novel_unique") is True
        and row.get("official_hull_status") == "known"
        and row.get("official_e_above_hull") is not None
    ]
    energies = [float(row["official_e_above_hull"]) for row in eligible]
    strict_stable = sum(row.get("strict_stable") is True for row in reconstructed)
    strict_sun = sum(row.get("strict_sun") is True for row in reconstructed)
    meta_stable = sum(row.get("meta_stable") is True for row in reconstructed)
    meta_sun = sum(row.get("meta_sun") is True for row in reconstructed)
    return {
        "reconstructed": len(reconstructed),
        "hull_known_novel_unique": len(eligible),
        "strict_stable": strict_stable,
        "strict_sun": strict_sun,
        "strict_stable_to_sun_retention": rate(strict_sun, strict_stable),
        "meta_stable": meta_stable,
        "meta_sun": meta_sun,
        "meta_stable_to_sun_retention": rate(meta_sun, meta_stable),
        "e_hull_quantiles": {
            "q10": quantile(energies, 0.10),
            "q25": quantile(energies, 0.25),
            "q50": quantile(energies, 0.50),
            "q75": quantile(energies, 0.75),
            "q90": quantile(energies, 0.90),
        },
        "e_hull_threshold_counts": {
            "eq_0": sum(value <= 0.0 for value in energies),
            "le_0p01": sum(value <= 0.01 for value in energies),
            "le_0p05": sum(value <= 0.05 for value in energies),
            "le_0p10": sum(value <= 0.10 for value in energies),
        },
    }


def training_evidence(training_run: Path) -> dict:
    result: dict[str, dict] = {}
    for arm in ARMS:
        rows = read_jsonl(training_run / arm / "training_log.jsonl")
        evaluations = [
            row for row in rows if row.get("event") == "eval" and row.get("val_loss") is not None
        ]
        evidence = {
            "validation_step": int(evaluations[-1]["step"]),
            "val_loss": float(evaluations[-1]["val_loss"]),
        }
        if arm == "candidate":
            grounded = [
                row
                for row in rows
                if row.get("event") == "train"
                and int(row.get("counterfactual_grounding_samples") or 0) > 0
                and row.get("counterfactual_grounding_margin") is not None
            ]
            samples = sum(int(row["counterfactual_grounding_samples"]) for row in grounded)
            evidence["true_vs_counterfactual_margin"] = (
                None
                if samples == 0
                else sum(
                    float(row["counterfactual_grounding_margin"])
                    * int(row["counterfactual_grounding_samples"])
                    for row in grounded
                )
                / samples
            )
            evidence["grounding_samples"] = samples
        result[arm] = evidence
    result["candidate_minus_control"] = (
        result["candidate"]["val_loss"] - result["control"]["val_loss"]
    )
    return result


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
    eval_run = args.eval_run.resolve()
    generation_run = args.generation_run.resolve()
    training_run = args.training_run.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    cache = eval_run / "official_mp_cache"
    phase_diagrams = _phase_diagrams(cache / "official_slim_cache.jsonl")
    unresolved = {
        str(row["chemsys"])
        for row in protocol.read_jsonl(cache / "unresolved_chemsys.jsonl")
    }

    rows_by_arm: dict[str, list[dict]] = {}
    reports: dict[str, dict] = {}
    cells: list[dict] = []
    energy: dict[str, dict] = {}
    for arm in ARMS:
        cell = eval_run / arm
        rows, report = _evaluate_cell(
            cell_id=arm,
            labels_path=cell / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl",
            generation_path=cell / "generation/generation.jsonl",
            direct_path=cell / "evaluation/direct/report.json",
            phase_diagrams=phase_diagrams,
            unresolved=unresolved,
            output_dir=output / "cells" / arm,
        )
        rows_by_arm[arm] = rows
        reports[arm] = report
        energy[arm] = energy_summary(rows)
        body = read_json(generation_run / arm / "body/sample_metrics.json")
        refine = read_json(generation_run / arm / "refine/refinement_metrics.json")
        counts = report["counts"]
        direct = report["direct"]
        cells.append(
            {
                "arm": arm,
                "requested": ATTEMPTS,
                "parsed": int(body["parse_success"]),
                "body": int(body["graph_success"]),
                "refined": int(refine["num_proposals"]),
                "reconstructed": int(counts["reconstructed"]),
                "direct_comp": int(direct["composition_valid"]),
                "direct_struct": int(direct["structure_valid"]),
                "direct_joint": int(direct["joint_valid"]),
                "novel": int(counts["novel"]),
                "unique": int(counts["unique_representatives"]),
                "novel_unique": int(counts["novel_unique"]),
                "hull_known": int(counts["hull_known_reconstructed"]),
                "hull_unknown": int(counts["hull_unknown_reconstructed"]),
                "strict": int(counts["strict_sun"]),
                "meta": int(counts["meta_sun"]),
                "body_rate": rate(int(body["graph_success"]), ATTEMPTS),
                "direct_joint_rate": rate(int(direct["joint_valid"]), ATTEMPTS),
                "novel_rate": rate(int(counts["novel"]), int(counts["reconstructed"])),
                "unique_rate": rate(
                    int(counts["unique_representatives"]), int(counts["reconstructed"])
                ),
                "strict_attempt_rate": rate(int(counts["strict_sun"]), ATTEMPTS),
                "meta_attempt_rate": rate(int(counts["meta_sun"]), ATTEMPTS),
                "strict_known_rate": rate(
                    int(counts["strict_sun"]), int(counts["hull_known_reconstructed"])
                ),
                "meta_known_rate": rate(
                    int(counts["meta_sun"]), int(counts["hull_known_reconstructed"])
                ),
                "strict_stable": energy[arm]["strict_stable"],
                "meta_stable": energy[arm]["meta_stable"],
                "strict_stable_to_sun_retention": energy[arm][
                    "strict_stable_to_sun_retention"
                ],
                "meta_stable_to_sun_retention": energy[arm][
                    "meta_stable_to_sun_retention"
                ],
            }
        )

    control = next(row for row in cells if row["arm"] == "control")
    candidate = next(row for row in cells if row["arm"] == "candidate")
    rate_keys = [key for key in control if key.endswith("_rate") or key.endswith("_retention")]
    deltas = {key: candidate[key] - control[key] for key in rate_keys}
    control_rows = {int(row["ordinal"]): row for row in rows_by_arm["control"]}
    candidate_rows = {int(row["ordinal"]): row for row in rows_by_arm["candidate"]}
    known_both = [
        ordinal
        for ordinal in range(ATTEMPTS)
        if control_rows[ordinal]["official_hull_status"] == "known"
        and candidate_rows[ordinal]["official_hull_status"] == "known"
    ]
    strict_mcnemar = _exact_mcnemar(
        [bool(control_rows[index]["strict_sun"]) for index in known_both],
        [bool(candidate_rows[index]["strict_sun"]) for index in known_both],
    )
    meta_mcnemar = _exact_mcnemar(
        [bool(control_rows[index]["meta_sun"]) for index in known_both],
        [bool(candidate_rows[index]["meta_sun"]) for index in known_both],
    )
    training = training_evidence(training_run)
    criteria = {
        "mechanism": training["candidate_minus_control"] < 0.0
        and (training["candidate"].get("true_vs_counterfactual_margin") or 0.0) > 0.0,
        "strict_direction": deltas["strict_attempt_rate"] > 0.0,
        "meta_direction": deltas["meta_attempt_rate"] > 0.0,
        "body_noninferiority_1pp": deltas["body_rate"] >= -0.01,
        "direct_joint_noninferiority_1pp": deltas["direct_joint_rate"] >= -0.01,
        "novelty_noninferiority_1pp": deltas["novel_rate"] >= -0.01
        and deltas["unique_rate"] >= -0.01,
        "strict_retention_noninferiority_1pp": deltas[
            "strict_stable_to_sun_retention"
        ]
        >= -0.01,
        "meta_retention_noninferiority_1pp": deltas["meta_stable_to_sun_retention"]
        >= -0.01,
    }
    criteria["contribution_candidate_pass"] = all(criteria.values())

    summary = {
        "schema": "h1a2_grounding_fixed_requested1000_final_v1",
        "design": {
            "attempts": ATTEMPTS,
            "cohort": "first 1000 parsed epoch-2 Plans; no survivor filtering",
            "pairing": "same Plans and seed-by-sample-index DLM/refiner streams",
            "checkpoint": "full-epoch grounding_34700 control/candidate final",
            "public_headline_separation": "this requested-1000 cohort does not replace or redefine 105/488",
        },
        "unknown_policy": "excluded from hull-known denominators; never mapped to unstable",
        "training": training,
        "cells": cells,
        "cell_reports": reports,
        "energy": energy,
        "known_both": len(known_both),
        "deltas": deltas,
        "strict_mcnemar": strict_mcnemar,
        "meta_mcnemar": meta_mcnemar,
        "criteria": criteria,
    }
    stem = "GROUNDING_FIXED1000_FINAL"
    (output / f"{stem}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)

    lines = [
        "# Counterfactual grounding — fixed requested-1000 confirmation",
        "",
        f"Contribution candidate pass: **{criteria['contribution_candidate_pass']}**",
        "",
        "| Arm | Body | Direct C/S/J | N/U/N∩U | Hull K/U | Strict stable/SUN | Meta stable/SUN | Strict retention | Meta retention |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cells:
        lines.append(
            f"| {row['arm']} | {row['body']}/1000 | "
            f"{row['direct_comp']}/{row['direct_struct']}/{row['direct_joint']} | "
            f"{row['novel']}/{row['unique']}/{row['novel_unique']} | "
            f"{row['hull_known']}/{row['hull_unknown']} | "
            f"{row['strict_stable']}/{row['strict']} | "
            f"{row['meta_stable']}/{row['meta']} | "
            f"{row['strict_stable_to_sun_retention']:.2%} | "
            f"{row['meta_stable_to_sun_retention']:.2%} |"
        )
    lines.extend(["", "## Candidate minus control", ""])
    for key, value in deltas.items():
        lines.append(f"- {key}: `{value:+.4%}`")
    lines.extend(["", "## Frozen criteria", ""])
    for key, value in criteria.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            f"Known-both exact McNemar: Strict candidate-only/control-only "
            f"`{strict_mcnemar['candidate_only']}/{strict_mcnemar['control_only']}`, "
            f"p=`{strict_mcnemar['two_sided_exact_p']:.4g}`; Meta "
            f"`{meta_mcnemar['candidate_only']}/{meta_mcnemar['control_only']}`, "
            f"p=`{meta_mcnemar['two_sided_exact_p']:.4g}`.",
            "",
            "The public 105/1000 Strict and 488/1000 Meta headline remains unchanged and is not redefined by this cohort.",
        ]
    )
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
