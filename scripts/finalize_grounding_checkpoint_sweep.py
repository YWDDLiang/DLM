#!/usr/bin/env python3
"""Finalize paired control/grounding screens at DLM steps 500 and 1000."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


STEPS = (500, 1000, 1696)
FULL_STEP = 1696
ARMS = ("control", "candidate")
ATTEMPTS = 256


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
    hull_known_novel_unique = [
        row
        for row in rows
        if row.get("novel_unique") is True
        and row.get("official_hull_status") == "known"
        and row.get("official_e_above_hull") is not None
    ]
    energies = [float(row["official_e_above_hull"]) for row in hull_known_novel_unique]
    strict_stable = sum(row.get("strict_stable") is True for row in reconstructed)
    strict_sun = sum(row.get("strict_sun") is True for row in reconstructed)
    meta_stable = sum(row.get("meta_stable") is True for row in reconstructed)
    meta_sun = sum(row.get("meta_sun") is True for row in reconstructed)
    return {
        "reconstructed": len(reconstructed),
        "hull_known_novel_unique": len(hull_known_novel_unique),
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


def training_evidence(sweep_run: Path, arm: str, step: int) -> dict:
    rows = read_jsonl(sweep_run / arm / "training_log.jsonl")
    evaluations = {
        int(row["step"]): float(row["val_loss"])
        for row in rows
        if row.get("event") == "eval" and row.get("val_loss") is not None
    }
    evidence = {"val_loss": evaluations.get(step)}
    if arm == "candidate":
        grounded = [
            row
            for row in rows
            if row.get("event") == "train"
            and int(row.get("step") or 0) <= step
            and int(row.get("counterfactual_grounding_samples") or 0) > 0
            and row.get("counterfactual_grounding_margin") is not None
        ]
        samples = sum(int(row["counterfactual_grounding_samples"]) for row in grounded)
        weighted_margin = sum(
            float(row["counterfactual_grounding_margin"])
            * int(row["counterfactual_grounding_samples"])
            for row in grounded
        )
        evidence.update(
            {
                "grounding_events": len(grounded),
                "grounding_samples": samples,
                "true_vs_counterfactual_margin": None if samples == 0 else weighted_margin / samples,
                "positive_margin_event_fraction": rate(
                    sum(float(row["counterfactual_grounding_margin"]) > 0.0 for row in grounded),
                    len(grounded),
                ),
            }
        )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--sweep-run", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--full-training-run", type=Path, required=True)
    parser.add_argument("--eval-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.eval_runtime.resolve()))
    import protocol  # noqa: PLC0415
    from finalize_official import _evaluate_cell, _exact_mcnemar, _phase_diagrams  # noqa: PLC0415

    if protocol.DENOMINATOR != ATTEMPTS:
        raise RuntimeError("H1_ACTIVE_DENOMINATOR must be 256")
    eval_run = args.eval_run.resolve()
    sweep_run = args.sweep_run.resolve()
    training_run = args.training_run.resolve()
    full_training_run = args.full_training_run.resolve()
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

    training = {
        f"step{step}_{arm}": training_evidence(training_run, arm, step)
        for step in (500, 1000)
        for arm in ARMS
    }
    for arm in ARMS:
        evidence = training_evidence(full_training_run, arm, 1500)
        evidence["checkpoint_step"] = FULL_STEP
        evidence["validation_step"] = 1500
        training[f"step{FULL_STEP}_{arm}"] = evidence
    rows_by_cell: dict[str, list[dict]] = {}
    reports: dict[str, dict] = {}
    cells: list[dict] = []
    energy: dict[str, dict] = {}
    for step in STEPS:
        label = f"step{step:04d}"
        for arm in ARMS:
            cell_id = f"s{step}_{arm}"
            cell = eval_run / label / arm
            rows, report = _evaluate_cell(
                cell_id=cell_id,
                labels_path=cell / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl",
                generation_path=cell / "generation/generation.jsonl",
                direct_path=cell / "evaluation/direct/report.json",
                phase_diagrams=phase_diagrams,
                unresolved=unresolved,
                output_dir=output / "cells" / cell_id,
            )
            rows_by_cell[cell_id] = rows
            reports[cell_id] = report
            energy[cell_id] = energy_summary(rows)
            body = read_json(sweep_run / label / arm / "body/sample_metrics.json")
            refine = read_json(sweep_run / label / arm / "refine/refinement_metrics.json")
            counts = report["counts"]
            direct = report["direct"]
            cells.append(
                {
                    "step": step,
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
                    "strict_stable_to_sun_retention": energy[cell_id][
                        "strict_stable_to_sun_retention"
                    ],
                    "meta_stable_to_sun_retention": energy[cell_id][
                        "meta_stable_to_sun_retention"
                    ],
                }
            )

    comparisons: list[dict] = []
    for step in STEPS:
        control_rows = {int(row["ordinal"]): row for row in rows_by_cell[f"s{step}_control"]}
        candidate_rows = {int(row["ordinal"]): row for row in rows_by_cell[f"s{step}_candidate"]}
        known_both = [
            ordinal
            for ordinal in range(ATTEMPTS)
            if control_rows[ordinal]["official_hull_status"] == "known"
            and candidate_rows[ordinal]["official_hull_status"] == "known"
        ]
        control = next(row for row in cells if row["step"] == step and row["arm"] == "control")
        candidate = next(row for row in cells if row["step"] == step and row["arm"] == "candidate")
        rate_keys = [key for key in control if key.endswith("_rate") or key.endswith("_retention")]
        deltas = {key: candidate[key] - control[key] for key in rate_keys}
        strict_control = [bool(control_rows[index]["strict_sun"]) for index in known_both]
        strict_candidate = [bool(candidate_rows[index]["strict_sun"]) for index in known_both]
        meta_control = [bool(control_rows[index]["meta_sun"]) for index in known_both]
        meta_candidate = [bool(candidate_rows[index]["meta_sun"]) for index in known_both]
        mechanism = {
            "control_val_loss": training[f"step{step}_control"]["val_loss"],
            "candidate_val_loss": training[f"step{step}_candidate"]["val_loss"],
            "candidate_minus_control": (
                None
                if training[f"step{step}_control"]["val_loss"] is None
                or training[f"step{step}_candidate"]["val_loss"] is None
                else training[f"step{step}_candidate"]["val_loss"]
                - training[f"step{step}_control"]["val_loss"]
            ),
            "candidate_true_vs_counterfactual_margin": training[f"step{step}_candidate"].get(
                "true_vs_counterfactual_margin"
            ),
        }
        criteria = {
            "mechanism": mechanism["candidate_minus_control"] is not None
            and mechanism["candidate_minus_control"] < 0.0
            and (mechanism["candidate_true_vs_counterfactual_margin"] or 0.0) > 0.0,
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
        }
        criteria["paired_screen_pass"] = all(criteria.values())
        comparisons.append(
            {
                "step": step,
                "known_both": len(known_both),
                "mechanism": mechanism,
                "deltas": deltas,
                "strict_mcnemar": _exact_mcnemar(strict_control, strict_candidate),
                "meta_mcnemar": _exact_mcnemar(meta_control, meta_candidate),
                "criteria": criteria,
            }
        )

    summary = {
        "schema": "h1a2_grounding_checkpoint_sweep_final_v1",
        "design": {
            "steps": list(STEPS),
            "epoch_fraction": {
                "500": 500 / FULL_STEP,
                "1000": 1000 / FULL_STEP,
                "1696": 1.0,
            },
            "attempts_per_cell": ATTEMPTS,
            "plans": "same fixed Plan cohort across every cell",
            "randomness": "same DLM/refiner seed-by-sample-index streams across arms and steps",
            "downstream": "D1 exact-plan, temperature 0.7, fixed model494, no safe-axis",
            "selection_warning": "checkpoint sweep is diagnostic; no best checkpoint may be reported alone",
            "full_step": (
                "step1696 is freshly generated on the same raw-256 Plans and seed streams; "
                "its validation loss is measured at step1500"
            ),
        },
        "unknown_policy": "excluded from hull-known denominators; never mapped to unstable",
        "training": training,
        "cells": cells,
        "cell_reports": reports,
        "energy": energy,
        "comparisons": comparisons,
        "qualifying_steps": [
            item["step"] for item in comparisons if item["criteria"]["paired_screen_pass"]
        ],
    }
    stem = "GROUNDING_CHECKPOINT_SWEEP_FINAL"
    (output / f"{stem}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)

    lines = [
        "# Counterfactual-grounding DLM checkpoint sweep",
        "",
        "This reports every frozen checkpoint arm; it does not select only the best checkpoint.",
        "Steps 500/1000/1696 correspond to approximately 0.295/0.590/1.000 training epoch.",
        "",
        "| Step | Arm | Body | Direct J | N∩U | Hull K/U | Strict | Meta | Strict stable→S.U.N. | Meta stable→S.U.N. |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cells:
        lines.append(
            f"| {row['step']} | {row['arm']} | {row['body']}/256 | {row['direct_joint']} | "
            f"{row['novel_unique']} | {row['hull_known']}/{row['hull_unknown']} | "
            f"{row['strict']} | {row['meta']} | "
            f"{row['strict_stable_to_sun_retention']:.2%} | "
            f"{row['meta_stable_to_sun_retention']:.2%} |"
        )
    lines.extend(["", "## Paired comparisons", ""])
    for item in comparisons:
        lines.append(
            f"- step {item['step']}: ΔStrict `{item['deltas']['strict_attempt_rate']:+.2%}`, "
            f"ΔMeta `{item['deltas']['meta_attempt_rate']:+.2%}`, "
            f"ΔDirect-J `{item['deltas']['direct_joint_rate']:+.2%}`, "
            f"ΔStrict stable→S.U.N. retention "
            f"`{item['deltas']['strict_stable_to_sun_retention']:+.2%}`; "
            f"screen pass `{item['criteria']['paired_screen_pass']}`."
        )
    lines.extend(
        [
            "",
            f"Qualifying steps: `{summary['qualifying_steps']}`.",
            "",
            "The public 105/1000 Strict and 488/1000 Meta headline remains unchanged.",
        ]
    )
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
