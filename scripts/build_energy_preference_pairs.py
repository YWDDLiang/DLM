#!/usr/bin/env python3
"""Build one conservative CHGNet energy pair per frozen exact rich Plan."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def ordinal_from_attempt(row: dict[str, Any]) -> int:
    if row.get("ordinal") is not None:
        return int(row["ordinal"])
    return int(str(row["attempt_id"]).rsplit("-", 1)[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-cohort", type=Path, required=True)
    parser.add_argument("--body-run", type=Path, required=True)
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-gap", type=float, default=0.06)
    parser.add_argument("--min-train-pairs", type=int, default=96)
    parser.add_argument("--min-validation-pairs", type=int, default=24)
    args = parser.parse_args()

    cohort_rows = read_jsonl(args.plan_cohort)
    cohort = {int(row["sample_idx"]): row for row in cohort_rows}
    if len(cohort_rows) != 256 or set(cohort) != set(range(256)):
        raise ValueError("energy pair cohort must cover sample_idx 0..255")

    stream_candidates: dict[int, dict[int, dict[str, Any]]] = {}
    stream_reports: list[dict[str, Any]] = []
    for stream in range(4):
        body = {
            int(row["sample_idx"]): row
            for row in read_jsonl(args.body_run / f"stream{stream}/body/raw_generations.jsonl")
        }
        labels = {
            int(row["ordinal"]): row
            for row in read_jsonl(
                args.eval_run
                / f"stream{stream}/evaluation/full_reconstructed/attempt_labels_preofficial.jsonl"
            )
        }
        direct = {
            ordinal_from_attempt(row): row
            for row in read_jsonl(
                args.eval_run / f"stream{stream}/evaluation/direct/attempt_metrics.jsonl"
            )
        }
        if set(body) != set(range(256)) or set(labels) != set(range(256)) or set(direct) != set(range(256)):
            raise ValueError(f"stream{stream} does not preserve all requested ordinals")
        eligible = 0
        values: dict[int, dict[str, Any]] = {}
        for sample_idx in range(256):
            source = body[sample_idx]
            label = labels[sample_idx]
            metric = direct[sample_idx]
            expected_hash = str(cohort[sample_idx]["source_plan_state_sha256"])
            observed_hash = canonical_sha256(source.get("plan_state"))
            if observed_hash != expected_hash:
                raise ValueError(f"stream{stream} Plan identity changed at {sample_idx}")
            energy = label.get("chgnet_energy_per_atom")
            text = source.get("text")
            ok = (
                source.get("parsed") is True
                and label.get("reconstructed") is True
                and label.get("chgnet_relaxation_known") is True
                and energy is not None
                and math.isfinite(float(energy))
                and metric.get("valid") is True
                and isinstance(text, str)
                and bool(text.strip())
            )
            if not ok:
                continue
            eligible += 1
            values[sample_idx] = {
                "stream": stream,
                "energy_per_atom": float(energy),
                "text": str(text),
                "conditioning_prompt": str(source.get("conditioning_prompt") or ""),
                "cif_sha256": hashlib.sha256(str(source.get("cif") or "").encode("utf-8")).hexdigest(),
            }
        stream_candidates[stream] = values
        stream_reports.append({"stream": stream, "eligible_direct_chgnet": eligible})

    pairs: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    gap_bins: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    gaps: list[float] = []
    for sample_idx in range(256):
        values = [
            stream_candidates[stream][sample_idx]
            for stream in range(4)
            if sample_idx in stream_candidates[stream]
        ]
        if len(values) < 2:
            failures["fewer_than_two_eligible_streams"] += 1
            continue
        values.sort(key=lambda row: (float(row["energy_per_atom"]), int(row["stream"])))
        positive = values[0]
        negative = values[-1]
        if positive["text"] == negative["text"]:
            failures["identical_extreme_body_text"] += 1
            continue
        gap = float(negative["energy_per_atom"]) - float(positive["energy_per_atom"])
        if gap < float(args.min_gap):
            failures["gap_below_threshold"] += 1
            continue
        prompts = {str(value["conditioning_prompt"]) for value in values}
        if len(prompts) != 1:
            raise ValueError(f"conditioning prompt changed across streams at {sample_idx}")
        plan_row = cohort[sample_idx]
        plan = plan_row["plan_state"]
        split = str(plan_row["pair_split"])
        if gap < 0.10:
            gap_bin = "0.06_0.10"
        elif gap < 0.20:
            gap_bin = "0.10_0.20"
        else:
            gap_bin = "ge_0.20"
        gap_bins[f"{split}:{gap_bin}"] += 1
        stratum = "|".join(
            (
                split,
                str(plan.get("anion_framework") or "other"),
                f"arity{len(plan.get('elements') or ())}",
                f"N{int(plan.get('N') or 0)}",
            )
        )
        strata[stratum] += 1
        gaps.append(gap)
        pairs.append(
            {
                "schema": "h1a2_same_plan_energy_preference_pair_v1",
                "pair_id": f"energy-pair-{sample_idx:04d}",
                "sample_idx": sample_idx,
                "pair_split": split,
                "plan_state": plan,
                "source_plan_state_sha256": plan_row["source_plan_state_sha256"],
                "conditioning_prompt": positive["conditioning_prompt"],
                "positive_text": positive["text"],
                "negative_text": negative["text"],
                "positive_stream": positive["stream"],
                "negative_stream": negative["stream"],
                "positive_energy_per_atom": positive["energy_per_atom"],
                "negative_energy_per_atom": negative["energy_per_atom"],
                "energy_gap_eV_per_atom": gap,
                "positive_cif_sha256": positive["cif_sha256"],
                "negative_cif_sha256": negative["cif_sha256"],
                "eligible_streams": len(values),
                "direct_required": True,
                "novelty_used": False,
                "stability_token_in_prompt": False,
            }
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for split in ("train", "validation"):
        with (output / f"{split}.jsonl").open("x", encoding="utf-8") as handle:
            for row in pairs:
                if row["pair_split"] == split:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    train_pairs = sum(row["pair_split"] == "train" for row in pairs)
    validation_pairs = sum(row["pair_split"] == "validation" for row in pairs)
    gate = {
        "train_pairs_at_least_minimum": train_pairs >= int(args.min_train_pairs),
        "validation_pairs_at_least_minimum": validation_pairs
        >= int(args.min_validation_pairs),
        "gap_threshold_unchanged": float(args.min_gap) == 0.06,
    }
    gate["preference_training_authorized"] = all(gate.values())
    report = {
        "schema": "h1a2_energy_preference_pair_manifest_v1",
        "plan_cohort": str(args.plan_cohort.resolve()),
        "body_run": str(args.body_run.resolve()),
        "eval_run": str(args.eval_run.resolve()),
        "plans": 256,
        "streams": 4,
        "min_gap_eV_per_atom": float(args.min_gap),
        "train_pairs": train_pairs,
        "validation_pairs": validation_pairs,
        "total_pairs": len(pairs),
        "stream_reports": stream_reports,
        "failures": dict(sorted(failures.items())),
        "gap_bins": dict(sorted(gap_bins.items())),
        "gap_quantiles": {
            key: quantile(gaps, probability)
            for key, probability in (("q10", 0.10), ("q25", 0.25), ("q50", 0.50), ("q75", 0.75), ("q90", 0.90))
        },
        "strata": dict(sorted(strata.items())),
        "gate": gate,
    }
    (output / "DLM_STABILITY_PAIR_DATA_MANIFEST.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "DLM_STABILITY_PAIR_DATA_MANIFEST.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "pair_id",
                "sample_idx",
                "pair_split",
                "positive_stream",
                "negative_stream",
                "positive_energy_per_atom",
                "negative_energy_per_atom",
                "energy_gap_eV_per_atom",
                "eligible_streams",
            ),
        )
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in pairs)
    lines = [
        "# DLM same-Plan stability pair data",
        "",
        f"Preference training authorized: **{gate['preference_training_authorized']}**",
        "",
        f"- Plans/streams: `256/4`",
        f"- Train/validation pairs: `{train_pairs}/{validation_pairs}`",
        f"- Minimum gap: `{float(args.min_gap):.3f} eV/atom`",
        f"- Failure reasons: `{dict(sorted(failures.items()))}`",
        f"- Gap quantiles: `{report['gap_quantiles']}`",
        "",
        "Unknown CHGNet energies are missing and never become negatives. Each Plan contributes at most one primary low/high-energy pair; novelty is not a training label.",
    ]
    (output / "DLM_STABILITY_PAIR_DATA_MANIFEST.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
