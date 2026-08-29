#!/usr/bin/env python3
"""Build composition-normalized soft-D3PO pairs from frozen outcome assets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SPLIT_SALT = "h1a2-shared-noise-d3po-chemsys-v1"
FAMILIES = {
    "oxide",
    "halide",
    "sulfide",
    "chalcogenide",
    "nitride",
    "phosphide_or_phosphate",
    "other",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_composition(plan: Mapping[str, Any]) -> tuple[list[str], list[int]]:
    elements = [str(value) for value in (plan.get("elements") or ())]
    counts = [int(value) for value in (plan.get("counts") or ())]
    if not elements or len(elements) != len(counts):
        raise ValueError("Plan elements/counts are malformed")
    pairs = sorted(zip(elements, counts))
    if any(count <= 0 for _, count in pairs):
        raise ValueError("Plan counts must be positive")
    n_value = int(plan.get("N") or 0)
    if n_value <= 0 or sum(count for _, count in pairs) != n_value:
        raise ValueError("Plan N/count conservation failed")
    return [element for element, _ in pairs], [count for _, count in pairs]


def formula_from_counts(elements: Sequence[str], counts: Sequence[int]) -> str:
    return "".join(
        element if count == 1 else f"{element}{count}"
        for element, count in zip(elements, counts)
    )


def minimal_prompt_from_plan(plan: Mapping[str, Any]) -> tuple[str | None, str]:
    elements, counts = canonical_composition(plan)
    family = str(plan.get("anion_framework") or "")
    if family not in FAMILIES:
        return None, "unsupported_family"
    bucket = str(plan.get("charge_bucket") or "")
    validator = plan.get("validator") or {}
    certificate = plan.get("ccfd_v2_certificate") or {}
    reason = str(
        validator.get("reason")
        or certificate.get("benchmark_reason")
        or certificate.get("reason")
        or ""
    )
    if bucket == "all_metal" or reason == "all_metal_shortcut":
        charge = "all_metal"
    elif bucket in {"neutral_plausible", "certified_neutral"} or reason in {
        "charge_neutral_pauling_valid",
        "validator_boolean",
        "independent_benchmark_pass",
    }:
        charge = "certified_neutral"
    elif certificate.get("benchmark_valid") is True:
        charge = "certified_neutral"
    else:
        return None, f"unsupported_charge:{bucket or reason or 'missing'}"
    spec = {
        "N": sum(counts),
        "charge": charge,
        "counts": counts,
        "elements": elements,
        "family": family,
        "formula": formula_from_counts(elements, counts),
    }
    payload = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return payload + "\ndynamic_crystal_body:", "ok"


def composition_identity(plan: Mapping[str, Any]) -> str:
    elements, counts = canonical_composition(plan)
    divisor = 0
    for count in counts:
        divisor = math.gcd(divisor, count)
    return "|".join(
        f"{element}:{count // divisor}" for element, count in zip(elements, counts)
    )


def chemsys(plan: Mapping[str, Any]) -> str:
    elements, _ = canonical_composition(plan)
    return "-".join(elements)


def chemsys_split(value: str, validation_percent: int = 15) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}|{value}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "validation" if bucket < validation_percent else "train"


def ordinal_from_attempt(row: Mapping[str, Any]) -> int:
    if row.get("ordinal") is not None:
        return int(row["ordinal"])
    return int(str(row["attempt_id"]).rsplit("-", 1)[-1])


def deduplicate_outcomes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_text: dict[str, dict[str, Any]] = {}
    for row in rows:
        text = str(row["answer"])
        existing = by_text.get(text)
        if existing is None or float(row["energy_per_atom"]) < float(
            existing["energy_per_atom"]
        ):
            by_text[text] = dict(row)
    return sorted(
        by_text.values(),
        key=lambda row: (float(row["energy_per_atom"]), str(row["source"])),
    )


def build_pairs_for_composition(
    outcomes: Sequence[Mapping[str, Any]],
    *,
    energy_temperature: float = 0.03,
    margin_cap: float = 0.06,
) -> list[dict[str, Any]]:
    unique = deduplicate_outcomes(outcomes)
    unnormalized: list[dict[str, Any]] = []
    for left, right in itertools.combinations(unique, 2):
        left_energy = float(left["energy_per_atom"])
        right_energy = float(right["energy_per_atom"])
        gap = abs(left_energy - right_energy)
        if gap < 1e-12:
            continue
        winner, loser = (left, right) if left_energy < right_energy else (right, left)
        soft_target = 1.0 / (1.0 + math.exp(-gap / energy_temperature))
        raw_weight = min(1.0, gap / margin_cap)
        unnormalized.append(
            {
                "winner": winner,
                "loser": loser,
                "energy_gap_eV_per_atom": gap,
                "soft_target": soft_target,
                "raw_weight": raw_weight,
            }
        )
    total = sum(float(row["raw_weight"]) for row in unnormalized)
    for row in unnormalized:
        row["pair_weight"] = float(row.pop("raw_weight")) / total
    return unnormalized


def noisy_outcomes(
    plan_cohort: Path,
    body_runs: Sequence[Path],
    eval_runs: Sequence[Path],
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    cohort = {int(row["sample_idx"]): row for row in read_jsonl(plan_cohort)}
    if len(body_runs) != 2 or len(eval_runs) != 2:
        raise ValueError("eight-stream source requires two body and two eval runs")
    grouped: dict[str, list[dict[str, Any]]] = {}
    exclusions: Counter[str] = Counter()
    for stream in range(8):
        body_root = body_runs[0 if stream < 4 else 1]
        eval_root = eval_runs[0 if stream < 4 else 1]
        bodies = {
            int(row["sample_idx"]): row
            for row in read_jsonl(
                body_root / f"stream{stream}/body/raw_generations.jsonl"
            )
        }
        labels = {
            int(row["ordinal"]): row
            for row in read_jsonl(
                eval_root
                / f"stream{stream}/evaluation/full_reconstructed/attempt_labels_preofficial.jsonl"
            )
        }
        direct = {
            ordinal_from_attempt(row): row
            for row in read_jsonl(
                eval_root / f"stream{stream}/evaluation/direct/attempt_metrics.jsonl"
            )
        }
        for sample_idx, cohort_row in cohort.items():
            source = bodies[sample_idx]
            label = labels[sample_idx]
            metric = direct[sample_idx]
            if canonical_sha256(source.get("plan_state")) != str(
                cohort_row["source_plan_state_sha256"]
            ):
                raise ValueError(f"noisy stream{stream} Plan changed at {sample_idx}")
            prompt, reason = minimal_prompt_from_plan(source["plan_state"])
            if prompt is None:
                exclusions[reason] += 1
                continue
            energy = label.get("chgnet_energy_per_atom")
            if not (
                source.get("parsed") is True
                and source.get("text")
                and source.get("cif")
                and label.get("reconstructed") is True
                and label.get("chgnet_relaxation_known") is True
                and energy is not None
                and metric.get("valid") is True
            ):
                exclusions["ineligible_outcome"] += 1
                continue
            identity = composition_identity(source["plan_state"])
            grouped.setdefault(identity, []).append(
                {
                    "answer": str(source["text"]),
                    "cif": str(source["cif"]),
                    "energy_per_atom": float(energy),
                    "prompt": prompt,
                    "plan": dict(source["plan_state"]),
                    "source": f"noisy_stream{stream}",
                    "source_ordinal": sample_idx,
                }
            )
    return grouped, exclusions


def l7_outcomes(
    generation_root: Path, final_root: Path
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for arm in ("base", "g0_all", "g1_strict"):
        bodies = {
            int(row["ordinal"]): row
            for row in read_jsonl(
                generation_root / f"seed18/{arm}/body/raw_generations.jsonl"
            )
        }
        labels = {
            int(row["ordinal"]): row
            for row in read_jsonl(
                final_root
                / f"cells/seed18/{arm}/attempt_results_official.jsonl"
            )
        }
        for ordinal, source in bodies.items():
            label = labels[ordinal]
            prompt, reason = minimal_prompt_from_plan(source["plan_state"])
            if prompt is None:
                raise ValueError(f"L7 minimal prompt failed: {reason}")
            energy = label.get("chgnet_energy_per_atom")
            if not (
                source.get("parsed") is True
                and source.get("text")
                and source.get("cif")
                and label.get("reconstructed") is True
                and label.get("chgnet_relaxation_known") is True
                and energy is not None
            ):
                continue
            identity = composition_identity(source["plan_state"])
            grouped.setdefault(identity, []).append(
                {
                    "answer": str(source["text"]),
                    "cif": str(source["cif"]),
                    "energy_per_atom": float(energy),
                    "prompt": prompt,
                    "plan": dict(source["plan_state"]),
                    "source": f"l7_{arm}",
                    "source_ordinal": ordinal,
                }
            )
    return grouped


def merge_groups(*sources: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        for identity, rows in source.items():
            merged.setdefault(identity, []).extend(dict(row) for row in rows)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy-plan-cohort", type=Path, required=True)
    parser.add_argument("--noisy-body-run", type=Path, action="append", required=True)
    parser.add_argument("--noisy-eval-run", type=Path, action="append", required=True)
    parser.add_argument("--l7-generation", type=Path, required=True)
    parser.add_argument("--l7-final", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    noisy, exclusions = noisy_outcomes(
        args.noisy_plan_cohort.resolve(),
        [path.resolve() for path in args.noisy_body_run],
        [path.resolve() for path in args.noisy_eval_run],
    )
    l7 = l7_outcomes(args.l7_generation.resolve(), args.l7_final.resolve())
    groups = merge_groups(noisy, l7)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    group_counts: Counter[str] = Counter()
    source_winners: Counter[str] = Counter()
    source_losers: Counter[str] = Counter()
    pair_index = 0
    for identity, outcomes in sorted(groups.items()):
        prompts = {str(row["prompt"]) for row in outcomes}
        if len(prompts) != 1:
            raise ValueError(f"minimal prompt changed within {identity}")
        plan = outcomes[0]["plan"]
        split = chemsys_split(chemsys(plan))
        pairs = build_pairs_for_composition(outcomes)
        if not pairs:
            continue
        group_counts[split] += 1
        for pair in pairs:
            winner = pair.pop("winner")
            loser = pair.pop("loser")
            source_winners[str(winner["source"])] += 1
            source_losers[str(loser["source"])] += 1
            rows_by_split[split].append(
                {
                    "schema": "h1a2_shared_noise_soft_d3po_pair_v1",
                    "pair_id": f"d3po-pair-{pair_index:07d}",
                    "split": split,
                    "composition_id": identity,
                    "chemsys": chemsys(plan),
                    "N": int(plan["N"]),
                    "prompt": str(winner["prompt"]),
                    "winner_answer": str(winner["answer"]),
                    "loser_answer": str(loser["answer"]),
                    "winner_energy_per_atom": float(winner["energy_per_atom"]),
                    "loser_energy_per_atom": float(loser["energy_per_atom"]),
                    "winner_source": str(winner["source"]),
                    "loser_source": str(loser["source"]),
                    **pair,
                }
            )
            pair_index += 1

    hashes = {
        split: write_jsonl(output / f"{split}.jsonl", rows)
        for split, rows in rows_by_split.items()
    }
    manifest = {
        "schema": "h1a2_shared_noise_soft_d3po_pair_manifest_v1",
        "split_salt": SPLIT_SALT,
        "energy_temperature_eV_per_atom": 0.03,
        "margin_cap_eV_per_atom": 0.06,
        "composition_weight_sum": 1.0,
        "pair_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "composition_counts": dict(group_counts),
        "source_winner_counts": dict(sorted(source_winners.items())),
        "source_loser_counts": dict(sorted(source_losers.items())),
        "noisy_exclusions": dict(sorted(exclusions.items())),
        "hashes": hashes,
        "l7_retired_as_test": True,
        "gpu_jobs_used": 0,
    }
    (output / "D3PO_PAIR_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
