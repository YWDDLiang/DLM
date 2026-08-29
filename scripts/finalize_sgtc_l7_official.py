#!/usr/bin/env python3
"""Finalize the one-time official-hull SGTC requested-1000 L7 gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


SEED = 18
ARMS = ("base", "g0_all", "g1_strict")
ATTEMPTS = 1000
HULL_THRESHOLDS = (0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.20, 0.50)
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rate(value: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else value / denominator


def fmt_optional(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _threshold_key(value: float) -> str:
    if value == 0.0:
        return "le_0"
    return f"le_{value:.2f}".replace(".", "p")


def continuous_distribution(values: Sequence[float]) -> dict[str, Any]:
    observed = [float(value) for value in values]
    return {
        "known": len(observed),
        "mean": None if not observed else statistics.fmean(observed),
        "quantiles": {
            "q10": quantile(observed, 0.10),
            "q25": quantile(observed, 0.25),
            "q50": quantile(observed, 0.50),
            "q75": quantile(observed, 0.75),
            "q90": quantile(observed, 0.90),
        },
        "ecdf": {
            _threshold_key(threshold): {
                "threshold": threshold,
                "count": sum(value <= threshold for value in observed),
                "rate": rate(
                    sum(value <= threshold for value in observed), len(observed)
                ),
            }
            for threshold in HULL_THRESHOLDS
        },
    }


def continuous_pair_summary(
    control_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    require_official_known: bool = False,
) -> dict[str, Any]:
    control = {int(row["ordinal"]): row for row in control_rows}
    candidate = {int(row["ordinal"]): row for row in candidate_rows}
    differences: list[float] = []
    for ordinal in sorted(set(control) & set(candidate)):
        left = control[ordinal]
        right = candidate[ordinal]
        if require_official_known and (
            left.get("official_hull_status") != "known"
            or right.get("official_hull_status") != "known"
        ):
            continue
        if left.get(field) is None or right.get(field) is None:
            continue
        if left.get("chemsys") != right.get("chemsys"):
            raise RuntimeError(f"paired composition changed at ordinal {ordinal}")
        left_identity = row_composition_identity(left)
        right_identity = row_composition_identity(right)
        if (
            left_identity is not None
            and right_identity is not None
            and left_identity != right_identity
        ):
            raise RuntimeError(
                f"paired reduced composition changed at ordinal {ordinal}"
            )
        differences.append(float(right[field]) - float(left[field]))
    if not differences:
        return {
            "known_both": 0,
            "candidate_minus_control_mean": None,
            "candidate_minus_control_median": None,
            "fraction_lower": None,
            "lower": 0,
            "higher": 0,
            "ties": 0,
            "normal95_lower": None,
            "normal95_upper": None,
        }
    mean = statistics.fmean(differences)
    standard_error = (
        0.0
        if len(differences) == 1
        else statistics.stdev(differences) / math.sqrt(len(differences))
    )
    radius = 1.959963984540054 * standard_error
    epsilon = 1e-12
    lower = sum(value < -epsilon for value in differences)
    higher = sum(value > epsilon for value in differences)
    ties = len(differences) - lower - higher
    return {
        "known_both": len(differences),
        "candidate_minus_control_mean": mean,
        "candidate_minus_control_median": statistics.median(differences),
        "fraction_lower": rate(lower, len(differences)),
        "lower": lower,
        "higher": higher,
        "ties": ties,
        "normal95_lower": mean - radius,
        "normal95_upper": mean + radius,
    }


def composition_identity(elements: Sequence[str], counts: Sequence[float]) -> str:
    if len(elements) != len(counts) or not elements:
        raise ValueError("composition elements/counts are malformed")
    integral: list[tuple[str, int]] = []
    for element, raw_count in zip(elements, counts):
        count = int(round(float(raw_count)))
        if count <= 0 or abs(float(raw_count) - count) > 1e-6:
            raise ValueError("composition count is not a positive integer")
        integral.append((str(element), count))
    divisor = 0
    for _, count in integral:
        divisor = math.gcd(divisor, count)
    return "|".join(
        f"{element}:{count // divisor}" for element, count in sorted(integral)
    )


def row_composition_identity(row: Mapping[str, Any]) -> str | None:
    composition = row.get("chgnet_composition")
    if not isinstance(composition, Mapping) or not composition:
        return None
    return composition_identity(list(composition), list(composition.values()))


def training_support(curriculum_root: Path) -> dict[str, dict[str, set[str]]]:
    support: dict[str, dict[str, set[str]]] = {}
    for arm in ("g0_all", "g1_strict"):
        identities: set[str] = set()
        chemsys: set[str] = set()
        for row in read_jsonl(curriculum_root / arm / "train.jsonl"):
            minimal = row.get("minimal_spec") or {}
            elements = [str(value) for value in minimal.get("elements") or []]
            counts = [float(value) for value in minimal.get("counts") or []]
            identities.add(composition_identity(elements, counts))
            chemsys.add("-".join(sorted(elements)))
        support[arm] = {"identities": identities, "chemsys": chemsys}
    return support


def _support_stratum(
    rows: Sequence[Mapping[str, Any]],
    support: set[str],
    *,
    field: str,
    seen: bool,
) -> dict[str, Any]:
    selected: list[Mapping[str, Any]] = []
    for row in rows:
        if row.get("reconstructed") is not True:
            continue
        identity = row_composition_identity(row)
        if field != "identity":
            identity = row.get("chemsys")
        if identity is None or ((str(identity) in support) is not seen):
            continue
        selected.append(row)
    hull_values = [
        float(row["official_e_above_hull"])
        for row in selected
        if row.get("official_hull_status") == "known"
        and row.get("official_e_above_hull") is not None
    ]
    return {
        "reconstructed": len(selected),
        "hull_known": len(hull_values),
        "strict_stable": sum(row.get("strict_stable") is True for row in selected),
        "strict_sun": sum(row.get("strict_sun") is True for row in selected),
        "meta_stable": sum(row.get("meta_stable") is True for row in selected),
        "meta_sun": sum(row.get("meta_sun") is True for row in selected),
        "e_hull": continuous_distribution(hull_values),
    }


def seen_unseen_summary(
    rows: Sequence[Mapping[str, Any]], support: Mapping[str, set[str]]
) -> dict[str, Any]:
    return {
        "exact_identity": {
            "seen": _support_stratum(
                rows, support["identities"], field="identity", seen=True
            ),
            "unseen": _support_stratum(
                rows, support["identities"], field="identity", seen=False
            ),
        },
        "chemsys": {
            "seen": _support_stratum(
                rows, support["chemsys"], field="chemsys", seen=True
            ),
            "unseen": _support_stratum(
                rows, support["chemsys"], field="chemsys", seen=False
            ),
        },
    }


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
    parser.add_argument("--curriculum-root", type=Path, required=True)
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

    support = training_support(args.curriculum_root.resolve())
    continuous = {
        "stage_availability": {
            "raw_chgnet": False,
            "raw_official_hull": False,
            "refined_chgnet": True,
            "refined_official_hull": True,
            "reason": (
                "The frozen L7 contract evaluated only reconstructed/refined cells; "
                "raw Direct/CHGNet/hull was not run and is not inferred post hoc."
            ),
        },
        "cells": {},
        "pairwise": {},
    }
    seen_unseen: dict[str, Any] = {
        "training_support": {
            arm: {
                "exact_identities": len(values["identities"]),
                "chemsys": len(values["chemsys"]),
            }
            for arm, values in support.items()
        },
        "cells": {},
    }
    for arm, rows in rows_by_arm.items():
        official_all = [
            float(row["official_e_above_hull"])
            for row in rows
            if row.get("official_hull_status") == "known"
            and row.get("official_e_above_hull") is not None
        ]
        official_novel_unique = [
            float(row["official_e_above_hull"])
            for row in rows
            if row.get("novel_unique") is True
            and row.get("official_hull_status") == "known"
            and row.get("official_e_above_hull") is not None
        ]
        chgnet = [
            float(row["chgnet_energy_per_atom"])
            for row in rows
            if row.get("reconstructed") is True
            and row.get("chgnet_relaxation_known") is True
            and row.get("chgnet_energy_per_atom") is not None
        ]
        continuous["cells"][arm] = {
            "official_e_above_hull_all_known": continuous_distribution(official_all),
            "official_e_above_hull_novel_unique_known": continuous_distribution(
                official_novel_unique
            ),
            "chgnet_energy_per_atom_refined": continuous_distribution(chgnet),
        }
        seen_unseen["cells"][arm] = {
            training_arm: seen_unseen_summary(rows, values)
            for training_arm, values in support.items()
        }
    for candidate, control in (
        ("g0_all", "base"),
        ("g1_strict", "base"),
        ("g1_strict", "g0_all"),
    ):
        continuous["pairwise"][f"{candidate}-{control}"] = {
            "official_e_above_hull": continuous_pair_summary(
                rows_by_arm[control],
                rows_by_arm[candidate],
                field="official_e_above_hull",
                require_official_known=True,
            ),
            "chgnet_energy_per_atom_refined": continuous_pair_summary(
                rows_by_arm[control],
                rows_by_arm[candidate],
                field="chgnet_energy_per_atom",
            ),
        }

    report = {
        "schema": "h1a2_sgtc_l7_official_final_v2",
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
        "continuous": continuous,
        "seen_unseen": seen_unseen,
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
            "## Official e_hull distribution (all known reconstructed attempts)",
            "",
            "| Arm | Known | q10 | q50 | q90 | <=0.01 | <=0.05 | <=0.10 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        distribution = continuous["cells"][arm][
            "official_e_above_hull_all_known"
        ]
        ecdf = distribution["ecdf"]
        lines.append(
            f"| {arm} | {distribution['known']} | "
            f"{fmt_optional(distribution['quantiles']['q10'])} | "
            f"{fmt_optional(distribution['quantiles']['q50'])} | "
            f"{fmt_optional(distribution['quantiles']['q90'])} | "
            f"{ecdf['le_0p01']['count']} | {ecdf['le_0p05']['count']} | "
            f"{ecdf['le_0p10']['count']} |"
        )
    lines.extend(
        [
            "",
            "## Matched continuous effects",
            "",
            "Positive candidate-minus-control energy is adverse; lower fractions above 0.5 are favorable.",
            "",
            "| Pair | Metric | N | Mean delta | Median delta | Fraction lower | Normal 95% CI |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for pair, metrics in continuous["pairwise"].items():
        for metric, summary in metrics.items():
            lines.append(
                f"| {pair} | {metric} | {summary['known_both']} | "
                f"{fmt_optional(summary['candidate_minus_control_mean'], 6)} | "
                f"{fmt_optional(summary['candidate_minus_control_median'], 6)} | "
                f"{fmt_optional(summary['fraction_lower'], 4)} | "
                f"[{fmt_optional(summary['normal95_lower'], 6)}, "
                f"{fmt_optional(summary['normal95_upper'], 6)}] |"
            )
    lines.extend(
        [
            "",
            "## Exact paired binary effects",
            "",
            "| Pair | Metric | Known both | Delta | Wald 95% CI | McNemar p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for pair, metrics in pairwise.items():
        for metric in ("strict_sun", "meta_sun"):
            summary = metrics[metric]
            lines.append(
                f"| {pair} | {metric} | {summary['known_both']} | "
                f"{summary['candidate_minus_control']:+.4%} | "
                f"[{summary['wald95_lower']:+.4%}, {summary['wald95_upper']:+.4%}] | "
                f"{summary['mcnemar']['two_sided_exact_p']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Seen/unseen support relative to G1 strict training chemsys",
            "",
            "| Arm | Seen reconstructed/known | Seen Strict/Meta SUN | Unseen reconstructed/known | Unseen Strict/Meta SUN |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        strata = seen_unseen["cells"][arm]["g1_strict"]["chemsys"]
        seen = strata["seen"]
        unseen = strata["unseen"]
        lines.append(
            f"| {arm} | {seen['reconstructed']}/{seen['hull_known']} | "
            f"{seen['strict_sun']}/{seen['meta_sun']} | "
            f"{unseen['reconstructed']}/{unseen['hull_known']} | "
            f"{unseen['strict_sun']}/{unseen['meta_sun']} |"
        )
    lines.extend(
        [
            "",
            "The gate uses the requested-1000 denominator. Paired intervals and exact McNemar results are in the JSON artifact.",
            "The frozen L7 contract evaluated only reconstructed/refined cells. Raw CHGNet and raw official hull are unavailable and are not inferred post hoc.",
            "The existing public 105/488 headline is unchanged by this internal confirmation.",
        ]
    )
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
