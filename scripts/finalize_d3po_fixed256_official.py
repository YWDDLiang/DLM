#!/usr/bin/env python3
"""Finalize the frozen D3PO fixed256 official-hull experiment.

The independent unit is one of the 256 frozen compositions.  The two common
sampling streams are process replicates: they are averaged within composition
before any training-seed or fixed-cohort bootstrap is computed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence


ATTEMPTS = 256
STREAMS = (17, 18)
ARMS = ("base", "d3po_seed81017", "d3po_seed81018")
CANDIDATE_SEEDS = {
    "d3po_seed81017": 81017,
    "d3po_seed81018": 81018,
}
STAGES = ("refined", "raw")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260830
HULL_THRESHOLDS = (0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.20, 0.50)
STEM = "D3PO_FIXED256_OFFICIAL_FINAL"
UNKNOWN_POLICY = "missing; never unstable"
ENDPOINT_HIERARCHY = (
    "refined_chgnet_paired_delta",
    "official_e_hull_paired_delta",
    "raw_chgnet_paired_delta",
    "meta_stable_and_sun",
    "strict_direct_nu_safety",
)


@dataclass(frozen=True)
class RuntimeAPI:
    denominator: int
    phase_diagrams: Callable[[Path], Any]
    evaluate_cell: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise RuntimeError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    write_text_exclusive(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def continuous_distribution(values: Sequence[float]) -> dict[str, Any]:
    observed = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "known": len(observed),
        "mean": statistics.fmean(observed) if observed else None,
        "quantiles": {
            key: quantile(observed, probability)
            for key, probability in (
                ("q10", 0.10),
                ("q25", 0.25),
                ("q50", 0.50),
                ("q75", 0.75),
                ("q90", 0.90),
            )
        },
        "ecdf": {
            f"le_{threshold:g}": {
                "threshold": threshold,
                "count": sum(value <= threshold for value in observed),
                "rate": rate(sum(value <= threshold for value in observed), len(observed)),
            }
            for threshold in HULL_THRESHOLDS
        },
    }


def composition_identity(row: Mapping[str, Any]) -> str | None:
    composition = row.get("chgnet_composition")
    if not isinstance(composition, Mapping) or not composition:
        return None
    integral: list[tuple[str, int]] = []
    for element, raw_count in composition.items():
        count = int(round(float(raw_count)))
        if count <= 0 or abs(float(raw_count) - count) > 1e-6:
            raise RuntimeError("non-integral CHGNet composition")
        integral.append((str(element), count))
    divisor = 0
    for _, count in integral:
        divisor = math.gcd(divisor, count)
    return "|".join(
        f"{element}:{count // divisor}" for element, count in sorted(integral)
    )


def indexed_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    indexed = {int(row["ordinal"]): dict(row) for row in rows}
    if len(indexed) != ATTEMPTS or set(indexed) != set(range(ATTEMPTS)):
        raise RuntimeError(f"{label} does not cover ordinals 0..255 exactly")
    return indexed


def paired_continuous_deltas(
    control_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    require_official_known: bool,
    label: str,
) -> dict[int, float]:
    control = indexed_rows(control_rows, f"{label}:control")
    candidate = indexed_rows(candidate_rows, f"{label}:candidate")
    deltas: dict[int, float] = {}
    for ordinal in range(ATTEMPTS):
        left = control[ordinal]
        right = candidate[ordinal]
        if require_official_known and (
            left.get("official_hull_status") != "known"
            or right.get("official_hull_status") != "known"
        ):
            continue
        if left.get(field) is None or right.get(field) is None:
            continue
        left_identity = composition_identity(left)
        right_identity = composition_identity(right)
        if left.get("chemsys") != right.get("chemsys"):
            raise RuntimeError(f"paired chemsys changed at {label}:{ordinal}")
        if left_identity is None or right_identity is None or left_identity != right_identity:
            raise RuntimeError(f"paired composition changed at {label}:{ordinal}")
        delta = float(right[field]) - float(left[field])
        if not math.isfinite(delta):
            raise RuntimeError(f"nonfinite paired delta at {label}:{ordinal}")
        deltas[ordinal] = delta
    return deltas


def average_delta_maps(maps: Sequence[Mapping[int, float]]) -> dict[int, float]:
    if not maps:
        return {}
    common = set(maps[0])
    for values in maps[1:]:
        common &= set(values)
    return {
        ordinal: statistics.fmean(float(values[ordinal]) for values in maps)
        for ordinal in sorted(common)
    }


def delta_summary(values: Mapping[int, float]) -> dict[str, Any]:
    ordered = [float(values[index]) for index in sorted(values)]
    return {
        "compositions_requested": ATTEMPTS,
        "compositions_observed": len(ordered),
        "compositions_missing": ATTEMPTS - len(ordered),
        "mean_delta": statistics.fmean(ordered) if ordered else None,
        "median_delta": statistics.median(ordered) if ordered else None,
        "fraction_lower": (
            sum(value < 0.0 for value in ordered) / len(ordered) if ordered else None
        ),
        "lower": sum(value < 0.0 for value in ordered),
        "higher": sum(value > 0.0 for value in ordered),
        "ties": sum(value == 0.0 for value in ordered),
    }


def cluster_bootstrap_summary(
    values: Mapping[int, float],
    *,
    label: str,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    summary = delta_summary(values)
    ordered = [float(values[index]) for index in sorted(values)]
    if not ordered:
        return {
            **summary,
            "bootstrap": {
                "unit": "composition",
                "replicates": replicates,
                "seed": None,
                "ci95_lower": None,
                "ci95_upper": None,
            },
        }
    label_seed = int.from_bytes(
        hashlib.sha256(label.encode("utf-8")).digest()[:8], "big"
    )
    seed = BOOTSTRAP_SEED ^ label_seed
    rng = random.Random(seed)
    count = len(ordered)
    boot_means = [
        sum(ordered[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(replicates)
    ]
    return {
        **summary,
        "bootstrap": {
            "unit": "composition",
            "replicates": replicates,
            "seed": seed,
            "ci95_lower": quantile(boot_means, 0.025),
            "ci95_upper": quantile(boot_means, 0.975),
        },
    }


def exact_mcnemar(control: Sequence[bool], candidate: Sequence[bool]) -> dict[str, Any]:
    if len(control) != len(candidate):
        raise RuntimeError("paired binary endpoint lengths differ")
    control_only = sum(left and not right for left, right in zip(control, candidate))
    candidate_only = sum(right and not left for left, right in zip(control, candidate))
    discordant = control_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(control_only, candidate_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "known_both": len(control),
        "control_only": control_only,
        "candidate_only": candidate_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
        "role": "descriptive_only",
    }


def paired_binary_summary(
    control_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    require_official_known: bool,
    label: str,
) -> dict[str, Any]:
    control = indexed_rows(control_rows, f"{label}:control")
    candidate = indexed_rows(candidate_rows, f"{label}:candidate")
    left_values: list[bool] = []
    right_values: list[bool] = []
    for ordinal in range(ATTEMPTS):
        left = control[ordinal]
        right = candidate[ordinal]
        if require_official_known and (
            left.get("official_hull_status") != "known"
            or right.get("official_hull_status") != "known"
        ):
            continue
        left_values.append(bool(left.get(field)))
        right_values.append(bool(right.get(field)))
    differences = [int(right) - int(left) for left, right in zip(left_values, right_values)]
    return {
        "candidate_minus_base_rate": (
            statistics.fmean(differences) if differences else None
        ),
        "mcnemar": exact_mcnemar(left_values, right_values),
    }


def summarize_cell(
    rows: Sequence[Mapping[str, Any]],
    direct_report: Mapping[str, Any],
    *,
    stream: int,
    arm: str,
    stage: str,
    body_manifest: Mapping[str, Any],
    refinement_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    indexed_rows(rows, f"stream{stream}:{stage}:{arm}")
    reconstructed = sum(row.get("reconstructed") is True for row in rows)
    hull_values = [
        float(row["official_e_above_hull"])
        for row in rows
        if row.get("official_hull_status") == "known"
        and row.get("official_e_above_hull") is not None
    ]
    chgnet_values = [
        float(row["chgnet_energy_per_atom"])
        for row in rows
        if row.get("chgnet_energy_per_atom") is not None
    ]
    direct = direct_report["direct"]
    counts = {
        "requested": ATTEMPTS,
        "parsed": int(body_manifest["parsed"]),
        "body": int(body_manifest["graphs"]),
        "refined": (
            None
            if refinement_manifest is None
            else int(refinement_manifest["num_proposals"])
        ),
        "reconstructed": reconstructed,
        "direct_comp": int(direct["composition_valid"]),
        "direct_struct": int(direct["structure_valid"]),
        "direct_joint": int(direct["joint_valid"]),
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
        "strict_sun": sum(row.get("strict_sun") is True for row in rows),
        "meta_stable": sum(row.get("meta_stable") is True for row in rows),
        "meta_sun": sum(row.get("meta_sun") is True for row in rows),
    }
    return {
        "stream": stream,
        "arm": arm,
        "training_seed": CANDIDATE_SEEDS.get(arm),
        "stage": stage,
        **counts,
        "body_rate": rate(counts["body"], ATTEMPTS),
        "direct_joint_rate": rate(counts["direct_joint"], ATTEMPTS),
        "novel_unique_attempt_rate": rate(counts["novel_unique"], ATTEMPTS),
        "strict_sun_attempt_rate": rate(counts["strict_sun"], ATTEMPTS),
        "meta_sun_attempt_rate": rate(counts["meta_sun"], ATTEMPTS),
        "chgnet_energy_per_atom": continuous_distribution(chgnet_values),
        "official_e_above_hull": continuous_distribution(hull_values),
        "unknown_policy": UNKNOWN_POLICY,
    }


def build_continuous_endpoint(
    rows_by_cell: Mapping[tuple[str, int, str], Sequence[Mapping[str, Any]]],
    *,
    stage: str,
    field: str,
    require_official_known: bool,
    endpoint: str,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    stream_maps: dict[str, dict[int, dict[int, float]]] = {}
    stream_reports: list[dict[str, Any]] = []
    seed_maps: dict[int, dict[int, float]] = {}
    seed_reports: list[dict[str, Any]] = []
    for arm, training_seed in CANDIDATE_SEEDS.items():
        stream_maps[arm] = {}
        for stream in STREAMS:
            values = paired_continuous_deltas(
                rows_by_cell[(stage, stream, "base")],
                rows_by_cell[(stage, stream, arm)],
                field=field,
                require_official_known=require_official_known,
                label=f"{endpoint}:seed{training_seed}:stream{stream}",
            )
            stream_maps[arm][stream] = values
            stream_reports.append(
                {
                    "training_seed": training_seed,
                    "arm": arm,
                    "stream": stream,
                    **delta_summary(values),
                }
            )
        averaged = average_delta_maps([stream_maps[arm][stream] for stream in STREAMS])
        seed_maps[training_seed] = averaged
        seed_reports.append(
            {
                "training_seed": training_seed,
                "arm": arm,
                "stream_handling": "average streams 17/18 within composition first",
                **cluster_bootstrap_summary(
                    averaged,
                    label=f"{endpoint}:seed{training_seed}",
                    replicates=bootstrap_replicates,
                ),
            }
        )
    fixed_cohort = average_delta_maps([seed_maps[seed] for seed in sorted(seed_maps)])
    return {
        "endpoint": endpoint,
        "direction": "negative candidate-minus-base is favorable",
        "per_stream": stream_reports,
        "per_training_seed": seed_reports,
        "fixed_two_training_seed_cohort": {
            "estimand": "mean of the two frozen training-seed policies on this composition cohort",
            "does_not_estimate_training_seed_population_variance": True,
            **cluster_bootstrap_summary(
                fixed_cohort,
                label=f"{endpoint}:fixed-two-seed-cohort",
                replicates=bootstrap_replicates,
            ),
        },
    }


def build_binary_effects(
    cells: Sequence[Mapping[str, Any]],
    rows_by_cell: Mapping[tuple[str, int, str], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    by_key = {
        (str(row["stage"]), int(row["stream"]), str(row["arm"])): row
        for row in cells
    }
    endpoint_fields = {
        "meta_sun": "meta_sun_attempt_rate",
        "strict_sun": "strict_sun_attempt_rate",
        "direct_joint": "direct_joint_rate",
        "novel_unique": "novel_unique_attempt_rate",
    }
    result: dict[str, Any] = {}
    for arm, training_seed in CANDIDATE_SEEDS.items():
        per_stream: list[dict[str, Any]] = []
        seed_deltas: dict[str, float] = {}
        for endpoint, cell_field in endpoint_fields.items():
            deltas: list[float] = []
            for stream in STREAMS:
                control = by_key[("refined", stream, "base")]
                candidate = by_key[("refined", stream, arm)]
                delta = float(candidate[cell_field]) - float(control[cell_field])
                deltas.append(delta)
                per_stream.append(
                    {
                        "stream": stream,
                        "endpoint": endpoint,
                        "candidate_minus_base_requested_rate": delta,
                    }
                )
            seed_deltas[endpoint] = statistics.fmean(deltas)
        mcnemar: dict[str, Any] = {}
        for stream in STREAMS:
            mcnemar[str(stream)] = {
                endpoint: paired_binary_summary(
                    rows_by_cell[("refined", stream, "base")],
                    rows_by_cell[("refined", stream, arm)],
                    field=endpoint,
                    require_official_known=True,
                    label=f"seed{training_seed}:stream{stream}:{endpoint}",
                )
                for endpoint in ("meta_sun", "strict_sun")
            }
        result[str(training_seed)] = {
            "arm": arm,
            "per_stream_requested_rate_deltas": per_stream,
            "training_seed_mean_of_stream_rate_deltas": seed_deltas,
            "mcnemar": mcnemar,
            "mcnemar_role": "descriptive_only; S.U.N. uniqueness is cohort-level and nonlinear",
        }
    return result


def _means_by_seed(endpoint: Mapping[str, Any]) -> dict[int, float]:
    return {
        int(row["training_seed"]): float(row["mean_delta"])
        for row in endpoint["per_training_seed"]
        if row.get("mean_delta") is not None
    }


def _means_by_stream(endpoint: Mapping[str, Any]) -> list[float]:
    return [
        float(row["mean_delta"])
        for row in endpoint["per_stream"]
        if row.get("mean_delta") is not None
    ]


def classify_result(
    *,
    refined: Mapping[str, Any] | None = None,
    official: Mapping[str, Any] | None = None,
    raw: Mapping[str, Any] | None = None,
    binary: Mapping[str, Any] | None = None,
    engineering_failure: str | None = None,
) -> dict[str, Any]:
    """Assign exactly one preregistered evidence class; never a hard gate."""
    if engineering_failure is not None:
        return {
            "code": "E",
            "label": "engineering",
            "reason": engineering_failure,
            "hard_gate": False,
        }
    if refined is None or official is None or raw is None or binary is None:
        raise ValueError("scientific classification requires all frozen endpoints")
    refined_stream = _means_by_stream(refined)
    refined_seed = _means_by_seed(refined)
    official_seed = _means_by_seed(official)
    raw_stream = _means_by_stream(raw)
    raw_seed = _means_by_seed(raw)
    pooled_ci_upper = refined["fixed_two_training_seed_cohort"]["bootstrap"][
        "ci95_upper"
    ]
    expected_seeds = set(CANDIDATE_SEEDS.values())
    complete = (
        len(refined_stream) == 4
        and len(raw_stream) == 4
        and set(refined_seed) == expected_seeds
        and set(official_seed) == expected_seeds
        and set(raw_seed) == expected_seeds
        and pooled_ci_upper is not None
    )
    if not complete:
        return {
            "code": "E",
            "label": "engineering",
            "reason": "required fixed256 paired endpoint coverage is incomplete",
            "hard_gate": False,
        }
    binary_seed = {
        int(seed): value["training_seed_mean_of_stream_rate_deltas"]
        for seed, value in binary.items()
    }
    all_four_refined_left = all(value < 0.0 for value in refined_stream)
    both_refined_seed_left = all(value < 0.0 for value in refined_seed.values())
    both_official_seed_left = all(value < 0.0 for value in official_seed.values())
    all_four_raw_left = all(value < 0.0 for value in raw_stream)
    both_raw_seed_left = all(value < 0.0 for value in raw_seed.values())
    meta_both_improve = all(
        binary_seed[seed]["meta_sun"] > 0.0 for seed in expected_seeds
    )
    direct_nu_retained = all(
        binary_seed[seed]["direct_joint"] >= 0.0
        and binary_seed[seed]["novel_unique"] >= 0.0
        for seed in expected_seeds
    )
    evidence = {
        "all_four_seed_stream_refined_means_lt_zero": all_four_refined_left,
        "both_training_seed_refined_means_lt_zero": both_refined_seed_left,
        "fixed_cohort_refined_bootstrap_ci_upper_lt_zero": pooled_ci_upper < 0.0,
        "both_training_seed_official_hull_means_lt_zero": both_official_seed_left,
        "all_four_seed_stream_raw_means_lt_zero": all_four_raw_left,
        "both_training_seed_raw_means_lt_zero": both_raw_seed_left,
        "meta_requested_rate_improves_for_both_training_seeds": meta_both_improve,
        "direct_and_nu_requested_rates_retained_for_both_training_seeds": direct_nu_retained,
    }
    if (
        all_four_refined_left
        and both_refined_seed_left
        and pooled_ci_upper < 0.0
        and both_official_seed_left
        and meta_both_improve
        and direct_nu_retained
    ):
        code, label = "P", "robust endpoint positive"
    elif all_four_refined_left and both_refined_seed_left:
        code, label = "M", "continuous positive"
    elif all_four_raw_left and both_raw_seed_left and (
        not both_refined_seed_left or not all_four_refined_left
    ):
        code, label = "I", "interface erasure"
    elif not any(value < 0.0 for value in raw_seed.values()):
        code, label = "N", "objective negative"
    else:
        code, label = "U", "seed-unstable"
    return {
        "code": code,
        "label": label,
        "reason": {
            "P": "replicated refined and official left shift with Meta improvement and safety retention",
            "M": "all refined cells improve, but confidence/threshold/official criteria are weaker",
            "I": "both raw policies improve but tau800 removes or reverses consistency",
            "U": "raw improvement is not replicated across both training seeds",
            "N": "neither training-seed aggregate shifts raw energy left",
        }[code],
        "hard_gate": False,
        "evidence": evidence,
    }


def _load_runtime(eval_runtime: Path) -> RuntimeAPI:
    sys.path.insert(0, str(eval_runtime.resolve()))
    protocol = importlib.import_module("protocol")
    official = importlib.import_module("finalize_official")
    return RuntimeAPI(
        denominator=int(protocol.DENOMINATOR),
        phase_diagrams=official._phase_diagrams,
        evaluate_cell=official._evaluate_cell,
    )


def _cache_dir(path: Path) -> Path:
    direct = path.resolve()
    if (direct / "completion_SUCCESS").is_file():
        return direct
    nested = direct / "official_mp_cache"
    if (nested / "completion_SUCCESS").is_file():
        return nested
    raise RuntimeError("official cache completion_SUCCESS is missing")


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _write_input_manifest(output: Path, inputs: Mapping[str, Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    lines = ["sha256\tbytes\tlogical_name\tpath"]
    for logical_name, path in sorted(inputs.items()):
        location = _require_file(path).resolve()
        digest = sha256_file(location)
        size = location.stat().st_size
        rows.append(
            {
                "logical_name": logical_name,
                "path": str(location),
                "bytes": size,
                "sha256": digest,
            }
        )
        lines.append(f"{digest}\t{size}\t{logical_name}\t{location}")
    manifest = output / "INPUTS.sha256"
    write_text_exclusive(manifest, "\n".join(lines) + "\n")
    return {
        "manifest": manifest.name,
        "manifest_sha256": sha256_file(manifest),
        "files": rows,
    }


def _write_output_manifest(output: Path) -> None:
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name not in {"OUTPUTS.sha256", "_SUCCESS"}
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}" for path in files
    ]
    write_text_exclusive(output / "OUTPUTS.sha256", "\n".join(lines) + "\n")


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _render_markdown(report: Mapping[str, Any]) -> str:
    classification = report["classification"]
    continuous = report["continuous_endpoints"]
    lines = [
        "# D3PO fixed256 official final",
        "",
        f"Evidence class: **{classification['code']} — {classification['label']}**",
        "",
        f"Reason: {classification['reason']}",
        "",
        "This is an evidence classification, not a hard promotion gate. The independent unit is 256 compositions; streams 17/18 are averaged within composition.",
    ]
    titles = (
        ("refined_chgnet", "1. Refined CHGNet paired delta"),
        ("official_e_hull", "2. Official e_hull paired delta"),
        ("raw_chgnet", "3. Raw CHGNet paired delta"),
    )
    for key, title in titles:
        endpoint = continuous[key]
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "Negative candidate-minus-base values are favorable.",
                "",
                "| Training seed | Stream | Known compositions | Mean delta | Fraction lower |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in endpoint["per_stream"]:
            lines.append(
                f"| {row['training_seed']} | {row['stream']} | {row['compositions_observed']} | "
                f"{_fmt(row['mean_delta'])} | {_fmt(row['fraction_lower'], 4)} |"
            )
        lines.extend(
            [
                "",
                "| Training seed | Streams averaged first | Known compositions | Mean delta | Composition-bootstrap 95% CI |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for row in endpoint["per_training_seed"]:
            bootstrap = row["bootstrap"]
            lines.append(
                f"| {row['training_seed']} | yes | {row['compositions_observed']} | "
                f"{_fmt(row['mean_delta'])} | [{_fmt(bootstrap['ci95_lower'])}, {_fmt(bootstrap['ci95_upper'])}] |"
            )
        pooled = endpoint["fixed_two_training_seed_cohort"]
        lines.append("")
        lines.append(
            f"Fixed two-seed cohort: n={pooled['compositions_observed']}, mean={_fmt(pooled['mean_delta'])}, "
            f"95% CI=[{_fmt(pooled['bootstrap']['ci95_lower'])}, {_fmt(pooled['bootstrap']['ci95_upper'])}]. "
            "This CI does not estimate training-seed population variance."
        )
    lines.extend(
        [
            "",
            "## 4. Meta stable and S.U.N.",
            "",
            "| Training seed | Mean stream Meta S.U.N. requested-rate delta |",
            "|---:|---:|",
        ]
    )
    for seed, values in sorted(report["binary_effects"].items()):
        delta = values["training_seed_mean_of_stream_rate_deltas"]["meta_sun"]
        lines.append(f"| {seed} | {delta:+.4%} |")
    lines.extend(
        [
            "",
            "## 5. Strict, Direct and NU safety",
            "",
            "| Training seed | Strict S.U.N. delta | Direct-J delta | NU delta |",
            "|---:|---:|---:|---:|",
        ]
    )
    for seed, values in sorted(report["binary_effects"].items()):
        delta = values["training_seed_mean_of_stream_rate_deltas"]
        lines.append(
            f"| {seed} | {delta['strict_sun']:+.4%} | {delta['direct_joint']:+.4%} | "
            f"{delta['novel_unique']:+.4%} |"
        )
    lines.extend(
        [
            "",
            "## Raw/refined cells",
            "",
            "| Stage | Stream | Arm | Body | Reconstructed | Direct J | N/U/NU | Hull K/U | Strict stable/SUN | Meta stable/SUN |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["cells"]:
        lines.append(
            f"| {row['stage']} | {row['stream']} | {row['arm']} | {row['body']} | "
            f"{row['reconstructed']} | {row['direct_joint']} | "
            f"{row['novel']}/{row['unique']}/{row['novel_unique']} | "
            f"{row['hull_known']}/{row['hull_unknown']} | "
            f"{row['strict_stable']}/{row['strict_sun']} | "
            f"{row['meta_stable']}/{row['meta_sun']} |"
        )
    lines.extend(
        [
            "",
            "## Official e_hull ECDF and quantiles",
            "",
            "| Stage | Stream | Arm | Known | q10 | q50 | q90 | <=0 | <=0.05 | <=0.10 |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["cells"]:
        distribution = row["official_e_above_hull"]
        lines.append(
            f"| {row['stage']} | {row['stream']} | {row['arm']} | {distribution['known']} | "
            f"{_fmt(distribution['quantiles']['q10'], 4)} | {_fmt(distribution['quantiles']['q50'], 4)} | "
            f"{_fmt(distribution['quantiles']['q90'], 4)} | {distribution['ecdf']['le_0']['count']} | "
            f"{distribution['ecdf']['le_0.05']['count']} | {distribution['ecdf']['le_0.1']['count']} |"
        )
    lines.extend(
        [
            "",
            "## Statistical notes",
            "",
            "- McNemar tests in JSON are descriptive only.",
            "- Official-hull unknown rows are missing and are never mapped to unstable.",
            "- INPUTS.sha256 and OUTPUTS.sha256 freeze all consumed and emitted artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def _csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in report["cells"]:
        rows.append(
            {
                "record_type": "cell",
                "stage": cell["stage"],
                "stream": cell["stream"],
                "training_seed": cell["training_seed"],
                "arm": cell["arm"],
                "endpoint": "cell_counts",
                "known": cell["hull_known"],
                "requested": cell["requested"],
                "body": cell["body"],
                "direct_joint": cell["direct_joint"],
                "novel_unique": cell["novel_unique"],
                "strict_sun": cell["strict_sun"],
                "meta_sun": cell["meta_sun"],
                "mean": cell["official_e_above_hull"]["mean"],
                "q50": cell["official_e_above_hull"]["quantiles"]["q50"],
            }
        )
    for endpoint_name, endpoint in report["continuous_endpoints"].items():
        for item in endpoint["per_stream"]:
            rows.append(
                {
                    "record_type": "continuous_stream",
                    "stream": item["stream"],
                    "training_seed": item["training_seed"],
                    "arm": item["arm"],
                    "endpoint": endpoint_name,
                    "known": item["compositions_observed"],
                    "mean_delta": item["mean_delta"],
                    "fraction_lower": item["fraction_lower"],
                }
            )
        for item in endpoint["per_training_seed"]:
            rows.append(
                {
                    "record_type": "continuous_training_seed",
                    "training_seed": item["training_seed"],
                    "arm": item["arm"],
                    "endpoint": endpoint_name,
                    "known": item["compositions_observed"],
                    "mean_delta": item["mean_delta"],
                    "fraction_lower": item["fraction_lower"],
                    "ci95_lower": item["bootstrap"]["ci95_lower"],
                    "ci95_upper": item["bootstrap"]["ci95_upper"],
                }
            )
        pooled = endpoint["fixed_two_training_seed_cohort"]
        rows.append(
            {
                "record_type": "continuous_fixed_cohort",
                "endpoint": endpoint_name,
                "known": pooled["compositions_observed"],
                "mean_delta": pooled["mean_delta"],
                "fraction_lower": pooled["fraction_lower"],
                "ci95_lower": pooled["bootstrap"]["ci95_lower"],
                "ci95_upper": pooled["bootstrap"]["ci95_upper"],
            }
        )
    rows.append(
        {
            "record_type": "classification",
            "endpoint": "evidence_class",
            "classification": report["classification"]["code"],
        }
    )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "record_type",
        "stage",
        "stream",
        "training_seed",
        "arm",
        "endpoint",
        "classification",
        "requested",
        "known",
        "body",
        "direct_joint",
        "novel_unique",
        "strict_sun",
        "meta_sun",
        "mean",
        "q50",
        "mean_delta",
        "fraction_lower",
        "ci95_lower",
        "ci95_upper",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def finalize(
    args: argparse.Namespace,
    *,
    runtime: RuntimeAPI | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    generation_run = args.generation_run.resolve()
    eval_run = args.eval_run.resolve()
    eval_runtime = args.eval_runtime.resolve()
    cache = _cache_dir(args.official_cache_run)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    _require_file(generation_run / "_SUCCESS")
    _require_file(eval_run / "_OFFLINE_SUCCESS")
    runtime_api = runtime or _load_runtime(eval_runtime)
    if runtime_api.denominator != ATTEMPTS:
        raise RuntimeError("H1_ACTIVE_DENOMINATOR must be 256")
    output.mkdir(parents=True, exist_ok=False)
    phase_diagrams = runtime_api.phase_diagrams(cache / "official_slim_cache.jsonl")
    unresolved = {
        str(row["chemsys"]) for row in read_jsonl(cache / "unresolved_chemsys.jsonl")
    }

    inputs: dict[str, Path] = {
        "generation_success": generation_run / "_SUCCESS",
        "eval_offline_success": eval_run / "_OFFLINE_SUCCESS",
        "official_completion": cache / "completion_SUCCESS",
        "official_slim_cache": cache / "official_slim_cache.jsonl",
        "official_unresolved": cache / "unresolved_chemsys.jsonl",
    }
    for optional_name, path in (
        ("generation_contract", generation_run / "scientific_contract.tsv"),
        ("generation_inputs", generation_run / "inputs.sha256"),
        ("generation_outputs", generation_run / "OUTPUTS.sha256"),
        ("eval_contract", eval_run / "generation_scientific_contract.tsv"),
        ("eval_inputs", eval_run / "inputs.sha256"),
        ("eval_outputs", eval_run / "OUTPUTS.sha256"),
        ("official_manifest", cache / "completion_manifest.json"),
        ("runtime_protocol", eval_runtime / "protocol.py"),
        ("runtime_finalizer", eval_runtime / "finalize_official.py"),
    ):
        if path.is_file():
            inputs[optional_name] = path

    rows_by_cell: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    cells: list[dict[str, Any]] = []
    for stage in STAGES:
        for stream in STREAMS:
            for arm in ARMS:
                eval_arm = arm if stage == "refined" else f"raw_{arm}"
                cell_root = eval_run / f"stream{stream}/{eval_arm}"
                generation_cell = generation_run / f"stream{stream}/{arm}"
                labels = cell_root / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl"
                generation = cell_root / "generation/generation.jsonl"
                direct = cell_root / "evaluation/direct/report.json"
                full_success = cell_root / "evaluation/full_reconstructed/_SUCCESS"
                inputs[f"{stage}.stream{stream}.{arm}.labels"] = labels
                inputs[f"{stage}.stream{stream}.{arm}.generation"] = generation
                inputs[f"{stage}.stream{stream}.{arm}.direct"] = direct
                inputs[f"{stage}.stream{stream}.{arm}.full_success"] = full_success
                body_path = generation_cell / "body/SGTC_BODY_MANIFEST.json"
                inputs[f"stream{stream}.{arm}.body_manifest"] = body_path
                refinement_path = generation_cell / "refine/refinement_metrics.json"
                if stage == "refined":
                    inputs[f"stream{stream}.{arm}.refinement_manifest"] = refinement_path
                rows, direct_report = runtime_api.evaluate_cell(
                    cell_id=f"stream{stream}_{stage}_{arm}",
                    labels_path=labels,
                    generation_path=generation,
                    direct_path=direct,
                    phase_diagrams=phase_diagrams,
                    unresolved=unresolved,
                    output_dir=output / f"cells/stream{stream}/{stage}/{arm}",
                )
                rows_by_cell[(stage, stream, arm)] = rows
                cells.append(
                    summarize_cell(
                        rows,
                        direct_report,
                        stream=stream,
                        arm=arm,
                        stage=stage,
                        body_manifest=read_json(body_path),
                        refinement_manifest=(
                            read_json(refinement_path) if stage == "refined" else None
                        ),
                    )
                )

    continuous = {
        "refined_chgnet": build_continuous_endpoint(
            rows_by_cell,
            stage="refined",
            field="chgnet_energy_per_atom",
            require_official_known=False,
            endpoint="refined_chgnet_energy_per_atom",
            bootstrap_replicates=bootstrap_replicates,
        ),
        "official_e_hull": build_continuous_endpoint(
            rows_by_cell,
            stage="refined",
            field="official_e_above_hull",
            require_official_known=True,
            endpoint="official_e_above_hull",
            bootstrap_replicates=bootstrap_replicates,
        ),
        "raw_chgnet": build_continuous_endpoint(
            rows_by_cell,
            stage="raw",
            field="chgnet_energy_per_atom",
            require_official_known=False,
            endpoint="raw_chgnet_energy_per_atom",
            bootstrap_replicates=bootstrap_replicates,
        ),
    }
    binary = build_binary_effects(cells, rows_by_cell)
    classification = classify_result(
        refined=continuous["refined_chgnet"],
        official=continuous["official_e_hull"],
        raw=continuous["raw_chgnet"],
        binary=binary,
    )
    input_hashes = _write_input_manifest(output, inputs)
    report: dict[str, Any] = {
        "schema": "h1a2_d3po_fixed256_official_final_v1",
        "status": "complete",
        "classification": classification,
        "design": {
            "attempts": ATTEMPTS,
            "streams": list(STREAMS),
            "arms": list(ARMS),
            "training_seeds": sorted(CANDIDATE_SEEDS.values()),
            "independent_unit": "composition",
            "stream_role": "paired process replicate; averaged within composition",
            "rerank_replacement_checkpoint_or_seed_selection": False,
        },
        "endpoint_hierarchy": list(ENDPOINT_HIERARCHY),
        "cells": cells,
        "continuous_endpoints": continuous,
        "binary_effects": binary,
        "statistical_contract": {
            "bootstrap": "deterministic paired cluster bootstrap over compositions",
            "bootstrap_replicates": bootstrap_replicates,
            "pooled_ci_limitation": "fixed two-policy cohort only; does not estimate training-seed population variance",
            "mcnemar": "descriptive only",
        },
        "unknown_policy": UNKNOWN_POLICY,
        "hashes": {
            "inputs": input_hashes,
            "outputs_manifest": "OUTPUTS.sha256",
            "outputs_manifest_excludes": ["OUTPUTS.sha256", "_SUCCESS"],
        },
    }
    write_json_exclusive(output / f"{STEM}.json", report)
    write_text_exclusive(output / f"{STEM}.md", _render_markdown(report))
    _write_csv(output / f"{STEM}.csv", _csv_rows(report))
    _write_output_manifest(output)
    (output / "_SUCCESS").touch(exist_ok=False)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-run", type=Path, required=True)
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--official-cache-run", type=Path, required=True)
    parser.add_argument("--eval-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = finalize(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "classification": report["classification"]["code"],
                "output": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
