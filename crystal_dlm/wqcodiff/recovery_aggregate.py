"""Aggregate Day-7 recovery cells and execute the preregistered DLM gate."""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import write_json_exclusive


DLM = "B-WQ-DLM-MONO"
BASELINES = ("B-WQ-AR", "B-WQ-D3PM")
LEVELS = (0.3, 0.5, 0.7, 0.9)
FAILED_EDIT_DISTANCE = 20.0
FAILED_TANGENT_ERROR = 1.0


@dataclasses.dataclass(frozen=True, slots=True)
class PairedEstimate:
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    pairs: int
    corruption_seeds: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _read(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("schema") != "wqcodiff_recovery_attempt_v1":
                    raise ValueError(f"{path}:{line_number}: wrong recovery schema")
                key = (
                    payload.get("method"),
                    payload.get("material_id"),
                    payload.get("corruption_seed"),
                    float(payload.get("corruption_level")),
                    payload.get("operator"),
                    payload.get("geometry_condition"),
                    payload.get("schedule"),
                    payload.get("control", "none"),
                    float(payload.get("revision_threshold", 0.7)),
                )
                if key in seen:
                    raise ValueError(f"duplicate recovery cell record: {key}")
                seen.add(key)
                records.append(payload)
    if not records:
        raise ValueError("no recovery records were provided")
    return records


def _select(
    records: Sequence[Mapping[str, Any]],
    *,
    method: str,
    levels: Sequence[float] | None = None,
    geometry: str | None = None,
    schedule: str | None = None,
    control: str | None = None,
) -> list[Mapping[str, Any]]:
    level_set = None if levels is None else {float(value) for value in levels}
    return [
        record
        for record in records
        if record.get("method") == method
        and (level_set is None or float(record["corruption_level"]) in level_set)
        and (geometry is None or record.get("geometry_condition") == geometry)
        and (schedule is None or record.get("schedule") == schedule)
        and (control is None or record.get("control", "none") == control)
    ]


def _pair_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("material_id"),
        int(record.get("corruption_seed")),
        float(record.get("corruption_level")),
        record.get("operator"),
    )


def _exact(record: Mapping[str, Any]) -> float:
    return float(
        record.get("status") == "succeeded"
        and bool(record.get("exact_full_protostructure_recovery"))
    )


def _edit(record: Mapping[str, Any]) -> float:
    if record.get("status") != "succeeded":
        return FAILED_EDIT_DISTANCE
    value = record.get("topology_edit_distance_after")
    return FAILED_EDIT_DISTANCE if value is None else float(value)


def _tangent(record: Mapping[str, Any]) -> float:
    if record.get("status") != "succeeded":
        return FAILED_TANGENT_ERROR
    value = record.get("tangent_coordinate_error")
    return FAILED_TANGENT_ERROR if value is None else float(value)


def _net(record: Mapping[str, Any]) -> float:
    if record.get("status") != "succeeded":
        return 0.0
    return float(record.get("mechanism", {}).get("net_correction", 0.0))


def _mechanism_value(name: str) -> Callable[[Mapping[str, Any]], float | None]:
    def extract(record: Mapping[str, Any]) -> float | None:
        if record.get("status") != "succeeded":
            return None
        value = record.get("mechanism", {}).get(name)
        return None if value is None else float(value)

    return extract


def _rate(records: Sequence[Mapping[str, Any]]) -> float | None:
    return None if not records else float(np.mean([_exact(record) for record in records]))


def _mean(
    records: Sequence[Mapping[str, Any]],
    extractor: Callable[[Mapping[str, Any]], float | None],
) -> float | None:
    values = [value for record in records if (value := extractor(record)) is not None]
    return None if not values else float(np.mean(values))


def _paired_bootstrap(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    *,
    first_value: Callable[[Mapping[str, Any]], float | None],
    second_value: Callable[[Mapping[str, Any]], float | None],
    repetitions: int,
    seed: int,
) -> PairedEstimate:
    left = {_pair_key(record): record for record in first}
    right = {_pair_key(record): record for record in second}
    keys = sorted(set(left) & set(right), key=repr)
    pairs: list[tuple[tuple[Any, ...], float]] = []
    for key in keys:
        a = first_value(left[key])
        b = second_value(right[key])
        if a is not None and b is not None:
            pairs.append((key, float(a) - float(b)))
    if not pairs:
        return PairedEstimate(None, None, None, 0, 0)
    by_seed: dict[int, list[float]] = defaultdict(list)
    for key, difference in pairs:
        by_seed[int(key[1])].append(difference)
    seeds = sorted(by_seed)
    generator = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        selected_seeds = generator.choice(seeds, size=len(seeds), replace=True)
        sampled: list[float] = []
        for selected_seed in selected_seeds:
            values = np.asarray(by_seed[int(selected_seed)], dtype=np.float64)
            sampled.extend(
                generator.choice(values, size=len(values), replace=True).tolist()
            )
        draws[repetition] = float(np.mean(sampled))
    return PairedEstimate(
        estimate=float(np.mean([value for _, value in pairs])),
        ci_low=float(np.quantile(draws, 0.025)),
        ci_high=float(np.quantile(draws, 0.975)),
        pairs=len(pairs),
        corruption_seeds=len(seeds),
    )


def aggregate_recovery(
    paths: Sequence[str | Path],
    *,
    output_path: str | Path,
    primary_geometry: str = "noisy",
    primary_schedule: str = "fixed",
    repetitions: int = 10_000,
    seed: int = 20260710,
) -> dict[str, Any]:
    if repetitions != 10_000:
        raise ValueError("registered recovery bootstrap requires 10,000 repetitions")
    records = _read(paths)
    thresholds = {
        float(record.get("revision_threshold", 0.7)) for record in records
    }
    if len(thresholds) != 1:
        raise ValueError(
            "Day-7 DLM aggregation requires one frozen revision threshold; "
            "calibration artifacts must be aggregated separately"
        )
    if any(record.get("operator") == "none" for record in records):
        raise ValueError("clean threshold-calibration cells cannot enter the DLM gate")
    subsets = sorted({str(record.get("subset_hash")) for record in records})
    cell_table: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record.get("method"),
            float(record["corruption_level"]),
            record.get("operator"),
            record.get("geometry_condition"),
            record.get("schedule"),
            record.get("control", "none"),
        )
        grouped[key].append(record)
    for key, values in sorted(grouped.items(), key=lambda item: repr(item[0])):
        cell_table.append(
            {
                "method": key[0],
                "level": key[1],
                "operator": key[2],
                "geometry": key[3],
                "schedule": key[4],
                "control": key[5],
                "attempts": len(values),
                "successful": sum(value.get("status") == "succeeded" for value in values),
                "exact_attempt_rate": _rate(values),
                "mean_edit_distance_after": _mean(values, _edit),
                "mean_tangent_coordinate_error": _mean(values, _tangent),
                "mean_net_correction": _mean(values, _net),
                "revision_precision": _mean(
                    values, _mechanism_value("revision_precision")
                ),
                "revision_recall": _mean(
                    values, _mechanism_value("revision_recall")
                ),
                "revision_churn": _mean(
                    values, _mechanism_value("revision_churn")
                ),
                "dimension_change_count": _mean(
                    values, _mechanism_value("dimension_change_count")
                ),
                "event_counts": {
                    event: sum(
                        int(value.get("mechanism", {}).get("event_counts", {}).get(event, 0))
                        for value in values
                    )
                    for event in (
                        "orbit_birth",
                        "orbit_death",
                        "wyckoff_type_change",
                        "species_change",
                    )
                },
            }
        )

    high_results: dict[str, Any] = {}
    exact_ci_pass = True
    high_gain_three_pp = False
    edit_wins = 0
    all_level_results: dict[str, Any] = {}
    for level in LEVELS:
        dlm_records = _select(
            records,
            method=DLM,
            levels=(level,),
            geometry=primary_geometry,
            schedule=primary_schedule,
            control="none",
        )
        baseline_records = {
            method: _select(
                records,
                method=method,
                levels=(level,),
                geometry=primary_geometry,
                schedule=primary_schedule,
                control="none",
            )
            for method in BASELINES
        }
        baseline_rates = {method: _rate(values) for method, values in baseline_records.items()}
        available = {
            method: rate for method, rate in baseline_rates.items() if rate is not None
        }
        best_method = max(available, key=available.get) if available else None
        best_records = [] if best_method is None else baseline_records[best_method]
        comparison = _paired_bootstrap(
            dlm_records,
            best_records,
            first_value=_exact,
            second_value=_exact,
            repetitions=repetitions,
            seed=seed + int(level * 100),
        )
        dlm_edit = _mean(dlm_records, _edit)
        baseline_edit = _mean(best_records, _edit)
        won_edit = (
            dlm_edit is not None
            and baseline_edit is not None
            and dlm_edit < baseline_edit
        )
        edit_wins += int(won_edit)
        level_result = {
            "dlm_attempt_rate": _rate(dlm_records),
            "baseline_attempt_rates": baseline_rates,
            "best_baseline": best_method,
            "paired_exact_gain": comparison.to_dict(),
            "dlm_edit_distance": dlm_edit,
            "best_baseline_edit_distance": baseline_edit,
            "edit_distance_won": won_edit,
        }
        all_level_results[str(level)] = level_result
        if level in {0.7, 0.9}:
            high_results[str(level)] = level_result
            exact_ci_pass = exact_ci_pass and (
                comparison.ci_low is not None and comparison.ci_low > 0.0
            )
            high_gain_three_pp = high_gain_three_pp or (
                comparison.estimate is not None and comparison.estimate >= 0.03
            )

    adaptive = _select(
        records,
        method="M-WQ-STRAT-GEO",
        levels=(0.7, 0.9),
        geometry=primary_geometry,
        schedule="geometry-adaptive",
        control="none",
    )
    fixed = _select(
        records,
        method="M-WQ-STRAT-GEO",
        levels=(0.7, 0.9),
        geometry=primary_geometry,
        schedule="fixed",
        control="none",
    )
    absent = _select(
        records,
        method="M-WQ-STRAT-GEO",
        levels=(0.7, 0.9),
        geometry="absent",
        schedule="geometry-adaptive",
        control="none",
    )
    continuous_first = _select(
        records,
        method="M-WQ-STRAT-GEO",
        levels=(0.7, 0.9),
        geometry=primary_geometry,
        schedule="continuous-first",
        control="none",
    )
    adaptive_vs_fixed = _paired_bootstrap(
        adaptive,
        fixed,
        first_value=_exact,
        second_value=_exact,
        repetitions=repetitions,
        seed=seed + 701,
    )
    geometry_to_discrete = _paired_bootstrap(
        adaptive,
        absent,
        first_value=_exact,
        second_value=_exact,
        repetitions=repetitions,
        seed=seed + 702,
    )
    # Positive means the adaptive discrete state improves the final continuous
    # chart error relative to a continuous-first-only intervention.
    discrete_to_geometry = _paired_bootstrap(
        continuous_first,
        adaptive,
        first_value=_tangent,
        second_value=_tangent,
        repetitions=repetitions,
        seed=seed + 703,
    )
    net_correction = _paired_bootstrap(
        adaptive,
        fixed,
        first_value=_net,
        second_value=_net,
        repetitions=repetitions,
        seed=seed + 704,
    )
    controls: dict[str, Any] = {}
    controls_do_not_match = True
    for offset, control in enumerate(
        ("random-count", "shuffled-geometry", "extra-call"), start=1
    ):
        control_records = _select(
            records,
            method="M-WQ-STRAT-GEO",
            levels=(0.7, 0.9),
            geometry=primary_geometry,
            schedule="geometry-adaptive",
            control=control,
        )
        estimate = _paired_bootstrap(
            adaptive,
            control_records,
            first_value=_exact,
            second_value=_exact,
            repetitions=repetitions,
            seed=seed + 710 + offset,
        )
        controls[control] = estimate.to_dict()
        controls_do_not_match = controls_do_not_match and (
            estimate.estimate is not None and estimate.estimate > 0.0
        )

    mechanism_pass = all(
        estimate.ci_low is not None and estimate.ci_low > 0.0
        for estimate in (
            geometry_to_discrete,
            discrete_to_geometry,
            adaptive_vs_fixed,
            net_correction,
        )
    ) and controls_do_not_match
    gates = {
        "high_corruption_exact_ci_lower_positive": exact_ci_pass,
        "one_high_corruption_gain_at_least_3pp": high_gain_three_pp,
        "edit_distance_levels_won_at_least_3": edit_wins >= 3,
        "geometry_to_discrete_positive": geometry_to_discrete.ci_low is not None
        and geometry_to_discrete.ci_low > 0.0,
        "discrete_to_geometry_positive": discrete_to_geometry.ci_low is not None
        and discrete_to_geometry.ci_low > 0.0,
        "adaptive_beats_fixed_same_calls": adaptive_vs_fixed.ci_low is not None
        and adaptive_vs_fixed.ci_low > 0.0,
        "net_correction_positive": net_correction.ci_low is not None
        and net_correction.ci_low > 0.0,
        "random_shuffled_extra_controls_do_not_match": controls_do_not_match,
    }
    dlm_promoted = all(gates.values())
    result = {
        "schema": "wqcodiff_day7_dlm_gate_v1",
        "records": len(records),
        "subset_hashes": subsets,
        "revision_threshold": next(iter(thresholds)),
        "attempt_denominator_failure_penalties": {
            "topology_edit_distance": FAILED_EDIT_DISTANCE,
            "tangent_coordinate_error": FAILED_TANGENT_ERROR,
            "net_correction": 0.0,
        },
        "primary_cell": {
            "geometry": primary_geometry,
            "schedule": primary_schedule,
            "control": "none",
        },
        "all_levels": all_level_results,
        "high_corruption": high_results,
        "edit_distance_levels_won": edit_wins,
        "interventions": {
            "adaptive_vs_fixed": adaptive_vs_fixed.to_dict(),
            "geometry_to_discrete": geometry_to_discrete.to_dict(),
            "discrete_to_geometry": discrete_to_geometry.to_dict(),
            "net_correction": net_correction.to_dict(),
            "controls": controls,
        },
        "gates": gates,
        "mechanism_pass": mechanism_pass,
        "dlm_promoted": dlm_promoted,
        "required_claim_action": (
            "retain_dlm_superiority_claim"
            if dlm_promoted
            else "delete_dlm_superiority_claim_and_use_best_ar_or_d3pm"
        ),
        "cell_table": cell_table,
    }
    location = Path(output_path)
    write_json_exclusive(location, result)
    return result
