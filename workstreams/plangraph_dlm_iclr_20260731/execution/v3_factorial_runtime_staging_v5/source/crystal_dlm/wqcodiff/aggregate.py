"""Registered attempt-level final aggregation and ICLR oral gates."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .contracts import write_json_exclusive
from .vocabulary import crystal_system_from_space_group


EVALUATORS = ("chgnet", "mattersim", "mace")
STAGES = ("raw", "common-refiner", "relaxed")
SENSITIVITIES = ("strict", "standard", "lenient")
TRAINING_SEEDS = (11, 23, 47)
REGISTERED_SEED_COUNTS = {11: 3334, 23: 3333, 47: 3333}


@dataclasses.dataclass(frozen=True, slots=True)
class FinalAggregateConfig:
    inputs: tuple[str, ...]
    output: str
    champion: str
    final_method: str
    train_data: tuple[str, ...] = ()
    usage_inputs: tuple[str, ...] = ()
    headline_stage: str = "relaxed"
    bootstrap_repetitions: int = 10_000
    bootstrap_seed: int = 20260710
    allow_nonpaper_counts: bool = False

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("at least one evaluation input is required")
        if not self.champion or not self.final_method or self.champion == self.final_method:
            raise ValueError("distinct champion and final method names are required")
        if self.headline_stage not in STAGES:
            raise ValueError("headline stage must be raw/common-refiner/relaxed")
        if self.bootstrap_repetitions != 10_000:
            raise ValueError("registered final bootstrap requires 10,000 repetitions")


@dataclasses.dataclass(frozen=True, slots=True)
class PairedEstimate:
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    p_one_sided: float | None
    pairs: int
    final_only: int
    champion_only: int
    train_seeds: int
    sampling_seeds: int
    duplicate_clusters: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _read_jsonl(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    locks: dict[tuple[str, str], tuple[str, str, str]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema") != "wqcodiff_mlip_sun_attempt_v1":
                    raise ValueError(f"{path}:{line_number}: invalid evaluation schema")
                key = (
                    str(row.get("attempt_id")),
                    str(row.get("evaluator")),
                    str(row.get("stage")),
                )
                if key in seen:
                    raise ValueError(f"duplicate evaluation row: {key}")
                seen.add(key)
                lock_key = (key[1], key[2])
                lock_value = (
                    str(row.get("contract_hash")),
                    str(row.get("hull_sha256")),
                    str(row.get("matcher_contract_sha256")),
                )
                previous = locks.setdefault(lock_key, lock_value)
                if previous != lock_value:
                    raise ValueError(f"mixed evaluator/hull/matcher contracts for {lock_key}")
                records.append(row)
    if not records:
        raise ValueError("no evaluation rows were supplied")
    return records


def _read_usage(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    job_ids: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "wqcodiff_slurm_usage_v1":
            raise ValueError(f"invalid Slurm usage schema: {path}")
        job_id = str(payload.get("slurm_job_id") or "")
        if not job_id or job_id in job_ids:
            raise ValueError(f"missing/duplicate Slurm job id: {job_id}")
        if float(payload.get("gpu_hours", -1.0)) < 0.0:
            raise ValueError(f"invalid GPU-hour value: {path}")
        job_ids.add(job_id)
        rows.append(payload)
    return rows


def _select(
    records: Sequence[Mapping[str, Any]],
    *,
    method: str,
    evaluator: str | None = None,
    stage: str | None = None,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in records
        if row.get("method") == method
        and (evaluator is None or row.get("evaluator") == evaluator)
        and (stage is None or row.get("stage") == stage)
    ]


def _path_value(row: Mapping[str, Any], path: str) -> float:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return 0.0
        value = value.get(part)
    return float(bool(value)) if isinstance(value, bool) or value is None else float(value)


def _metric_table(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["method"]), str(row["evaluator"]), str(row["stage"]))].append(row)
    table: list[dict[str, Any]] = []
    for (method, evaluator, stage), rows in sorted(groups.items()):
        denominator = len(rows)
        table.append(
            {
                "method": method,
                "evaluator": evaluator,
                "stage": stage,
                "attempts": denominator,
                "succeeded": sum(row.get("status") == "succeeded" for row in rows),
                "stability_at_0p0": float(np.mean([_path_value(row, "stable_at_0p0") for row in rows])),
                "stability_at_0p1": float(np.mean([_path_value(row, "stable_at_0p1") for row in rows])),
                "uniqueness_standard": float(
                    np.mean([_path_value(row, "unique.standard") for row in rows])
                ),
                "full_structure_novel_standard": float(
                    np.mean([_path_value(row, "full_novel.standard") for row in rows])
                ),
                "anonymous_prototype_novel": float(
                    np.mean([_path_value(row, "anonymous_prototype_novel") for row in rows])
                ),
                "species_wyckoff_protostructure_novel": float(
                    np.mean([_path_value(row, "protostructure_novel") for row in rows])
                ),
                "substitution_aware_novel": float(
                    np.mean([_path_value(row, "substitution_aware_novel") for row in rows])
                ),
                "novel_unique_standard": float(
                    np.mean([_path_value(row, "novel_unique_standard") for row in rows])
                ),
                "mlip_sun_at_0p0": float(np.mean([_path_value(row, "mlip_sun_at_0p0") for row in rows])),
                "mlip_sun_at_0p1": float(np.mean([_path_value(row, "mlip_sun_at_0p1") for row in rows])),
                "substitution_aware_mlip_sun_at_0p1": float(
                    np.mean(
                        [
                            _path_value(row, "substitution_aware_mlip_sun_at_0p1")
                            for row in rows
                        ]
                    )
                ),
                "failure_rate": float(
                    np.mean([row.get("status") != "succeeded" for row in rows])
                ),
            }
        )
    return table


def _mechanism_table(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize one immutable mechanism record per generated attempt.

    Evaluation rows repeat generation metadata across evaluators and stages.
    Deduplicating here prevents a method with more evaluator rows from appearing
    to perform more topology/revision events.  Any disagreement is fatal.
    """

    fields = (
        "revision_control",
        "topology_changed",
        "orbit_count_changed",
        "dimension_changed",
        "revision_events",
        "revision_fills",
        "revision_churn",
        "topology_event_counts",
    )
    attempts: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        key = (str(row["method"]), str(row["attempt_id"]))
        current = {field: row.get(field) for field in fields}
        previous = attempts.setdefault(key, current)
        if previous != current:
            raise ValueError(f"inconsistent repeated mechanism metadata for {key}")

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for (method, _attempt_id), row in attempts.items():
        groups[(method, str(row.get("revision_control") or "none"))].append(row)

    result: list[dict[str, Any]] = []
    event_types = (
        "orbit_birth",
        "orbit_death",
        "wyckoff_type_change",
        "species_change",
    )
    for (method, control), rows in sorted(groups.items()):
        denominator = len(rows)
        event_totals = {
            event: sum(
                int(row.get("topology_event_counts", {}).get(event, 0))
                for row in rows
                if isinstance(row.get("topology_event_counts"), Mapping)
            )
            for event in event_types
        }
        churn = [
            float(row["revision_churn"])
            for row in rows
            if row.get("revision_churn") is not None
        ]
        result.append(
            {
                "method": method,
                "revision_control": control,
                "attempts": denominator,
                "topology_changed_rate": float(
                    np.mean([bool(row.get("topology_changed")) for row in rows])
                ),
                "orbit_count_changed_rate": float(
                    np.mean([bool(row.get("orbit_count_changed")) for row in rows])
                ),
                "dimension_changed_rate": float(
                    np.mean([bool(row.get("dimension_changed")) for row in rows])
                ),
                "revision_attempt_rate": float(
                    np.mean([int(row.get("revision_events") or 0) > 0 for row in rows])
                ),
                "revision_events_total": sum(
                    int(row.get("revision_events") or 0) for row in rows
                ),
                "revision_events_per_attempt": float(
                    np.mean([int(row.get("revision_events") or 0) for row in rows])
                ),
                "revision_fills_total": sum(
                    int(row.get("revision_fills") or 0) for row in rows
                ),
                "revision_fills_per_attempt": float(
                    np.mean([int(row.get("revision_fills") or 0) for row in rows])
                ),
                "revision_churn_mean": None
                if len(churn) != denominator
                else float(np.mean(churn)),
                "topology_event_counts_total": event_totals,
                "topology_event_counts_per_attempt": {
                    event: count / denominator for event, count in event_totals.items()
                },
            }
        )
    return result


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[right] = left


def _duplicate_components(
    pairs: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any], float]],
) -> list[int]:
    union = _UnionFind(len(pairs))
    labels: dict[str, int] = {}
    for index, (_, final, champion, _) in enumerate(pairs):
        for row in (final, champion):
            cluster = row.get("duplicate_cluster", {})
            label = cluster.get("standard") if isinstance(cluster, Mapping) else None
            if not label:
                continue
            # A duplicate structure induces dependence irrespective of which
            # method produced it.  Cross-method components therefore share a
            # single label namespace.
            key = str(label)
            if key in labels:
                union.union(index, labels[key])
            else:
                labels[key] = index
    return [union.find(index) for index in range(len(pairs))]


def _seed_for(base: int, label: str) -> int:
    digest = hashlib.sha256(f"{base}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _paired_estimate(
    final_rows: Sequence[Mapping[str, Any]],
    champion_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    repetitions: int,
    seed: int,
    predicate: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
) -> PairedEstimate:
    def index(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            pair_id = str(row.get("pair_id") or "")
            if not pair_id:
                raise ValueError(f"{label} row lacks method-independent pair_id")
            if pair_id in result:
                raise ValueError(f"duplicate {label} pair_id: {pair_id}")
            result[pair_id] = row
        return result

    final_index = index(final_rows, "final")
    champion_index = index(champion_rows, "champion")
    common = sorted(set(final_index) & set(champion_index))
    pairs: list[tuple[str, Mapping[str, Any], Mapping[str, Any], float]] = []
    for pair_id in common:
        final = final_index[pair_id]
        champion = champion_index[pair_id]
        if (
            int(final["training_seed"]) != int(champion["training_seed"])
            or int(final["sampling_seed"]) != int(champion["sampling_seed"])
        ):
            raise ValueError(f"paired seed mismatch for {pair_id}")
        if predicate is not None and not predicate(final, champion):
            continue
        pairs.append(
            (pair_id, final, champion, _path_value(final, metric) - _path_value(champion, metric))
        )
    if not pairs:
        return PairedEstimate(
            None,
            None,
            None,
            None,
            0,
            len(set(final_index) - set(champion_index)),
            len(set(champion_index) - set(final_index)),
            0,
            0,
            0,
        )

    components = _duplicate_components(pairs)
    hierarchy: dict[int, dict[int, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for component, (_, final, _, difference) in zip(components, pairs):
        hierarchy[int(final["training_seed"])][int(final["sampling_seed"])][component].append(
            difference
        )
    training_seeds = sorted(hierarchy)
    generator = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        selected_training = generator.choice(
            training_seeds, size=len(training_seeds), replace=True
        )
        train_values: list[float] = []
        for training_seed in selected_training:
            sampling_groups = hierarchy[int(training_seed)]
            sampling_seeds = sorted(sampling_groups)
            selected_sampling = generator.choice(
                sampling_seeds, size=len(sampling_seeds), replace=True
            )
            sampling_values: list[float] = []
            for sampling_seed in selected_sampling:
                clusters = sampling_groups[int(sampling_seed)]
                cluster_means = np.asarray(
                    [float(np.mean(values)) for values in clusters.values()],
                    dtype=np.float64,
                )
                selected_clusters = generator.choice(
                    cluster_means, size=len(cluster_means), replace=True
                )
                sampling_values.append(float(np.mean(selected_clusters)))
            train_values.append(float(np.mean(sampling_values)))
        draws[repetition] = float(np.mean(train_values))
    estimate = float(np.mean([item[3] for item in pairs]))
    nonpositive = int(np.count_nonzero(draws <= 0.0))
    sampling_count = len(
        {(int(item[1]["training_seed"]), int(item[1]["sampling_seed"])) for item in pairs}
    )
    return PairedEstimate(
        estimate=estimate,
        ci_low=float(np.quantile(draws, 0.025)),
        ci_high=float(np.quantile(draws, 0.975)),
        p_one_sided=(nonpositive + 1.0) / (repetitions + 1.0),
        pairs=len(pairs),
        final_only=len(set(final_index) - set(champion_index)),
        champion_only=len(set(champion_index) - set(final_index)),
        train_seeds=len(training_seeds),
        sampling_seeds=sampling_count,
        duplicate_clusters=len(set(components)),
    )


def _multi_mlip_rows(
    records: Sequence[Mapping[str, Any]], method: str, stage: str
) -> list[dict[str, Any]]:
    by_evaluator: dict[str, dict[str, Mapping[str, Any]]] = {}
    for evaluator in EVALUATORS:
        selected = _select(records, method=method, evaluator=evaluator, stage=stage)
        current: dict[str, Mapping[str, Any]] = {}
        for row in selected:
            pair_id = str(row.get("pair_id") or "")
            if not pair_id:
                raise ValueError("multi-MLIP row lacks pair_id")
            if pair_id in current:
                raise ValueError(f"duplicate multi-MLIP pair {method}/{evaluator}/{pair_id}")
            current[pair_id] = row
        by_evaluator[evaluator] = current
    common = set.intersection(*(set(values) for values in by_evaluator.values()))
    mattersim_ids = set(by_evaluator["mattersim"])
    expected_size = min(6_000, len(mattersim_ids))
    expected_common = set(
        sorted(
            mattersim_ids,
            key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
        )[:expected_size]
    )
    if common != expected_common or any(
        set(by_evaluator[evaluator]) != expected_common
        for evaluator in ("chgnet", "mace")
    ):
        raise ValueError(
            "multi-MLIP rows are not the frozen lowest-SHA256 method-independent pair subset"
        )
    result: list[dict[str, Any]] = []
    for pair_id in sorted(common):
        source = by_evaluator["mattersim"][pair_id]
        outcomes = {
            evaluator: bool(by_evaluator[evaluator][pair_id].get("mlip_sun_at_0p1"))
            for evaluator in EVALUATORS
        }
        result.append(
            {
                "pair_id": pair_id,
                "method": method,
                "training_seed": int(source["training_seed"]),
                "sampling_seed": int(source["sampling_seed"]),
                "duplicate_cluster": source.get("duplicate_cluster", {}),
                "two_of_three": sum(outcomes.values()) >= 2,
                "three_of_three": all(outcomes.values()),
                "unanimous_mattersim_mace": outcomes["mattersim"] and outcomes["mace"],
                "outcomes": outcomes,
            }
        )
    return result


def _multi_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    denominator = len(rows)
    return {
        "attempts": denominator,
        "pair_id_subset_sha256": hashlib.sha256(
            "\n".join(sorted(str(row["pair_id"]) for row in rows)).encode("utf-8")
        ).hexdigest(),
        "selection": "lowest_sha256_method_independent_pair_id",
        "two_of_three": None
        if not rows
        else float(np.mean([bool(row["two_of_three"]) for row in rows])),
        "three_of_three": None
        if not rows
        else float(np.mean([bool(row["three_of_three"]) for row in rows])),
        "unanimous_mattersim_mace": None
        if not rows
        else float(np.mean([bool(row["unanimous_mattersim_mace"]) for row in rows])),
    }


def _holm(p_values: Mapping[str, float | None]) -> dict[str, float | None]:
    available = sorted(
        ((name, float(value)) for name, value in p_values.items() if value is not None),
        key=lambda item: item[1],
    )
    adjusted: dict[str, float | None] = {name: None for name in p_values}
    running = 0.0
    count = len(available)
    for rank, (name, value) in enumerate(available):
        running = max(running, min(1.0, value * (count - rank)))
        adjusted[name] = running
    return adjusted


def _jsd(first: Sequence[Any], second: Sequence[Any]) -> float | None:
    if not first or not second:
        return None
    keys = sorted(set(first) | set(second), key=repr)
    left_count, right_count = Counter(first), Counter(second)
    left = np.asarray([left_count[key] / len(first) for key in keys], dtype=np.float64)
    right = np.asarray([right_count[key] / len(second) for key in keys], dtype=np.float64)
    middle = 0.5 * (left + right)
    left_mask, right_mask = left > 0, right > 0
    divergence = 0.5 * float(np.sum(left[left_mask] * np.log(left[left_mask] / middle[left_mask])))
    divergence += 0.5 * float(
        np.sum(right[right_mask] * np.log(right[right_mask] / middle[right_mask]))
    )
    return divergence


def _wasserstein(first: Sequence[float], second: Sequence[float]) -> float | None:
    if not first or not second:
        return None
    left = np.sort(np.asarray(first, dtype=np.float64))
    right = np.sort(np.asarray(second, dtype=np.float64))
    values = np.sort(np.concatenate((left, right)))
    if len(values) < 2:
        return 0.0
    deltas = np.diff(values)
    left_cdf = np.searchsorted(left, values[:-1], side="right") / len(left)
    right_cdf = np.searchsorted(right, values[:-1], side="right") / len(right)
    return float(np.sum(np.abs(left_cdf - right_cdf) * deltas))


def _reference_distribution(paths: Sequence[str | Path]) -> dict[str, list[Any]] | None:
    if not paths:
        return None
    from pymatgen.core import Structure

    result: dict[str, list[Any]] = defaultdict(list)
    for raw_path in paths:
        with Path(raw_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not record.get("selected"):
                    continue
                primary = record["decompositions"]["symprec_1e-02"]
                state = primary["state"]
                structure = Structure.from_dict(primary["primitive_structure"])
                space_group = int(state["space_group"])
                result["space_group"].append(space_group)
                result["crystal_system"].append(crystal_system_from_space_group(space_group))
                result["occupied_wyckoff_dof"].append(
                    sum(len(orbit.get("free_coordinate", ())) for orbit in state["orbits"])
                )
                result["atom_count"].append(sum(int(orbit["primitive_multiplicity"]) for orbit in state["orbits"]))
                result["orbit_count"].append(len(state["orbits"]))
                result["density_g_cm3"].append(float(structure.density))
                result["element_count"].append(len(structure.composition.elements))
    if not result.get("space_group"):
        raise ValueError("training-distribution inputs contain no selected primary records")
    return dict(result)


def _distribution_report(
    rows: Sequence[Mapping[str, Any]], reference: Mapping[str, Sequence[Any]] | None
) -> dict[str, Any] | None:
    if reference is None:
        return None
    successful = [row for row in rows if row.get("status") == "succeeded"]
    space_groups = [
        int(row["intended_space_group"])
        for row in successful
        if row.get("intended_space_group") is not None
    ]
    generated: dict[str, list[Any]] = {
        "space_group": space_groups,
        "crystal_system": [crystal_system_from_space_group(value) for value in space_groups],
    }
    for key in (
        "occupied_wyckoff_dof",
        "atom_count",
        "orbit_count",
        "density_g_cm3",
        "element_count",
    ):
        generated[key] = [row[key] for row in successful if row.get(key) is not None]
    symmetry_rows = [
        row
        for row in successful
        if row.get("intended_space_group") is not None
        and row.get("redetected_space_group") is not None
    ]
    multiset_rows = [row for row in successful if row.get("wyckoff_multiset_match") is not None]
    return {
        "successful_structures": len(successful),
        "space_group_jsd": _jsd(reference["space_group"], generated["space_group"]),
        "crystal_system_jsd": _jsd(reference["crystal_system"], generated["crystal_system"]),
        "occupied_wyckoff_dof_jsd": _jsd(
            reference["occupied_wyckoff_dof"], generated["occupied_wyckoff_dof"]
        ),
        "atom_count_wasserstein": _wasserstein(reference["atom_count"], generated["atom_count"]),
        "orbit_count_wasserstein": _wasserstein(reference["orbit_count"], generated["orbit_count"]),
        "density_wasserstein_g_cm3": _wasserstein(
            reference["density_g_cm3"], generated["density_g_cm3"]
        ),
        "element_count_wasserstein": _wasserstein(
            reference["element_count"], generated["element_count"]
        ),
        "intended_redetected_space_group_agreement": None
        if not symmetry_rows
        else float(
            np.mean(
                [
                    int(row["intended_space_group"]) == int(row["redetected_space_group"])
                    for row in symmetry_rows
                ]
            )
        ),
        "intended_redetected_wyckoff_multiset_agreement": None
        if not multiset_rows
        else float(np.mean([bool(row["wyckoff_multiset_match"]) for row in multiset_rows])),
    }


def _compute_report(
    final_rows: Sequence[Mapping[str, Any]],
    champion_rows: Sequence[Mapping[str, Any]],
    usage: Sequence[Mapping[str, Any]],
    *,
    final_method: str,
    champion_method: str,
) -> dict[str, Any]:
    def values(
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[list[float], list[float], list[float]]:
        calls: list[float] = []
        walltimes: list[float] = []
        flops: list[float] = []
        for row in rows:
            call_map = row.get("generation_calls")
            if isinstance(call_map, Mapping):
                calls.append(float(sum(float(value) for value in call_map.values())))
            elif row.get("backbone_calls") is not None:
                calls.append(float(row["backbone_calls"]))
            if row.get("generation_walltime_s") is not None:
                walltimes.append(float(row["generation_walltime_s"]))
            if row.get("generation_flops_lower_bound") is not None:
                flops.append(float(row["generation_flops_lower_bound"]))
        return calls, walltimes, flops

    final_calls, final_time, final_flops = values(final_rows)
    champion_calls, champion_time, champion_flops = values(champion_rows)
    mean_final = None if not final_calls else float(np.mean(final_calls))
    mean_champion = None if not champion_calls else float(np.mean(champion_calls))
    call_ratio = (
        None
        if mean_final is None or mean_champion is None or mean_champion <= 0
        else mean_final / mean_champion
    )
    attempt_final_gpuh = None if len(final_time) != len(final_rows) else sum(final_time) / 3600.0
    attempt_champion_gpuh = (
        None if len(champion_time) != len(champion_rows) else sum(champion_time) / 3600.0
    )
    usage_by_method = {
        method: [
            row
            for row in usage
            if row.get("method") == method and row.get("stage") == "sample"
        ]
        for method in (final_method, champion_method)
    }
    final_usage = usage_by_method[final_method]
    champion_usage = usage_by_method[champion_method]
    final_gpuh = (
        None if not final_usage else sum(float(row["gpu_hours"]) for row in final_usage)
    )
    champion_gpuh = (
        None
        if not champion_usage
        else sum(float(row["gpu_hours"]) for row in champion_usage)
    )
    actual_gpu_ratio = (
        None
        if final_gpuh is None or champion_gpuh is None or champion_gpuh <= 0.0
        else final_gpuh / champion_gpuh
    )
    final_sun = sum(_path_value(row, "mlip_sun_at_0p1") for row in final_rows)
    champion_sun = sum(_path_value(row, "mlip_sun_at_0p1") for row in champion_rows)
    final_efficiency = None if not final_gpuh or final_gpuh <= 0 else final_sun / final_gpuh
    champion_efficiency = (
        None if not champion_gpuh or champion_gpuh <= 0 else champion_sun / champion_gpuh
    )
    pareto = bool(
        final_efficiency is not None
        and champion_efficiency is not None
        and final_efficiency > champion_efficiency
        and final_sun / max(len(final_rows), 1) > champion_sun / max(len(champion_rows), 1)
    )
    return {
        "mean_generation_calls": {"final": mean_final, "champion": mean_champion},
        "generation_call_ratio": call_ratio,
        "generation_flops_lower_bound": {
            "final": None if len(final_flops) != len(final_rows) else sum(final_flops),
            "champion": None
            if len(champion_flops) != len(champion_rows)
            else sum(champion_flops),
            "estimator": "2x_parameter_count_per_joint_call_lower_bound_not_actual_flops",
        },
        "attempt_walltime_gpu_hours_proxy": {
            "final": attempt_final_gpuh,
            "champion": attempt_champion_gpuh,
        },
        "actual_slurm_gpu_hours": {"final": final_gpuh, "champion": champion_gpuh},
        "actual_slurm_gpu_hour_ratio": actual_gpu_ratio,
        "peak_memory_mib": {
            "final": max(
                (float(row["peak_memory_mib"]) for row in final_usage if row.get("peak_memory_mib") is not None),
                default=None,
            ),
            "champion": max(
                (
                    float(row["peak_memory_mib"])
                    for row in champion_usage
                    if row.get("peak_memory_mib") is not None
                ),
                default=None,
            ),
        },
        "slurm_usage_records": {
            "final": len(final_usage),
            "champion": len(champion_usage),
        },
        "sun_candidates_per_gpuh": {
            "final": final_efficiency,
            "champion": champion_efficiency,
        },
        "quality_compute_pareto": pareto,
        "compute_evidence_complete": bool(final_usage and champion_usage),
        "within_two_x_or_pareto": bool(
            actual_gpu_ratio is not None and (actual_gpu_ratio <= 2.0 or pareto)
        ),
    }


def aggregate_final(config: FinalAggregateConfig) -> dict[str, Any]:
    records = _read_jsonl(config.inputs)
    usage = _read_usage(config.usage_inputs)
    methods = {str(row["method"]) for row in records}
    if config.champion not in methods or config.final_method not in methods:
        raise ValueError("champion/final methods are absent from evaluation inputs")
    comparisons: dict[str, Any] = {}
    estimates: dict[str, PairedEstimate] = {}

    for stage in STAGES:
        final_rows = _select(records, method=config.final_method, evaluator="mattersim", stage=stage)
        champion_rows = _select(records, method=config.champion, evaluator="mattersim", stage=stage)
        estimates[f"mattersim_sun_at_0p1/{stage}"] = _paired_estimate(
            final_rows,
            champion_rows,
            metric="mlip_sun_at_0p1",
            repetitions=config.bootstrap_repetitions,
            seed=_seed_for(config.bootstrap_seed, f"stage/{stage}"),
        )

    headline_final = _select(
        records,
        method=config.final_method,
        evaluator="mattersim",
        stage=config.headline_stage,
    )
    headline_champion = _select(
        records,
        method=config.champion,
        evaluator="mattersim",
        stage=config.headline_stage,
    )
    estimates["novel_unique/headline"] = _paired_estimate(
        headline_final,
        headline_champion,
        metric="novel_unique_standard",
        repetitions=config.bootstrap_repetitions,
        seed=_seed_for(config.bootstrap_seed, "novel_unique"),
    )
    for sensitivity in SENSITIVITIES:
        estimates[f"matcher/{sensitivity}"] = _paired_estimate(
            headline_final,
            headline_champion,
            metric=f"matcher_sensitivity_sun_at_0p1.{sensitivity}",
            repetitions=config.bootstrap_repetitions,
            seed=_seed_for(config.bootstrap_seed, f"matcher/{sensitivity}"),
        )
    for changed in (False, True):
        label = "changed" if changed else "unchanged"
        estimates[f"orbit_count/{label}"] = _paired_estimate(
            headline_final,
            headline_champion,
            metric="mlip_sun_at_0p1",
            repetitions=config.bootstrap_repetitions,
            seed=_seed_for(config.bootstrap_seed, f"orbit/{label}"),
            predicate=lambda final, _champion, expected=changed: bool(
                final.get("orbit_count_changed")
            )
            == expected,
        )
    for training_seed in TRAINING_SEEDS:
        estimates[f"training_seed/{training_seed}"] = _paired_estimate(
            headline_final,
            headline_champion,
            metric="mlip_sun_at_0p1",
            repetitions=config.bootstrap_repetitions,
            seed=_seed_for(config.bootstrap_seed, f"train/{training_seed}"),
            predicate=lambda final, _champion, expected=training_seed: int(
                final["training_seed"]
            )
            == expected,
        )

    common_family_counts = Counter(
        str(final.get("material_family"))
        for final_by_pair, champion_by_pair in [
            (
                {str(row.get("pair_id")): row for row in headline_final},
                {str(row.get("pair_id")): row for row in headline_champion},
            )
        ]
        for pair_id, final in final_by_pair.items()
        if pair_id in champion_by_pair
        and final.get("material_family") == champion_by_pair[pair_id].get("material_family")
        and final.get("material_family") not in {None, "", "unknown"}
    )
    family_total = sum(common_family_counts.values())
    family_minimum = 1 if config.allow_nonpaper_counts else max(100, math.ceil(0.05 * family_total))
    major_families = sorted(
        family for family, count in common_family_counts.items() if count >= family_minimum
    )
    for family in major_families:
        estimates[f"material_family/{family}"] = _paired_estimate(
            headline_final,
            headline_champion,
            metric="mlip_sun_at_0p1",
            repetitions=config.bootstrap_repetitions,
            seed=_seed_for(config.bootstrap_seed, f"family/{family}"),
            predicate=lambda final, champion, expected=family: (
                final.get("material_family") == expected
                and champion.get("material_family") == expected
            ),
        )

    multi: dict[str, Any] = {}
    multi_rows: dict[str, list[dict[str, Any]]] = {}
    for method in (config.champion, config.final_method):
        current = _multi_mlip_rows(records, method, config.headline_stage)
        multi_rows[method] = current
        multi[method] = _multi_summary(current)
    unanimous = _paired_estimate(
        multi_rows[config.final_method],
        multi_rows[config.champion],
        metric="unanimous_mattersim_mace",
        repetitions=config.bootstrap_repetitions,
        seed=_seed_for(config.bootstrap_seed, "unanimous_mattersim_mace"),
    )
    estimates["unanimous_mattersim_mace/headline"] = unanimous

    for name, estimate in estimates.items():
        comparisons[name] = estimate.to_dict()
    secondary_names = [
        name
        for name in estimates
        if name.startswith(("matcher/", "orbit_count/", "material_family/"))
    ]
    holm = _holm({name: estimates[name].p_one_sided for name in secondary_names})
    for name, value in holm.items():
        comparisons[name]["holm_adjusted_p"] = value

    seed_counts: dict[str, dict[str, int]] = {}
    counts_registered = True
    for method, rows in (
        (config.champion, headline_champion),
        (config.final_method, headline_final),
    ):
        counts = Counter(int(row["training_seed"]) for row in rows)
        seed_counts[method] = {str(seed): counts.get(seed, 0) for seed in TRAINING_SEEDS}
        counts_registered = counts_registered and (
            len(rows) == 10_000
            and all(
                counts.get(seed, 0) == REGISTERED_SEED_COUNTS[seed]
                for seed in TRAINING_SEEDS
            )
        )
    all_stage_counts = all(
        len(_select(records, method=method, evaluator="mattersim", stage=stage)) == 10_000
        for method in (config.champion, config.final_method)
        for stage in STAGES
    )
    common_6k = all(multi[method]["attempts"] == 6_000 for method in multi)
    common_subset_aligned = (
        multi[config.champion]["pair_id_subset_sha256"]
        == multi[config.final_method]["pair_id_subset_sha256"]
    )
    matched_10k = estimates[
        f"mattersim_sun_at_0p1/{config.headline_stage}"
    ].pairs == 10_000
    counts_registered = (
        counts_registered
        and all_stage_counts
        and common_6k
        and common_subset_aligned
        and matched_10k
    )

    headline = estimates[f"mattersim_sun_at_0p1/{config.headline_stage}"]
    novelty = estimates["novel_unique/headline"]
    minimum_subgroup_pairs = 1 if config.allow_nonpaper_counts else 100
    gates = {
        "registered_attempt_counts": counts_registered,
        "mattersim_gain_at_least_5pp": bool(
            headline.estimate is not None and headline.estimate >= 0.05
        ),
        "mattersim_ci_low_at_least_2pp": bool(
            headline.ci_low is not None and headline.ci_low >= 0.02
        ),
        "mattersim_mace_unanimous_gain_at_least_3pp": bool(
            unanimous.estimate is not None and unanimous.estimate >= 0.03
        ),
        "three_training_seeds_positive": all(
            estimates[f"training_seed/{seed}"].estimate is not None
            and estimates[f"training_seed/{seed}"].estimate > 0.0
            for seed in TRAINING_SEEDS
        ),
        "raw_common_refiner_relaxed_positive": all(
            estimates[f"mattersim_sun_at_0p1/{stage}"].estimate is not None
            and estimates[f"mattersim_sun_at_0p1/{stage}"].estimate > 0.0
            for stage in STAGES
        ),
        "novel_unique_drop_no_more_than_2pp": bool(
            novelty.estimate is not None and novelty.estimate >= -0.02
        ),
        "matcher_sensitivity_all_positive": all(
            estimates[f"matcher/{value}"].estimate is not None
            and estimates[f"matcher/{value}"].estimate > 0.0
            for value in SENSITIVITIES
        ),
        "orbit_count_changed_and_unchanged_positive": all(
            estimates[f"orbit_count/{label}"].pairs >= minimum_subgroup_pairs
            and estimates[f"orbit_count/{label}"].estimate is not None
            and estimates[f"orbit_count/{label}"].estimate > 0.0
            for label in ("changed", "unchanged")
        ),
        "all_major_common_material_families_positive": bool(major_families)
        and all(
            estimates[f"material_family/{family}"].estimate is not None
            and estimates[f"material_family/{family}"].estimate > 0.0
            for family in major_families
        ),
    }
    compute = _compute_report(
        headline_final,
        headline_champion,
        usage,
        final_method=config.final_method,
        champion_method=config.champion,
    )
    gates["compute_within_two_x_or_pareto"] = compute["within_two_x_or_pareto"]
    gates["actual_compute_evidence_complete"] = compute["compute_evidence_complete"]

    reference = _reference_distribution(config.train_data)
    distributions = {
        method: _distribution_report(
            _select(
                records,
                method=method,
                evaluator="mattersim",
                stage=config.headline_stage,
            ),
            reference,
        )
        for method in (config.champion, config.final_method)
    }
    gates["distribution_and_symmetry_panel_complete"] = reference is not None and all(
        value is not None for value in distributions.values()
    )
    matcher_diagnostics: dict[str, Any] = {}
    for method, rows in (
        (config.champion, headline_champion),
        (config.final_method, headline_final),
    ):
        counts: Counter[str] = Counter()
        affected = 0
        for row in rows:
            values = {
                str(key): int(value)
                for key, value in row.get("matcher_diagnostics", {}).items()
            }
            affected += int(any(value > 0 for value in values.values()))
            counts.update(values)
        matcher_diagnostics[method] = {
            "attempts": len(rows),
            "attempts_with_timeout_or_error": affected,
            "attempt_rate": affected / max(len(rows), 1),
            "event_counts": dict(sorted(counts.items())),
            "conservative_policy": True,
        }
    oral_eligible = all(gates.values()) and not config.allow_nonpaper_counts
    poster_signal = bool(
        headline.estimate is not None
        and headline.estimate >= 0.02
        and headline.ci_low is not None
        and headline.ci_low > 0.0
        and novelty.estimate is not None
        and novelty.estimate >= -0.02
    )
    result = {
        "schema": "wqcodiff_final_aggregate_v1",
        "denominator": "all_submitted_attempts",
        "champion": config.champion,
        "final_method": config.final_method,
        "headline_evaluator": "mattersim",
        "headline_stage": config.headline_stage,
        "bootstrap": {
            "repetitions": config.bootstrap_repetitions,
            "levels": ["training_seed", "sampling_seed", "duplicate_cluster"],
            "paired_on": "method_independent_pair_id",
            "seed": config.bootstrap_seed,
        },
        "metric_table": _metric_table(records),
        "topology_and_revision_table": _mechanism_table(records),
        "multi_mlip": multi,
        "comparisons": comparisons,
        "major_common_material_families": {
            "definition": "same-family attempt pairs (failures retained), >=5% and >=100 pairs",
            "minimum_pairs": family_minimum,
            "assigned_common_pairs": family_total,
            "assignment_rate": family_total / max(headline.pairs, 1),
            "counts": dict(sorted(common_family_counts.items())),
            "major": major_families,
        },
        "seed_counts": seed_counts,
        "compute": compute,
        "distribution_and_symmetry": distributions,
        "matcher_diagnostics": matcher_diagnostics,
        "gates": gates,
        "oral_eligible": oral_eligible,
        "poster_signal_if_not_oral": poster_signal and not oral_eligible,
        "decision": (
            "oral_thresholds_met"
            if oral_eligible
            else "poster_level_signal_only"
            if poster_signal
            else "core_claim_not_supported"
        ),
    }
    output = Path(config.output)
    write_json_exclusive(output, result)
    return result
