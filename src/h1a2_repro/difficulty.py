"""Lightweight proposal/realization analysis for H1-A2 attempt ledgers.

The module deliberately uses only the Python standard library.  It normalizes
the small set of fields needed by the paper, removes evaluator replays by a
caller-supplied scientific cohort id, and estimates a cross-fitted difficulty
baseline without turning sparse exact formulas into headline strata.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HALOGENS = frozenset({"F", "Cl", "Br", "I"})
CHALCOGENS = frozenset({"S", "Se", "Te"})
COMMON_ANIONS = HALOGENS | CHALCOGENS | frozenset({"O", "N", "P", "As"})
METAL_EXCEPTIONS = frozenset({"H", "B", "C", "Si", "Ge", "Sb"}) | COMMON_ANIONS
PRIMARY_FEATURES = ("family", "arity", "n_bin", "all_metal")


def _first(row: Mapping[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        value: Any = row
        ok = True
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                ok = False
                break
            value = value[part]
        if ok and value is not None:
            return value
    return default


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def n_bin(num_atoms: int | None) -> str:
    if num_atoms is None or num_atoms <= 0:
        return "unknown"
    lower = ((int(num_atoms) - 1) // 4) * 4 + 1
    return f"{lower:02d}-{lower + 3:02d}"


def broad_family(elements: Sequence[str], *, all_metal: bool = False) -> str:
    symbols = frozenset(str(value) for value in elements if str(value))
    if all_metal:
        return "all_metal"
    has_o = "O" in symbols
    has_halogen = bool(symbols & HALOGENS)
    if has_o and has_halogen:
        return "oxyhalide"
    if has_o:
        return "oxide"
    if has_halogen:
        return "halide"
    if symbols & CHALCOGENS:
        return "chalcogenide"
    if "N" in symbols:
        return "nitride"
    if symbols & {"P", "As"}:
        return "pnictide"
    return "other"


def _plan_features(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _first(row, "planner_plan_features", "plan_state", "r5_plan_state", default={})
    return value if isinstance(value, Mapping) else {}


def _attempt_key(row: Mapping[str, Any], cohort_id: str) -> str:
    explicit = _first(
        row,
        "scientific_attempt_id",
        "plan_sha256",
        "plan_hash",
        "attempt_id",
        "id",
    )
    if explicit is not None:
        return f"{cohort_id}:{explicit}"
    features = _plan_features(row)
    identity = {
        "cohort": cohort_id,
        "repeat": _first(row, "repeat", "planner_repeat", default=0),
        "ordinal": _first(row, "ordinal", "cohort_ordinal", "planner_ordinal"),
        "formula": _first(features, "formula", default=_first(row, "formula", "reduced_formula")),
        "N": _first(features, "N", default=_first(row, "N", "num_atoms")),
    }
    return sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Attempt:
    key: str
    cohort_id: str
    method: str
    formula: str
    elements: tuple[str, ...]
    num_atoms: int | None
    family: str
    arity: str
    n_bin: str
    all_metal: str
    charge_bucket: str | None
    strict_sun: bool | None
    meta_sun: bool | None
    hull_known: bool | None
    raw: Mapping[str, Any]

    @property
    def reward(self) -> float | None:
        if self.hull_known is False:
            return None
        if self.strict_sun is None or self.meta_sun is None:
            return None
        return float(int(self.meta_sun) + int(self.strict_sun))

    def feature(self, name: str) -> str:
        value = getattr(self, name)
        return "unknown" if value is None else str(value)


def normalize_attempt(row: Mapping[str, Any], *, cohort_id: str, method: str) -> Attempt:
    features = _plan_features(row)
    elements_value = _first(features, "elements", default=_first(row, "elements", default=[]))
    elements = tuple(str(value) for value in elements_value) if isinstance(elements_value, Sequence) and not isinstance(elements_value, str) else ()
    num_atoms_value = _first(features, "N", default=_first(row, "N", "num_atoms"))
    try:
        num_atoms = int(num_atoms_value) if num_atoms_value is not None else None
    except (TypeError, ValueError):
        num_atoms = None
    charge = _first(features, "charge_bucket", default=_first(row, "charge_bucket"))
    all_metal = bool(str(charge).lower() == "all_metal" or (elements and not (set(elements) & COMMON_ANIONS) and not (set(elements) & METAL_EXCEPTIONS)))
    formula = str(_first(features, "formula", default=_first(row, "formula", "reduced_formula", default="")))
    strict = _bool_or_none(_first(row, "strict_sun_intersection", "strict_sun", "strict_stable"))
    meta = _bool_or_none(_first(row, "meta_sun_intersection", "meta_sun", "meta_stable"))
    hull_known = _bool_or_none(_first(row, "official_hull_known", "hull_known", "ehull_known"))
    if hull_known is None and _first(row, "official_e_above_hull", "e_above_hull") is not None:
        hull_known = True
    return Attempt(
        key=_attempt_key(row, cohort_id),
        cohort_id=str(cohort_id),
        method=str(method),
        formula=formula,
        elements=elements,
        num_atoms=num_atoms,
        family=broad_family(elements, all_metal=all_metal),
        arity=str(len(set(elements))) if elements else "unknown",
        n_bin=n_bin(num_atoms),
        all_metal="yes" if all_metal else "no",
        charge_bucket=None if charge is None else str(charge),
        strict_sun=strict,
        meta_sun=meta,
        hull_known=hull_known,
        raw=row,
    )


def load_jsonl(path: Path, *, cohort_id: str, method: str) -> list[Attempt]:
    attempts: list[Attempt] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                attempts.append(normalize_attempt(json.loads(line), cohort_id=cohort_id, method=method))
    return attempts


def deduplicate(attempts: Iterable[Attempt]) -> tuple[list[Attempt], int]:
    kept: dict[str, Attempt] = {}
    duplicate_count = 0
    for attempt in attempts:
        if attempt.key in kept:
            duplicate_count += 1
            continue
        kept[attempt.key] = attempt
    return list(kept.values()), duplicate_count


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (math.nan, math.nan)
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize(attempts: Sequence[Attempt], feature: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Attempt]] = defaultdict(list)
    for attempt in attempts:
        groups[attempt.feature(feature)].append(attempt)
    rows: list[dict[str, Any]] = []
    for value, group in sorted(groups.items()):
        known = [item for item in group if item.reward is not None]
        strict = sum(int(bool(item.strict_sun)) for item in known)
        meta = sum(int(bool(item.meta_sun)) for item in known)
        rows.append(
            {
                "feature": feature,
                "value": value,
                "attempts": len(group),
                "hull_known": len(known),
                "hull_unknown": len(group) - len(known),
                "strict": strict,
                "strict_rate": strict / len(known) if known else None,
                "strict_ci95": wilson_interval(strict, len(known)) if known else None,
                "meta": meta,
                "meta_rate": meta / len(known) if known else None,
                "meta_ci95": wilson_interval(meta, len(known)) if known else None,
            }
        )
    return rows


def _stable_fold(key: str, folds: int) -> int:
    return int(sha256(key.encode("utf-8")).hexdigest()[:16], 16) % int(folds)


def cross_fitted_difficulty(
    attempts: Sequence[Attempt],
    *,
    features: Sequence[str] = PRIMARY_FEATURES,
    folds: int = 5,
    prior_strength: float = 20.0,
) -> dict[str, float]:
    eligible = [item for item in attempts if item.reward is not None]
    if len(eligible) < max(2, folds):
        raise ValueError("not enough hull-known attempts for cross-fitted difficulty")
    predictions: dict[str, float] = {}
    for fold in range(folds):
        train = [item for item in eligible if _stable_fold(item.key, folds) != fold]
        test = [item for item in eligible if _stable_fold(item.key, folds) == fold]
        if not train:
            continue
        global_mean = sum(float(item.reward) for item in train) / len(train)
        level_stats: dict[tuple[str, str], tuple[float, int]] = {}
        accum: dict[tuple[str, str], list[float]] = defaultdict(list)
        for item in train:
            for feature in features:
                accum[(feature, item.feature(feature))].append(float(item.reward))
        for key, values in accum.items():
            shrunk = (sum(values) + prior_strength * global_mean) / (len(values) + prior_strength)
            level_stats[key] = (shrunk, len(values))
        for item in test:
            residuals = [
                level_stats.get((feature, item.feature(feature)), (global_mean, 0))[0] - global_mean
                for feature in features
            ]
            prediction = global_mean + (sum(residuals) / max(1, len(residuals)))
            predictions[item.key] = min(2.0, max(0.0, prediction))
    return predictions


def effective_sample_size(weights: Sequence[float]) -> float:
    total = sum(weights)
    square = sum(value * value for value in weights)
    return 0.0 if square <= 0.0 else total * total / square


def difficulty_weights(
    attempts: Sequence[Attempt],
    baselines: Mapping[str, float],
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    temperature: float = 1.0,
    max_weight: float = 5.0,
    min_ess_ratio: float = 0.5,
) -> tuple[dict[str, float], dict[str, Any]]:
    eligible = [item for item in attempts if item.reward is not None and item.key in baselines]
    if not eligible:
        raise ValueError("no eligible attempts with cross-fitted baselines")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    mean_baseline = sum(float(baselines[item.key]) for item in eligible) / len(eligible)

    def build(scale: float) -> list[float]:
        shift_factors: list[float] = []
        within_factors: list[float] = []
        strata: list[tuple[str, ...]] = []
        for item in eligible:
            baseline = float(baselines[item.key])
            advantage = float(item.reward) - baseline
            shift_exponent = scale * alpha * (baseline - mean_baseline) / temperature
            within_exponent = scale * beta * advantage / temperature
            shift_factors.append(math.exp(max(-20.0, min(20.0, shift_exponent))))
            within_factors.append(math.exp(max(-20.0, min(20.0, within_exponent))))
            strata.append(tuple(item.feature(feature) for feature in PRIMARY_FEATURES))
        within_by_stratum: dict[tuple[str, ...], list[float]] = defaultdict(list)
        for stratum, value in zip(strata, within_factors):
            within_by_stratum[stratum].append(value)
        within_means = {
            stratum: sum(values) / len(values)
            for stratum, values in within_by_stratum.items()
        }
        values = [
            min(max_weight, shift * within / within_means[stratum])
            for shift, within, stratum in zip(shift_factors, within_factors, strata)
        ]
        mean = sum(values) / len(values)
        return [value / mean for value in values]

    scale = 1.0
    weights = build(scale)
    target_ess = min_ess_ratio * len(weights)
    if effective_sample_size(weights) < target_ess:
        low, high = 0.0, 1.0
        for _ in range(40):
            mid = (low + high) / 2.0
            candidate = build(mid)
            if effective_sample_size(candidate) >= target_ess:
                low = mid
            else:
                high = mid
        scale = low
        weights = build(scale)
    result = {item.key: weight for item, weight in zip(eligible, weights)}
    report = {
        "count": len(weights),
        "alpha": alpha,
        "beta": beta,
        "factorization": "proposal_shift_times_within_stratum_normalized_advantage",
        "temperature": temperature,
        "scale": scale,
        "max_weight": max(weights),
        "min_weight": min(weights),
        "ess": effective_sample_size(weights),
        "ess_ratio": effective_sample_size(weights) / len(weights),
    }
    return result, report


def kitagawa_decomposition(
    baseline: Sequence[Attempt],
    candidate: Sequence[Attempt],
    *,
    feature: str = "family",
    endpoint: str = "meta_sun",
) -> dict[str, Any]:
    if endpoint not in {"strict_sun", "meta_sun"}:
        raise ValueError("endpoint must be strict_sun or meta_sun")

    def rates(items: Sequence[Attempt]) -> tuple[dict[str, float], dict[str, float]]:
        known = [item for item in items if item.reward is not None]
        total = max(1, len(known))
        groups: dict[str, list[Attempt]] = defaultdict(list)
        for item in known:
            groups[item.feature(feature)].append(item)
        mix = {key: len(value) / total for key, value in groups.items()}
        outcome = {
            key: sum(int(bool(getattr(item, endpoint))) for item in value) / len(value)
            for key, value in groups.items()
        }
        return mix, outcome

    p0, r0 = rates(baseline)
    p1, r1 = rates(candidate)
    common = sorted(set(p0) & set(p1))
    mix_effect = sum(0.5 * (p1[h] - p0[h]) * (r1[h] + r0[h]) for h in common)
    conditional_effect = sum(0.5 * (p1[h] + p0[h]) * (r1[h] - r0[h]) for h in common)
    return {
        "feature": feature,
        "endpoint": endpoint,
        "common_strata": common,
        "baseline_common_mass": sum(p0[h] for h in common),
        "candidate_common_mass": sum(p1[h] for h in common),
        "proposal_mix_effect": mix_effect,
        "conditional_realization_effect": conditional_effect,
        "common_support_total": mix_effect + conditional_effect,
    }


__all__ = [
    "Attempt",
    "PRIMARY_FEATURES",
    "broad_family",
    "cross_fitted_difficulty",
    "deduplicate",
    "difficulty_weights",
    "effective_sample_size",
    "kitagawa_decomposition",
    "load_jsonl",
    "n_bin",
    "normalize_attempt",
    "summarize",
    "wilson_interval",
]

