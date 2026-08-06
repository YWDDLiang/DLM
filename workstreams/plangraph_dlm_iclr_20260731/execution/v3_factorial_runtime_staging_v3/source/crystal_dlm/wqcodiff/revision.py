"""Registered true-remask controller and validation-only threshold selection."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


REVISION_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
REVISION_FIELDS = ("existence", "wyckoff_type", "species")


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class FieldRef:
    orbit_id: str
    field: str

    def __post_init__(self) -> None:
        if not self.orbit_id or self.field not in REVISION_FIELDS:
            raise ValueError("invalid revisable field")


@dataclasses.dataclass(frozen=True, slots=True)
class RevisionDecision:
    selected: tuple[FieldRef, ...]
    threshold: float
    eligible: int
    remaining_total_budget: int


class RevisionBudget:
    """Enforce per-field, per-step, and total churn limits."""

    def __init__(self, initial_field_count: int) -> None:
        if initial_field_count <= 0:
            raise ValueError("initial_field_count must be positive")
        self.initial_field_count = int(initial_field_count)
        self.total_limit = max(1, int(math.floor(0.5 * initial_field_count)))
        self._total = 0
        self._counts: dict[FieldRef, int] = {}

    @property
    def total(self) -> int:
        return self._total

    @property
    def remaining(self) -> int:
        return self.total_limit - self._total

    @property
    def churn(self) -> float:
        return self._total / self.initial_field_count

    def count(self, field: FieldRef) -> int:
        return self._counts.get(field, 0)

    def select(
        self,
        scores: Mapping[FieldRef, float],
        *,
        threshold: float,
        current_field_count: int,
    ) -> RevisionDecision:
        if threshold not in REVISION_THRESHOLDS:
            raise ValueError("threshold is not in the registered grid")
        if current_field_count <= 0:
            raise ValueError("current_field_count must be positive")
        eligible = [
            (field, float(score))
            for field, score in scores.items()
            if math.isfinite(float(score))
            and float(score) >= threshold
            and self.count(field) < 2
        ]
        eligible.sort(key=lambda item: (-item[1], item[0].orbit_id, item[0].field))
        per_step = max(1, int(math.ceil(0.1 * current_field_count)))
        count = min(per_step, self.remaining, len(eligible))
        selected = tuple(field for field, _ in eligible[:count])
        for field in selected:
            self._counts[field] = self.count(field) + 1
        self._total += len(selected)
        return RevisionDecision(
            selected=selected,
            threshold=threshold,
            eligible=len(eligible),
            remaining_total_budget=self.remaining,
        )

    def preview(
        self,
        scores: Mapping[FieldRef, float],
        *,
        threshold: float,
        current_field_count: int,
    ) -> RevisionDecision:
        """Return the registered selection without consuming any budget."""

        counts = dict(self._counts)
        total = self._total
        try:
            return self.select(
                scores,
                threshold=threshold,
                current_field_count=current_field_count,
            )
        finally:
            self._counts = counts
            self._total = total


@dataclasses.dataclass(frozen=True, slots=True)
class ThresholdResult:
    threshold: float
    clean_false_remask_rate: float
    net_correction: int


def select_revision_threshold(
    *,
    clean_scores: Sequence[float],
    wrong_scores: Sequence[float],
    right_to_wrong_if_selected: Mapping[float, int] | None = None,
    wrong_to_right_if_selected: Mapping[float, int] | None = None,
) -> ThresholdResult:
    """Apply the preregistered Day-7 threshold rule exactly.

    Choose among thresholds with clean false-remask <= 5%; maximize net
    correction, then prefer the higher threshold to avoid unregistered churn.
    """

    if not clean_scores or not wrong_scores:
        raise ValueError("clean and wrong validation scores are required")
    right_to_wrong_if_selected = right_to_wrong_if_selected or {}
    wrong_to_right_if_selected = wrong_to_right_if_selected or {}
    candidates: list[ThresholdResult] = []
    for threshold in REVISION_THRESHOLDS:
        false_rate = sum(float(value) >= threshold for value in clean_scores) / len(clean_scores)
        default_wrong = sum(float(value) >= threshold for value in wrong_scores)
        wrong_to_right = int(wrong_to_right_if_selected.get(threshold, default_wrong))
        right_to_wrong = int(
            right_to_wrong_if_selected.get(
                threshold,
                sum(float(value) >= threshold for value in clean_scores),
            )
        )
        if false_rate <= 0.05:
            candidates.append(
                ThresholdResult(threshold, false_rate, wrong_to_right - right_to_wrong)
            )
    if not candidates:
        # This is a gate failure rather than an off-grid threshold escape hatch.
        raise ValueError("no registered threshold satisfies clean false-remask <= 5%")
    return max(candidates, key=lambda result: (result.net_correction, result.threshold))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibrate_revision_threshold_from_recovery(
    paths: Sequence[str | Path],
    *,
    output: str | Path,
    protocol_name: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Freeze the Day-7 threshold from attempt-level clean/corrupt recovery rows."""

    rows: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        source_artifacts.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema") != "wqcodiff_recovery_attempt_v1":
                    raise ValueError(f"{path}:{line_number}: invalid recovery schema")
                if row.get("method") != "M-WQ-STRAT-GEO":
                    raise ValueError("threshold calibration only accepts M-WQ-STRAT-GEO")
                if row.get("schedule") != "geometry-adaptive" or row.get("control") != "none":
                    raise ValueError("threshold calibration requires geometry-adaptive/no-control cells")
                threshold = float(row.get("revision_threshold", -1.0))
                if threshold not in REVISION_THRESHOLDS:
                    raise ValueError("calibration row has an off-grid or missing threshold")
                rows.append(row)
    if not rows:
        raise ValueError("no recovery attempts were supplied for threshold calibration")
    observed = {float(row["revision_threshold"]) for row in rows}
    if observed != set(REVISION_THRESHOLDS):
        raise ValueError(f"threshold grid is incomplete: {sorted(observed)}")

    fingerprints: dict[float, dict[str, set[tuple[Any, ...]]]] = {
        threshold: {"clean": set(), "corrupt": set()}
        for threshold in REVISION_THRESHOLDS
    }
    table: list[dict[str, Any]] = []
    for threshold in REVISION_THRESHOLDS:
        selected = [row for row in rows if float(row["revision_threshold"]) == threshold]
        clean = [row for row in selected if row.get("operator") == "none"]
        corrupt = [row for row in selected if row.get("operator") != "none"]
        if not clean or not corrupt:
            raise ValueError(f"threshold {threshold} lacks clean or corrupt attempts")
        for kind, values in (("clean", clean), ("corrupt", corrupt)):
            fingerprints[threshold][kind] = {
                (
                    row.get("material_id"),
                    int(row.get("corruption_seed")),
                    float(row.get("corruption_level")),
                    row.get("operator"),
                    row.get("geometry_condition"),
                )
                for row in values
            }
        clean_fields = sum(
            max(1, int(row.get("initial_revisable_field_count") or 1)) for row in clean
        )
        false_remasks = sum(
            (
                max(1, int(row.get("initial_revisable_field_count") or 1))
                if row.get("status") != "succeeded"
                else int(row.get("mechanism", {}).get("revision_selected_actions", 0))
            )
            for row in clean
        )
        net_values = [
            (
                float(row.get("mechanism", {}).get("net_correction", 0.0))
                if row.get("status") == "succeeded"
                else 0.0
            )
            for row in corrupt
        ]
        table.append(
            {
                "threshold": threshold,
                "clean_attempts": len(clean),
                "clean_revisable_fields": clean_fields,
                "clean_false_remasks": false_remasks,
                "clean_false_remask_rate": false_remasks / clean_fields,
                "corrupt_attempts": len(corrupt),
                "mean_net_correction_attempt_denominator": sum(net_values) / len(net_values),
                "wrong_to_right": sum(
                    int(row.get("mechanism", {}).get("wrong_to_right", 0))
                    for row in corrupt
                    if row.get("status") == "succeeded"
                ),
                "right_to_wrong": sum(
                    int(row.get("mechanism", {}).get("right_to_wrong", 0))
                    for row in corrupt
                    if row.get("status") == "succeeded"
                ),
            }
        )
    baseline = fingerprints[REVISION_THRESHOLDS[0]]
    for threshold in REVISION_THRESHOLDS[1:]:
        if fingerprints[threshold] != baseline:
            raise ValueError("threshold cells do not use identical hash-fixed attempts")
    eligible = [row for row in table if row["clean_false_remask_rate"] <= 0.05]
    if not eligible:
        raise ValueError("no threshold satisfies clean false-remask <= 5%")
    winner = max(
        eligible,
        key=lambda row: (
            row["mean_net_correction_attempt_denominator"],
            row["threshold"],
        ),
    )
    result = {
        "schema": "wqcodiff_revision_threshold_lock_v1",
        "protocol_name": protocol_name,
        "protocol_sha256": protocol_sha256,
        "frozen_day": 7,
        "selection_rule": "clean_false_remask_le_0p05_then_max_net_correction_then_higher_threshold",
        "selected_threshold": winner["threshold"],
        "selected_clean_false_remask_rate": winner["clean_false_remask_rate"],
        "selected_mean_net_correction": winner[
            "mean_net_correction_attempt_denominator"
        ],
        "threshold_table": table,
        "source_artifacts": source_artifacts,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def load_revision_threshold_lock(
    path: str | Path,
    *,
    protocol_name: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "wqcodiff_revision_threshold_lock_v1":
        raise ValueError("invalid revision threshold lock schema")
    if (
        payload.get("protocol_name") != protocol_name
        or payload.get("protocol_sha256") != protocol_sha256
    ):
        raise ValueError("revision threshold lock/protocol mismatch")
    threshold = float(payload.get("selected_threshold", -1.0))
    if threshold not in REVISION_THRESHOLDS:
        raise ValueError("revision threshold lock selected an off-grid value")
    return payload
