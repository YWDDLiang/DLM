"""Deterministic Day-14 comparator selection from matched validation attempts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_CONFIGURATIONS = {
    "best-discrete-engine",
    "joint-no-revision",
    "disc-once-tau-0p25",
    "disc-once-tau-0p5",
    "disc-once-tau-0p75",
    "disc-once-tau-1p0",
    "atom-joint",
    "stratified-geometry",
}
COMPARATOR_CONFIGURATIONS = {
    "best-discrete-engine",
    "joint-no-revision",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    location = Path(path).resolve()
    return {
        "path": str(location),
        "bytes": location.stat().st_size,
        "sha256": sha256_file(location),
    }


def parse_named_path(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path or name in result:
            raise ValueError("evaluation inputs must be unique CONFIG=PATH entries")
        result[name] = Path(raw_path).resolve()
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "wqcodiff_mlip_sun_attempt_v1":
                raise ValueError(f"{path}:{line_number}: invalid MLIP-SUN schema")
            rows.append(row)
    if not rows:
        raise ValueError(f"screening evaluation is empty: {path}")
    return rows


def _identity_set(rows: Sequence[Mapping[str, Any]], key: str) -> set[str]:
    return {str(row.get(key) or "") for row in rows}


def _ranking_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, str]:
    return (
        -float(row["mlip_sun_at_0p1"]),
        -float(row["novel_unique_standard"]),
        -float(row["mlip_sun_at_0p0"]),
        float(row["failure_rate"]),
        float(row["mean_generation_calls"]),
        str(row["configuration_id"]),
    )


def freeze_week2_champion(
    *,
    sampling_plan_path: str | Path,
    evaluation_paths: Mapping[str, str | Path],
    output: str | Path,
    expected_attempts_per_configuration: int = 3000,
    expected_attempts_per_sampling_seed: int = 1000,
    allow_nonpaper_attempts: bool = False,
) -> dict[str, Any]:
    """Freeze DISC-ONCE tau and the strongest eligible WQ comparator."""

    if (
        expected_attempts_per_configuration != 3000
        or expected_attempts_per_sampling_seed != 1000
    ) and not allow_nonpaper_attempts:
        raise ValueError("paper screening requires 3 x 1000 attempts per configuration")
    if expected_attempts_per_configuration != 3 * expected_attempts_per_sampling_seed:
        raise ValueError("screening denominator must be three equal sampling seeds")
    plan_path = Path(sampling_plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "wqcodiff_week2_sampling_plan_v1":
        raise ValueError("unsupported Week-2 sampling plan schema")
    development_jobs = [job for job in plan["jobs"] if job.get("phase") == "development"]
    specifications: dict[str, dict[str, Any]] = {}
    for job in development_jobs:
        configuration_id = str(job["configuration_id"])
        candidate = {
            "route": str(job["route"]),
            "variant": str(job["variant"]),
            "disc_once_tau": job.get("disc_once_tau"),
            "pairing_id": str(job["pairing_id"]),
        }
        previous = specifications.setdefault(configuration_id, candidate)
        if previous != candidate:
            raise ValueError("sampling plan configuration metadata is inconsistent")
    if set(specifications) != EXPECTED_CONFIGURATIONS:
        raise ValueError("sampling plan lacks the frozen eight configurations")
    supplied = {str(name): Path(path).resolve() for name, path in evaluation_paths.items()}
    if set(supplied) != EXPECTED_CONFIGURATIONS:
        raise ValueError("champion selection requires exactly eight evaluation artifacts")

    metrics: list[dict[str, Any]] = []
    common_pair_ids: set[str] | None = None
    common_contract: dict[str, str] | None = None
    source_artifacts: list[dict[str, Any]] = []
    for configuration_id in sorted(EXPECTED_CONFIGURATIONS):
        path = supplied[configuration_id]
        rows = _read_jsonl(path)
        source_artifacts.append({"configuration_id": configuration_id, **file_identity(path)})
        if len(rows) != expected_attempts_per_configuration:
            raise ValueError(f"{configuration_id} has the wrong attempt denominator")
        attempt_ids = _identity_set(rows, "attempt_id")
        pair_ids = _identity_set(rows, "pair_id")
        if "" in attempt_ids or "" in pair_ids:
            raise ValueError("screening rows require attempt_id and pair_id")
        if len(attempt_ids) != len(rows) or len(pair_ids) != len(rows):
            raise ValueError("screening evaluation contains duplicate attempt or pair IDs")
        if _identity_set(rows, "status") - {"succeeded", "failed"}:
            raise ValueError("screening evaluation contains a nonterminal status")
        if common_pair_ids is None:
            common_pair_ids = pair_ids
        elif pair_ids != common_pair_ids:
            raise ValueError("screening configurations do not use identical matched pairs")
        seed_counts: dict[int, int] = {}
        for row in rows:
            seed = int(row.get("sampling_seed", -1))
            seed_counts[seed] = seed_counts.get(seed, 0) + 1
        if seed_counts != {
            101: expected_attempts_per_sampling_seed,
            202: expected_attempts_per_sampling_seed,
            303: expected_attempts_per_sampling_seed,
        }:
            raise ValueError("screening sampling-seed counts differ from the frozen split")
        if {int(row.get("training_seed", -1)) for row in rows} != {11}:
            raise ValueError("Week-2 screening only accepts training seed 11")
        if _identity_set(rows, "method") != {specifications[configuration_id]["variant"]}:
            raise ValueError("screening method differs from its sampling-plan configuration")
        expected_tau = specifications[configuration_id]["disc_once_tau"]
        observed_tau = {row.get("disc_once_tau") for row in rows}
        if observed_tau != {expected_tau}:
            raise ValueError("screening DISC-ONCE tau metadata is missing or mismatched")
        if _identity_set(rows, "pairing_id") != {
            specifications[configuration_id]["pairing_id"]
        }:
            raise ValueError("screening pairing namespace differs from its frozen plan")
        if _identity_set(rows, "evaluator") != {"mattersim"}:
            raise ValueError("champion selection is frozen to MatterSim")
        if _identity_set(rows, "stage") != {"raw"}:
            raise ValueError("champion selection is frozen to the raw stage")
        contract = {
            key: next(iter(_identity_set(rows, key)))
            for key in (
                "contract_hash",
                "hull_sha256",
                "novelty_reference_sha256",
                "matcher_contract_sha256",
                "subset_hash",
            )
        }
        if any(not value for value in contract.values()):
            raise ValueError("screening evaluation lacks a frozen evaluator identity")
        for key in contract:
            if len(_identity_set(rows, key)) != 1:
                raise ValueError(f"screening artifact mixes {key} values")
        if common_contract is None:
            common_contract = contract
        elif contract != common_contract:
            raise ValueError("screening artifacts use different evaluator/reference contracts")

        succeeded = sum(row.get("status") == "succeeded" for row in rows)
        total_calls = 0.0
        for row in rows:
            calls = row.get("generation_calls") or {}
            total_calls += sum(float(value) for value in calls.values())
        denominator = len(rows)
        metrics.append(
            {
                "configuration_id": configuration_id,
                **specifications[configuration_id],
                "attempts": denominator,
                "succeeded": succeeded,
                "mlip_sun_at_0p1": sum(bool(row.get("mlip_sun_at_0p1")) for row in rows)
                / denominator,
                "novel_unique_standard": sum(
                    bool(row.get("novel_unique_standard")) for row in rows
                )
                / denominator,
                "mlip_sun_at_0p0": sum(bool(row.get("mlip_sun_at_0p0")) for row in rows)
                / denominator,
                "failure_rate": (denominator - succeeded) / denominator,
                "mean_generation_calls": total_calls / denominator,
            }
        )

    by_id = {row["configuration_id"]: row for row in metrics}
    disc_once_winner = min(
        (row for row in metrics if row["route"] == "disc-once"),
        key=_ranking_key,
    )
    eligible_ids = set(COMPARATOR_CONFIGURATIONS) | {
        str(disc_once_winner["configuration_id"])
    }
    champion = min((by_id[value] for value in eligible_ids), key=_ranking_key)

    training_plan_path = Path(str(plan["training_plan"]["path"])).resolve()
    if file_identity(training_plan_path) != dict(plan["training_plan"]):
        raise ValueError("Week-2 training plan changed after sampling-plan freeze")
    training_plan = json.loads(training_plan_path.read_text(encoding="utf-8"))
    route_jobs = {
        str(job.get("route")): job
        for job in training_plan["jobs"]
        if job.get("phase") == "screen-60000-to-85000"
    }

    def continuation_identity(route: str) -> dict[str, Any]:
        path = Path(str(route_jobs[route]["continuation_checkpoint"]))
        if not path.is_absolute():
            project_root = Path(str(plan.get("project_root") or "")).resolve()
            if not str(plan.get("project_root") or ""):
                raise ValueError("sampling plan lacks its project root")
            path = project_root / path
        return file_identity(path)

    pair_set_sha256 = hashlib.sha256(
        "\n".join(sorted(common_pair_ids or set())).encode("utf-8")
    ).hexdigest()
    result = {
        "schema": "wqcodiff_week2_champion_lock_v1",
        "frozen_day": 14,
        "run_id": plan["run_id"],
        "protocol_sha256": plan["protocol_sha256"],
        "source_bundle_sha256": plan["source_bundle_sha256"],
        "selection_evaluator": "mattersim",
        "selection_stage": "raw",
        "selection_rule": (
            "max_sun0p1_then_novel_unique_then_sun0p0_then_lower_failure_"
            "then_lower_calls_then_lexicographically_lower_configuration_id"
        ),
        "eligible_comparator_configurations": sorted(eligible_ids),
        "selected_disc_once_configuration": disc_once_winner["configuration_id"],
        "selected_disc_once_tau": disc_once_winner["disc_once_tau"],
        "selected_champion": champion,
        "selected_champion_continuation_checkpoint": continuation_identity(
            str(champion["route"])
        ),
        "stratified_geometry_continuation_checkpoint": continuation_identity(
            "stratified-geometry"
        ),
        "atom_baseline_is_champion_eligible": False,
        "stratified_geometry_is_champion_eligible": False,
        "matched_pair_set_sha256": pair_set_sha256,
        "evaluator_contract": common_contract,
        "metrics": sorted(metrics, key=lambda row: row["configuration_id"]),
        "sampling_plan": file_identity(plan_path),
        "source_artifacts": source_artifacts,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result
