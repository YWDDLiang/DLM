#!/usr/bin/env python3
"""Reconcile a final Day-7 recovery cell killed by the Slurm time limit.

This utility is deliberately accounting-only: it never invokes the model and
never retries an attempt.  It may be used only after immutable ``sacct``
evidence says that the original lane reached ``TIMEOUT``, and only when the
timed cell is the final cell assigned to that lane.  Completed artifact rows
are retained byte-for-byte; every missing planned attempt is appended once as
``timeout`` so the registered denominator remains intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from crystal_dlm.wqcodiff.contracts import (
    ArtifactLedger,
    AttemptLedger,
    AttemptRecord,
    AttemptStatus,
    SeedDeriver,
    write_json_exclusive,
)
from crystal_dlm.wqcodiff.protocol import ACTIVE_PROTOCOL_NAME

from scripts.a800.run_day7_lane import _append_event, file_identity, sha256_file


TIMEOUT_EVIDENCE_SCHEMA = "wqcodiff_slurm_terminal_evidence_v1"


def _flag_values(argv: Sequence[str], flag: str) -> tuple[str, ...]:
    return tuple(argv[index + 1] for index, value in enumerate(argv[:-1]) if value == flag)


def _flag_value(argv: Sequence[str], flag: str) -> str:
    values = _flag_values(argv, flag)
    if len(values) != 1:
        raise ValueError(f"expected exactly one {flag}, observed {len(values)}")
    return values[0]


def _optional_flag_value(argv: Sequence[str], flag: str) -> str | None:
    values = _flag_values(argv, flag)
    if len(values) > 1:
        raise ValueError(f"expected at most one {flag}, observed {len(values)}")
    return values[0] if values else None


def _read_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}:{line_number}: dataset row is not an object")
                rows.append(payload)
    return rows


def _hash_fixed_records(
    paths: Sequence[str | Path], structures: int
) -> tuple[list[dict[str, Any]], str]:
    records = _read_jsonl(paths)
    records.sort(
        key=lambda row: hashlib.sha256(str(row["material_id"]).encode("utf-8")).hexdigest()
    )
    if structures > len(records):
        raise ValueError("timeout reconciliation requested more structures than the dataset")
    selected = records[:structures]
    subset_hash = hashlib.sha256(
        "\n".join(str(row["material_id"]) for row in selected).encode("utf-8")
    ).hexdigest()
    return selected, subset_hash


def _initial_field_count(record: Mapping[str, Any]) -> int:
    state = record["decompositions"]["symprec_1e-02"]["state"]
    committed = bool(state.get("space_group_committed", True))
    return 3 * len(state["orbits"]) + (0 if committed else 1)


def _load_timeout_evidence(path: str | Path, expected_job_id: str) -> dict[str, Any]:
    location = Path(path).resolve()
    payload = json.loads(location.read_text(encoding="utf-8"))
    if payload.get("schema") != TIMEOUT_EVIDENCE_SCHEMA:
        raise ValueError("invalid Slurm timeout evidence schema")
    if str(payload.get("job_id")) != str(expected_job_id):
        raise ValueError("Slurm timeout evidence job ID mismatch")
    if not str(payload.get("state", "")).upper().startswith("TIMEOUT"):
        raise ValueError("Slurm evidence does not record a TIMEOUT state")
    payload["artifact"] = file_identity(location)
    return payload


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "wqcodiff_day7_lane_event_v1":
                raise ValueError(f"invalid lane event at line {line_number}")
            rows.append(row)
    return rows


def _cell_config(job: Mapping[str, Any]) -> dict[str, Any]:
    argv = tuple(str(value) for value in job["argv"])
    return {
        "argv": argv,
        "protocol_path": _flag_value(argv, "--protocol"),
        "dataset_paths": _flag_values(argv, "--dataset"),
        "experiment_id": _flag_value(argv, "--experiment-id"),
        "pairing_id": _flag_value(argv, "--pairing-id"),
        "runtime_source_bundle_sha256": _optional_flag_value(
            argv, "--runtime-source-bundle-sha256"
        ),
        "method": _flag_value(argv, "--variant"),
        "training_seed": int(_flag_value(argv, "--training-seed")),
        "corruption_seed": int(_flag_value(argv, "--corruption-seed")),
        "structures": int(_flag_value(argv, "--structures")),
        "corruption_level": float(_flag_value(argv, "--corruption-level")),
        "operator": _flag_value(argv, "--operator"),
        "geometry_condition": _flag_value(argv, "--geometry-condition"),
        "schedule": _flag_value(argv, "--schedule"),
        "control": _flag_value(argv, "--control"),
        "revision_threshold": float(_flag_value(argv, "--revision-threshold")),
        "calls": int(_flag_value(argv, "--calls")),
        "inference_batch_size": int(_flag_value(argv, "--inference-batch-size")),
    }


def reconcile_recovery_cell_timeout(
    job: Mapping[str, Any],
    *,
    slurm_job_id: str,
    timeout_reason: str,
) -> dict[str, Any]:
    """Append terminal timeout rows for the uncompleted planned attempts."""

    config = _cell_config(job)
    records, subset_hash = _hash_fixed_records(
        config["dataset_paths"], config["structures"]
    )
    output = Path(str(job["output"]))
    ledger_path = Path(str(job["ledger"]))
    summary_path = output.with_suffix(".summary.json")
    if summary_path.exists():
        raise FileExistsError("timed recovery cell already has a summary")
    artifacts = ArtifactLedger(output)
    attempts = AttemptLedger(ledger_path)
    artifact_rows = artifacts.records()
    if not artifact_rows:
        raise ValueError("refusing timeout reconciliation without any completed artifact row")
    if len(artifact_rows) >= config["structures"]:
        raise ValueError("timed recovery cell already has the full artifact denominator")
    by_artifact = {str(row["attempt_id"]): row for row in artifact_rows}
    if len(by_artifact) != len(artifact_rows):
        raise ValueError("duplicate recovery artifact attempt ID before reconciliation")

    first = artifact_rows[0]
    checkpoint_sha256 = str(first["checkpoint_sha256"])
    source_bundle_sha256 = str(first["source_bundle_sha256"])
    runtime_source_bundle_sha256 = str(
        first.get(
            "runtime_source_bundle_sha256",
            config["runtime_source_bundle_sha256"] or source_bundle_sha256,
        )
    )
    if config["runtime_source_bundle_sha256"] not in {
        None,
        runtime_source_bundle_sha256,
    }:
        raise ValueError("partial artifact/runtime source-bundle SHA256 mismatch")
    if str(first["subset_hash"]) != subset_hash:
        raise ValueError("partial recovery artifact uses a different hash-fixed subset")

    deriver = SeedDeriver(ACTIVE_PROTOCOL_NAME, config["experiment_id"])
    pair_deriver = SeedDeriver(ACTIVE_PROTOCOL_NAME, config["pairing_id"])
    ledger_records = attempts.records()
    ledger_by_attempt: dict[str, list[AttemptRecord]] = {}
    for row in ledger_records:
        ledger_by_attempt.setdefault(row.attempt_id, []).append(row)

    expected_attempt_ids: list[str] = []
    timeout_attempt_ids: list[str] = []
    reason = f"SlurmTimeLimit:job_id={slurm_job_id}:{timeout_reason}"
    started = time.monotonic()
    for ordinal, record in enumerate(records):
        attempt_id = deriver.attempt_id(
            training_seed=config["training_seed"],
            sampling_seed=config["corruption_seed"],
            ordinal=ordinal,
            method=config["method"],
        )
        expected_attempt_ids.append(attempt_id)
        existing_ledger = ledger_by_attempt.get(attempt_id, [])
        terminal = [row for row in existing_ledger if row.status.terminal]
        artifact = by_artifact.get(attempt_id)
        if artifact is not None:
            if len(terminal) != 1:
                raise ValueError("completed artifact does not have exactly one terminal ledger row")
            continue
        if terminal:
            raise ValueError("terminal attempt lacks its immutable recovery artifact")

        seed = deriver.derive(
            training_seed=config["training_seed"],
            sampling_seed=config["corruption_seed"],
            attempt_id=attempt_id,
            stage="recovery",
        )
        pair_id = pair_deriver.pair_id(
            training_seed=config["training_seed"],
            sampling_seed=config["corruption_seed"],
            ordinal=ordinal,
        )
        paired_seed = pair_deriver.paired_derive(
            training_seed=config["training_seed"],
            sampling_seed=config["corruption_seed"],
            ordinal=ordinal,
            stage="recovery_corruption_and_reverse",
        )
        initial_fields = _initial_field_count(record)
        artifact_payload = {
            "schema": "wqcodiff_recovery_attempt_v1",
            "attempt_id": attempt_id,
            "material_id": record["material_id"],
            "method": config["method"],
            "training_seed": config["training_seed"],
            "corruption_seed": config["corruption_seed"],
            "ordinal": ordinal,
            "pair_id": pair_id,
            "paired_seed": paired_seed,
            "corruption_level": config["corruption_level"],
            "operator": config["operator"],
            "geometry_condition": config["geometry_condition"],
            "schedule": config["schedule"],
            "control": config["control"],
            "revision_threshold": config["revision_threshold"],
            "subset_hash": subset_hash,
            "checkpoint_sha256": checkpoint_sha256,
            "source_bundle_sha256": source_bundle_sha256,
            "runtime_source_bundle_sha256": runtime_source_bundle_sha256,
            "status": AttemptStatus.TIMEOUT.value,
            "applicable": True,
            "reason": reason,
            "initial_revisable_field_count": initial_fields,
            "mechanism": {
                "revision_selected_actions": 0,
                "wrong_to_right": 0,
                "right_to_wrong": 0,
                "net_correction": 0,
            },
            "calls": {"joint": 0, "bridge": 0, "projection": 0},
            "walltime_s": 0.0,
            "inference_batch_size": 0,
            "inference_batch_elapsed_s": 0.0,
            "walltime_allocation": "slurm_timeout_unexecuted_or_unfinished",
        }
        digest = artifacts.append(artifact_payload)
        attempts.append(
            AttemptRecord(
                attempt_id=attempt_id,
                method=config["method"],
                training_seed=config["training_seed"],
                sampling_seed=config["corruption_seed"],
                stage="recovery",
                status=AttemptStatus.TIMEOUT,
                reason=reason,
                artifact_hash=digest,
                seed=seed,
                calls={"joint": 0, "bridge": 0, "projection": 0},
                flops=0.0,
                walltime_s=0.0,
                metadata={
                    "ordinal": ordinal,
                    "pair_id": pair_id,
                    "paired_seed": paired_seed,
                    "subset_hash": subset_hash,
                    "checkpoint_sha256": checkpoint_sha256,
                    "source_bundle_sha256": source_bundle_sha256,
                    "runtime_source_bundle_sha256": runtime_source_bundle_sha256,
                    "slurm_job_id": str(slurm_job_id),
                    "timeout_reconciled": True,
                },
            )
        )
        timeout_attempt_ids.append(attempt_id)

    final_rows = artifacts.records()
    if len(final_rows) != config["structures"]:
        raise RuntimeError("timeout reconciliation did not restore the full denominator")
    audit = attempts.audit(
        seed_deriver=deriver,
        terminal_stage="recovery",
        expected_attempt_ids=expected_attempt_ids,
    )
    if not audit.ok:
        raise RuntimeError(f"attempt ledger audit failed after timeout reconciliation: {audit}")
    succeeded_rows = [row for row in final_rows if row.get("status") == "succeeded"]
    summary = {
        "ok": True,
        "schema": "wqcodiff_recovery_cell_v1",
        "protocol_name": ACTIVE_PROTOCOL_NAME,
        "protocol_sha256": sha256_file(Path(config["protocol_path"])),
        "method": config["method"],
        "pairing_id": config["pairing_id"],
        "structures": config["structures"],
        "succeeded": len(succeeded_rows),
        "failed": config["structures"] - len(succeeded_rows),
        "all_attempts_terminal": True,
        "all_attempts_succeeded": len(succeeded_rows) == config["structures"],
        "not_applicable": sum(not bool(row.get("applicable", True)) for row in final_rows),
        "subset_hash": subset_hash,
        "model_provenance": {
            "checkpoint_sha256": checkpoint_sha256,
            "source_bundle_sha256": source_bundle_sha256,
        },
        "runtime_source_bundle_sha256": runtime_source_bundle_sha256,
        "exact_recovery_attempt_rate": sum(
            bool(row.get("exact_full_protostructure_recovery", False)) for row in final_rows
        )
        / config["structures"],
        "exact_recovery_success_rate": sum(
            bool(row.get("exact_full_protostructure_recovery", False)) for row in succeeded_rows
        )
        / max(len(succeeded_rows), 1),
        "mean_edit_distance_before": None,
        "mean_edit_distance_after": None,
        "elapsed_s": time.monotonic() - started,
        "timeout_reconciliation": {
            "slurm_job_id": str(slurm_job_id),
            "reason": timeout_reason,
            "attempts": len(timeout_attempt_ids),
            "attempt_ids_sha256": hashlib.sha256(
                "\n".join(timeout_attempt_ids).encode("utf-8")
            ).hexdigest(),
            "no_model_invocation": True,
            "no_retry": True,
        },
        "cell": {
            "corruption_seed": config["corruption_seed"],
            "level": config["corruption_level"],
            "operator": config["operator"],
            "geometry_condition": config["geometry_condition"],
            "schedule": config["schedule"],
            "control": config["control"],
            "revision_threshold": config["revision_threshold"],
            "inference_batch_size": config["inference_batch_size"],
            "calls": config["calls"],
        },
    }
    write_json_exclusive(summary_path, summary)
    return {
        "summary": summary,
        "timeout_attempts": len(timeout_attempt_ids),
        "output_sha256": sha256_file(output),
        "summary_sha256": sha256_file(summary_path),
        "ledger_sha256": sha256_file(ledger_path),
    }


def reconcile_terminal_lane_timeout(
    *,
    manifest_path: str | Path,
    events_path: str | Path,
    complete_path: str | Path,
    timeout_evidence_path: str | Path,
    slurm_job_id: str,
) -> dict[str, Any]:
    manifest_location = Path(manifest_path).resolve()
    events_location = Path(events_path).resolve()
    complete_location = Path(complete_path).resolve()
    if complete_location.exists():
        raise FileExistsError("lane already has a complete manifest")
    lane = json.loads(manifest_location.read_text(encoding="utf-8"))
    if lane.get("schema") != "wqcodiff_day7_lane_plan_v1":
        raise ValueError("invalid Day-7 lane manifest")
    evidence = _load_timeout_evidence(timeout_evidence_path, slurm_job_id)
    events = _load_events(events_location)
    if not events or events[-1].get("event") != "started":
        raise ValueError("lane does not end in one unterminated started cell")
    started_cell = str(events[-1]["cell_id"])
    jobs = list(lane["jobs"])
    if not jobs or str(jobs[-1]["cell_id"]) != started_cell:
        raise ValueError("timeout reconciliation is allowed only for the lane's final cell")
    prior_terminal_list = [
        str(row["cell_id"]) for row in events if row.get("event") == "terminal"
    ]
    prior_terminal_cells = set(prior_terminal_list)
    expected_prior = {str(row["cell_id"]) for row in jobs[:-1]}
    if prior_terminal_cells != expected_prior or len(prior_terminal_list) != len(expected_prior):
        raise ValueError("prior lane cells are not exactly and uniquely terminal")
    if any(int(row.get("returncode", 1)) != 0 for row in events if row.get("event") == "terminal"):
        raise ValueError("a prior lane cell failed before the final Slurm timeout")

    result = reconcile_recovery_cell_timeout(
        jobs[-1],
        slurm_job_id=str(slurm_job_id),
        timeout_reason=str(evidence.get("reason", evidence.get("state"))),
    )
    _append_event(
        events_location,
        {
            "schema": "wqcodiff_day7_lane_event_v1",
            "event": "terminal",
            "cell_id": started_cell,
            "phase_ordinal": jobs[-1]["phase_ordinal"],
            "returncode": 124,
            "terminal_status": AttemptStatus.TIMEOUT.value,
            "timeout_attempts": result["timeout_attempts"],
            "output_sha256": result["output_sha256"],
            "summary_sha256": result["summary_sha256"],
            "ledger_sha256": result["ledger_sha256"],
            "timeout_evidence_sha256": evidence["artifact"]["sha256"],
        },
    )
    payload = {
        "schema": "wqcodiff_day7_lane_complete_v1",
        "ok": True,
        "phase": lane["phase"],
        "lane_index": lane["lane_index"],
        "cells": lane["cells"],
        "completed_cells": lane["cells"],
        "attempts": lane["attempts"],
        "backbone_calls": lane["backbone_calls"],
        "manifest_sha256": sha256_file(manifest_location),
        "events_sha256": sha256_file(events_location),
        "slurm_timeout_reconciled": True,
        "timeout_attempts": result["timeout_attempts"],
        "all_attempts_terminal": True,
        "all_attempts_succeeded": False,
        "planned_backbone_calls_include_timed_out_attempts": True,
        "no_retry": True,
        "timeout_evidence": evidence["artifact"],
    }
    write_json_exclusive(complete_location, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--complete", type=Path, required=True)
    parser.add_argument("--timeout-evidence", type=Path, required=True)
    parser.add_argument("--slurm-job-id", required=True)
    args = parser.parse_args()
    result = reconcile_terminal_lane_timeout(
        manifest_path=args.manifest,
        events_path=args.events,
        complete_path=args.complete,
        timeout_evidence_path=args.timeout_evidence,
        slurm_job_id=args.slurm_job_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
