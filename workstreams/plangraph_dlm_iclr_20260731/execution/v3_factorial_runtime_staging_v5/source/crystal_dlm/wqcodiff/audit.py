"""Cross-artifact integrity audit for registered paper evidence."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import AttemptLedger, SeedDeriver, write_json_exclusive
from .mlip import EvaluatorLock, sha256_file
from .protocol import load_protocol
from .revision import load_revision_threshold_lock


@dataclasses.dataclass(frozen=True, slots=True)
class WorkflowAuditConfig:
    ledgers: tuple[tuple[str, str], ...]
    artifacts: tuple[str, ...]
    output: str
    formal_reports: tuple[str, ...] = ()
    dataset_reports: tuple[str, ...] = ()
    final_aggregate: str | None = None
    source_manifest: str | None = None
    asset_lock: str | None = None
    model_root: str | None = None
    revision_lock: str | None = None


def _ledger_audit(
    experiment_id: str,
    path: str | Path,
    *,
    protocol_name: str,
) -> tuple[dict[str, Any], set[str]]:
    ledger = AttemptLedger(path)
    records = ledger.records()
    deriver = SeedDeriver(protocol_name, experiment_id)
    by_key: dict[tuple[str, str], list[Any]] = defaultdict(list)
    bad_seed: list[tuple[str, str]] = []
    artifact_hashes: set[str] = set()
    for record in records:
        by_key[record.key].append(record)
        expected = deriver.derive(
            training_seed=record.training_seed,
            sampling_seed=record.sampling_seed,
            attempt_id=record.attempt_id,
            stage=record.stage,
        )
        if record.seed != expected:
            bad_seed.append(record.key)
        if record.artifact_hash:
            artifact_hashes.add(record.artifact_hash)
    lifecycle_errors: list[dict[str, Any]] = []
    for key, values in sorted(by_key.items()):
        counts = Counter(value.status.value for value in values)
        terminal = [value for value in values if value.status.terminal]
        if counts.get("submitted", 0) != 1 or len(terminal) != 1 or any(
            count > 1 for count in counts.values()
        ):
            lifecycle_errors.append(
                {"attempt_id": key[0], "stage": key[1], "statuses": dict(counts)}
            )
    pair_seed_values: dict[str, set[int]] = defaultdict(set)
    for record in records:
        pair_id = record.metadata.get("pair_id")
        paired_seed = record.metadata.get("paired_seed")
        if pair_id and paired_seed is not None:
            pair_seed_values[str(pair_id)].add(int(paired_seed))
    pair_conflicts = {
        pair_id: sorted(values)
        for pair_id, values in pair_seed_values.items()
        if len(values) != 1
    }
    result = {
        "path": str(Path(path).resolve()),
        "experiment_id": experiment_id,
        "records": len(records),
        "attempt_stages": len(by_key),
        "attempts": len({record.attempt_id for record in records}),
        "lifecycle_errors": lifecycle_errors,
        "seed_mismatches": sorted(set(bad_seed)),
        "pair_seed_conflicts": pair_conflicts,
    }
    result["ok"] = not lifecycle_errors and not bad_seed and not pair_conflicts
    return result, artifact_hashes


def _artifact_audit(paths: Sequence[str | Path]) -> tuple[dict[str, Any], set[str]]:
    seen_keys: set[tuple[Any, ...]] = set()
    duplicate_keys: list[tuple[Any, ...]] = []
    line_hashes: set[str] = set()
    schemas: Counter[str] = Counter()
    revision_lock_hashes: set[str] = set()
    revision_rows_missing_lock: list[tuple[str, int, str]] = []
    records = 0
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    raise ValueError(f"{path}:{line_number}: invalid artifact JSON: {exc}") from exc
                schema = str(row.get("schema") or "")
                schemas[schema] += 1
                records += 1
                revision_hash = row.get("revision_lock_sha256")
                if revision_hash:
                    revision_lock_hashes.add(str(revision_hash))
                if (
                    schema == "wqcodiff_generation_attempt_v1"
                    and row.get("revision_control") not in {None, "none"}
                    and not revision_hash
                ):
                    revision_rows_missing_lock.append(
                        (str(path.resolve()), line_number, str(row.get("attempt_id")))
                    )
                line_hashes.add(hashlib.sha256(line).hexdigest())
                if "attempt_id" in row:
                    key = (
                        schema,
                        row.get("attempt_id"),
                        row.get("evaluator"),
                        row.get("stage"),
                        row.get("hull_sha256"),
                        row.get("corruption_level"),
                        row.get("operator"),
                        row.get("geometry_condition"),
                        row.get("schedule"),
                        row.get("control"),
                    )
                elif "material_id" in row:
                    key = (
                        schema,
                        row.get("material_id"),
                        row.get("evaluator"),
                        row.get("stage"),
                    )
                else:
                    key = (schema, str(path.resolve()), line_number)
                if key in seen_keys:
                    duplicate_keys.append(key)
                seen_keys.add(key)
    result = {
        "files": [str(Path(path).resolve()) for path in paths],
        "records": records,
        "schemas": dict(sorted(schemas.items())),
        "duplicate_keys": duplicate_keys,
        "revision_lock_hashes": sorted(revision_lock_hashes),
        "revision_rows_missing_lock": revision_rows_missing_lock,
        "ok": not duplicate_keys,
    }
    return result, line_hashes


def _report_audit(paths: Sequence[str | Path], expected_field: str) -> list[dict[str, Any]]:
    results = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        passed = bool(payload.get(expected_field) or payload.get("ok"))
        results.append(
            {
                "path": str(path.resolve()),
                "schema": payload.get("schema"),
                "passed": passed,
                "sha256": sha256_file(path),
            }
        )
    return results


def _source_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "wqcodiff_source_manifest_v1":
        raise ValueError("invalid source manifest schema")
    root = Path(payload["source_root"])
    errors: list[str] = []
    for entry in payload["files"]:
        location = root / entry["path"]
        if not location.is_file():
            errors.append(f"missing:{entry['path']}")
        elif sha256_file(location) != entry["sha256"]:
            errors.append(f"hash:{entry['path']}")
        elif location.stat().st_size != int(entry["bytes"]):
            errors.append(f"size:{entry['path']}")
    return {
        "path": str(manifest_path),
        "files": len(payload["files"]),
        "errors": errors,
        "ok": not errors,
    }


def _required_claim_evidence(
    *,
    ledger_results: Sequence[Mapping[str, Any]],
    artifact_result: Mapping[str, Any],
    formal_reports: Sequence[Mapping[str, Any]],
    dataset_reports: Sequence[Mapping[str, Any]],
    source_result: Mapping[str, Any] | None,
    asset_result: Mapping[str, Any] | None,
    revision_result: Mapping[str, Any] | None,
    aggregate_result: Mapping[str, Any] | None,
) -> dict[str, bool]:
    """Return the non-optional evidence checklist for a paper headline claim.

    ``integrity_passed`` can be useful for partial/day-level audits.  It must not,
    however, make an incomplete evidence bundle eligible for the final claim.
    Schemas are checked explicitly so an unrelated report with ``ok=true`` cannot
    stand in for either of the two formal gates or the P1 dataset audit.
    """

    passed_formal_schemas = {
        str(report.get("schema"))
        for report in formal_reports
        if bool(report.get("passed"))
    }
    passed_dataset_schemas = {
        str(report.get("schema"))
        for report in dataset_reports
        if bool(report.get("passed"))
    }
    required_formal = {
        "wqcodiff_formal_audit_v1",
        "wqcodiff_pyxtal_chart_audit_v1",
    }
    return {
        "attempt_ledgers": bool(ledger_results)
        and all(bool(result.get("ok")) for result in ledger_results),
        "attempt_artifacts": bool(artifact_result.get("records", 0))
        and bool(artifact_result.get("ok")),
        "formal_and_chart_gates": required_formal <= passed_formal_schemas,
        "p1_dataset_gate": "wqcodiff_p1_dataset_audit_v1"
        in passed_dataset_schemas,
        "source_manifest": bool(source_result and source_result.get("ok")),
        "mlip_asset_lock": bool(asset_result and asset_result.get("ok")),
        "revision_threshold_lock": bool(
            revision_result and revision_result.get("ok")
        ),
        "final_aggregate": aggregate_result is not None,
    }


def audit_workflow(
    config: WorkflowAuditConfig,
    *,
    protocol_path: str | Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    ledger_results = []
    referenced_hashes: set[str] = set()
    for experiment_id, path in config.ledgers:
        result, hashes = _ledger_audit(
            experiment_id,
            path,
            protocol_name=protocol.name,
        )
        ledger_results.append(result)
        referenced_hashes.update(hashes)
    artifact_result, artifact_hashes = _artifact_audit(config.artifacts)
    missing_artifacts = sorted(referenced_hashes - artifact_hashes)
    formal = _report_audit(config.formal_reports, "passed")
    dataset = _report_audit(config.dataset_reports, "ok")

    aggregate_result: dict[str, Any] | None = None
    if config.final_aggregate:
        path = Path(config.final_aggregate).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "wqcodiff_final_aggregate_v1":
            raise ValueError("invalid final aggregate schema")
        aggregate_result = {
            "path": str(path),
            "sha256": sha256_file(path),
            "oral_eligible": bool(payload.get("oral_eligible")),
            "decision": payload.get("decision"),
            "gates": payload.get("gates"),
        }

    source_result = _source_manifest(config.source_manifest) if config.source_manifest else None
    asset_result: dict[str, Any] | None = None
    if config.asset_lock or config.model_root:
        if not config.asset_lock or not config.model_root:
            raise ValueError("asset_lock and model_root must be provided together")
        lock = EvaluatorLock.load(config.asset_lock)
        assets = {
            evaluator: lock.verify(
                evaluator,
                config.model_root,
                verify_installed=False,
            ).checkpoint_sha256
            for evaluator in ("chgnet", "mattersim", "mace")
        }
        asset_result = {
            "path": str(Path(config.asset_lock).resolve()),
            "sha256": sha256_file(config.asset_lock),
            "assets": assets,
            "ok": True,
        }
    revision_result: dict[str, Any] | None = None
    if config.revision_lock:
        lock_path = Path(config.revision_lock).resolve()
        lock = load_revision_threshold_lock(
            lock_path,
            protocol_name=protocol.name,
            protocol_sha256=protocol.sha256,
        )
        revision_result = {
            "path": str(lock_path),
            "sha256": sha256_file(lock_path),
            "selected_threshold": lock["selected_threshold"],
            "clean_false_remask_rate": lock["selected_clean_false_remask_rate"],
            "ok": float(lock["selected_clean_false_remask_rate"]) <= 0.05,
        }
        observed_locks = set(artifact_result["revision_lock_hashes"])
        expected_lock = revision_result["sha256"]
        revision_result["artifact_lock_hashes"] = sorted(observed_locks)
        revision_result["artifact_rows_missing_lock"] = artifact_result[
            "revision_rows_missing_lock"
        ]
        revision_result["ok"] = bool(
            revision_result["ok"]
            and observed_locks == {expected_lock}
            and not artifact_result["revision_rows_missing_lock"]
        )

    integrity = (
        all(result["ok"] for result in ledger_results)
        and artifact_result["ok"]
        and not missing_artifacts
        and all(result["passed"] for result in formal)
        and all(result["passed"] for result in dataset)
        and (source_result is None or source_result["ok"])
        and (asset_result is None or asset_result["ok"])
        and (revision_result is None or revision_result["ok"])
    )
    claim_evidence = _required_claim_evidence(
        ledger_results=ledger_results,
        artifact_result=artifact_result,
        formal_reports=formal,
        dataset_reports=dataset,
        source_result=source_result,
        asset_result=asset_result,
        revision_result=revision_result,
        aggregate_result=aggregate_result,
    )
    result = {
        "schema": "wqcodiff_workflow_audit_v1",
        "protocol_name": protocol.name,
        "protocol_sha256": protocol.sha256,
        "integrity_passed": integrity,
        "main_claim_eligible": bool(
            integrity
            and all(claim_evidence.values())
            and aggregate_result is not None
            and aggregate_result["oral_eligible"]
        ),
        "required_claim_evidence": claim_evidence,
        "ledgers": ledger_results,
        "artifacts": artifact_result,
        "ledger_artifact_hashes_missing": missing_artifacts,
        "formal_reports": formal,
        "dataset_reports": dataset,
        "source_manifest": source_result,
        "asset_lock": asset_result,
        "revision_threshold_lock": revision_result,
        "final_aggregate": aggregate_result,
    }
    write_json_exclusive(config.output, result)
    return result
