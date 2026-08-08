"""Immutable cross-machine ledgers for local-only exact SMACT 4 audits.

The Evidence-First route deliberately keeps SMACT 4 off A800.  This module
contains only JSON/hash/identity checks, so the A800 SMACT 3.1 runtime can
verify and consume ledgers produced by the separately frozen local evaluator
without importing SMACT 4.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from crystal_dlm.h1_chemistry_first_sft import canonical_json_sha256
from crystal_dlm.h1_nocharge_ion_aux import canonicalize_ion_witness


WITNESS_LEDGER_SCHEMA = "h1_local_smact4_witness_ledger_v1"
WITNESS_MANIFEST_SCHEMA = "h1_local_smact4_witness_manifest_v1"
STAGE_AUDIT_MANIFEST_SCHEMA = "h1_local_smact4_stage_audit_manifest_v1"
EXPECTED_SMACT4_VERSION = "4.0.0"
EXPECTED_SMACT4_WHEEL_SHA256 = (
    "e3eb968da92d47a8ef9a4af42af5589a6de61cccbca9d329937e1f4e402f0551"
)
EXPECTED_SMACT4_CONTRACT_SHA256 = (
    "ad070f3ad8d025ec9d7918dc1aa84e3f8faaf4ee0a124fbe8602208d5609ca19"
)
LEGACY_PRIMARY_REASON = "charge_neutral_pauling_valid"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def frozen_source_row_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact source identity shared by local and A800 joins."""

    return {
        "split": str(row.get("split", "")),
        "row_idx": int(row.get("row_idx", -1)),
        "material_id": str(row.get("material_id", "")),
        "plan": row.get("plan"),
        "legacy": row.get("legacy"),
    }


def frozen_source_row_sha256(row: Mapping[str, Any]) -> str:
    return canonical_json_sha256(frozen_source_row_payload(row))


def _legacy_primary(row: Mapping[str, Any]) -> bool:
    legacy = row.get("legacy")
    return bool(
        isinstance(legacy, Mapping)
        and legacy.get("valid") is True
        and legacy.get("reason") == LEGACY_PRIMARY_REASON
    )


def _validate_smact4_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("smact_version") != EXPECTED_SMACT4_VERSION
        or contract.get("release_wheel_sha256") != EXPECTED_SMACT4_WHEEL_SHA256
        or contract.get("contract_sha256") != EXPECTED_SMACT4_CONTRACT_SHA256
    ):
        raise ValueError("exact local SMACT4 contract identity mismatch")


def build_witness_ledger_payload(
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    source_inventory_sha256: str,
    legacy_report: Mapping[str, Any],
    smact4_contract: Mapping[str, Any],
    witness_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a complete one-row-per-source-row local witness ledger."""

    _validate_smact4_contract(smact4_contract)
    if len(source_inventory_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in source_inventory_sha256
    ):
        raise ValueError("invalid frozen source inventory SHA")
    split_payloads: dict[str, Any] = {}
    for split in ("train", "val"):
        rows = list(source_rows[split])
        report = witness_reports[split]
        if report.get("official_witness_parity") is not True:
            raise ValueError(f"{split} exact-SMACT4 witness parity failed")
        ledger_rows: list[dict[str, Any]] = []
        stable_indices: list[int] = []
        for expected_idx, row in enumerate(rows):
            if int(row.get("row_idx", -1)) != expected_idx:
                raise ValueError(f"{split} source ordinals are not exact")
            primary = _legacy_primary(row)
            upgraded = row.get("smact4")
            if primary:
                if not isinstance(upgraded, Mapping):
                    raise ValueError(f"{split}:{expected_idx} lacks SMACT4 audit")
                if (
                    upgraded.get("valid") is True
                    and upgraded.get("stratum") == "uniform_primary"
                    and upgraded.get("witness")
                ):
                    stable_indices.append(expected_idx)
                audit: dict[str, Any] | None = dict(upgraded)
                local_status = "audited"
            else:
                if upgraded is not None:
                    raise ValueError(
                        f"{split}:{expected_idx} unexpectedly audited outside legacy-primary"
                    )
                audit = None
                local_status = "not_legacy_nonshortcut_primary"
            plan = row.get("plan")
            if not isinstance(plan, Mapping):
                raise ValueError(f"{split}:{expected_idx} lacks Plan payload")
            ledger_rows.append(
                {
                    "split": split,
                    "row_idx": expected_idx,
                    "material_id": str(row.get("material_id", "")),
                    "formula": str(plan.get("formula", "")),
                    "source_row_sha256": frozen_source_row_sha256(row),
                    "legacy_primary": primary,
                    "local_audit_status": local_status,
                    "smact4": audit,
                }
            )
        reported = [int(value) for value in report["stable_primary_indices"]]
        if stable_indices != reported:
            raise ValueError(f"{split} stable-primary index identity mismatch")
        split_report = (legacy_report.get("splits") or {}).get(split)
        if not isinstance(split_report, Mapping):
            raise ValueError(f"legacy snapshot omitted {split} report")
        split_payloads[split] = {
            "row_count": len(rows),
            "snapshot_jsonl_sha256": split_report.get("snapshot_jsonl_sha256"),
            "source_csv_sha256": split_report.get("source_csv_sha256"),
            "stable_primary_indices": stable_indices,
            "stable_primary_count": len(stable_indices),
            "witness_report": dict(report),
            "rows": ledger_rows,
        }
    payload = {
        "schema": WITNESS_LEDGER_SCHEMA,
        "status": "pass",
        "execution_location": "local_windows_only",
        "a800_smact4_execution": False,
        "source_inventory_sha256": source_inventory_sha256,
        "legacy_snapshot_contract_sha256": legacy_report.get("contract_sha256"),
        "smact4_contract": dict(smact4_contract),
        "splits": split_payloads,
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    return payload


def write_witness_bundle(output_dir: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    ledger_path = output / "witness_ledger.json"
    ledger_path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": WITNESS_MANIFEST_SCHEMA,
        "status": "pass",
        "execution_location": "local_windows_only",
        "a800_smact4_execution": False,
        "source_inventory_sha256": payload.get("source_inventory_sha256"),
        "witness_ledger": ledger_path.name,
        "witness_ledger_sha256": sha256_file(ledger_path),
        "witness_ledger_bytes": ledger_path.stat().st_size,
        "legacy_snapshot_contract_sha256": payload.get(
            "legacy_snapshot_contract_sha256"
        ),
        "smact4_contract_sha256": (
            payload.get("smact4_contract") or {}
        ).get("contract_sha256"),
        "split_counts": {
            split: int(value["row_count"])
            for split, value in (payload.get("splits") or {}).items()
        },
    }
    manifest_path = output / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_sha = sha256_file(manifest_path)
    success = {
        "schema": WITNESS_MANIFEST_SCHEMA,
        "complete": True,
        "manifest_sha256": manifest_sha,
        "witness_ledger_sha256": manifest["witness_ledger_sha256"],
    }
    (output / "_SUCCESS").write_text(
        json.dumps(success, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**manifest, "manifest_sha256": manifest_sha}


def load_and_attach_witness_bundle(
    bundle_dir: Path,
    source_rows: MutableMapping[str, list[dict[str, Any]]],
    *,
    legacy_report: Mapping[str, Any],
    expected_source_inventory_sha256: str,
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Verify a local ledger and attach only exact matching SMACT4 witnesses."""

    root = bundle_dir.resolve()
    manifest_path = root / "MANIFEST.json"
    success_path = root / "_SUCCESS"
    manifest = read_object(manifest_path)
    success = read_object(success_path)
    manifest_sha = sha256_file(manifest_path)
    if expected_manifest_sha256 and manifest_sha != expected_manifest_sha256:
        raise ValueError("local witness manifest SHA mismatch")
    if manifest.get("witness_ledger") != "witness_ledger.json":
        raise ValueError("local witness ledger filename is not frozen")
    ledger_path = root / "witness_ledger.json"
    observed_files = {path.name for path in root.iterdir() if path.is_file()}
    expected_files = {"MANIFEST.json", "_SUCCESS", "witness_ledger.json"}
    if (
        manifest.get("schema") != WITNESS_MANIFEST_SCHEMA
        or manifest.get("status") != "pass"
        or manifest.get("execution_location") != "local_windows_only"
        or manifest.get("a800_smact4_execution") is not False
        or manifest.get("source_inventory_sha256")
        != expected_source_inventory_sha256
        or success.get("schema") != WITNESS_MANIFEST_SCHEMA
        or success.get("complete") is not True
        or success.get("manifest_sha256") != manifest_sha
        or success.get("witness_ledger_sha256")
        != manifest.get("witness_ledger_sha256")
        or manifest.get("legacy_snapshot_contract_sha256")
        != legacy_report.get("contract_sha256")
        or manifest.get("smact4_contract_sha256")
        != EXPECTED_SMACT4_CONTRACT_SHA256
        or manifest.get("split_counts")
        != {split: len(source_rows[split]) for split in ("train", "val")}
        or not ledger_path.is_file()
        or sha256_file(ledger_path) != manifest.get("witness_ledger_sha256")
        or ledger_path.stat().st_size != int(manifest.get("witness_ledger_bytes", -1))
        or observed_files != expected_files
    ):
        raise ValueError("local witness bundle identity mismatch")
    ledger = read_object(ledger_path)
    payload_for_sha = dict(ledger)
    recorded_payload_sha = payload_for_sha.pop("payload_sha256", None)
    if (
        ledger.get("schema") != WITNESS_LEDGER_SCHEMA
        or ledger.get("status") != "pass"
        or ledger.get("execution_location") != "local_windows_only"
        or ledger.get("a800_smact4_execution") is not False
        or ledger.get("source_inventory_sha256")
        != expected_source_inventory_sha256
        or recorded_payload_sha != canonical_json_sha256(payload_for_sha)
        or ledger.get("legacy_snapshot_contract_sha256")
        != legacy_report.get("contract_sha256")
    ):
        raise ValueError("local witness ledger contract mismatch")
    contract = ledger.get("smact4_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("local witness ledger omitted SMACT4 contract")
    _validate_smact4_contract(contract)

    reports: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        rows = source_rows[split]
        split_ledger = (ledger.get("splits") or {}).get(split)
        split_report = (legacy_report.get("splits") or {}).get(split)
        if not isinstance(split_ledger, Mapping) or not isinstance(
            split_report, Mapping
        ):
            raise ValueError(f"local witness ledger omitted {split}")
        entries = split_ledger.get("rows")
        if (
            not isinstance(entries, list)
            or len(entries) != len(rows)
            or int(split_ledger.get("row_count", -1)) != len(rows)
            or split_ledger.get("snapshot_jsonl_sha256")
            != split_report.get("snapshot_jsonl_sha256")
            or split_ledger.get("source_csv_sha256")
            != split_report.get("source_csv_sha256")
        ):
            raise ValueError(f"{split} witness-ledger parent identity mismatch")
        stable: list[int] = []
        strata: Counter[str] = Counter()
        for expected_idx, (row, entry) in enumerate(zip(rows, entries)):
            if not isinstance(entry, Mapping):
                raise ValueError(f"{split}:{expected_idx} ledger row is not an object")
            plan = row.get("plan")
            if not isinstance(plan, Mapping):
                raise ValueError(f"{split}:{expected_idx} source row lacks Plan")
            if (
                entry.get("split") != split
                or int(entry.get("row_idx", -1)) != expected_idx
                or str(entry.get("material_id", ""))
                != str(row.get("material_id", ""))
                or str(entry.get("formula", "")) != str(plan.get("formula", ""))
                or entry.get("source_row_sha256") != frozen_source_row_sha256(row)
                or bool(entry.get("legacy_primary")) != _legacy_primary(row)
            ):
                raise ValueError(f"{split}:{expected_idx} local ledger join mismatch")
            upgraded = entry.get("smact4")
            if _legacy_primary(row):
                if entry.get("local_audit_status") != "audited" or not isinstance(
                    upgraded, Mapping
                ):
                    raise ValueError(f"{split}:{expected_idx} missing SMACT4 audit")
                if upgraded.get("official_witness_parity") is not True:
                    raise ValueError(f"{split}:{expected_idx} witness parity failed")
                row["smact4"] = dict(upgraded)
                strata[str(upgraded.get("stratum", "unknown"))] += 1
                if (
                    upgraded.get("valid") is True
                    and upgraded.get("stratum") == "uniform_primary"
                    and upgraded.get("witness")
                ):
                    witness = canonicalize_ion_witness(
                        [tuple(value) for value in upgraded["witness"]]
                    )
                    elements = list(plan.get("elements") or [])
                    counts = [int(value) for value in (plan.get("counts") or [])]
                    expected_counts = Counter(
                        symbol
                        for symbol, count in zip(elements, counts)
                        for _ in range(count)
                    )
                    if (
                        Counter(symbol for symbol, _oxidation in witness)
                        != expected_counts
                        or sum(int(oxidation) for _symbol, oxidation in witness) != 0
                    ):
                        raise ValueError(
                            f"{split}:{expected_idx} witness arithmetic mismatch"
                        )
                    stable.append(expected_idx)
            else:
                if (
                    entry.get("local_audit_status")
                    != "not_legacy_nonshortcut_primary"
                    or upgraded is not None
                ):
                    raise ValueError(
                        f"{split}:{expected_idx} nonprimary row has local audit payload"
                    )
        recorded_stable = [
            int(value) for value in split_ledger.get("stable_primary_indices", [])
        ]
        if stable != recorded_stable:
            raise ValueError(f"{split} stable-primary join mismatch")
        reports[split] = {
            "legacy_primary_count": sum(_legacy_primary(row) for row in rows),
            "stable_primary_indices": stable,
            "stable_primary_count": len(stable),
            "strata": dict(sorted(strata.items())),
            "official_witness_parity_failures": [],
            "official_witness_parity": True,
            "local_manifest_sha256": manifest_sha,
        }
    return {
        **manifest,
        "manifest_sha256": manifest_sha,
        "smact4_contract": dict(contract),
    }, reports


__all__ = [
    "EXPECTED_SMACT4_CONTRACT_SHA256",
    "EXPECTED_SMACT4_VERSION",
    "EXPECTED_SMACT4_WHEEL_SHA256",
    "STAGE_AUDIT_MANIFEST_SCHEMA",
    "WITNESS_LEDGER_SCHEMA",
    "WITNESS_MANIFEST_SCHEMA",
    "build_witness_ledger_payload",
    "frozen_source_row_payload",
    "frozen_source_row_sha256",
    "load_and_attach_witness_bundle",
    "read_object",
    "sha256_file",
    "write_witness_bundle",
]
