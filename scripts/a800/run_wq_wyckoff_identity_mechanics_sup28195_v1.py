#!/usr/bin/env python3
"""Development-only permutation-safe WTB mechanics regression.

This execution reuses the already exposed job28195 source panel only to verify
the source-identity repair and R/U/T mechanics.  It never evaluates CHGNet or
S.U.N., generates a new WQ proposal, trains, retries, reranks, or promotes a
scientific result.

The legacy arm engine remains byte-pinned.  This wrapper supplies:

* a new execution identity and new arm attempt IDs;
* a 256-row source round-trip audit using composition multisets;
* the first 32 frozen source/noise cells as a development mechanics panel;
* canonical atom ordering for every parent-model input;
* a non-blocking diagnostic for the legacy ordered-atom mismatch.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crystal_dlm.wqcodiff.charts import PyXtalChartCatalog  # noqa: E402
from crystal_dlm.wqcodiff.contracts import SeedDeriver, write_json_exclusive  # noqa: E402
from crystal_dlm.wqcodiff.crysllmgen.gate import sha256_file  # noqa: E402
from crystal_dlm.wqcodiff.crysllmgen.wtb_confirmatory import (  # noqa: E402
    IDENTITY as JOB28195_IDENTITY,
    PROTOCOL_NAME,
    SAMPLING_SEED,
    SOURCE_METHOD as JOB28195_SOURCE_METHOD,
    TRAINING_SEED,
    build_confirmatory_cells,
)
from crystal_dlm.wqcodiff.crysllmgen.wtb_identity_v2 import (  # noqa: E402
    audit_legacy_source_row,
    composition_multiset_signature,
)
from crystal_dlm.wqcodiff.runtime import expand_state  # noqa: E402
from scripts.a800 import (  # noqa: E402
    run_wq_wyckoff_chart_retraction_arms256_v1 as legacy,
)


IDENTITY = "wq_wyckoff_identity_mechanics_sup28195_v1"
CONTRACT_SCHEMA = "wq_wyckoff_identity_mechanics_contract_v1"
TERMINAL_SCHEMA = "wq_wyckoff_identity_mechanics_terminal_v1"
SOURCE_IDENTITY_SCHEMA = "wq_wyckoff_source_identity_audit_v2"
ATTEMPTS = 32
SOURCE_ATTEMPTS = 256
START_ORDINAL = 512
END_ORDINAL_INCLUSIVE = START_ORDINAL + ATTEMPTS - 1
ARM_METHODS = {
    "R": "WTB-IDV2-R-RAW-WQ-CANONICAL",
    "U": "WTB-IDV2-U-PARENT-SCHEDULE32-CANONICAL",
    "T": "WTB-IDV2-T-EVERY-STEP-GLOBAL-CHART-RETRACTION-CANONICAL",
}
ARM_EXPERIMENT_ID = f"{IDENTITY}-arms"
STAGE = "permutation_safe_rut32_mechanics"

_ACTIVE_CONTRACT: dict[str, Any] | None = None
_SOURCE_AUDIT_ROWS: list[dict[str, Any]] = []
_SOURCE_AUDIT_SUMMARY: dict[str, Any] = {}


def _require_sha256(value: str, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be one lowercase SHA256")
    return text


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    global _ACTIVE_CONTRACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != CONTRACT_SCHEMA
        or payload.get("identity") != IDENTITY
        or payload.get("status") != "authorized_development_execution"
    ):
        raise ValueError("unexpected permutation-safe mechanics contract")
    lineage = payload["lineage"]
    if (
        int(lineage["supersedes_failed_job_id"]) != 28195
        or lineage["reuse_purpose"] != "mechanics_regression_only"
        or lineage["confirmatory_evidence"] is not False
        or lineage["job28195_reinterpreted"] is not False
    ):
        raise ValueError("job28195 lineage/evidence boundary changed")
    identity = payload["identity_contract"]
    if (
        identity["composition_identity"] != "element_count_multiset"
        or identity["topology_identity"] != "exact_species_wyckoff_topology_hash"
        or identity["legacy_ordered_atom_identity_role"] != "diagnostic_only"
        or identity["legacy_ordered_atom_mismatch_blocking"] is not False
        or identity["parent_input_order"] != "canonical_orbit_expansion"
    ):
        raise ValueError("permutation-safe identity contract changed")
    matrix = payload["matrix"]
    if (
        int(matrix["source_identity_audit_attempts"]) != SOURCE_ATTEMPTS
        or int(matrix["mechanics_attempts_per_arm"]) != ATTEMPTS
        or matrix["arms"] != ARM_METHODS
        or int(matrix["start_ordinal"]) != START_ORDINAL
        or int(matrix["end_ordinal_inclusive"]) != END_ORDINAL_INCLUSIVE
        or int(matrix["reverse_steps"]) != legacy.REVERSE_STEPS
        or int(matrix["decoder_calls_per_diffusion_arm"])
        != legacy.DECODER_CALLS_PER_DIFFUSION_ARM
    ):
        raise ValueError("permutation-safe mechanics matrix changed")
    resources = payload["resources"]
    if (
        int(resources["a800"]) != 1
        or int(resources["cpus"]) != 8
        or int(resources["cpus"]) > 8 * int(resources["a800"])
    ):
        raise ValueError("resource contract exceeds 8 CPU per A800")
    if not all(bool(value) for value in payload["forbidden_actions"].values()):
        raise ValueError("a forbidden mechanics action was enabled")
    _ACTIVE_CONTRACT = payload
    return payload, sha256_file(path)


def _development_cells() -> tuple[Any, ...]:
    source_cells = build_confirmatory_cells()[:ATTEMPTS]
    arm_deriver = SeedDeriver(PROTOCOL_NAME, ARM_EXPERIMENT_ID)
    return tuple(
        dataclasses.replace(
            cell,
            arm_attempt_ids={
                arm: arm_deriver.attempt_id(
                    training_seed=TRAINING_SEED,
                    sampling_seed=SAMPLING_SEED,
                    ordinal=cell.ordinal,
                    method=method,
                )
                for arm, method in ARM_METHODS.items()
            },
        )
        for cell in source_cells
    )


def _load_source_evidence(
    *,
    source_jsonl: Path,
    source_report: Path,
    contract_sha256: str,
    execution_patch_sha256: str,
) -> list[dict[str, Any]]:
    del contract_sha256, execution_patch_sha256
    global _SOURCE_AUDIT_ROWS, _SOURCE_AUDIT_SUMMARY
    if _ACTIVE_CONTRACT is None:
        raise RuntimeError("contract must load before source evidence")
    frozen = _ACTIVE_CONTRACT["frozen_job28195_sources"]
    if (
        sha256_file(source_jsonl) != frozen["source_attempts_sha256"]
        or sha256_file(source_report) != frozen["source_report_sha256"]
    ):
        raise ValueError("job28195 frozen source bytes changed")
    report = json.loads(source_report.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "wq_wyckoff_chart_retraction_source_report_v1"
        or report.get("ok") is not True
        or report.get("acceptance") != "PASS"
        or int(report.get("submitted", -1)) != SOURCE_ATTEMPTS
        or int(report.get("terminal", -1)) != SOURCE_ATTEMPTS
        or report.get("contract_sha256")
        != frozen["job28195_scientific_contract_sha256"]
        or report.get("execution_patch_sha256")
        != frozen["job28195_execution_patch_sha256"]
        or report.get("retry_or_replacement_used") is not False
    ):
        raise ValueError("job28195 source report is not the frozen PASS")
    rows = [
        json.loads(line)
        for line in source_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cells = build_confirmatory_cells()
    if len(rows) != SOURCE_ATTEMPTS:
        raise ValueError("job28195 source denominator changed")
    catalog = PyXtalChartCatalog()
    normalized: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row, cell in zip(rows, cells, strict=True):
        if (
            row.get("schema")
            != "wq_wyckoff_chart_retraction_source_attempt_v1"
            or row.get("identity") != JOB28195_IDENTITY
            or row.get("method") != JOB28195_SOURCE_METHOD
            or row.get("attempt_id") != cell.source_attempt_id
            or row.get("pair_id") != cell.pair_id
            or int(row.get("ordinal", -1)) != cell.ordinal
            or int(row.get("proposal_seed", -1)) != cell.proposal_seed
            or row.get("arm_attempt_ids") != dict(cell.arm_attempt_ids)
            or row.get("contract_sha256")
            != frozen["job28195_scientific_contract_sha256"]
            or row.get("execution_patch_sha256")
            != frozen["job28195_execution_patch_sha256"]
            or row.get("retry_or_replacement_used") is not False
            or row.get("best_of_or_rerank_used") is not False
            or row.get("status") != "succeeded"
        ):
            raise ValueError(f"job28195 source row changed at ordinal {cell.ordinal}")
        try:
            payload = row.get("proposal_state")
            if not isinstance(payload, Mapping):
                raise ValueError("successful source has no proposal state")
            state = legacy.StratifiedState.from_dict(dict(payload))
            canonical_payload = state.to_dict(canonical_storage=True)
            canonical_state = legacy.StratifiedState.from_dict(canonical_payload)
            expanded = expand_state(
                canonical_state,
                catalog,
                redetect_space_group=False,
            )
            canonical_state, canonical_payload, audit = audit_legacy_source_row(
                row,
                canonical_atomic_numbers=expanded.atomic_numbers,
            )
            audit_row = {
                "ordinal": cell.ordinal,
                "source_attempt_id": cell.source_attempt_id,
                "pair_id": cell.pair_id,
                **audit.to_dict(),
            }
            audit_rows.append(audit_row)
            if cell.ordinal <= END_ORDINAL_INCLUSIVE:
                updated = dict(row)
                updated.update(
                    {
                        "proposal_state": canonical_payload,
                        "proposal_topology_hash": audit.canonical_topology_hash,
                        "composition_signature": (
                            audit.canonical_composition_multiset_signature
                        ),
                        "source_signature": audit.source_signature_v2,
                        "legacy_order_identity": audit.legacy_order_identity,
                        "legacy_order_mismatch_is_blocking": False,
                        "identity_contract": (
                            "permutation_safe_composition_multiset_v2"
                        ),
                    }
                )
                normalized.append(updated)
        except Exception as exc:
            failures.append(
                {
                    "ordinal": cell.ordinal,
                    "source_attempt_id": cell.source_attempt_id,
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
    _SOURCE_AUDIT_ROWS = audit_rows + failures
    _SOURCE_AUDIT_SUMMARY = {
        "schema": SOURCE_IDENTITY_SCHEMA,
        "identity": IDENTITY,
        "registered_sources": SOURCE_ATTEMPTS,
        "permutation_safe_identity_passed": len(audit_rows),
        "permutation_safe_identity_failed": len(failures),
        "legacy_order_identity_matched": sum(
            row.get("legacy_order_identity") is True for row in audit_rows
        ),
        "legacy_order_identity_mismatched": sum(
            row.get("legacy_order_identity") is False for row in audit_rows
        ),
        "legacy_order_mismatch_is_blocking": False,
        "composition_identity": "element_count_multiset",
        "topology_identity": "exact_species_wyckoff_topology_hash",
        "parent_input_order": "canonical_orbit_expansion",
        "source_attempts_sha256": sha256_file(source_jsonl),
        "source_report_sha256": sha256_file(source_report),
        "ok": len(audit_rows) == SOURCE_ATTEMPTS and not failures,
        "acceptance": (
            "PASS"
            if len(audit_rows) == SOURCE_ATTEMPTS and not failures
            else "FAIL"
        ),
    }
    if not _SOURCE_AUDIT_SUMMARY["ok"]:
        raise ValueError("permutation-safe full source identity audit failed")
    if len(normalized) != ATTEMPTS:
        raise ValueError("development mechanics subset changed")
    return normalized


def _validate_source_success(
    row: Mapping[str, Any],
    *,
    catalog: PyXtalChartCatalog,
) -> tuple[Any, Any]:
    payload = row.get("proposal_state")
    if not isinstance(payload, Mapping):
        raise ValueError("successful source has no proposal state")
    state = legacy.StratifiedState.from_dict(dict(payload))
    expanded = expand_state(
        state,
        catalog,
        redetect_space_group=False,
    )
    _, _, audit = audit_legacy_source_row(
        row,
        canonical_atomic_numbers=expanded.atomic_numbers,
    )
    if (
        audit.canonical_composition_multiset_signature
        != row.get("composition_signature")
        or audit.source_signature_v2 != row.get("source_signature")
    ):
        raise ValueError("normalized permutation-safe source identity changed")
    return state, expanded


def _terminal_gate(output: Path, arms_report: Mapping[str, Any]) -> dict[str, Any]:
    mechanics = [
        json.loads(line)
        for line in (output / "arm_mechanics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    generations = {
        arm: [
            json.loads(line)
            for line in (output / f"{arm.lower()}_generation.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        for arm in ARM_METHODS
    }
    t_mechanics = [row for row in mechanics if row.get("arm") == "T"]
    checks = {
        "source_permutation_safe_identity_256_of_256": (
            _SOURCE_AUDIT_SUMMARY.get("ok") is True
            and _SOURCE_AUDIT_SUMMARY.get("permutation_safe_identity_passed")
            == SOURCE_ATTEMPTS
        ),
        "legacy_order_mismatch_nonblocking": (
            _SOURCE_AUDIT_SUMMARY.get("legacy_order_mismatch_is_blocking")
            is False
        ),
        "R_generation_32_of_32": all(
            row.get("status") == "succeeded" for row in generations["R"]
        ),
        "U_generation_32_of_32": all(
            row.get("status") == "succeeded" for row in generations["U"]
        ),
        "T_generation_32_of_32": all(
            row.get("status") == "succeeded" for row in generations["T"]
        ),
        "all_terminal_composition_multisets_exact": all(
            row.get("status") == "succeeded"
            and row.get("observed_composition_signature")
            == row.get("composition_signature")
            for rows in generations.values()
            for row in rows
        ),
        "T_exact_topology_32_of_32": all(
            row.get("status") == "succeeded"
            and row.get("details", {}).get("topology_hash_unchanged") is True
            for row in t_mechanics
        ),
        "T_global_chart_retraction_32_of_32": all(
            row.get("status") == "succeeded"
            and int(row.get("projection_calls", -1))
            == legacy.DECODER_CALLS_PER_DIFFUSION_ARM
            and row.get("details", {})
            .get("mechanics", {})
            .get("lattice_projection_methods")
            == [legacy.PROJECTION_METHOD]
            and row.get("details", {})
            .get("mechanics", {})
            .get("all_chart_retraction_audit_values_finite")
            is True
            for row in t_mechanics
        ),
        "no_retry_or_replacement": all(
            row.get("retry_or_replacement_used") is False
            for rows in generations.values()
            for row in rows
        ),
    }
    ok = (
        arms_report.get("ok") is True
        and len(mechanics) == ATTEMPTS * len(ARM_METHODS)
        and all(len(rows) == ATTEMPTS for rows in generations.values())
        and all(checks.values())
    )
    return {
        "schema": TERMINAL_SCHEMA,
        "identity": IDENTITY,
        "ok": ok,
        "acceptance": "PASS" if ok else "FAIL",
        "development_panel_reused": True,
        "confirmatory_evidence": False,
        "supersedes_failed_job_id": 28195,
        "job28195_reinterpreted": False,
        "source_identity_summary": _SOURCE_AUDIT_SUMMARY,
        "mechanics_gate_checks": checks,
        "arm_terminal_counts": arms_report.get("terminal_counts"),
        "legacy_ordered_atom_identity_role": "diagnostic_only",
        "training_performed": False,
        "new_generation_performed": False,
        "mlip_used": False,
        "sun_evaluated": False,
        "external_api_used": False,
        "retry_or_replacement_used": False,
        "automatic_confirmatory_authorized": False,
    }


def _configure_legacy_engine() -> None:
    legacy.IDENTITY = IDENTITY
    legacy.ATTEMPTS = ATTEMPTS
    legacy.START_ORDINAL = START_ORDINAL
    legacy.END_ORDINAL_INCLUSIVE = END_ORDINAL_INCLUSIVE
    legacy.ARM_METHODS = ARM_METHODS
    legacy.ARM_EXPERIMENT_ID = ARM_EXPERIMENT_ID
    legacy.STAGE = STAGE
    legacy.MECHANICS_SCHEMA = "wq_wyckoff_identity_mechanics_row_v1"
    legacy.REPORT_SCHEMA = "wq_wyckoff_identity_mechanics_arms_report_v1"
    legacy.build_confirmatory_cells = _development_cells
    legacy._load_contract = _load_contract
    legacy._load_source_evidence = _load_source_evidence
    legacy._validate_source_success = _validate_source_success
    legacy.composition_signature = composition_multiset_signature


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-patch-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for name in (
        "contract",
        "source_jsonl",
        "source_report",
        "checkpoint",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    _require_sha256(args.execution_patch_sha256, name="execution patch")
    _configure_legacy_engine()
    try:
        arms_report = legacy._run(args, args.output_dir)
        write_json_exclusive(
            args.output_dir / "source_identity_audit.json",
            {
                **_SOURCE_AUDIT_SUMMARY,
                "rows": _SOURCE_AUDIT_ROWS,
            },
        )
        terminal = _terminal_gate(args.output_dir, arms_report)
        write_json_exclusive(args.output_dir / "terminal_report.json", terminal)
        print(json.dumps(terminal, sort_keys=True, allow_nan=False))
        if not terminal["ok"]:
            raise SystemExit(2)
    except Exception:
        raise


if __name__ == "__main__":
    main()
