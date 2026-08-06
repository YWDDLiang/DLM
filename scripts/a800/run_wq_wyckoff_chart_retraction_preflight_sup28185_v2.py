#!/usr/bin/env python3
"""Run the development-only chart-retraction mechanics regression.

This is a new method identity after the immutable scientific failure of
job28185.  It deliberately reuses the same 8 x 4 F/T panel only to verify the
mechanical regression; because that panel exposed the defect, no output from
this runner is confirmatory evidence.

The execution engine and immutable-U validation remain byte-pinned to the v1
runner.  This wrapper replaces only the lattice-audit extraction and gates,
then invokes that engine exactly once.  It does not submit jobs, generate
proposals, train, retry, replace, query an API, or call an MLIP.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crystal_dlm.wqcodiff.contracts import write_json_exclusive  # noqa: E402
from crystal_dlm.wqcodiff.crysllmgen.gate import sha256_file  # noqa: E402
from scripts.a800 import (  # noqa: E402
    run_wq_wyckoff_tangent_bridge_preflight_v1 as legacy,
)


CONTRACT_SCHEMA = "wq_wyckoff_chart_retraction_preflight_contract_v2"
TERMINAL_SCHEMA = "wq_wyckoff_chart_retraction_preflight_terminal_v2"
CELL_SCHEMA = "wq_wyckoff_chart_retraction_preflight_cell_v2"
REFERENCE_SCHEMA = "wq_wyckoff_chart_retraction_u_reference_v2"
IDENTITY = "wq_wyckoff_chart_retraction_preflight_sup28185_v2"
PROJECTION_METHOD = "global_chart_retraction_v1"

_LEGACY_MECHANICS_FROM_STEP = legacy._mechanics_from_step
_LEGACY_RUN_TANGENT_TRAJECTORY = legacy._run_tangent_trajectory
_LEGACY_GATE_ROWS = legacy._gate_rows
_LEGACY_APPEND_JSONL = legacy._append_jsonl
_LEGACY_WRITE_JSON_EXCLUSIVE = legacy.write_json_exclusive


def _load_contract_v2(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unexpected chart-retraction contract schema")
    if payload.get("identity") != IDENTITY:
        raise ValueError("unexpected chart-retraction execution identity")
    if payload.get("status") != "local_built_remote_execution_not_authorized":
        raise ValueError("chart-retraction contract is not local-only")
    if int(payload.get("supersedes_job_id", -1)) != 28185:
        raise ValueError("chart-retraction contract must supersede job28185")
    evidence = payload["evidence_classification"]
    if (
        not bool(evidence["development_panel_reused"])
        or bool(evidence["confirmatory_evidence"])
        or evidence["reuse_purpose"] != "mechanics_regression_only"
    ):
        raise ValueError("job28185-informed panel is not development-only")
    matrix = payload["matrix"]
    if (
        matrix["reference_u_cells"] != legacy.BRIDGE_CELL_COUNT
        or matrix["new_cells_per_arm"] != legacy.BRIDGE_CELL_COUNT
        or matrix["new_arms"] != ["F", "T"]
    ):
        raise ValueError("chart-retraction contract is not the frozen F/T32 matrix")
    resources = payload["future_resource_envelope_not_authorized"]
    if resources["cpus"] > 8 * resources["a800"]:
        raise ValueError("resource contract exceeds 8 CPU per A800")
    if not all(bool(value) for value in payload["forbidden_actions"].values()):
        raise ValueError("one forbidden chart-retraction action was enabled")
    return payload, sha256_file(path)


def _mechanics_from_step_v2(audit: Mapping[str, Any]) -> dict[str, Any]:
    mechanics = _LEGACY_MECHANICS_FROM_STEP(audit)
    lattice = audit["lattice_audit"]
    finite_values = (
        float(lattice["input_update_norm"]),
        float(lattice["chart_update_norm"]),
        float(lattice["tangent_update_norm"]),
        float(lattice["retracted_update_norm"]),
        float(lattice["normal_residual_norm"]),
        float(lattice["condition_number_after"]),
        float(audit["primitive_transform_consistency_max_abs_error"]),
        float(audit["primitive_lattice_consistency_max_abs_error"]),
        float(audit["primitive_lattice_consistency_relative_error"]),
        float(audit["primitive_lattice_scale"]),
    )
    mechanics.update(
        {
            "lattice_projection_methods": [
                str(lattice["projection_method"])
            ],
            "maximum_lattice_input_update_norm": finite_values[0],
            "maximum_lattice_chart_update_norm": finite_values[1],
            "maximum_lattice_linearized_update_norm": finite_values[2],
            "maximum_lattice_retracted_update_norm": finite_values[3],
            "maximum_lattice_normal_residual_norm": finite_values[4],
            "maximum_lattice_condition_number_after": finite_values[5],
            "maximum_primitive_transform_consistency_max_abs_error": (
                finite_values[6]
            ),
            "maximum_primitive_lattice_consistency_max_abs_error": (
                finite_values[7]
            ),
            "maximum_primitive_lattice_consistency_relative_error": (
                finite_values[8]
            ),
            "maximum_primitive_lattice_scale": finite_values[9],
            "all_chart_retraction_audit_values_finite": all(
                math.isfinite(value) for value in finite_values
            ),
        }
    )
    return mechanics


def _run_tangent_trajectory_v2(*args: Any, **kwargs: Any):
    final_structure, final_wq_state, details = (
        _LEGACY_RUN_TANGENT_TRAJECTORY(*args, **kwargs)
    )
    records = list(details["trajectory"]["projection_records"])
    step_mechanics = [
        _mechanics_from_step_v2(record["audit"]) for record in records
    ]
    mechanics = dict(details["mechanics"])
    mechanics.update(
        {
            "lattice_projection_methods": sorted(
                {
                    method
                    for value in step_mechanics
                    for method in value["lattice_projection_methods"]
                }
            ),
            "maximum_lattice_input_update_norm": max(
                value["maximum_lattice_input_update_norm"]
                for value in step_mechanics
            ),
            "maximum_lattice_chart_update_norm": max(
                value["maximum_lattice_chart_update_norm"]
                for value in step_mechanics
            ),
            "maximum_lattice_linearized_update_norm": max(
                value["maximum_lattice_linearized_update_norm"]
                for value in step_mechanics
            ),
            "maximum_lattice_retracted_update_norm": max(
                value["maximum_lattice_retracted_update_norm"]
                for value in step_mechanics
            ),
            "maximum_lattice_normal_residual_norm": max(
                value["maximum_lattice_normal_residual_norm"]
                for value in step_mechanics
            ),
            "maximum_lattice_condition_number_after": max(
                value["maximum_lattice_condition_number_after"]
                for value in step_mechanics
            ),
            "maximum_primitive_transform_consistency_max_abs_error": max(
                value[
                    "maximum_primitive_transform_consistency_max_abs_error"
                ]
                for value in step_mechanics
            ),
            "maximum_primitive_lattice_consistency_max_abs_error": max(
                value[
                    "maximum_primitive_lattice_consistency_max_abs_error"
                ]
                for value in step_mechanics
            ),
            "maximum_primitive_lattice_consistency_relative_error": max(
                value[
                    "maximum_primitive_lattice_consistency_relative_error"
                ]
                for value in step_mechanics
            ),
            "maximum_primitive_lattice_scale": max(
                value["maximum_primitive_lattice_scale"]
                for value in step_mechanics
            ),
            "all_chart_retraction_audit_values_finite": all(
                value["all_chart_retraction_audit_values_finite"]
                for value in step_mechanics
            ),
        }
    )
    details = dict(details)
    details["mechanics"] = mechanics
    return final_structure, final_wq_state, details


def _gate_rows_v2(
    rows: list[dict[str, Any]],
    *,
    contract: Mapping[str, Any],
    schedule_ok: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summaries, checks = _LEGACY_GATE_ROWS(
        rows,
        contract=contract,
        schedule_ok=schedule_ok,
    )
    gates = contract["mechanics_gates"]
    for arm in ("F", "T"):
        succeeded = [
            row
            for row in rows
            if row["arm"] == arm and row["status"] == "succeeded"
        ]
        mechanics = [row["mechanics"] for row in succeeded]
        summary = summaries[arm]
        summary.update(
            {
                "lattice_projection_methods": sorted(
                    {
                        method
                        for value in mechanics
                        for method in value["lattice_projection_methods"]
                    }
                ),
                "maximum_lattice_input_update_norm": max(
                    (
                        float(value["maximum_lattice_input_update_norm"])
                        for value in mechanics
                    ),
                    default=math.inf,
                ),
                "maximum_lattice_chart_update_norm": max(
                    (
                        float(value["maximum_lattice_chart_update_norm"])
                        for value in mechanics
                    ),
                    default=math.inf,
                ),
                "maximum_lattice_retracted_update_norm": max(
                    (
                        float(value["maximum_lattice_retracted_update_norm"])
                        for value in mechanics
                    ),
                    default=math.inf,
                ),
                "maximum_lattice_condition_number_after": max(
                    (
                        float(
                            value[
                                "maximum_lattice_condition_number_after"
                            ]
                        )
                        for value in mechanics
                    ),
                    default=math.inf,
                ),
                "maximum_primitive_transform_consistency_max_abs_error": max(
                    (
                        float(
                            value[
                                "maximum_primitive_transform_consistency_max_abs_error"
                            ]
                        )
                        for value in mechanics
                    ),
                    default=math.inf,
                ),
                "maximum_primitive_lattice_consistency_max_abs_error": max(
                    (
                        float(
                            value[
                                "maximum_primitive_lattice_consistency_max_abs_error"
                            ]
                        )
                        for value in mechanics
                    ),
                    default=math.inf,
                ),
                "maximum_primitive_lattice_consistency_relative_error": max(
                    (
                        float(
                            value[
                                "maximum_primitive_lattice_consistency_relative_error"
                            ]
                        )
                        for value in mechanics
                    ),
                    default=math.inf,
                ),
                "maximum_primitive_lattice_scale": max(
                    (
                        float(value["maximum_primitive_lattice_scale"])
                        for value in mechanics
                    ),
                    default=math.inf,
                ),
                "all_chart_retraction_audit_values_finite": bool(mechanics)
                and all(
                    bool(value["all_chart_retraction_audit_values_finite"])
                    for value in mechanics
                ),
            }
        )
        checks[f"{arm}_global_chart_retraction"] = (
            summary["lattice_projection_methods"]
            == [gates["required_lattice_projection_method"]]
        )
        checks[f"{arm}_primitive_transform"] = (
            summary[
                "maximum_primitive_transform_consistency_max_abs_error"
            ]
            <= gates["primitive_transform_consistency_max_abs_error"]
        )
        checks[f"{arm}_primitive_lattice_consistency"] = (
            summary[
                "maximum_primitive_lattice_consistency_max_abs_error"
            ]
            <= gates["primitive_lattice_consistency_max_abs_error"]
            and summary[
                "maximum_primitive_lattice_consistency_relative_error"
            ]
            <= gates["primitive_lattice_consistency_relative_error"]
        )
        checks[f"{arm}_lattice_scale_safety"] = (
            summary["maximum_primitive_lattice_scale"]
            <= gates["primitive_lattice_max_abs_entry"]
        )
        checks[f"{arm}_lattice_condition_safety"] = (
            summary["maximum_lattice_condition_number_after"]
            <= gates["lattice_condition_number_after_max"]
        )
        checks[f"{arm}_chart_retraction_finite"] = summary[
            "all_chart_retraction_audit_values_finite"
        ]
    return summaries, checks


def _append_jsonl_v2(handle: Any, row: Mapping[str, Any]) -> None:
    payload = dict(row)
    if payload.get("schema") == "wq_wyckoff_tangent_preflight_cell_v1":
        payload["schema"] = CELL_SCHEMA
    _LEGACY_APPEND_JSONL(handle, payload)


def _write_json_exclusive_v2(path: Path, payload: Mapping[str, Any]) -> None:
    converted = dict(payload)
    if path.name == "u_reference_manifest.json":
        converted["schema"] = REFERENCE_SCHEMA
        converted["development_panel_reused"] = True
        converted["confirmatory_evidence"] = False
    elif path.name == "terminal_report.json":
        converted["schema"] = TERMINAL_SCHEMA
        converted["development_panel_reused"] = True
        converted["confirmatory_evidence"] = False
        converted["supersedes_job_id"] = 28185
        converted["lattice_projection_method"] = PROJECTION_METHOD
    _LEGACY_WRITE_JSON_EXCLUSIVE(path, converted)


def _activate_v2_engine() -> None:
    legacy._load_contract = _load_contract_v2
    legacy._mechanics_from_step = _mechanics_from_step_v2
    legacy._run_tangent_trajectory = _run_tangent_trajectory_v2
    legacy._gate_rows = _gate_rows_v2
    legacy._append_jsonl = _append_jsonl_v2
    legacy.write_json_exclusive = _write_json_exclusive_v2


def _verify_implementation(contract: Mapping[str, Any]) -> None:
    implementation = contract["implementation"]
    checks = (
        ("v2_runner_source", Path(__file__).resolve()),
        (
            "legacy_execution_engine_source",
            ROOT / implementation["legacy_execution_engine_source"],
        ),
        (
            "tangent_bridge_source",
            ROOT / implementation["tangent_bridge_source"],
        ),
        ("runtime_source", ROOT / implementation["runtime_source"]),
    )
    for field, path in checks:
        expected = implementation[f"{field}_sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"installed {field} SHA256 changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "configs/experiments/wyckoff_codiffusion"
        / "wq_wyckoff_chart_retraction_preflight_sup28185_v2.json",
    )
    parser.add_argument("--source-generation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--u-terminal-audit", type=Path, required=True)
    parser.add_argument("--u-selection-manifest", type=Path, required=True)
    parser.add_argument("--u-attempts", type=Path, required=True)
    parser.add_argument("--u-terminal-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-patch-sha256", required=True)
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cuda"),
    )
    args = parser.parse_args()
    for name in (
        "contract",
        "source_generation",
        "checkpoint",
        "u_terminal_audit",
        "u_selection_manifest",
        "u_attempts",
        "u_terminal_report",
    ):
        setattr(args, name, getattr(args, name).resolve())
    contract, _ = _load_contract_v2(args.contract)
    _verify_implementation(contract)
    _activate_v2_engine()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = legacy._run(args, output)
    except Exception as exc:
        terminal = output / "terminal_report.json"
        if not terminal.exists():
            write_json_exclusive(
                terminal,
                {
                    "schema": TERMINAL_SCHEMA,
                    "ok": False,
                    "acceptance": "FAIL",
                    "identity": IDENTITY,
                    "reason": f"{type(exc).__name__}:{exc}",
                    "traceback": traceback.format_exc(),
                    "development_panel_reused": True,
                    "confirmatory_evidence": False,
                    "training_performed": False,
                    "new_generation_performed": False,
                    "u_rerun": False,
                    "retry_or_replacement_used": False,
                    "mlip_used": False,
                    "external_api_used": False,
                },
            )
        raise
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
