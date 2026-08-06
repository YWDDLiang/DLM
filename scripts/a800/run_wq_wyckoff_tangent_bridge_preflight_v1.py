#!/usr/bin/env python3
"""Run the MLIP-free F/T Wyckoff-tangent mechanics preflight.

Arm U is loaded only as immutable job28081 evidence.  Arm F projects each
stored U terminal structure once.  Arm T reuses the released CrysLLMGen
checkpoint and scalar schedules while restricting forward noise and every
reverse substep to the fixed Wyckoff quotient.  No proposal is generated and
no scientific outcome is used for selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.a800.run_wq_schedule_correct_bridge_parity_v1 import (  # noqa: E402
    _canonical_json,
    _load_source_rows,
    _module_state_sha256,
    _schedule_audit,
    _select_rows,
)
from crystal_dlm.wqcodiff.charts import PyXtalChartCatalog  # noqa: E402
from crystal_dlm.wqcodiff.contracts import write_json_exclusive  # noqa: E402
from crystal_dlm.wqcodiff.crysllmgen.atom_sampling import (  # noqa: E402
    _output_structure,
    expanded_state_to_parent_batch,
    load_registered_csp,
)
from crystal_dlm.wqcodiff.crysllmgen.bridge_parity import (  # noqa: E402
    BRIDGE_CELL_COUNT,
    build_bridge_cells,
    build_numpy_parent_schedules,
)
from crystal_dlm.wqcodiff.crysllmgen.gate import sha256_file  # noqa: E402
from crystal_dlm.wqcodiff.crysllmgen.tangent_bridge import (  # noqa: E402
    ParentLatticeFrame,
    WyckoffTangentProjector,
    forward_noise_manifold_numpy,
    run_parent_reverse_on_wyckoff_manifold,
)
from crystal_dlm.wqcodiff.runtime import expand_state  # noqa: E402
from crystal_dlm.wqcodiff.state import StratifiedState  # noqa: E402


def _require_sha256(value: str, *, name: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be one lowercase SHA256")
    return value


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "wq_wyckoff_tangent_preflight_contract_v1":
        raise ValueError("unexpected Wyckoff-tangent contract schema")
    if payload["status"] != "local_built_remote_execution_not_authorized":
        raise ValueError("Wyckoff-tangent contract status is not local-only")
    matrix = payload["matrix"]
    if (
        matrix["reference_u_cells"] != BRIDGE_CELL_COUNT
        or matrix["new_cells_per_arm"] != BRIDGE_CELL_COUNT
        or matrix["new_arms"] != ["F", "T"]
    ):
        raise ValueError("Wyckoff-tangent contract is not the frozen 32-cell matrix")
    resources = payload["future_resource_envelope_not_authorized"]
    if resources["cpus"] > 8 * resources["a800"]:
        raise ValueError("resource contract exceeds 8 CPU per A800")
    forbidden = payload["forbidden_actions"]
    if not all(bool(value) for value in forbidden.values()):
        raise ValueError("one forbidden Wyckoff-tangent action was enabled")
    return payload, sha256_file(path)


def _load_json_exact(path: Path, expected_sha256: str, *, name: str) -> dict[str, Any]:
    expected = _require_sha256(expected_sha256, name=f"{name} expected SHA256")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{name} SHA256 differs from the frozen contract")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return payload


def _load_jsonl_exact(
    path: Path,
    expected_sha256: str,
    *,
    name: str,
) -> list[dict[str, Any]]:
    expected = _require_sha256(expected_sha256, name=f"{name} expected SHA256")
    if sha256_file(path) != expected:
        raise ValueError(f"{name} SHA256 differs from the frozen contract")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except Exception as exc:
                raise ValueError(f"{name} row {line_number} is invalid") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{name} row {line_number} is not an object")
            rows.append(row)
    return rows


def _validate_u_reference(
    *,
    contract: Mapping[str, Any],
    terminal_audit_path: Path,
    selection_path: Path,
    attempts_path: Path,
    terminal_report_path: Path,
    selected_source_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    reference = contract["immutable_u_reference"]
    terminal_audit = _load_json_exact(
        terminal_audit_path,
        reference["terminal_audit_sha256"],
        name="U terminal audit",
    )
    selection = _load_json_exact(
        selection_path,
        reference["selection_manifest_sha256"],
        name="U selection manifest",
    )
    attempts = _load_jsonl_exact(
        attempts_path,
        reference["attempts_jsonl_sha256"],
        name="U attempts",
    )
    terminal_report = _load_json_exact(
        terminal_report_path,
        reference["terminal_report_sha256"],
        name="U terminal report",
    )
    if (
        terminal_audit.get("status") != "PASS"
        or terminal_audit.get("acceptance") != "PASS"
        or str(terminal_audit["submission"]["job_id"]) != "28081"
        or terminal_audit["submission"]["slurm"]["state"] != "COMPLETED"
        or terminal_audit["submission"]["slurm"]["exit_code"] != "0:0"
    ):
        raise ValueError("U terminal audit is not the immutable PASS job28081")
    terminal_hashes = terminal_audit["terminal_artifacts"]
    for key in (
        "selection_manifest_sha256",
        "attempts_jsonl_sha256",
        "terminal_report_sha256",
    ):
        if terminal_hashes[key] != reference[key]:
            raise ValueError(f"U terminal audit disagrees on {key}")
    if (
        not terminal_report.get("ok")
        or terminal_report.get("acceptance") != "PASS"
        or terminal_report.get("attempts_jsonl_sha256")
        != reference["attempts_jsonl_sha256"]
        or terminal_report.get("selection_manifest_sha256")
        != reference["selection_manifest_sha256"]
    ):
        raise ValueError("U terminal report is not the frozen passing report")
    expected_attempt_ids = [row["attempt_id"] for row in selected_source_rows]
    if selection.get("attempt_ids") != expected_attempt_ids:
        raise ValueError("U selection manifest no longer matches the source panel")

    cells = build_bridge_cells(base_seed=int(contract["matrix"]["base_noise_seed"]))
    expected_by_id = {cell.cell_id: cell for cell in cells}
    if len(attempts) != BRIDGE_CELL_COUNT:
        raise ValueError("U attempts do not contain exactly 32 rows")
    by_id: dict[str, dict[str, Any]] = {}
    for row in attempts:
        cell_id = str(row.get("cell_id", ""))
        if not cell_id or cell_id in by_id or cell_id not in expected_by_id:
            raise ValueError("U attempts contain an unknown or duplicate cell")
        cell = expected_by_id[cell_id]
        expected_source = selected_source_rows[cell.panel_index]["attempt_id"]
        if (
            row.get("status") != "succeeded"
            or row.get("source_attempt_id") != expected_source
            or int(row.get("timestep", -1)) != cell.timestep
            or bool(row.get("retry_or_replacement_used"))
            or bool(row.get("clean_condition_used_as_reverse_state"))
            or not isinstance(row.get("final_structure"), Mapping)
        ):
            raise ValueError(f"U cell {cell_id} violates its immutable contract")
        by_id[cell_id] = row
    if set(by_id) != set(expected_by_id):
        raise ValueError("U attempts are not the exact frozen cell set")
    return by_id, {
        "terminal_audit_sha256": reference["terminal_audit_sha256"],
        "selection_manifest_sha256": reference["selection_manifest_sha256"],
        "attempts_jsonl_sha256": reference["attempts_jsonl_sha256"],
        "terminal_report_sha256": reference["terminal_report_sha256"],
        "job_id": 28081,
        "cells": len(by_id),
        "rerun": False,
    }


def _append_jsonl(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def _parent_clean_lattice(batch: Any) -> np.ndarray:
    data_utils = __import__(
        "models_ddpm.data_utils",
        fromlist=["lattice_params_to_matrix_torch"],
    )
    lattice = data_utils.lattice_params_to_matrix_torch(
        batch.lengths,
        batch.angles,
    )
    return lattice[0].detach().double().cpu().numpy()


def _structure_from_arrays(
    *,
    lattice: np.ndarray,
    atomic_numbers: np.ndarray,
    fractional_coordinates: np.ndarray,
) -> Any:
    from pymatgen.core import Structure
    from pymatgen.core.lattice import Lattice

    return Structure(
        lattice=Lattice(np.asarray(lattice, dtype=np.float64)),
        species=[int(value) for value in atomic_numbers],
        coords=np.asarray(fractional_coordinates, dtype=np.float64) % 1.0,
        coords_are_cartesian=False,
        to_unit_cell=True,
    )


def _mechanics_from_step(audit: Mapping[str, Any]) -> dict[str, Any]:
    orbit_audits = list(audit["orbit_audits"])
    lattice_audit = audit["lattice_audit"]
    return {
        "discrete_fields_unchanged": bool(audit["discrete_fields_unchanged"]),
        "topology_hash_unchanged": (
            audit["topology_hash_before"] == audit["topology_hash_after"]
        ),
        "orbit_chart_roundtrip_checked": bool(
            audit["orbit_chart_roundtrip_checked"]
        ),
        "orbit_chart_roundtrip_max_abs_error": float(
            audit["maximum_orbit_chart_roundtrip_max_abs_error"]
        ),
        "lattice_chart_roundtrip_max_abs_error": float(
            lattice_audit["chart_roundtrip_max_abs_error"]
        ),
        "lattice_matrix_roundtrip_max_abs_error": float(
            lattice_audit["lattice_roundtrip_max_abs_error"]
        ),
        "maximum_fixed_site_drift": max(
            (
                float(value["fixed_site_drift"])
                for value in orbit_audits
            ),
            default=0.0,
        ),
        "primitive_volume": float(audit["primitive_volume_after"]),
        "retry_or_replacement_used": bool(
            audit["retry_or_replacement_used"]
        ),
        "topology_revision_used": bool(audit["topology_revision_used"]),
    }


def _run_final_projection(
    *,
    proposal: StratifiedState,
    expanded: Any,
    parent_clean_lattice: np.ndarray,
    u_row: Mapping[str, Any],
    projector: WyckoffTangentProjector,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from pymatgen.core import Structure

    u_structure = Structure.from_dict(dict(u_row["final_structure"]))
    u_atomic_numbers = np.asarray(
        [int(site.specie.Z) for site in u_structure],
        dtype=np.int64,
    )
    if not np.array_equal(u_atomic_numbers, expanded.atomic_numbers):
        raise ValueError("U terminal species/order differs from the WQ source")
    frame = ParentLatticeFrame.from_reference_lattices(
        wq_reference_lattice=expanded.primitive_lattice,
        parent_reference_lattice=parent_clean_lattice,
    )
    result = projector.project_parent_step(
        state=proposal,
        proposed_fractional_coordinates=np.asarray(
            u_structure.frac_coords,
            dtype=np.float64,
        ),
        proposed_primitive_lattice=frame.to_wq(
            np.asarray(u_structure.lattice.matrix, dtype=np.float64)
        ),
    )
    final_structure = _structure_from_arrays(
        lattice=frame.to_parent(result.projected_primitive_lattice),
        atomic_numbers=expanded.atomic_numbers,
        fractional_coordinates=result.projected_fractional_coordinates,
    )
    return (
        final_structure,
        result.state.to_dict(),
        {
            "step_audit": result.audit.to_dict(),
            "mechanics": _mechanics_from_step(result.audit.to_dict()),
            "lattice_frame_audit": frame.audit.to_dict(),
            "parent_decoder_calls": 0,
            "projection_calls": 1,
        },
    )


def _run_tangent_trajectory(
    *,
    proposal: StratifiedState,
    expanded: Any,
    parent_clean_lattice: np.ndarray,
    batch: Any,
    model: Any,
    cell: Any,
    projector: WyckoffTangentProjector,
    numpy_schedules: Any,
    reverse_steps: int,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    forward = forward_noise_manifold_numpy(
        state=proposal,
        catalog=projector.catalog,
        schedules=numpy_schedules,
        cell=cell,
        parent_clean_primitive_lattice=parent_clean_lattice,
    )
    random.seed(cell.reverse_noise_seed)
    np.random.seed(cell.reverse_noise_seed % (1 << 32))
    torch.manual_seed(cell.reverse_noise_seed)
    torch.cuda.manual_seed_all(cell.reverse_noise_seed)
    output_state, trajectory = run_parent_reverse_on_wyckoff_manifold(
        model=model,
        batch=batch,
        forward_state=forward,
        projector=projector,
        reverse_steps=reverse_steps,
    )
    final_structure = _output_structure(output_state)
    projection_records = list(trajectory["projection_records"])
    if len(projection_records) != 2 * reverse_steps:
        raise RuntimeError("T trajectory projection accounting changed")
    step_mechanics = [
        _mechanics_from_step(record["audit"])
        for record in projection_records
    ]
    mechanics = {
        "discrete_fields_unchanged": all(
            value["discrete_fields_unchanged"] for value in step_mechanics
        ),
        "topology_hash_unchanged": all(
            value["topology_hash_unchanged"] for value in step_mechanics
        ),
        "orbit_chart_roundtrip_checked": all(
            value["orbit_chart_roundtrip_checked"] for value in step_mechanics
        ),
        "orbit_chart_roundtrip_max_abs_error": max(
            value["orbit_chart_roundtrip_max_abs_error"]
            for value in step_mechanics
        ),
        "lattice_chart_roundtrip_max_abs_error": max(
            value["lattice_chart_roundtrip_max_abs_error"]
            for value in step_mechanics
        ),
        "lattice_matrix_roundtrip_max_abs_error": max(
            value["lattice_matrix_roundtrip_max_abs_error"]
            for value in step_mechanics
        ),
        "maximum_fixed_site_drift": max(
            value["maximum_fixed_site_drift"] for value in step_mechanics
        ),
        "primitive_volume": float(final_structure.volume),
        "first_reverse_lattice_valid": bool(
            trajectory["first_reverse_lattice_valid"]
        ),
        "all_trajectory_values_finite": bool(
            trajectory["all_trajectory_values_finite"]
        ),
        "input_identity_verified": bool(
            trajectory["input_identity_verified"]
        ),
        "retry_or_replacement_used": bool(
            trajectory["retry_or_replacement_used"]
        ),
        "topology_revision_used": bool(
            trajectory["topology_revision_used"]
        ),
    }
    return (
        final_structure,
        dict(output_state["wq_state"]),
        {
            "forward_audit": forward.audit.to_dict(),
            "trajectory": trajectory,
            "mechanics": mechanics,
            "parent_decoder_calls": int(trajectory["decoder_calls"]),
            "projection_calls": int(trajectory["projection_calls"]),
            "source_atom_count": int(expanded.atom_count),
        },
    )


def _gate_rows(
    rows: list[dict[str, Any]],
    *,
    contract: Mapping[str, Any],
    schedule_ok: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gates = contract["mechanics_gates"]
    summaries: dict[str, Any] = {}
    checks: dict[str, Any] = {"schedule_parity": bool(schedule_ok)}
    for arm in ("F", "T"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        succeeded = [row for row in arm_rows if row["status"] == "succeeded"]
        mechanics = [row["mechanics"] for row in succeeded]
        summary = {
            "terminal_cells": len(arm_rows),
            "succeeded_cells": len(succeeded),
            "failed_cells": len(arm_rows) - len(succeeded),
            "positive_volume_outputs": sum(
                math.isfinite(float(value["primitive_volume"]))
                and float(value["primitive_volume"]) > 0.0
                for value in mechanics
            ),
            "maximum_orbit_chart_roundtrip_max_abs_error": max(
                (
                    float(value["orbit_chart_roundtrip_max_abs_error"])
                    for value in mechanics
                ),
                default=math.inf,
            ),
            "maximum_lattice_chart_roundtrip_max_abs_error": max(
                (
                    float(value["lattice_chart_roundtrip_max_abs_error"])
                    for value in mechanics
                ),
                default=math.inf,
            ),
            "maximum_lattice_matrix_roundtrip_max_abs_error": max(
                (
                    float(value["lattice_matrix_roundtrip_max_abs_error"])
                    for value in mechanics
                ),
                default=math.inf,
            ),
            "maximum_fixed_site_drift": max(
                (
                    float(value["maximum_fixed_site_drift"])
                    for value in mechanics
                ),
                default=math.inf,
            ),
            "all_discrete_fields_unchanged": bool(mechanics)
            and all(
                bool(value["discrete_fields_unchanged"])
                for value in mechanics
            ),
            "all_topology_hashes_unchanged": bool(mechanics)
            and all(bool(value["topology_hash_unchanged"]) for value in mechanics),
            "all_orbit_roundtrips_checked": bool(mechanics)
            and all(
                bool(value["orbit_chart_roundtrip_checked"])
                for value in mechanics
            ),
            "retry_or_replacement_count": sum(
                bool(value["retry_or_replacement_used"])
                for value in mechanics
            ),
            "topology_revision_count": sum(
                bool(value["topology_revision_used"])
                for value in mechanics
            ),
            "parent_decoder_call_counts": sorted(
                {
                    int(row["details"]["parent_decoder_calls"])
                    for row in succeeded
                }
            ),
            "projection_call_counts": sorted(
                {
                    int(row["details"]["projection_calls"])
                    for row in succeeded
                }
            ),
        }
        if arm == "T":
            summary.update(
                {
                    "first_reverse_invalid_lattice_count": sum(
                        not bool(value["first_reverse_lattice_valid"])
                        for value in mechanics
                    ),
                    "nonfinite_trajectory_count": sum(
                        not bool(value["all_trajectory_values_finite"])
                        for value in mechanics
                    ),
                    "input_identity_failure_count": sum(
                        not bool(value["input_identity_verified"])
                        for value in mechanics
                    ),
                    "forward_contract_failure_count": sum(
                        bool(
                            row["details"]["forward_audit"][
                                "parent_schedule_changed"
                            ]
                        )
                        or bool(
                            row["details"]["forward_audit"][
                                "clean_condition_used_as_reverse_state"
                            ]
                        )
                        or bool(
                            row["details"]["forward_audit"][
                                "retry_or_replacement_used"
                            ]
                        )
                        or not bool(
                            row["details"]["forward_audit"]["noise_audit"][
                                "paired_parent_noise_transform"
                            ]
                        )
                        for row in succeeded
                    ),
                }
            )
        summaries[arm] = summary
        checks[f"{arm}_terminal"] = (
            summary["terminal_cells"] == gates["terminal_cells_per_arm"]
        )
        checks[f"{arm}_success"] = (
            summary["succeeded_cells"] == gates["successful_cells_per_arm"]
        )
        checks[f"{arm}_positive_volume"] = (
            summary["positive_volume_outputs"]
            == gates["positive_volume_outputs_per_arm"]
        )
        checks[f"{arm}_discrete"] = (
            summary["all_discrete_fields_unchanged"]
            and summary["all_topology_hashes_unchanged"]
            and summary["retry_or_replacement_count"] == 0
            and summary["topology_revision_count"] == 0
        )
        checks[f"{arm}_orbit_roundtrip"] = (
            summary["all_orbit_roundtrips_checked"]
            and summary["maximum_orbit_chart_roundtrip_max_abs_error"]
            <= gates["orbit_chart_roundtrip_max_abs_error"]
        )
        checks[f"{arm}_lattice_roundtrip"] = (
            summary["maximum_lattice_chart_roundtrip_max_abs_error"]
            <= gates["lattice_chart_roundtrip_max_abs_error"]
            and summary["maximum_lattice_matrix_roundtrip_max_abs_error"]
            <= gates["lattice_matrix_roundtrip_max_abs_error"]
        )
        checks[f"{arm}_fixed_site"] = (
            summary["maximum_fixed_site_drift"]
            <= gates["fixed_site_drift_max_abs"]
        )
        checks[f"{arm}_call_budget"] = (
            summary["parent_decoder_call_counts"]
            == [gates[f"{arm}_parent_decoder_calls_per_cell"]]
            and summary["projection_call_counts"]
            == [gates[f"{arm}_projection_calls_per_cell"]]
        )
    checks["T_trajectory"] = (
        summaries["T"]["first_reverse_invalid_lattice_count"] == 0
        and summaries["T"]["nonfinite_trajectory_count"] == 0
        and summaries["T"]["input_identity_failure_count"] == 0
        and summaries["T"]["forward_contract_failure_count"] == 0
    )
    return summaries, checks


def _run(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    started = time.time()
    contract, contract_sha256 = _load_contract(args.contract)
    _require_sha256(args.execution_patch_sha256, name="execution patch")
    implementation = contract["implementation"]
    tangent_source = ROOT / implementation["tangent_bridge_source"]
    if sha256_file(tangent_source) != (
        implementation["tangent_bridge_source_sha256"]
    ):
        raise ValueError("installed tangent bridge source SHA256 changed")
    if args.device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("WTB-32 requires one CUDA A800 allocation")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("WTB-32 requires exactly one visible GPU")
    device_name = torch.cuda.get_device_name(args.device)
    if "A800" not in device_name:
        raise RuntimeError(f"WTB-32 requires A800, observed {device_name}")
    if "SLURM_JOB_ID" in os.environ:
        allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
        if allocated_cpus <= 0 or allocated_cpus > 8:
            raise RuntimeError("WTB-32 Slurm allocation exceeds 8 CPU per A800")

    source = contract["source_panel"]
    source_rows = _load_source_rows(args.source_generation, source)
    selected = _select_rows(source_rows, source)
    u_by_cell, u_summary = _validate_u_reference(
        contract=contract,
        terminal_audit_path=args.u_terminal_audit,
        selection_path=args.u_selection_manifest,
        attempts_path=args.u_attempts,
        terminal_report_path=args.u_terminal_report,
        selected_source_rows=selected,
    )
    reference_payload = {
        "schema": "wq_wyckoff_tangent_u_reference_v1",
        "contract_sha256": contract_sha256,
        "source_generation_sha256": source["generation_jsonl_sha256"],
        "source_attempt_ids": [row["attempt_id"] for row in selected],
        "source_proposal_state_sha256": [
            hashlib.sha256(_canonical_json(row["proposal_state"])).hexdigest()
            for row in selected
        ],
        "u": u_summary,
        "u_rerun": False,
        "selection_uses_scientific_outcomes": False,
    }
    reference_path = output / "u_reference_manifest.json"
    write_json_exclusive(reference_path, reference_payload)

    if sha256_file(args.checkpoint) != contract["parent"]["checkpoint_sha256"]:
        raise ValueError("parent checkpoint SHA256 changed")
    model, parent_identity = load_registered_csp(
        snapshot_root=ROOT / "crystal_dlm/wqcodiff/crysllmgen/upstream",
        checkpoint=args.checkpoint,
        device=args.device,
    )
    schedule = _schedule_audit(
        model,
        float(
            contract["mechanics_gates"][
                "parent_schedule_max_absolute_error"
            ]
        ),
    )
    if not schedule["ok"]:
        raise RuntimeError("strict-loaded parent schedule parity failed")
    strict_load = {
        "ok": True,
        "parent_identity": parent_identity,
        "decoder_state_sha256": _module_state_sha256(model.decoder),
        "time_embedding_state_sha256": _module_state_sha256(
            model.time_embedding
        ),
        "beta_scheduler_state_sha256": _module_state_sha256(
            model.beta_scheduler
        ),
        "sigma_scheduler_state_sha256": _module_state_sha256(
            model.sigma_scheduler
        ),
    }

    cells = build_bridge_cells(base_seed=int(contract["matrix"]["base_noise_seed"]))
    numpy_schedules = build_numpy_parent_schedules()
    catalog = PyXtalChartCatalog()
    projector = WyckoffTangentProjector(catalog)
    reverse_steps = int(contract["matrix"]["reverse_steps_per_t_cell"])
    attempt_rows: list[dict[str, Any]] = []
    attempts_path = output / "attempts.jsonl"
    with attempts_path.open("x", encoding="utf-8") as attempts_handle:
        for cell in cells:
            source_row = selected[cell.panel_index]
            proposal = StratifiedState.from_dict(
                dict(source_row["proposal_state"])
            )
            expanded = expand_state(
                proposal,
                catalog,
                redetect_space_group=False,
            )
            batch = expanded_state_to_parent_batch(expanded, args.device)
            clean_parent_lattice = _parent_clean_lattice(batch)
            for arm in ("F", "T"):
                row_started = time.time()
                base = {
                    "schema": "wq_wyckoff_tangent_preflight_cell_v1",
                    "arm": arm,
                    "cell_id": cell.cell_id,
                    "timestep": cell.timestep,
                    "panel_index": cell.panel_index,
                    "source_attempt_id": source_row["attempt_id"],
                    "forward_noise_seed": cell.forward_noise_seed,
                    "reverse_noise_seed": cell.reverse_noise_seed,
                    "u_reference_row_sha256": hashlib.sha256(
                        _canonical_json(u_by_cell[cell.cell_id])
                    ).hexdigest(),
                    "execution_patch_sha256": args.execution_patch_sha256,
                    "retry_or_replacement_used": False,
                    "topology_revision_used": False,
                    "new_generation_performed": False,
                    "u_rerun": False,
                }
                try:
                    if arm == "F":
                        (
                            final_structure,
                            final_wq_state,
                            details,
                        ) = _run_final_projection(
                            proposal=proposal,
                            expanded=expanded,
                            parent_clean_lattice=clean_parent_lattice,
                            u_row=u_by_cell[cell.cell_id],
                            projector=projector,
                        )
                    else:
                        (
                            final_structure,
                            final_wq_state,
                            details,
                        ) = _run_tangent_trajectory(
                            proposal=proposal,
                            expanded=expanded,
                            parent_clean_lattice=clean_parent_lattice,
                            batch=batch,
                            model=model,
                            cell=cell,
                            projector=projector,
                            numpy_schedules=numpy_schedules,
                            reverse_steps=reverse_steps,
                        )
                    volume = float(final_structure.volume)
                    if not math.isfinite(volume) or volume <= 0.0:
                        raise ValueError("terminal structure has non-positive volume")
                    final_state = StratifiedState.from_dict(final_wq_state)
                    if final_state.topology_hash() != proposal.topology_hash():
                        raise ValueError("terminal WQ topology hash changed")
                    row = {
                        **base,
                        "status": "succeeded",
                        "reason": "",
                        "source_topology_hash": proposal.topology_hash(),
                        "final_topology_hash": final_state.topology_hash(),
                        "final_wq_state": final_wq_state,
                        "final_structure": final_structure.as_dict(),
                        "final_volume": volume,
                        "mechanics": details["mechanics"],
                        "details": details,
                        "walltime_s": time.time() - row_started,
                    }
                except Exception as exc:
                    row = {
                        **base,
                        "status": "failed",
                        "reason": f"{type(exc).__name__}:{exc}",
                        "traceback": traceback.format_exc(),
                        "walltime_s": time.time() - row_started,
                    }
                attempt_rows.append(row)
                _append_jsonl(attempts_handle, row)

    attempts_sha256 = sha256_file(attempts_path)
    summaries, gate_checks = _gate_rows(
        attempt_rows,
        contract=contract,
        schedule_ok=bool(schedule["ok"]),
    )
    ok = all(bool(value) for value in gate_checks.values())
    report = {
        "schema": "wq_wyckoff_tangent_preflight_terminal_v1",
        "ok": ok,
        "acceptance": "PASS" if ok else "FAIL",
        "run_id": contract["run_id"],
        "identity": contract["identity"],
        "contract_sha256": contract_sha256,
        "execution_patch_sha256": args.execution_patch_sha256,
        "tangent_bridge_source_sha256": (
            implementation["tangent_bridge_source_sha256"]
        ),
        "source_generation_sha256": source["generation_jsonl_sha256"],
        "u_reference_manifest_sha256": sha256_file(reference_path),
        "attempts_jsonl_sha256": attempts_sha256,
        "u_reference": u_summary,
        "strict_load": strict_load,
        "schedule_audit": schedule,
        "arm_summaries": summaries,
        "gate_checks": gate_checks,
        "training_performed": False,
        "new_generation_performed": False,
        "u_rerun": False,
        "mlip_used": False,
        "external_api_used": False,
        "scientific_outcome_selection_performed": False,
        "retry_or_replacement_used": False,
        "walltime_s": time.time() - started,
        "gpu": {
            "name": device_name,
            "peak_memory_bytes": int(
                torch.cuda.max_memory_allocated(args.device)
            ),
        },
    }
    write_json_exclusive(output / "terminal_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "configs/experiments/wyckoff_codiffusion"
        / "wq_wyckoff_tangent_bridge_preflight_v1.json",
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
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = _run(args, output)
    except Exception as exc:
        terminal = output / "terminal_report.json"
        if not terminal.exists():
            write_json_exclusive(
                terminal,
                {
                    "schema": "wq_wyckoff_tangent_preflight_terminal_v1",
                    "ok": False,
                    "acceptance": "FAIL",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "traceback": traceback.format_exc(),
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
