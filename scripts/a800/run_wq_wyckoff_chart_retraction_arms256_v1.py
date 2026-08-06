#!/usr/bin/env python3
"""Run the paired R/U/T arms for the immutable WTB-256 source panel.

R is the losslessly expanded WQ proposal.  U starts the released parent from
the exact training-time forward-noised geometry.  T uses the same source,
start timestep, and paired base-noise identities while retracting every parent
corrector/predictor proposal to the fixed Wyckoff chart.  Every source yields
one terminal row per arm; failures remain in all downstream denominators.
"""

from __future__ import annotations

import argparse
import dataclasses
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
    _module_state_sha256,
    _schedule_audit,
)
from scripts.a800.run_wq_wyckoff_chart_retraction_preflight_sup28185_v2 import (  # noqa: E402
    PROJECTION_METHOD,
    _run_tangent_trajectory_v2,
)
from scripts.a800.run_wq_wyckoff_tangent_bridge_preflight_v1 import (  # noqa: E402
    _parent_clean_lattice,
)
from crystal_dlm.wqcodiff.charts import PyXtalChartCatalog  # noqa: E402
from crystal_dlm.wqcodiff.contracts import (  # noqa: E402
    ArtifactLedger,
    AttemptLedger,
    AttemptRecord,
    AttemptStatus,
    SeedDeriver,
    write_json_exclusive,
)
from crystal_dlm.wqcodiff.crysllmgen.atom_sampling import (  # noqa: E402
    _output_structure,
    expanded_state_to_parent_batch,
    load_registered_csp,
)
from crystal_dlm.wqcodiff.crysllmgen.bridge_parity import (  # noqa: E402
    CleanProposalCondition,
    build_numpy_parent_schedules,
    forward_noise_numpy,
    forward_noise_torch,
    run_parent_reverse_from_noisy_state,
)
from crystal_dlm.wqcodiff.crysllmgen.gate import sha256_file  # noqa: E402
from crystal_dlm.wqcodiff.crysllmgen.tangent_bridge import (  # noqa: E402
    WyckoffTangentProjector,
)
from crystal_dlm.wqcodiff.crysllmgen.wtb_confirmatory import (  # noqa: E402
    ARM_EXPERIMENT_ID,
    ARM_METHODS,
    ATTEMPTS,
    DECODER_CALLS_PER_DIFFUSION_ARM,
    END_ORDINAL_INCLUSIVE,
    IDENTITY,
    REVERSE_START_TIMESTEP,
    REVERSE_STEPS,
    SAMPLING_SEED,
    SOURCE_METHOD,
    START_ORDINAL,
    TRAINING_SEED,
    build_confirmatory_cells,
    composition_signature,
    source_signature,
)
from crystal_dlm.wqcodiff.runtime import expand_state  # noqa: E402
from crystal_dlm.wqcodiff.state import StratifiedState  # noqa: E402


CONTRACT_SCHEMA = "wq_wyckoff_chart_retraction_confirmatory256_contract_v1"
SOURCE_SCHEMA = "wq_wyckoff_chart_retraction_source_attempt_v1"
SOURCE_REPORT_SCHEMA = "wq_wyckoff_chart_retraction_source_report_v1"
GENERATION_SCHEMA = "wqcodiff_generation_attempt_v1"
MECHANICS_SCHEMA = "wq_wyckoff_chart_retraction_arm_mechanics_v1"
REPORT_SCHEMA = "wq_wyckoff_chart_retraction_arms_report_v1"
STAGE = "paired_rut_generation"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: str, *, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be one lowercase SHA256")
    return value


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA or payload.get("identity") != IDENTITY:
        raise ValueError("unexpected WTB-256 contract identity")
    if payload.get("status") != "local_built_remote_execution_not_authorized":
        raise ValueError("WTB-256 contract status changed")
    matrix = payload["matrix"]
    if (
        matrix["arms"] != ARM_METHODS
        or int(matrix["reverse_start_timestep"]) != REVERSE_START_TIMESTEP
        or int(matrix["reverse_steps"]) != REVERSE_STEPS
        or int(matrix["decoder_calls_per_diffusion_arm"])
        != DECODER_CALLS_PER_DIFFUSION_ARM
        or matrix["tangent_projection_method"] != PROJECTION_METHOD
    ):
        raise ValueError("WTB-256 R/U/T matrix changed")
    resources = payload["resources"]
    if int(resources["a800"]) != 1 or int(resources["cpus"]) > 8:
        raise ValueError("WTB-256 resource contract exceeds 8 CPU per A800")
    if not all(bool(value) for value in payload["forbidden_actions"].values()):
        raise ValueError("a forbidden WTB-256 action was enabled")
    return payload, sha256_file(path)


def _load_source_evidence(
    *,
    source_jsonl: Path,
    source_report: Path,
    contract_sha256: str,
    execution_patch_sha256: str,
) -> list[dict[str, Any]]:
    report = json.loads(source_report.read_text(encoding="utf-8"))
    if (
        report.get("schema") != SOURCE_REPORT_SCHEMA
        or report.get("ok") is not True
        or report.get("acceptance") != "PASS"
        or int(report.get("submitted", -1)) != ATTEMPTS
        or int(report.get("terminal", -1)) != ATTEMPTS
        or report.get("contract_sha256") != contract_sha256
        or report.get("execution_patch_sha256") != execution_patch_sha256
        or report.get("source_attempts_sha256") != sha256_file(source_jsonl)
        or report.get("retry_or_replacement_used") is not False
    ):
        raise ValueError("WTB-256 source report is not an immutable PASS")
    rows = [
        json.loads(line)
        for line in source_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cells = build_confirmatory_cells()
    if len(rows) != ATTEMPTS:
        raise ValueError("WTB-256 source denominator changed")
    for row, cell in zip(rows, cells, strict=True):
        if (
            row.get("schema") != SOURCE_SCHEMA
            or row.get("identity") != IDENTITY
            or row.get("method") != SOURCE_METHOD
            or row.get("attempt_id") != cell.source_attempt_id
            or row.get("pair_id") != cell.pair_id
            or int(row.get("ordinal", -1)) != cell.ordinal
            or int(row.get("proposal_seed", -1)) != cell.proposal_seed
            or row.get("arm_attempt_ids") != dict(cell.arm_attempt_ids)
            or row.get("contract_sha256") != contract_sha256
            or row.get("execution_patch_sha256") != execution_patch_sha256
            or row.get("retry_or_replacement_used") is not False
            or row.get("best_of_or_rerank_used") is not False
            or row.get("status") not in {"succeeded", "failed"}
        ):
            raise ValueError(f"WTB-256 source row changed at ordinal {cell.ordinal}")
    return rows


def _require_runtime() -> str:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("WTB-256 R/U/T must run through Slurm")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if not 1 <= cpus <= 8:
        raise RuntimeError("WTB-256 R/U/T exceeds 8 CPU per A800")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("WTB-256 R/U/T requires one visible GPU")
    device_name = torch.cuda.get_device_name(0)
    if "A800" not in device_name:
        raise RuntimeError(f"WTB-256 requires A800, observed {device_name}")
    return device_name


def _validate_source_success(
    row: Mapping[str, Any],
    *,
    catalog: PyXtalChartCatalog,
) -> tuple[StratifiedState, Any]:
    if row["status"] != "succeeded":
        raise ValueError("source row is not successful")
    proposal_payload = row.get("proposal_state")
    if not isinstance(proposal_payload, Mapping):
        raise ValueError("successful source has no proposal state")
    proposal = StratifiedState.from_dict(dict(proposal_payload))
    expanded = expand_state(
        proposal,
        catalog,
        redetect_space_group=False,
    )
    topology_hash = proposal.topology_hash()
    if (
        topology_hash != row.get("proposal_topology_hash")
        or composition_signature(expanded.atomic_numbers)
        != row.get("composition_signature")
        or source_signature(
            proposal_state=proposal.to_dict(),
            topology_hash=topology_hash,
            atomic_numbers=expanded.atomic_numbers,
        )
        != row.get("source_signature")
        or [int(value) for value in expanded.atomic_numbers]
        != row.get("atomic_numbers")
    ):
        raise ValueError("successful source expansion identity changed")
    return proposal, expanded


def _generation_base(
    *,
    arm: str,
    source: Mapping[str, Any],
    cell: Any,
    contract_sha256: str,
    execution_patch_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": GENERATION_SCHEMA,
        "identity": IDENTITY,
        "attempt_id": cell.arm_attempt_ids[arm],
        "method": ARM_METHODS[arm],
        "arm": arm,
        "status": "failed",
        "reason": "",
        "ordinal": cell.ordinal,
        "source_attempt_id": cell.source_attempt_id,
        "pair_id": cell.pair_id,
        "proposal_seed": cell.proposal_seed,
        "forward_noise_seed": cell.forward_noise_seed,
        "reverse_noise_seed": cell.reverse_noise_seed,
        "training_seed": TRAINING_SEED,
        "sampling_seed": SAMPLING_SEED,
        "source_signature": source.get("source_signature"),
        "source_topology_hash": source.get("proposal_topology_hash"),
        "composition_signature": source.get("composition_signature"),
        "structure": None,
        "minimum_pair_distance_angstrom": None,
        "volume_per_atom_angstrom3": None,
        "density_g_cm3": None,
        "collision_free_at_0p5_angstrom": False,
        "retry_or_replacement_used": False,
        "best_of_or_rerank_used": False,
        "training_performed": False,
        "mlip_used": False,
        "external_api_used": False,
        "contract_sha256": contract_sha256,
        "execution_patch_sha256": execution_patch_sha256,
    }


def _geometry_descriptors(structure: Any) -> dict[str, Any]:
    atom_count = int(len(structure))
    if atom_count <= 0:
        raise ValueError("terminal structure contains no atoms")
    distances = np.asarray(structure.distance_matrix, dtype=np.float64)
    if distances.shape != (atom_count, atom_count):
        raise ValueError("terminal distance matrix shape changed")
    masked = distances + np.eye(atom_count, dtype=np.float64) * 10.5
    minimum_distance = float(masked.min())
    volume = float(structure.volume)
    density = float(structure.density)
    volume_per_atom = volume / atom_count
    if (
        not math.isfinite(minimum_distance)
        or minimum_distance < 0.0
        or any(
            not math.isfinite(value) or value <= 0.0
            for value in (volume, density, volume_per_atom)
        )
    ):
        raise ValueError("terminal geometry descriptors are nonfinite/nonpositive")
    return {
        "minimum_pair_distance_angstrom": minimum_distance,
        "volume_per_atom_angstrom3": volume_per_atom,
        "density_g_cm3": density,
        "collision_free_at_0p5_angstrom": (
            minimum_distance >= 0.5 and volume >= 0.1
        ),
    }


def _run(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    started_all = time.monotonic()
    device_name = _require_runtime()
    contract, contract_sha256 = _load_contract(args.contract)
    execution_patch = _require_sha256(
        args.execution_patch_sha256,
        name="execution patch",
    )
    sources = _load_source_evidence(
        source_jsonl=args.source_jsonl,
        source_report=args.source_report,
        contract_sha256=contract_sha256,
        execution_patch_sha256=execution_patch,
    )
    if sha256_file(args.checkpoint) != contract["frozen_assets"]["parent_checkpoint_sha256"]:
        raise ValueError("WTB-256 parent checkpoint changed")

    output.mkdir(parents=True, exist_ok=False)
    arm_ledgers = {
        arm: ArtifactLedger(output / f"{arm.lower()}_generation.jsonl")
        for arm in ARM_METHODS
    }
    mechanics = ArtifactLedger(output / "arm_mechanics.jsonl")
    tangent_trajectory_path = output / "t_trajectory_evidence.jsonl"
    tangent_trajectory_path.touch(exist_ok=False)
    tangent_trajectories = ArtifactLedger(tangent_trajectory_path)
    attempt_ledger = AttemptLedger(output / "attempt_ledger.jsonl")
    arm_deriver = SeedDeriver(
        "crysllmgen_wyckoff_georev_v4",
        ARM_EXPERIMENT_ID,
    )
    cells = build_confirmatory_cells()
    for cell in cells:
        for arm, method in ARM_METHODS.items():
            attempt_ledger.append(
                AttemptRecord(
                    attempt_id=cell.arm_attempt_ids[arm],
                    method=method,
                    training_seed=TRAINING_SEED,
                    sampling_seed=SAMPLING_SEED,
                    stage=STAGE,
                    status=AttemptStatus.SUBMITTED,
                    seed=arm_deriver.derive(
                        training_seed=TRAINING_SEED,
                        sampling_seed=SAMPLING_SEED,
                        attempt_id=cell.arm_attempt_ids[arm],
                        stage=STAGE,
                    ),
                    metadata={
                        "arm": arm,
                        "ordinal": cell.ordinal,
                        "pair_id": cell.pair_id,
                        "source_attempt_id": cell.source_attempt_id,
                    },
                )
            )

    model, parent_identity = load_registered_csp(
        snapshot_root=ROOT / "crystal_dlm/wqcodiff/crysllmgen/upstream",
        checkpoint=args.checkpoint,
        device=args.device,
    )
    schedule = _schedule_audit(
        model,
        float(contract["integrity_gates"]["parent_schedule_max_absolute_error"]),
    )
    if not schedule["ok"]:
        raise RuntimeError("WTB-256 strict-loaded parent schedule parity failed")
    strict_load = {
        "ok": True,
        "parent_identity": parent_identity,
        "decoder_state_sha256": _module_state_sha256(model.decoder),
        "time_embedding_state_sha256": _module_state_sha256(model.time_embedding),
        "beta_scheduler_state_sha256": _module_state_sha256(model.beta_scheduler),
        "sigma_scheduler_state_sha256": _module_state_sha256(model.sigma_scheduler),
    }
    catalog = PyXtalChartCatalog()
    projector = WyckoffTangentProjector(catalog)
    numpy_schedules = build_numpy_parent_schedules()
    terminal_counts = {
        arm: {"succeeded": 0, "failed": 0}
        for arm in ARM_METHODS
    }

    for source, cell in zip(sources, cells, strict=True):
        proposal = None
        expanded = None
        batch = None
        parent_clean_lattice = None
        source_failure = ""
        if source["status"] == "succeeded":
            try:
                proposal, expanded = _validate_source_success(
                    source,
                    catalog=catalog,
                )
                batch = expanded_state_to_parent_batch(expanded, args.device)
                parent_clean_lattice = _parent_clean_lattice(batch)
            except Exception as exc:
                source_failure = f"source_revalidation:{type(exc).__name__}:{exc}"
        else:
            source_failure = "source_generation:" + str(source.get("reason", "failed"))

        for arm, method in ARM_METHODS.items():
            started = time.monotonic()
            row = _generation_base(
                arm=arm,
                source=source,
                cell=cell,
                contract_sha256=contract_sha256,
                execution_patch_sha256=execution_patch,
            )
            mechanics_row: dict[str, Any] = {
                "schema": MECHANICS_SCHEMA,
                "identity": IDENTITY,
                "attempt_id": cell.arm_attempt_ids[arm],
                "method": method,
                "arm": arm,
                "ordinal": cell.ordinal,
                "source_attempt_id": cell.source_attempt_id,
                "pair_id": cell.pair_id,
                "status": "failed",
                "reason": "",
                "parent_decoder_calls": 0,
                "projection_calls": 0,
                "retry_or_replacement_used": False,
                "topology_revision_used": False,
                "contract_sha256": contract_sha256,
                "execution_patch_sha256": execution_patch,
            }
            reason = source_failure
            calls: dict[str, int] = {"parent_decoder": 0, "projection": 0}
            status = AttemptStatus.FAILED
            try:
                if reason:
                    raise ValueError(reason)
                assert proposal is not None
                assert expanded is not None
                assert batch is not None
                assert parent_clean_lattice is not None
                if arm == "R":
                    final_structure = expanded.pymatgen_structure()
                    mechanics_detail = {
                        "raw_source_identity": True,
                        "topology_hash_unchanged": True,
                        "composition_unchanged": True,
                        "parent_decoder_calls": 0,
                        "projection_calls": 0,
                    }
                elif arm == "U":
                    condition = CleanProposalCondition(
                        frac_coords=expanded.fractional_coordinates,
                        lattice=parent_clean_lattice,
                    )
                    forward = forward_noise_numpy(
                        condition,
                        schedules=numpy_schedules,
                        cell=cell,
                    )
                    clean_x = torch.tensor(
                        np.array(condition.frac_coords, copy=True),
                        dtype=batch.frac_coords.dtype,
                        device=args.device,
                    )
                    clean_l = torch.tensor(
                        np.array(condition.lattice, copy=True)[None, :, :],
                        dtype=batch.lengths.dtype,
                        device=args.device,
                    )
                    coordinate_noise = torch.tensor(
                        np.array(forward.state.coordinate_noise, copy=True),
                        dtype=clean_x.dtype,
                        device=args.device,
                    )
                    lattice_noise = torch.tensor(
                        np.array(forward.state.lattice_noise, copy=True)[None, :, :],
                        dtype=clean_l.dtype,
                        device=args.device,
                    )
                    bridge_state = forward_noise_torch(
                        clean_frac_coords=clean_x,
                        clean_lattice=clean_l,
                        coordinate_noise=coordinate_noise,
                        lattice_noise=lattice_noise,
                        timestep=cell.timestep,
                        alphas_cumprod=model.beta_scheduler.alphas_cumprod,
                        coordinate_sigmas=model.sigma_scheduler.sigmas,
                    )
                    random.seed(cell.reverse_noise_seed)
                    np.random.seed(cell.reverse_noise_seed % (1 << 32))
                    torch.manual_seed(cell.reverse_noise_seed)
                    torch.cuda.manual_seed_all(cell.reverse_noise_seed)
                    output_state, trajectory = run_parent_reverse_from_noisy_state(
                        model=model,
                        batch=batch,
                        bridge_state=bridge_state,
                        reverse_steps=REVERSE_STEPS,
                    )
                    final_structure = _output_structure(output_state)
                    calls["parent_decoder"] = int(trajectory["decoder_calls"])
                    mechanics_detail = {
                        "forward_condition_sha256": condition.sha256,
                        "forward_state_condition_sha256": (
                            forward.state.condition_sha256
                        ),
                        "timestep": int(cell.timestep),
                        "reverse_steps": int(trajectory["reverse_steps"]),
                        "timestep_grid": list(trajectory["timestep_grid"]),
                        "first_reverse_lattice_valid": bool(
                            trajectory["first_reverse_lattice_valid"]
                        ),
                        "all_trajectory_values_finite": bool(
                            trajectory["all_trajectory_values_finite"]
                        ),
                        "clean_condition_used_as_reverse_state": bool(
                            trajectory["clean_condition_used_as_reverse_state"]
                        ),
                        "topology_hash_unchanged": None,
                        "composition_unchanged": True,
                        "parent_decoder_calls": calls["parent_decoder"],
                        "projection_calls": 0,
                    }
                else:
                    final_structure, final_wq_state, details = (
                        _run_tangent_trajectory_v2(
                            proposal=proposal,
                            expanded=expanded,
                            parent_clean_lattice=parent_clean_lattice,
                            batch=batch,
                            model=model,
                            cell=cell,
                            projector=projector,
                            numpy_schedules=numpy_schedules,
                            reverse_steps=REVERSE_STEPS,
                        )
                    )
                    final_state = StratifiedState.from_dict(final_wq_state)
                    if final_state.topology_hash() != proposal.topology_hash():
                        raise ValueError("T terminal topology hash changed")
                    calls["parent_decoder"] = int(details["parent_decoder_calls"])
                    calls["projection"] = int(details["projection_calls"])
                    projection_records = list(
                        details["trajectory"]["projection_records"]
                    )
                    tangent_trajectories.append(
                        {
                            "schema": "wq_wyckoff_chart_retraction_trajectory_v1",
                            "attempt_id": cell.arm_attempt_ids[arm],
                            "ordinal": cell.ordinal,
                            "pair_id": cell.pair_id,
                            "projection_records": projection_records,
                            "projection_records_sha256": hashlib.sha256(
                                _canonical_json(projection_records)
                            ).hexdigest(),
                        }
                    )
                    mechanics_detail = {
                        "forward_audit": details["forward_audit"],
                        "mechanics": details["mechanics"],
                        "timestep": int(cell.timestep),
                        "reverse_steps": REVERSE_STEPS,
                        "timestep_grid": list(details["trajectory"]["timestep_grid"]),
                        "projection_records": len(projection_records),
                        "projection_records_sha256": hashlib.sha256(
                            _canonical_json(projection_records)
                        ).hexdigest(),
                        "topology_hash_unchanged": True,
                        "composition_unchanged": True,
                        "parent_decoder_calls": calls["parent_decoder"],
                        "projection_calls": calls["projection"],
                    }

                volume = float(final_structure.volume)
                geometry = _geometry_descriptors(final_structure)
                atomic_numbers = [
                    int(value) for value in final_structure.atomic_numbers
                ]
                observed_composition = composition_signature(atomic_numbers)
                if (
                    not math.isfinite(volume)
                    or volume <= 0.0
                    or observed_composition != source["composition_signature"]
                ):
                    raise ValueError("terminal geometry/composition integrity failed")
                if (
                    arm in {"U", "T"}
                    and calls["parent_decoder"]
                    != DECODER_CALLS_PER_DIFFUSION_ARM
                ):
                    raise ValueError("parent decoder call budget changed")
                if arm == "T" and (
                    calls["projection"] != DECODER_CALLS_PER_DIFFUSION_ARM
                    or mechanics_detail["mechanics"][
                        "lattice_projection_methods"
                    ]
                    != [PROJECTION_METHOD]
                    or not mechanics_detail["mechanics"][
                        "all_chart_retraction_audit_values_finite"
                    ]
                ):
                    raise ValueError("T every-step global chart retraction changed")
                row.update(
                    {
                        "status": "succeeded",
                        "structure": final_structure.as_dict(),
                        "structure_sha256": hashlib.sha256(
                            _canonical_json(final_structure.as_dict())
                        ).hexdigest(),
                        "volume": volume,
                        "atom_count": len(final_structure),
                        "atomic_numbers": atomic_numbers,
                        "observed_composition_signature": observed_composition,
                        **geometry,
                        "calls": calls,
                    }
                )
                mechanics_row.update(
                    {
                        "status": "succeeded",
                        "parent_decoder_calls": calls["parent_decoder"],
                        "projection_calls": calls["projection"],
                        "details": mechanics_detail,
                    }
                )
                status = AttemptStatus.SUCCEEDED
                reason = ""
                terminal_counts[arm]["succeeded"] += 1
            except Exception as exc:
                if not reason:
                    reason = f"{type(exc).__name__}:{exc}"
                row.update(
                    {
                        "reason": reason,
                        "traceback": traceback.format_exc(),
                        "calls": calls,
                    }
                )
                mechanics_row.update(
                    {
                        "reason": reason,
                        "traceback": traceback.format_exc(),
                        "parent_decoder_calls": calls["parent_decoder"],
                        "projection_calls": calls["projection"],
                    }
                )
                terminal_counts[arm]["failed"] += 1
            elapsed = time.monotonic() - started
            row["walltime_s"] = elapsed
            mechanics_row["walltime_s"] = elapsed
            digest = arm_ledgers[arm].append(row)
            mechanics.append(mechanics_row)
            attempt_ledger.append(
                AttemptRecord(
                    attempt_id=cell.arm_attempt_ids[arm],
                    method=method,
                    training_seed=TRAINING_SEED,
                    sampling_seed=SAMPLING_SEED,
                    stage=STAGE,
                    status=status,
                    reason=reason,
                    artifact_hash=digest,
                    seed=arm_deriver.derive(
                        training_seed=TRAINING_SEED,
                        sampling_seed=SAMPLING_SEED,
                        attempt_id=cell.arm_attempt_ids[arm],
                        stage=STAGE,
                    ),
                    calls=calls,
                    walltime_s=elapsed,
                    metadata={
                        "arm": arm,
                        "ordinal": cell.ordinal,
                        "pair_id": cell.pair_id,
                        "source_attempt_id": cell.source_attempt_id,
                    },
                )
            )

    expected_ids = [
        cell.arm_attempt_ids[arm]
        for cell in cells
        for arm in ARM_METHODS
    ]
    audit = attempt_ledger.audit(
        seed_deriver=arm_deriver,
        terminal_stage=STAGE,
        expected_attempt_ids=expected_ids,
    )
    denominator_ok = all(
        values["succeeded"] + values["failed"] == ATTEMPTS
        for values in terminal_counts.values()
    )
    report = {
        "schema": REPORT_SCHEMA,
        "ok": bool(audit.ok and denominator_ok),
        "acceptance": "PASS" if audit.ok and denominator_ok else "FAIL",
        "identity": IDENTITY,
        "contract_sha256": contract_sha256,
        "execution_patch_sha256": execution_patch,
        "source_attempts_sha256": sha256_file(args.source_jsonl),
        "source_report_sha256": sha256_file(args.source_report),
        "start_ordinal": START_ORDINAL,
        "end_ordinal_inclusive": END_ORDINAL_INCLUSIVE,
        "attempts_per_arm": ATTEMPTS,
        "arm_methods": ARM_METHODS,
        "terminal_counts": terminal_counts,
        "reverse_start_timestep": REVERSE_START_TIMESTEP,
        "reverse_steps": REVERSE_STEPS,
        "decoder_calls_per_diffusion_arm": (
            DECODER_CALLS_PER_DIFFUSION_ARM
        ),
        "tangent_projection_method": PROJECTION_METHOD,
        "strict_load": strict_load,
        "schedule_audit": schedule,
        "attempt_audit": dataclasses.asdict(audit),
        "artifacts": {
            f"{arm}_generation_sha256": sha256_file(
                output / f"{arm.lower()}_generation.jsonl"
            )
            for arm in ARM_METHODS
        },
        "mechanics_sha256": sha256_file(output / "arm_mechanics.jsonl"),
        "t_trajectory_evidence_sha256": sha256_file(tangent_trajectory_path),
        "attempt_ledger_sha256": sha256_file(output / "attempt_ledger.jsonl"),
        "retry_or_replacement_used": False,
        "best_of_or_rerank_used": False,
        "training_performed": False,
        "mlip_used": False,
        "external_api_used": False,
        "gpu": {
            "name": device_name,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(args.device)),
        },
        "walltime_s": time.monotonic() - started_all,
    }
    write_json_exclusive(output / "arms_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
        "source_jsonl",
        "source_report",
        "checkpoint",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    try:
        report = _run(args, args.output_dir)
    except Exception as exc:
        if args.output_dir.is_dir():
            terminal = args.output_dir / "arms_report.json"
            if not terminal.exists():
                write_json_exclusive(
                    terminal,
                    {
                        "schema": REPORT_SCHEMA,
                        "ok": False,
                        "acceptance": "FAIL",
                        "identity": IDENTITY,
                        "reason": f"{type(exc).__name__}:{exc}",
                        "traceback": traceback.format_exc(),
                        "retry_or_replacement_used": False,
                        "training_performed": False,
                        "mlip_used": False,
                        "external_api_used": False,
                    },
                )
        raise
    print(json.dumps(report, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
