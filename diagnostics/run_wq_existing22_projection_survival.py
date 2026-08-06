#!/usr/bin/env python3
"""Evaluate the frozen 22 composition projections without generation or MLIP use."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SCHEMA = "wqcodiff_existing22_projection_survival_contract_v1"
PANEL_ROW_SCHEMA = "wqcodiff_composition_mechanism_panel_input_row_v1"
PROJECTION_ROW_SCHEMA = "wqcodiff_composition_mechanism_panel_row_v1"
PROJECTION_SCHEMA = "wqcodiff_fixed_topology_composition_projection_v1"
THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class SurvivalAuditError(RuntimeError):
    """A fail-closed contract, identity, or topology error."""


@dataclasses.dataclass(frozen=True)
class FrozenProjection:
    attempt_id: str
    ordinal: int
    input_line_number: int
    original_formula: str
    projected_formula: str
    changed_orbit_ids: tuple[str, ...]
    original_state: dict[str, Any]
    projected_state: dict[str, Any]
    projected_state_sha256: str


@dataclasses.dataclass(frozen=True)
class RenderedStructure:
    structure: Any
    structure_dict: dict[str, Any]
    atom_count: int
    volume: float
    redetected_space_group: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SurvivalAuditError(f"{path} is not a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SurvivalAuditError(
                    f"{path}: invalid JSON on line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise SurvivalAuditError(
                    f"{path}: line {line_number} is not a JSON object"
                )
            row["_physical_line_number"] = line_number
            rows.append(row)
    return rows


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def validate_runtime(contract: Mapping[str, Any]) -> dict[str, Any]:
    runtime = contract["runtime"]
    expected_python = Path(str(runtime["python"])).resolve()
    actual_python = Path(sys.executable).resolve()
    if actual_python != expected_python:
        raise SurvivalAuditError(
            f"wrong Python: {actual_python} != {expected_python}"
        )
    if os.environ.get("SLURM_JOB_ID"):
        raise SurvivalAuditError("this evaluation is frozen to the login node, not Slurm")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(
        runtime["cuda_visible_devices"]
    ):
        raise SurvivalAuditError("CUDA_VISIBLE_DEVICES must be explicitly empty")
    expected_threads = int(runtime["numerical_threads"])
    observed_threads = {
        name: int(os.environ.get(name, "0")) for name in THREAD_ENVIRONMENT
    }
    if any(value != expected_threads for value in observed_threads.values()):
        raise SurvivalAuditError(
            f"numerical thread contract mismatch: {observed_threads}"
        )
    return {
        "python": str(actual_python),
        "python_version": sys.version.split()[0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": None,
        "numerical_threads": observed_threads,
    }


def validate_input_identities(
    contract: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, str]]:
    inputs = contract["inputs"]
    paths = {
        "panel": resolve_project_path(inputs["panel"]["path"]),
        "panel_manifest": resolve_project_path(inputs["panel_manifest"]["path"]),
        "projections": resolve_project_path(inputs["projections"]["path"]),
        "projection_report": resolve_project_path(
            inputs["projection_report"]["path"]
        ),
        "snapshot_root": resolve_project_path(inputs["upstream_snapshot"]["root"]),
        "snapshot_manifest": resolve_project_path(
            inputs["upstream_snapshot"]["manifest"]
        ),
    }
    paths["compute_metrics"] = paths["snapshot_root"] / "compute_metrics.py"
    paths["eval_utils"] = paths["snapshot_root"] / "eval_utils.py"
    required_files = {
        "panel": (paths["panel"], inputs["panel"]["sha256"]),
        "panel_manifest": (
            paths["panel_manifest"],
            inputs["panel_manifest"]["sha256"],
        ),
        "projections": (paths["projections"], inputs["projections"]["sha256"]),
        "projection_report": (
            paths["projection_report"],
            inputs["projection_report"]["sha256"],
        ),
        "snapshot_manifest": (
            paths["snapshot_manifest"],
            inputs["upstream_snapshot"]["manifest_sha256"],
        ),
        "compute_metrics": (
            paths["compute_metrics"],
            inputs["upstream_snapshot"]["compute_metrics_sha256"],
        ),
        "eval_utils": (
            paths["eval_utils"],
            inputs["upstream_snapshot"]["eval_utils_sha256"],
        ),
    }
    observed: dict[str, str] = {}
    for name, (path, expected) in required_files.items():
        if not path.is_file():
            raise SurvivalAuditError(f"required input is absent: {path}")
        observed[name] = sha256(path)
        if observed[name] != str(expected):
            raise SurvivalAuditError(
                f"{name} SHA256 mismatch: {observed[name]} != {expected}"
            )
    return paths, observed


def _orbit_geometry(orbit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "orbit_id": orbit["orbit_id"],
        "wyckoff_type": int(orbit["wyckoff_type"]),
        "multiplicity": int(orbit["multiplicity"]),
        "primitive_multiplicity": int(orbit["primitive_multiplicity"]),
        "chart_dimension": int(orbit["chart_dimension"]),
        "free_coordinate": [float(value) for value in orbit["free_coordinate"]],
    }


def _element_set(state: Mapping[str, Any]) -> set[int]:
    return {int(orbit["species"]) for orbit in state["orbits"]}


def _atom_count(state: Mapping[str, Any]) -> int:
    return sum(int(orbit["primitive_multiplicity"]) for orbit in state["orbits"])


def validate_topology_invariants(
    original: Mapping[str, Any],
    projected: Mapping[str, Any],
    declared_changed_orbits: Sequence[str],
) -> None:
    exact_fields = (
        "attempt_id",
        "space_group",
        "lattice_system",
        "lattice_chart",
        "timestep",
        "space_group_committed",
    )
    for field in exact_fields:
        if original.get(field) != projected.get(field):
            raise SurvivalAuditError(f"projected state changed frozen field {field}")
    original_orbits = list(original.get("orbits", ()))
    projected_orbits = list(projected.get("orbits", ()))
    if len(original_orbits) != len(projected_orbits):
        raise SurvivalAuditError("projected state changed orbit count")
    changed: list[str] = []
    for old, new in zip(original_orbits, projected_orbits):
        if _orbit_geometry(old) != _orbit_geometry(new):
            raise SurvivalAuditError(
                f"projected state changed topology/geometry of orbit "
                f"{old.get('orbit_id')}"
            )
        if int(old["species"]) != int(new["species"]):
            changed.append(str(old["orbit_id"]))
    if sorted(changed) != sorted(str(value) for value in declared_changed_orbits):
        raise SurvivalAuditError("declared changed_orbit_ids do not match states")
    if not changed:
        raise SurvivalAuditError("a projected row must change at least one orbit")
    if _atom_count(original) != _atom_count(projected):
        raise SurvivalAuditError("projected state changed primitive atom count")
    if _element_set(original) != _element_set(projected):
        raise SurvivalAuditError("projected state changed the original element set")


def select_frozen_projections(
    panel_rows: Sequence[Mapping[str, Any]],
    projection_rows: Sequence[Mapping[str, Any]],
    *,
    expected_panel_rows: int,
    expected_projection_rows: int,
    expected_ordinals: Sequence[int],
) -> list[FrozenProjection]:
    if len(panel_rows) != expected_panel_rows:
        raise SurvivalAuditError(
            f"panel row count {len(panel_rows)} != {expected_panel_rows}"
        )
    if len(projection_rows) != expected_projection_rows:
        raise SurvivalAuditError(
            f"projection row count {len(projection_rows)} != "
            f"{expected_projection_rows}"
        )
    panel_by_id: dict[str, Mapping[str, Any]] = {}
    panel_order: list[str] = []
    for row in panel_rows:
        if row.get("schema") != PANEL_ROW_SCHEMA:
            raise SurvivalAuditError("invalid mechanism panel row schema")
        attempt_id = str(row.get("attempt_id", ""))
        if not attempt_id or attempt_id in panel_by_id:
            raise SurvivalAuditError("missing or duplicate panel attempt_id")
        panel_by_id[attempt_id] = row
        panel_order.append(attempt_id)

    projection_order: list[str] = []
    selected: list[FrozenProjection] = []
    seen: set[str] = set()
    for row in projection_rows:
        if row.get("schema") != PROJECTION_ROW_SCHEMA:
            raise SurvivalAuditError("invalid composition projection row schema")
        attempt_id = str(row.get("attempt_id", ""))
        if not attempt_id or attempt_id in seen:
            raise SurvivalAuditError("missing or duplicate projection attempt_id")
        seen.add(attempt_id)
        projection_order.append(attempt_id)
        if attempt_id not in panel_by_id:
            raise SurvivalAuditError("projection attempt is absent from frozen panel")
        projection = row.get("projection")
        if not isinstance(projection, Mapping):
            raise SurvivalAuditError("projection payload is absent")
        if projection.get("schema") != PROJECTION_SCHEMA:
            raise SurvivalAuditError("invalid projection payload schema")
        if projection.get("attempt_id") != attempt_id:
            raise SurvivalAuditError("projection attempt_id mismatch")
        if projection.get("status") != "projected":
            continue
        source = panel_by_id[attempt_id]
        if source.get("panel_group") != "no_neutral":
            raise SurvivalAuditError("projected attempt is not in no_neutral group")
        if projection.get("source_reason") != "charge_neutrality_fail":
            raise SurvivalAuditError("projected attempt has wrong source reason")
        original_state = source.get("state")
        projected_state = projection.get("state")
        if not isinstance(original_state, Mapping) or not isinstance(
            projected_state, Mapping
        ):
            raise SurvivalAuditError("projection row has no original/projected state")
        changed = tuple(str(value) for value in projection["changed_orbit_ids"])
        validate_topology_invariants(original_state, projected_state, changed)
        selected.append(
            FrozenProjection(
                attempt_id=attempt_id,
                ordinal=int(source["ordinal"]),
                input_line_number=int(row["input_line_number"]),
                original_formula=str(projection["original_formula"]),
                projected_formula=str(projection["projected_formula"]),
                changed_orbit_ids=changed,
                original_state=dict(original_state),
                projected_state=dict(projected_state),
                projected_state_sha256=canonical_sha256(projected_state),
            )
        )
    if panel_order != projection_order:
        raise SurvivalAuditError("projection rows do not preserve frozen panel order")
    selected.sort(key=lambda value: value.ordinal)
    observed_ordinals = [value.ordinal for value in selected]
    if observed_ordinals != [int(value) for value in expected_ordinals]:
        raise SurvivalAuditError(
            f"projected ordinal identity mismatch: {observed_ordinals}"
        )
    if len({value.attempt_id for value in selected}) != len(selected):
        raise SurvivalAuditError("selected projection attempt IDs are not unique")
    return selected


def prepare_backend(
    snapshot_root: Path,
) -> tuple[
    Callable[[Mapping[str, Any]], RenderedStructure],
    Callable[[Any], dict[str, Any]],
]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from crystal_dlm.wqcodiff.charts import PyXtalChartCatalog
    from crystal_dlm.wqcodiff.runtime import expand_state
    from crystal_dlm.wqcodiff.state import StratifiedState

    sys.path.insert(0, str(snapshot_root))
    try:
        from compute_metrics import Crystal
    finally:
        sys.path.pop(0)
    import numpy as np

    catalog = PyXtalChartCatalog(hall_style="spglib")

    def render(state_payload: Mapping[str, Any]) -> RenderedStructure:
        state = StratifiedState.from_dict(dict(state_payload))
        expanded = expand_state(state, catalog, redetect_space_group=True)
        structure = expanded.pymatgen_structure()
        return RenderedStructure(
            structure=structure,
            structure_dict=dict(structure.as_dict()),
            atom_count=int(expanded.atom_count),
            volume=float(expanded.volume),
            redetected_space_group=int(expanded.redetected_space_group),
        )

    def metric(structure: Any) -> dict[str, Any]:
        crystal = Crystal(
            {
                "frac_coords": structure.frac_coords,
                "atom_types": np.asarray(structure.atomic_numbers),
                "lengths": np.asarray(structure.lattice.abc),
                "angles": np.asarray(structure.lattice.angles),
            }
        )
        comp_valid = bool(crystal.comp_valid)
        struct_valid = bool(crystal.struct_valid)
        valid = bool(crystal.valid)
        if valid:
            reason = ""
        elif not bool(getattr(crystal, "constructed", False)):
            reason = str(
                getattr(crystal, "invalid_reason", "construction_invalid")
            )
        elif not comp_valid:
            reason = "composition_invalid"
        elif not struct_valid:
            reason = "structure_invalid"
        else:
            reason = "fingerprint_invalid"
        return {
            "constructed": bool(getattr(crystal, "constructed", False)),
            "comp_valid": comp_valid,
            "struct_valid": struct_valid,
            "valid": valid,
            "fingerprint_valid": bool(
                getattr(crystal, "comp_fp", None) is not None
                and getattr(crystal, "struct_fp", None) is not None
            ),
            "reason": reason,
        }

    return render, metric


def evaluate_projections(
    projections: Sequence[FrozenProjection],
    *,
    renderer: Callable[[Mapping[str, Any]], RenderedStructure],
    metric: Callable[[Any], Mapping[str, Any]],
    minimum_structural_valid: int,
    minimum_joint_valid: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    structures: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for projection in projections:
        structure_record: dict[str, Any] = {
            "schema": "wqcodiff_existing22_rendered_structure_v1",
            "attempt_id": projection.attempt_id,
            "ordinal": projection.ordinal,
            "source": "existing_fixed_topology_projection",
            "original_formula": projection.original_formula,
            "projected_formula": projection.projected_formula,
            "changed_orbit_ids": list(projection.changed_orbit_ids),
            "projected_state_sha256": projection.projected_state_sha256,
            "retry_or_replacement_used": False,
        }
        metric_record: dict[str, Any] = {
            "schema": "wqcodiff_existing22_crysllmgen_metric_attempt_v1",
            "attempt_id": projection.attempt_id,
            "ordinal": projection.ordinal,
            "projected_formula": projection.projected_formula,
            "render_ok": False,
            "constructed": False,
            "comp_valid": False,
            "struct_valid": False,
            "valid": False,
            "fingerprint_valid": False,
            "reason": "",
            "retry_or_replacement_used": False,
        }
        try:
            rendered = renderer(projection.projected_state)
            if (
                rendered.atom_count != _atom_count(projection.projected_state)
                or not math.isfinite(rendered.volume)
                or rendered.volume <= 0
            ):
                raise SurvivalAuditError(
                    "rendered atom count or volume violates state contract"
                )
            structure_record.update(
                {
                    "status": "succeeded",
                    "atom_count": rendered.atom_count,
                    "volume": rendered.volume,
                    "redetected_space_group": rendered.redetected_space_group,
                    "structure": rendered.structure_dict,
                }
            )
            metric_values = dict(metric(rendered.structure))
            required = {
                "constructed",
                "comp_valid",
                "struct_valid",
                "valid",
                "fingerprint_valid",
                "reason",
            }
            if not required.issubset(metric_values):
                raise SurvivalAuditError("metric backend omitted required fields")
            metric_record.update(metric_values)
            metric_record["render_ok"] = True
        except Exception as exc:
            reason = f"{type(exc).__name__}:{exc}"
            structure_record.update({"status": "failed", "reason": reason})
            metric_record["reason"] = reason
        structures.append(structure_record)
        metrics.append(metric_record)

    attempts = len(projections)
    rendered = sum(bool(row["render_ok"]) for row in metrics)
    comp_valid = sum(bool(row["comp_valid"]) for row in metrics)
    struct_valid = sum(bool(row["struct_valid"]) for row in metrics)
    valid = sum(bool(row["valid"]) for row in metrics)
    checks = {
        "rendered_count_exact": rendered == attempts,
        "composition_valid_count_exact": comp_valid == attempts,
        "minimum_structural_valid_count": struct_valid
        >= int(minimum_structural_valid),
        "minimum_joint_valid_count": valid >= int(minimum_joint_valid),
        "one_structure_record_per_attempt": len(structures) == attempts,
        "one_metric_record_per_attempt": len(metrics) == attempts,
        "attempt_ids_unique": len({row["attempt_id"] for row in metrics})
        == attempts,
        "retry_or_replacement_used_false": not any(
            bool(row["retry_or_replacement_used"]) for row in metrics
        ),
    }
    summary = {
        "attempts": attempts,
        "denominator": "all_existing_projected_states",
        "rendered_count": rendered,
        "composition_valid_count": comp_valid,
        "structural_valid_count": struct_valid,
        "joint_valid_count": valid,
        "rendered_rate": rendered / attempts,
        "composition_valid_rate": comp_valid / attempts,
        "structural_valid_rate": struct_valid / attempts,
        "joint_valid_rate": valid / attempts,
        "minimum_structural_valid_count": int(minimum_structural_valid),
        "minimum_joint_valid_count": int(minimum_joint_valid),
        "checks": checks,
        "ok": all(checks.values()),
    }
    return structures, metrics, summary


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("numpy", "pymatgen", "pyxtal", "smact"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


def prepare(
    contract_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Path],
    dict[str, str],
    dict[str, Any],
    list[FrozenProjection],
    Callable[[Mapping[str, Any]], RenderedStructure],
    Callable[[Any], dict[str, Any]],
]:
    contract = load_json(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise SurvivalAuditError("invalid survival contract schema")
    output = resolve_project_path(contract["output"]["directory"])
    if output.exists():
        raise SurvivalAuditError(f"frozen output identity already exists: {output}")
    runtime = validate_runtime(contract)
    paths, observed_hashes = validate_input_identities(contract)
    panel_rows = load_jsonl(paths["panel"])
    projection_rows = load_jsonl(paths["projections"])
    projection_report = load_json(paths["projection_report"])
    expected_recovered = int(
        contract["inputs"]["projection_report"]["no_neutral_recovered"]
    )
    if (
        int(projection_report.get("no_neutral_recovered", -1))
        != expected_recovered
        or bool(projection_report.get("retry_or_replacement_used", True))
    ):
        raise SurvivalAuditError("frozen projection report contract mismatch")
    selected = select_frozen_projections(
        panel_rows,
        projection_rows,
        expected_panel_rows=int(contract["inputs"]["panel"]["rows"]),
        expected_projection_rows=int(contract["inputs"]["projections"]["rows"]),
        expected_ordinals=contract["expected_projected_ordinals"],
    )
    expected_attempts = int(contract["denominator"]["attempts"])
    if len(selected) != expected_attempts or expected_recovered != expected_attempts:
        raise SurvivalAuditError("frozen all-22 denominator mismatch")
    renderer, metric = prepare_backend(paths["snapshot_root"])
    runtime["packages"] = package_versions()
    return (
        contract,
        paths,
        observed_hashes,
        runtime,
        selected,
        renderer,
        metric,
    )


def execute(contract_path: Path, *, preflight_only: bool) -> int:
    (
        contract,
        paths,
        observed_hashes,
        runtime,
        selected,
        renderer,
        metric,
    ) = prepare(contract_path)
    preflight = {
        "schema": "wqcodiff_existing22_projection_survival_preflight_v1",
        "ok": True,
        "contract": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "input_sha256": observed_hashes,
        "runtime": runtime,
        "selected_attempts": len(selected),
        "selected_ordinals": [value.ordinal for value in selected],
        "new_generation": False,
        "training": False,
        "projector_rerun": False,
        "mlip_calls": 0,
        "sun_calls": 0,
        "retry_or_replacement_used": False,
    }
    if preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    output = resolve_project_path(contract["output"]["directory"])
    output.mkdir(parents=True, exist_ok=False)
    claim_path = output / str(contract["output"]["claim"])
    claim = {
        "schema": "wqcodiff_existing22_projection_survival_claim_v1",
        "status": "claimed",
        "scientific_call_index": 1,
        "scientific_call_limit": int(contract["scope"]["scientific_call_limit"]),
        "contract": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "inputs": observed_hashes,
        "attempts": len(selected),
        "ordinals": [value.ordinal for value in selected],
        "retry_or_replacement_used": False,
    }
    _write_json_exclusive(claim_path, claim)
    started = time.monotonic()
    try:
        structures, metrics, summary = evaluate_projections(
            selected,
            renderer=renderer,
            metric=metric,
            minimum_structural_valid=int(
                contract["acceptance"]["minimum_structural_valid_count"]
            ),
            minimum_joint_valid=int(
                contract["acceptance"]["minimum_joint_valid_count"]
            ),
        )
        structures_path = output / str(contract["output"]["structures"])
        metrics_path = output / str(contract["output"]["attempt_metrics"])
        _write_jsonl_exclusive(structures_path, structures)
        _write_jsonl_exclusive(metrics_path, metrics)
        report = {
            "schema": "wqcodiff_existing22_projection_survival_report_v1",
            "ok": bool(summary["ok"]),
            "run_id": contract["run_id"],
            "contract": str(contract_path),
            "contract_sha256": sha256(contract_path),
            "claim": str(claim_path),
            "claim_sha256": sha256(claim_path),
            "inputs": {
                name: {"path": str(paths[name]), "sha256": digest}
                for name, digest in observed_hashes.items()
                if name in paths
            },
            "runtime": runtime,
            "summary": summary,
            "ordinals": [value.ordinal for value in selected],
            "projected_state_sha256": {
                str(value.ordinal): value.projected_state_sha256
                for value in selected
            },
            "structures": str(structures_path),
            "structures_sha256": sha256(structures_path),
            "attempt_metrics": str(metrics_path),
            "attempt_metrics_sha256": sha256(metrics_path),
            "scientific_generation_attempts_created": 0,
            "projector_calls": 0,
            "mlip_calls": 0,
            "sun_calls": 0,
            "external_api_calls": 0,
            "retry_or_replacement_used": False,
            "walltime_s": time.monotonic() - started,
            "decision": (
                contract["decision"]["pass"]
                if summary["ok"]
                else contract["decision"]["fail"]
            ),
        }
        report_path = output / str(contract["output"]["report"])
        _write_json_exclusive(report_path, report)
        terminal = {
            "schema": "wqcodiff_existing22_projection_survival_terminal_v1",
            "ok": bool(report["ok"]),
            "status": "PASS" if report["ok"] else "FAIL",
            "report": str(report_path),
            "report_sha256": sha256(report_path),
            "structures_sha256": report["structures_sha256"],
            "attempt_metrics_sha256": report["attempt_metrics_sha256"],
            "attempts": summary["attempts"],
            "rendered_count": summary["rendered_count"],
            "composition_valid_count": summary["composition_valid_count"],
            "structural_valid_count": summary["structural_valid_count"],
            "joint_valid_count": summary["joint_valid_count"],
            "retry_or_replacement_used": False,
        }
        terminal_path = output / str(contract["output"]["terminal_acceptance"])
        _write_json_exclusive(terminal_path, terminal)
        print(json.dumps(terminal, indent=2, sort_keys=True))
        return 0 if terminal["ok"] else 4
    except BaseException as exc:
        failure_path = output / "failure.json"
        if not failure_path.exists():
            _write_json_exclusive(
                failure_path,
                {
                    "schema": "wqcodiff_existing22_projection_survival_failure_v1",
                    "ok": False,
                    "error": f"{type(exc).__name__}:{exc}",
                    "claim": str(claim_path),
                    "claim_sha256": sha256(claim_path),
                    "retry_or_replacement_used": False,
                },
            )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    return execute(args.contract.resolve(), preflight_only=args.preflight_only)


if __name__ == "__main__":
    raise SystemExit(main())
