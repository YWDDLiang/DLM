"""Validation and command discovery for the immutable paper pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping


MAINLINE_STAGE_ORDER = (
    "build-scientific-support",
    "build-planner-data",
    "train-planner",
    "sample-plan",
    "build-dlm-data",
    "train-dlm",
    "train-periodic-dlm",
    "generate-refine-headline",
    "evaluate-headline",
    "finalize-headline",
)
MAINLINE_SCHEMA = "c3fd_g2_paper_pipeline_v1"
DEFAULT_CONFIG = Path("configs/paper/mainline_v1.json")
REQUIRED_INVARIANTS = {
    "one_plan_one_trajectory": True,
    "exact_N_and_composition": True,
    "train_serve_plan_schema_equal": True,
    "fixed_requested_denominator": True,
    "retry_replacement_rerank_best_of_n": False,
    "test_outcome_training": False,
}
EXPECTED_COMPONENT_SCHEMAS = {
    "configs/paper/planner_fused_v1.json": "paper_science_constrained_llm_planner_v1",
    "configs/paper/compact_v2_sft_v1.json": "paper_plan_conditioned_crystal_dlm_v1",
    "configs/paper/g2_pbc_r_v1.json": "paper_periodic_relational_denoising_v1",
    "configs/paper/inference_stream17_v1.json": "paper_fixed_one_trajectory_inference_v1",
    "configs/paper/model494_tau800_v1.json": "paper_fixed_terminal_diffusion_v1",
    "configs/paper/evaluation_fixed256_v1.json": "paper_fixed256_physical_evaluation_v1",
}


class ManifestError(ValueError):
    """Raised when the paper manifest no longer matches the frozen method."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read JSON manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"manifest must be a JSON object: {path}")
    return value


def _relative_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{field} must be a nonempty repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ManifestError(f"{field} escapes the repository: {value}")
    return path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def validate_manifest(
    manifest: Mapping[str, Any], *, root: Path | None = None
) -> dict[str, Any]:
    """Validate method identity, paths, ordering, and scientific invariants."""

    root = (root or repository_root()).resolve()
    _require(manifest.get("schema") == MAINLINE_SCHEMA, "paper pipeline schema changed")
    factorization = manifest.get("factorization")
    _require(isinstance(factorization, Mapping), "factorization is missing")
    _require(
        set(factorization) == {"plan", "raw_structure", "final_structure"},
        "coupled factorization must contain plan/raw/final transitions",
    )

    stages = manifest.get("stages")
    _require(isinstance(stages, list), "stages must be a list")
    stage_ids = tuple(row.get("id") for row in stages if isinstance(row, Mapping))
    _require(stage_ids == MAINLINE_STAGE_ORDER, "mainline stage order changed")

    checked_paths: list[str] = []
    for row in stages:
        _require(isinstance(row, Mapping), "every stage must be an object")
        contract = _relative_path(row.get("contract"), field=f"{row.get('id')}.contract")
        _require((root / contract).is_file(), f"missing stage contract: {contract}")
        checked_paths.append(contract.as_posix())
        implementations = row.get("implementation")
        _require(
            isinstance(implementations, list) and implementations,
            f"{row.get('id')} implementation list is empty",
        )
        for index, value in enumerate(implementations):
            path = _relative_path(value, field=f"{row.get('id')}.implementation[{index}]")
            _require((root / path).is_file(), f"missing implementation: {path}")
            checked_paths.append(path.as_posix())

    component_paths = manifest.get("component_configs")
    _require(isinstance(component_paths, list), "component_configs must be a list")
    _require(
        set(component_paths) == set(EXPECTED_COMPONENT_SCHEMAS),
        "component config set changed",
    )
    components: dict[str, dict[str, Any]] = {}
    for value in component_paths:
        relative = _relative_path(value, field="component_configs")
        path = root / relative
        _require(path.is_file(), f"missing component config: {relative}")
        component = _read_json(path)
        _require(
            component.get("schema") == EXPECTED_COMPONENT_SCHEMAS[value],
            f"component schema changed: {value}",
        )
        components[value] = component
        checked_paths.append(relative.as_posix())

    invariants = manifest.get("invariants")
    _require(isinstance(invariants, Mapping), "scientific invariants are missing")
    for name, expected in REQUIRED_INVARIANTS.items():
        _require(invariants.get(name) is expected, f"invariant changed: {name}")

    planner = components["configs/paper/planner_fused_v1.json"]
    _require(planner.get("plan_schema") == "C3FD_NATIVE_PLAN_V2", "Plan schema changed")
    _require(
        planner.get("fusion", {}).get("legal_mask_applied_inside_decoding") is True,
        "C3FD support is no longer inside LLM decoding",
    )
    _require(
        planner.get("fusion", {}).get("retry_filter_replacement_rerank_best_of_n") is False,
        "Planner introduced post-hoc result selection",
    )

    dlm = components["configs/paper/compact_v2_sft_v1.json"]
    representation = dlm.get("representation", {})
    _require(representation.get("formula") == "7+4N", "DLM representation changed")
    _require(representation.get("exact_N_and_composition") is True, "exact chemistry lost")

    g2 = components["configs/paper/g2_pbc_r_v1.json"]
    _require(g2.get("method") == "G2-PBC-R", "promoted G2 method changed")
    _require(
        g2.get("periodic_geometry", {}).get("image_count") == 125,
        "strict triclinic PBC contract changed",
    )
    _require(
        g2.get("relation_residual", {}).get("uncertainty_gate") is False,
        "rejected G2-B uncertainty gate became active",
    )

    inference = components["configs/paper/inference_stream17_v1.json"]
    _require(
        inference.get("retry_filter_replacement_rerank_best_of_n") is False,
        "inference result-selection policy changed",
    )
    _require(
        inference.get("dlm", {}).get("trajectories_per_plan") == 1,
        "one-Plan/one-trajectory contract changed",
    )

    return {
        "schema": MAINLINE_SCHEMA,
        "status": "valid",
        "stage_count": len(stages),
        "component_count": len(components),
        "checked_repository_paths": sorted(set(checked_paths)),
        "stage_order": list(stage_ids),
    }


def load_and_validate(
    config: Path | str | None = None, *, root: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = (root or repository_root()).resolve()
    path = root / (Path(config) if config is not None else DEFAULT_CONFIG)
    manifest = _read_json(path)
    return manifest, validate_manifest(manifest, root=root)


def stage_spec(manifest: Mapping[str, Any], stage_id: str) -> Mapping[str, Any]:
    for row in manifest.get("stages", []):
        if isinstance(row, Mapping) and row.get("id") == stage_id:
            return row
    raise ManifestError(f"unknown paper pipeline stage: {stage_id}")


def command_for_stage(manifest: Mapping[str, Any], stage_id: str) -> dict[str, Any]:
    """Return the audited command contract without executing it."""

    row = stage_spec(manifest, stage_id)
    contract = str(row["contract"])
    command = ["sbatch", contract] if contract.endswith(".sbatch") else [sys.executable, contract]
    return {
        "stage": stage_id,
        "command": command,
        "contract_environment": dict(row.get("contract_environment", {})),
        "implementation": list(row["implementation"]),
        "execution": "not_started_by_this_read_only_command",
    }
