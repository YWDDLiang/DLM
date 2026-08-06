"""Fail-closed contract for MatterSim guidance and CHGNet evaluation.

This module intentionally uses only the Python standard library.  It can be
imported on a login node before torch, MatterSim, or CHGNet is available.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "wqcodiff_mattersim_guidance_chgnet_evaluation_v1"
MATTERSIM_CHECKPOINT_SHA256 = (
    "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"
)
CHGNET_CHECKPOINT_SHA256 = (
    "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1"
)


class GuidanceContractError(ValueError):
    """Raised when the guide/evaluator firewall is not fail-closed."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GuidanceContractError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise GuidanceContractError(f"{name} must be a sequence")
    return value


def _require(value: bool, message: str) -> None:
    if not value:
        raise GuidanceContractError(message)


def _require_sha256(value: Any, name: str) -> str:
    text = str(value)
    _require(
        len(text) == 64 and all(character in "0123456789abcdef" for character in text),
        f"{name} must be a lowercase SHA256",
    )
    return text


def _require_sorted_unique_numbers(
    value: Any,
    name: str,
    *,
    allow_zero: bool,
) -> tuple[float, ...]:
    items = tuple(float(item) for item in _sequence(value, name))
    _require(bool(items), f"{name} must not be empty")
    _require(
        list(items) == sorted(set(items)),
        f"{name} must be sorted and unique",
    )
    if allow_zero:
        _require(all(item >= 0 for item in items), f"{name} must be non-negative")
    else:
        _require(all(item > 0 for item in items), f"{name} must be positive")
    return items


def canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Return the SHA256 of canonical compact JSON for *payload*."""

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA256 for *path*."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_guidance_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a MatterSim-guide / CHGNet-evaluator contract.

    The return value is a compact, serializable acceptance summary.  Any
    ambiguity or role leakage raises :class:`GuidanceContractError`.
    """

    root = _mapping(payload, "payload")
    _require(root.get("schema") == SCHEMA, "invalid guidance-contract schema")
    _require(
        root.get("status") == "local_preparation_only_no_remote_submission",
        "contract must not itself authorize remote submission",
    )

    revision = _mapping(root.get("protocol_revision"), "protocol_revision")
    _require(revision.get("kind") == "addendum", "protocol change must be an addendum")
    _require(
        revision.get("scope") == "future_MLIP_role_assignment_only",
        "protocol addendum scope is too broad",
    )
    _require(
        revision.get("does_not_modify_historical_results") is True,
        "historical results must remain immutable",
    )

    roles = _mapping(root.get("roles"), "roles")
    guide = _mapping(roles.get("guidance"), "roles.guidance")
    evaluator = _mapping(roles.get("headline_evaluator"), "roles.headline_evaluator")
    _require(guide.get("model") == "mattersim", "guidance model must be MatterSim")
    _require(
        evaluator.get("model") == "chgnet",
        "headline evaluator must be CHGNet",
    )
    _require(guide.get("model") != evaluator.get("model"), "guide/evaluator must differ")
    _require(guide.get("package_version") == "1.1.2", "MatterSim version changed")
    _require(
        guide.get("checkpoint") == "MatterSim-v1.0.0-5M.pth",
        "MatterSim checkpoint changed",
    )
    _require(
        _require_sha256(
            guide.get("checkpoint_sha256"),
            "MatterSim checkpoint_sha256",
        )
        == MATTERSIM_CHECKPOINT_SHA256,
        "MatterSim checkpoint SHA256 changed",
    )
    _require(evaluator.get("package_version") == "0.4.2", "CHGNet version changed")
    _require(evaluator.get("model_semantics") == "0.3.0", "CHGNet model changed")
    _require(
        _require_sha256(
            evaluator.get("checkpoint_sha256"),
            "CHGNet checkpoint_sha256",
        )
        == CHGNET_CHECKPOINT_SHA256,
        "CHGNet checkpoint SHA256 changed",
    )
    _require(
        evaluator.get("runtime")
        == "/public/home/jiaosz/miniconda3/envs/diff_meets_diff",
        "CHGNet must use the diff_meets_diff environment",
    )
    _require(
        evaluator.get("protocol") == "exact_R5C_A100_protocol_on_A800",
        "CHGNet evaluation protocol changed",
    )

    guide_allowed = set(_sequence(guide.get("allowed_uses"), "guidance.allowed_uses"))
    guide_forbidden = set(
        _sequence(guide.get("forbidden_uses"), "guidance.forbidden_uses")
    )
    _require(
        "fixed_composition_low_noise_position_corrector" in guide_allowed,
        "MatterSim allowed use must be fixed-composition low-noise correction",
    )
    _require(
        "headline_SUN_evaluation" in guide_forbidden,
        "MatterSim headline evaluation must be forbidden",
    )
    required_guide_forbidden = {
        "training_loss",
        "distillation_teacher",
        "training_label_generation",
        "training_checkpoint_selection",
    }
    _require(
        required_guide_forbidden.issubset(guide_forbidden),
        "MatterSim training/distillation firewall is incomplete",
    )
    _require(not guide_allowed.intersection(guide_forbidden), "guide use lists overlap")

    evaluator_allowed = set(
        _sequence(evaluator.get("allowed_uses"), "headline_evaluator.allowed_uses")
    )
    evaluator_forbidden = set(
        _sequence(evaluator.get("forbidden_uses"), "headline_evaluator.forbidden_uses")
    )
    required_evaluator_forbidden = {
        "training_loss",
        "guidance",
        "schedule_selection",
        "hyperparameter_selection",
        "checkpoint_selection",
        "retry_decision",
    }
    _require(
        required_evaluator_forbidden.issubset(evaluator_forbidden),
        "CHGNet training/tuning/guidance firewall is incomplete",
    )
    _require(
        "frozen_heldout_SUN_evaluation" in evaluator_allowed,
        "CHGNet must be restricted to frozen held-out evaluation",
    )
    _require(
        not evaluator_allowed.intersection(evaluator_forbidden),
        "evaluator use lists overlap",
    )

    guidance = _mapping(root.get("guidance_contract"), "guidance_contract")
    _require(
        guidance.get("composition_scope") == "fixed_per_attempt",
        "MatterSim guidance must keep composition fixed",
    )
    _require(
        guidance.get("state_scope") == "predicted_clean_x0_at_low_noise",
        "MatterSim may only see predicted clean low-noise states",
    )
    _require(
        guidance.get("coordinate_scope") == "cartesian_positions_only",
        "v1 guidance must be position-only",
    )
    _require(
        guidance.get("energy_comparison_scope")
        == "within_same_composition_trajectory_only",
        "MatterSim energies may only be compared within one composition trajectory",
    )
    _require(
        guidance.get("cross_composition_energy_ranking") is False,
        "cross-composition MatterSim ranking is forbidden",
    )
    _require(
        guidance.get("lattice_guidance_enabled") is False,
        "lattice guidance requires a later audited contract",
    )
    _require(
        guidance.get(
            "lattice_guidance_requires_separate_stress_finite_difference_audit"
        )
        is True,
        "stress finite-difference audit must guard lattice guidance",
    )
    _require(
        guidance.get("force_sign_and_unit_audit_required") is True,
        "force sign/unit audit is required",
    )
    calls = _require_sorted_unique_numbers(
        guidance.get("corrector_calls_grid"),
        "corrector_calls_grid",
        allow_zero=True,
    )
    _require(0.0 in calls, "corrector_calls_grid must include unguided K=0")
    _require_sorted_unique_numbers(
        guidance.get("max_cartesian_displacement_angstrom_grid"),
        "max_cartesian_displacement_angstrom_grid",
        allow_zero=False,
    )
    _require_sorted_unique_numbers(
        guidance.get("low_noise_start_timestep_grid"),
        "low_noise_start_timestep_grid",
        allow_zero=False,
    )
    _require(
        guidance.get("fixed_calls_per_attempt") is True,
        "guidance calls must be fixed per attempt",
    )
    for key in (
        "line_search_enabled",
        "retry_enabled",
        "replacement_enabled",
        "best_of_n_enabled",
    ):
        _require(guidance.get(key) is False, f"{key} must be false")
    _require(
        guidance.get("failure_policy")
        == "retain_attempt_as_failure_without_unguided_fallback",
        "failure policy cannot hide MatterSim failures",
    )
    development_sources = {
        str(value).lower()
        for value in _sequence(
            guidance.get("development_selection_sources"),
            "development_selection_sources",
        )
    }
    _require(
        not any("chgnet" in value for value in development_sources),
        "CHGNet cannot be used to tune MatterSim guidance",
    )

    firewall = _mapping(root.get("evaluation_firewall"), "evaluation_firewall")
    required_true = (
        "guidance_configuration_frozen_before_chgnet",
    )
    required_false = (
        "chgnet_visible_during_guidance_development",
        "chgnet_used_for_tuning",
        "mattersim_used_as_headline_evaluator",
        "heldout_panel_reusable_after_tuning",
        "coverage_adjusted_used_for_selection",
        "new_dft_allowed",
    )
    for key in required_true:
        _require(firewall.get(key) is True, f"{key} must be true")
    for key in required_false:
        _require(firewall.get(key) is False, f"{key} must be false")
    _require(
        firewall.get("main_denominator") == "all_registered_attempts",
        "main denominator must include every registered attempt",
    )
    thresholds = tuple(float(value) for value in firewall.get("sun_thresholds_ev_per_atom", ()))
    _require(thresholds == (0.0, 0.1), "S.U.N. thresholds must remain 0.0/0.1")
    _require(
        firewall.get("mp_reference_prefetch")
        == "login_node_no_slurm_then_frozen_read_only_cache",
        "MP reference prefetch contract changed",
    )

    training = _mapping(
        root.get("new_diffusion_training_policy"),
        "new_diffusion_training_policy",
    )
    _require(
        training.get("long_training_authorized_by_this_file") is False,
        "preparation contract cannot authorize long training",
    )
    _require(
        training.get("training_mode") == "MLIP_free",
        "new diffusion training must remain MLIP-free",
    )
    _require(
        training.get("mattersim_allowed_in_training_or_distillation") is False,
        "MatterSim cannot enter training or distillation",
    )
    _require(
        training.get("chgnet_allowed_in_training_or_selection") is False,
        "CHGNet cannot enter training or selection",
    )
    training_sources = {
        str(value).lower()
        for value in _sequence(
            training.get("allowed_training_signal_sources"),
            "allowed_training_signal_sources",
        )
    }
    forbidden_source_tokens = ("mattersim", "chgnet", "mace", "mlip")
    _require(
        not any(
            token in source
            for source in training_sources
            for token in forbidden_source_tokens
        ),
        "MLIP-derived training signals are forbidden",
    )

    resources = _mapping(root.get("resource_contract"), "resource_contract")
    max_cpus = int(resources.get("max_cpus_per_a800", -1))
    _require(max_cpus == 8, "resource contract must cap CPUs at 8 per A800")
    _require(
        resources.get("claim_must_be_created_after_resource_preflight") is True,
        "resource preflight must precede the submission claim",
    )
    profiles = _sequence(resources.get("job_profiles"), "resource_contract.job_profiles")
    _require(bool(profiles), "at least one job profile is required")
    for index, raw_profile in enumerate(profiles):
        profile = _mapping(raw_profile, f"job_profiles[{index}]")
        gpus = int(profile.get("a800", 0))
        cpus = int(profile.get("cpus", 0))
        _require(gpus > 0, f"job_profiles[{index}] must request an A800")
        _require(cpus > 0, f"job_profiles[{index}] must request CPUs")
        _require(
            cpus <= max_cpus * gpus,
            f"job_profiles[{index}] exceeds 8 CPUs per A800",
        )

    return {
        "schema": "wqcodiff_guidance_contract_acceptance_v1",
        "ok": True,
        "contract_schema": SCHEMA,
        "run_id": str(root.get("run_id")),
        "guide": "mattersim",
        "headline_evaluator": "chgnet",
        "guide_checkpoint_sha256": MATTERSIM_CHECKPOINT_SHA256,
        "evaluator_checkpoint_sha256": CHGNET_CHECKPOINT_SHA256,
        "main_denominator": "all_registered_attempts",
        "max_cpus_per_a800": max_cpus,
        "mlip_free_training": True,
        "long_training_authorized": False,
        "canonical_payload_sha256": canonical_payload_sha256(root),
    }


def load_and_validate_guidance_contract(path: str | Path) -> dict[str, Any]:
    """Load JSON from *path*, validate it, and include the byte-level SHA."""

    location = Path(path).resolve()
    payload = json.loads(location.read_text(encoding="utf-8"))
    result = validate_guidance_contract(payload)
    result["path"] = str(location)
    result["file_sha256"] = file_sha256(location)
    return result


__all__ = [
    "CHGNET_CHECKPOINT_SHA256",
    "GuidanceContractError",
    "MATTERSIM_CHECKPOINT_SHA256",
    "SCHEMA",
    "canonical_payload_sha256",
    "file_sha256",
    "load_and_validate_guidance_contract",
    "validate_guidance_contract",
]
