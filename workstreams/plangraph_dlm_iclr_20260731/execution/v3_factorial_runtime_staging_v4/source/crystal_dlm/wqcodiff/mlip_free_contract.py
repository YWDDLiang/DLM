"""Fail-closed contract for the MLIP-free ICLR WQ experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "wqcodiff_iclr_mlip_free_experiment_v1"
CHGNET_CHECKPOINT_SHA256 = (
    "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1"
)
FORBIDDEN_TRAINING_TOKENS = ("mattersim", "chgnet", "mace", "mlip", "sun")


class MLIPFreeContractError(ValueError):
    """Raised when the MLIP-free experiment boundary is violated."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MLIPFreeContractError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MLIPFreeContractError(f"{name} must be a sequence")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MLIPFreeContractError(message)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_mlip_free_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, "payload")
    _require(root.get("schema") == SCHEMA, "invalid MLIP-free contract schema")
    _require(
        root.get("status") == "local_preparation_only_no_remote_submission",
        "contract must not authorize remote execution",
    )

    supersession = _mapping(root.get("supersession"), "supersession")
    _require(
        supersession.get("cancelled_before_scientific_attempts") is True,
        "cancelled guidance branch must have zero scientific attempts",
    )
    _require(
        int(supersession.get("cancelled_scientific_attempt_count", -1)) == 0,
        "cancelled guidance scientific attempt count must be zero",
    )
    _require(
        int(supersession.get("cancelled_remote_submission_count", -1)) == 0,
        "cancelled guidance remote submission count must be zero",
    )
    _require(
        supersession.get("preserve_cancelled_files_as_decision_history") is True,
        "cancelled decision history must remain immutable",
    )

    training = _mapping(root.get("training_contract"), "training_contract")
    _require(training.get("mlip_free") is True, "training must be MLIP-free")
    sources = {
        str(value).lower()
        for value in _sequence(
            training.get("allowed_signal_sources"),
            "training_contract.allowed_signal_sources",
        )
    }
    for source in sources:
        _require(
            not any(token in source for token in FORBIDDEN_TRAINING_TOKENS),
            f"forbidden training signal source: {source}",
        )
    forbidden_sources = {
        str(value).lower()
        for value in _sequence(
            training.get("forbidden_signal_sources"),
            "training_contract.forbidden_signal_sources",
        )
    }
    for required in ("mattersim", "chgnet", "mace"):
        _require(
            any(required in source for source in forbidden_sources),
            f"{required} must be explicitly forbidden in training",
        )
    _require(
        training.get("long_training_authorized_by_this_contract") is False,
        "local contract cannot authorize long training",
    )

    sampling = _mapping(root.get("sampling_contract"), "sampling_contract")
    _require(sampling.get("mlip_free") is True, "sampling must be MLIP-free")
    for key in (
        "energy_guidance_enabled",
        "property_guidance_enabled",
        "reranking_enabled",
        "retry_enabled",
        "replacement_enabled",
        "best_of_n_enabled",
    ):
        _require(sampling.get(key) is False, f"{key} must be false")
    _require(
        sampling.get("one_output_per_registered_attempt") is True,
        "each attempt must have one output",
    )
    _require(
        sampling.get("all_registered_attempts_in_denominator") is True,
        "all registered attempts must remain in the denominator",
    )

    projection = _mapping(
        root.get("composition_projection"),
        "composition_projection",
    )
    _require(
        projection.get("applicable_source_reason") == "charge_neutrality_fail",
        "projector may only act on charge-neutrality failures",
    )
    identity_reasons = set(
        _sequence(
            projection.get("identity_source_reasons"),
            "composition_projection.identity_source_reasons",
        )
    )
    _require(
        {
            "charge_neutral_pauling_valid",
            "pauling_fail_or_ratio_rejected",
            "single_element_shortcut",
            "all_metal_shortcut",
        }.issubset(identity_reasons),
        "identity reasons do not protect valid/Pauling/shortcut cases",
    )
    required_preserved = {
        "space_group",
        "lattice_system",
        "lattice_chart",
        "orbit_ids",
        "Wyckoff_types",
        "orbit_multiplicities",
        "primitive_multiplicities",
        "chart_dimensions",
        "free_coordinates",
        "atom_count",
        "exact_original_element_set",
    }
    _require(
        required_preserved.issubset(
            set(_sequence(projection.get("preserve"), "composition_projection.preserve"))
        ),
        "fixed-topology preservation contract is incomplete",
    )
    _require(
        projection.get("allowed_change") == "whole_orbit_species_reassignment_only",
        "projector may only reassign complete orbits",
    )
    _require(
        list(projection.get("objective_order") or [])
        == [
            "changed_orbit_count",
            "affected_primitive_atom_count",
            "raw_composition_count_L1",
            "canonical_assignment_SHA256",
        ],
        "projection objective order changed",
    )
    _require(
        1 <= int(projection.get("max_changed_orbits", 0)) <= 20,
        "invalid max_changed_orbits",
    )
    _require(
        1 <= int(projection.get("max_candidate_assignments", 0)) <= 1_000_000,
        "invalid max_candidate_assignments",
    )
    _require(
        projection.get("no_solution_policy") == "retain_original_invalid_attempt",
        "no-solution attempts cannot be replaced",
    )
    _require(
        projection.get("budget_exhaustion_policy")
        == "retain_original_invalid_attempt_and_fail_closed",
        "search-budget exhaustion must fail closed",
    )

    panel = _mapping(
        root.get("composition_mechanism_panel"),
        "composition_mechanism_panel",
    )
    no_neutral = int(panel.get("no_charge_neutral_assignment", -1))
    pauling = int(panel.get("pauling_only", -1))
    controls = int(panel.get("matched_valid_controls", -1))
    _require(
        no_neutral + pauling + controls == int(panel.get("attempts", -1)) == 64,
        "composition mechanism panel must be 36+12+16=64",
    )
    gate = _mapping(panel.get("gate"), "composition_mechanism_panel.gate")
    _require(
        int(gate.get("minimum_no_neutral_recovered", -1)) == 24,
        "composition mechanism recovery gate changed",
    )
    _require(
        int(gate.get("valid_controls_byte_identical", -1)) == 16,
        "valid controls must all remain identical",
    )
    _require(
        int(gate.get("pauling_only_byte_identical", -1)) == 12,
        "Pauling-only cases must all remain identical",
    )

    evaluator = _mapping(root.get("evaluation_contract"), "evaluation_contract")
    _require(evaluator.get("model") == "chgnet", "held-out evaluator must be CHGNet")
    _require(evaluator.get("package_version") == "0.4.2", "CHGNet package changed")
    _require(evaluator.get("model_semantics") == "0.3.0", "CHGNet model changed")
    _require(
        evaluator.get("checkpoint_sha256") == CHGNET_CHECKPOINT_SHA256,
        "CHGNet checkpoint SHA changed",
    )
    _require(
        evaluator.get("runtime")
        == "/public/home/jiaosz/miniconda3/envs/diff_meets_diff",
        "CHGNet must use diff_meets_diff",
    )
    _require(
        evaluator.get("protocol") == "exact_R5C_A100_protocol_on_A800",
        "CHGNet evaluation protocol changed",
    )
    _require(
        evaluator.get("visible_only_after_method_and_checkpoint_freeze") is True,
        "held-out CHGNet must remain hidden before method freeze",
    )
    evaluator_forbidden = {
        str(value)
        for value in _sequence(
            evaluator.get("forbidden_uses"),
            "evaluation_contract.forbidden_uses",
        )
    }
    _require(
        {
            "training",
            "guidance",
            "reranking",
            "schedule_tuning",
            "hyperparameter_tuning",
            "checkpoint_selection",
            "retry_or_replacement_decision",
        }.issubset(evaluator_forbidden),
        "CHGNet evaluator firewall is incomplete",
    )
    _require(
        evaluator.get("main_denominator") == "all_registered_attempts",
        "CHGNet main denominator changed",
    )
    _require(
        tuple(float(value) for value in evaluator.get("sun_thresholds_ev_per_atom", ()))
        == (0.0, 0.1),
        "S.U.N. thresholds changed",
    )
    _require(
        evaluator.get("coverage_adjusted_used_for_selection") is False,
        "coverage-adjusted results cannot select methods",
    )
    _require(evaluator.get("new_dft_allowed") is False, "new DFT is forbidden")

    resources = _mapping(root.get("resource_contract"), "resource_contract")
    _require(
        int(resources.get("max_cpus_per_a800", -1)) == 8,
        "resource contract must cap CPUs at 8 per A800",
    )
    _require(
        resources.get("resource_preflight_before_claim") is True,
        "resource preflight must precede claim creation",
    )
    _require(
        resources.get("existing_output_or_claim_is_fatal") is True,
        "existing identities must fail closed",
    )

    authorization = _mapping(root.get("authorization"), "authorization")
    _require(
        authorization.get("local_code_config_tests_and_documents") is True,
        "local preparation must be authorized",
    )
    _require(
        authorization.get("remote_submission") is False,
        "remote submission is not authorized by this contract",
    )
    _require(
        authorization.get("long_training") is False,
        "long training is not authorized by this contract",
    )

    return {
        "schema": "wqcodiff_iclr_mlip_free_contract_acceptance_v1",
        "ok": True,
        "run_id": str(root.get("run_id")),
        "training_mlip_free": True,
        "sampling_mlip_free": True,
        "heldout_evaluator": "chgnet",
        "composition_panel_attempts": 64,
        "minimum_no_neutral_recovered": 24,
        "max_cpus_per_a800": 8,
        "remote_submission_authorized": False,
        "long_training_authorized": False,
        "canonical_payload_sha256": _canonical_sha256(root),
    }


def load_and_validate_mlip_free_contract(path: str | Path) -> dict[str, Any]:
    location = Path(path).resolve()
    payload = json.loads(location.read_text(encoding="utf-8"))
    result = validate_mlip_free_contract(payload)
    result["path"] = str(location)
    result["file_sha256"] = file_sha256(location)
    return result


__all__ = [
    "CHGNET_CHECKPOINT_SHA256",
    "MLIPFreeContractError",
    "SCHEMA",
    "file_sha256",
    "load_and_validate_mlip_free_contract",
    "validate_mlip_free_contract",
]
