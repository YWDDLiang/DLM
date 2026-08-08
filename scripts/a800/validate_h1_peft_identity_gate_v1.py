#!/usr/bin/env python3
"""Independent fail-closed admission checks for the V6 PEFT identity repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


EXPECTED_ADAPTER_TENSORS = 448
EXPECTED_WEIGHT_SHA256 = (
    "65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a"
)
EXPECTED_CONFIG_SHA256 = (
    "a40299dfbef59bd74210707240d0908e8e2b219fba10ae3f24c9b6ef7cbfbfda"
)
EXPECTED_DTYPE_COUNTS = {"torch.float32": EXPECTED_ADAPTER_TENSORS}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _source_checks(prefix: str, report: Mapping[str, Any]) -> dict[str, bool]:
    return {
        f"{prefix}_passed": report.get("passed") is True,
        f"{prefix}_weight_sha": report.get("weight_sha256")
        == EXPECTED_WEIGHT_SHA256
        and report.get("expected_weight_sha256") == EXPECTED_WEIGHT_SHA256,
        f"{prefix}_config_sha": report.get("config_sha256")
        == EXPECTED_CONFIG_SHA256
        and report.get("expected_config_sha256") == EXPECTED_CONFIG_SHA256,
        f"{prefix}_tensor_count": report.get("source_tensor_count")
        == EXPECTED_ADAPTER_TENSORS
        and report.get("model_tensor_count") == EXPECTED_ADAPTER_TENSORS,
        f"{prefix}_topology": report.get("missing_source") == []
        and report.get("missing_model") == []
        and report.get("shape_mismatch_count") == 0,
        f"{prefix}_fp32": report.get("dtype_mismatch_count") == 0
        and report.get("source_dtypes") == EXPECTED_DTYPE_COUNTS
        and report.get("model_dtypes") == EXPECTED_DTYPE_COUNTS,
        f"{prefix}_finite": report.get("nonfinite_count") == 0,
        f"{prefix}_values": report.get("mismatched_count") == 0
        and report.get("max_abs_diff") == 0.0
        and report.get("source_tensor_sha256")
        == report.get("model_tensor_sha256"),
    }


def _pair_checks(
    prefix: str,
    report: Mapping[str, Any],
    *,
    require_exact: bool,
) -> dict[str, bool]:
    exact = (
        report.get("passed") is True
        and report.get("mismatched_count") == 0
        and report.get("max_abs_diff") == 0.0
        and report.get("candidate_sha256") == report.get("reference_sha256")
    )
    diagnostic = (
        report.get("passed") is False
        and isinstance(report.get("mismatched_count"), int)
        and report.get("mismatched_count", 0) > 0
        and isinstance(report.get("max_abs_diff"), (int, float))
        and report.get("max_abs_diff", 0.0) > 0.0
        and report.get("candidate_sha256") != report.get("reference_sha256")
    )
    activation = report.get("activation") or {}
    candidate_names = report.get("candidate_parameter_names") or []
    trainable_names = report.get("trainable_parameter_names") or []
    return {
        f"{prefix}_verdict": exact if require_exact else diagnostic,
        f"{prefix}_tensor_count": report.get("candidate_tensor_count")
        == EXPECTED_ADAPTER_TENSORS
        and report.get("reference_tensor_count") == EXPECTED_ADAPTER_TENSORS,
        f"{prefix}_topology": report.get("missing_candidate") == []
        and report.get("missing_reference") == []
        and report.get("shape_mismatch_count") == 0,
        f"{prefix}_fp32": report.get("dtype_mismatch_count") == 0
        and report.get("candidate_dtypes") == EXPECTED_DTYPE_COUNTS
        and report.get("reference_dtypes") == EXPECTED_DTYPE_COUNTS,
        f"{prefix}_finite": report.get("nonfinite_count") == 0,
        f"{prefix}_independent_storage": report.get("storage_overlap_count") == 0,
        f"{prefix}_candidate_only_trainable": len(candidate_names)
        == EXPECTED_ADAPTER_TENSORS
        and len(trainable_names) == EXPECTED_ADAPTER_TENSORS
        and report.get("missing_candidate_trainable") == []
        and report.get("noncandidate_trainable") == [],
        f"{prefix}_activation": activation.get("passed") is True,
    }


def _copy_checks(prefix: str, report: Mapping[str, Any]) -> dict[str, bool]:
    return {
        f"{prefix}_passed": report.get("passed") is True,
        f"{prefix}_tensor_count": report.get("tensor_count")
        == EXPECTED_ADAPTER_TENSORS,
        f"{prefix}_operation": report.get("copy_operation")
        == "torch_tensor_copy_in_place",
        f"{prefix}_post_exact": report.get("post_copy_mismatched_count") == 0,
        f"{prefix}_post_independent": report.get(
            "post_copy_storage_overlap_count"
        )
        == 0,
    }


def _triplet_checks(
    prefix: str,
    report: Mapping[str, Any],
    *,
    require_precopy_rounding: bool,
) -> dict[str, bool]:
    before = report.get("candidate_source_before_reference") or {}
    pre_copy = report.get("pre_copy_pair_diagnostic") or {}
    copy = report.get("copy_report") or {}
    post = report.get("post_setup_pair_identity") or {}
    candidate = report.get("candidate_source_after_setup") or {}
    reference = report.get("reference_source_after_setup") or {}
    checks = {
        f"{prefix}_schema": report.get("schema")
        == "h1_protected_p0_candidate_reference_identity_v2",
        f"{prefix}_passed": report.get("passed") is True,
    }
    checks.update(_source_checks(f"{prefix}_source_before", before))
    if require_precopy_rounding:
        checks.update(_pair_checks(f"{prefix}_precopy", pre_copy, require_exact=False))
    else:
        checks[f"{prefix}_precopy_recorded"] = bool(pre_copy)
    checks.update(_copy_checks(f"{prefix}_copy", copy))
    checks.update(_pair_checks(f"{prefix}_post", post, require_exact=True))
    checks.update(_source_checks(f"{prefix}_candidate_source", candidate))
    checks.update(_source_checks(f"{prefix}_reference_source", reference))
    source_hashes = {
        item.get("source_tensor_sha256") for item in (before, candidate, reference)
    }
    model_hashes = {
        item.get("model_tensor_sha256") for item in (before, candidate, reference)
    }
    checks[f"{prefix}_single_source_value_hash"] = (
        len(source_hashes) == 1
        and None not in source_hashes
        and source_hashes == model_hashes
        and post.get("candidate_sha256") == post.get("reference_sha256")
    )
    return checks


def validate_probe(
    report_path: Path,
    *,
    expected_source_inventory_sha256: str,
) -> dict[str, Any]:
    report = read_json(report_path)
    checks: dict[str, bool] = {
        "schema": report.get("schema")
        == "h1_peft_protected_p0_real_adapter_probe_v1",
        "reported_pass": report.get("passed") is True,
        "source_inventory": report.get("source_inventory_sha256")
        == expected_source_inventory_sha256,
        "a800": "A800" in str((report.get("runtime") or {}).get("cuda_device", "")),
        "no_forward": report.get("forward_performed") is False,
        "no_optimizer": report.get("optimizer_constructed") is False,
        "no_training": report.get("training_performed") is False,
        "no_smact4": report.get("smact4_executed") is False,
        "expected_rounding_signature": report.get("expected_rounding_signature")
        is True,
    }
    candidate_load = report.get("candidate_after_load") or {}
    checks.update(_source_checks("candidate_load", candidate_load))
    checks.update(
        _pair_checks("precopy_pair", report.get("pair_before_copy") or {}, require_exact=False)
    )
    checks.update(_copy_checks("copy", report.get("copy_report") or {}))
    for name in ("pair_after_copy_cpu", "pair_after_device", "pair_after_setup"):
        checks.update(_pair_checks(name, report.get(name) or {}, require_exact=True))
    candidate_setup = report.get("candidate_after_setup") or {}
    reference_setup = report.get("reference_after_setup") or {}
    checks.update(_source_checks("candidate_setup", candidate_setup))
    checks.update(_source_checks("reference_setup", reference_setup))
    source_hashes = {
        item.get("source_tensor_sha256")
        for item in (candidate_load, candidate_setup, reference_setup)
    }
    model_hashes = {
        item.get("model_tensor_sha256")
        for item in (candidate_load, candidate_setup, reference_setup)
    }
    checks["single_source_value_hash"] = (
        len(source_hashes) == 1
        and None not in source_hashes
        and source_hashes == model_hashes
    )
    passed = all(checks.values())
    return {
        "schema": "h1_peft_identity_probe_admission_v1",
        "mode": "probe",
        "input_path": str(report_path),
        "input_sha256": sha256_file(report_path),
        "expected_source_inventory_sha256": expected_source_inventory_sha256,
        "checks": checks,
        "failures": sorted(name for name, value in checks.items() if not value),
        "passed": passed,
    }


def validate_smoke(
    smoke_dir: Path,
    *,
    candidate: str,
    expected_source_inventory_sha256: str,
) -> dict[str, Any]:
    paths = {
        "config": smoke_dir / "train_config.json",
        "triplet": smoke_dir / "protected_p0_triplet_identity.json",
        "pair": smoke_dir / "dual_adapter_identity.json",
        "source_before": smoke_dir / "candidate_source_identity_before_reference.json",
        "precopy": smoke_dir / "dual_adapter_identity_before_copy.json",
    }
    payloads = {name: read_json(path) for name, path in paths.items()}
    config = payloads["config"]
    triplet = payloads["triplet"]
    smoke = config.get("preflight_smoke") or {}
    optimizer = config.get("optimizer_parameter_identity") or {}
    checks: dict[str, bool] = {
        "schema": config.get("schema") == "h1_chemistry_first_sft_v1",
        "candidate": config.get("candidate") == candidate,
        "cuda_bf16": str(config.get("device", "")).startswith("cuda")
        and config.get("bf16") is True,
        "source_inventory_bound": bool(expected_source_inventory_sha256),
        "triplet_file_matches_embedded": config.get("protected_p0_triplet_identity")
        == triplet,
        "pair_file_matches_embedded": config.get("dual_adapter_identity")
        == payloads["pair"],
        "source_file_matches_triplet": payloads["source_before"]
        == triplet.get("candidate_source_before_reference"),
        "precopy_file_matches_triplet": payloads["precopy"]
        == triplet.get("pre_copy_pair_diagnostic"),
        "logits_identical": smoke.get(
            "fixed_validation_record_candidate_reference_logits_identical"
        )
        is True,
        "logits_finite": smoke.get("fixed_validation_record_logits_finite") is True,
        "loss_finite": isinstance(smoke.get("loss"), (int, float))
        and math.isfinite(float(smoke["loss"])),
        "candidate_gradients": isinstance(smoke.get("gradient_tensor_count"), int)
        and smoke.get("gradient_tensor_count", 0) > 0
        and smoke.get("finite_gradients") is True,
        "no_unexpected_gradients": smoke.get("unexpected_gradient_names") == [],
        "reference_unchanged": bool(smoke.get("reference_sha_before"))
        and smoke.get("reference_sha_before")
        == smoke.get("reference_sha_after_backward"),
        "candidate_unchanged": bool(smoke.get("candidate_sha_before"))
        and smoke.get("candidate_sha_before")
        == smoke.get("candidate_sha_after_backward"),
        "no_optimizer_step": smoke.get("optimizer_step_performed") is False,
        "optimizer_candidate_only": optimizer.get("passed") is True
        and optimizer.get("candidate_parameter_count") == EXPECTED_ADAPTER_TENSORS
        and optimizer.get("trainable_parameter_count") == EXPECTED_ADAPTER_TENSORS
        and optimizer.get("optimizer_parameter_count") == EXPECTED_ADAPTER_TENSORS
        and optimizer.get("optimizer_unique_parameter_count")
        == EXPECTED_ADAPTER_TENSORS
        and optimizer.get("missing_candidate_in_optimizer") == []
        and optimizer.get("unexpected_optimizer_parameters") == []
        and optimizer.get("reference_in_optimizer") == [],
    }
    checks.update(
        _triplet_checks("triplet", triplet, require_precopy_rounding=False)
    )
    passed = all(checks.values())
    return {
        "schema": "h1_peft_identity_smoke_admission_v1",
        "mode": "smoke",
        "candidate": candidate,
        "smoke_dir": str(smoke_dir),
        "input_sha256": {
            name: sha256_file(path) for name, path in sorted(paths.items())
        },
        "expected_source_inventory_sha256": expected_source_inventory_sha256,
        "checks": checks,
        "failures": sorted(name for name, value in checks.items() if not value),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--report", type=Path, required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--smoke-dir", type=Path, required=True)
    smoke.add_argument("--candidate", choices=("sft_v2", "sft_v2_c"), required=True)
    for child in (probe, smoke):
        child.add_argument("--expected-source-inventory-sha256", required=True)
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == "probe":
        result = validate_probe(
            args.report,
            expected_source_inventory_sha256=args.expected_source_inventory_sha256,
        )
    else:
        result = validate_smoke(
            args.smoke_dir,
            candidate=args.candidate,
            expected_source_inventory_sha256=args.expected_source_inventory_sha256,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
