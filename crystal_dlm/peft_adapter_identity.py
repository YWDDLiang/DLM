"""Fail-closed identity helpers for duplicated PEFT adapters.

The protected Planner adapter is stored in FP32.  PEFT 0.16 creates a second
LoRA adapter at the base-layer dtype before loading its checkpoint, so a BF16
base can round that second copy before PEFT widens it back to FP32.  These
helpers attest the on-disk source, copy without aliasing, and reject pairwise
equality that is not also source identity.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch


PROTECTED_P0_ADAPTER_WEIGHT_SHA256 = (
    "65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a"
)
PROTECTED_P0_ADAPTER_CONFIG_SHA256 = (
    "a40299dfbef59bd74210707240d0908e8e2b219fba10ae3f24c9b6ef7cbfbfda"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_adapter_key(key: str, adapter_name: str) -> str:
    needle = f".{adapter_name}."
    if key.count(needle) != 1:
        raise RuntimeError(
            f"adapter state key {key!r} does not contain exactly one {needle!r}"
        )
    return key.replace(needle, ".")


def adapter_state_tensors(
    model: torch.nn.Module,
    adapter_name: str,
    *,
    keep_vars: bool,
) -> dict[str, torch.Tensor]:
    needle = f".{adapter_name}."
    tensors: dict[str, torch.Tensor] = {}
    for key, tensor in model.state_dict(keep_vars=keep_vars).items():
        if needle not in key:
            continue
        canonical = _canonical_adapter_key(key, adapter_name)
        if canonical in tensors:
            raise RuntimeError(f"canonical adapter key collision: {canonical}")
        tensors[canonical] = tensor
    if not tensors:
        raise RuntimeError(f"adapter {adapter_name!r} has no state tensors")
    return tensors


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    return cpu.view(torch.uint8).numpy().tobytes()


def _update_tensor_digest(
    digest: "hashlib._Hash",
    key: str,
    tensor: torch.Tensor,
) -> None:
    digest.update(key.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(_tensor_bytes(tensor))


def _tensor_is_finite(tensor: torch.Tensor) -> bool:
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return True
    return bool(torch.isfinite(tensor.detach()).all().item())


def _storage_interval(tensor: torch.Tensor) -> tuple[str, int, int]:
    if not tensor.is_contiguous():
        raise RuntimeError("adapter tensors must be contiguous for storage audit")
    start = int(tensor.data_ptr())
    end = start + int(tensor.numel()) * int(tensor.element_size())
    return str(tensor.device), start, end


def _intervals_overlap(
    left: tuple[str, int, int],
    right: tuple[str, int, int],
) -> bool:
    if left[0] != right[0]:
        return False
    return max(left[1], right[1]) < min(left[2], right[2])


def adapter_activation_report(
    model: torch.nn.Module,
    *,
    expected_adapter: str,
) -> dict[str, Any]:
    from peft.tuners.tuners_utils import BaseTunerLayer

    layers = []
    failures = []
    for name, module in model.named_modules():
        if not isinstance(module, BaseTunerLayer):
            continue
        active = list(module.active_adapters)
        merged = list(getattr(module, "merged_adapters", []))
        item = {"module": name, "active_adapters": active, "merged_adapters": merged}
        layers.append(item)
        if active != [expected_adapter] or merged:
            failures.append(item)
    return {
        "expected_adapter": expected_adapter,
        "tuner_layer_count": len(layers),
        "failure_count": len(failures),
        "failures": failures[:32],
        "passed": bool(layers) and not failures,
    }


def adapter_pair_identity_report(
    model: torch.nn.Module,
    *,
    candidate_name: str = "candidate",
    reference_name: str = "reference",
    expected_dtype: torch.dtype = torch.float32,
    expected_active_adapter: str | None = "candidate",
) -> dict[str, Any]:
    candidate = adapter_state_tensors(model, candidate_name, keep_vars=True)
    reference = adapter_state_tensors(model, reference_name, keep_vars=True)
    missing_candidate = sorted(set(reference) - set(candidate))
    missing_reference = sorted(set(candidate) - set(reference))
    common = sorted(set(candidate) & set(reference))

    candidate_digest = hashlib.sha256()
    reference_digest = hashlib.sha256()
    mismatched: list[dict[str, Any]] = []
    nonfinite: list[str] = []
    dtype_mismatch: list[dict[str, str]] = []
    shape_mismatch: list[str] = []
    storage_overlap: list[str] = []
    maximum = 0.0
    candidate_dtypes: Counter[str] = Counter()
    reference_dtypes: Counter[str] = Counter()

    for key in common:
        left = candidate[key]
        right = reference[key]
        candidate_dtypes[str(left.dtype)] += 1
        reference_dtypes[str(right.dtype)] += 1
        _update_tensor_digest(candidate_digest, key, left)
        _update_tensor_digest(reference_digest, key, right)
        if left.shape != right.shape:
            shape_mismatch.append(key)
            continue
        if left.dtype != expected_dtype or right.dtype != expected_dtype:
            dtype_mismatch.append(
                {"key": key, "candidate": str(left.dtype), "reference": str(right.dtype)}
            )
        left_finite = _tensor_is_finite(left)
        right_finite = _tensor_is_finite(right)
        if not left_finite or not right_finite:
            nonfinite.append(key)
        if _intervals_overlap(_storage_interval(left), _storage_interval(right)):
            storage_overlap.append(key)
        equal = bool(torch.equal(left.detach(), right.detach()))
        difference: float | None = None
        if left_finite and right_finite and left.shape == right.shape:
            difference = float((left.detach().float() - right.detach().float()).abs().max().item())
            maximum = max(maximum, difference)
        if not equal:
            mismatched.append({"key": key, "max_abs_diff": difference})

    candidate_parameters = sorted(
        name
        for name, _ in model.named_parameters()
        if f".{candidate_name}." in name
    )
    trainable = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    missing_candidate_trainable = sorted(set(candidate_parameters) - set(trainable))
    unexpected_trainable = sorted(set(trainable) - set(candidate_parameters))
    activation = (
        adapter_activation_report(model, expected_adapter=expected_active_adapter)
        if expected_active_adapter is not None
        else {"passed": True, "not_checked": True}
    )

    passed = all(
        [
            bool(candidate),
            not missing_candidate,
            not missing_reference,
            not shape_mismatch,
            not dtype_mismatch,
            not nonfinite,
            not mismatched,
            not storage_overlap,
            bool(candidate_parameters),
            not missing_candidate_trainable,
            not unexpected_trainable,
            bool(activation["passed"]),
        ]
    )
    return {
        "candidate_tensor_count": len(candidate),
        "reference_tensor_count": len(reference),
        "missing_candidate": missing_candidate,
        "missing_reference": missing_reference,
        "shape_mismatch_count": len(shape_mismatch),
        "shape_mismatch": shape_mismatch[:32],
        "dtype_mismatch_count": len(dtype_mismatch),
        "dtype_mismatch": dtype_mismatch[:32],
        "candidate_dtypes": dict(sorted(candidate_dtypes.items())),
        "reference_dtypes": dict(sorted(reference_dtypes.items())),
        "nonfinite_count": len(nonfinite),
        "nonfinite": nonfinite[:32],
        "max_abs_diff": maximum,
        "mismatched_count": len(mismatched),
        "mismatched": mismatched[:32],
        "storage_overlap_count": len(storage_overlap),
        "storage_overlap": storage_overlap[:32],
        "candidate_sha256": candidate_digest.hexdigest(),
        "reference_sha256": reference_digest.hexdigest(),
        "candidate_parameter_names": candidate_parameters,
        "trainable_parameter_names": trainable,
        "missing_candidate_trainable": missing_candidate_trainable,
        "noncandidate_trainable": unexpected_trainable,
        "activation": activation,
        "passed": passed,
    }


def copy_adapter_state_exact(
    model: torch.nn.Module,
    *,
    source_adapter: str = "candidate",
    target_adapter: str = "reference",
    expected_dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    source = adapter_state_tensors(model, source_adapter, keep_vars=True)
    target = adapter_state_tensors(model, target_adapter, keep_vars=True)
    missing_source = sorted(set(target) - set(source))
    missing_target = sorted(set(source) - set(target))
    if missing_source or missing_target:
        raise RuntimeError(
            f"adapter topology mismatch source_missing={missing_source} target_missing={missing_target}"
        )

    pre_copy_mismatch = 0
    pre_copy_maximum = 0.0
    with torch.no_grad():
        for key in sorted(source):
            left = source[key]
            right = target[key]
            if left.shape != right.shape:
                raise RuntimeError(f"adapter shape mismatch: {key}")
            if left.dtype != expected_dtype or right.dtype != expected_dtype:
                raise RuntimeError(
                    f"adapter dtype mismatch for {key}: {left.dtype} versus {right.dtype}"
                )
            if not _tensor_is_finite(left) or not _tensor_is_finite(right):
                raise RuntimeError(f"non-finite adapter tensor: {key}")
            if _intervals_overlap(_storage_interval(left), _storage_interval(right)):
                raise RuntimeError(f"candidate/reference storage overlap before copy: {key}")
            difference = float((left.detach().float() - right.detach().float()).abs().max().item())
            pre_copy_maximum = max(pre_copy_maximum, difference)
            if not torch.equal(left.detach(), right.detach()):
                pre_copy_mismatch += 1
            right.copy_(left)

    for name, parameter in model.named_parameters():
        if f".{target_adapter}." in name:
            parameter.requires_grad_(False)

    post_copy_mismatch = []
    post_copy_storage_overlap = []
    for key in sorted(source):
        left = source[key]
        right = target[key]
        if not torch.equal(left.detach(), right.detach()):
            post_copy_mismatch.append(key)
        if _intervals_overlap(_storage_interval(left), _storage_interval(right)):
            post_copy_storage_overlap.append(key)
    if post_copy_mismatch or post_copy_storage_overlap:
        raise RuntimeError(
            "exact adapter copy failed: "
            f"mismatch={post_copy_mismatch[:8]} overlap={post_copy_storage_overlap[:8]}"
        )
    return {
        "source_adapter": source_adapter,
        "target_adapter": target_adapter,
        "tensor_count": len(source),
        "pre_copy_mismatched_count": pre_copy_mismatch,
        "pre_copy_max_abs_diff": pre_copy_maximum,
        "post_copy_mismatched_count": 0,
        "post_copy_storage_overlap_count": 0,
        "copy_operation": "torch_tensor_copy_in_place",
        "passed": True,
    }


def adapter_source_identity_report(
    model: torch.nn.Module,
    adapter_name: str,
    adapter_dir: Path,
    *,
    expected_weight_sha256: str,
    expected_config_sha256: str,
    expected_dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    from safetensors import safe_open

    adapter_dir = Path(adapter_dir)
    weight_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    weight_sha = sha256_file(weight_path)
    config_sha = sha256_file(config_path)
    model_tensors = adapter_state_tensors(model, adapter_name, keep_vars=True)

    source_digest = hashlib.sha256()
    model_digest = hashlib.sha256()
    mismatched: list[dict[str, Any]] = []
    nonfinite: list[str] = []
    dtype_mismatch: list[dict[str, str]] = []
    shape_mismatch: list[str] = []
    source_dtypes: Counter[str] = Counter()
    model_dtypes: Counter[str] = Counter()
    maximum = 0.0

    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        source_keys = sorted(handle.keys())
        missing_source = sorted(set(model_tensors) - set(source_keys))
        missing_model = sorted(set(source_keys) - set(model_tensors))
        for key in sorted(set(source_keys) & set(model_tensors)):
            source = handle.get_tensor(key)
            actual = model_tensors[key].detach().cpu().clone()
            source_dtypes[str(source.dtype)] += 1
            model_dtypes[str(actual.dtype)] += 1
            _update_tensor_digest(source_digest, key, source)
            _update_tensor_digest(model_digest, key, actual)
            if source.shape != actual.shape:
                shape_mismatch.append(key)
                continue
            if source.dtype != expected_dtype or actual.dtype != expected_dtype:
                dtype_mismatch.append(
                    {"key": key, "source": str(source.dtype), "model": str(actual.dtype)}
                )
            source_finite = _tensor_is_finite(source)
            actual_finite = _tensor_is_finite(actual)
            if not source_finite or not actual_finite:
                nonfinite.append(key)
            difference: float | None = None
            if source_finite and actual_finite:
                difference = float((source.float() - actual.float()).abs().max().item())
                maximum = max(maximum, difference)
            if not torch.equal(source, actual):
                mismatched.append({"key": key, "max_abs_diff": difference})

    passed = all(
        [
            weight_sha == expected_weight_sha256,
            config_sha == expected_config_sha256,
            bool(model_tensors),
            not missing_source,
            not missing_model,
            not shape_mismatch,
            not dtype_mismatch,
            not nonfinite,
            not mismatched,
            source_digest.hexdigest() == model_digest.hexdigest(),
        ]
    )
    return {
        "adapter_name": adapter_name,
        "weight_path": str(weight_path),
        "weight_sha256": weight_sha,
        "expected_weight_sha256": expected_weight_sha256,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "expected_config_sha256": expected_config_sha256,
        "source_tensor_count": len(source_keys),
        "model_tensor_count": len(model_tensors),
        "missing_source": missing_source,
        "missing_model": missing_model,
        "shape_mismatch_count": len(shape_mismatch),
        "shape_mismatch": shape_mismatch[:32],
        "dtype_mismatch_count": len(dtype_mismatch),
        "dtype_mismatch": dtype_mismatch[:32],
        "source_dtypes": dict(sorted(source_dtypes.items())),
        "model_dtypes": dict(sorted(model_dtypes.items())),
        "nonfinite_count": len(nonfinite),
        "nonfinite": nonfinite[:32],
        "mismatched_count": len(mismatched),
        "mismatched": mismatched[:32],
        "max_abs_diff": maximum,
        "source_tensor_sha256": source_digest.hexdigest(),
        "model_tensor_sha256": model_digest.hexdigest(),
        "passed": passed,
    }


def adapter_value_sha256(model: torch.nn.Module, adapter_name: str) -> str:
    digest = hashlib.sha256()
    tensors = adapter_state_tensors(model, adapter_name, keep_vars=True)
    for key in sorted(tensors):
        _update_tensor_digest(digest, key, tensors[key])
    return digest.hexdigest()
