"""Tokenizer and CPU-mask preflight for a frozen PlanGraph body dataset."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch

from crystal_dlm.plangraph_dataset import (
    PLANGRAPH_DATASET_VERSION,
    sha256_file,
)
from scripts.llada_sft import (
    DataCollator,
    JsonlSftDataset,
    forward_process,
    normalize_planned_corruption_policy,
)


PLANNED_PREFLIGHT_VERSION = "plangraph_dlm_tokenizer_mask_preflight_v1"


class PlannedPreflightError(RuntimeError):
    """Raised when immutable input integrity or preflight invariants fail."""


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def register_data_vocab(tokenizer, vocab_file: str | Path) -> Dict[str, Any]:
    """Apply the same ordered additional-special-token operation as training."""

    path = Path(vocab_file).expanduser().resolve()
    if not path.is_file():
        raise PlannedPreflightError(f"missing data vocab: {path}")
    tokens = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not tokens:
        raise PlannedPreflightError(f"data vocab is empty: {path}")
    if len(tokens) != len(set(tokens)):
        raise PlannedPreflightError(f"data vocab contains duplicate tokens: {path}")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise PlannedPreflightError(
                "tokenizer has neither a pad token nor an eos token"
            )
        tokenizer.pad_token = tokenizer.eos_token
    vocab_before = tokenizer.get_vocab()
    missing_before = [token for token in tokens if token not in vocab_before]
    added = int(
        tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens}
        )
    )
    vocab_after = tokenizer.get_vocab()
    missing_after = [token for token in tokens if token not in vocab_after]
    if missing_after:
        raise PlannedPreflightError(
            f"{len(missing_after)} data-vocab tokens remain unregistered"
        )
    multi_token: list[str] = []
    for token in tokens:
        token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            multi_token.append(token)
    if multi_token:
        raise PlannedPreflightError(
            f"{len(multi_token)} registered data tokens do not encode as one token"
        )
    return {
        "vocab_file": str(path),
        "vocab_file_sha256": sha256_file(path),
        "data_token_count": len(tokens),
        "missing_before_count": len(missing_before),
        "added_token_count": added,
        "missing_after_count": len(missing_after),
        "tokenizer_vocab_size_before": len(vocab_before),
        "tokenizer_vocab_size_after": len(vocab_after),
        "tokenizer_vocab_sha256_after": _sha256_json(
            sorted((str(token), int(token_id)) for token, token_id in vocab_after.items())
        ),
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": (
            None
            if tokenizer.eos_token_id is None
            else int(tokenizer.eos_token_id)
        ),
    }


def verify_published_dataset(data_dir: str | Path) -> Dict[str, Any]:
    """Verify the builder manifest, success marker, and every published file."""

    body_dir = Path(data_dir).expanduser().resolve()
    root = body_dir.parent
    manifest_path = root / "manifest.json"
    success_path = root / "_SUCCESS"
    if body_dir.name != "body":
        raise PlannedPreflightError(
            "preflight expects the published PlanGraph dataset's body directory"
        )
    if not manifest_path.is_file() or not success_path.is_file():
        raise PlannedPreflightError(
            f"missing manifest/_SUCCESS beside body directory: {root}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        marker = json.loads(success_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise PlannedPreflightError(f"invalid published dataset metadata: {exc}") from exc
    if manifest.get("dataset_version") != PLANGRAPH_DATASET_VERSION:
        raise PlannedPreflightError("unexpected PlanGraph dataset version")
    if manifest.get("published") is not True:
        raise PlannedPreflightError("dataset manifest is not marked published")
    actual_manifest_sha256 = sha256_file(manifest_path)
    if marker.get("manifest_sha256") != actual_manifest_sha256:
        raise PlannedPreflightError("manifest SHA-256 disagrees with _SUCCESS")

    output_hashes = manifest.get("output_file_sha256")
    if not isinstance(output_hashes, dict) or not output_hashes:
        raise PlannedPreflightError("manifest has no output file hash ledger")
    verified_files = 0
    for relative, expected in sorted(output_hashes.items()):
        candidate = (root / str(relative)).resolve()
        if root != candidate and root not in candidate.parents:
            raise PlannedPreflightError(
                f"manifest output path escapes dataset root: {relative!r}"
            )
        if not candidate.is_file():
            raise PlannedPreflightError(f"published output is missing: {relative}")
        actual = sha256_file(candidate)
        if actual != expected:
            raise PlannedPreflightError(
                f"published output SHA-256 mismatch: {relative}"
            )
        verified_files += 1
    return {
        "dataset_root": str(root),
        "body_dir": str(body_dir),
        "dataset_version": manifest["dataset_version"],
        "manifest_sha256": actual_manifest_sha256,
        "verified_output_file_count": verified_files,
        "fixed_validation_panel": manifest.get("fixed_validation_panel"),
        "split_manifest": manifest.get("splits"),
    }


def _planned_mask_invariant_counts(
    *,
    batch: Dict[str, torch.Tensor],
    output: Dict[str, torch.Tensor],
    require_all_planned: bool,
) -> Dict[str, int]:
    planned_flags = output["planned_sample_mask"].bool()
    if require_all_planned and not bool(planned_flags.all()):
        raise PlannedPreflightError("planned-only smoke produced an iid sample")
    if bool(output["input_masked_indices"][:, : batch["prompt_lengths"].min()].any()):
        raise PlannedPreflightError("corruption masked a prompt token")

    counts = {
        "samples": int(batch["input_ids"].shape[0]),
        "planned_samples": int(planned_flags.sum().item()),
        "iid_samples": int((~planned_flags).sum().item()),
        "supervised_tokens": int(output["masked_indices"].sum().item()),
        "input_masked_tokens": int(output["input_masked_indices"].sum().item()),
        "future_masked_tokens": 0,
    }
    for sample_index in range(batch["input_ids"].shape[0]):
        prompt_length = int(batch["prompt_lengths"][sample_index].item())
        if bool(
            output["input_masked_indices"][
                sample_index, :prompt_length
            ].any()
        ):
            raise PlannedPreflightError("corruption masked a prompt token")
        group_ids = batch["planned_group_ids"][sample_index]
        valid_group_ids = group_ids[group_ids >= 0]
        answer_length = int(valid_group_ids.numel())
        loss_mask = output["masked_indices"][
            sample_index, prompt_length : prompt_length + answer_length
        ]
        input_mask = output["input_masked_indices"][
            sample_index, prompt_length : prompt_length + answer_length
        ]
        if not bool(loss_mask.any()):
            raise PlannedPreflightError("a smoke sample has zero supervised tokens")
        if bool((loss_mask & ~input_mask).any()):
            raise PlannedPreflightError("a supervised token is visible in the input")

        if not bool(planned_flags[sample_index]):
            if not torch.equal(loss_mask, input_mask):
                raise PlannedPreflightError(
                    "iid sample input mask differs from its loss mask"
                )
            if int(output["active_group_indices"][sample_index].item()) != -1:
                raise PlannedPreflightError("iid sample has an active planned group")
            continue

        active = int(output["active_group_indices"][sample_index].item())
        sample_groups = valid_group_ids
        prerequisite = sample_groups < active
        active_positions = sample_groups == active
        future = sample_groups > active
        if active < 0 or not bool(active_positions.any()):
            raise PlannedPreflightError("planned sample has an invalid active group")
        if bool(input_mask[prerequisite].any()):
            raise PlannedPreflightError("planned corruption hid a prerequisite token")
        if not bool(input_mask[future].all()):
            raise PlannedPreflightError("planned corruption left a future token visible")
        if bool(loss_mask[future].any()):
            raise PlannedPreflightError("planned corruption supervised a future token")
        if bool(loss_mask[~active_positions].any()):
            raise PlannedPreflightError(
                "planned corruption supervised outside the active group"
            )
        if not torch.equal(input_mask[active_positions], loss_mask[active_positions]):
            raise PlannedPreflightError(
                "active-group input and supervision masks disagree"
            )
        counts["future_masked_tokens"] += int(future.sum().item())
    return counts


def _merge_counts(
    accumulator: Dict[str, int],
    update: Mapping[str, int],
) -> None:
    for key, value in update.items():
        accumulator[key] = int(accumulator.get(key, 0)) + int(value)


def _iter_batches(
    items: Sequence[Dict[str, Any]],
    *,
    batch_size: int,
) -> Iterable[Sequence[Dict[str, Any]]]:
    for offset in range(0, len(items), int(batch_size)):
        yield items[offset : offset + int(batch_size)]


def _run_mask_smoke(
    *,
    items: Sequence[Dict[str, Any]],
    tokenizer,
    batch_size: int,
    corruption_seed: int,
) -> Dict[str, Any]:
    if not items:
        raise PlannedPreflightError("mask smoke has no valid rows")
    collator = DataCollator(tokenizer)
    planned_counts: Dict[str, int] = {}
    mixture_counts: Dict[str, int] = {}
    fixed_iid_counts: Dict[str, int] = {}
    deterministic_repeats = 0

    for batch_index, item_batch in enumerate(
        _iter_batches(items, batch_size=batch_size)
    ):
        batch = collator(list(item_batch))
        common = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "prompt_lengths": batch["prompt_lengths"],
            "mask_policy_ids": batch["mask_policy_ids"],
            "planned_group_ids": batch["planned_group_ids"],
            "planned_group_counts": batch["planned_group_counts"],
            "corruption_keys": batch["corruption_keys"],
            "corruption_step": batch_index,
            "corruption_seed": int(corruption_seed),
        }
        planned = forward_process(
            **common,
            iid_fraction=0.0,
            planned_fraction=1.0,
        )
        planned_repeat = forward_process(
            **common,
            iid_fraction=0.0,
            planned_fraction=1.0,
        )
        for key in (
            "masked_indices",
            "input_masked_indices",
            "planned_sample_mask",
            "active_group_indices",
        ):
            if not torch.equal(planned[key], planned_repeat[key]):
                raise PlannedPreflightError(
                    f"stateless planned mask is not repeatable: {key}"
                )
        deterministic_repeats += int(batch["input_ids"].shape[0])
        _merge_counts(
            planned_counts,
            _planned_mask_invariant_counts(
                batch=batch,
                output=planned,
                require_all_planned=True,
            ),
        )

        mixture = forward_process(
            **common,
            iid_fraction=2.0,
            planned_fraction=1.0,
        )
        _merge_counts(
            mixture_counts,
            _planned_mask_invariant_counts(
                batch=batch,
                output=mixture,
                require_all_planned=False,
            ),
        )

        fixed_iid = forward_process(
            **common,
            iid_fraction=1.0,
            planned_fraction=0.0,
            stateless_iid=True,
        )
        fixed_iid_repeat = forward_process(
            **common,
            iid_fraction=1.0,
            planned_fraction=0.0,
            stateless_iid=True,
        )
        if not torch.equal(
            fixed_iid["masked_indices"],
            fixed_iid_repeat["masked_indices"],
        ):
            raise PlannedPreflightError("fixed-panel iid mask is not repeatable")
        _merge_counts(
            fixed_iid_counts,
            _planned_mask_invariant_counts(
                batch=batch,
                output=fixed_iid,
                require_all_planned=False,
            ),
        )

    return {
        "device": "cpu",
        "smoke_row_count": len(items),
        "batch_size": int(batch_size),
        "corruption_seed": int(corruption_seed),
        "deterministic_repeat_rows": deterministic_repeats,
        "planned_only": planned_counts,
        "iid_to_planned_2_to_1": mixture_counts,
        "fixed_stateless_iid": fixed_iid_counts,
        "all_invariants_passed": True,
    }


def preflight_planned_data(
    *,
    data_dir: str | Path,
    tokenizer,
    splits: Sequence[str],
    max_length: int,
    policy: str,
    corruption_seed: int = 20260731,
    mask_smoke_rows: int = 32,
    mask_smoke_batch_size: int = 8,
    verify_manifest: bool = True,
) -> Dict[str, Any]:
    """Run a full-denominator tokenizer audit and bounded CPU mask smoke."""

    body_dir = Path(data_dir).expanduser().resolve()
    normalized_policy = normalize_planned_corruption_policy(policy)
    if normalized_policy not in {"d1", "d2", "d2_shuffle"}:
        raise PlannedPreflightError(
            "preflight policy must be d1, d2, or d2_shuffle"
        )
    if int(max_length) <= 0:
        raise PlannedPreflightError("max_length must be positive")
    if int(mask_smoke_rows) <= 0 or int(mask_smoke_batch_size) <= 0:
        raise PlannedPreflightError("mask smoke sizes must be positive")
    manifest_report = verify_published_dataset(body_dir) if verify_manifest else None
    vocab_report = register_data_vocab(tokenizer, body_dir / "vocab_tokens.txt")

    split_reports: Dict[str, Dict[str, Any]] = {}
    smoke_items: list[Dict[str, Any]] = []
    all_failures: list[Dict[str, Any]] = []
    total_rows = 0
    passed_rows = 0
    for split in splits:
        path = body_dir / f"{split}.jsonl"
        if not path.is_file():
            raise PlannedPreflightError(f"missing requested body split: {path}")
        dataset = JsonlSftDataset(
            path,
            tokenizer,
            int(max_length),
            planned_corruption_policy=normalized_policy,
            planned_corruption_seed=int(corruption_seed),
        )
        model_lengths: Counter[int] = Counter()
        answer_lengths: Counter[int] = Counter()
        group_counts: Counter[int] = Counter()
        failures: list[Dict[str, Any]] = []
        seen_keys: Counter[int] = Counter()
        schedule_digest = hashlib.sha256()
        for ordinal in range(len(dataset)):
            total_rows += 1
            try:
                item = dataset[ordinal]
            except Exception as exc:  # noqa: BLE001 - retain full denominator.
                failure = {
                    "split": split,
                    "ordinal": ordinal,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                if len(all_failures) < 50:
                    all_failures.append(failure)
                continue
            passed_rows += 1
            model_lengths[int(item["input_ids"].shape[0])] += 1
            answer_lengths[int(item["planned_group_ids"].shape[0])] += 1
            group_counts[int(item["planned_group_count"])] += 1
            seen_keys[int(item["corruption_key"])] += 1
            schedule_digest.update(
                (
                    _canonical_json(
                        {
                            "ordinal": ordinal,
                            "corruption_key": int(item["corruption_key"]),
                            "planned_group_count": int(
                                item["planned_group_count"]
                            ),
                            "planned_group_ids": [
                                int(value)
                                for value in item["planned_group_ids"].tolist()
                            ],
                        }
                    )
                    + "\n"
                ).encode("utf-8")
            )
            if len(smoke_items) < int(mask_smoke_rows):
                smoke_items.append(item)
        split_reports[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "total_rows": len(dataset),
            "passed_rows": len(dataset) - len(failures),
            "failed_rows": len(failures),
            "all_rows_passed": len(dataset) > 0 and not failures,
            "max_model_length": max(model_lengths, default=0),
            "max_answer_token_count": max(answer_lengths, default=0),
            "model_length_distribution": {
                str(key): int(model_lengths[key]) for key in sorted(model_lengths)
            },
            "answer_token_count_distribution": {
                str(key): int(answer_lengths[key]) for key in sorted(answer_lengths)
            },
            "planned_group_count_distribution": {
                str(key): int(group_counts[key]) for key in sorted(group_counts)
            },
            "duplicate_corruption_key_rows": sum(
                count - 1 for count in seen_keys.values() if count > 1
            ),
            "ordered_planned_schedule_sha256": schedule_digest.hexdigest(),
            "failure_examples": failures[:20],
        }

    all_rows_passed = (
        total_rows > 0
        and passed_rows == total_rows
        and all(report["total_rows"] > 0 for report in split_reports.values())
    )
    mask_smoke = None
    mask_smoke_error = None
    if all_rows_passed:
        try:
            mask_smoke = _run_mask_smoke(
                items=smoke_items,
                tokenizer=tokenizer,
                batch_size=int(mask_smoke_batch_size),
                corruption_seed=int(corruption_seed),
            )
        except Exception as exc:  # noqa: BLE001 - report the exact smoke failure.
            mask_smoke_error = f"{type(exc).__name__}: {exc}"
    gate_passed = (
        all_rows_passed
        and mask_smoke is not None
        and mask_smoke.get("all_invariants_passed") is True
    )
    return {
        "preflight_version": PLANNED_PREFLIGHT_VERSION,
        "data_dir": str(body_dir),
        "policy": normalized_policy,
        "max_length": int(max_length),
        "corruption_mix": {
            "iid_fraction": 2.0,
            "planned_fraction": 1.0,
        },
        "validation_corruption": "stateless_iid_fixed_panel_v1",
        "manifest_verification": manifest_report,
        "tokenizer": vocab_report,
        "splits": split_reports,
        "total_rows": total_rows,
        "passed_rows": passed_rows,
        "failed_rows": total_rows - passed_rows,
        "all_rows_passed": all_rows_passed,
        "failure_examples": all_failures,
        "mask_smoke": mask_smoke,
        "mask_smoke_error": mask_smoke_error,
        "preflight_gate_passed": gate_passed,
        "gpu_used": False,
        "model_loaded": False,
    }


__all__ = [
    "PLANNED_PREFLIGHT_VERSION",
    "PlannedPreflightError",
    "preflight_planned_data",
    "register_data_vocab",
    "verify_published_dataset",
]
