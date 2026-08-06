"""Pure runtime checks for H1-A2 two-factor generation.

The model-facing scripts import this module to keep the scientific identities
independent of rank, batching, and filesystem enumeration.  It deliberately
contains no model loading, optimizer, retry, replacement, or selection logic.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from crystal_dlm.h1a2_factorial_contract import (
    FACTORIAL_ARM_COMPONENTS,
    FACTORIAL_ARMS,
    MODEL_SAMPLED_PLAN_PROVENANCE,
    PLANNER_ARMS,
    ordered_factorial_attempts,
)
from crystal_dlm.ordinal_rng import ordered_ordinal_records, sha256_text
from crystal_dlm.planned_corruption import (
    h1a2_generation_schedule,
    plan_condition_sha256,
)
from crystal_dlm.plangraph_v1 import plangraph_from_plan_state, plangraph_to_json
from crystal_dlm.r5_dynamic_length import exact_body_token_count


H1A2_FACTORIAL_RUNTIME_SCHEMA = "h1a2_factorial_runtime_v1"
ATTEMPT_STATUSES = ("complete", "failed")
BODY_ARMS = ("B0", "Bstar")
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _flat_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = list(value[0])
    if not isinstance(value, list) or any(
        isinstance(item, (list, tuple, Mapping)) for item in value
    ):
        raise ValueError("tokenizer returned non-flat input_ids")
    return [int(item) for item in value]


def tokenizer_vocab_sha256(tokenizer: Any) -> str:
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, Mapping) or not vocab:
        raise ValueError("body tokenizer vocabulary is missing or empty")
    return canonical_json_sha256(
        sorted((str(token), int(token_id)) for token, token_id in vocab.items())
    )


def assert_body_tokenizer_identity(
    tokenizer: Any,
    *,
    expected_vocab_sha256: str,
) -> dict[str, Any]:
    expected = str(expected_vocab_sha256).strip().lower()
    if _HEX_SHA256.fullmatch(expected) is None:
        raise ValueError("expected body tokenizer vocabulary SHA is malformed")
    observed = tokenizer_vocab_sha256(tokenizer)
    if observed != expected:
        raise ValueError(
            "body tokenizer vocabulary SHA mismatch: "
            f"expected={expected} observed={observed}"
        )
    return {
        "schema": H1A2_FACTORIAL_RUNTIME_SCHEMA,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "") or ""),
        "class": type(tokenizer).__name__,
        "vocab_size": int(len(tokenizer)) if hasattr(tokenizer, "__len__") else None,
        "vocab_sha256": observed,
    }


def assert_additive_body_tokenization(
    tokenizer: Any,
    *,
    prompt: str,
    answer: str,
    generated_token_ids: Sequence[int],
    expected_answer_token_count: int,
) -> dict[str, Any]:
    """Prove inference uses the same additive prompt/answer boundary as SFT."""

    prompt_text = str(prompt)
    answer_text = str(answer)
    prompt_ids = _flat_ids(
        tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    )
    answer_ids = _flat_ids(
        tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    )
    full_ids = _flat_ids(
        tokenizer(prompt_text + answer_text, add_special_tokens=False)["input_ids"]
    )
    generated_ids = [int(token_id) for token_id in generated_token_ids]
    expected_count = int(expected_answer_token_count)
    if full_ids != [*prompt_ids, *answer_ids]:
        raise ValueError("body tokenizer is non-additive at the prompt/answer boundary")
    if answer_ids != generated_ids:
        raise ValueError("decoded body text does not round-trip to generated token IDs")
    if len(answer_ids) != expected_count:
        raise ValueError(
            "body answer token count mismatch: "
            f"expected={expected_count} observed={len(answer_ids)}"
        )
    return {
        "schema": H1A2_FACTORIAL_RUNTIME_SCHEMA,
        "prompt_sha256": sha256_text(prompt_text),
        "answer_sha256": sha256_text(answer_text),
        "prompt_input_ids_sha256": canonical_json_sha256(prompt_ids),
        "answer_input_ids_sha256": canonical_json_sha256(answer_ids),
        "full_input_ids_sha256": canonical_json_sha256(full_ids),
        "prompt_token_count": len(prompt_ids),
        "answer_token_count": len(answer_ids),
        "additive_tokenization": True,
        "decoded_generated_ids_roundtrip": True,
    }


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(payload)
    return records


def ordered_planner_attempts(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_count: int,
    expected_planner_arm: str,
) -> list[Mapping[str, Any]]:
    arm = str(expected_planner_arm)
    if arm not in PLANNER_ARMS:
        raise ValueError(f"unsupported Planner arm {arm!r}")
    ordered = ordered_ordinal_records(
        records,
        expected_count=int(expected_count),
        require_complete=True,
    )
    for expected_idx, record in enumerate(ordered):
        if int(record["sample_idx"]) != expected_idx:
            raise ValueError("Planner attempt ledger is not ordinal-complete")
        if record.get("planner_arm") != arm:
            raise ValueError(
                f"Planner attempt {expected_idx} is not from expected arm {arm}"
            )
        status = str(record.get("attempt_status") or "")
        if status not in ATTEMPT_STATUSES:
            raise ValueError(
                f"Planner attempt {expected_idx} has invalid status {status!r}"
            )
        if status == "complete":
            if record.get("plan_provenance") != MODEL_SAMPLED_PLAN_PROVENANCE:
                raise ValueError(
                    f"Planner attempt {expected_idx} is not model-sampled"
                )
            if record.get("model_proposed_plan") is not True:
                raise ValueError(
                    f"Planner attempt {expected_idx} is not marked model-proposed"
                )
        elif record.get("earliest_failure_stage") != "planner":
            raise ValueError(
                f"failed Planner attempt {expected_idx} lacks planner failure stage"
            )
    return ordered


def load_planner_attempts(
    path: Path,
    *,
    expected_count: int,
    expected_planner_arm: str,
) -> list[Mapping[str, Any]]:
    return ordered_planner_attempts(
        read_jsonl_objects(Path(path)),
        expected_count=expected_count,
        expected_planner_arm=expected_planner_arm,
    )


def ordered_single_arm_attempts(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_count: int,
    expected_factorial_arm: str,
) -> list[Mapping[str, Any]]:
    arm = str(expected_factorial_arm)
    if arm not in FACTORIAL_ARMS:
        raise ValueError(f"unsupported factorial arm {arm!r}")
    ordered = ordered_ordinal_records(
        records,
        expected_count=int(expected_count),
        require_complete=True,
    )
    for expected_idx, record in enumerate(ordered):
        if int(record["sample_idx"]) != expected_idx:
            raise ValueError("single-arm attempt ledger is not ordinal-complete")
        if record.get("factorial_arm") != arm:
            raise ValueError(
                f"attempt {expected_idx} is not from expected arm {arm}"
            )
        if str(record.get("attempt_status") or "") not in ATTEMPT_STATUSES:
            raise ValueError(f"attempt {expected_idx} has invalid status")
        if int(record.get("evaluation_order", -1)) != expected_idx:
            raise ValueError(f"attempt {expected_idx} changed evaluation order")
    return ordered


def compile_body_condition(
    body_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile B0/B* schedules from the persisted sampled Plan only."""

    arm = str(body_input.get("factorial_arm") or "")
    if arm not in FACTORIAL_ARM_COMPONENTS:
        raise ValueError(f"unsupported factorial arm {arm!r}")
    expected_planner, expected_body = FACTORIAL_ARM_COMPONENTS[arm]
    if body_input.get("planner_arm") != expected_planner:
        raise ValueError("factorial Planner identity mismatch")
    if body_input.get("body_arm") != expected_body:
        raise ValueError("factorial body identity mismatch")
    if body_input.get("plan_provenance") != MODEL_SAMPLED_PLAN_PROVENANCE:
        raise ValueError("body condition did not originate from a model-sampled Plan")

    plan_state = body_input.get("plan_state")
    if not isinstance(plan_state, Mapping):
        raise ValueError("body condition is missing plan_state")
    prompt = str(body_input.get("body_prompt") or "")
    if sha256_text(prompt) != body_input.get("body_prompt_sha256"):
        raise ValueError("body condition prompt SHA mismatch")
    expected_tokens = exact_body_token_count(plan_state)

    policy = "d1" if expected_body == "B0" else "d2"
    schedule = h1a2_generation_schedule(plan_state, policy=policy)
    payload: dict[str, Any] = {
        **dict(body_input),
        "runtime_schema": H1A2_FACTORIAL_RUNTIME_SCHEMA,
        "generation_policy": policy,
        "generation_schedule": schedule,
        "generation_schedule_sha256": canonical_json_sha256(schedule),
        "expected_answer_token_count": expected_tokens,
        "compiled_plangraph": None,
        "compiled_plangraph_sha256": None,
        "plan_condition_sha256": None,
    }
    if policy == "d2":
        graph = plangraph_from_plan_state(plan_state)
        graph_json = plangraph_to_json(graph)
        payload["compiled_plangraph"] = graph
        payload["compiled_plangraph_sha256"] = sha256_text(graph_json)
        payload["plan_condition_sha256"] = plan_condition_sha256(
            prompt=prompt,
            graph=graph,
        )
    return payload


def propagated_planner_failure(
    planner_attempt: Mapping[str, Any],
    *,
    factorial_arm: str,
    ordinal_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Carry a Planner failure into the raw factorial denominator unchanged."""

    arm = str(factorial_arm)
    if arm not in FACTORIAL_ARM_COMPONENTS:
        raise ValueError(f"unsupported factorial arm {arm!r}")
    planner_arm, body_arm = FACTORIAL_ARM_COMPONENTS[arm]
    sample_idx = int(ordinal_record["sample_idx"])
    if int(planner_attempt.get("sample_idx", -1)) != sample_idx:
        raise ValueError("Planner failure ordinal mismatch")
    if planner_attempt.get("planner_arm") != planner_arm:
        raise ValueError("Planner failure arm mismatch")
    if planner_attempt.get("attempt_status") != "failed":
        raise ValueError("only failed Planner attempts may be propagated")
    return {
        "runtime_schema": H1A2_FACTORIAL_RUNTIME_SCHEMA,
        "sample_idx": sample_idx,
        "evaluation_order": int(ordinal_record["evaluation_order"]),
        "factorial_arm": arm,
        "planner_arm": planner_arm,
        "body_arm": body_arm,
        "attempt_status": "failed",
        "earliest_failure_stage": "planner",
        "failure_reason": planner_attempt.get("failure_reason"),
        "failure_message": planner_attempt.get("failure_message"),
        "planner_sampling_seed": int(ordinal_record["planner_sampling_seed"]),
        "body_sampling_seed": int(ordinal_record["body_sampling_seed"]),
        "refiner_sampling_seed": int(ordinal_record["refiner_sampling_seed"]),
        "retry_used": False,
        "replacement_used": False,
        "repair_used": False,
        "filter_used": False,
        "rerank_used": False,
    }


def assert_factorial_pairing(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_count: int,
) -> dict[str, Any]:
    """Validate four-arm completeness and all registered pairing identities."""

    ordered = ordered_factorial_attempts(
        records,
        expected_count=int(expected_count),
    )
    by_identity = {
        (int(record["sample_idx"]), str(record["factorial_arm"])): record
        for record in ordered
    }
    plan_pairs_checked = 0
    planner_failures_paired = 0
    for sample_idx in range(int(expected_count)):
        arms = {
            arm: by_identity[(sample_idx, arm)]
            for arm in FACTORIAL_ARMS
        }
        for key in (
            "planner_sampling_seed",
            "body_sampling_seed",
            "refiner_sampling_seed",
            "evaluation_order",
        ):
            observed = {int(record[key]) for record in arms.values()}
            if len(observed) != 1:
                raise ValueError(
                    f"ordinal {sample_idx} changed paired {key}: {sorted(observed)}"
                )
        if int(arms["M00"]["evaluation_order"]) != sample_idx:
            raise ValueError(f"ordinal {sample_idx} changed evaluation order")

        for left, right in (("M00", "M01"), ("M10", "M11")):
            left_planner_failed = (
                arms[left].get("earliest_failure_stage") == "planner"
            )
            right_planner_failed = (
                arms[right].get("earliest_failure_stage") == "planner"
            )
            if left_planner_failed != right_planner_failed:
                raise ValueError(
                    f"ordinal {sample_idx} {left}/{right} Planner failure mismatch"
                )
            if left_planner_failed:
                planner_failures_paired += 1
                continue
            for key in (
                "raw_plan_text_sha256",
                "plan_text_sha256",
                "body_prompt_sha256",
            ):
                left_value = arms[left].get(key)
                right_value = arms[right].get(key)
                if not left_value or left_value != right_value:
                    raise ValueError(
                        f"ordinal {sample_idx} {left}/{right} pairing mismatch "
                        f"for {key}"
                    )
            plan_pairs_checked += 1
    return {
        "schema": H1A2_FACTORIAL_RUNTIME_SCHEMA,
        "expected_count_per_arm": int(expected_count),
        "total_attempts": len(ordered),
        "plan_pairs_checked": plan_pairs_checked,
        "planner_failure_pairs": planner_failures_paired,
        "paired_planner_seed": True,
        "paired_body_seed": True,
        "paired_refiner_seed": True,
        "paired_evaluation_order": True,
        "within_planner_plan_prompt_identity": True,
    }


__all__ = [
    "ATTEMPT_STATUSES",
    "BODY_ARMS",
    "H1A2_FACTORIAL_RUNTIME_SCHEMA",
    "assert_additive_body_tokenization",
    "assert_body_tokenizer_identity",
    "assert_factorial_pairing",
    "canonical_json_sha256",
    "compile_body_condition",
    "load_planner_attempts",
    "ordered_planner_attempts",
    "ordered_single_arm_attempts",
    "propagated_planner_failure",
    "read_jsonl_objects",
    "tokenizer_vocab_sha256",
]
