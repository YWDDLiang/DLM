"""Fail-closed identities for the H1-A2 Planner/body 2x2 experiment.

This module contains no model inference and no scientific selection logic.
It freezes the byte/token input identity, sampled-Plan provenance, per-ordinal
random seeds, and within-Planner body-pair identities that a future runner
must consume.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from crystal_dlm.h1_llm_planner import (
    H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
    canonical_plan_record_for_style,
    format_planner_prompt,
)
from crystal_dlm.ordinal_rng import derive_ordinal_seed, sha256_text


FACTORIAL_CONTRACT_SCHEMA = "h1a2_factorial_contract_v1"
MODEL_SAMPLED_PLAN_PROVENANCE = "model_sampled_h1a2_planner"
STRUCTURE_DERIVED_TEACHER_PROVENANCE = "structure_derived_teacher_plan_state"
PLANNER_ARMS = ("P0", "Pstar")
FACTORIAL_ARMS = ("M00", "M10", "M01", "M11")
FACTORIAL_ARM_COMPONENTS = {
    "M00": ("P0", "B0"),
    "M10": ("Pstar", "B0"),
    "M01": ("P0", "Bstar"),
    "M11": ("Pstar", "Bstar"),
}
_FACTORIAL_ARM_ORDER = {arm: index for index, arm in enumerate(FACTORIAL_ARMS)}
_STRICT_RICH_PLAN = re.compile(
    r"\A"
    r"formula: ([^\n]+)\n"
    r"anion: ([^\n]+)\n"
    r"charge: ([^\n]+)\n"
    r"lattice: ([^\n]+)\n"
    r"spacegroup: ([^\n]+)\n"
    r"volume: ([^\n]+)\n"
    r"end: plan"
    r"\Z"
)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _flatten_input_ids(value: Any) -> list[int]:
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
        raise ValueError("planner tokenizer returned non-flat input_ids")
    return [int(item) for item in value]


def build_planner_input_contract(
    tokenizer: Any,
    *,
    planner_arm: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Build the exact no-sample-id H1-A2 Planner inference input identity."""

    arm = str(planner_arm)
    if arm not in PLANNER_ARMS:
        raise ValueError(f"unsupported planner arm {arm!r}")
    checkpoint_sha = str(checkpoint_sha256).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha):
        raise ValueError("checkpoint_sha256 must be a 64-character hex digest")

    prompt = format_planner_prompt(
        tokenizer,
        sample_idx=None,
        prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
    )
    if re.search(r"(?im)^\s*sample_id\s*:", prompt):
        raise ValueError("H1-A2 Planner inference prompt must not contain sample_id")
    encoded = tokenizer(prompt, add_special_tokens=False)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise ValueError("planner tokenizer output is missing input_ids")
    token_ids = _flatten_input_ids(encoded["input_ids"])
    if not token_ids:
        raise ValueError("planner tokenizer produced an empty input")

    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    tokenizer_identity = {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "") or ""),
        "class": type(tokenizer).__name__,
        "vocab_size": int(len(tokenizer)) if hasattr(tokenizer, "__len__") else None,
        "chat_template_sha256": sha256_text(chat_template),
    }
    return {
        "schema": FACTORIAL_CONTRACT_SCHEMA,
        "planner_arm": arm,
        "checkpoint_sha256": checkpoint_sha,
        "prompt_style": H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
        "include_sample_id": False,
        "prompt_text": prompt,
        "prompt_sha256": sha256_text(prompt),
        "input_ids": token_ids,
        "input_ids_sha256": _json_sha256(token_ids),
        "tokenizer_identity": tokenizer_identity,
        "tokenizer_identity_sha256": _json_sha256(tokenizer_identity),
    }


def assert_planner_input_identity(
    p0_contract: Mapping[str, Any],
    pstar_contract: Mapping[str, Any],
) -> None:
    """Reject any P0/P* inference difference other than checkpoint identity."""

    if p0_contract.get("planner_arm") != "P0":
        raise ValueError("first Planner input contract must be P0")
    if pstar_contract.get("planner_arm") != "Pstar":
        raise ValueError("second Planner input contract must be Pstar")
    required_equal = (
        "schema",
        "prompt_style",
        "include_sample_id",
        "prompt_text",
        "prompt_sha256",
        "input_ids",
        "input_ids_sha256",
        "tokenizer_identity",
        "tokenizer_identity_sha256",
    )
    mismatches = [
        key for key in required_equal if p0_contract.get(key) != pstar_contract.get(key)
    ]
    if mismatches:
        raise ValueError(f"P0/Pstar Planner input mismatch: {mismatches}")


def persist_model_sampled_plan(
    raw_model_plan_text: str,
    *,
    planner_arm: str,
    sample_idx: int,
    planner_sampling_seed: int,
    planner_input_contract: Mapping[str, Any],
    max_atoms: int = 20,
) -> dict[str, Any]:
    """Persist one strict seven-line model sample and compile its body prompt.

    No value is filled, repaired, or substituted. Formula-derived counts and
    atom count remain non-visible compiler fields, as in frozen H1-A2.
    """

    arm = str(planner_arm)
    if arm not in PLANNER_ARMS:
        raise ValueError(f"unsupported planner arm {arm!r}")
    if planner_input_contract.get("planner_arm") != arm:
        raise ValueError("Planner arm/input-contract mismatch")
    if planner_input_contract.get("include_sample_id") is not False:
        raise ValueError("sample-id Planner input is ineligible")
    if re.search(
        r"(?im)^\s*sample_id\s*:",
        str(planner_input_contract.get("prompt_text") or ""),
    ):
        raise ValueError("Planner input contains sample_id")

    raw_text = str(raw_model_plan_text)
    match = _STRICT_RICH_PLAN.fullmatch(raw_text)
    if match is None or any(not value.strip() for value in match.groups()):
        raise ValueError("model output is not exactly the seven-line H1-A2 Plan")

    record = canonical_plan_record_for_style(
        raw_text,
        sample_idx=int(sample_idx),
        max_atoms=int(max_atoms),
        prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
    )
    generated_rich = record["plan_state"].get("generated_rich_fields")
    if not isinstance(generated_rich, Mapping) or set(generated_rich) != {
        "anion",
        "charge",
        "lattice",
        "spacegroup",
        "volume",
    }:
        raise ValueError("model output did not supply every rich Plan field")
    record.update(
        {
            "schema": FACTORIAL_CONTRACT_SCHEMA,
            "planner_arm": arm,
            "plan_provenance": MODEL_SAMPLED_PLAN_PROVENANCE,
            "model_proposed_plan": True,
            "raw_model_sampled_plan_text": raw_text,
            "planner_prompt_sha256": planner_input_contract.get("prompt_sha256"),
            "planner_input_ids_sha256": planner_input_contract.get(
                "input_ids_sha256"
            ),
            "planner_tokenizer_identity_sha256": planner_input_contract.get(
                "tokenizer_identity_sha256"
            ),
            "planner_sampling_seed": int(planner_sampling_seed),
            "retry_used": False,
            "replacement_used": False,
            "repair_used": False,
            "filter_used": False,
            "rerank_used": False,
        }
    )
    return record


def persist_parser_accepted_model_sampled_plan(
    raw_model_plan_text: str,
    canonical_plan_text: str,
    *,
    planner_arm: str,
    sample_idx: int,
    planner_sampling_seed: int,
    planner_input_contract: Mapping[str, Any],
    max_atoms: int = 20,
) -> dict[str, Any]:
    """Persist a parser-accepted sample without conflating raw and canonical bytes.

    The caller supplies the immutable raw model output and the canonical text
    already accepted by the frozen H1-A2 parser.  The canonical text alone is
    compiled into the body prompt; the raw text remains analysis evidence.
    This helper never infers, retries, replaces, or repairs a Plan value.
    """

    raw_text = str(raw_model_plan_text)
    canonical_text = str(canonical_plan_text)
    if not raw_text:
        raise ValueError("raw model-sampled Plan text is empty")
    if not canonical_text:
        raise ValueError("canonical parser-accepted Plan text is empty")

    record = persist_model_sampled_plan(
        canonical_text,
        planner_arm=planner_arm,
        sample_idx=sample_idx,
        planner_sampling_seed=planner_sampling_seed,
        planner_input_contract=planner_input_contract,
        max_atoms=max_atoms,
    )
    raw_contract_warning: str | None = None
    try:
        persist_model_sampled_plan(
            raw_text,
            planner_arm=planner_arm,
            sample_idx=sample_idx,
            planner_sampling_seed=planner_sampling_seed,
            planner_input_contract=planner_input_contract,
            max_atoms=max_atoms,
        )
    except ValueError as exc:
        raw_contract_warning = str(exc)

    record.update(
        {
            "raw_model_sampled_plan_text": raw_text,
            "raw_plan_text_sha256": sha256_text(raw_text),
            "frozen_canonical_plan_text": canonical_text,
            "raw_plan_format_gate": "advisory_nonblocking",
            "raw_plan_contract_conforming": raw_contract_warning is None,
            "raw_plan_contract_warning": raw_contract_warning,
            "canonicalization_used": raw_text != canonical_text,
        }
    )
    return record


def build_factorial_ordinal_record(
    base_seed: int,
    *,
    sample_idx: int,
) -> dict[str, Any]:
    """Freeze rank-independent Planner/body/refiner randomness for one ordinal."""

    ordinal = int(sample_idx)
    return {
        "schema": FACTORIAL_CONTRACT_SCHEMA,
        "sample_idx": ordinal,
        "planner_sampling_seed": derive_ordinal_seed(
            base_seed,
            sample_idx=ordinal,
            stage="planner_sampling",
            role="shared",
        ),
        "body_sampling_seed": derive_ordinal_seed(
            base_seed,
            sample_idx=ordinal,
            stage="body_sampling",
            role="shared",
        ),
        "refiner_sampling_seed": derive_ordinal_seed(
            base_seed,
            sample_idx=ordinal,
            stage="refiner_sampling",
            role="shared",
        ),
        "evaluation_order": ordinal,
    }


def _validate_persisted_sampled_plan(
    record: Mapping[str, Any],
    *,
    expected_arm: str,
    expected_sample_idx: int,
    expected_planner_seed: int,
) -> None:
    if record.get("planner_arm") != expected_arm:
        raise ValueError(f"expected {expected_arm} sampled Plan")
    if int(record.get("sample_idx", -1)) != int(expected_sample_idx):
        raise ValueError("sampled Plan ordinal mismatch")
    if record.get("plan_provenance") != MODEL_SAMPLED_PLAN_PROVENANCE:
        raise ValueError("body input is not a persisted model-sampled Plan")
    if record.get("model_proposed_plan") is not True:
        raise ValueError("body input is not marked model-proposed")
    if record.get("source_plan_provenance") == STRUCTURE_DERIVED_TEACHER_PROVENANCE:
        raise ValueError("structure-derived teacher Plan is ineligible at inference")
    if int(record.get("planner_sampling_seed", -1)) != int(
        expected_planner_seed
    ):
        raise ValueError("sampled Plan seed does not match ordinal ledger")

    raw_text = str(record.get("raw_model_sampled_plan_text") or "")
    if sha256_text(raw_text) != record.get("raw_plan_text_sha256"):
        raise ValueError("raw sampled Plan SHA mismatch")
    plan_text = str(record.get("plan_text") or "")
    if sha256_text(plan_text) != record.get("plan_text_sha256"):
        raise ValueError("canonical sampled Plan SHA mismatch")
    prompt = str(record.get("prompt") or "")
    if sha256_text(prompt) != record.get("body_prompt_sha256"):
        raise ValueError("compiled body prompt SHA mismatch")


def build_factorial_body_inputs(
    p0_plan: Mapping[str, Any],
    pstar_plan: Mapping[str, Any],
    *,
    ordinal_record: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Create four arm inputs while preserving exact within-Planner Plan bytes."""

    sample_idx = int(ordinal_record["sample_idx"])
    planner_seed = int(ordinal_record["planner_sampling_seed"])
    _validate_persisted_sampled_plan(
        p0_plan,
        expected_arm="P0",
        expected_sample_idx=sample_idx,
        expected_planner_seed=planner_seed,
    )
    _validate_persisted_sampled_plan(
        pstar_plan,
        expected_arm="Pstar",
        expected_sample_idx=sample_idx,
        expected_planner_seed=planner_seed,
    )

    arm_sources = {
        "M00": p0_plan,
        "M10": pstar_plan,
        "M01": p0_plan,
        "M11": pstar_plan,
    }
    result: dict[str, dict[str, Any]] = {}
    for arm in FACTORIAL_ARMS:
        result[arm] = build_factorial_arm_input(
            arm_sources[arm],
            factorial_arm=arm,
            ordinal_record=ordinal_record,
        )

    for left, right in (("M00", "M01"), ("M10", "M11")):
        for key in (
            "raw_plan_text_sha256",
            "plan_text",
            "plan_text_sha256",
            "plan_state",
            "body_prompt",
            "body_prompt_sha256",
            "body_sampling_seed",
            "refiner_sampling_seed",
            "evaluation_order",
        ):
            if result[left][key] != result[right][key]:
                raise ValueError(f"{left}/{right} pairing mismatch for {key}")
    return result


def build_factorial_arm_input(
    sampled_plan: Mapping[str, Any],
    *,
    factorial_arm: str,
    ordinal_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one body input from one persisted model-sampled Planner attempt."""

    arm = str(factorial_arm)
    if arm not in FACTORIAL_ARM_COMPONENTS:
        raise ValueError(f"unsupported factorial_arm {arm!r}")
    planner_arm, body_arm = FACTORIAL_ARM_COMPONENTS[arm]
    sample_idx = int(ordinal_record["sample_idx"])
    planner_seed = int(ordinal_record["planner_sampling_seed"])
    _validate_persisted_sampled_plan(
        sampled_plan,
        expected_arm=planner_arm,
        expected_sample_idx=sample_idx,
        expected_planner_seed=planner_seed,
    )
    return {
        "schema": FACTORIAL_CONTRACT_SCHEMA,
        "sample_idx": sample_idx,
        "factorial_arm": arm,
        "planner_arm": planner_arm,
        "body_arm": body_arm,
        "plan_provenance": MODEL_SAMPLED_PLAN_PROVENANCE,
        "raw_plan_text_sha256": sampled_plan["raw_plan_text_sha256"],
        "plan_text": sampled_plan["plan_text"],
        "plan_text_sha256": sampled_plan["plan_text_sha256"],
        "plan_state": dict(sampled_plan["plan_state"]),
        "body_prompt": sampled_plan["prompt"],
        "body_prompt_sha256": sampled_plan["body_prompt_sha256"],
        "planner_sampling_seed": planner_seed,
        "body_sampling_seed": int(ordinal_record["body_sampling_seed"]),
        "refiner_sampling_seed": int(ordinal_record["refiner_sampling_seed"]),
        "evaluation_order": int(ordinal_record["evaluation_order"]),
    }


def ordered_factorial_attempts(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_count: int,
    expected_arms: Sequence[str] = FACTORIAL_ARMS,
) -> list[Mapping[str, Any]]:
    """Sort four-arm output and reject duplicate, missing, or stray attempts."""

    count = int(expected_count)
    if count < 0:
        raise ValueError("expected_count must be non-negative")
    arms = tuple(str(arm) for arm in expected_arms)
    if len(set(arms)) != len(arms):
        raise ValueError("expected_arms contains duplicates")
    if set(arms) != set(FACTORIAL_ARMS):
        raise ValueError(f"expected_arms must be exactly {FACTORIAL_ARMS}")

    by_identity: dict[tuple[int, str], Mapping[str, Any]] = {}
    for record in records:
        if "sample_idx" not in record or "factorial_arm" not in record:
            raise ValueError("factorial record is missing sample_idx or factorial_arm")
        sample_idx = int(record["sample_idx"])
        arm = str(record["factorial_arm"])
        if not 0 <= sample_idx < count:
            raise ValueError(
                f"sample_idx {sample_idx} is outside registered range [0, {count})"
            )
        if arm not in arms:
            raise ValueError(f"unexpected factorial_arm {arm!r}")
        identity = (sample_idx, arm)
        if identity in by_identity:
            raise ValueError(f"duplicate factorial attempt {identity}")
        by_identity[identity] = record

    expected = {(sample_idx, arm) for sample_idx in range(count) for arm in arms}
    missing = sorted(expected - set(by_identity))
    if missing:
        raise ValueError(f"factorial attempt ledger mismatch: missing={missing[:16]}")
    return [
        by_identity[(sample_idx, arm)]
        for sample_idx in range(count)
        for arm in sorted(arms, key=_FACTORIAL_ARM_ORDER.__getitem__)
    ]


__all__ = [
    "FACTORIAL_ARMS",
    "FACTORIAL_ARM_COMPONENTS",
    "FACTORIAL_CONTRACT_SCHEMA",
    "MODEL_SAMPLED_PLAN_PROVENANCE",
    "PLANNER_ARMS",
    "STRUCTURE_DERIVED_TEACHER_PROVENANCE",
    "assert_planner_input_identity",
    "build_factorial_arm_input",
    "build_factorial_body_inputs",
    "build_factorial_ordinal_record",
    "build_planner_input_contract",
    "ordered_factorial_attempts",
    "persist_model_sampled_plan",
    "persist_parser_accepted_model_sampled_plan",
]
