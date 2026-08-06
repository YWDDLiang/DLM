#!/usr/bin/env python3
"""Build the matched C0/C1 no-charge ion-auxiliary H1 SFT ledger."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from crystal_dlm.fixed_slot import metadata_from_csv_row  # noqa: E402
from crystal_dlm.h1_llm_planner import (  # noqa: E402
    H1_PLANNER_PROMPT_STYLE_RICH_NOCHARGE,
    H1_PLANNER_SYSTEM_PROMPT,
    build_planner_messages,
    teacher_formula_answer,
)
from crystal_dlm.h1_nocharge_ion_aux import (  # noqa: E402
    H1_NOCHARGE_ION_AUX_SCHEMA,
    H1_NOCHARGE_ION_AUX_SEED,
    H1_NOCHARGE_ION_AUX_TASK_COUNTS,
    H1_NOCHARGE_ION_AUX_VALIDATION_TASK_COUNTS,
    SMACT4_ICSD24_FILTER,
    SMACT4_VERSION,
    assert_task_contract,
    canonical_json_sha256,
    deterministic_task_schedule,
    format_atom_sequence,
    format_ion_sequence,
    formula_from_atom_sequence,
    formula_from_ion_sequence,
    formula_weight_span,
    ion_charge_sum,
    load_smact4_icsd24_oxidation_map,
    oxidation_to_code,
    payload_weight_span,
    raked_select_indices,
    smact4_validity_with_witness,
)
from crystal_dlm.r5_plan_body import semantic_fields_from_plan  # noqa: E402
from crystal_dlm.r5_plan_state import PLAN_STATE_VERSION, plan_state_from_arrays  # noqa: E402
from scripts.build_r5c_plan_body_sft_data import read_rows, structure_row_to_arrays  # noqa: E402


RAKING_FIELDS = (
    "N",
    "arity",
    "family",
    "anion_framework",
    "lattice_system",
    "spacegroup_bucket",
    "volume_per_atom_bin",
)
POSITIVE_TASKS = ("direct_nocharge_plan", "sequence_to_formula", "oxidation_infill")
FULL_MP20_TASKS = ("conditional_mp20_anchor", "p0_kl_anchor")
LEGACY_SNAPSHOT_SCHEMA = "h1_nocharge_mp20_legacy_snapshot_v1"
LEGACY_SMACT_VERSION = "3.1.0"
LEGACY_EVALUATOR_SHA256 = "ca1c94f583e0c97a172b5c9b7ba96505257fd74dedfc618b584c34486ac1f178"


def scaled_task_counts(template: Mapping[str, int], total: int) -> dict[str, int]:
    """Scale the frozen mixture by largest remainder for fixture-only audits."""

    requested = int(total)
    if requested < len(template):
        raise ValueError(f"fixture total must be at least {len(template)}, got {requested}")
    denominator = sum(int(value) for value in template.values())
    quotas = {key: requested * int(value) / denominator for key, value in template.items()}
    counts = {key: int(value) for key, value in quotas.items()}
    for key in sorted(quotas, key=lambda item: (-(quotas[item] - counts[item]), item))[: requested - sum(counts.values())]:
        counts[key] += 1
    if any(value <= 0 for value in counts.values()):
        raise ValueError(f"fixture task scaling removed a task: {counts}")
    return counts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tokenizer(path: str | None):
    if not path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_messages(tokenizer, messages: Sequence[Mapping[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True))
    return f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"


def task_messages(user_content: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": H1_PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": str(user_content)},
    ]


def load_source_split(csv_path: Path, *, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(read_rows(csv_path, limit=limit)):
        metadata = metadata_from_csv_row(row)
        arrays = structure_row_to_arrays(row)
        plan = plan_state_from_arrays(arrays, metadata=metadata)
        plan.update(semantic_fields_from_plan(plan))
        rows.append(
            {
                "split": str(split),
                "row_idx": int(row_idx),
                "material_id": str(metadata.get("material_id", row.get("material_id", ""))),
                "plan": plan,
                "legacy": dict(plan.get("validator") or {}),
            }
        )
    return rows


def load_legacy_snapshot(
    snapshot_dir: Path,
    *,
    splits: Sequence[str] = ("train", "val", "test"),
    require_frozen: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    report_path = snapshot_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("schema") != LEGACY_SNAPSHOT_SCHEMA
        or report.get("status") != "pass"
        or report.get("legacy_smact_version") != LEGACY_SMACT_VERSION
        or report.get("legacy_evaluator_sha256") != LEGACY_EVALUATOR_SHA256
    ):
        raise RuntimeError("legacy MP20 snapshot evaluator identity mismatch")
    expected_contract = str(report.get("contract_sha256") or "")
    contract_payload = dict(report)
    contract_payload.pop("contract_sha256", None)
    if not expected_contract or canonical_json_sha256(contract_payload) != expected_contract:
        raise RuntimeError("legacy MP20 snapshot contract SHA mismatch")
    success = json.loads((snapshot_dir / "_SUCCESS").read_text(encoding="utf-8"))
    if (
        not isinstance(success, dict)
        or success.get("complete") is not True
        or success.get("contract_sha256") != expected_contract
    ):
        raise RuntimeError("legacy MP20 snapshot success marker mismatch")
    if require_frozen and report.get("fixture_only") is not False:
        raise RuntimeError("frozen data build cannot consume a fixture legacy snapshot")
    source_rows: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        path = snapshot_dir / f"{split}.jsonl"
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{path} contains a non-object row")
                    rows.append(value)
        split_report = (report.get("splits") or {}).get(split)
        if (
            not isinstance(split_report, Mapping)
            or len(rows) != int(split_report.get("row_count", -1))
            or sha256_file(path) != split_report.get("snapshot_jsonl_sha256")
        ):
            raise RuntimeError(f"legacy snapshot {split} identity mismatch")
        if [int(row.get("row_idx", -1)) for row in rows] != list(range(len(rows))):
            raise RuntimeError(f"legacy snapshot {split} ordinals are not exact")
        source_rows[split] = rows
    return source_rows, report


def attach_smact4_witnesses(
    rows: list[dict[str, Any]],
    oxidation_map: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    legacy_primary = []
    stats: Counter[str] = Counter()
    parity_failures: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        legacy = row["legacy"]
        if legacy.get("valid") is True and legacy.get("reason") == "charge_neutral_pauling_valid":
            legacy_primary.append(idx)
        else:
            stats[f"legacy:{legacy.get('reason', 'unknown')}"] += 1

    positive = []
    for idx in legacy_primary:
        row = rows[idx]
        plan = row["plan"]
        upgraded = smact4_validity_with_witness(
            str(plan["formula"]),
            list(plan["elements"]),
            list(plan["counts"]),
            oxidation_map,
        )
        row["smact4"] = upgraded
        stats[f"smact4:{upgraded['stratum']}"] += 1
        if not upgraded["official_witness_parity"]:
            parity_failures.append(
                {
                    "row_idx": idx,
                    "material_id": row["material_id"],
                    "formula": plan["formula"],
                    "upgraded": upgraded,
                }
            )
        if upgraded["valid"] and upgraded["stratum"] == "uniform_primary" and upgraded["witness"]:
            positive.append(idx)

    return {
        "legacy_primary_count": len(legacy_primary),
        "stable_primary_indices": positive,
        "stable_primary_count": len(positive),
        "strata": dict(sorted(stats.items())),
        "official_witness_parity_failures": parity_failures,
        "official_witness_parity": not parity_failures,
    }


def auxiliary_cursor(row: Mapping[str, Any], *, split: str, role: str, seed: int) -> int:
    num_atoms = int((row.get("plan") or {})["N"])
    digest = hashlib.sha256(
        f"{seed}|{split}|{role}|{row['row_idx']}|{row['material_id']}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % num_atoms


def auxiliary_infill_kind(row: Mapping[str, Any], *, split: str, seed: int) -> str:
    digest = hashlib.sha256(
        f"{seed}|{split}|oxidation_infill_kind|{row['row_idx']}|{row['material_id']}".encode("utf-8")
    ).digest()
    return "element" if digest[0] % 2 == 0 else "attribute"


def conditional_anchor_answer(plan: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"anion: {plan['anion_framework']}",
            f"lattice: {plan['lattice_system']}",
            f"spacegroup: {plan['spacegroup_bucket']}",
            f"volume: {plan['volume_per_atom_bin']}",
            "end: anchor",
        ]
    )


def _record_base(
    row: Mapping[str, Any],
    *,
    split: str,
    ordinal: int,
    task: str,
    cursor: int | None,
    infill_kind: str | None,
) -> dict[str, Any]:
    plan = row["plan"]
    return {
        "schema": H1_NOCHARGE_ION_AUX_SCHEMA,
        "record_id": f"{split}:{ordinal:04d}:{task}",
        "split": str(split),
        "ledger_ordinal": int(ordinal),
        "task": str(task),
        "source_row_idx": int(row["row_idx"]),
        "material_id": str(row["material_id"]),
        "infill_cursor": None if cursor is None else int(cursor),
        "infill_kind": infill_kind,
        "formula": str(plan["formula"]),
        "reduced_formula": str(plan["reduced_formula"]),
        "num_atoms": int(plan["N"]),
        "num_elements": len(plan["elements"]),
        "plan_state_version": PLAN_STATE_VERSION,
        "sample_weight": 1.0,
        "source_is_legacy_primary": bool(
            row["legacy"].get("valid") is True
            and row["legacy"].get("reason") == "charge_neutral_pauling_valid"
        ),
        "source_smact4_stratum": (row.get("smact4") or {}).get("stratum"),
    }


def make_paired_records(
    row: Mapping[str, Any],
    *,
    split: str,
    ordinal: int,
    task: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = row["plan"]
    cursor = auxiliary_cursor(row, split=split, role=task, seed=seed) if task == "oxidation_infill" else None
    infill_kind = auxiliary_infill_kind(row, split=split, seed=seed) if task == "oxidation_infill" else None
    base = _record_base(
        row,
        split=split,
        ordinal=ordinal,
        task=task,
        cursor=cursor,
        infill_kind=infill_kind,
    )

    if task == "direct_nocharge_plan":
        answer = teacher_formula_answer(plan, prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_NOCHARGE)
        common = {
            **base,
            "messages": build_planner_messages(sample_idx=None, prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_NOCHARGE),
            "answer": answer,
            "loss_mode": "sft",
            "weighted_answer_spans": formula_weight_span(answer, str(plan["formula"])),
        }
        return {**common, "arm": "c0_neutral_aux"}, {**common, "arm": "c1_ion_aux"}

    if task == "sequence_to_formula":
        witness = (row.get("smact4") or {}).get("witness")
        c0_sequence = format_atom_sequence(plan["elements"], plan["counts"])
        c1_sequence = format_ion_sequence(plan["elements"], plan["counts"], witness)
        answer = f"formula: {plan['formula']}\nend: formula"
        c0_user = (
            "Training-only atom-sequence arithmetic task. Convert the repeated neutral atom sequence to its exact "
            "flat integer-count formula. Return exactly `formula: ...` and `end: formula`.\n"
            f"atoms: {c0_sequence}"
        )
        c1_user = (
            "Training-only ion-sequence arithmetic task. Verify the repeated ion witness and convert it to its exact "
            "flat integer-count formula. Return exactly `formula: ...` and `end: formula`.\n"
            f"ions: {c1_sequence}"
        )
        common = {
            **base,
            "answer": answer,
            "loss_mode": "sft",
            "weighted_answer_spans": formula_weight_span(answer, str(plan["formula"])),
            "witness_charge_sum": ion_charge_sum(c1_sequence),
        }
        return (
            {**common, "arm": "c0_neutral_aux", "messages": task_messages(c0_user), "aux_sequence": c0_sequence},
            {**common, "arm": "c1_ion_aux", "messages": task_messages(c1_user), "aux_sequence": c1_sequence},
        )

    if task == "oxidation_infill":
        witness = (row.get("smact4") or {}).get("witness")
        c0_sequence = format_atom_sequence(plan["elements"], plan["counts"])
        c1_sequence = format_ion_sequence(plan["elements"], plan["counts"], witness)
        c0_tokens = c0_sequence[2:].split(",")
        c1_tokens = c1_sequence[2:].split(",")
        assert cursor is not None
        if infill_kind == "element":
            c0_target = c0_tokens[cursor].split(":", 1)[0]
            c1_target = c1_tokens[cursor].split(":", 1)[0]
            c0_tokens[cursor] = "<MASK_ELEMENT>:" + c0_tokens[cursor].split(":", 1)[1]
            c1_tokens[cursor] = "<MASK_ELEMENT>:" + c1_tokens[cursor].split(":", 1)[1]
            c0_answer = f"element: {c0_target}\nend: element"
            c1_answer = f"element: {c1_target}\nend: element"
            c0_task = "Fill the masked element without changing the repeated atom count."
            c1_task = "Fill the masked element without changing the frozen oxidation witness."
        else:
            c0_target = c0_tokens[cursor].split(":", 1)[1]
            c1_target = c1_tokens[cursor].split(":", 1)[1]
            c0_tokens[cursor] = c0_tokens[cursor].split(":", 1)[0] + ":<MASK_COUNT>"
            c1_tokens[cursor] = c1_tokens[cursor].split(":", 1)[0] + ":<MASK_STATE>"
            c0_answer = f"count: {c0_target}\nend: count"
            c1_answer = f"state: {c1_target}\nend: state"
            c0_task = "Fill the masked repeated-atom count code; the only legal target is C001."
            c1_task = "Fill the masked oxidation state while preserving the frozen charge-neutral witness."
        c0_masked = "A=" + ",".join(c0_tokens)
        c1_masked = "I=" + ",".join(c1_tokens)
        c0_user = (
            f"Training-only matched element/count infill task. {c0_task} Return only the requested field and end marker.\n"
            f"formula: {plan['formula']}\nsequence: {c0_masked}"
        )
        c1_user = (
            f"Training-only matched element/oxidation infill task. {c1_task} Return only the requested field and end marker.\n"
            f"formula: {plan['formula']}\nsequence: {c1_masked}"
        )
        return (
            {
                **base,
                "arm": "c0_neutral_aux",
                "messages": task_messages(c0_user),
                "answer": c0_answer,
                "loss_mode": "sft",
                "weighted_answer_spans": payload_weight_span(c0_answer, c0_target),
                "aux_sequence": c0_masked,
                "infill_target": c0_target,
            },
            {
                **base,
                "arm": "c1_ion_aux",
                "messages": task_messages(c1_user),
                "answer": c1_answer,
                "loss_mode": "sft",
                "weighted_answer_spans": payload_weight_span(c1_answer, c1_target),
                "aux_sequence": c1_masked,
                "infill_target": c1_target,
                "witness_charge_sum": sum(int(value) for _symbol, value in witness),
            },
        )

    if task == "conditional_mp20_anchor":
        answer = conditional_anchor_answer(plan)
        user = (
            "Training-only conditional MP-20 anchor. The formula is fixed input and is not a generation target. "
            "Predict only the four non-formula fields and the end marker; never repeat or alter the formula.\n"
            f"formula: {plan['formula']}"
        )
        common = {
            **base,
            "messages": task_messages(user),
            "answer": answer,
            "loss_mode": "sft",
            "weighted_answer_spans": [],
            "formula_is_input_only": True,
        }
        return {**common, "arm": "c0_neutral_aux"}, {**common, "arm": "c1_ion_aux"}

    if task == "p0_kl_anchor":
        answer = conditional_anchor_answer(plan)
        user = (
            "Training-only conditional function-space anchor. The formula is fixed input and is never a generation "
            "target. Preserve the reference Planner distribution over only the four non-formula fields and end marker; "
            "never repeat or alter the formula.\n"
            f"formula: {plan['formula']}"
        )
        common = {
            **base,
            "messages": task_messages(user),
            "answer": answer,
            "loss_mode": "kl_only",
            "weighted_answer_spans": [],
            "kl_mask_scope": "answer",
            "formula_is_input_only": True,
            "formula_is_sft_target": False,
            "generated_charge_field": False,
        }
        return {**common, "arm": "c0_neutral_aux"}, {**common, "arm": "c1_ion_aux"}

    raise ValueError(f"unknown no-charge ion-aux task {task!r}")


def build_split_records(
    rows: list[dict[str, Any]],
    stable_primary_indices: Sequence[int],
    *,
    split: str,
    task_counts: Mapping[str, int],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    total = sum(int(value) for value in task_counts.values())
    assert_task_contract(task_counts, expected_total=total)
    positive_count = sum(int(task_counts[task]) for task in POSITIVE_TASKS)
    positive_selected, raking_report = raked_select_indices(
        rows,
        stable_primary_indices,
        rows,
        positive_count,
        fields=RAKING_FIELDS,
        seed=int(seed),
        role=f"{split}:stable_primary",
        exact_formula_cap=4,
        reduced_formula_cap=4,
    )
    full_count = sum(int(task_counts[task]) for task in FULL_MP20_TASKS)
    excluded = set(positive_selected)
    full_population = [idx for idx in range(len(rows)) if idx not in excluded]
    full_selected, full_raking_report = raked_select_indices(
        rows,
        full_population,
        rows,
        full_count,
        fields=RAKING_FIELDS,
        seed=int(seed) + 17,
        role=f"{split}:full_mp20_anchor",
        exact_formula_cap=4,
        reduced_formula_cap=4,
    )

    queues: dict[str, deque[int]] = {}
    cursor = 0
    for task in POSITIVE_TASKS:
        count = int(task_counts[task])
        queues[task] = deque(positive_selected[cursor : cursor + count])
        cursor += count
    cursor = 0
    for task in FULL_MP20_TASKS:
        count = int(task_counts[task])
        queues[task] = deque(full_selected[cursor : cursor + count])
        cursor += count

    c0_records: list[dict[str, Any]] = []
    c1_records: list[dict[str, Any]] = []
    schedule = deterministic_task_schedule(task_counts, seed=int(seed))
    for ordinal, task in enumerate(schedule):
        source_idx = queues[task].popleft()
        c0, c1 = make_paired_records(
            rows[source_idx],
            split=split,
            ordinal=ordinal,
            task=task,
            seed=int(seed),
        )
        c0_records.append(c0)
        c1_records.append(c1)

    report = audit_paired_records(c0_records, c1_records, expected_task_counts=task_counts)
    report["raking"] = raking_report
    report["full_anchor_raking"] = full_raking_report
    report["source_row_exposure_max"] = max(
        Counter(record["source_row_idx"] for record in c0_records).values(), default=0
    )
    report["exact_formula_exposure_max"] = max(
        Counter(str(record["formula"]) for record in c0_records).values(), default=0
    )
    report["reduced_formula_exposure_max"] = max(
        Counter(str(record["reduced_formula"]) for record in c0_records).values(), default=0
    )
    return c0_records, c1_records, report


def audit_paired_records(
    c0_records: Sequence[Mapping[str, Any]],
    c1_records: Sequence[Mapping[str, Any]],
    *,
    expected_task_counts: Mapping[str, int],
) -> dict[str, Any]:
    failures: list[str] = []
    if len(c0_records) != len(c1_records):
        failures.append("arm_count_mismatch")
    c0_task_counts = Counter(str(row["task"]) for row in c0_records)
    c1_task_counts = Counter(str(row["task"]) for row in c1_records)
    if dict(c0_task_counts) != {key: int(value) for key, value in expected_task_counts.items()}:
        failures.append("c0_task_count_mismatch")
    if c0_task_counts != c1_task_counts:
        failures.append("arm_task_count_mismatch")

    nonaux_identity = 0
    ion_roundtrip = 0
    for c0, c1 in zip(c0_records, c1_records):
        identity_fields = ("record_id", "split", "ledger_ordinal", "task", "source_row_idx", "material_id", "infill_cursor", "infill_kind", "formula")
        if any(c0.get(field) != c1.get(field) for field in identity_fields):
            failures.append(f"pair_identity:{c0.get('record_id')}")
        if c0["task"] not in {"sequence_to_formula", "oxidation_infill"}:
            if c0.get("messages") != c1.get("messages") or c0.get("answer") != c1.get("answer"):
                failures.append(f"nonaux_content_mismatch:{c0.get('record_id')}")
            else:
                nonaux_identity += 1
        if c1["task"] == "sequence_to_formula":
            if formula_from_ion_sequence(str(c1["aux_sequence"])) != c1["formula"]:
                failures.append(f"ion_formula_roundtrip:{c1.get('record_id')}")
            if ion_charge_sum(str(c1["aux_sequence"])) != 0:
                failures.append(f"ion_charge_sum:{c1.get('record_id')}")
            ion_roundtrip += 1
            if formula_from_atom_sequence(str(c0["aux_sequence"])) != c0["formula"]:
                failures.append(f"atom_formula_roundtrip:{c0.get('record_id')}")
        if c0["task"] == "direct_nocharge_plan" and "charge:" in str(c0["answer"]).lower():
            failures.append(f"charge_leak:{c0.get('record_id')}")
        if c0["task"] == "conditional_mp20_anchor" and str(c0["formula"]) in str(c0["answer"]):
            failures.append(f"conditional_formula_target_leak:{c0.get('record_id')}")

    return {
        "passed": not failures,
        "failures": failures,
        "arm_count": len(c0_records),
        "task_counts": dict(sorted(c0_task_counts.items())),
        "nonaux_content_identity_count": nonaux_identity,
        "ion_roundtrip_count": ion_roundtrip,
        "pair_contract_sha256": canonical_json_sha256(
            [
                {
                    "record_id": row["record_id"],
                    "task": row["task"],
                    "source_row_idx": row["source_row_idx"],
                    "infill_cursor": row["infill_cursor"],
                    "infill_kind": row["infill_kind"],
                }
                for row in c0_records
            ]
        ),
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def tokenizer_audit(records: Sequence[Mapping[str, Any]], tokenizer, *, max_length: int) -> dict[str, Any]:
    if tokenizer is None:
        return {"performed": False}
    max_prompt = 0
    max_answer = 0
    max_total = 0
    overflow: list[str] = []
    for row in records:
        prompt = format_messages(tokenizer, row["messages"])
        answer = str(row["answer"]).strip() + (tokenizer.eos_token or "")
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        answer_len = len(tokenizer(answer, add_special_tokens=False)["input_ids"])
        max_prompt = max(max_prompt, prompt_len)
        max_answer = max(max_answer, answer_len)
        max_total = max(max_total, prompt_len + answer_len)
        if answer_len >= int(max_length):
            overflow.append(str(row["record_id"]))
    return {
        "performed": True,
        "max_prompt_tokens": max_prompt,
        "max_answer_tokens": max_answer,
        "max_total_tokens_before_prompt_left_truncation": max_total,
        "max_length": int(max_length),
        "answer_overflow_records": overflow,
        "passed": not overflow,
    }


def leakage_report(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    names = sorted(splits)
    for left_idx, left in enumerate(names):
        for right in names[left_idx + 1 :]:
            left_ids = {str(row["material_id"]) for row in splits[left] if row["material_id"]}
            right_ids = {str(row["material_id"]) for row in splits[right] if row["material_id"]}
            left_formula = {str(row["plan"]["reduced_formula"]) for row in splits[left]}
            right_formula = {str(row["plan"]["reduced_formula"]) for row in splits[right]}
            result[f"{left}__{right}"] = {
                "material_id_overlap": len(left_ids & right_ids),
                "reduced_formula_overlap": len(left_formula & right_formula),
                "left_unique_reduced_formula": len(left_formula),
                "right_unique_reduced_formula": len(right_formula),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--legacy-snapshot-dir", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=H1_NOCHARGE_ION_AUX_SEED)
    parser.add_argument("--limit", type=int, default=None, help="Fixture-only row limit; forbidden for a frozen run.")
    parser.add_argument("--fixture-train-records", type=int, default=64)
    parser.add_argument("--fixture-val-records", type=int, default=32)
    args = parser.parse_args()

    oxidation_map, smact4_contract = load_smact4_icsd24_oxidation_map()
    if str(smact4_contract["smact_version"]) != SMACT4_VERSION:
        raise RuntimeError(
            f"exact SMACT {SMACT4_VERSION} is required, found {smact4_contract['smact_version']!r}"
        )

    source_rows: dict[str, list[dict[str, Any]]] = {}
    source_hashes: dict[str, str] = {}
    legacy_snapshot_report: dict[str, Any] | None = None
    if args.legacy_snapshot_dir is not None:
        if args.limit is not None:
            raise RuntimeError("--limit cannot be combined with a frozen legacy snapshot")
        source_rows, legacy_snapshot_report = load_legacy_snapshot(
            args.legacy_snapshot_dir,
            require_frozen=True,
        )
        source_hashes = {
            split: str(legacy_snapshot_report["splits"][split]["source_csv_sha256"])
            for split in ("train", "val", "test")
        }
    else:
        if args.limit is None:
            raise RuntimeError(
                "a frozen build requires --legacy-snapshot-dir from SMACT 3.1.0; "
                "running legacy labels inside the SMACT4 process is forbidden"
            )
        for split in ("train", "val", "test"):
            csv_path = args.input_dir / f"{split}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(csv_path)
            source_hashes[split] = sha256_file(csv_path)
            source_rows[split] = load_source_split(csv_path, split=split, limit=args.limit)
    if args.limit is None:
        expected = {"train": 27136, "val": 9047}
        for split, count in expected.items():
            if len(source_rows[split]) != count:
                raise RuntimeError(f"frozen MP-20 {split} count mismatch: {len(source_rows[split])} != {count}")

    witness_reports = {
        split: attach_smact4_witnesses(source_rows[split], oxidation_map)
        for split in ("train", "val")
    }
    for split, report in witness_reports.items():
        if not report["official_witness_parity"]:
            raise RuntimeError(f"SMACT4 official/witness parity failed for {split}")

    train_task_counts = (
        H1_NOCHARGE_ION_AUX_TASK_COUNTS
        if args.limit is None
        else scaled_task_counts(H1_NOCHARGE_ION_AUX_TASK_COUNTS, int(args.fixture_train_records))
    )
    val_task_counts = (
        H1_NOCHARGE_ION_AUX_VALIDATION_TASK_COUNTS
        if args.limit is None
        else scaled_task_counts(H1_NOCHARGE_ION_AUX_VALIDATION_TASK_COUNTS, int(args.fixture_val_records))
    )
    train_c0, train_c1, train_report = build_split_records(
        source_rows["train"],
        witness_reports["train"]["stable_primary_indices"],
        split="train",
        task_counts=train_task_counts,
        seed=int(args.seed),
    )
    val_c0, val_c1, val_report = build_split_records(
        source_rows["val"],
        witness_reports["val"]["stable_primary_indices"],
        split="val",
        task_counts=val_task_counts,
        seed=int(args.seed) + 1,
    )
    if not train_report["passed"] or not val_report["passed"]:
        raise RuntimeError("paired C0/C1 ledger audit failed")

    tokenizer = load_tokenizer(args.tokenizer_path)
    token_audit = {
        "c0": tokenizer_audit(train_c0 + val_c0, tokenizer, max_length=int(args.max_length)),
        "c1": tokenizer_audit(train_c1 + val_c1, tokenizer, max_length=int(args.max_length)),
    }
    if any(audit.get("performed") and not audit.get("passed") for audit in token_audit.values()):
        raise RuntimeError("tokenizer answer-length audit failed")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "c0" / "train.jsonl", train_c0)
    write_jsonl(args.output_dir / "c0" / "val.jsonl", val_c0)
    write_jsonl(args.output_dir / "c1" / "train.jsonl", train_c1)
    write_jsonl(args.output_dir / "c1" / "val.jsonl", val_c1)
    summary = {
        "schema": H1_NOCHARGE_ION_AUX_SCHEMA,
        "status": "pass",
        "seed": int(args.seed),
        "fixture_only": args.limit is not None,
        "task_counts": {"train": train_task_counts, "val": val_task_counts},
        "source_csv_sha256": source_hashes,
        "source_counts": {split: len(rows) for split, rows in source_rows.items()},
        "legacy_snapshot": (
            None
            if legacy_snapshot_report is None
            else {
                "schema": legacy_snapshot_report["schema"],
                "legacy_smact_version": legacy_snapshot_report["legacy_smact_version"],
                "legacy_evaluator_sha256": legacy_snapshot_report["legacy_evaluator_sha256"],
                "contract_sha256": legacy_snapshot_report["contract_sha256"],
            }
        ),
        "smact4_contract": smact4_contract,
        "legacy_evaluator_source_sha256": sha256_file(PROJECT_ROOT / "crystal_dlm/composition_validity.py"),
        "witness_reports": witness_reports,
        "train_pair_report": train_report,
        "val_pair_report": val_report,
        "tokenizer_audit": token_audit,
        "leakage": leakage_report(source_rows),
        "inference_prompt_style": H1_PLANNER_PROMPT_STYLE_RICH_NOCHARGE,
        "kl_anchor_prompt_style": "formula_input_only_nonformula_fields_v1",
        "smact4_filter": dict(SMACT4_ICSD24_FILTER),
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    (args.output_dir / "audit_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_SUCCESS").write_text(
        json.dumps(
            {
                "schema": H1_NOCHARGE_ION_AUX_SCHEMA,
                "complete": True,
                "summary_sha256": summary["summary_sha256"],
                "train_pair_contract_sha256": train_report["pair_contract_sha256"],
                "val_pair_contract_sha256": val_report["pair_contract_sha256"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
