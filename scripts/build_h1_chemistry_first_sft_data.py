#!/usr/bin/env python3
"""Build the immutable SFT-v2/SFT-v2-C chemistry-first Planner ledgers."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from crystal_dlm.h1_chemistry_first_sft import (  # noqa: E402
    H1_CHEMISTRY_FIRST_AUX_TASKS,
    H1_CHEMISTRY_FIRST_GRAD_ACCUM,
    H1_CHEMISTRY_FIRST_INFERENCE_MESSAGES_SHA256,
    H1_CHEMISTRY_FIRST_PROMPT_STYLE,
    H1_CHEMISTRY_FIRST_SFT_SCHEMA,
    H1_CHEMISTRY_FIRST_SFT_SEED,
    accumulation_group_size,
    add_weight_span,
    assign_auxiliary_tasks,
    canonical_json_sha256,
    curriculum_order,
    deterministic_mask_cursor,
    hash_shuffle,
    optimizer_update_count,
    order_pair_audit,
    record_order_sha256,
    validate_weight_spans,
    warmup_step_count,
)
from crystal_dlm.h1_llm_planner import (  # noqa: E402
    H1_PLANNER_PROMPT_STYLE_RICH_NOCHARGE,
    H1_PLANNER_SYSTEM_PROMPT,
    build_planner_messages,
    teacher_formula_answer,
)
from crystal_dlm.h1_nocharge_ion_aux import (  # noqa: E402
    SMACT4_ICSD24_FILTER,
    SMACT4_VERSION,
    format_atom_sequence,
    format_ion_sequence,
    formula_from_atom_sequence,
    formula_from_ion_sequence,
    ion_charge_sum,
    load_smact4_icsd24_oxidation_map,
)
from crystal_dlm.r5_plan_state import PLAN_STATE_VERSION  # noqa: E402
from scripts.build_h1_nocharge_ion_aux_sft_data import (  # noqa: E402
    LEGACY_EVALUATOR_SHA256,
    LEGACY_SMACT_VERSION,
    attach_smact4_witnesses,
    format_messages,
    leakage_report,
    load_legacy_snapshot,
    load_tokenizer,
    sha256_file,
    write_jsonl,
)


FROZEN_MP20_COUNTS = {"train": 27136, "val": 9047}
EXPECTED_WEIGHT_LABELS = {
    "conditional_structural_anchor": (),
    "direct_nocharge_plan": ("formula",),
    "atoms_to_formula": ("formula",),
    "ions_to_charge_sum_formula": ("charge_sum", "formula"),
    "masked_oxidation": ("oxidation",),
    "formula_to_elements_counts_n": ("elements", "counts"),
}
FORMULA_TARGET_TASKS = {
    "direct_nocharge_plan",
    "atoms_to_formula",
    "ions_to_charge_sum_formula",
}


def task_messages(user_content: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": H1_PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": str(user_content)},
    ]


def anchor_answer(plan: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"anion: {plan['anion_framework']}",
            f"lattice: {plan['lattice_system']}",
            f"spacegroup: {plan['spacegroup_bucket']}",
            f"volume: {plan['volume_per_atom_bin']}",
            "end: plan",
        ]
    )


def record_base(
    row: Mapping[str, Any],
    *,
    split: str,
    task: str,
    source_is_pos: bool,
) -> dict[str, Any]:
    plan = row["plan"]
    witness = (row.get("smact4") or {}).get("witness")
    witness_sha = None if witness is None else canonical_json_sha256(witness)
    return {
        "schema": H1_CHEMISTRY_FIRST_SFT_SCHEMA,
        "record_id": f"{split}:{int(row['row_idx']):05d}:{task}",
        "split": str(split),
        "task": str(task),
        "source_row_idx": int(row["row_idx"]),
        "material_id": str(row.get("material_id", "")),
        "formula": str(plan["formula"]),
        "reduced_formula": str(plan["reduced_formula"]),
        "num_atoms": int(plan["N"]),
        "num_elements": len(plan["elements"]),
        "plan_state_version": PLAN_STATE_VERSION,
        "sample_weight": 1.0,
        "loss_mode": "sft",
        "source_is_pos": bool(source_is_pos),
        "source_is_legacy_primary": bool(
            row["legacy"].get("valid") is True
            and row["legacy"].get("reason") == "charge_neutral_pauling_valid"
        ),
        "source_smact4_stratum": (row.get("smact4") or {}).get("stratum"),
        "smact4_witness_sha256": witness_sha,
        "generated_charge_field": False,
    }


def make_anchor_record(row: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    plan = row["plan"]
    answer = anchor_answer(plan)
    user = (
        "Training-only formula-conditioned MP-20 structural anchor. The formula is fixed input and is not a "
        "generation target. Predict only anion, lattice, spacegroup, volume, and the Plan end marker. Never repeat "
        "or alter the formula and never emit charge or oxidation fields.\n"
        f"formula: {plan['formula']}"
    )
    return {
        **record_base(row, split=split, task="conditional_structural_anchor", source_is_pos=False),
        "messages": task_messages(user),
        "answer": answer,
        "weighted_answer_spans": [],
        "formula_is_input_only": True,
        "formula_is_unconditional_target": False,
    }


def make_direct_record(row: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    plan = row["plan"]
    answer = teacher_formula_answer(
        plan,
        prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_NOCHARGE,
    )
    return {
        **record_base(row, split=split, task="direct_nocharge_plan", source_is_pos=True),
        "messages": build_planner_messages(
            sample_idx=None,
            prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_NOCHARGE,
        ),
        "answer": answer,
        "weighted_answer_spans": [
            add_weight_span(answer, str(plan["formula"]), label="formula")
        ],
        "formula_is_input_only": False,
        "formula_is_unconditional_target": True,
    }


def _masked_ion_sequence(sequence: str, cursor: int) -> tuple[str, str]:
    value = str(sequence)
    if not value.startswith("I="):
        raise ValueError("ion sequence must begin with I=")
    tokens = value[2:].split(",")
    if not (0 <= int(cursor) < len(tokens)):
        raise ValueError("masked oxidation cursor is out of range")
    symbol, state = tokens[int(cursor)].split(":", 1)
    tokens[int(cursor)] = f"{symbol}:<MASK_OXIDATION>"
    return "I=" + ",".join(tokens), state


def make_auxiliary_record(
    row: Mapping[str, Any],
    *,
    split: str,
    task: str,
    seed: int,
) -> dict[str, Any]:
    if task not in H1_CHEMISTRY_FIRST_AUX_TASKS:
        raise ValueError(f"unknown chemistry-first auxiliary task {task!r}")
    plan = row["plan"]
    witness = (row.get("smact4") or {}).get("witness")
    if not witness or (row.get("smact4") or {}).get("stratum") != "uniform_primary":
        raise ValueError("auxiliary tasks require a stable uniform-primary SMACT4 witness")
    base = record_base(row, split=split, task=task, source_is_pos=True)

    if task == "atoms_to_formula":
        sequence = format_atom_sequence(plan["elements"], plan["counts"])
        answer = f"formula: {plan['formula']}\nend: formula"
        user = (
            "Training-only composition arithmetic. Convert the repeated atom sequence to the exact flat integer-"
            "count formula. Return exactly two lines: `formula: ...` and `end: formula`.\n"
            f"atoms: {sequence}"
        )
        return {
            **base,
            "messages": task_messages(user),
            "answer": answer,
            "weighted_answer_spans": [
                add_weight_span(answer, str(plan["formula"]), label="formula")
            ],
            "aux_sequence": sequence,
            "formula_is_input_only": False,
            "formula_is_unconditional_target": True,
        }

    ion_sequence = format_ion_sequence(plan["elements"], plan["counts"], witness)
    if ion_charge_sum(ion_sequence) != 0:
        raise ValueError("frozen POS ion witness is not charge neutral")
    if task == "ions_to_charge_sum_formula":
        answer = f"charge_sum: 0\nformula: {plan['formula']}\nend: ions"
        user = (
            "Training-only oxidation arithmetic. Sum the repeated ion charges and derive the exact flat integer-"
            "count formula. Return exactly `charge_sum: ...`, `formula: ...`, and `end: ions`.\n"
            f"ions: {ion_sequence}"
        )
        return {
            **base,
            "messages": task_messages(user),
            "answer": answer,
            "weighted_answer_spans": [
                add_weight_span(answer, "0", label="charge_sum"),
                add_weight_span(answer, str(plan["formula"]), label="formula"),
            ],
            "aux_sequence": ion_sequence,
            "witness_charge_sum": 0,
            "formula_is_input_only": False,
            "formula_is_unconditional_target": True,
        }

    if task == "masked_oxidation":
        cursor = deterministic_mask_cursor(
            int(plan["N"]),
            row_idx=int(row["row_idx"]),
            material_id=str(row.get("material_id", "")),
            seed=int(seed),
        )
        masked, target = _masked_ion_sequence(ion_sequence, cursor)
        answer = f"oxidation: {target}\nend: oxidation"
        user = (
            "Training-only oxidation infill. Fill the one masked oxidation code without changing any element, "
            "multiplicity, or other oxidation code. Return exactly `oxidation: ...` and `end: oxidation`.\n"
            f"formula: {plan['formula']}\nions: {masked}"
        )
        return {
            **base,
            "messages": task_messages(user),
            "answer": answer,
            "weighted_answer_spans": [
                add_weight_span(answer, target, label="oxidation")
            ],
            "aux_sequence": masked,
            "infill_cursor": cursor,
            "infill_target": target,
            "witness_charge_sum": 0,
            "formula_is_input_only": True,
            "formula_is_unconditional_target": False,
        }

    elements = ",".join(str(value) for value in plan["elements"])
    counts = ",".join(str(int(value)) for value in plan["counts"])
    answer = (
        f"elements: {elements}\n"
        f"counts: {counts}\n"
        f"N: {int(plan['N'])}\n"
        "end: composition"
    )
    user = (
        "Training-only formula decomposition. Return the canonical element order, aligned integer counts, total N, "
        "and end marker. Do not propose a new formula.\n"
        f"formula: {plan['formula']}"
    )
    return {
        **base,
        "messages": task_messages(user),
        "answer": answer,
        "weighted_answer_spans": [
            add_weight_span(answer, elements, label="elements"),
            add_weight_span(answer, counts, label="counts"),
        ],
        "formula_is_input_only": True,
        "formula_is_unconditional_target": False,
    }


def build_common_train_records(
    rows: Sequence[Mapping[str, Any]],
    stable_primary_indices: Sequence[int],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positive_indices = sorted(set(int(value) for value in stable_primary_indices))
    positive_set = set(positive_indices)
    positive_rows = [rows[idx] for idx in positive_indices]
    assignments = assign_auxiliary_tasks(positive_rows, seed=int(seed))
    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(make_anchor_record(row, split="train"))
        row_idx = int(row["row_idx"])
        if row_idx not in positive_set:
            continue
        records.append(make_direct_record(row, split="train"))
        records.append(
            make_auxiliary_record(
                row,
                split="train",
                task=assignments[row_idx],
                seed=int(seed),
            )
        )
    expected = len(rows) + 2 * len(positive_indices)
    if len(records) != expected:
        raise RuntimeError("chemistry-first record census mismatch")
    return records, {
        "all_count": len(rows),
        "pos_count": len(positive_indices),
        "record_count": len(records),
        "expected_record_count": expected,
        "auxiliary_task_counts": dict(
            sorted(Counter(assignments.values()).items())
        ),
        "positive_source_indices_sha256": canonical_json_sha256(positive_indices),
        "auxiliary_assignment_sha256": canonical_json_sha256(assignments),
    }


def build_validation_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [make_anchor_record(row, split="val") for row in rows]


def audit_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_all_count: int,
    expected_pos_count: int,
) -> dict[str, Any]:
    failures: list[str] = []
    ids = [str(row.get("record_id", "")) for row in records]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        failures.append("record_id_identity")
    task_counts = Counter(str(row.get("task")) for row in records)
    if task_counts["conditional_structural_anchor"] != int(expected_all_count):
        failures.append("anchor_census")
    if task_counts["direct_nocharge_plan"] != int(expected_pos_count):
        failures.append("direct_pos_census")
    if sum(task_counts[task] for task in H1_CHEMISTRY_FIRST_AUX_TASKS) != int(expected_pos_count):
        failures.append("auxiliary_pos_census")
    aux_counts = [task_counts[task] for task in H1_CHEMISTRY_FIRST_AUX_TASKS]
    if aux_counts and max(aux_counts) - min(aux_counts) > 1:
        failures.append("auxiliary_cycle_imbalance")

    invalid_unconditional_formula_targets = []
    charge_plan_leaks = []
    weighting_failures = []
    source_exposure = Counter()
    formula_roundtrips = 0
    neutral_ion_roundtrips = 0
    for row in records:
        task = str(row["task"])
        source_exposure[int(row["source_row_idx"])] += 1
        answer = str(row["answer"])
        answer_lines = [line.strip().lower() for line in answer.splitlines()]
        if any(line.startswith("charge:") for line in answer_lines):
            charge_plan_leaks.append(str(row["record_id"]))
        if task in FORMULA_TARGET_TASKS and not bool(row.get("source_is_pos")):
            invalid_unconditional_formula_targets.append(str(row["record_id"]))
        if bool(row.get("formula_is_unconditional_target")) != (task in FORMULA_TARGET_TASKS):
            invalid_unconditional_formula_targets.append(str(row["record_id"]))
        if task == "conditional_structural_anchor" and any(
            line.startswith("formula:") for line in answer_lines
        ):
            invalid_unconditional_formula_targets.append(str(row["record_id"]))
        spans = list(row.get("weighted_answer_spans") or [])
        try:
            validate_weight_spans(answer, spans)
        except (KeyError, TypeError, ValueError):
            weighting_failures.append(str(row["record_id"]))
            continue
        labels = tuple(str(span.get("label")) for span in spans)
        if labels != EXPECTED_WEIGHT_LABELS[task]:
            weighting_failures.append(str(row["record_id"]))
        if float(row.get("sample_weight", -1.0)) != 1.0:
            weighting_failures.append(str(row["record_id"]))
        if task == "atoms_to_formula":
            if formula_from_atom_sequence(str(row["aux_sequence"])) != str(row["formula"]):
                failures.append(f"atom_formula_roundtrip:{row['record_id']}")
            formula_roundtrips += 1
        if task == "ions_to_charge_sum_formula":
            if formula_from_ion_sequence(str(row["aux_sequence"])) != str(row["formula"]):
                failures.append(f"ion_formula_roundtrip:{row['record_id']}")
            if ion_charge_sum(str(row["aux_sequence"])) != 0:
                failures.append(f"ion_charge_sum:{row['record_id']}")
            formula_roundtrips += 1
            neutral_ion_roundtrips += 1
    if invalid_unconditional_formula_targets:
        failures.append("invalid_formula_unconditional_target")
    if charge_plan_leaks:
        failures.append("generated_charge_field_leak")
    if weighting_failures:
        failures.append("token_weight_contract")
    return {
        "passed": not failures,
        "failures": failures,
        "record_count": len(records),
        "task_counts": dict(sorted(task_counts.items())),
        "invalid_unconditional_formula_target_count": len(
            set(invalid_unconditional_formula_targets)
        ),
        "generated_charge_field_leak_count": len(set(charge_plan_leaks)),
        "token_weight_failure_count": len(set(weighting_failures)),
        "formula_roundtrip_count": formula_roundtrips,
        "neutral_ion_roundtrip_count": neutral_ion_roundtrips,
        "source_exposure_histogram": dict(
            sorted(Counter(source_exposure.values()).items())
        ),
    }


def tokenizer_audit(
    records: Sequence[Mapping[str, Any]],
    tokenizer,
    *,
    max_length: int,
) -> dict[str, Any]:
    if tokenizer is None:
        return {"performed": False, "passed": False}
    maximum_prompt = 0
    maximum_answer = 0
    maximum_total = 0
    failures: list[dict[str, Any]] = []
    for row in records:
        prompt = format_messages(tokenizer, row["messages"])
        answer_text = str(row["answer"]).strip() + (tokenizer.eos_token or "")
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        encoded_answer = tokenizer(
            answer_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        answer_ids = encoded_answer["input_ids"]
        offsets = encoded_answer.get("offset_mapping")
        total = len(prompt_ids) + len(answer_ids)
        maximum_prompt = max(maximum_prompt, len(prompt_ids))
        maximum_answer = max(maximum_answer, len(answer_ids))
        maximum_total = max(maximum_total, total)
        reasons: list[str] = []
        if total > int(max_length):
            reasons.append("would_truncate")
        if offsets is None:
            reasons.append("missing_offsets")
        else:
            for span in row.get("weighted_answer_spans") or []:
                if not any(
                    int(end) > int(span["start"]) and int(start) < int(span["end"])
                    for start, end in offsets
                ):
                    reasons.append(f"uncovered_weight_span:{span.get('label')}")
        if reasons:
            failures.append({"record_id": row["record_id"], "reasons": reasons})
    return {
        "performed": True,
        "passed": not failures,
        "max_length": int(max_length),
        "maximum_prompt_tokens": maximum_prompt,
        "maximum_answer_tokens": maximum_answer,
        "maximum_total_tokens": maximum_total,
        "failure_count": len(failures),
        "failures": failures[:100],
    }


def write_candidate_dataset(
    output_dir: Path,
    *,
    train_records: Sequence[Mapping[str, Any]],
    val_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)
    return {
        "train_path": str(train_path),
        "train_sha256": sha256_file(train_path),
        "train_rows": len(train_records),
        "train_order_sha256": record_order_sha256(train_records),
        "val_path": str(val_path),
        "val_sha256": sha256_file(val_path),
        "val_rows": len(val_records),
        "val_order_sha256": record_order_sha256(val_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=H1_CHEMISTRY_FIRST_SFT_SEED)
    parser.add_argument("--allow-fixture-snapshot", action="store_true")
    args = parser.parse_args()

    source_rows, legacy_report = load_legacy_snapshot(
        args.legacy_snapshot_dir,
        splits=("train", "val"),
        require_frozen=not bool(args.allow_fixture_snapshot),
    )
    if not args.allow_fixture_snapshot:
        for split, expected in FROZEN_MP20_COUNTS.items():
            if len(source_rows[split]) != expected:
                raise RuntimeError(
                    f"frozen MP20 {split} count mismatch: {len(source_rows[split])} != {expected}"
                )

    oxidation_map, smact4_contract = load_smact4_icsd24_oxidation_map()
    if str(smact4_contract.get("smact_version")) != SMACT4_VERSION:
        raise RuntimeError("exact SMACT4 runtime identity mismatch")
    witness_reports = {
        split: attach_smact4_witnesses(source_rows[split], oxidation_map)
        for split in ("train", "val")
    }
    if any(not report["official_witness_parity"] for report in witness_reports.values()):
        raise RuntimeError("SMACT4 official/witness parity failed")

    common_records, census = build_common_train_records(
        source_rows["train"],
        witness_reports["train"]["stable_primary_indices"],
        seed=int(args.seed),
    )
    base_records = hash_shuffle(
        common_records,
        seed=int(args.seed),
        role="sft_v2_base_hash_shuffle",
    )
    curriculum_records = curriculum_order(common_records, seed=int(args.seed))
    val_records = hash_shuffle(
        build_validation_records(source_rows["val"]),
        seed=int(args.seed),
        role="validation_anchor_hash_shuffle",
    )
    common_audit = audit_records(
        common_records,
        expected_all_count=len(source_rows["train"]),
        expected_pos_count=witness_reports["train"]["stable_primary_count"],
    )
    pair_audit = order_pair_audit(base_records, curriculum_records)
    val_audit = audit_records(
        val_records,
        expected_all_count=len(source_rows["val"]),
        expected_pos_count=0,
    )
    # Validation is intentionally anchor-only.
    val_audit["passed"] = bool(val_audit["passed"]) and val_audit["task_counts"] == {
        "conditional_structural_anchor": len(source_rows["val"])
    }
    if not val_audit["passed"] and "validation_not_anchor_only" not in val_audit["failures"]:
        val_audit["failures"].append("validation_not_anchor_only")

    inference_messages = build_planner_messages(
        sample_idx=None,
        prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_NOCHARGE,
    )
    inference_messages_sha256 = canonical_json_sha256(inference_messages)
    if inference_messages_sha256 != H1_CHEMISTRY_FIRST_INFERENCE_MESSAGES_SHA256:
        raise RuntimeError("six-line no-charge inference prompt byte identity changed")
    split_leakage = leakage_report(source_rows)
    if any(
        int(report.get("material_id_overlap", -1)) != 0
        for report in split_leakage.values()
    ):
        raise RuntimeError("MP20 material-id leakage across frozen splits")

    tokenizer = load_tokenizer(args.tokenizer_path)
    token_audits = {
        "base_train": tokenizer_audit(base_records, tokenizer, max_length=int(args.max_length)),
        "curriculum_train": tokenizer_audit(
            curriculum_records, tokenizer, max_length=int(args.max_length)
        ),
        "validation": tokenizer_audit(val_records, tokenizer, max_length=int(args.max_length)),
    }
    if not common_audit["passed"] or not pair_audit["passed"] or not val_audit["passed"]:
        raise RuntimeError("chemistry-first data contract audit failed")
    if any(not audit["passed"] for audit in token_audits.values()):
        raise RuntimeError("chemistry-first tokenizer no-truncation audit failed")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "sft_v2": write_candidate_dataset(
            output_dir / "sft_v2",
            train_records=base_records,
            val_records=val_records,
        ),
        "sft_v2_c": write_candidate_dataset(
            output_dir / "sft_v2_c",
            train_records=curriculum_records,
            val_records=val_records,
        ),
    }
    total_updates = optimizer_update_count(len(common_records), H1_CHEMISTRY_FIRST_GRAD_ACCUM)
    final_group = accumulation_group_size(
        len(common_records) - 1,
        total_microbatches=len(common_records),
        grad_accum=H1_CHEMISTRY_FIRST_GRAD_ACCUM,
    )
    order_ledger = {
        "schema": H1_CHEMISTRY_FIRST_SFT_SCHEMA,
        "seed": int(args.seed),
        "record_multiset_sha256": pair_audit["record_multiset_sha256"],
        "sft_v2_order_sha256": pair_audit["base_order_sha256"],
        "sft_v2_c_order_sha256": pair_audit["curriculum_order_sha256"],
        "sft_v2_record_ids": [str(row["record_id"]) for row in base_records],
        "sft_v2_c_record_ids": [str(row["record_id"]) for row in curriculum_records],
    }
    (output_dir / "ORDER_LEDGER.json").write_text(
        json.dumps(order_ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": H1_CHEMISTRY_FIRST_SFT_SCHEMA,
        "status": "pass",
        "fixture_only": bool(args.allow_fixture_snapshot),
        "seed": int(args.seed),
        "prompt_style": H1_CHEMISTRY_FIRST_PROMPT_STYLE,
        "inference_messages_sha256": inference_messages_sha256,
        "legacy_contract": {
            "smact_version": LEGACY_SMACT_VERSION,
            "legacy_evaluator_sha256": LEGACY_EVALUATOR_SHA256,
            "snapshot_contract_sha256": legacy_report["contract_sha256"],
        },
        "smact4_contract": smact4_contract,
        "smact4_filter": dict(SMACT4_ICSD24_FILTER),
        "source_counts": {key: len(value) for key, value in source_rows.items()},
        "source_csv_sha256": {
            split: legacy_report["splits"][split]["source_csv_sha256"]
            for split in ("train", "val")
        },
        "witness_reports": witness_reports,
        "census": census,
        "common_record_audit": common_audit,
        "order_pair_audit": pair_audit,
        "validation_audit": val_audit,
        "tokenizer_audits": token_audits,
        "leakage": split_leakage,
        "optimizer_geometry": {
            "batch_size": 1,
            "gradient_accumulation": H1_CHEMISTRY_FIRST_GRAD_ACCUM,
            "record_count": len(common_records),
            "total_updates": total_updates,
            "warmup_steps": warmup_step_count(total_updates),
            "final_accumulation_microbatches": final_group,
            "one_epoch_exact": True,
            "drop_last": False,
            "repeat_records": False,
        },
        "outputs": outputs,
        "order_ledger_sha256": sha256_file(output_dir / "ORDER_LEDGER.json"),
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    (output_dir / "audit_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    success = {
        "schema": H1_CHEMISTRY_FIRST_SFT_SCHEMA,
        "complete": True,
        "summary_sha256": summary["summary_sha256"],
        "record_multiset_sha256": pair_audit["record_multiset_sha256"],
        "sft_v2_train_sha256": outputs["sft_v2"]["train_sha256"],
        "sft_v2_c_train_sha256": outputs["sft_v2_c"]["train_sha256"],
    }
    (output_dir / "_SUCCESS").write_text(
        json.dumps(success, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
