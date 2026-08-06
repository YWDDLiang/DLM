#!/usr/bin/env python3
"""Build H1 Llama formula-planner SFT data from MP-20 CSV splits."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from crystal_dlm.h1_llm_planner import (  # noqa: E402
    H1_PLANNER_PROMPT_VERSION,
    H1_PLANNER_SYSTEM_PROMPT,
    build_planner_messages,
    build_planner_user_prompt,
    format_planner_prompt,
    normalize_prompt_style,
    teacher_formula_answer,
)
from crystal_dlm.fixed_slot import metadata_from_csv_row, write_json  # noqa: E402
from crystal_dlm.r5_plan_body import format_composition_plan  # noqa: E402
from crystal_dlm.r5_plan_state import (  # noqa: E402
    ALLOWED_LATTICE_SYSTEMS,
    ALLOWED_SPACEGROUP_BUCKETS,
    CHARGE_BUCKET_TO_CODE,
    PLAN_STATE_VERSION,
    plan_state_from_arrays,
)
from scripts.build_r5c_plan_body_sft_data import (  # noqa: E402
    histogram_add,
    read_rows,
    split_names,
    structure_row_to_arrays,
)

ANION_VALUES = (
    "oxide",
    "sulfide",
    "chalcogenide",
    "halide",
    "nitride",
    "phosphide_or_phosphate",
    "other",
)
CHARGE_VALUES = tuple(CHARGE_BUCKET_TO_CODE)
LATTICE_VALUES = tuple(sorted(ALLOWED_LATTICE_SYSTEMS))
SPACEGROUP_VALUES = tuple(sorted(ALLOWED_SPACEGROUP_BUCKETS))
VOLUME_VALUES = tuple(f"volpa_{start:03d}_{start + 4:03d}" for start in range(0, 40, 5))
H1A3_SAMPLE_TYPES = ("direct_plan", "correct_plan", "consistency_explain", "formula_count_check")


def load_tokenizer(tokenizer_path: str | None):
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def token_len(tokenizer: Any, text: str) -> int | None:
    if tokenizer is None:
        return None
    return int(len(tokenizer(text, add_special_tokens=False)["input_ids"]))


def format_messages_prompt(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> str | None:
    if tokenizer is None:
        return None
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True))
    if len(messages) >= 2:
        return f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"
    raise ValueError("planner messages must contain system and user turns")


def _alternate(value: Any, choices: Sequence[str], *, offset: int) -> str:
    current = str(value)
    pool = [item for item in choices if item != current]
    if not pool:
        return current
    return pool[int(offset) % len(pool)]


def corrupted_rich_plan(plan_state: Mapping[str, Any], *, row_idx: int) -> tuple[str, list[str]]:
    """Return a deterministic corrupted rich plan and concise violation labels."""

    fields = {
        "formula": str(plan_state.get("formula", "")),
        "anion": str(plan_state.get("anion_framework", "other")),
        "charge": str(plan_state.get("charge_bucket", "validator_unavailable")),
        "lattice": str(plan_state.get("lattice_system", "triclinic")),
        "spacegroup": str(plan_state.get("spacegroup_bucket", "sg_001_002")),
        "volume": str(plan_state.get("volume_per_atom_bin", "volpa_000_004")),
    }
    mode = int(row_idx) % 6
    labels: list[str] = []
    if mode == 0:
        fields["anion"] = _alternate(fields["anion"], ANION_VALUES, offset=row_idx)
        labels.append("anion_formula_mismatch")
    elif mode == 1:
        fields["charge"] = _alternate(fields["charge"], CHARGE_VALUES, offset=row_idx)
        labels.append("charge_formula_mismatch")
    elif mode == 2:
        fields["lattice"] = _alternate(fields["lattice"], LATTICE_VALUES, offset=row_idx)
        labels.append("lattice_spacegroup_mismatch")
    elif mode == 3:
        fields["spacegroup"] = _alternate(fields["spacegroup"], SPACEGROUP_VALUES, offset=row_idx)
        labels.append("spacegroup_lattice_mismatch")
    elif mode == 4:
        fields["volume"] = _alternate(fields["volume"], VOLUME_VALUES, offset=row_idx)
        labels.append("volume_formula_mismatch")
    else:
        fields["charge"] = _alternate(fields["charge"], CHARGE_VALUES, offset=row_idx)
        fields["anion"] = _alternate(fields["anion"], ANION_VALUES, offset=row_idx + 1)
        labels.extend(["charge_formula_mismatch", "anion_formula_mismatch"])
    text = "\n".join(
        [
            f"formula: {fields['formula']}",
            f"anion: {fields['anion']}",
            f"charge: {fields['charge']}",
            f"lattice: {fields['lattice']}",
            f"spacegroup: {fields['spacegroup']}",
            f"volume: {fields['volume']}",
            "end: plan",
        ]
    )
    return text, labels


def consistency_messages(*, sample_idx: int | None, prompt_style: str, corrupted_plan: str, mode: str) -> list[dict[str, str]]:
    base_prompt = build_planner_user_prompt(sample_idx=sample_idx, prompt_style=prompt_style)
    if mode == "correct_plan":
        task = (
            "Training-only consistency task. The rich plan below may contain inconsistent fields. "
            "Return the corrected seven-line rich plan only, using the same output format as above.\n\n"
            f"visible_wrong_plan:\n{corrupted_plan}"
        )
    elif mode == "consistency_explain":
        task = (
            "Training-only consistency task. The rich plan below may contain inconsistent fields. "
            "Return exactly two lines: `violation_labels: <comma-separated labels>` and `end: labels`.\n\n"
            f"visible_wrong_plan:\n{corrupted_plan}"
        )
    elif mode == "formula_count_check":
        task = (
            "Training-only arithmetic task. For the formula below, return exactly five lines: "
            "`formula: ...`, `elements: ...`, `counts: ...`, `N: ...`, `end: check`.\n\n"
            f"formula: {corrupted_plan}"
        )
    else:
        raise ValueError(f"unknown consistency mode {mode!r}")
    return [
        {"role": "system", "content": H1_PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": base_prompt + "\n\n" + task},
    ]


def formula_count_answer(plan_state: Mapping[str, Any]) -> str:
    elements = [str(value) for value in plan_state.get("elements") or []]
    counts = [str(int(value)) for value in plan_state.get("counts") or []]
    return "\n".join(
        [
            f"formula: {plan_state.get('formula')}",
            f"elements: {', '.join(elements)}",
            f"counts: {', '.join(counts)}",
            f"N: {int(plan_state['N'])}",
            "end: check",
        ]
    )


def build_record(
    *,
    split: str,
    row_idx: int,
    plan_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    tokenizer: Any = None,
    sample_weight: float = 1.0,
    prompt_style: str = "chat_formula_end_v1",
    include_sample_id: bool = True,
    task: str = "direct_plan",
    messages: list[dict[str, str]] | None = None,
    answer: str | None = None,
) -> dict[str, Any]:
    sample_idx = row_idx if include_sample_id else None
    answer = teacher_formula_answer(plan_state, prompt_style=prompt_style) if answer is None else str(answer)
    if messages is None:
        messages = build_planner_messages(sample_idx=sample_idx, prompt_style=prompt_style)
        prompt_text = (
            format_planner_prompt(tokenizer, sample_idx=sample_idx, prompt_style=prompt_style)
            if tokenizer is not None
            else None
        )
    else:
        prompt_text = format_messages_prompt(tokenizer, messages)
    return {
        "task": f"h1_llm_{task}",
        "h1a3_sample_type": task,
        "representation": f"h1_llm_plan_{prompt_style}",
        "prompt_version": H1_PLANNER_PROMPT_VERSION,
        "prompt_style": prompt_style,
        "plan_state_version": PLAN_STATE_VERSION,
        "split": split,
        "row_idx": int(row_idx),
        "include_sample_id": bool(include_sample_id),
        "messages": messages,
        "prompt": prompt_text,
        "answer": answer,
        "text": None if prompt_text is None else prompt_text + answer,
        "plan_text": answer,
        "plan_state": dict(plan_state),
        "metadata": dict(metadata),
        "num_atoms": int(plan_state["N"]),
        "num_elements": len(plan_state.get("elements") or []),
        "sample_weight": float(sample_weight),
        "prompt_model_length": token_len(tokenizer, prompt_text) if prompt_text is not None else None,
        "answer_model_length": token_len(tokenizer, answer),
    }


def build_records_for_plan(
    *,
    split: str,
    row_idx: int,
    plan_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    tokenizer: Any,
    prompt_style: str,
    include_sample_id: bool,
    sample_types: Sequence[str],
    weights: Mapping[str, float],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sample_idx = row_idx if include_sample_id else None
    corrupted_plan, labels = corrupted_rich_plan(plan_state, row_idx=row_idx)
    for sample_type in sample_types:
        if sample_type == "direct_plan":
            records.append(
                build_record(
                    split=split,
                    row_idx=row_idx,
                    plan_state=plan_state,
                    metadata=metadata,
                    tokenizer=tokenizer,
                    sample_weight=float(weights.get(sample_type, 1.0)),
                    prompt_style=prompt_style,
                    include_sample_id=include_sample_id,
                    task=sample_type,
                )
            )
        elif sample_type == "correct_plan":
            records.append(
                build_record(
                    split=split,
                    row_idx=row_idx,
                    plan_state=plan_state,
                    metadata=metadata,
                    tokenizer=tokenizer,
                    sample_weight=float(weights.get(sample_type, 0.5)),
                    prompt_style=prompt_style,
                    include_sample_id=include_sample_id,
                    task=sample_type,
                    messages=consistency_messages(
                        sample_idx=sample_idx,
                        prompt_style=prompt_style,
                        corrupted_plan=corrupted_plan,
                        mode=sample_type,
                    ),
                    answer=format_composition_plan(plan_state, plan_style=prompt_style),
                )
            )
        elif sample_type == "consistency_explain":
            records.append(
                build_record(
                    split=split,
                    row_idx=row_idx,
                    plan_state=plan_state,
                    metadata=metadata,
                    tokenizer=tokenizer,
                    sample_weight=float(weights.get(sample_type, 0.25)),
                    prompt_style=prompt_style,
                    include_sample_id=include_sample_id,
                    task=sample_type,
                    messages=consistency_messages(
                        sample_idx=sample_idx,
                        prompt_style=prompt_style,
                        corrupted_plan=corrupted_plan,
                        mode=sample_type,
                    ),
                    answer=f"violation_labels: {', '.join(labels)}\nend: labels",
                )
            )
        elif sample_type == "formula_count_check":
            records.append(
                build_record(
                    split=split,
                    row_idx=row_idx,
                    plan_state=plan_state,
                    metadata=metadata,
                    tokenizer=tokenizer,
                    sample_weight=float(weights.get(sample_type, 0.25)),
                    prompt_style=prompt_style,
                    include_sample_id=include_sample_id,
                    task=sample_type,
                    messages=consistency_messages(
                        sample_idx=sample_idx,
                        prompt_style=prompt_style,
                        corrupted_plan=str(plan_state.get("formula")),
                        mode=sample_type,
                    ),
                    answer=formula_count_answer(plan_state),
                )
            )
        else:
            raise ValueError(f"unknown H1-A3 sample type {sample_type!r}")
    return records


def build_split(
    *,
    split: str,
    input_csv: Path,
    output_jsonl: Path,
    tokenizer: Any,
    limit: int | None,
    progress_every: int,
    sample_weight: float,
    prompt_style: str,
    include_sample_id: bool,
    sample_types: Sequence[str],
    weights: Mapping[str, float],
) -> Dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, Any] = {
        "split": split,
        "input_csv": str(input_csv),
        "rows_seen": 0,
        "rows_written": 0,
        "failures": 0,
        "prompt_model_lengths": {},
        "answer_model_lengths": {},
        "atom_count_histogram": {},
        "num_elements_histogram": {},
        "formula_histogram": {},
        "family_histogram": {},
        "arity_histogram": {},
        "size_histogram": {},
        "anion_framework_histogram": {},
        "charge_bucket_histogram": {},
        "lattice_system_histogram": {},
        "spacegroup_bucket_histogram": {},
        "volume_per_atom_bin_histogram": {},
        "sample_type_counts": {},
        "sample_type_weights": dict(weights),
    }
    failure_path = output_jsonl.with_suffix(".failure_cases.jsonl")
    with output_jsonl.open("w", encoding="utf-8") as out, failure_path.open("w", encoding="utf-8") as failures:
        for row_idx, row in enumerate(read_rows(input_csv, limit=limit)):
            stats["rows_seen"] += 1
            try:
                metadata = metadata_from_csv_row(row)
                arrays = structure_row_to_arrays(row)
                plan = plan_state_from_arrays(arrays, metadata=metadata)
                records = build_records_for_plan(
                    split=split,
                    row_idx=row_idx,
                    plan_state=plan,
                    metadata=metadata,
                    tokenizer=tokenizer,
                    prompt_style=prompt_style,
                    include_sample_id=include_sample_id,
                    sample_types=sample_types,
                    weights=weights,
                )
                for record in records:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["rows_written"] += 1
                    histogram_add(stats["sample_type_counts"], record.get("h1a3_sample_type", "unknown"))
                histogram_add(stats["atom_count_histogram"], record["num_atoms"])
                histogram_add(stats["num_elements_histogram"], record["num_elements"])
                histogram_add(stats["formula_histogram"], plan.get("formula", "unknown"))
                histogram_add(stats["family_histogram"], plan.get("family", "unknown"))
                histogram_add(stats["arity_histogram"], plan.get("arity", "unknown"))
                histogram_add(stats["size_histogram"], plan.get("size", "unknown"))
                histogram_add(stats["anion_framework_histogram"], plan.get("anion_framework", "unknown"))
                histogram_add(stats["charge_bucket_histogram"], plan.get("charge_bucket", "unknown"))
                histogram_add(stats["lattice_system_histogram"], plan.get("lattice_system", "unknown"))
                histogram_add(stats["spacegroup_bucket_histogram"], plan.get("spacegroup_bucket", "sg_unknown"))
                histogram_add(stats["volume_per_atom_bin_histogram"], plan.get("volume_per_atom_bin", "volpa_unknown"))
                if record["prompt_model_length"] is not None:
                    histogram_add(stats["prompt_model_lengths"], record["prompt_model_length"])
                if record["answer_model_length"] is not None:
                    histogram_add(stats["answer_model_lengths"], record["answer_model_length"])
            except Exception as exc:  # noqa: BLE001
                stats["failures"] += 1
                failures.write(
                    json.dumps(
                        {
                            "split": split,
                            "row_idx": row_idx,
                            "material_id": row.get("material_id"),
                            "reason": type(exc).__name__,
                            "message": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if progress_every > 0 and stats["rows_seen"] % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "h1_formula_sft_builder_progress",
                            "split": split,
                            "rows_seen": stats["rows_seen"],
                            "rows_written": stats["rows_written"],
                            "failures": stats["failures"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return stats


def parse_splits(value: str | Sequence[str]) -> list[str]:
    names = split_names(value)
    if not names:
        raise ValueError("at least one split is required")
    return names


def parse_sample_types(value: str | Sequence[str]) -> list[str]:
    names = split_names(value)
    if not names:
        raise ValueError("at least one H1-A3 sample type is required")
    unknown = [name for name in names if name not in H1A3_SAMPLE_TYPES]
    if unknown:
        raise ValueError(f"unknown H1-A3 sample types {unknown}; expected {H1A3_SAMPLE_TYPES}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20_h1_llm_formula_sft")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--sample-weight", type=float, default=1.0)
    parser.add_argument("--mixture", default="direct_plan")
    parser.add_argument("--direct-plan-weight", type=float, default=1.0)
    parser.add_argument("--correct-plan-weight", type=float, default=0.5)
    parser.add_argument("--consistency-explain-weight", type=float, default=0.25)
    parser.add_argument("--formula-count-check-weight", type=float, default=0.25)
    parser.add_argument("--prompt-style", default="chat_formula_end_v1")
    parser.add_argument("--include-sample-id", dest="include_sample_id", action="store_true", default=True)
    parser.add_argument("--no-include-sample-id", dest="include_sample_id", action="store_false")
    parser.add_argument("--allow-missing-splits", action="store_true")
    args = parser.parse_args()

    prompt_style = normalize_prompt_style(args.prompt_style)
    sample_types = parse_sample_types(args.mixture)
    weights = {
        "direct_plan": float(args.direct_plan_weight) * float(args.sample_weight),
        "correct_plan": float(args.correct_plan_weight) * float(args.sample_weight),
        "consistency_explain": float(args.consistency_explain_weight) * float(args.sample_weight),
        "formula_count_check": float(args.formula_count_check_weight) * float(args.sample_weight),
    }
    tokenizer = load_tokenizer(args.tokenizer_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits: Dict[str, Any] = {}
    for split in parse_splits(args.splits):
        input_csv = args.input_dir / f"{split}.csv"
        if not input_csv.exists():
            if args.allow_missing_splits:
                splits[split] = {
                    "split": split,
                    "input_csv": str(input_csv),
                    "missing": True,
                    "rows_seen": 0,
                    "rows_written": 0,
                    "failures": 0,
                }
                continue
            raise FileNotFoundError(f"Requested split CSV does not exist: {input_csv}")
        splits[split] = build_split(
            split=split,
            input_csv=input_csv,
            output_jsonl=args.output_dir / f"{split}.jsonl",
            tokenizer=tokenizer,
            limit=args.limit,
            progress_every=args.progress_every,
            sample_weight=args.sample_weight,
            prompt_style=prompt_style,
            include_sample_id=bool(args.include_sample_id),
            sample_types=sample_types,
            weights=weights,
        )

    summary = {
        "representation": f"h1_llm_plan_{prompt_style}",
        "prompt_version": H1_PLANNER_PROMPT_VERSION,
        "prompt_style": prompt_style,
        "plan_state_version": PLAN_STATE_VERSION,
        "answer_layout": "prompt-style dependent ordinary-text plan; no body coordinates",
        "include_sample_id": bool(args.include_sample_id),
        "splits": splits,
        "sample_weight": float(args.sample_weight),
        "mixture": sample_types,
        "sample_type_weights": weights,
    }
    write_json(str(args.output_dir / "stats.json"), summary)
    write_json(
        str(args.output_dir / "_SUCCESS"),
        {
            "representation": f"h1_llm_plan_{prompt_style}",
            "prompt_version": H1_PLANNER_PROMPT_VERSION,
            "prompt_style": prompt_style,
            "complete": True,
            "splits": {
                split: {
                    "missing": bool(stats.get("missing", False)),
                    "rows_seen": stats.get("rows_seen", 0),
                    "rows_written": stats.get("rows_written", 0),
                    "failures": stats.get("failures", 0),
                }
                for split, stats in splits.items()
            },
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
