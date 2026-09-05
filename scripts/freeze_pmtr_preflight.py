#!/usr/bin/env python3
"""Freeze one head-learnability split and one actual-SPAD transfer cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crystal_dlm.dynamic_crystal import parse_dynamic_answer  # noqa: E402


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            yield value


def _source_index(row: Mapping[str, Any], fallback: int) -> int:
    return int(row.get("source_row_idx", fallback))


def _hash_order(seed: int, namespace: str, source_index: int) -> bytes:
    return hashlib.sha256(
        f"pmtr-preflight-v1:{int(seed)}:{namespace}:{int(source_index)}".encode()
    ).digest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def _index_unique(rows: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for fallback, row in enumerate(rows):
        index = _source_index(row, fallback)
        if index in result:
            raise ValueError(f"duplicate source_row_idx {index}")
        normalized = dict(row)
        normalized["source_row_idx"] = index
        result[index] = normalized
    return result


def _validate_transfer_state(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if row.get("outcomes_read") not in (False, None):
        raise ValueError("actual-SPAD state has read outcomes")
    if bool(row.get("replacement", False)):
        raise ValueError("actual-SPAD state used replacement")
    source_index = int(row["mp20_train_source_row_idx"])
    answer = str(row.get("final_body") or "")
    try:
        parsed = parse_dynamic_answer(answer, strict=True)
    except Exception:
        return None
    plan = row.get("plan_state")
    program = row.get("species_program")
    if not isinstance(plan, Mapping) or not isinstance(program, list):
        raise ValueError("actual-SPAD state lacks Plan or species program")
    if int(plan.get("N", -1)) != int(parsed["num_atoms"]):
        raise ValueError("actual-SPAD final body N differs from Plan")
    if set(str(value) for value in program) != set(str(value) for value in plan["elements"]):
        raise ValueError("actual-SPAD species program differs from Plan")
    return {
        "schema": "pmtr_actual_spad_transfer_v1",
        "source_row_idx": source_index,
        "source_split": "train",
        "sample_idx": int(row.get("sample_idx", row.get("preflight_idx", source_index))),
        "prompt": str(row["prompt"]),
        "plan_state": dict(plan),
        "species_program": [str(value) for value in program],
        "raw_answer": answer,
        "num_atoms": int(parsed["num_atoms"]),
        "outcomes_read": False,
        "replacement": False,
    }


def freeze(
    *,
    teacher_rows: Iterable[dict[str, Any]],
    pointer_rows: Iterable[dict[str, Any]],
    actual_states: Iterable[dict[str, Any]],
    seed: int,
    fit_size: int,
    holdout_size: int,
    transfer_size: int,
) -> dict[str, Any]:
    teacher = _index_unique(teacher_rows)
    pointer = _index_unique(pointer_rows)
    if set(teacher) != set(pointer):
        missing_pointer = set(teacher) - set(pointer)
        missing_teacher = set(pointer) - set(teacher)
        raise ValueError(
            f"teacher/pointer source support differs: "
            f"missing_pointer={len(missing_pointer)} missing_teacher={len(missing_teacher)}"
        )

    transfer_by_source: dict[int, dict[str, Any]] = {}
    for state in actual_states:
        normalized = _validate_transfer_state(state)
        if normalized is None:
            continue
        source_index = int(normalized["source_row_idx"])
        transfer_by_source.setdefault(source_index, normalized)
    transfer_order = sorted(
        transfer_by_source,
        key=lambda index: _hash_order(seed, "actual-spad-transfer", index),
    )
    if len(transfer_order) < int(transfer_size):
        raise ValueError("not enough parseable unique actual-SPAD train states")
    transfer_ids = transfer_order[: int(transfer_size)]

    coherent_order = sorted(
        (index for index in teacher if index not in set(transfer_ids)),
        key=lambda index: _hash_order(seed, "coherent-corruption", index),
    )
    required = int(fit_size) + int(holdout_size)
    if len(coherent_order) < required:
        raise ValueError("not enough teacher rows after transfer exclusion")
    fit_ids = coherent_order[: int(fit_size)]
    holdout_ids = coherent_order[int(fit_size) : required]
    return {
        "fit_sources": [teacher[index] for index in fit_ids],
        "fit_pointers": [pointer[index] for index in fit_ids],
        "holdout_sources": [teacher[index] for index in holdout_ids],
        "holdout_pointers": [pointer[index] for index in holdout_ids],
        "transfer": [transfer_by_source[index] for index in transfer_ids],
        "fit_ids": fit_ids,
        "holdout_ids": holdout_ids,
        "transfer_ids": transfer_ids,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-train", type=Path, required=True)
    parser.add_argument("--pointer-train", type=Path, required=True)
    parser.add_argument("--actual-spad-states", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--fit-size", type=int, default=384)
    parser.add_argument("--holdout-size", type=int, default=128)
    parser.add_argument("--transfer-size", type=int, default=128)
    args = parser.parse_args(argv)
    for name in ("fit_size", "holdout_size", "transfer_size"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = freeze(
        teacher_rows=iter_jsonl(args.teacher_train),
        pointer_rows=iter_jsonl(args.pointer_train),
        actual_states=iter_jsonl(args.actual_spad_states),
        seed=int(args.seed),
        fit_size=int(args.fit_size),
        holdout_size=int(args.holdout_size),
        transfer_size=int(args.transfer_size),
    )
    for name in (
        "fit_sources",
        "fit_pointers",
        "holdout_sources",
        "holdout_pointers",
        "transfer",
    ):
        _write_jsonl(args.output_dir / f"{name}.jsonl", result[name])
    manifest = {
        "schema": "pmtr_preflight_freeze_v1",
        "seed": int(args.seed),
        "fit_sources": len(result["fit_ids"]),
        "holdout_sources": len(result["holdout_ids"]),
        "actual_spad_transfer": len(result["transfer_ids"]),
        "pairwise_disjoint": not (
            set(result["fit_ids"]) & set(result["holdout_ids"])
            or set(result["fit_ids"]) & set(result["transfer_ids"])
            or set(result["holdout_ids"]) & set(result["transfer_ids"])
        ),
        "selection": "seeded_hash_before_PMTR_training_or_outcomes",
        "outcomes_read": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
