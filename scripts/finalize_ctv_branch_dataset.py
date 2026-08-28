#!/usr/bin/env python3
"""Audit frozen train/validation CTV Branch returns before Q extraction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crystal_dlm.ctv_value_data import validate_branch_split  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_split(
    *,
    name: str,
    rollout_run: Path,
    label_run: Path,
    expected_branches: int,
    expected_states: int,
    continuations: int,
) -> dict[str, Any]:
    return validate_branch_split(
        split=name,
        branches=read_jsonl(rollout_run / "branch/branches.jsonl"),
        states=read_jsonl(rollout_run / "branch/states.jsonl"),
        labels=read_jsonl(label_run / "labels/CTV_BRANCH_TERMINAL_LABELS.jsonl"),
        expected_branches=expected_branches,
        expected_states=expected_states,
        continuations_per_action=continuations,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rollout-run", type=Path, required=True)
    parser.add_argument("--train-label-run", type=Path, required=True)
    parser.add_argument("--validation-rollout-run", type=Path, required=True)
    parser.add_argument("--validation-label-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train = load_split(
        name="train",
        rollout_run=args.train_rollout_run,
        label_run=args.train_label_run,
        expected_branches=2048,
        expected_states=256,
        continuations=1,
    )
    validation = load_split(
        name="validation",
        rollout_run=args.validation_rollout_run,
        label_run=args.validation_label_run,
        expected_branches=1024,
        expected_states=64,
        continuations=2,
    )
    overlap = sorted(set(train["composition_ids"]) & set(validation["composition_ids"]))
    if overlap:
        raise ValueError("CTV train and validation compositions overlap")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state_path = output / "CTV_BRANCH_STATE_AUDIT.csv"
    state_rows = [*train["state_summaries"], *validation["state_summaries"]]
    fieldnames = list(state_rows[0])
    with state_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(state_rows)

    report = {
        "schema": "h1a2_ctv_branch_dataset_audit_v1",
        "train": train["report"],
        "validation": validation["report"],
        "composition_overlap": len(overlap),
        "unknown_policy": "missing; never negative or high energy",
        "dataset_authorized": True,
        "state_csv_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
    }
    json_path = output / "CTV_BRANCH_DATASET_AUDIT.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# CTV Branch dataset audit",
        "",
        "This is a denominator/alignment and proxy-label audit, not a Q or S.U.N. result.",
        "",
        f"- Train branches/states/plans: `{train['report']['branches']}/"
        f"{train['report']['states']}/{train['report']['plans']}`",
        f"- Validation branches/states/plans: `{validation['report']['branches']}/"
        f"{validation['report']['states']}/{validation['report']['plans']}`",
        f"- Known returns: `{train['report']['known_returns']}/2048` train, "
        f"`{validation['report']['known_returns']}/1024` validation",
        f"- Direct valid: `{train['report']['direct_valid']}/2048` train, "
        f"`{validation['report']['direct_valid']}/1024` validation",
        f"- Validation cross-continuation pair agreement: "
        f"`{validation['report']['cross_pair_agreement']}`",
        f"- Validation mean state Spearman: "
        f"`{validation['report']['cross_continuation_spearman_mean']}`",
        "- Reduced-composition overlap: `0`",
        "- Dataset authorization: `true`",
    ]
    (output / "CTV_BRANCH_DATASET_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
