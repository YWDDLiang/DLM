#!/usr/bin/env python3
"""Reward MP-20 fixed-slot DLM RL rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.composition_validity import composition_record, pbc_duplicate_record
from crystal_dlm.fixed_slot import arrays_to_structure, parse_fixed_slot_answer, write_json
from crystal_dlm.rl_utils import REWARD_MODES, score_rollout_records
from scripts.sample_llada_crystals import graph_from_arrays, import_process_one


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def diagnose_row(row: dict[str, Any], process_one) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "parse_ok": False,
        "pymatgen_ok": False,
        "graph_ok": False,
        "smact_valid": False,
    }
    text = str(row.get("response", row.get("text", "")))
    try:
        arrays = parse_fixed_slot_answer(text)
        diagnostics["parse_ok"] = True
        diagnostics["num_atoms"] = int(arrays["num_atoms"])
        diagnostics["atom_types"] = list(arrays["atom_types"])
        arrays_to_structure(arrays)
        diagnostics["pymatgen_ok"] = True
        diagnostics.update(pbc_duplicate_record(arrays["frac_coords"]))
        graph_from_arrays(arrays, process_one)
        diagnostics["graph_ok"] = True
        comp = composition_record(arrays["atom_types"])
        diagnostics.update(
            {
                "smact_valid": bool(comp["comp_valid"]),
                "smact_reason": comp["reason"],
                "formula": comp["formula"],
                "symbols": comp["symbols"],
                "num_elements": comp["num_elements"],
            }
        )
    except Exception as exc:  # keep reward robust; details are diagnostics
        diagnostics["failure_type"] = type(exc).__name__
        diagnostics["failure_message"] = str(exc)
    return diagnostics


def write_markdown(summary: dict[str, Any], path: Path, reward_mode: str) -> None:
    if reward_mode == "reason_aware":
        conclusion = (
            "本文件记录本轮 online rollout 的 reason-aware reward 分布。该模式把 "
            "`charge_neutral_pauling_valid`、all-metal shortcut、single-element shortcut、"
            "charge-neutrality failure、Pauling/ratio failure 和 PBC-equivalent duplicate 分开计分，"
            "避免把所有 SMACT-valid 输出都当作同等正样本。"
        )
    elif reward_mode == "shortcut_strict":
        conclusion = (
            "本文件记录本轮 online rollout 的 v2 strict reward 分布。该模式仍不使用 preference / "
            "contrastive warmup，但会把 all-metal shortcut 当作 invalid composition 处理，并对 "
            "single-element shortcut 给更强负奖励，用于修复 shortcut 继续上升的问题。"
        )
    elif reward_mode == "shortcut_aware":
        conclusion = (
            "本文件记录本轮 online rollout 的 v2 reward 分布。v2 仍只使用 failure、SMACT valid、"
            "formula 重复和 element-set 重复惩罚，但不再把 all-metal / single-element shortcut 当成满分："
            "`charge_neutral_pauling_valid` 才给主要正奖励。"
        )
    else:
        conclusion = (
            "本文件记录本轮 online rollout 的 v1 reward 分布。第一版 reward 只使用 failure、SMACT valid、"
            "formula 重复和 element-set 重复惩罚；all-metal / single-element shortcut 只监控，不直接扣分。"
        )
    lines = [
        "# RL Reward Diagnosis",
        "",
        "## 中文结论",
        "",
        conclusion,
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen")
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default="reason_aware")
    args = parser.parse_args()

    process_one = import_process_one(args.crysllmgen_dir)
    rows = read_jsonl(args.input_jsonl)
    diagnosed = []
    for row in rows:
        new_row = dict(row)
        new_row["diagnostics"] = diagnose_row(row, process_one)
        diagnosed.append(new_row)
    scored, summary = score_rollout_records(diagnosed, reward_mode=args.reward_mode)
    write_jsonl(args.output_jsonl, scored)
    write_json(str(args.summary_json), summary)
    write_markdown(summary, args.summary_md, args.reward_mode)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
