#!/usr/bin/env python3
"""Attach composition rewards to R5 plan-state rollouts for training updates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.fixed_slot import write_json  # noqa: E402


REWARD_TABLES = {
    "reason_v1": {
        "charge_neutral_pauling_valid": 1.0,
        "all_metal_shortcut": 0.15,
        "single_element_shortcut": -0.6,
        "charge_neutrality_fail": -1.0,
        "pauling_fail_or_ratio_rejected": -0.7,
        "oxidation_state_missing": -0.5,
    },
    # D13: strict plan reward. This is a training signal over all direct
    # rollouts, not a sampling prior or output selector.
    "strict_v2": {
        "charge_neutral_pauling_valid": 1.2,
        "all_metal_shortcut": -0.55,
        "single_element_shortcut": -0.8,
        "charge_neutrality_fail": -1.2,
        "pauling_fail_or_ratio_rejected": -0.9,
        "oxidation_state_missing": -0.8,
    },
}

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def reward_for_record(record: Mapping[str, Any], *, reward_mode: str = "reason_v1") -> tuple[float, dict[str, Any]]:
    if reward_mode not in REWARD_TABLES:
        raise ValueError(f"Unknown reward_mode: {reward_mode}")
    if not record.get("parsed"):
        diagnostics = {
            "parsed": False,
            "valid_plan": False,
            "validator_reason": "parse_failure",
            "formula": None,
            "num_elements": 0,
        }
        return -1.2, diagnostics

    plan = record.get("plan_state") or {}
    validation = record.get("plan_validation") or {}
    validator = plan.get("validator") or record.get("smact") or {}
    reason = str(validator.get("reason") or "unknown")
    valid_plan = bool(validation.get("valid"))
    reward_table = REWARD_TABLES[reward_mode]
    reward = float(reward_table.get(reason, -0.8))
    if not valid_plan:
        reward = min(reward, -1.0)
    diagnostics = {
        "parsed": True,
        "valid_plan": valid_plan,
        "validator_valid": validator.get("valid"),
        "validator_reason": reason,
        "formula": plan.get("formula"),
        "N": plan.get("N"),
        "num_elements": len(plan.get("elements") or []),
        "charge_bucket": plan.get("charge_bucket"),
        "anion_framework": plan.get("anion_framework"),
        "lattice_system": plan.get("lattice_system"),
        "spacegroup_bucket": plan.get("spacegroup_bucket"),
        "volume_per_atom_bin": plan.get("volume_per_atom_bin"),
        "reward_mode": reward_mode,
    }
    return reward, diagnostics


def build_rewarded_rollouts(
    rows: list[Mapping[str, Any]], *, reward_mode: str = "reason_v1"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    reward_values: list[float] = []
    reason_counts: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    for idx, record in enumerate(rows):
        reward, diagnostics = reward_for_record(record, reward_mode=reward_mode)
        reason_counts[str(diagnostics.get("validator_reason"))] += 1
        prompt = str(record.get("conditioning_prompt") or record.get("prompt") or "").rstrip()
        response = str(record.get("text") or record.get("response") or "")
        prompt_counts[prompt[:120]] += 1
        reward_values.append(float(reward))
        output.append(
            {
                "sample_idx": int(record.get("sample_idx", idx)),
                "prompt": prompt,
                "response": response,
                "reward": float(reward),
                "diagnostics": diagnostics,
                "source_parsed": bool(record.get("parsed")),
                "source_representation": record.get("representation"),
                "source_plan_state": record.get("plan_state"),
                "source_plan_validation": record.get("plan_validation"),
                "reward_policy": f"r5_plan_{reward_mode}_all_rollouts",
            }
        )
    mean = sum(reward_values) / max(1, len(reward_values))
    var = sum((value - mean) ** 2 for value in reward_values) / max(1, len(reward_values))
    summary = {
        "count": len(output),
        "reward_policy": f"r5_plan_{reward_mode}_all_rollouts",
        "reward_mode": reward_mode,
        "reward_mean": mean,
        "reward_std": var ** 0.5,
        "reward_min": min(reward_values) if reward_values else None,
        "reward_max": max(reward_values) if reward_values else None,
        "reason_counts": dict(reason_counts.most_common()),
        "prompt_count": len(prompt_counts),
        "all_rollouts_retained": True,
        "uses_candidate_selection": False,
        "uses_rejection_or_retry": False,
        "uses_training_filter": False,
        "uses_sampling_prior": False,
    }
    return output, summary


def write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# R5 Plan Rollout Reward Summary",
        "",
        "All direct rollout samples are retained. Rewards are training signals only; they do not select, retry, filter, or condition samples.",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--reward-mode", choices=sorted(REWARD_TABLES), default="reason_v1")
    args = parser.parse_args()

    output, summary = build_rewarded_rollouts(read_jsonl(args.input_jsonl), reward_mode=args.reward_mode)
    write_jsonl(args.output_jsonl, output)
    write_json(str(args.summary_json), summary)
    write_markdown(summary, args.summary_md)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
