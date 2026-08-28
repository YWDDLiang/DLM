#!/usr/bin/env python3
"""Outcome-blind CPU proposal simulation for C³FD-v2.1 Step 4."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.c3fd_calibration import (  # type: ignore[import-not-found]
    StratumInteraction,
    calibrated_top_p_probabilities,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    value = row["proposal_targets"]
    return int(value["family"]), int(value["N"]), int(value["arity"])


def tvd(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    left_total = sum(int(value) for value in left.values())
    right_total = sum(int(value) for value in right.values())
    keys = set(left) | set(right)
    if left_total == 0 or right_total == 0:
        return 1.0
    return 0.5 * sum(
        abs(
            int(left.get(value, 0)) / left_total
            - int(right.get(value, 0)) / right_total
        )
        for value in keys
    )


def distribution(rows: Iterable[tuple[int, int, int]], index: int) -> dict[str, int]:
    return dict(sorted(Counter(str(row[index]) for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--v2-diagnostic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    train = list(iter_jsonl(args.data_dir / "train.jsonl"))
    all_keys = [key(row) for row in train]
    supported = {key(row) for row in train if row.get("composition_supervision") is True}
    supported_keys = [value for value in all_keys if value in supported]
    interaction = StratumInteraction.fit(supported_keys, alpha=1.0)
    family_size = max(value[0] for value in all_keys) + 1
    family_counts = Counter(value[0] for value in supported_keys)
    n_counts = Counter(value[1] for value in supported_keys)
    arity_counts = Counter(value[2] for value in supported_keys)
    family_logits = torch.tensor(
        [math.log(family_counts.get(index, 0) + 1.0) for index in range(family_size)]
    )
    n_logits = torch.tensor(
        [math.log(n_counts.get(index, 0) + 1.0) for index in range(1, 21)]
    )
    arity_logits = torch.tensor(
        [math.log(arity_counts.get(index, 0) + 1.0) for index in range(1, 8)]
    )
    joint_scores = interaction.joint_scores(family_logits, n_logits, arity_logits)
    probabilities = calibrated_top_p_probabilities(
        joint_scores, temperature=1.0, top_p=float(args.top_p)
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    sampled_indices = torch.multinomial(
        probabilities,
        int(args.draws),
        replacement=True,
        generator=generator,
    ).tolist()
    sampled = [interaction.strata[int(index)] for index in sampled_indices]
    reference = {
        "family": distribution(all_keys, 0),
        "N": distribution(all_keys, 1),
        "arity": distribution(all_keys, 2),
    }
    simulated = {
        "family": distribution(sampled, 0),
        "N": distribution(sampled, 1),
        "arity": distribution(sampled, 2),
    }
    distances = {
        name: tvd(simulated[name], reference[name])
        for name in ("family", "N", "arity")
    }
    v2 = json.loads(args.v2_diagnostic.read_text(encoding="utf-8"))
    p0_distance = v2["distance_to_training"]["p0"]["train_full"]
    train_all_metal = float(v2["groups"]["train_full"]["all_metal_rate"])
    gate = {
        "draws_at_least_100000": int(args.draws) >= 100_000,
        "outcome_labels_unused": True,
        "supported_train_mass_at_least_99pct": len(supported_keys) / len(all_keys) >= 0.99,
        "N_tvd_not_worse_than_p0_plus_0p01": distances["N"] <= float(p0_distance["N"]) + 0.01,
        "arity_tvd_not_worse_than_p0_plus_0p01": distances["arity"] <= float(p0_distance["arity"]) + 0.01,
        "family_tvd_not_worse_than_p0_plus_0p01": distances["family"] <= float(p0_distance["family"]) + 0.01,
        "species_top_k_zero": True,
        "global_pair_prior_zero": True,
    }
    gate["step4_pass"] = all(gate.values())
    payload = {
        "schema": "h1a2_c3fd_v21_step4_proposal_sim_v1",
        "draws": int(args.draws),
        "seed": int(args.seed),
        "top_p": float(args.top_p),
        "supported_strata": len(interaction.strata),
        "supported_train_mass": len(supported_keys) / len(all_keys),
        "full_train_all_metal_reference": train_all_metal,
        "all_metal_note": "not determined by proposal tuple; retained as a pilot-stage train-reference gate",
        "reference": reference,
        "simulated": simulated,
        "tvd_to_full_train": distances,
        "p0_tvd_to_full_train": p0_distance,
        "gate": gate,
    }
    args.output_dir.mkdir(parents=True)
    stem = "C3FD_V21_STEP4_PROPOSAL_SIM"
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["field", "candidate_tvd", "p0_tvd", "gate_limit"],
        )
        writer.writeheader()
        for name in ("family", "N", "arity"):
            writer.writerow(
                {
                    "field": name,
                    "candidate_tvd": distances[name],
                    "p0_tvd": p0_distance[name],
                    "gate_limit": float(p0_distance[name]) + 0.01,
                }
            )
    lines = [
        "# C³FD-v2.1 Step-4 proposal simulation",
        "",
        f"Step 4 pass: **{gate['step4_pass']}**",
        f"Draws: `{args.draws}`; supported train mass: `{len(supported_keys) / len(all_keys):.4%}`",
        "",
        "| Field | Candidate TVD to full train | P0 TVD | Gate limit |",
        "|---|---:|---:|---:|",
    ]
    for name in ("family", "N", "arity"):
        lines.append(
            f"| {name} | {distances[name]:.6f} | {float(p0_distance[name]):.6f} | "
            f"{float(p0_distance[name]) + 0.01:.6f} |"
        )
    lines.extend(["", "## Gates", ""])
    lines.extend(f"- {name}: `{value}`" for name, value in gate.items())
    (args.output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
