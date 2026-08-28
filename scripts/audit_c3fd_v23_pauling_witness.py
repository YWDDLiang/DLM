#!/usr/bin/env python3
"""Audit the compiled C3FD-v2.3 Pauling-witness viability kernel."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.ccfd import FormulaToken  # noqa: E402
from crystal_dlm.ccfd_v2 import CCFDv2State, SetAtomCount  # noqa: E402
from crystal_dlm.composition_pair_prior import ValenceNode  # noqa: E402
from crystal_dlm.family_reachability import (  # noqa: E402
    PaulingWitnessReachability,
    state_symbols,
)
from crystal_dlm.r5_plan_state import anion_framework_from_symbols  # noqa: E402


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def synthetic_gate() -> dict[str, bool]:
    lithium = FormulaToken.from_symbol("Li", 1, 1)
    oxygen = FormulaToken.from_symbol("O", -2, 1)
    fluorine = FormulaToken.from_symbol("F", -1, 1)
    iron = FormulaToken.from_symbol("Fe", 2, 1)
    nodes = tuple(
        ValenceNode(token.atomic_number, token.oxidation_state)
        for token in (lithium, oxygen, fluorine, iron)
    )
    eneg = {
        lithium.atomic_number: 0.98,
        oxygen.atomic_number: 3.44,
        fluorine.atomic_number: 3.98,
        iron.atomic_number: 1.83,
    }
    oracle = PaulingWitnessReachability(
        nodes,
        electronegativity_by_atomic_number=eneg,
        metal_atomic_numbers={lithium.atomic_number, iron.atomic_number},
    )
    state = CCFDv2State.start().apply(SetAtomCount(2))
    legal = oracle.legal_species_counts(
        state, family="oxide", target_arity=2, max_species=2
    )
    inverted = PaulingWitnessReachability(
        tuple(
            ValenceNode(token.atomic_number, token.oxidation_state)
            for token in (lithium, fluorine)
        ),
        electronegativity_by_atomic_number={
            lithium.atomic_number: 4.0,
            fluorine.atomic_number: 1.0,
        },
        metal_atomic_numbers={lithium.atomic_number},
    )
    return {
        "split_false_positive_removed": lithium not in legal,
        "valid_oxide_path_retained": oxygen in legal,
        "pauling_inversion_rejected": not inverted.can_complete(
            state, family="halide", target_arity=2, max_species=2
        ),
    }


def teacher_witness_counts(
    path: Path,
    *,
    nodes: list[ValenceNode],
    family_values: list[str],
    oracle: PaulingWitnessReachability,
) -> dict[str, int | float]:
    total = valid = 0
    for row in iter_jsonl(path):
        if row.get("composition_supervision") is not True:
            continue
        total += 1
        proposal = row["proposal_targets"]
        target_n = int(proposal["N"])
        target_arity = int(proposal["arity"])
        family = str(family_values[int(proposal["family"])])
        state = CCFDv2State.start().apply(SetAtomCount(target_n))
        species = [int(value) for value in row["species_labels"]]
        counts = [int(value) for value in row["count_targets"]]
        if len(species) != len(counts):
            continue
        try:
            for species_id, count in zip(species, counts):
                node = nodes[species_id]
                state = state.apply(
                    FormulaToken(node.atomic_number, node.oxidation_state, count),
                    max_species=7,
                )
        except ValueError:
            continue
        valid += int(
            oracle.terminal_witness_valid(
                state, family=family, target_arity=target_arity
            )
        )
    return {
        "total": total,
        "valid": valid,
        "rate": 0.0 if total == 0 else valid / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-limit-sec", type=float, default=120.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.perf_counter()

    vocabulary = json.loads(
        (args.data_dir / "vocabulary.json").read_text(encoding="utf-8")
    )
    family_values = list(vocabulary["soft_vocabulary"]["anion_framework"])
    nodes = [
        ValenceNode(int(row["atomic_number"]), int(row["oxidation_state"]))
        for row in sorted(vocabulary["species"], key=lambda value: int(value["id"]))
    ]
    oracle = PaulingWitnessReachability(nodes)
    teacher = {
        split: teacher_witness_counts(
            args.data_dir / f"{split}.jsonl",
            nodes=nodes,
            family_values=family_values,
            oracle=oracle,
        )
        for split in ("train", "val")
    }
    supported_strata = sorted(
        {
            (
                int(row["proposal_targets"]["family"]),
                int(row["proposal_targets"]["N"]),
                int(row["proposal_targets"]["arity"]),
            )
            for row in iter_jsonl(args.data_dir / "train.jsonl")
            if row.get("composition_supervision") is True
        }
    )
    records: list[dict[str, Any]] = []
    for family_id, target_n, target_arity in supported_strata:
        family = str(family_values[family_id])
        state = CCFDv2State.start().apply(SetAtomCount(target_n))
        root_reachable = oracle.can_complete(
            state, family=family, target_arity=target_arity, max_species=7
        )
        dead_end = False
        steps = 0
        if root_reachable:
            while int(state.remaining_atoms or 0) > 0:
                legal = oracle.legal_species_counts(
                    state,
                    family=family,
                    target_arity=target_arity,
                    max_species=7,
                )
                if not legal:
                    dead_end = True
                    break
                state = state.apply(legal[0], max_species=7)
                steps += 1
        terminal_witness = bool(
            root_reachable
            and not dead_end
            and oracle.terminal_witness_valid(
                state, family=family, target_arity=target_arity
            )
        )
        independent_terminal = False
        if terminal_witness and state.eos_legal:
            independent_terminal = bool(
                state.end().certificate().benchmark_compatible
                and anion_framework_from_symbols(state_symbols(state)) == family
            )
        records.append(
            {
                "family_id": family_id,
                "family": family,
                "N": target_n,
                "arity": target_arity,
                "root_reachable": root_reachable,
                "dead_end": dead_end,
                "terminal_witness": terminal_witness,
                "independent_terminal": independent_terminal,
                "steps": steps,
            }
        )
    elapsed = time.perf_counter() - started
    synthetic = synthetic_gate()
    gate = {
        **synthetic,
        "train_teacher_witness_100pct": teacher["train"]["rate"] == 1.0,
        "val_teacher_witness_100pct": teacher["val"]["rate"] == 1.0,
        "all_supported_strata_root_reachable": all(
            bool(row["root_reachable"]) for row in records
        ),
        "deterministic_trajectories_zero_dead_end": not any(
            bool(row["dead_end"]) for row in records
        ),
        "all_terminals_have_constructive_witness": all(
            bool(row["terminal_witness"]) for row in records
        ),
        "all_terminals_independent_benchmark_valid": all(
            bool(row["independent_terminal"]) for row in records
        ),
        "model_weights_unused": True,
        "outcome_labels_unused": True,
        "runtime_within_frozen_limit": elapsed <= float(args.runtime_limit_sec),
    }
    gate["step2_pass"] = all(gate.values())
    payload = {
        "schema": "h1a2_c3fd_v23_pauling_witness_audit_v1",
        "data_dir": str(args.data_dir.resolve()),
        "teacher_witness": teacher,
        "supported_strata": len(records),
        "root_reachable": sum(bool(row["root_reachable"]) for row in records),
        "dead_ends": sum(bool(row["dead_end"]) for row in records),
        "terminal_witness": sum(bool(row["terminal_witness"]) for row in records),
        "independent_terminal": sum(
            bool(row["independent_terminal"]) for row in records
        ),
        "elapsed_sec": elapsed,
        "runtime_limit_sec": float(args.runtime_limit_sec),
        "oracle_stats": dict(oracle.stats()),
        "failures": [
            row
            for row in records
            if not row["root_reachable"]
            or row["dead_end"]
            or not row["terminal_witness"]
            or not row["independent_terminal"]
        ],
        "gate": gate,
    }
    args.output_dir.mkdir(parents=True)
    stem = "C3FD_V23_PAULING_WITNESS_AUDIT"
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / f"{stem}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# C³FD-v2.3 constructive Pauling-witness audit",
        "",
        f"Step 2 pass: **{gate['step2_pass']}**",
        f"Teacher witness train/val: `{teacher['train']['valid']}/{teacher['train']['total']}` / "
        f"`{teacher['val']['valid']}/{teacher['val']['total']}`.",
        f"Supported strata: `{len(records)}`; reachable: `{payload['root_reachable']}`; "
        f"dead ends: `{payload['dead_ends']}`; independent terminals: "
        f"`{payload['independent_terminal']}`.",
        f"Elapsed: `{elapsed:.3f}` s (limit `{float(args.runtime_limit_sec):.1f}` s).",
        "",
        "## Gates",
        "",
        *[f"- {key}: `{value}`" for key, value in gate.items()],
        "",
        "## Oracle cache",
        "",
        f"`{json.dumps(dict(oracle.stats()), sort_keys=True)}`",
    ]
    (args.output_dir / f"{stem}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
