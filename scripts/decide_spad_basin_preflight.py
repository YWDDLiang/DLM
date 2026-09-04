#!/usr/bin/env python3
"""Make the preregistered train-only basin-headroom decision."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def selected_k10_e0_deltas(groups: list[Mapping[str, Any]]) -> list[float]:
    deltas: list[float] = []
    for group in groups:
        candidates = list(group.get("candidates") or ())
        no_op = next((row for row in candidates if row.get("source") == "no_op"), None)
        if no_op is None:
            continue
        no_op_e0 = finite(no_op.get("E0_energy_eV_per_atom"))
        known = [
            (finite(row.get("K10_energy_eV_per_atom")), row)
            for row in candidates
            if row.get("terminal_legal") is True
        ]
        known = [(value, row) for value, row in known if value is not None]
        if no_op_e0 is None or not known:
            continue
        _value, best = min(known, key=lambda item: float(item[0]))
        best_e0 = finite(best.get("E0_energy_eV_per_atom"))
        if best_e0 is not None:
            deltas.append(best_e0 - no_op_e0)
    return deltas


def decide(report: Mapping[str, Any], groups: list[Mapping[str, Any]]) -> dict[str, Any]:
    if report.get("schema") != "spad_basin_preflight_value_final_v1":
        raise ValueError("unexpected preflight value report")
    if int(report.get("groups", -1)) != 128 or len(groups) != 128:
        raise ValueError("preflight decision requires all 128 train groups")
    k10 = report["headroom"]["K10"]
    coverage = report["coverage"]["K10"]
    known_groups = int(k10["groups_with_known_no_op_and_candidate"])
    median_headroom = finite(k10["headroom_meV_per_atom"]["median"])
    above10 = int(k10["groups_above_headroom_threshold_meV"]["10"])
    gates = {
        "K10_candidate_coverage_at_least_90pct": float(coverage["coverage"] or 0.0) >= 0.90,
        "K10_known_paired_groups_at_least_64": known_groups >= 64,
        "K10_median_headroom_positive": median_headroom is not None and median_headroom > 0.0,
        "K10_groups_above_10meV_at_least_24": above10 >= 24,
    }
    e0_deltas = selected_k10_e0_deltas(groups)
    authorized = all(gates.values())
    return {
        "schema": "spad_basin_preflight_training_decision_v1",
        "authorized": authorized,
        "passed": authorized,
        "primary_route": "k10_basin_consistent",
        "diagnostic_route": "e0_single_point",
        "gates": gates,
        "K10": {
            "coverage": coverage["coverage"],
            "known_paired_groups": known_groups,
            "median_headroom_meV_per_atom": median_headroom,
            "groups_above_10meV": above10,
        },
        "K10_selected_action_E0_delta_eV_per_atom": {
            "count": len(e0_deltas),
            "median": None if not e0_deltas else statistics.median(e0_deltas),
            "lower": sum(value < 0 for value in e0_deltas),
            "higher": sum(value > 0 for value in e0_deltas),
        },
        "K10_vs_K20_kendall": report.get("kendall_tau_b", {}).get(
            "K10_vs_K20"
        ),
        "decision_uses_train_only_preflight": True,
        "prospective_outcomes_read": False,
        "no_parameter_or_candidate_change": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    label = args.label_run.resolve(strict=True)
    report = json.loads((label / "PRELIGHT_VALUE_FINAL.json").read_text())
    groups = list(iter_jsonl(label / "labelled_groups.jsonl"))
    decision = decide(report, groups)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    name = (
        "PRELIGHT_TRAINING_AUTHORIZED"
        if decision["authorized"]
        else "PRELIGHT_TRAINING_REJECTED"
    )
    (args.output_dir / name).write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "DECISION.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
