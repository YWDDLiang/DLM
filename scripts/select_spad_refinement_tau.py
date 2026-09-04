#!/usr/bin/env python3
"""Select one refinement tau from stream17 development before held-out use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


TAUS = (400, 600, 800)


def select(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("schema") != "spad_low_noise_tau_calibration_final_v1":
        raise ValueError("unexpected tau calibration schema")
    by_tau = {int(row["tau"]): row for row in report.get("cells") or ()}
    if set(by_tau) != set(TAUS):
        raise ValueError("tau calibration must contain exactly 400/600/800")
    scored = []
    for tau in TAUS:
        rates = by_tau[tau]["rates"]
        strict = float(rates["strict_sun"])
        meta = float(rates["meta_sun"])
        scored.append(
            {
                "tau": tau,
                "strict_sun": strict,
                "meta_sun": meta,
                "balanced_target_fraction": min(strict / 0.10, meta / 0.50),
            }
        )
    # Development selection is fixed: maximize the weaker target-normalized
    # endpoint, then prefer less destructive noise when numerically tied.
    best = max(row["balanced_target_fraction"] for row in scored)
    winner = min(
        (
            row
            for row in scored
            if abs(row["balanced_target_fraction"] - best) <= 1.0e-12
        ),
        key=lambda row: row["tau"],
    )
    return {
        "schema": "spad_refinement_tau_selection_v1",
        "selected_tau": int(winner["tau"]),
        "selection_rule": "max_min(strict_sun/0.10,meta_sun/0.50)_tie_smallest_tau",
        "development_stream": 17,
        "heldout_stream": 18,
        "scores": scored,
        "heldout_outcomes_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-final", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    report = json.loads(args.calibration_final.read_text(encoding="utf-8"))
    result = select(report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "TAU_SELECTION.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "SELECTED_TAU").write_text(
        str(result["selected_tau"]) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
