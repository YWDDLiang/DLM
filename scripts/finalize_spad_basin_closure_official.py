#!/usr/bin/env python3
"""Finalize native basin-closure S.U.N. without rerunning Direct."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_RUNTIME = PROJECT_ROOT / "eval_runtime"
if str(EVAL_RUNTIME) not in sys.path:
    sys.path.insert(0, str(EVAL_RUNTIME))
if os.environ.get("H1_ACTIVE_DENOMINATOR") != "256":
    raise RuntimeError("H1_ACTIVE_DENOMINATOR=256 is required")

import protocol  # noqa: E402
from finalize_official import _e_above_hull, _exact_mcnemar, _phase_diagrams  # noqa: E402


DENOMINATOR = 256


def indexed_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = protocol.read_jsonl(path.resolve())
    indexed = {int(row["ordinal"]): dict(row) for row in rows}
    if len(rows) != DENOMINATOR or set(indexed) != set(range(DENOMINATOR)):
        raise protocol.ContractError(f"fixed256 ordinal coverage changed: {path}")
    return indexed


def paired_binary(
    candidate: Sequence[bool], baseline: Sequence[bool]
) -> dict[str, Any]:
    if len(candidate) != DENOMINATOR or len(baseline) != DENOMINATOR:
        raise ValueError("paired binary vectors must cover fixed256")
    result = _exact_mcnemar(list(baseline), list(candidate))
    return {
        **result,
        "wins": result["candidate_only"],
        "losses": result["control_only"],
        "both_true": sum(left and right for left, right in zip(candidate, baseline, strict=True)),
        "both_false": sum(not left and not right for left, right in zip(candidate, baseline, strict=True)),
    }


def describe(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "mean": None if not finite else float(statistics.fmean(finite)),
        "median": None if not finite else float(statistics.median(finite)),
        "minimum": None if not finite else float(min(finite)),
        "maximum": None if not finite else float(max(finite)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    labels = indexed_rows(args.preofficial_labels)
    generation = indexed_rows(args.generation_jsonl)
    baseline = indexed_rows(args.baseline_official_attempts)
    raw_screen = json.loads(args.raw_screen_json.read_text(encoding="utf-8"))
    if raw_screen.get("schema") != "spad_basin_closure_native_raw_screen_v1":
        raise protocol.ContractError("raw screen schema changed")
    if raw_screen["arms"]["closure_ce"]["composition_valid"] != DENOMINATOR:
        raise protocol.ContractError("closure composition validity changed")

    cache = args.official_cache_dir.resolve()
    if not (cache / "completion_SUCCESS").is_file():
        raise protocol.ContractError("official query is incomplete")
    phase_diagrams = _phase_diagrams(cache / "official_slim_cache.jsonl")
    unresolved = {
        str(row["chemsys"])
        for row in protocol.read_jsonl(cache / "unresolved_chemsys.jsonl")
    }

    evaluated: list[dict[str, Any]] = []
    for ordinal in range(DENOMINATOR):
        label = labels[ordinal]
        generated = generation[ordinal]
        if str(label["attempt_id"]) != str(generated["attempt_id"]):
            raise protocol.ContractError(f"attempt identity changed at {ordinal}")
        reconstructed = label.get("reconstructed") is True
        chemsys = None if label.get("chemsys") is None else str(label["chemsys"])
        energy = label.get("chgnet_energy_per_atom")
        e_hull: float | None = None
        hull_status = "not_reconstructed"
        if reconstructed and energy is None:
            hull_status = "chgnet_energy_unknown"
        elif reconstructed and chemsys in unresolved:
            hull_status = "official_hull_unknown"
        elif reconstructed:
            pd = phase_diagrams.get(str(chemsys))
            if pd is None:
                raise protocol.ContractError(f"official cache omitted {chemsys}")
            e_hull = _e_above_hull(
                pd, label["chgnet_composition"], float(energy)
            )
            if not math.isfinite(e_hull):
                raise protocol.ContractError("non-finite official hull value")
            hull_status = "known"
        novel_unique = bool(label.get("novel")) and bool(
            label.get("unique_representative")
        )
        strict_stable = e_hull is not None and e_hull <= 0.0
        meta_stable = e_hull is not None and e_hull <= 0.1
        evaluated.append(
            {
                **label,
                "schema": "spad_basin_closure_official_attempt_v1",
                "official_hull_status": hull_status,
                "official_e_above_hull": e_hull,
                "novel_unique": novel_unique,
                "strict_stable": strict_stable,
                "meta_stable": meta_stable,
                "strict_sun": strict_stable and novel_unique,
                "meta_sun": meta_stable and novel_unique,
            }
        )

    counts = {
        "attempts": DENOMINATOR,
        "generation_succeeded": sum(
            row.get("status") == "succeeded" for row in generation.values()
        ),
        "reconstructed": sum(row["reconstructed"] for row in evaluated),
        "novel_unique": sum(row["novel_unique"] for row in evaluated),
        "chgnet_energy_known": sum(row["chgnet_relaxation_known"] for row in evaluated),
        "hull_known": sum(row["official_hull_status"] == "known" for row in evaluated),
        "hull_unknown": sum(
            row["reconstructed"] and row["official_hull_status"] != "known"
            for row in evaluated
        ),
        "strict_stable": sum(row["strict_stable"] for row in evaluated),
        "meta_stable": sum(row["meta_stable"] for row in evaluated),
        "strict_sun": sum(row["strict_sun"] for row in evaluated),
        "meta_sun": sum(row["meta_sun"] for row in evaluated),
    }
    hull_values = [
        float(row["official_e_above_hull"])
        for row in evaluated
        if row["official_e_above_hull"] is not None
    ]
    paired_hull_deltas: list[float] = []
    for ordinal, row in enumerate(evaluated):
        base = baseline[ordinal]
        if (
            row["official_e_above_hull"] is not None
            and base.get("official_e_above_hull") is not None
        ):
            paired_hull_deltas.append(
                float(row["official_e_above_hull"])
                - float(base["official_e_above_hull"])
            )
    paired = {
        field: paired_binary(
            [bool(row[field]) for row in evaluated],
            [bool(baseline[index].get(field)) for index in range(DENOMINATOR)],
        )
        for field in ("strict_stable", "meta_stable", "strict_sun", "meta_sun")
    }
    paired["e_above_hull_eV_per_atom"] = {
        "definition": "closure_ce_minus_frozen_BS",
        "delta": describe(paired_hull_deltas),
        "lower": sum(value < 0 for value in paired_hull_deltas),
        "equal": sum(value == 0 for value in paired_hull_deltas),
        "higher": sum(value > 0 for value in paired_hull_deltas),
    }
    report = {
        "schema": "spad_basin_closure_official_final_v1",
        "endpoint": "native_closure_ce_raw_stream17",
        "fixed_denominator": DENOMINATOR,
        "counts": counts,
        "rates_all_attempts": {
            "strict_sun": counts["strict_sun"] / DENOMINATOR,
            "meta_sun": counts["meta_sun"] / DENOMINATOR,
            "strict_stable": counts["strict_stable"] / DENOMINATOR,
            "meta_stable": counts["meta_stable"] / DENOMINATOR,
        },
        "official_e_above_hull_eV_per_atom": describe(hull_values),
        "paired_vs_frozen_BS_raw_s17": paired,
        "fast_raw_validity": raw_screen["arms"]["closure_ce"],
        "official_query_status": "complete_with_explicit_unresolved",
        "official_unresolved_chemsys": len(unresolved),
        "direct_run": False,
        "model494": False,
        "selection_retry_replacement": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    protocol.write_jsonl_exclusive(output / "attempt_results_official.jsonl", evaluated)
    protocol.write_json_exclusive(output / "BASIN_CLOSURE_OFFICIAL_FINAL.json", report)
    (output / "_SUCCESS").touch(exist_ok=False)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preofficial-labels", type=Path, required=True)
    parser.add_argument("--generation-jsonl", type=Path, required=True)
    parser.add_argument("--official-cache-dir", type=Path, required=True)
    parser.add_argument("--baseline-official-attempts", type=Path, required=True)
    parser.add_argument("--raw-screen-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
