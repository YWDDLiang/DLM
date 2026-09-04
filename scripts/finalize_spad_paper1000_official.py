#!/usr/bin/env python3
"""Finalize parser-conditioned SPAD paper1000 S.U.N. from an existing MP cache."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_RUNTIME = PROJECT_ROOT / "eval_runtime"
if str(EVAL_RUNTIME) not in sys.path:
    sys.path.insert(0, str(EVAL_RUNTIME))

import protocol  # noqa: E402
from finalize_official import _e_above_hull, _phase_diagrams  # noqa: E402


DENOMINATOR = 1000


def indexed_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = protocol.read_jsonl(path.resolve())
    indexed = {int(row["ordinal"]): dict(row) for row in rows}
    if len(rows) != DENOMINATOR or set(indexed) != set(range(DENOMINATOR)):
        raise protocol.ContractError(f"paper1000 ordinal coverage changed: {path}")
    return indexed


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
    if os.environ.get("H1_ACTIVE_DENOMINATOR") != str(DENOMINATOR):
        raise RuntimeError("H1_ACTIVE_DENOMINATOR=1000 is required")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    labels = indexed_rows(args.preofficial_labels)
    generation = indexed_rows(args.generation_jsonl)
    cache = args.official_cache_dir.resolve()
    if not (cache / "completion_SUCCESS").is_file():
        raise protocol.ContractError("official cache is incomplete")
    phase_diagrams = _phase_diagrams(cache / "official_slim_cache.jsonl")
    unresolved = {
        str(row["chemsys"])
        for row in protocol.read_jsonl(cache / "unresolved_chemsys.jsonl")
    }

    evaluated: list[dict[str, Any]] = []
    cache_not_covered: set[str] = set()
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
            hull_status = "official_cache_explicitly_unresolved"
        elif reconstructed:
            pd = phase_diagrams.get(str(chemsys))
            if pd is None:
                hull_status = "official_cache_not_covered"
                cache_not_covered.add(str(chemsys))
            else:
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
                "schema": "spad_paper1000_official_attempt_v1",
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
        "chgnet_energy_known": sum(
            row["chgnet_relaxation_known"] for row in evaluated
        ),
        "hull_known": sum(
            row["official_hull_status"] == "known" for row in evaluated
        ),
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
    report = {
        "schema": "spad_paper1000_official_final_v1",
        "endpoint": str(args.endpoint),
        "conditional_parser_valid_denominator": DENOMINATOR,
        "counts": counts,
        "rates_all_attempts": {
            name: counts[name] / DENOMINATOR
            for name in ("strict_sun", "meta_sun", "strict_stable", "meta_stable")
        },
        "official_e_above_hull_eV_per_atom": describe(hull_values),
        "official_cache_explicitly_unresolved_chemsys": len(unresolved),
        "official_cache_not_covered_chemsys": sorted(cache_not_covered),
        "uncovered_cache_rows_count_as_not_stable": True,
        "new_official_query": False,
        "direct_run": False,
        "model494": bool(args.model494),
        "model494_tau": None if args.model494_tau is None else int(args.model494_tau),
        "selection_retry_replacement": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    protocol.write_jsonl_exclusive(output / "attempt_results_official.jsonl", evaluated)
    protocol.write_json_exclusive(output / "PAPER1000_OFFICIAL_FINAL.json", report)
    (output / "_SUCCESS").touch(exist_ok=False)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preofficial-labels", type=Path, required=True)
    parser.add_argument("--generation-jsonl", type=Path, required=True)
    parser.add_argument("--official-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model494", action="store_true")
    parser.add_argument("--model494-tau", type=int)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
