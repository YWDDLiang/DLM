#!/usr/bin/env python3
"""Finalize fixed-stream Potential-Closure raw/tau800 S.U.N. from a reused MP cache."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ARMS = ("BS", "closure_control", "potential_closed")
ENDPOINTS = ("raw", "tau800")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_runtime(source: Path):
    runtime_dir = source / "eval_runtime"
    sys.path.insert(0, str(runtime_dir))
    spec = importlib.util.spec_from_file_location("potential_finalize_official", runtime_dir / "finalize_official.py")
    if spec is None or spec.loader is None:
        raise ImportError(runtime_dir / "finalize_official.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired(left: list[dict[str, Any]], right: list[dict[str, Any]], field: str) -> dict[str, int]:
    left = sorted(left, key=lambda row: int(row["ordinal"]))
    right = sorted(right, key=lambda row: int(row["ordinal"]))
    if len(left) != 256 or len(right) != 256:
        raise ValueError("paired S.U.N. cells must retain 256 attempts")
    wins = losses = equal = 0
    for a, b in zip(left, right, strict=True):
        av, bv = bool(a[field]), bool(b[field])
        wins += int(bv and not av)
        losses += int(av and not bv)
        equal += int(av == bv)
    return {"potential_wins": wins, "potential_losses": losses, "equal": equal}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--preofficial-run", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--tau800-run", type=Path, required=True)
    parser.add_argument("--raw-eval-run", type=Path, required=True)
    parser.add_argument("--tau800-eval-run", type=Path, required=True)
    parser.add_argument("--official-query-run", type=Path, required=True)
    parser.add_argument("--bs-final-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    runtime = load_runtime(args.source_root.resolve())
    cache = args.official_query_run.resolve() / "official_mp_cache"
    if not (cache / "completion_SUCCESS").is_file():
        raise RuntimeError("reused official cache is not complete")
    phase_diagrams = runtime._phase_diagrams(cache / "official_slim_cache.jsonl")
    unresolved = {str(row["chemsys"]) for row in read_jsonl(cache / "unresolved_chemsys.jsonl")}
    args.output_dir.mkdir(parents=True)
    cells_dir = args.output_dir / "cells"
    cells_dir.mkdir()
    eval_reports = {
        "raw": read_json(args.raw_eval_run / "final" / "NATIVE_RAW_FINAL.json"),
        "tau800": read_json(args.tau800_eval_run / "final" / "TAU800_FINAL.json"),
    }
    generation = {
        (mode, "raw"): args.native_run / mode / "stream17" / "raw_generation" / "generation.jsonl"
        for mode in ("closure_control", "potential_closed")
    }
    generation.update(
        {
            (mode, "tau800"): args.tau800_run / mode / "stream17" / "generation" / "generation.jsonl"
            for mode in ("closure_control", "potential_closed")
        }
    )
    rows_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    reports: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        for mode in ("closure_control", "potential_closed"):
            source_validity = eval_reports[endpoint]["direct"][mode]
            validity_path = cells_dir / f"{mode}_{endpoint}_validity.json"
            validity_path.write_text(
                json.dumps(
                    {
                        "attempts": 256,
                        "comp_valid_count": int(source_validity["composition_valid"]),
                        "struct_valid_count": int(source_validity["structure_valid"]),
                        "valid_count": int(source_validity["direct_joint"]),
                        "metrics_unchanged_upstream": "two-validity-only; Direct not run",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            preofficial = (
                args.preofficial_run
                / "cells"
                / f"{mode}_{endpoint}_s17"
                / "evaluation"
                / "full_reconstructed"
            )
            if not (preofficial / "_SUCCESS").is_file():
                raise RuntimeError(f"incomplete preofficial cell: {mode}/{endpoint}")
            output = cells_dir / f"{mode}_{endpoint}_s17"
            rows, report = runtime._evaluate_cell(
                cell_id=f"{mode}_{endpoint}_s17",
                labels_path=preofficial / "attempt_labels_preofficial.jsonl",
                generation_path=generation[(mode, endpoint)],
                direct_path=validity_path,
                phase_diagrams=phase_diagrams,
                unresolved=unresolved,
                output_dir=output,
            )
            rows_by_cell[(mode, endpoint)] = rows
            reports.append({"arm": mode, "endpoint": endpoint, **report["counts"]})

    for endpoint, source_name in (("raw", "BS_raw_s17"), ("tau800", "BS_refined_s17")):
        source = args.bs_final_run / "cells" / source_name
        if not (source / "_SUCCESS").is_file():
            raise RuntimeError(f"missing reused BS official cell: {source_name}")
        rows = read_jsonl(source / "attempt_results_official.jsonl")
        report = read_json(source / "report.json")
        rows_by_cell[("BS", endpoint)] = rows
        reports.append({"arm": "BS", "endpoint": endpoint, **report["counts"]})

    reports.sort(key=lambda row: (ENDPOINTS.index(row["endpoint"]), ARMS.index(row["arm"])))
    paired_report = {}
    for endpoint in ENDPOINTS:
        paired_report[endpoint] = {
            field: paired(
                rows_by_cell[("closure_control", endpoint)],
                rows_by_cell[("potential_closed", endpoint)],
                field,
            )
            for field in ("strict_sun", "meta_sun")
        }
    result = {
        "schema": "potential_closure_fixed256_sun_final_v1",
        "stream": 17,
        "denominator": 256,
        "official_cache_reused": True,
        "fresh_query_run": False,
        "direct_run": False,
        "cells": reports,
        "potential_vs_control": paired_report,
    }
    (args.output_dir / "POTENTIAL_CLOSURE_SUN_FINAL.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Potential-Closure fixed256 S.U.N.",
        "",
        "| Endpoint | Arm | Comp-valid | Struct-valid | N∩U | Hull-known | Strict S.U.N. | Meta S.U.N. |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in reports:
        validity = eval_reports[row["endpoint"]]["direct"][row["arm"]]
        lines.append(
            f"| {row['endpoint']} | {row['arm']} | {validity['composition_valid']}/256 | "
            f"{validity['structure_valid']}/256 | {row['novel_unique']}/256 | "
            f"{row['hull_known_reconstructed']}/256 | {row['strict_sun']}/256 "
            f"({100*row['strict_sun']/256:.2f}%) | {row['meta_sun']}/256 "
            f"({100*row['meta_sun']/256:.2f}%) |"
        )
    (args.output_dir / "POTENTIAL_CLOSURE_SUN_FINAL.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
