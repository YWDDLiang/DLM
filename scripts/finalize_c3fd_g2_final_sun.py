#!/usr/bin/env python3
"""Finalize the matched BASE/G2 prospective official S.U.N. endpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import finalize_c3fd_llama_prospective_sun as common


ATTEMPTS = 256
STREAM = 17
ROUTES = ("BASE", "G2")
STAGES = ("raw", "refined")
STEM = "C3FD_G2_FINAL_SUN"


def distribution(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "known": len(finite),
        "quantiles": {
            key: common.quantile(finite, probability)
            for key, probability in (
                ("min", 0.0),
                ("q25", 0.25),
                ("median", 0.5),
                ("q75", 0.75),
                ("q90", 0.9),
                ("max", 1.0),
            )
        },
        "ecdf": {
            str(threshold): (
                None
                if not finite
                else sum(value <= threshold for value in finite) / len(finite)
            )
            for threshold in (0.0, 0.05, 0.1, 0.2, 0.5)
        },
    }


def summarize_cell(stage: str, route: str, report: Mapping[str, Any], rows):
    summary = common.summarize_cell(
        "prospective", stage, STREAM, route, report
    )
    summary["chgnet_energy"] = distribution(
        [
            row["chgnet_energy_per_atom"]
            for row in rows
            if row.get("chgnet_relaxation_known") is True
            and row.get("chgnet_energy_per_atom") is not None
        ]
    )
    summary["official_e_above_hull"] = distribution(
        [
            row["official_e_above_hull"]
            for row in rows
            if row.get("official_hull_status") == "known"
            and row.get("official_e_above_hull") is not None
        ]
    )
    return summary


def paired_boolean(runtime, base_rows, g2_rows, field: str) -> dict[str, Any]:
    base = common.indexed(base_rows, label=f"BASE {field}")
    g2 = common.indexed(g2_rows, label=f"G2 {field}")
    left = []
    right = []
    for index in range(ATTEMPTS):
        if (
            base[index].get("requested_exact_composition_identity")
            != g2[index].get("requested_exact_composition_identity")
        ):
            raise ValueError(f"paired composition changed at {index}")
        left.append(bool(base[index].get(field)))
        right.append(bool(g2[index].get(field)))
    return runtime._exact_mcnemar(left, right)


def render(report: Mapping[str, Any]) -> str:
    lines = [
        "# C3FD + periodic-relation G2 final prospective S.U.N.",
        "",
        "The raw endpoint measures DLM realization; refined S.U.N. measures the complete DLM + model494 system. All rates use the fixed requested denominator 256.",
        "",
        "| Stage | Arm | Direct | N | U | NU | Hull known | Strict S.U.N. | Meta S.U.N. |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["cells"]:
        lines.append(
            f"| {row['stage']} | {row['route']} | {row['direct_joint']}/256 | "
            f"{row['novel']}/256 | {row['unique']}/256 | {row['novel_unique']}/256 | "
            f"{row['hull_known']}/256 | {row['strict_sun']}/256 "
            f"({100*row['strict_sun_rate']:.3f}%) | {row['meta_sun']}/256 "
            f"({100*row['meta_sun_rate']:.3f}%) |"
        )
    lines.extend(["", "## Target evaluation", ""])
    for key, value in report["target_evaluation"].items():
        lines.append(
            f"- {key}: Strict {100*value['strict_rate']:.3f}% "
            f"({'PASS' if value['strict_met'] else 'MISS'}), Meta "
            f"{100*value['meta_rate']:.3f}% "
            f"({'PASS' if value['meta_met'] else 'MISS'})."
        )
    lines.extend(["", "## Paired G2−BASE effects", ""])
    for key, value in report["paired_continuous"].items():
        lines.append(
            f"- {key}: n={value['known']}, mean={value['mean']}, "
            f"95% CI={value['ci95']} (negative favors G2)."
        )
    for key, value in report["paired_binary"].items():
        lines.append(
            f"- {key}: BASE-only={value['control_only']}, "
            f"G2-only={value['candidate_only']}, exact p={value['two_sided_exact_p']}."
        )
    unresolved = report["official_cache"]["query_unresolved"]
    lines.extend(
        [
            "",
            f"Official cache resolved {report['official_cache']['query_resolved']}/"
            f"{report['official_cache']['query_count']} chemsystems; {unresolved} unresolved "
            "rows remain unknown and are never mapped to stable.",
            "",
            "This is one pre-registered stream, so it is not a multi-seed robustness claim. "
            "No failed sample, composition, checkpoint, or result was replaced or selected.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--eval-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    cache = args.official_run.resolve() / "official_mp_cache"
    if not (cache / "completion_SUCCESS").is_file():
        raise RuntimeError("official query is not complete")

    runtime = common.load_runtime(args.eval_runtime.resolve())
    protocol = __import__("protocol")
    phase_diagrams = runtime._phase_diagrams(cache / "official_slim_cache.jsonl")
    unresolved = {
        str(row["chemsys"])
        for row in protocol.read_jsonl(cache / "unresolved_chemsys.jsonl")
    }
    output.mkdir(parents=True)

    cells = []
    rows_by_key = {}
    input_hashes = {
        "official_manifest": common.sha256_file(cache / "completion_manifest.json")
    }
    for stage in STAGES:
        for route in ROUTES:
            root = args.eval_run.resolve() / route / stage
            paths = {
                "labels": root
                / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl",
                "generation": root / "generation/generation.jsonl",
                "direct": root / "evaluation/direct/report.json",
            }
            for name, path in paths.items():
                input_hashes[f"{stage}.{route}.{name}"] = common.sha256_file(path)
            rows, cell_report = runtime._evaluate_cell(
                cell_id=f"{stage}_{route}",
                labels_path=paths["labels"],
                generation_path=paths["generation"],
                direct_path=paths["direct"],
                phase_diagrams=phase_diagrams,
                unresolved=unresolved,
                output_dir=output / f"cells/{stage}/{route}",
            )
            rows = common.attach_requested_identity(
                rows,
                protocol.read_jsonl(paths["generation"]),
                label=f"{stage} {route}",
            )
            rows_by_key[(stage, route)] = rows
            cells.append(summarize_cell(stage, route, cell_report, rows))

    paired_continuous = {}
    for stage in STAGES:
        for field, hull_known in (
            ("chgnet_energy_per_atom", False),
            ("official_e_above_hull", True),
        ):
            name = f"{stage}_{field}"
            delta = common.paired_stream_delta(
                rows_by_key[(stage, "BASE")],
                rows_by_key[(stage, "G2")],
                field=field,
                require_hull_known=hull_known,
            )
            paired_continuous[name] = common.bootstrap(delta, name)

    paired_binary = {}
    for stage in STAGES:
        for field in ("strict_sun", "meta_sun"):
            paired_binary[f"{stage}_{field}"] = paired_boolean(
                runtime,
                rows_by_key[(stage, "BASE")],
                rows_by_key[(stage, "G2")],
                field,
            )

    cells.sort(key=lambda row: (STAGES.index(row["stage"]), ROUTES.index(row["route"])))
    target_evaluation = {
        f"{row['stage']}_{row['route']}": {
            "strict_rate": row["strict_sun_rate"],
            "meta_rate": row["meta_sun_rate"],
            "strict_met": row["strict_sun_rate"] >= 0.10,
            "meta_met": row["meta_sun_rate"] >= 0.50,
        }
        for row in cells
    }
    report = {
        "schema": "c3fd_g2_final_prospective_sun_v1",
        "status": "complete",
        "requested_denominator": ATTEMPTS,
        "stream": STREAM,
        "stream_scope": "single pre-registered stream; not a seed-robust claim",
        "cells": cells,
        "target_evaluation": target_evaluation,
        "paired_continuous": paired_continuous,
        "paired_binary": paired_binary,
        "targets": {"strict": 0.10, "meta": 0.50, "not_result_deletion_gates": True},
        "selection_retry_replacement_rerank": False,
        "inputs": input_hashes,
        "official_cache": common.read_json(cache / "completion_manifest.json"),
    }
    (output / f"{STEM}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / f"{STEM}.md").write_text(render(report), encoding="utf-8")
    with (output / f"{STEM}.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "stage",
                "route",
                "requested",
                "reconstructed",
                "direct_joint",
                "novel",
                "unique",
                "novel_unique",
                "hull_known",
                "hull_unknown",
                "strict_sun",
                "strict_sun_rate",
                "meta_sun",
                "meta_sun_rate",
            ),
        )
        writer.writeheader()
        for row in cells:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    files = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "OUTPUTS.sha256").write_text(
        "".join(
            f"{common.sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    (output / "_SUCCESS").touch()
    print(json.dumps({"status": "complete", "output": str(output)}))


if __name__ == "__main__":
    main()
