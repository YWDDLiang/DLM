#!/usr/bin/env python3
"""Finalize the 12-cell raw/refined rich recovery development canary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("M0", "RCF", "R0")
STREAMS = (17, 18)
STAGES = ("raw", "refined")
COMPARISONS = (
    ("R0_minus_RCF", "RCF", "R0", "soft structural tuple alignment"),
    ("R0_minus_M0", "M0", "R0", "rich package recovery"),
    ("RCF_minus_M0", "M0", "RCF", "rich-trained checkpoint/package without aligned tuple"),
)


def load_common():
    path = ROOT / "scripts" / "finalize_d3po_fixed256_official.py"
    spec = importlib.util.spec_from_file_location("d3po_common_for_rich", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules while the
    # imported module is executing, so register this dynamic module first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def cell_root(eval_run: Path, stage: str, stream: int, arm: str) -> Path:
    directory = arm if stage == "refined" else f"raw_{arm}"
    return eval_run / f"stream{stream}" / directory


def summarize_cell(
    common: Any,
    eval_run: Path,
    generation_cells: Mapping[tuple[int, str], Mapping[str, Any]],
    stage: str,
    stream: int,
    arm: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = cell_root(eval_run, stage, stream, arm)
    labels_path = root / "evaluation" / "full_reconstructed" / "attempt_labels_preofficial.jsonl"
    direct_path = root / "evaluation" / "direct" / "report.json"
    rows = common.read_jsonl(labels_path)
    common.indexed_rows(rows, f"{stage}:{stream}:{arm}")
    direct = read_json(direct_path)
    energies = [
        float(row["chgnet_energy_per_atom"])
        for row in rows
        if row.get("chgnet_energy_per_atom") is not None
        and math.isfinite(float(row["chgnet_energy_per_atom"]))
    ]
    generation = generation_cells[(stream, arm)]
    counts = {
        "requested": 256,
        "body_parsed": int(generation["parsed"]),
        "reconstructed": sum(row.get("reconstructed") is True for row in rows),
        "direct_comp": int(direct["comp_valid_count"]),
        "direct_struct": int(direct["struct_valid_count"]),
        "direct_joint": int(direct["valid_count"]),
        "novel": sum(row.get("novel") is True for row in rows),
        "unique": sum(row.get("unique_representative") is True for row in rows),
        "novel_unique": sum(row.get("novel_unique") is True for row in rows),
        "energy_known": len(energies),
        "energy_unknown": 256 - len(energies),
    }
    summary = {
        "stage": stage,
        "stream": int(stream),
        "arm": arm,
        **counts,
        "body_rate": counts["body_parsed"] / 256,
        "direct_joint_rate": counts["direct_joint"] / 256,
        "novel_unique_rate": counts["novel_unique"] / 256,
        "chgnet_energy_per_atom": common.continuous_distribution(energies),
        "attempt_labels_sha256": sha256_file(labels_path),
        "direct_report_sha256": sha256_file(direct_path),
    }
    return rows, summary


def comparison_endpoint(
    common: Any,
    rows_by_cell: Mapping[tuple[str, int, str], Sequence[Mapping[str, Any]]],
    *,
    stage: str,
    name: str,
    control: str,
    candidate: str,
    estimand: str,
) -> dict[str, Any]:
    per_stream = []
    maps = []
    for stream in STREAMS:
        values = common.paired_continuous_deltas(
            rows_by_cell[(stage, stream, control)],
            rows_by_cell[(stage, stream, candidate)],
            field="chgnet_energy_per_atom",
            require_official_known=False,
            label=f"rich:{stage}:{name}:stream{stream}",
        )
        maps.append(values)
        per_stream.append({"stream": stream, **common.delta_summary(values)})
    averaged = common.average_delta_maps(maps)
    return {
        "name": name,
        "stage": stage,
        "control": control,
        "candidate": candidate,
        "estimand": estimand,
        "direction": "negative candidate-minus-control is favorable",
        "per_stream": per_stream,
        "streams_averaged_within_composition": common.cluster_bootstrap_summary(
            averaged,
            label=f"rich:{stage}:{name}:stream-average",
            replicates=10_000,
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Rich recovery canary offline final",
        "",
        "Continuous CHGNet energy is reported before threshold metrics. This is",
        "a development canary; no official MP query was run.",
        "",
        "## Cell accounting",
        "",
        "| stage | stream | arm | body | reconstructed | Direct J | N/U/NU | energy K/U | mean energy |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        energy = cell["chgnet_energy_per_atom"]
        lines.append(
            f"| {cell['stage']} | {cell['stream']} | {cell['arm']} | "
            f"{cell['body_parsed']} | {cell['reconstructed']} | {cell['direct_joint']} | "
            f"{cell['novel']}/{cell['unique']}/{cell['novel_unique']} | "
            f"{cell['energy_known']}/{cell['energy_unknown']} | {energy['mean']:.6f} |"
        )
    lines.extend(["", "## Paired continuous effects", ""])
    for endpoint in report["continuous_effects"]:
        pooled = endpoint["streams_averaged_within_composition"]
        bootstrap = pooled["bootstrap"]
        lines.extend(
            [
                f"### {endpoint['stage']} — {endpoint['name']}",
                "",
                f"- estimand: {endpoint['estimand']};",
                f"- observed compositions: {pooled['compositions_observed']}/256;",
                f"- mean delta: {pooled['mean_delta']:.6f} eV/atom;",
                f"- median delta: {pooled['median_delta']:.6f} eV/atom;",
                f"- fraction lower: {pooled['fraction_lower']:.4f};",
                f"- paired composition-bootstrap 95% CI: [{bootstrap['ci95_lower']:.6f}, {bootstrap['ci95_upper']:.6f}].",
                "",
            ]
        )
    resource = report["resources"]
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "- `R0-RCF` isolates alignment of the three predicted soft fields on",
            "  the fixed rich-trained DLM package.",
            "- `R0-M0` is a package recovery comparison and is not a single-factor",
            "  Planner causal effect.",
            "- A refined-only benefit is refiner-mediated and is not evidence that",
            "  the raw DLM learned stability.",
            "- Historical H1-A2 references require official hull and are not",
            "  directly recomputed in this offline canary.",
            "",
            f"Observed evaluation use: {resource['observed_gpu_hours']:.4f} A800-hours; scheduler kill ceiling: {resource['kill_ceiling_gpu_hours']:.1f}.",
            "",
            f"Overall accounting pass: **{report['gate']['pass']}**.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--generation-final", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--elapsed-seconds", type=int, default=7134)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not (args.eval_run / "_OFFLINE_SUCCESS").is_file():
        raise RuntimeError("offline success marker missing")
    common = load_common()
    generation = read_json(args.generation_final / "RICH_RECOVERY_GENERATION_FINAL.json")
    generation_cells = {
        (int(row["stream"]), str(row["arm"])): row for row in generation["cells"]
    }
    rows_by_cell = {}
    cells = []
    for stage in STAGES:
        for stream in STREAMS:
            for arm in ARMS:
                rows, summary = summarize_cell(
                    common, args.eval_run, generation_cells, stage, stream, arm
                )
                rows_by_cell[(stage, stream, arm)] = rows
                cells.append(summary)
    effects = [
        comparison_endpoint(
            common,
            rows_by_cell,
            stage=stage,
            name=name,
            control=control,
            candidate=candidate,
            estimand=estimand,
        )
        for stage in STAGES
        for name, control, candidate, estimand in COMPARISONS
    ]
    report = {
        "schema": "h1a2_rich_recovery_offline_final_v1",
        "eval_run": str(args.eval_run.resolve()),
        "generation_final": str(args.generation_final.resolve()),
        "cells": cells,
        "continuous_effects": effects,
        "resources": {
            "elapsed_seconds": int(args.elapsed_seconds),
            "gpus": 6,
            "observed_gpu_hours": 6 * int(args.elapsed_seconds) / 3600,
            "kill_ceiling_gpu_hours": 6 * 10,
        },
        "official_query_run": False,
        "fixed_denominator": 256,
        "retry_replacement_rerank": False,
        "gate": {
            "twelve_cells": len(cells) == 12,
            "all_requested_256": all(cell["requested"] == 256 for cell in cells),
            "all_direct_reports_ok": all(cell["direct_joint"] <= 256 for cell in cells),
            "all_energy_accounted": all(
                cell["energy_known"] + cell["energy_unknown"] == 256 for cell in cells
            ),
        },
    }
    report["gate"]["pass"] = all(report["gate"].values())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    json_path = args.output_dir / "RICH_RECOVERY_OFFLINE_FINAL.json"
    md_path = args.output_dir / "RICH_RECOVERY_OFFLINE_FINAL.md"
    csv_path = args.output_dir / "RICH_RECOVERY_OFFLINE_FINAL.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "stage", "stream", "arm", "body_parsed", "reconstructed",
                "direct_comp", "direct_struct", "direct_joint", "novel",
                "unique", "novel_unique", "energy_known", "energy_unknown",
                "mean_energy",
            ),
        )
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    **{key: cell[key] for key in writer.fieldnames if key in cell},
                    "mean_energy": cell["chgnet_energy_per_atom"]["mean"],
                }
            )
    outputs = {path.name: sha256_file(path) for path in (json_path, md_path, csv_path)}
    (args.output_dir / "OUTPUTS.sha256.json").write_text(
        json.dumps(outputs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    marker = args.output_dir / ("_SUCCESS" if report["gate"]["pass"] else "_FAILED")
    marker.write_text(sha256_file(json_path) + "\n", encoding="utf-8")
    if not report["gate"]["pass"]:
        raise SystemExit(3)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
