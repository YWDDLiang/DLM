#!/usr/bin/env python3
"""Finalize the fixed raw-first C3FD-native SFT development canary."""

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
POLICIES = (82017, 82018)
STREAMS = (17, 18)
STAGES = ("raw", "refined")
SCOPES = ("all", "train", "val")


def load_common() -> Any:
    path = ROOT / "scripts" / "finalize_d3po_fixed256_official.py"
    spec = importlib.util.spec_from_file_location("d3po_common_for_native_canary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def attempt_index(row: Mapping[str, Any]) -> int:
    attempt_id = str(row.get("attempt_id") or "")
    try:
        return int(attempt_id.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot recover sample index from {attempt_id!r}") from exc


def cell_root(eval_run: Path, stage: str, stream: int, policy: int) -> Path:
    arm = f"policy{policy}" if stage == "refined" else f"raw_policy{policy}"
    return eval_run / f"stream{stream}" / arm


def scope_indices(ledger: Sequence[Mapping[str, Any]]) -> dict[str, set[int]]:
    indices = {"all": set(), "train": set(), "val": set()}
    for row in ledger:
        sample_idx = int(row["sample_idx"])
        split = str(row["source_split"])
        if split not in {"train", "val"}:
            raise ValueError(f"unknown ledger split {split}")
        indices["all"].add(sample_idx)
        indices[split].add(sample_idx)
    if indices["all"] != set(range(256)):
        raise ValueError("canary ledger is not fixed256 contiguous")
    if len(indices["train"]) != 128 or len(indices["val"]) != 128:
        raise ValueError("canary ledger is not train128/val128")
    return indices


def energy_map(
    rows: Sequence[Mapping[str, Any]], allowed: set[int]
) -> dict[int, float]:
    output: dict[int, float] = {}
    for row in rows:
        sample_idx = int(row["ordinal"])
        if sample_idx not in allowed:
            continue
        value = row.get("chgnet_energy_per_atom")
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            output[sample_idx] = number
    return output


def delta_map(
    control: Mapping[int, float], candidate: Mapping[int, float]
) -> dict[int, float]:
    return {
        sample_idx: float(candidate[sample_idx]) - float(control[sample_idx])
        for sample_idx in sorted(set(control) & set(candidate))
    }


def average_maps(maps: Sequence[Mapping[int, float]]) -> dict[int, float]:
    if not maps:
        return {}
    common = set(maps[0])
    for values in maps[1:]:
        common &= set(values)
    return {
        sample_idx: statistics.fmean(float(values[sample_idx]) for values in maps)
        for sample_idx in sorted(common)
    }


def summarize_cell(
    common: Any,
    *,
    eval_run: Path,
    generation_cell: Mapping[str, Any],
    indices: Mapping[str, set[int]],
    stage: str,
    stream: int,
    policy: int,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    root = cell_root(eval_run, stage, stream, policy)
    labels_path = root / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl"
    direct_metrics_path = root / "evaluation/direct/attempt_metrics.jsonl"
    direct_report_path = root / "evaluation/direct/report.json"
    rows = read_jsonl(labels_path)
    common.indexed_rows(rows, f"native-canary:{stage}:{stream}:{policy}")
    labels = {int(row["ordinal"]): row for row in rows}
    direct_rows = read_jsonl(direct_metrics_path)
    direct = {attempt_index(row): row for row in direct_rows}
    if set(labels) != set(range(256)) or set(direct) != set(range(256)):
        raise ValueError("offline labels or Direct attempts do not cover fixed256")
    missing = {int(value) for value in generation_cell["missing_sample_indices"]}
    summaries: list[dict[str, Any]] = []
    for scope in SCOPES:
        allowed = indices[scope]
        scoped_labels = [labels[sample_idx] for sample_idx in sorted(allowed)]
        scoped_direct = [direct[sample_idx] for sample_idx in sorted(allowed)]
        energies = [
            float(row["chgnet_energy_per_atom"])
            for row in scoped_labels
            if row.get("chgnet_energy_per_atom") is not None
            and math.isfinite(float(row["chgnet_energy_per_atom"]))
        ]
        requested = len(allowed)
        counts = {
            "requested": requested,
            "body_parsed": requested - len(missing & allowed),
            "reconstructed": sum(row.get("reconstructed") is True for row in scoped_labels),
            "direct_comp": sum(row.get("comp_valid") is True for row in scoped_direct),
            "direct_struct": sum(row.get("struct_valid") is True for row in scoped_direct),
            "direct_joint": sum(row.get("valid") is True for row in scoped_direct),
            "novel": sum(row.get("novel") is True for row in scoped_labels),
            "unique": sum(row.get("unique_representative") is True for row in scoped_labels),
            "novel_unique": sum(row.get("novel_unique") is True for row in scoped_labels),
            "energy_known": len(energies),
            "energy_unknown": requested - len(energies),
        }
        summaries.append(
            {
                "stage": stage,
                "stream": stream,
                "policy_seed": policy,
                "scope": scope,
                **counts,
                "body_rate": counts["body_parsed"] / requested,
                "direct_joint_rate": counts["direct_joint"] / requested,
                "novel_unique_rate": counts["novel_unique"] / requested,
                "chgnet_energy_per_atom": common.continuous_distribution(energies),
                "attempt_labels_sha256": sha256_file(labels_path),
                "direct_metrics_sha256": sha256_file(direct_metrics_path),
                "direct_report_sha256": sha256_file(direct_report_path),
            }
        )
    return rows, direct, summaries


def effect_summary(
    common: Any,
    *,
    name: str,
    stage: str,
    scope: str,
    per_stream_maps: Sequence[Mapping[int, float]],
    estimand: str,
) -> dict[str, Any]:
    per_stream = [
        {"stream": stream, **common.delta_summary(values)}
        for stream, values in zip(STREAMS, per_stream_maps)
    ]
    averaged = average_maps(per_stream_maps)
    return {
        "name": name,
        "stage": stage,
        "scope": scope,
        "estimand": estimand,
        "direction": "negative second-minus-first is favorable",
        "per_stream": per_stream,
        "streams_averaged_within_composition": common.cluster_bootstrap_summary(
            averaged,
            label=f"native-canary:{name}:{stage}:{scope}",
            replicates=10_000,
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# C3FD-native SFT canary offline final",
        "",
        "This fixed development canary reports raw evidence before refined evidence.",
        "No official MP query was run and no policy seed was selected.",
        "",
        "## All-composition cell accounting",
        "",
        "| Stage | Stream | Policy | Body | Direct J | Reconstructed | N/U/NU | Energy K/U | Mean E |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        if cell["scope"] != "all":
            continue
        energy = cell["chgnet_energy_per_atom"]
        lines.append(
            f"| {cell['stage']} | {cell['stream']} | {cell['policy_seed']} | "
            f"{cell['body_parsed']} | {cell['direct_joint']} | {cell['reconstructed']} | "
            f"{cell['novel']}/{cell['unique']}/{cell['novel_unique']} | "
            f"{cell['energy_known']}/{cell['energy_unknown']} | {energy['mean']:.6f} |"
        )
    lines.extend(["", "## Policy aggregates", ""])
    for aggregate in report["policy_aggregates"]:
        if aggregate["scope"] != "all":
            continue
        lines.append(
            f"- {aggregate['stage']} policy{aggregate['policy_seed']}: body "
            f"{aggregate['body_parsed']}/{aggregate['requested']}, Direct J "
            f"{aggregate['direct_joint']}/{aggregate['requested']}, NU "
            f"{aggregate['novel_unique']}/{aggregate['requested']}."
        )
    lines.extend(["", "## Continuous paired effects", ""])
    for effect in report["continuous_effects"]:
        if effect["scope"] != "all":
            continue
        pooled = effect["streams_averaged_within_composition"]
        bootstrap = pooled["bootstrap"]
        lines.extend(
            [
                f"### {effect['name']} ({effect['stage']})",
                "",
                f"- estimand: {effect['estimand']};",
                f"- compositions observed: {pooled['compositions_observed']}/256;",
                f"- mean delta: {pooled['mean_delta']:.6f} eV/atom;",
                f"- fraction lower: {pooled['fraction_lower']:.4f};",
                f"- paired composition-bootstrap CI: [{bootstrap['ci95_lower']:.6f}, {bootstrap['ci95_upper']:.6f}].",
                "",
            ]
        )
    decision = report["decision"]
    lines.extend(
        [
            "## Decision",
            "",
            f"- execution recovered: **{decision['execution_recovered']}**;",
            f"- raw geometry remains insufficient: **{decision['raw_geometry_insufficient']}**;",
            f"- model494 rescue is large: **{decision['refiner_rescue_large']}**;",
            f"- next route: **{decision['next_route']}**.",
            "",
            "The canary supports Planner-interface adaptation but does not support a claim",
            "that teacher-only CE learned raw stability. The next route uses a fresh MP20-train",
            "on-policy pool; the historical 3,614 candidates remain excluded from formal training.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--generation-final", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--elapsed-seconds", type=int, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not (args.eval_run / "_OFFLINE_SUCCESS").is_file():
        raise RuntimeError("offline success marker missing")
    common = load_common()
    ledger = read_jsonl(args.ledger)
    indices = scope_indices(ledger)
    generation = read_json(args.generation_final)
    generation_cells = {
        (int(row["stream"]), int(row["policy_seed"])): row
        for row in generation["cells"]
    }
    rows_by_cell: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    cells: list[dict[str, Any]] = []
    for stage in STAGES:
        for stream in STREAMS:
            for policy in POLICIES:
                rows, _direct, summaries = summarize_cell(
                    common,
                    eval_run=args.eval_run,
                    generation_cell=generation_cells[(stream, policy)],
                    indices=indices,
                    stage=stage,
                    stream=stream,
                    policy=policy,
                )
                rows_by_cell[(stage, stream, policy)] = rows
                cells.extend(summaries)
    policy_aggregates: list[dict[str, Any]] = []
    for stage in STAGES:
        for policy in POLICIES:
            for scope in SCOPES:
                selected = [
                    cell
                    for cell in cells
                    if cell["stage"] == stage
                    and cell["policy_seed"] == policy
                    and cell["scope"] == scope
                ]
                policy_aggregates.append(
                    {
                        "stage": stage,
                        "policy_seed": policy,
                        "scope": scope,
                        **{
                            key: sum(int(cell[key]) for cell in selected)
                            for key in (
                                "requested",
                                "body_parsed",
                                "reconstructed",
                                "direct_comp",
                                "direct_struct",
                                "direct_joint",
                                "novel",
                                "unique",
                                "novel_unique",
                                "energy_known",
                                "energy_unknown",
                            )
                        },
                    }
                )
    effects: list[dict[str, Any]] = []
    for stage in STAGES:
        for scope in SCOPES:
            allowed = indices[scope]
            effects.append(
                effect_summary(
                    common,
                    name="policy_seed_sensitivity_82018_minus_82017",
                    stage=stage,
                    scope=scope,
                    per_stream_maps=[
                        delta_map(
                            energy_map(rows_by_cell[(stage, stream, 82017)], allowed),
                            energy_map(rows_by_cell[(stage, stream, 82018)], allowed),
                        )
                        for stream in STREAMS
                    ],
                    estimand="independent policy82018 minus policy82017; diagnostic only, never seed selection",
                )
            )
    for policy in POLICIES:
        for scope in SCOPES:
            allowed = indices[scope]
            effects.append(
                effect_summary(
                    common,
                    name=f"model494_refined_minus_raw_policy{policy}",
                    stage="raw_to_refined",
                    scope=scope,
                    per_stream_maps=[
                        delta_map(
                            energy_map(rows_by_cell[("raw", stream, policy)], allowed),
                            energy_map(rows_by_cell[("refined", stream, policy)], allowed),
                        )
                        for stream in STREAMS
                    ],
                    estimand=f"model494-refined minus raw CHGNet energy for policy{policy}",
                )
            )
    aggregate_lookup = {
        (row["stage"], row["policy_seed"], row["scope"]): row
        for row in policy_aggregates
    }
    raw_rates = {
        str(policy): aggregate_lookup[("raw", policy, "all")]["direct_joint"] / 512
        for policy in POLICIES
    }
    refined_rates = {
        str(policy): aggregate_lookup[("refined", policy, "all")]["direct_joint"] / 512
        for policy in POLICIES
    }
    parse_rates = {
        str(policy): aggregate_lookup[("raw", policy, "all")]["body_parsed"] / 512
        for policy in POLICIES
    }
    decision = {
        "execution_recovered": all(rate >= 0.95 for rate in parse_rates.values()),
        "raw_geometry_insufficient": all(rate < 0.5 for rate in raw_rates.values()),
        "refiner_rescue_large": all(
            refined_rates[str(policy)] - raw_rates[str(policy)] > 0.4
            for policy in POLICIES
        ),
        "next_route": "fresh_mp20_train_on_policy_same_composition_alignment",
        "historical_3614_formal_training_allowed": False,
        "seed_or_checkpoint_selection": False,
    }
    report = {
        "schema": "c3fd_native_sft_canary_offline_final_v1",
        "eval_run": str(args.eval_run.resolve()),
        "generation_final": str(args.generation_final.resolve()),
        "ledger": str(args.ledger.resolve()),
        "cells": cells,
        "policy_aggregates": policy_aggregates,
        "continuous_effects": effects,
        "headline": {
            "body_parse_rate": parse_rates,
            "raw_direct_joint_rate": raw_rates,
            "refined_direct_joint_rate": refined_rates,
        },
        "decision": decision,
        "resources": {
            "elapsed_seconds": int(args.elapsed_seconds),
            "gpus": 4,
            "observed_a800_hours": 4 * int(args.elapsed_seconds) / 3600,
            "kill_ceiling_a800_hours": 32.0,
        },
        "official_query_run": False,
        "fixed_denominator": 256,
        "retry_replacement_rerank": False,
        "gate": {
            "eight_cells": len(
                {
                    (cell["stage"], cell["stream"], cell["policy_seed"])
                    for cell in cells
                }
            )
            == 8,
            "all_requested_accounted": all(
                cell["energy_known"] + cell["energy_unknown"] == cell["requested"]
                for cell in cells
            ),
            "all_direct_bounded": all(
                0 <= cell["direct_joint"] <= cell["requested"] for cell in cells
            ),
            "split_accounting": all(
                cell["requested"] == (256 if cell["scope"] == "all" else 128)
                for cell in cells
            ),
        },
    }
    report["gate"]["pass"] = all(report["gate"].values())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    json_path = args.output_dir / "C3FD_NATIVE_SFT_CANARY_OFFLINE_FINAL.json"
    md_path = args.output_dir / "C3FD_NATIVE_SFT_CANARY_OFFLINE_FINAL.md"
    csv_path = args.output_dir / "C3FD_NATIVE_SFT_CANARY_OFFLINE_CELLS.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        fields = (
            "stage",
            "stream",
            "policy_seed",
            "scope",
            "requested",
            "body_parsed",
            "reconstructed",
            "direct_comp",
            "direct_struct",
            "direct_joint",
            "novel",
            "unique",
            "novel_unique",
            "energy_known",
            "energy_unknown",
            "mean_energy",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    **{key: cell[key] for key in fields if key in cell},
                    "mean_energy": cell["chgnet_energy_per_atom"]["mean"],
                }
            )
    hashes = {path.name: sha256_file(path) for path in (json_path, md_path, csv_path)}
    (args.output_dir / "OUTPUTS.sha256.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    marker = args.output_dir / ("_SUCCESS" if report["gate"]["pass"] else "_FAILED")
    marker.write_text(sha256_file(json_path) + "\n", encoding="utf-8")
    if not report["gate"]["pass"]:
        raise SystemExit(3)
    print(json.dumps(report["headline"], indent=2, sort_keys=True))
    print(json.dumps(report["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
