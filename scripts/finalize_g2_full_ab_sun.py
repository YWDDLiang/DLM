#!/usr/bin/env python3
"""Finalize the matched full-epoch G2 A/B development endpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import finalize_c3fd_g2_final_sun as base
import finalize_c3fd_llama_prospective_sun as common


ATTEMPTS = 256
STREAM = 17
ROUTES = ("A", "B")
STAGES = ("raw", "refined")
STEM = "G2_FULL_EPOCH_AB_SUN"
METHOD_NAMES = {
    "A": "G2-PBC-R",
    "B": "G2-PBC-RU",
}


def paired_boolean(runtime, a_rows, b_rows, field: str) -> dict[str, Any]:
    a = common.indexed(a_rows, label=f"A {field}")
    b = common.indexed(b_rows, label=f"B {field}")
    left = []
    right = []
    for index in range(ATTEMPTS):
        if (
            a[index].get("requested_exact_composition_identity")
            != b[index].get("requested_exact_composition_identity")
        ):
            raise ValueError(f"paired composition changed at {index}")
        left.append(bool(a[index].get(field)))
        right.append(bool(b[index].get(field)))
    return runtime._exact_mcnemar(left, right)


def summarize_cell(stage: str, route: str, report: Mapping[str, Any], rows):
    summary = common.summarize_cell("development", stage, STREAM, route, report)
    summary["method_name"] = METHOD_NAMES[route]
    summary["chgnet_energy"] = base.distribution(
        [
            row["chgnet_energy_per_atom"]
            for row in rows
            if row.get("chgnet_relaxation_known") is True
            and row.get("chgnet_energy_per_atom") is not None
        ]
    )
    summary["official_e_above_hull"] = base.distribution(
        [
            row["official_e_above_hull"]
            for row in rows
            if row.get("official_hull_status") == "known"
            and row.get("official_e_above_hull") is not None
        ]
    )
    return summary


def promotion_decision(cells, paired_continuous) -> dict[str, Any]:
    by_key = {(row["stage"], row["route"]): row for row in cells}
    a_raw = by_key[("raw", "A")]
    b_raw = by_key[("raw", "B")]
    energy = paired_continuous["raw_chgnet_energy_per_atom"]
    energy_ci = energy["ci95"]
    direct_rule = (
        b_raw["reconstructed"] >= a_raw["reconstructed"]
        and b_raw["direct_joint"] >= a_raw["direct_joint"] + 8
    )
    energy_rule = (
        energy_ci[0] is not None
        and energy_ci[1] is not None
        and float(energy_ci[1]) < 0.0
    )
    promoted = "B" if direct_rule or energy_rule else "A"
    return {
        "registered_rule": (
            "B only if body is no lower and raw Direct is at least A+8, "
            "or the paired raw-energy 95% CI is entirely below zero; otherwise A"
        ),
        "direct_rule_met": direct_rule,
        "raw_energy_rule_met": energy_rule,
        "promoted_route": promoted,
        "promoted_method": METHOD_NAMES[promoted],
        "not_a_seed_or_checkpoint_selection": True,
    }


def render(report: Mapping[str, Any]) -> str:
    lines = [
        "# G2 full-epoch A/B cached-official endpoint",
        "",
        "This is a matched post-outcome development comparison on one frozen "
        "Plan/noise stream. All rates use the fixed requested denominator 256.",
        "",
        "| Stage | Arm | Method | Body | Direct | N/U/NU | Hull known | Strict S.U.N. | Meta S.U.N. |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["cells"]:
        lines.append(
            f"| {row['stage']} | {row['route']} | {row['method_name']} | "
            f"{row['reconstructed']}/256 | {row['direct_joint']}/256 | "
            f"{row['novel']}/{row['unique']}/{row['novel_unique']} | "
            f"{row['hull_known']}/256 | {row['strict_sun']}/256 "
            f"({100*row['strict_sun_rate']:.3f}%) | {row['meta_sun']}/256 "
            f"({100*row['meta_sun_rate']:.3f}%) |"
        )
    lines.extend(["", "## Paired B−A continuous effects", ""])
    for key, value in report["paired_continuous"].items():
        lines.append(
            f"- {key}: n={value['known']}, mean={value['mean']}, "
            f"95% CI={value['ci95']} (negative favors B)."
        )
    lines.extend(["", "## Paired B−A binary effects", ""])
    for key, value in report["paired_binary"].items():
        lines.append(
            f"- {key}: A-only={value['control_only']}, "
            f"B-only={value['candidate_only']}, exact p={value['two_sided_exact_p']}."
        )
    decision = report["promotion_decision"]
    lines.extend(
        [
            "",
            "## Registered method decision",
            "",
            f"Promoted method: **{decision['promoted_method']} ({decision['promoted_route']})**.",
            f"Direct rule met: `{decision['direct_rule_met']}`; raw-energy rule met: "
            f"`{decision['raw_energy_rule_met']}`.",
            "",
            "No Plan, sample, failed row, seed, checkpoint, or evaluation result was replaced.",
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
        raise RuntimeError("official cache is not complete")

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
        for field, require_hull_known in (
            ("chgnet_energy_per_atom", False),
            ("official_e_above_hull", True),
        ):
            name = f"{stage}_{field}"
            delta = common.paired_stream_delta(
                rows_by_key[(stage, "A")],
                rows_by_key[(stage, "B")],
                field=field,
                require_hull_known=require_hull_known,
            )
            paired_continuous[name] = common.bootstrap(delta, name)

    paired_binary = {}
    for stage in STAGES:
        for field in ("strict_sun", "meta_sun"):
            paired_binary[f"{stage}_{field}"] = paired_boolean(
                runtime,
                rows_by_key[(stage, "A")],
                rows_by_key[(stage, "B")],
                field,
            )

    cells.sort(key=lambda row: (STAGES.index(row["stage"]), ROUTES.index(row["route"])))
    report = {
        "schema": "g2_full_epoch_ab_cached_official_v1",
        "status": "complete",
        "requested_denominator": ATTEMPTS,
        "stream": STREAM,
        "claim_scope": "matched post-outcome development; not seed robustness",
        "methods": METHOD_NAMES,
        "cells": cells,
        "paired_continuous": paired_continuous,
        "paired_binary": paired_binary,
        "promotion_decision": promotion_decision(cells, paired_continuous),
        "inputs": input_hashes,
        "official_cache": common.read_json(cache / "completion_manifest.json"),
        "selection_retry_replacement_rerank": False,
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
                "method_name",
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
