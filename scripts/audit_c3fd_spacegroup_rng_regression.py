#!/usr/bin/env python3
"""Verify that SG-head sampling preserves the legacy Planner RNG prefix."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


PREFIX_FIELDS = (
    "N",
    "elements",
    "counts",
    "formula",
    "anion_framework",
    "charge_bucket",
    "lattice_system",
    "volume_per_atom_bin",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if raw.strip():
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise TypeError(f"non-object at {path}:{line_number}")
                yield value


def by_index(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    for row in iter_jsonl(path):
        index = int(row["sample_idx"])
        if index in rows:
            raise ValueError(f"duplicate sample_idx {index} in {path}")
        rows[index] = row
    return rows


def prefix_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {field: plan.get(field) for field in PREFIX_FIELDS}


def compare_pair(*, name: str, legacy_dir: Path, corrected_dir: Path) -> dict[str, Any]:
    legacy_plan_path = legacy_dir / "plans_for_dlm.jsonl"
    corrected_plan_path = corrected_dir / "plans_for_dlm.jsonl"
    legacy_raw_path = legacy_dir / "raw_generations.jsonl"
    corrected_raw_path = corrected_dir / "raw_generations.jsonl"
    legacy = by_index(legacy_plan_path)
    corrected = by_index(corrected_plan_path)
    legacy_raw = by_index(legacy_raw_path)
    corrected_raw = by_index(corrected_raw_path)
    indices = sorted(set(legacy) | set(corrected))
    if not indices or set(legacy) != set(corrected):
        raise ValueError(f"{name}: legacy/corrected plan ordinals differ")
    if set(legacy_raw) != set(corrected_raw) or set(legacy_raw) != set(indices):
        raise ValueError(f"{name}: raw ordinal coverage differs")
    prefix_mismatch = []
    trace_mismatch = []
    sg_changed = 0
    sg_transitions: Counter[str] = Counter()
    for index in indices:
        legacy_plan = legacy[index]["plan_state"]
        corrected_plan = corrected[index]["plan_state"]
        if canonical_json(prefix_payload(legacy_plan)) != canonical_json(
            prefix_payload(corrected_plan)
        ):
            prefix_mismatch.append(index)
        legacy_sg = str(legacy_plan["spacegroup_bucket"])
        corrected_sg = str(corrected_plan["spacegroup_bucket"])
        if legacy_sg != corrected_sg:
            sg_changed += 1
        sg_transitions[f"{legacy_sg}->{corrected_sg}"] += 1
        raw_left = legacy_raw[index]
        raw_right = corrected_raw[index]
        for field in ("semantic_trace", "target_proposal", "certificate", "failure"):
            if canonical_json(raw_left.get(field)) != canonical_json(raw_right.get(field)):
                trace_mismatch.append({"sample_idx": index, "field": field})
    result = {
        "name": name,
        "rows": len(indices),
        "sample_idx_min": indices[0],
        "sample_idx_max": indices[-1],
        "prefix_mismatch": prefix_mismatch,
        "semantic_or_certificate_mismatch": trace_mismatch,
        "spacegroup_changed": sg_changed,
        "spacegroup_unchanged": len(indices) - sg_changed,
        "spacegroup_transitions": dict(sorted(sg_transitions.items())),
        "inputs": {
            "legacy_plans_sha256": sha256_file(legacy_plan_path),
            "corrected_plans_sha256": sha256_file(corrected_plan_path),
            "legacy_raw_sha256": sha256_file(legacy_raw_path),
            "corrected_raw_sha256": sha256_file(corrected_raw_path),
        },
    }
    result["gate"] = {
        "same_ordinal_coverage": True,
        "composition_lattice_volume_prefix_exact": not prefix_mismatch,
        "semantic_trace_certificate_exact": not trace_mismatch,
        "corrected_spacegroup_draw_active": sg_changed > 0,
    }
    result["gate"]["pass"] = all(result["gate"].values())
    return result


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# C3FD SG-head RNG-prefix regression",
        "",
        "The corrected sampler adds the SG-head draw only after the legacy",
        "metric-lattice and volume draws. This zero-GPU audit compares real",
        "checkpoint outputs at identical seeds and ordinals.",
        "",
        "| pair | rows | prefix mismatches | trace mismatches | SG changed | pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pair in report["pairs"]:
        lines.append(
            f"| {pair['name']} | {pair['rows']} | {len(pair['prefix_mismatch'])} | "
            f"{len(pair['semantic_or_certificate_mismatch'])} | "
            f"{pair['spacegroup_changed']} | {pair['gate']['pass']} |"
        )
    lines.extend(
        [
            "",
            "A passing result proves RNG-prefix preservation and that the new SG",
            "draw is active. It does not prove better DLM stability; that remains",
            "the matched development-canary estimand.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_pair(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0]:
        raise argparse.ArgumentTypeError("pair must be NAME=LEGACY_DIR,CORRECTED_DIR")
    directories = parts[1].split(",", 1)
    if len(directories) != 2:
        raise argparse.ArgumentTypeError("pair must be NAME=LEGACY_DIR,CORRECTED_DIR")
    return parts[0], Path(directories[0]), Path(directories[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    pairs = [
        compare_pair(name=name, legacy_dir=legacy, corrected_dir=corrected)
        for name, legacy, corrected in args.pair
    ]
    report = {
        "schema": "h1a2_c3fd_spacegroup_rng_regression_v1",
        "pairs": pairs,
        "outcomes_read": False,
        "gpu_jobs_used": 0,
        "gate": {
            "all_pairs_pass": all(pair["gate"]["pass"] for pair in pairs),
            "pair_count": len(pairs),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    json_path = args.output_dir / "C3FD_SPACEGROUP_RNG_REGRESSION.json"
    md_path = args.output_dir / "C3FD_SPACEGROUP_RNG_REGRESSION.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    outputs = {path.name: sha256_file(path) for path in (json_path, md_path)}
    if not report["gate"]["all_pairs_pass"]:
        (args.output_dir / "_FAILED").write_text(canonical_json(outputs) + "\n", encoding="utf-8")
        raise SystemExit(3)
    (args.output_dir / "_SUCCESS").write_text(canonical_json(outputs) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(pairs), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
