#!/usr/bin/env python3
"""Audit and freeze paired minimal/rich DLM interfaces from C3FD-v2.5."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


LATTICE_TO_SPACEGROUP = {
    "triclinic": "sg_001_002",
    "monoclinic": "sg_003_015",
    "orthorhombic": "sg_016_074",
    "tetragonal": "sg_075_142",
    "trigonal": "sg_143_167",
    "hexagonal": "sg_168_194",
    "cubic": "sg_195_230",
}
RICH_FIELDS = (
    "anion_framework",
    "charge_bucket",
    "lattice_system",
    "spacegroup_bucket",
    "volume_per_atom_bin",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


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


def exact_identity(plan: Mapping[str, Any]) -> str:
    elements = [str(value) for value in plan.get("elements") or ()]
    counts = [int(value) for value in plan.get("counts") or ()]
    if len(elements) != len(counts) or not elements:
        raise ValueError("elements/counts changed")
    if int(plan.get("N") or 0) != sum(counts):
        raise ValueError("N/count conservation failed")
    combined = Counter()
    for element, count in zip(elements, counts):
        if count <= 0:
            raise ValueError("count must be positive")
        combined[element] += count
    return "|".join(f"{element}:{combined[element]}" for element in sorted(combined))


def reduced_identity(plan: Mapping[str, Any]) -> str:
    counts = Counter()
    for element, count in zip(plan["elements"], plan["counts"]):
        counts[str(element)] += int(count)
    divisor = 0
    for value in counts.values():
        divisor = math.gcd(divisor, int(value))
    return "|".join(f"{element}:{counts[element] // divisor}" for element in sorted(counts))


def minimal_prompt(plan: Mapping[str, Any]) -> str:
    payload = {
        "N": int(plan["N"]),
        "charge": str(plan["charge_bucket"]),
        "counts": [int(value) for value in plan["counts"]],
        "elements": [str(value) for value in plan["elements"]],
        "family": str(plan["anion_framework"]),
        "formula": str(plan["formula"]),
    }
    return canonical_json(payload) + "\ndynamic_crystal_body:"


def expected_rich_text(plan: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            f"formula: {plan['formula']}",
            f"anion: {plan['anion_framework']}",
            f"charge: {plan['charge_bucket']}",
            f"lattice: {plan['lattice_system']}",
            f"spacegroup: {plan['spacegroup_bucket']}",
            f"volume: {plan['volume_per_atom_bin']}",
            "end: plan",
        )
    )


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    return -sum((value / total) * math.log(value / total) for value in counter.values() if value) if total else 0.0


def tvd(left: Counter[str], right: Counter[str]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left[key] / left_total - right[key] / right_total) for key in keys)


def audit_rows(rows: list[Mapping[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != 1000 or {int(row["sample_idx"]) for row in rows} != set(range(1000)):
        raise ValueError("C3FD source must cover sample_idx0..999")
    distributions = {field: Counter() for field in RICH_FIELDS}
    paired: list[dict[str, Any]] = []
    unknown = Counter()
    for row in sorted(rows, key=lambda value: int(value["sample_idx"])):
        plan = row.get("plan_state")
        if not isinstance(plan, Mapping):
            raise ValueError("row lacks plan_state")
        certificate = plan.get("ccfd_v2_certificate") or {}
        if certificate.get("benchmark_valid") is not True:
            raise ValueError("C3FD composition certificate changed")
        for field in RICH_FIELDS:
            value = str(plan.get(field) or "<UNKNOWN>")
            distributions[field][value] += 1
            unknown[field] += int(value == "<UNKNOWN>")
        lattice = str(plan["lattice_system"])
        if LATTICE_TO_SPACEGROUP.get(lattice) != str(plan["spacegroup_bucket"]):
            raise ValueError("lattice/spacegroup compatibility changed")
        rich = expected_rich_text(plan)
        if str(row.get("plan_text")) != rich:
            raise ValueError("stored rich Plan text changed")
        paired.append(
            {
                "schema": "h1a2_c3fd_paired_dlm_interface_v1",
                "seed": int(seed),
                "sample_idx": int(row["sample_idx"]),
                "exact_composition_identity": exact_identity(plan),
                "reduced_composition_identity": reduced_identity(plan),
                "plan_state": dict(plan),
                "minimal_prompt": minimal_prompt(plan),
                "rich_prompt": rich + "\ndynamic_crystal_body:",
                "rich_plan_text": rich,
            }
        )
    report = {
        "seed": int(seed),
        "rows": len(paired),
        "composition_valid": len(paired),
        "lattice_spacegroup_compatible": len(paired),
        "unknown": dict(unknown),
        "distributions": {field: dict(sorted(values.items())) for field, values in distributions.items()},
        "entropy_nats": {field: entropy(values) for field, values in distributions.items()},
    }
    return paired, report


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed17", type=Path, required=True)
    parser.add_argument("--seed18", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    sources = {17: args.seed17.resolve(), 18: args.seed18.resolve()}
    results = {}
    reports = {}
    for seed, path in sources.items():
        results[seed], reports[seed] = audit_rows(list(iter_jsonl(path)), seed=seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_hashes = {
        str(seed): write_jsonl(args.output_dir / f"paired_seed{seed}.jsonl", rows)
        for seed, rows in results.items()
    }
    distribution_tvd = {
        field: tvd(
            Counter(reports[17]["distributions"][field]),
            Counter(reports[18]["distributions"][field]),
        )
        for field in RICH_FIELDS
    }
    manifest = {
        "schema": "h1a2_c3fd_rich_interface_audit_v1",
        "sources": {
            str(seed): {"path": str(path), "sha256": sha256_file(path)}
            for seed, path in sources.items()
        },
        "reports": {str(seed): report for seed, report in reports.items()},
        "seed_distribution_tvd": distribution_tvd,
        "output_hashes": output_hashes,
        "outcomes_read": False,
        "gpu_jobs_used": 0,
        "gates": {
            "requested1000_both_seeds": all(report["rows"] == 1000 for report in reports.values()),
            "composition_valid_100pct": all(report["composition_valid"] == 1000 for report in reports.values()),
            "rich_unknown_zero": all(not any(report["unknown"].values()) for report in reports.values()),
            "lattice_sg_compatible_100pct": all(report["lattice_spacegroup_compatible"] == 1000 for report in reports.values()),
            "stored_rich_text_exact": True,
            "minimal_rich_composition_identity_paired": True,
        },
    }
    manifest["gates"]["audit_pass"] = all(manifest["gates"].values())
    manifest_path = args.output_dir / "C3FD_RICH_INTERFACE_AUDIT.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").write_text(sha256_file(manifest_path) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
