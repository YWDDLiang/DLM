#!/usr/bin/env python3
"""Attach frozen C³FD certificates and minimal prompts to D3PO test Plans."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_c3fd_planner_data import compile_row  # noqa: E402
from build_ctv_minimal_spec_data import convert_row  # noqa: E402


FORBIDDEN_OUTCOME_KEYS = frozenset(
    {
        "chgnet_energy_per_atom",
        "e_above_hull",
        "energy_per_atom",
        "official_e_hull",
        "reward",
        "stable",
        "strict_stable",
        "meta_stable",
        "sun",
        "generation_success",
    }
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"non-object row in {path}")
                yield value


def forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in FORBIDDEN_OUTCOME_KEYS:
                paths.append(child_path)
            paths.extend(forbidden_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(forbidden_paths(child, f"{prefix}[{index}]"))
    return paths


def convert_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    converted_rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        forbidden = forbidden_paths(row)
        if forbidden:
            raise ValueError(
                f"test row {row_index} contains outcome fields: {forbidden[:3]}"
            )
        if int(row.get("d3po_test_ordinal", -1)) != row_index:
            raise ValueError("D3PO test ordinals must be exactly 0..255")
        certificate = compile_row(row, row_index)
        if certificate.get("composition_supervision") is not True:
            reason = str(
                certificate.get("compile_error")
                or certificate.get("certificate_class")
                or "uncertified"
            )
            reasons[f"certificate_rejected:{reason}"] += 1
            continue
        converted, reason = convert_row(row, certificate)
        reasons[reason] += 1
        if converted is None:
            continue
        converted["d3po_test_ordinal"] = row_index
        converted["c3fd_certificate_class"] = str(
            certificate.get("certificate_class")
        )
        converted["c3fd_assignment_source"] = certificate.get(
            "assignment_source"
        )
        converted_rows.append(converted)
    return converted_rows, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=256)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    rows = list(iter_jsonl(args.input_jsonl))
    converted, reasons = convert_rows(rows)
    expected = int(args.expected)
    if len(rows) != expected or len(converted) != expected:
        raise RuntimeError(
            f"D3PO certified test count mismatch: source={len(rows)}, "
            f"converted={len(converted)}, expected={expected}, reasons={dict(reasons)}"
        )
    identities = [str(row["reduced_composition_identity"]) for row in converted]
    if len(set(identities)) != expected:
        raise RuntimeError("D3PO certified test reduced identities are not unique")

    args.output_dir.mkdir(parents=True)
    output_path = args.output_dir / "D3PO_TEST_CERTIFIED_PLANS.jsonl"
    with output_path.open("x", encoding="utf-8") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    source_hash = hashlib.sha256(args.input_jsonl.read_bytes()).hexdigest()
    output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    gate = {
        "source_count_exact": len(rows) == expected,
        "converted_count_exact": len(converted) == expected,
        "reduced_identities_unique": len(set(identities)) == expected,
        "minimal_prompts_present": all(bool(row.get("prompt")) for row in converted),
        "outcome_fields_absent": True,
        "selection_unchanged": True,
    }
    gate["d3po_certified_test_authorized"] = all(gate.values())
    manifest = {
        "schema": "h1a2_d3po_certified_test_cohort_v1",
        "source": str(args.input_jsonl.resolve()),
        "source_sha256": source_hash,
        "output": str(output_path.resolve()),
        "output_sha256": output_hash,
        "count": expected,
        "reasons": dict(reasons.most_common()),
        "gate": gate,
        "gpu_jobs_used": 0,
    }
    (args.output_dir / "D3PO_TEST_CERTIFIED_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if gate["d3po_certified_test_authorized"]:
        (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
