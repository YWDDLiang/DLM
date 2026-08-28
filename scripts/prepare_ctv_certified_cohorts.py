#!/usr/bin/env python3
"""Attach C³FD composition certificates and minimal prompts to frozen CTV sets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_c3fd_planner_data import compile_row  # noqa: E402
from build_ctv_minimal_spec_data import convert_row  # noqa: E402


FILES = (
    "CTV_BRANCH_CANARY_PLANS.jsonl",
    "CTV_BRANCH_TRAIN_PLANS.jsonl",
    "CTV_BRANCH_VAL_PLANS.jsonl",
    "CTV_DLM_L6_PLANS.jsonl",
    "CTV_DLM_L7_PLANS.jsonl",
)
EXPECTED_COUNTS = {
    "CTV_BRANCH_CANARY_PLANS.jsonl": 8,
    "CTV_BRANCH_TRAIN_PLANS.jsonl": 128,
    "CTV_BRANCH_VAL_PLANS.jsonl": 32,
    "CTV_DLM_L6_PLANS.jsonl": 256,
    "CTV_DLM_L7_PLANS.jsonl": 1000,
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"non-object row in {path}")
                yield value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-freeze-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    reports: dict[str, Any] = {}
    all_authorized = True
    for name in FILES:
        source = args.identity_freeze_dir / name
        rows = list(iter_jsonl(source))
        reasons: Counter[str] = Counter()
        converted_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            certificate = compile_row(row, row_index)
            if certificate.get("composition_supervision") is not True:
                reason = str(
                    certificate.get("compile_error")
                    or certificate.get("certificate_class")
                )
                reasons[f"certificate_rejected:{reason}"] += 1
                continue
            converted, reason = convert_row(row, certificate)
            reasons[reason] += 1
            if converted is None:
                continue
            converted["c3fd_certificate_class"] = str(certificate.get("certificate_class"))
            converted["c3fd_assignment_source"] = certificate.get("assignment_source")
            converted["c3fd_valence_witness"] = {
                "nodes": list(certificate.get("nodes") or ()),
                "counts": [int(value) for value in (certificate.get("counts") or ())],
            }
            converted_rows.append(converted)
        expected = int(EXPECTED_COUNTS[name])
        passed = len(rows) == expected and len(converted_rows) == expected
        all_authorized = all_authorized and passed
        output = args.output_dir / name
        with output.open("x", encoding="utf-8") as handle:
            for row in converted_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        reports[name] = {
            "source_rows": len(rows),
            "expected_rows": expected,
            "certified_rows": len(converted_rows),
            "all_rows_certified": passed,
            "reasons": dict(reasons.most_common()),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }

    gate = {
        "all_frozen_counts_preserved": all(
            value["source_rows"] == value["expected_rows"] for value in reports.values()
        ),
        "all_frozen_rows_c3fd_certified": all_authorized,
        "stability_energy_novelty_outcomes_unused": True,
        "minimal_prompts_only": True,
    }
    gate["ctv_certified_cohorts_authorized"] = all(gate.values())
    report = {
        "schema": "h1a2_ctv_certified_cohorts_v1",
        "identity_freeze_dir": str(args.identity_freeze_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "files": reports,
        "gate": gate,
    }
    (args.output_dir / "CTV_CERTIFIED_COHORTS_MANIFEST.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# CTV-DLM certified cohorts",
        "",
        f"Authorized: **{gate['ctv_certified_cohorts_authorized']}**",
        "",
    ]
    for name, value in reports.items():
        lines.append(
            f"- {name}: `{value['certified_rows']}/{value['expected_rows']}` certified"
        )
    lines.extend(
        [
            "",
            "The exact valence witness is a machine sidecar and is never visible to the DLM.",
            "No stability, energy, novelty or generation-success outcome is used.",
        ]
    )
    (args.output_dir / "CTV_CERTIFIED_COHORTS_MANIFEST.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if gate["ctv_certified_cohorts_authorized"]:
        (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
