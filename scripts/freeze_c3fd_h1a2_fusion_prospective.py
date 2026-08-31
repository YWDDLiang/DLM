#!/usr/bin/env python3
"""Freeze matched H1-A2-full and C3FD-V2 prospective Plan views."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.c3fd_native_plan import (  # noqa: E402
    build_native_inference_prompt,
    native_plan_from_parts,
)


RICH_BUILDER = PROJECT_ROOT / "scripts" / "freeze_rich_recovery_cohort.py"
SPEC = importlib.util.spec_from_file_location("rich_builder_for_fusion", RICH_BUILDER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(RICH_BUILDER)
RICH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RICH
SPEC.loader.exec_module(RICH)


SCHEMA = "c3fd_h1a2_fusion_prospective_cohort_v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def blocked_from_root(root: Path) -> tuple[set[str], list[dict[str, Any]]]:
    blocked: set[str] = set()
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.jsonl")):
        identities: set[str] = set()
        for row in read_jsonl(path):
            plan = RICH.find_plan_state(row)
            if plan is None:
                continue
            try:
                identities.add(RICH.exact_identity(plan))
            except Exception:
                continue
        if identities:
            blocked.update(identities)
            files.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "exact_identities": len(identities),
                }
            )
    return blocked, files


def freeze(
    source_rows: list[Mapping[str, Any]],
    *,
    blocked_exact: set[str],
    count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    v2_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    exclusions: dict[str, int] = {}
    for source_ordinal, row in enumerate(source_rows):
        plan = RICH.find_plan_state(row)
        if plan is None:
            exclusions["missing_plan"] = exclusions.get("missing_plan", 0) + 1
            continue
        try:
            exact = RICH.exact_identity(plan)
            reduced = RICH.reduced_identity(plan)
            system = RICH.chemsys(plan)
            full_plan = RICH.canonicalize_rich_plan(plan)
            full_prompt = RICH.build_body_prompt(full_plan).rstrip() + "\n"
            v2_plan = native_plan_from_parts(plan, plan)
            v2_prompt = build_native_inference_prompt(plan, plan)
        except Exception as exc:
            key = f"invalid:{type(exc).__name__}"
            exclusions[key] = exclusions.get(key, 0) + 1
            continue
        if exact in blocked_exact:
            exclusions["blocked_exact"] = exclusions.get("blocked_exact", 0) + 1
            continue
        if exact in seen:
            exclusions["duplicate_exact"] = exclusions.get("duplicate_exact", 0) + 1
            continue
        sample_idx = len(ledger)
        seen.add(exact)
        common = {
            "sample_idx": sample_idx,
            "source_sample_idx": int(row.get("sample_idx", source_ordinal)),
            "source_ordinal": source_ordinal,
            "exact_composition_identity": exact,
            "reduced_composition_identity": reduced,
            "chemsys": system,
        }
        ledger.append(dict(common))
        full_rows.append(
            {
                **common,
                "schema": "c3fd_h1a2_fusion_full_plan_row_v1",
                "view": "H1A2_FULL",
                "plan_state": full_plan,
                "prompt": full_prompt,
            }
        )
        v2_rows.append(
            {
                **common,
                "schema": "c3fd_h1a2_fusion_v2_plan_row_v1",
                "view": "C3FD_V2",
                "plan_state": v2_plan,
                "prompt": v2_prompt,
            }
        )
        if len(ledger) == count:
            break
    if len(ledger) != count:
        raise RuntimeError(f"only {len(ledger)} eligible prospective rows")
    return ledger, full_rows, v2_rows, {
        "selected": len(ledger),
        "unique_exact": len(seen),
        "unique_chemsys": len({row["chemsys"] for row in ledger}),
        "exclusions": dict(sorted(exclusions.items())),
    }


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plans", type=Path, required=True)
    parser.add_argument("--mp20-train", type=Path, required=True)
    parser.add_argument("--exclude-cohort-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source_sha = sha256_file(args.source_plans)
    mp20_exact: set[str] = set()
    for row in read_jsonl(args.mp20_train):
        plan = RICH.find_plan_state(row)
        if plan is not None:
            try:
                mp20_exact.add(RICH.exact_identity(plan))
            except Exception:
                continue
    cohort_exact, cohort_files = blocked_from_root(args.exclude_cohort_root)
    blocked = mp20_exact | cohort_exact
    source_rows = read_jsonl(args.source_plans)
    if sha256_file(args.source_plans) != source_sha:
        raise RuntimeError("fusion prospective source changed while reading")
    ledger, full_rows, v2_rows, selection = freeze(
        source_rows,
        blocked_exact=blocked,
        count=int(args.count),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "ledger.jsonl", ledger)
    write_jsonl(args.output_dir / "H1A2_FULL.jsonl", full_rows)
    write_jsonl(args.output_dir / "C3FD_V2.jsonl", v2_rows)
    (args.output_dir / "blocked_inputs.json").write_text(
        json.dumps(
            {
                "mp20_train": {
                    "path": str(args.mp20_train.resolve()),
                    "sha256": sha256_file(args.mp20_train),
                    "exact_identities": len(mp20_exact),
                },
                "cohorts": cohort_files,
                "union_exact_identities": len(blocked),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    hashes = {
        name: sha256_file(args.output_dir / name)
        for name in ("ledger.jsonl", "H1A2_FULL.jsonl", "C3FD_V2.jsonl", "blocked_inputs.json")
    }
    selected_exact = {row["exact_composition_identity"] for row in ledger}
    manifest = {
        "schema": SCHEMA,
        "planner_sampling_seed": 20,
        "source_sha256": source_sha,
        "selected": len(ledger),
        "selection": "first_eligible_source_ordinal",
        "selection_report": selection,
        "views": {
            "H1A2_FULL": "canonical legacy H1-A2 rich JSON",
            "C3FD_V2": "current compact C3FD-native V2 JSON",
        },
        "matched_composition_and_order": True,
        "outcomes_read": False,
        "output_sha256": hashes,
        "gates": {
            "fixed256": len(ledger) == 256,
            "sample_idx_contiguous": [row["sample_idx"] for row in ledger]
            == list(range(256)),
            "exact_identity_unique": len(selected_exact) == 256,
            "blocked_overlap_zero": not bool(selected_exact & blocked),
            "matched_view_identity": all(
                full_rows[index]["exact_composition_identity"]
                == v2_rows[index]["exact_composition_identity"]
                for index in range(256)
            ),
            "outcome_blind": True,
        },
    }
    if not all(manifest["gates"].values()):
        raise RuntimeError("fusion prospective gates failed")
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    hashes["manifest.json"] = sha256_file(manifest_path)
    with (args.output_dir / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for name in sorted(hashes):
            handle.write(f"{hashes[name]}  {name}\n")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
