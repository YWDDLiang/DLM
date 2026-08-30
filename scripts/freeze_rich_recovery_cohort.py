#!/usr/bin/env python3
"""Freeze one outcome-blind M0/R0/RCF rich-recovery development cohort."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_d3po_pairs import minimal_prompt_from_plan  # noqa: E402
from crystal_dlm.r5_plan_state import build_body_prompt, prototype_key  # noqa: E402


SOFT_TUPLE_FIELDS = (
    "lattice_system",
    "spacegroup_bucket",
    "volume_per_atom_bin",
)
HARD_FIELDS = (
    "N",
    "elements",
    "counts",
    "formula",
    "anion_framework",
    "charge_bucket",
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


def find_plan_state(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("plan_state", "source_plan_state", "r5_plan_state"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    for key in ("source_row", "record", "candidate"):
        value = row.get(key)
        if isinstance(value, Mapping):
            nested = find_plan_state(value)
            if nested is not None:
                return nested
    return None


def canonical_composition(plan: Mapping[str, Any]) -> list[tuple[str, int]]:
    elements = [str(value) for value in (plan.get("elements") or ())]
    counts = [int(value) for value in (plan.get("counts") or ())]
    if not elements or len(elements) != len(counts):
        raise ValueError("plan lacks aligned elements/counts")
    combined: Counter[str] = Counter()
    for element, count in zip(elements, counts):
        if count <= 0:
            raise ValueError("composition count must be positive")
        combined[element] += count
    if int(plan.get("N") or 0) != sum(combined.values()):
        raise ValueError("plan N/count conservation failed")
    return sorted(combined.items())


def exact_identity(plan: Mapping[str, Any]) -> str:
    return "|".join(f"{element}:{count}" for element, count in canonical_composition(plan))


def reduced_identity(plan: Mapping[str, Any]) -> str:
    pairs = canonical_composition(plan)
    divisor = 0
    for _element, count in pairs:
        divisor = math.gcd(divisor, count)
    return "|".join(f"{element}:{count // divisor}" for element, count in pairs)


def chemsys(plan: Mapping[str, Any]) -> str:
    return "-".join(element for element, _count in canonical_composition(plan))


def parse_blocked(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("blocked input must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("blocked input must be NAME=PATH")
    return name, Path(raw_path)


def load_blocked(paths: list[tuple[str, Path]]) -> tuple[set[str], dict[str, Any]]:
    all_exact: set[str] = set()
    report = {}
    for name, path in paths:
        exact: set[str] = set()
        systems: set[str] = set()
        rows = 0
        missing = 0
        for row in iter_jsonl(path):
            rows += 1
            plan = find_plan_state(row)
            if plan is None:
                missing += 1
                continue
            try:
                exact.add(exact_identity(plan))
                systems.add(chemsys(plan))
            except Exception:
                missing += 1
        all_exact.update(exact)
        report[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": rows,
            "identities": len(exact),
            "chemsys": sorted(systems),
            "missing_or_invalid": missing,
        }
    return all_exact, report


def soft_tuple(plan: Mapping[str, Any]) -> tuple[str, str, str]:
    return tuple(str(plan[field]) for field in SOFT_TUPLE_FIELDS)  # type: ignore[return-value]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
    return sha256_file(path)


def freeze(
    source_rows: list[Mapping[str, Any]],
    *,
    count: int,
    blocked_exact: set[str],
    rcf_shift: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    selected = []
    seen: set[str] = set()
    exclusions: Counter[str] = Counter()
    for source_ordinal, row in enumerate(source_rows):
        plan = find_plan_state(row)
        if plan is None:
            exclusions["missing_plan_state"] += 1
            continue
        try:
            exact = exact_identity(plan)
            reduced = reduced_identity(plan)
            system = chemsys(plan)
        except Exception as exc:
            exclusions[f"invalid_plan:{type(exc).__name__}"] += 1
            continue
        if exact in blocked_exact:
            exclusions["blocked_exact_identity"] += 1
            continue
        if exact in seen:
            exclusions["duplicate_exact_identity"] += 1
            continue
        seen.add(exact)
        selected.append(
            {
                "source_ordinal": source_ordinal,
                "source_sample_idx": int(row.get("sample_idx", source_ordinal)),
                "plan_state": deepcopy(dict(plan)),
                "exact_composition_identity": exact,
                "reduced_composition_identity": reduced,
                "chemsys": system,
            }
        )
        if len(selected) == int(count):
            break
    if len(selected) != int(count):
        raise RuntimeError(f"only {len(selected)} eligible rows for requested {count}")
    if int(rcf_shift) <= 0 or int(rcf_shift) >= len(selected):
        raise ValueError("RCF shift must be in 1..count-1")

    source_tuples = [soft_tuple(row["plan_state"]) for row in selected]
    ledger = []
    views = {"M0": [], "R0": [], "RCF": []}
    tuple_value_changes = 0
    for execution_ordinal, item in enumerate(selected):
        plan = deepcopy(item["plan_state"])
        ledger_row = {
            "schema": "h1a2_rich_recovery_ledger_v1",
            "sample_idx": execution_ordinal,
            "source_sample_idx": item["source_sample_idx"],
            "source_ordinal": item["source_ordinal"],
            "exact_composition_identity": item["exact_composition_identity"],
            "reduced_composition_identity": item["reduced_composition_identity"],
            "chemsys": item["chemsys"],
            "plan_state": plan,
        }
        ledger.append(ledger_row)

        minimal_prompt_text, minimal_reason = minimal_prompt_from_plan(plan)
        if minimal_prompt_text is None:
            raise ValueError(f"minimal prompt failed: {minimal_reason}")
        minimal_spec = json.loads(minimal_prompt_text.splitlines()[0])
        views["M0"].append(
            {
                **ledger_row,
                "schema": "h1a2_rich_recovery_view_v1",
                "view": "M0",
                "prompt": minimal_prompt_text,
                "minimal_spec": minimal_spec,
            }
        )
        views["R0"].append(
            {
                **ledger_row,
                "schema": "h1a2_rich_recovery_view_v1",
                "view": "R0",
                "prompt": build_body_prompt(plan).rstrip() + "\n",
            }
        )

        donor_index = (execution_ordinal + int(rcf_shift)) % len(selected)
        if donor_index == execution_ordinal:
            raise RuntimeError("RCF donor must be a different ordinal")
        donor_tuple = source_tuples[donor_index]
        rcf_plan = deepcopy(plan)
        for field, value in zip(SOFT_TUPLE_FIELDS, donor_tuple):
            rcf_plan[field] = value
        rcf_plan["prototype_key"] = prototype_key(rcf_plan)
        tuple_value_changes += int(soft_tuple(rcf_plan) != source_tuples[execution_ordinal])
        for field in HARD_FIELDS:
            if canonical_json(rcf_plan.get(field)) != canonical_json(plan.get(field)):
                raise RuntimeError(f"RCF changed hard field {field}")
        views["RCF"].append(
            {
                **ledger_row,
                "schema": "h1a2_rich_recovery_view_v1",
                "view": "RCF",
                "plan_state": rcf_plan,
                "prompt": build_body_prompt(rcf_plan).rstrip() + "\n",
                "rcf_donor_sample_idx": donor_index,
                "rcf_donor_source_sample_idx": selected[donor_index]["source_sample_idx"],
            }
        )
    if Counter(source_tuples) != Counter(soft_tuple(row["plan_state"]) for row in views["RCF"]):
        raise RuntimeError("RCF did not preserve the soft-tuple multiset")
    report = {
        "requested": int(count),
        "selected": len(selected),
        "exclusions": dict(sorted(exclusions.items())),
        "rcf_shift": int(rcf_shift),
        "rcf_donor_derangement": True,
        "rcf_joint_tuple_multiset_preserved": True,
        "rcf_tuple_value_changed": tuple_value_changes,
        "unique_exact_compositions": len(seen),
        "unique_chemsys": len({row["chemsys"] for row in selected}),
    }
    return ledger, views, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plans", type=Path, required=True)
    parser.add_argument("--blocked", action="append", type=parse_blocked, default=[])
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--rcf-shift", type=int, default=73)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source_sha_before = sha256_file(args.source_plans)
    blocked_exact, blocked_report = load_blocked(list(args.blocked))
    source_rows = list(iter_jsonl(args.source_plans))
    if sha256_file(args.source_plans) != source_sha_before:
        raise RuntimeError("source plans changed while reading")
    ledger, views, selection = freeze(
        source_rows,
        count=int(args.count),
        blocked_exact=blocked_exact,
        rcf_shift=int(args.rcf_shift),
    )
    selected_exact = {row["exact_composition_identity"] for row in ledger}
    if selected_exact & blocked_exact:
        raise RuntimeError("selected cohort overlaps blocked exact identities")
    selected_chemsys = {row["chemsys"] for row in ledger}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    hashes = {
        "ledger.jsonl": write_jsonl(args.output_dir / "ledger.jsonl", ledger),
        **{
            f"{name}.jsonl": write_jsonl(args.output_dir / f"{name}.jsonl", rows)
            for name, rows in views.items()
        },
    }
    manifest = {
        "schema": "h1a2_rich_recovery_cohort_manifest_v1",
        "source": {
            "path": str(args.source_plans.resolve()),
            "sha256": source_sha_before,
            "rows": len(source_rows),
        },
        "blocked": blocked_report,
        "selection": selection,
        "selected_chemsys_overlap": {
            name: len(selected_chemsys & set(report["chemsys"]))
            for name, report in blocked_report.items()
        },
        "output_hashes": hashes,
        "view_prompt_contract": {
            "M0": "ctv_minimal_spec_json",
            "R0": "r5_full_plan_state_json",
            "RCF": "r5_full_plan_state_json_with_joint_soft_tuple_derangement",
            "raw_seven_line_text_fed_to_dlm": False,
        },
        "outcomes_read": False,
        "gpu_jobs_used": 0,
        "gates": {
            "count_exact": len(ledger) == int(args.count),
            "exact_identity_blocked_overlap_zero": not bool(selected_exact & blocked_exact),
            "hard_fields_paired": True,
            "rcf_donor_derangement": selection["rcf_donor_derangement"],
            "rcf_joint_tuple_multiset_preserved": selection[
                "rcf_joint_tuple_multiset_preserved"
            ],
            "all_views_hashed": len(hashes) == 4,
            "outcomes_read_false": True,
        },
    }
    manifest["gates"]["pass"] = all(manifest["gates"].values())
    manifest_path = args.output_dir / "RICH_RECOVERY_COHORT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker = args.output_dir / ("_SUCCESS" if manifest["gates"]["pass"] else "_FAILED")
    marker.write_text(sha256_file(manifest_path) + "\n", encoding="utf-8")
    if not manifest["gates"]["pass"]:
        raise SystemExit(3)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
