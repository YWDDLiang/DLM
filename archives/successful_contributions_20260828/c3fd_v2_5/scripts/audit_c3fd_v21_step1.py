#!/usr/bin/env python3
"""Step-1 audit for C³FD-v2.1 proposal labels and semantic ledgers."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"non-object row in {path}")
                yield value


def stratum(row: Mapping[str, Any]) -> tuple[int, int, int] | None:
    proposal = row.get("proposal_targets") or {}
    values = (proposal.get("family"), proposal.get("N"), proposal.get("arity"))
    if any(value is None for value in values):
        return None
    family, n_value, arity = (int(value) for value in values)
    if n_value < 1 or n_value > 20 or arity < 1 or arity > 7:
        return None
    return family, n_value, arity


def validate_row(row: Mapping[str, Any]) -> dict[str, bool]:
    proposal = stratum(row)
    proposal_exact = bool(row.get("proposal_supervision") is True and proposal is not None)
    plan = row.get("plan_state") or {}
    if proposal is not None and isinstance(plan, Mapping):
        _family, n_value, arity = proposal
        proposal_exact = bool(
            proposal_exact
            and int(plan.get("N") or 0) == n_value
            and len(plan.get("elements") or ()) == arity
        )
    composition = row.get("composition_supervision") is True
    ledger = list(row.get("ledger_steps") or ())
    ledger_exact = not composition
    if composition:
        arity = int(proposal[2]) if proposal is not None else -1
        ledger_exact = bool(
            len(ledger) == arity + 2
            and ledger[0] == ledger[1]
            and int(ledger[0].get("remaining_atoms", -1)) == int(proposal[1])
            and int(ledger[0].get("net_charge", 1)) == 0
            and int(ledger[0].get("remaining_species", -1)) == arity
            and int(ledger[-1].get("remaining_atoms", -1)) == 0
            and int(ledger[-1].get("net_charge", 1)) == 0
            and int(ledger[-1].get("remaining_species", -1)) == 0
            and str(ledger[-1].get("branch")) in {"ionic", "alloy"}
        )
    return {
        "proposal_exact": proposal_exact,
        "ledger_exact": ledger_exact,
    }


def summarize(
    name: str,
    path: Path,
    *,
    reachable: set[tuple[int, int, int]],
) -> dict[str, Any]:
    rows = proposal = composition = proposal_exact = ledger_exact = reachable_rows = 0
    weighted_total = weighted_reachable = 0.0
    strata: Counter[str] = Counter()
    families: Counter[str] = Counter()
    n_values: Counter[str] = Counter()
    arities: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    for row in iter_jsonl(path):
        rows += 1
        weight = float(row.get("sample_weight", 1.0) or 1.0)
        weighted_total += weight
        flags = validate_row(row)
        proposal_exact += int(flags["proposal_exact"])
        ledger_exact += int(flags["ledger_exact"])
        composition += int(row.get("composition_supervision") is True)
        key = stratum(row)
        if key is None:
            failures["missing_or_invalid_proposal"] += 1
            continue
        proposal += 1
        label = "|".join(str(value) for value in key)
        strata[label] += 1
        families[str(key[0])] += 1
        n_values[str(key[1])] += 1
        arities[str(key[2])] += 1
        if key in reachable:
            reachable_rows += 1
            weighted_reachable += weight
    def rate(value: int, denominator: int = rows) -> float:
        return 0.0 if denominator == 0 else value / denominator
    return {
        "name": name,
        "path": str(path.resolve()),
        "rows": rows,
        "counts": {
            "proposal": proposal,
            "composition_supervision": composition,
            "proposal_exact": proposal_exact,
            "ledger_exact": ledger_exact,
            "reachable_rows": reachable_rows,
        },
        "rates": {
            "proposal_coverage": rate(proposal),
            "proposal_exact": rate(proposal_exact),
            "ledger_exact": rate(ledger_exact),
            "reachable_row_mass": rate(reachable_rows),
            "reachable_weighted_mass": (
                0.0 if weighted_total == 0 else weighted_reachable / weighted_total
            ),
        },
        "strata": dict(sorted(strata.items())),
        "family": dict(sorted(families.items())),
        "N": dict(sorted(n_values.items(), key=lambda item: int(item[0]))),
        "arity": dict(sorted(arities.items(), key=lambda item: int(item[0]))),
        "failures": dict(failures.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    train_rows = list(iter_jsonl(args.data_dir / "train.jsonl"))
    reachable = {
        key
        for row in train_rows
        if row.get("composition_supervision") is True
        for key in [stratum(row)]
        if key is not None
    }
    results = []
    for split in ("train", "val", "test"):
        path = args.data_dir / f"{split}.jsonl"
        if path.is_file():
            results.append(summarize(split, path, reachable=reachable))
    by_name = {row["name"]: row for row in results}
    gate = {
        "train_val_present": "train" in by_name and "val" in by_name,
        "all_rows_have_exact_proposal_labels": all(
            row["rates"]["proposal_exact"] == 1.0 for row in results
        ),
        "all_rows_have_exact_or_inapplicable_ledger": all(
            row["rates"]["ledger_exact"] == 1.0 for row in results
        ),
        "train_reachable_weighted_mass_at_least_99pct": by_name.get("train", {}).get("rates", {}).get("reachable_weighted_mass", 0.0) >= 0.99,
        "val_reachable_weighted_mass_at_least_99pct": by_name.get("val", {}).get("rates", {}).get("reachable_weighted_mass", 0.0) >= 0.99,
    }
    gate["step1_pass"] = all(gate.values())
    payload = {
        "schema": "h1a2_c3fd_v21_step1_data_audit_v1",
        "reachable_strata_from_train_benchmark": len(reachable),
        "datasets": results,
        "gate": gate,
    }
    args.output_dir.mkdir(parents=True)
    stem = "C3FD_V21_STEP1_DATA_AUDIT"
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["name", "rows", "proposal_exact", "ledger_exact", "reachable_weighted_mass"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "name": row["name"],
                    "rows": row["rows"],
                    **{key: row["rates"][key] for key in fields if key in row["rates"]},
                }
            )
    lines = [
        "# C³FD-v2.1 Step-1 data audit",
        "",
        f"Step 1 pass: **{gate['step1_pass']}**",
        "",
        "| Split | Rows | Proposal exact | Ledger exact | Reachable weighted mass |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results:
        rates = row["rates"]
        lines.append(
            f"| {row['name']} | {row['rows']} | {rates['proposal_exact']:.2%} | "
            f"{rates['ledger_exact']:.2%} | {rates['reachable_weighted_mass']:.2%} |"
        )
    lines.extend(["", "## Gates", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in gate.items())
    (args.output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
