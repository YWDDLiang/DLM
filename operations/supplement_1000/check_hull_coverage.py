#!/usr/bin/env python3
"""A cache completed for another cohort is not complete for the current one."""
import argparse
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def coverage(plans, resolved, unresolved):
    wanted = {"-".join(sorted(set(row["plan_state"]["elements"]))) for row in plans}
    known = {row["chemsys"] for row in resolved}
    unknown = {row["chemsys"] for row in unresolved}
    if known & unknown:
        raise ValueError("a chemical system is both resolved and unresolved")
    missing = wanted - known - unknown
    return {"schema": "cohort_official_hull_coverage_v1", "coverage_accounted": not missing,
            "wanted_chemsys": len(wanted), "resolved_chemsys": len(wanted & known),
            "explicitly_unresolved_chemsys": sorted(wanted & unknown),
            "unqueried_chemsys": sorted(missing),
            "unknown_is_not_zero_energy": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plans", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    if not (args.cache / "completion_SUCCESS").is_file():
        raise ValueError("official cache query is not complete")
    result = coverage(read(args.plans), read(args.cache / "official_slim_cache.jsonl"),
                      read(args.cache / "unresolved_chemsys.jsonl"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["coverage_accounted"] else 2)


if __name__ == "__main__":
    main()
