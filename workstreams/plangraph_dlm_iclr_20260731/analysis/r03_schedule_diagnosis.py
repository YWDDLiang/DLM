#!/usr/bin/env python3
"""Read-only paired diagnosis for the H1-A2 R03 body-schedule screen."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


ATOM_RE = re.compile(
    r"<E_([^>]+)><X_(-?\d+)><Y_(-?\d+)><Z_(-?\d+)>"
)
COUNT_RE = re.compile(r"<N_(\d+)>")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_body_text(text: str) -> dict[str, Any]:
    count_match = COUNT_RE.search(text)
    atoms = [
        {
            "slot": slot,
            "species": species,
            "coord": (int(x) % 100, int(y) % 100, int(z) % 100),
        }
        for slot, (species, x, y, z) in enumerate(ATOM_RE.findall(text))
    ]
    expected_n = int(count_match.group(1)) if count_match else None
    by_coord: dict[tuple[int, int, int], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    by_xy: dict[tuple[int, int], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for atom in atoms:
        by_coord[atom["coord"]].append(atom)
        by_xy[atom["coord"][:2]].append(atom)
    duplicate_clusters = {
        coord: members for coord, members in by_coord.items() if len(members) > 1
    }
    xy_clusters = {
        xy: members for xy, members in by_xy.items() if len(members) > 1
    }
    return {
        "expected_n": expected_n,
        "atoms": atoms,
        "parsed_atom_count": len(atoms),
        "duplicate_clusters": duplicate_clusters,
        "xy_clusters": xy_clusters,
    }


def pair_count(size: int) -> int:
    return size * (size - 1) // 2


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [math.nan, math.nan]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - half_width, center + half_width]


def summarize_numeric(values: Iterable[float]) -> dict[str, float | int | None]:
    data = list(values)
    if not data:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": len(data),
        "mean": sum(data) / len(data),
        "min": min(data),
        "max": max(data),
    }


def canonical_slot_groups(plan: dict[str, Any]) -> list[str]:
    groups: list[str] = []
    for element, count in zip(plan["elements"], plan["counts"]):
        groups.extend([str(element)] * int(count))
    return groups


def add_rate_bucket(
    buckets: dict[str, dict[str, int]], key: Any, failed: bool
) -> None:
    bucket = buckets.setdefault(str(key), {"attempts": 0, "failed": 0, "succeeded": 0})
    bucket["attempts"] += 1
    bucket["failed" if failed else "succeeded"] += 1


def finalize_rate_buckets(
    buckets: dict[str, dict[str, int]]
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for key, bucket in sorted(buckets.items(), key=lambda item: int(item[0])):
        total = bucket["attempts"]
        result[key] = {
            **bucket,
            "failure_rate": bucket["failed"] / total if total else math.nan,
        }
    return result


def analyze_attempt(
    attempt: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    parsed = parse_body_text(str(attempt.get("text") or ""))
    slot_groups = canonical_slot_groups(plan)
    duplicates: list[dict[str, Any]] = []
    same_group_pairs = 0
    cross_group_pairs = 0
    for coord, members in sorted(parsed["duplicate_clusters"].items()):
        annotated = []
        for member in members:
            slot = int(member["slot"])
            plan_group = slot_groups[slot] if slot < len(slot_groups) else None
            annotated.append({**member, "plan_group": plan_group})
        for left_index, left in enumerate(annotated):
            for right in annotated[left_index + 1 :]:
                if left["plan_group"] == right["plan_group"]:
                    same_group_pairs += 1
                else:
                    cross_group_pairs += 1
        duplicates.append(
            {
                "coord": list(coord),
                "members": annotated,
                "same_plan_group": len(
                    {member["plan_group"] for member in annotated}
                )
                == 1,
            }
        )
    atoms = parsed["atoms"]
    n = len(atoms)
    unique_x = len({atom["coord"][0] for atom in atoms})
    unique_y = len({atom["coord"][1] for atom in atoms})
    unique_z = len({atom["coord"][2] for atom in atoms})
    xy_pair_repetitions = sum(
        pair_count(len(members)) for members in parsed["xy_clusters"].values()
    )
    counts = [int(value) for value in plan["counts"]]
    return {
        "ordinal": int(attempt["ordinal"]),
        "status": str(attempt["status"]),
        "reason": str(attempt.get("reason") or ""),
        "n": int(plan["N"]),
        "elements": list(plan["elements"]),
        "counts": counts,
        "arity": len(counts),
        "max_multiplicity": max(counts),
        "d1_xyz_group_width": int(plan["N"]),
        "d2_max_coordinate_group_width": 3 * max(counts),
        "parsed_atom_count": parsed["parsed_atom_count"],
        "text_count_matches": parsed["expected_n"] == parsed["parsed_atom_count"],
        "duplicate_cluster_count": len(duplicates),
        "duplicate_pair_count": same_group_pairs + cross_group_pairs,
        "same_group_duplicate_pairs": same_group_pairs,
        "cross_group_duplicate_pairs": cross_group_pairs,
        "duplicates": duplicates,
        "unique_x": unique_x,
        "unique_y": unique_y,
        "unique_z": unique_z,
        "unique_x_fraction": unique_x / n if n else None,
        "unique_y_fraction": unique_y / n if n else None,
        "unique_z_fraction": unique_z / n if n else None,
        "repeated_xy_pair_count": xy_pair_repetitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument(
        "--view",
        choices=("full", "summary", "failures"),
        default="full",
    )
    args = parser.parse_args()

    control = read_jsonl(args.control)
    candidate = read_jsonl(args.candidate)
    ledger = read_jsonl(args.ledger)
    plans = {
        int(row["ordinal"]): row["arms"]["P0"]["plan_state"]
        for row in ledger
        if int(row["ordinal"]) < 32
    }
    control_by_ordinal = {int(row["ordinal"]): row for row in control}
    candidate_by_ordinal = {int(row["ordinal"]): row for row in candidate}
    ordinals = sorted(set(control_by_ordinal) | set(candidate_by_ordinal))
    if ordinals != list(range(32)):
        raise RuntimeError(f"expected ordinals 0..31, got {ordinals}")

    control_rows = [
        analyze_attempt(control_by_ordinal[ordinal], plans[ordinal])
        for ordinal in ordinals
    ]
    candidate_rows = [
        analyze_attempt(candidate_by_ordinal[ordinal], plans[ordinal])
        for ordinal in ordinals
    ]

    status_matrix = collections.Counter(
        (
            control_by_ordinal[ordinal]["status"],
            candidate_by_ordinal[ordinal]["status"],
        )
        for ordinal in ordinals
    )
    failure_by_n: dict[str, dict[str, int]] = {}
    failure_by_max_multiplicity: dict[str, dict[str, int]] = {}
    failure_by_d2_width: dict[str, dict[str, int]] = {}
    failure_by_arity: dict[str, dict[str, int]] = {}
    for row in candidate_rows:
        failed = row["status"] != "succeeded"
        add_rate_bucket(failure_by_n, row["n"], failed)
        add_rate_bucket(failure_by_max_multiplicity, row["max_multiplicity"], failed)
        add_rate_bucket(
            failure_by_d2_width, row["d2_max_coordinate_group_width"], failed
        )
        add_rate_bucket(failure_by_arity, row["arity"], failed)

    failed_candidate_rows = [
        row for row in candidate_rows if row["status"] != "succeeded"
    ]
    duplicate_failures = [
        row
        for row in failed_candidate_rows
        if "DuplicateCoordinateError" in row["reason"]
    ]
    same_group_failure_count = sum(
        row["same_group_duplicate_pairs"] > 0 for row in duplicate_failures
    )
    cross_group_failure_count = sum(
        row["cross_group_duplicate_pairs"] > 0 for row in duplicate_failures
    )
    candidate_duplicate_pairs = sum(
        row["duplicate_pair_count"] for row in candidate_rows
    )
    candidate_same_group_pairs = sum(
        row["same_group_duplicate_pairs"] for row in candidate_rows
    )
    candidate_cross_group_pairs = sum(
        row["cross_group_duplicate_pairs"] for row in candidate_rows
    )
    control_duplicate_pairs = sum(row["duplicate_pair_count"] for row in control_rows)

    discordant_control_only = status_matrix[("succeeded", "failed")]
    discordant_candidate_only = status_matrix[("failed", "succeeded")]
    discordant_total = discordant_control_only + discordant_candidate_only
    exact_mcnemar_p = (
        min(
            1.0,
            2.0
            * sum(
                math.comb(discordant_total, index)
                for index in range(
                    0, min(discordant_control_only, discordant_candidate_only) + 1
                )
            )
            / (2**discordant_total),
        )
        if discordant_total
        else 1.0
    )

    report = {
        "schema": "h1_a2_r03_schedule_diagnosis_v1",
        "denominator": len(ordinals),
        "status_matrix": {
            f"control_{left}__candidate_{right}": count
            for (left, right), count in sorted(status_matrix.items())
        },
        "completion": {
            "control": sum(row["status"] == "succeeded" for row in control_rows),
            "candidate": sum(
                row["status"] == "succeeded" for row in candidate_rows
            ),
            "candidate_duplicate_failures": len(duplicate_failures),
            "candidate_failure_rate": len(failed_candidate_rows) / len(ordinals),
            "candidate_failure_wilson_95": wilson_interval(
                len(failed_candidate_rows), len(ordinals)
            ),
            "paired_exact_mcnemar_p": exact_mcnemar_p,
        },
        "duplicate_topology": {
            "control_duplicate_pairs_all_outputs": control_duplicate_pairs,
            "candidate_duplicate_pairs_all_outputs": candidate_duplicate_pairs,
            "candidate_same_plan_group_pairs": candidate_same_group_pairs,
            "candidate_cross_plan_group_pairs": candidate_cross_group_pairs,
            "failed_samples_with_same_group_collision": same_group_failure_count,
            "failed_samples_with_cross_group_collision": cross_group_failure_count,
        },
        "candidate_failure_rates": {
            "by_n": finalize_rate_buckets(failure_by_n),
            "by_arity": finalize_rate_buckets(failure_by_arity),
            "by_max_multiplicity": finalize_rate_buckets(
                failure_by_max_multiplicity
            ),
            "by_d2_max_coordinate_group_width": finalize_rate_buckets(
                failure_by_d2_width
            ),
        },
        "group_width_comparison": {
            "failed": summarize_numeric(
                row["d2_max_coordinate_group_width"]
                for row in candidate_rows
                if row["status"] != "succeeded"
            ),
            "succeeded": summarize_numeric(
                row["d2_max_coordinate_group_width"]
                for row in candidate_rows
                if row["status"] == "succeeded"
            ),
            "width_at_least_18": {
                "attempts": sum(
                    row["d2_max_coordinate_group_width"] >= 18
                    for row in candidate_rows
                ),
                "failed": sum(
                    row["d2_max_coordinate_group_width"] >= 18
                    and row["status"] != "succeeded"
                    for row in candidate_rows
                ),
            },
            "width_below_18": {
                "attempts": sum(
                    row["d2_max_coordinate_group_width"] < 18
                    for row in candidate_rows
                ),
                "failed": sum(
                    row["d2_max_coordinate_group_width"] < 18
                    and row["status"] != "succeeded"
                    for row in candidate_rows
                ),
            },
        },
        "coordinate_concentration": {
            "control_unique_x_fraction": summarize_numeric(
                row["unique_x_fraction"] for row in control_rows
            ),
            "candidate_unique_x_fraction": summarize_numeric(
                row["unique_x_fraction"] for row in candidate_rows
            ),
            "control_unique_y_fraction": summarize_numeric(
                row["unique_y_fraction"] for row in control_rows
            ),
            "candidate_unique_y_fraction": summarize_numeric(
                row["unique_y_fraction"] for row in candidate_rows
            ),
            "control_unique_z_fraction": summarize_numeric(
                row["unique_z_fraction"] for row in control_rows
            ),
            "candidate_unique_z_fraction": summarize_numeric(
                row["unique_z_fraction"] for row in candidate_rows
            ),
            "control_repeated_xy_pairs": summarize_numeric(
                row["repeated_xy_pair_count"] for row in control_rows
            ),
            "candidate_repeated_xy_pairs": summarize_numeric(
                row["repeated_xy_pair_count"] for row in candidate_rows
            ),
        },
        "candidate_duplicate_failure_details": duplicate_failures,
        "integrity": {
            "all_text_counts_match": all(
                row["text_count_matches"] for row in control_rows + candidate_rows
            ),
            "all_failure_texts_parse_full": all(
                row["parsed_atom_count"] == row["n"] for row in failed_candidate_rows
            ),
        },
    }
    if args.view == "summary":
        report.pop("candidate_duplicate_failure_details")
        payload: Any = report
    elif args.view == "failures":
        payload = [
            {
                "ordinal": row["ordinal"],
                "n": row["n"],
                "counts": row["counts"],
                "d2_width": row["d2_max_coordinate_group_width"],
                "same_pairs": row["same_group_duplicate_pairs"],
                "cross_pairs": row["cross_group_duplicate_pairs"],
                "duplicate_coords": [
                    {
                        "coord": duplicate["coord"],
                        "groups": [
                            member["plan_group"] for member in duplicate["members"]
                        ],
                    }
                    for duplicate in row["duplicates"]
                ],
            }
            for row in duplicate_failures
        ]
    else:
        payload = report
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
