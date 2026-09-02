#!/usr/bin/env python3
"""Finalize paired raw validity for B0/BC/BP/BR SPAD cells."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ordinal_from_attempt_id(value: str) -> int:
    match = re.search(r"-(\d{4})$", str(value))
    if match is None:
        raise ValueError(f"attempt id lacks ordinal suffix: {value!r}")
    return int(match.group(1))


def exact_mcnemar(left: Sequence[bool], right: Sequence[bool]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("paired vectors have different lengths")
    right_only = sum((not a) and b for a, b in zip(left, right, strict=True))
    left_only = sum(a and (not b) for a, b in zip(left, right, strict=True))
    discordant = right_only + left_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(right_only, left_only)
        tail = sum(math.comb(discordant, value) for value in range(smaller + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "right_only": right_only,
        "left_only": left_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def metric_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = read_jsonl(path)
    indexed = {ordinal_from_attempt_id(row["attempt_id"]): row for row in rows}
    if len(rows) != 256 or set(indexed) != set(range(256)):
        raise ValueError(f"{path} does not cover exact requested ordinals")
    return indexed


def summarize(rows: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    result = {key: sum(bool(row[key]) for row in rows.values()) for key in ("comp_valid", "struct_valid", "valid")}
    result.update({f"{key}_rate": value / 256.0 for key, value in list(result.items())})
    return result


def text_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = read_jsonl(path)
    indexed = {int(row["sample_idx"]): row for row in rows}
    if len(rows) != 256 or set(indexed) != set(range(256)):
        raise ValueError("body rows do not cover exact sample indices")
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-run", type=Path, required=True)
    parser.add_argument("--program-plans", type=Path, required=True)
    parser.add_argument("--b0-body-run", type=Path, required=True)
    parser.add_argument("--bc-body-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    metrics: dict[str, dict[int, dict[int, dict[str, Any]]]] = defaultdict(dict)
    summaries: list[dict[str, Any]] = []
    for arm in ("B0", "BC", "BP", "BR"):
        for stream in (17, 18):
            rows = metric_rows(
                args.direct_run / arm / f"stream{stream}" / "direct/attempt_metrics.jsonl"
            )
            metrics[arm][stream] = rows
            summaries.append({"arm": arm, "stream": stream, **summarize(rows)})

    paired = {}
    for left, right, name in (
        ("B0", "BC", "transaction_schedule"),
        ("BC", "BP", "llama_species_program"),
        ("BP", "BR", "suffix_visible_backfill"),
    ):
        paired[name] = {}
        for stream in (17, 18):
            paired[name][str(stream)] = {
                metric: exact_mcnemar(
                    [bool(metrics[left][stream][index][metric]) for index in range(256)],
                    [bool(metrics[right][stream][index][metric]) for index in range(256)],
                )
                for metric in ("struct_valid", "valid")
            }

    plans = {int(row["sample_idx"]): row for row in read_jsonl(args.program_plans)}
    canonical_indices = {
        index
        for index, row in plans.items()
        if row["species_program_indices"]
        == list(range(len(row["species_program_indices"])))
    }
    isolation = {}
    token_pattern = re.compile(r"<[^>]+>")
    for stream in (17, 18):
        bc = text_rows(
            args.bc_body_run / "BC" / f"stream{stream}" / "body/raw_generations.jsonl"
        )
        bp = text_rows(
            args.b0_body_run / "BP" / f"stream{stream}" / "body/raw_generations.jsonl"
        )
        lattice_equal = sum(
            token_pattern.findall(bc[index]["text"])[:7]
            == token_pattern.findall(bp[index]["text"])[:7]
            for index in range(256)
        )
        exact_when_same_program = sum(
            bc[index]["text"] == bp[index]["text"] for index in canonical_indices
        )
        isolation[str(stream)] = {
            "lattice_prefix_equal": lattice_equal,
            "canonical_program_rows": len(canonical_indices),
            "full_text_equal_when_program_equal": exact_when_same_program,
        }

    pooled = {}
    for arm in ("B0", "BC", "BP", "BR"):
        rows = [row for row in summaries if row["arm"] == arm]
        pooled[arm] = {
            key: sum(float(row[key]) for row in rows) / len(rows)
            for key in ("comp_valid_rate", "struct_valid_rate", "valid_rate")
        }
    best_programmed = max(pooled["BP"]["valid_rate"], pooled["BR"]["valid_rate"])
    result = {
        "schema": "spad_raw_direct_final_v1",
        "requested_per_cell": 256,
        "cells": summaries,
        "pooled_stream_mean": pooled,
        "paired_mcnemar": paired,
        "isolation_audit": isolation,
        "schedule_training_direction_positive": best_programmed > pooled["BC"]["valid_rate"],
        "model494": False,
        "chgnet": False,
    }
    (args.output_dir / "SPAD_RAW_DIRECT_FINAL.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# SPAD raw Direct",
        "",
        "| Arm | Comp valid | Struct valid | Direct joint |",
        "|---|---:|---:|---:|",
    ]
    for arm in ("B0", "BC", "BP", "BR"):
        value = pooled[arm]
        lines.append(
            f"| {arm} | {100*value['comp_valid_rate']:.2f}% | "
            f"{100*value['struct_valid_rate']:.2f}% | {100*value['valid_rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "B0 is the retained confidence-ordered schedule; BC is canonical "
            "SPAD; BP adds the learned Llama species program; BR adds one "
            "suffix-visible backfill sweep.",
        ]
    )
    (args.output_dir / "SPAD_RAW_DIRECT_FINAL.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
