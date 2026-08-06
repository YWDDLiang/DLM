#!/usr/bin/env python3
"""Freeze the exact two-rank, 100-row H1-A2 body-DLM validation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_identity_sha256(identities: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for position, identity in enumerate(identities):
        digest.update(f"{position}\t{identity}\n".encode("utf-8"))
    return digest.hexdigest()


def ordered_index_sha256(indices: Iterable[int]) -> str:
    payload = "".join(f"{int(index)}\n" for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_ledger(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"ledger line {line_number} is not an object")
            expected_fields = {
                "ordinal",
                "model_visible_sha256",
                "plan_condition_sha256",
                "training_pair_sha256",
            }
            if set(row) != expected_fields:
                raise ValueError(
                    f"ledger line {line_number} fields changed: "
                    f"{sorted(row)}"
                )
            if int(row["ordinal"]) != len(rows):
                raise ValueError(
                    f"ledger ordinal mismatch at line {line_number}: "
                    f"{row['ordinal']} != {len(rows)}"
                )
            for field in expected_fields - {"ordinal"}:
                value = row[field]
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(
                        f"ledger line {line_number} has malformed {field}"
                    )
            rows.append(row)
    if not rows:
        raise ValueError("validation ledger is empty")
    return rows


def audit_panel(
    *,
    dataset_manifest_path: Path,
    validation_ledger_path: Path,
    world_size: int,
    rows_per_rank: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if world_size <= 0 or rows_per_rank <= 0:
        raise ValueError("world_size and rows_per_rank must be positive")

    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("dataset manifest must be an object")
    if manifest.get("dataset_version") != "h1a2_r5c_plangraph_sidecar_v2":
        raise ValueError("unexpected dataset version")
    if manifest.get("prompt_answer_byte_identity") is not True:
        raise ValueError("prompt/answer byte identity is not frozen true")
    if manifest.get("plangraph_visibility") != "collator_sidecar_only":
        raise ValueError("PlanGraph visibility contract changed")

    expected_ledger_sha = (
        manifest.get("output_file_sha256", {}).get("row_ledger/val.jsonl")
    )
    observed_ledger_sha = sha256_file(validation_ledger_path)
    if expected_ledger_sha != observed_ledger_sha:
        raise ValueError(
            "validation ledger SHA mismatch: "
            f"{observed_ledger_sha} != {expected_ledger_sha}"
        )

    rows = load_ledger(validation_ledger_path)
    expected_count = manifest.get("fixed_validation_panel", {}).get("row_count")
    if int(expected_count) != len(rows):
        raise ValueError(
            f"validation row count mismatch: {len(rows)} != {expected_count}"
        )

    full_shards = [
        list(range(rank, len(rows), world_size)) for rank in range(world_size)
    ]
    flattened = [index for shard in full_shards for index in shard]
    if len(flattened) != len(rows) or sorted(flattened) != list(range(len(rows))):
        raise ValueError("rank-strided validation shards are not an exact cover")

    selected_by_rank = [shard[:rows_per_rank] for shard in full_shards]
    selected_indices = [index for shard in selected_by_rank for index in shard]
    expected_panel_rows = world_size * rows_per_rank
    if len(set(selected_indices)) != expected_panel_rows:
        raise ValueError("fixed panel contains duplicate validation ordinals")
    global_sorted_indices = sorted(selected_indices)
    if global_sorted_indices != list(range(expected_panel_rows)):
        raise ValueError(
            "frozen rank-strided panel is not the first contiguous global rows"
        )

    panel_ledger: list[dict[str, Any]] = []
    for rank, indices in enumerate(selected_by_rank):
        for rank_position, index in enumerate(indices):
            row = rows[index]
            panel_ledger.append(
                {
                    "evaluation_order": len(panel_ledger),
                    "rank": rank,
                    "rank_position": rank_position,
                    "validation_ordinal": index,
                    "model_visible_sha256": row["model_visible_sha256"],
                    "plan_condition_sha256": row["plan_condition_sha256"],
                    "training_pair_sha256": row["training_pair_sha256"],
                }
            )

    globally_ordered_rows = [rows[index] for index in global_sorted_indices]
    report = {
        "schema": "h1a2_v3_dlm_fixed_100_row_panel_audit_v1",
        "status": "complete",
        "dataset_version": manifest["dataset_version"],
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "validation_ledger_path": str(validation_ledger_path),
        "validation_ledger_sha256": observed_ledger_sha,
        "validation_row_count": len(rows),
        "world_size": world_size,
        "batch_size_per_rank": 1,
        "eval_max_batches_per_rank": rows_per_rank,
        "global_panel_row_count": expected_panel_rows,
        "sampler": "rank_strided_no_padding",
        "panel_rule": (
            "first rows_per_rank indices from each frozen rank-strided "
            "no-padding validation shard"
        ),
        "rank_shards": [
            {
                "rank": rank,
                "full_count": len(full_shards[rank]),
                "selected_count": len(indices),
                "selected_first": indices[0],
                "selected_last": indices[-1],
                "selected_ordered_index_sha256": ordered_index_sha256(indices),
            }
            for rank, indices in enumerate(selected_by_rank)
        ],
        "rank_major_selected_indices_sha256": ordered_index_sha256(
            selected_indices
        ),
        "global_sorted_selected_indices_sha256": ordered_index_sha256(
            global_sorted_indices
        ),
        "global_sorted_model_visible_sha256": ordered_identity_sha256(
            [str(row["model_visible_sha256"]) for row in globally_ordered_rows]
        ),
        "global_sorted_plan_condition_sha256": ordered_identity_sha256(
            [str(row["plan_condition_sha256"]) for row in globally_ordered_rows]
        ),
        "global_sorted_training_pair_sha256": ordered_identity_sha256(
            [str(row["training_pair_sha256"]) for row in globally_ordered_rows]
        ),
        "B1_B2_same_panel": True,
        "initial_intermediate_terminal_same_panel": True,
        "no_padding_duplicates": True,
        "automatic_downstream": False,
        "scientific_training_authorized": False,
        "gate_passed": True,
    }
    return report, panel_ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--validation-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--rows-per-rank", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    report, panel_ledger = audit_panel(
        dataset_manifest_path=args.dataset_manifest.expanduser().resolve(),
        validation_ledger_path=args.validation_ledger.expanduser().resolve(),
        world_size=int(args.world_size),
        rows_per_rank=int(args.rows_per_rank),
    )
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.staging-",
        dir=output_dir.parent,
    ) as temporary_directory:
        stage = Path(temporary_directory)
        ledger_path = stage / "PANEL_LEDGER.jsonl"
        with ledger_path.open("x", encoding="utf-8") as handle:
            for row in panel_ledger:
                handle.write(canonical_json(row) + "\n")
        report["panel_ledger_sha256"] = sha256_file(ledger_path)
        report_path = stage / "FIXED_PANEL_AUDIT.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        success = {
            "schema": report["schema"],
            "report_sha256": sha256_file(report_path),
            "panel_ledger_sha256": report["panel_ledger_sha256"],
            "gate_passed": True,
        }
        (stage / "_SUCCESS").write_text(
            json.dumps(success, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage.rename(output_dir)

    print(json.dumps({**report, "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
