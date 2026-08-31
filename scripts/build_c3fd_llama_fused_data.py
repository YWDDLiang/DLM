#!/usr/bin/env python3
"""Build immutable typed residual-head data for the fused C3FD--Llama Planner."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from crystal_dlm.c3fd_llama_fused_plan import (  # noqa: E402
    FUSED_TYPED_PLAN_SCHEMA,
    stability_condition_from_e_above_hull,
    typed_targets_from_semantic_row,
)


DATASET_SCHEMA = "c3fd_llama_fused_typed_dataset_v1"
SPLITS = ("train", "val")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_row_idx(
    row: Mapping[str, Any], *, label: str, key: str
) -> int:
    if key not in row:
        raise ValueError(f"{label} lacks immutable {key}")
    value = row[key]
    if isinstance(value, bool):
        raise ValueError(f"{label} has invalid source_row_idx")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} has invalid source_row_idx") from exc
    if result < 0 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{label} has invalid source_row_idx")
    return result


def index_rows(
    path: Path, *, label: str, key: str = "source_row_idx"
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        source_idx = _source_row_idx(row, label=label, key=key)
        if source_idx in indexed:
            raise ValueError(f"{label} duplicates source_row_idx {source_idx}")
        indexed[source_idx] = row
    return indexed


def _split_marker_matches(row: Mapping[str, Any], split: str) -> bool:
    for key in ("source_split", "split"):
        if key in row and row[key] not in (None, ""):
            return str(row[key]) == split
    return True


def _stability_condition(source_row: Mapping[str, Any]) -> str:
    metadata = source_row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("missing source metadata")
    if "e_above_hull" not in metadata or metadata.get("e_above_hull") in (None, ""):
        raise ValueError("missing e_above_hull")
    return stability_condition_from_e_above_hull(metadata["e_above_hull"])


def _sample_weight(row: Mapping[str, Any]) -> float:
    try:
        value = float(row.get("sample_weight", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid sample_weight") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("invalid sample_weight")
    return value


def _skip_reason_for_metadata(exc: ValueError) -> str:
    message = str(exc)
    if message == "missing source metadata":
        return "missing_source_metadata"
    if message == "missing e_above_hull":
        return "missing_e_above_hull"
    if message == "malformed e_above_hull":
        return "malformed_e_above_hull"
    if message == "nonfinite e_above_hull":
        return "nonfinite_e_above_hull"
    return "invalid_source_metadata"


def _record_skip(
    reasons: Counter[str],
    indices: dict[str, list[int]],
    reason: str,
    source_idx: int,
) -> None:
    reasons[reason] += 1
    indices[reason].append(int(source_idx))


def build_split_rows(
    *,
    split: str,
    semantic_path: Path,
    source_path: Path,
    vocabulary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Perform an explicit key join and retain a complete skip ledger."""

    semantic = index_rows(
        semantic_path, label=f"semantic:{split}", key="source_row_idx"
    )
    source = index_rows(
        source_path,
        label=f"ctv_minimal:{split}",
        key="c3fd_certificate_source_row_idx",
    )
    output: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    skipped_indices: dict[str, list[int]] = defaultdict(list)
    tier_counts: Counter[str] = Counter()

    for source_idx in sorted(set(semantic) | set(source)):
        if source_idx not in semantic:
            _record_skip(
                skipped,
                skipped_indices,
                "missing_semantic_row",
                source_idx,
            )
            continue
        if source_idx not in source:
            _record_skip(
                skipped,
                skipped_indices,
                "missing_ctv_minimal_row",
                source_idx,
            )
            continue
        semantic_row = semantic[source_idx]
        source_row = source[source_idx]
        if not _split_marker_matches(semantic_row, split) or not _split_marker_matches(
            source_row, split
        ):
            _record_skip(
                skipped,
                skipped_indices,
                "split_marker_mismatch",
                source_idx,
            )
            continue
        try:
            stability_condition = _stability_condition(source_row)
        except ValueError as exc:
            _record_skip(
                skipped,
                skipped_indices,
                _skip_reason_for_metadata(exc),
                source_idx,
            )
            continue
        try:
            targets = typed_targets_from_semantic_row(semantic_row, vocabulary)
            sample_weight = _sample_weight(semantic_row)
        except (TypeError, ValueError):
            _record_skip(
                skipped,
                skipped_indices,
                "invalid_teacher_action_sequence",
                source_idx,
            )
            continue
        output.append(
            {
                "schema": DATASET_SCHEMA,
                "target_schema": FUSED_TYPED_PLAN_SCHEMA,
                "source_split": split,
                "source_row_idx": source_idx,
                "sample_weight": sample_weight,
                "stability_condition": stability_condition,
                "proposal_target": targets["proposal_target"],
                "species_ids": targets["species_ids"],
                "count_targets": targets["count_targets"],
                "ledger_steps": targets["ledger_steps"],
                "legal_action_indices": targets["legal_action_indices"],
                "max_count": targets["max_count"],
                "soft_targets": targets["soft_targets"],
                "audit_transcript": targets["audit_transcript"],
            }
        )
        tier_counts[stability_condition] += 1

    report = {
        "semantic_rows": len(semantic),
        "ctv_minimal_rows": len(source),
        "join_union_rows": len(set(semantic) | set(source)),
        "joined_rows": len(set(semantic) & set(source)),
        "kept_rows": len(output),
        "skipped_rows": sum(skipped.values()),
        "stability_condition_counts": dict(sorted(tier_counts.items())),
        "skipped_reasons": dict(sorted(skipped.items())),
        "skipped_source_row_indices": {
            reason: indices for reason, indices in sorted(skipped_indices.items())
        },
    }
    if report["kept_rows"] + report["skipped_rows"] != report["join_union_rows"]:
        raise AssertionError("join accounting does not close")
    return output, report


def _write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def build_dataset(
    *, semantic_dir: Path, ctv_minimal_dir: Path, output_dir: Path
) -> dict[str, Any]:
    """Build train/val only, publishing atomically to one exclusive path."""

    semantic_dir = semantic_dir.resolve()
    ctv_minimal_dir = ctv_minimal_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    preparing = output_dir.with_name(f".{output_dir.name}.preparing")
    failed = output_dir.with_name(f".{output_dir.name}.FAILED")
    if preparing.exists():
        raise FileExistsError(preparing)
    if failed.exists():
        raise FileExistsError(failed)
    vocabulary_path = semantic_dir / "vocabulary.json"
    if not vocabulary_path.is_file():
        raise FileNotFoundError(vocabulary_path)
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    if not isinstance(vocabulary, Mapping):
        raise ValueError("semantic vocabulary must be an object")

    input_paths = {
        "semantic_vocabulary": vocabulary_path,
        **{
            f"semantic_{split}": semantic_dir / f"{split}.jsonl"
            for split in SPLITS
        },
        **{
            f"ctv_minimal_{split}": ctv_minimal_dir / f"{split}.jsonl"
            for split in SPLITS
        },
    }
    for path in input_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    preparing.mkdir(parents=True, exist_ok=False)
    try:
        split_reports: dict[str, Any] = {}
        for split in SPLITS:
            rows, split_report = build_split_rows(
                split=split,
                semantic_path=input_paths[f"semantic_{split}"],
                source_path=input_paths[f"ctv_minimal_{split}"],
                vocabulary=vocabulary,
            )
            if not rows:
                raise ValueError(f"{split} has no valid fused Planner rows")
            output_path = preparing / f"{split}.jsonl"
            written = _write_jsonl_exclusive(output_path, rows)
            if written != len(rows):
                raise AssertionError(f"{split} output count changed")
            split_report["output_sha256"] = sha256_file(output_path)
            split_reports[split] = split_report

        manifest = {
            "schema": DATASET_SCHEMA,
            "status": "complete",
            "data_scope": "MP20 corresponding train/val splits only",
            "included_splits": list(SPLITS),
            "dev_test_prospective_rows": 0,
            "join_key": (
                "semantic.source_row_idx == "
                "ctv_minimal.c3fd_certificate_source_row_idx"
            ),
            "ordinal_zip_used": False,
            "raw_e_above_hull_copied_to_output": False,
            "structure_or_body_fields_copied_to_output": False,
            "stability_boundary_eV_per_atom": 0.1,
            "splits": split_reports,
            "input_sha256": {
                label: sha256_file(path) for label, path in sorted(input_paths.items())
            },
        }
        manifest_path = preparing / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksum_lines = [
            f"{sha256_file(path)}  {path.name}"
            for path in (
                preparing / "train.jsonl",
                preparing / "val.jsonl",
                manifest_path,
            )
        ]
        sums_path = preparing / "SHA256SUMS"
        sums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        success = {
            "manifest_sha256": sha256_file(manifest_path),
            "sha256sums_sha256": sha256_file(sums_path),
        }
        (preparing / "_SUCCESS").write_text(
            json.dumps(success, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        preparing.rename(output_dir)
        return manifest
    except Exception:
        if preparing.exists():
            shutil.move(str(preparing), str(failed))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--ctv-minimal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_dataset(
        semantic_dir=args.semantic_dir,
        ctv_minimal_dir=args.ctv_minimal_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
