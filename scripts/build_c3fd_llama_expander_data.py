#!/usr/bin/env python3
"""Build matched Route-F/Route-M H1 rich Expander data.

Both routes share the identical formula-prefill prompt and rich suffix target.
Route M alone carries a frozen C3FD feature vector consumed as soft prefix
embeddings by the Llama trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from crystal_dlm.c3fd_rich_expander import (  # noqa: E402
    C3FD_RICH_EXPANDER_VERSION,
    C3FD_SOFT_PREFIX_FEATURE_VERSION,
    FEATURE_DIM,
    ROUTE_FORMULA,
    ROUTE_SOFT_PREFIX,
    pack_soft_prefix_features,
    rich_suffix_from_plan_state,
)
from crystal_dlm.r5_plan_body import composition_plan_from_state  # noqa: E402


SCHEMA = "c3fd_llama_rich_expander_sft_v1"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_rows(path: Path, label: str) -> tuple[list[int], dict[int, dict[str, Any]]]:
    order: list[int] = []
    indexed: dict[int, dict[str, Any]] = {}
    for ordinal, row in enumerate(iter_jsonl(path)):
        source_idx = int(row.get("source_row_idx", ordinal))
        if source_idx in indexed:
            raise ValueError(f"{label} duplicates source_row_idx {source_idx}")
        order.append(source_idx)
        indexed[source_idx] = row
    return order, indexed


def minimal_plan(plan_state: Mapping[str, Any]) -> dict[str, Any]:
    composition = composition_plan_from_state(plan_state)
    return {
        "N": int(composition["N"]),
        "elements": list(composition["elements"]),
        "counts": list(composition["counts"]),
    }


def build_split(
    *,
    split: str,
    semantic_path: Path,
    predicted_path: Path,
    vocabulary: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    semantic_order, semantic = index_rows(semantic_path, f"semantic:{split}")
    predicted_order, predicted = index_rows(predicted_path, f"predicted:{split}")
    if semantic_order != predicted_order:
        raise ValueError(f"semantic/predicted source order changed for {split}")
    f_rows: list[dict[str, Any]] = []
    m_rows: list[dict[str, Any]] = []
    unavailable = 0
    for source_idx in semantic_order:
        semantic_row = semantic[source_idx]
        predicted_row = predicted[source_idx]
        plan_state = semantic_row.get("plan_state")
        if not isinstance(plan_state, Mapping):
            raise ValueError(f"semantic row {source_idx} lacks plan_state")
        plan = minimal_plan(plan_state)
        answer = rich_suffix_from_plan_state(plan_state)
        common = {
            "schema": SCHEMA,
            "expander_version": C3FD_RICH_EXPANDER_VERSION,
            "source_split": split,
            "source_row_idx": source_idx,
            "expander_plan_state": plan,
            "answer": answer,
            "sample_weight": float(
                semantic_row.get("sample_weight", 1.0) or 1.0
            ),
            "teacher_rich_fields_are_target_only": True,
            "outcomes_read": False,
        }
        f_rows.append({**common, "route": ROUTE_FORMULA})
        features = pack_soft_prefix_features(
            semantic_row, vocabulary, predicted_row
        )
        unavailable += int(features[2] == 0.0)
        m_rows.append(
            {
                **common,
                "route": ROUTE_SOFT_PREFIX,
                "soft_prefix_feature_version": C3FD_SOFT_PREFIX_FEATURE_VERSION,
                "soft_prefix_features": features,
            }
        )
    f_path = output_root / ROUTE_FORMULA / f"{split}.jsonl"
    m_path = output_root / ROUTE_SOFT_PREFIX / f"{split}.jsonl"
    f_count = write_jsonl_exclusive(f_path, f_rows)
    m_count = write_jsonl_exclusive(m_path, m_rows)
    if f_count != m_count or f_count != len(semantic_order):
        raise AssertionError("F/M row counts changed")
    return {
        "source_rows": len(semantic_order),
        "route_F_rows": f_count,
        "route_M_rows": m_count,
        "M_composition_trace_unavailable": unavailable,
        "F_sha256": sha256_file(f_path),
        "M_sha256": sha256_file(m_path),
        "visible_answers_identical": all(
            f_row["answer"] == m_row["answer"]
            and f_row["expander_plan_state"] == m_row["expander_plan_state"]
            for f_row, m_row in zip(f_rows, m_rows)
        ),
    }


def build_dataset(
    *,
    semantic_dir: Path,
    predicted_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    vocabulary_path = semantic_dir / "vocabulary.json"
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    preparing = output_dir.with_name(f".{output_dir.name}.preparing")
    if preparing.exists():
        raise FileExistsError(preparing)
    preparing.mkdir(parents=True)
    try:
        splits = {
            split: build_split(
                split=split,
                semantic_path=semantic_dir / f"{split}.jsonl",
                predicted_path=predicted_dir / f"{split}.jsonl",
                vocabulary=vocabulary,
                output_root=preparing,
            )
            for split in ("train", "val")
        }
        manifest = {
            "schema": SCHEMA,
            "status": "complete",
            "routes": [ROUTE_FORMULA, ROUTE_SOFT_PREFIX],
            "feature_dim": FEATURE_DIM,
            "soft_prefix_feature_version": C3FD_SOFT_PREFIX_FEATURE_VERSION,
            "source": "MP20 train/validation teacher rich fields",
            "C3FD_state_and_predictions_frozen": True,
            "predicted_checkpoint_selection": "none; seed17 and seed18 both encoded",
            "outcomes_read": False,
            "train_test_or_prospective_outcomes_read": [],
            "splits": splits,
            "inputs": {
                "semantic_vocabulary": sha256_file(vocabulary_path),
                **{
                    f"semantic_{split}": sha256_file(
                        semantic_dir / f"{split}.jsonl"
                    )
                    for split in ("train", "val")
                },
                **{
                    f"predicted_{split}": sha256_file(
                        predicted_dir / f"{split}.jsonl"
                    )
                    for split in ("train", "val")
                },
            },
        }
        manifest_path = preparing / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        sums = []
        for path in sorted(preparing.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                sums.append(
                    f"{sha256_file(path)}  {path.relative_to(preparing).as_posix()}"
                )
        (preparing / "SHA256SUMS").write_text(
            "\n".join(sums) + "\n", encoding="utf-8"
        )
        (preparing / "_SUCCESS").write_text(
            json.dumps(
                {
                    "manifest_sha256": sha256_file(manifest_path),
                    "sha256sums_sha256": sha256_file(preparing / "SHA256SUMS"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        preparing.rename(output_dir)
        return manifest
    except Exception:
        failed = output_dir.with_name(f".{output_dir.name}.FAILED")
        if preparing.exists():
            shutil.move(str(preparing), str(failed))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--predicted-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_dataset(
        semantic_dir=args.semantic_dir.resolve(),
        predicted_dir=args.predicted_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
