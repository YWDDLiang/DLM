"""Atomic, leakage-safe dataset builder for PlanGraph-DLM experiments.

The builder is deliberately stricter than the read-only schema audit:

* every non-empty source row remains in the denominator and in source order;
* no row is repaired, retried, replaced, filtered, or silently dropped;
* train/validation training-pair identities must be disjoint;
* publication is atomic and refuses an existing output directory; and
* source metadata and identifiers are not copied into SFT records or ledgers.

The output contains separate Planner and dynamic-body views.  The body view is
directly consumable by ``scripts/llada_sft.py``.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Mapping, Sequence

from crystal_dlm.h1_readonly_guard import assert_writable_output_path
from crystal_dlm.plangraph_v1 import (
    PLANGRAPH_VERSION,
    PlanGraphError,
    build_plangraph_sft_records,
)


PLANGRAPH_DATASET_VERSION = "plangraph_dlm_dataset_v1"

PLANNER_RECORD_FIELDS = frozenset(
    {
        "task",
        "representation",
        "prompt",
        "answer",
        "text",
        "loss_profile",
        "sample_weight",
        "training_pair_sha256",
    }
)

BODY_RECORD_FIELDS = frozenset(
    {
        "task",
        "representation",
        "plangraph_version",
        "prompt",
        "answer",
        "text",
        "loss_profile",
        "sample_weight",
        "num_atoms",
        "answer_semantic_length",
        "answer_token_count",
        "plangraph",
        "training_pair_sha256",
    }
)


class PlanGraphDatasetBuildError(RuntimeError):
    """Raised when a frozen dataset cannot be published safely."""

    def __init__(self, message: str, *, report: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.report = dict(report or {})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _ordered_identity_sha256(identities: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for ordinal, identity in enumerate(identities):
        digest.update(f"{ordinal}\t{identity}\n".encode("utf-8"))
    return digest.hexdigest()


def _validate_sft_record_shapes(records: Mapping[str, Mapping[str, Any]]) -> None:
    if set(records) != {"planner", "body"}:
        raise PlanGraphError("converter must return exactly planner and body records")
    planner_fields = set(records["planner"])
    body_fields = set(records["body"])
    if planner_fields != set(PLANNER_RECORD_FIELDS):
        raise PlanGraphError(
            "planner record fields changed: "
            f"missing={sorted(PLANNER_RECORD_FIELDS - planner_fields)} "
            f"extra={sorted(planner_fields - PLANNER_RECORD_FIELDS)}"
        )
    if body_fields != set(BODY_RECORD_FIELDS):
        raise PlanGraphError(
            "body record fields changed: "
            f"missing={sorted(BODY_RECORD_FIELDS - body_fields)} "
            f"extra={sorted(body_fields - BODY_RECORD_FIELDS)}"
        )
    planner_identity = records["planner"]["training_pair_sha256"]
    body_identity = records["body"]["training_pair_sha256"]
    if planner_identity != body_identity:
        raise PlanGraphError("planner/body training_pair_sha256 identities disagree")


def _failure_category(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return "json_decode"
    message = str(exc).lower()
    if "representation" in message:
        return "unsupported_representation"
    if "missing plan_state" in message:
        return "missing_plan_state"
    if "missing dynamic-v1 answer" in message:
        return "missing_answer"
    if "answer does not match plan_state" in message:
        return "answer_plan_mismatch"
    if isinstance(exc, PlanGraphError):
        return "plangraph_contract"
    return type(exc).__name__


def _prepare_split(
    *,
    split: str,
    source_path: Path,
    stage: Path,
    max_failure_examples: int,
) -> Dict[str, Any]:
    planner_path = stage / "planner" / f"{split}.jsonl"
    body_path = stage / "body" / f"{split}.jsonl"
    ledger_path = stage / "row_ledger" / f"{split}.jsonl"
    identities: list[str] = []
    answer_length_counts: Counter[int] = Counter()
    num_atom_counts: Counter[int] = Counter()
    failure_categories: Counter[str] = Counter()
    failure_examples: list[Dict[str, Any]] = []
    total_rows = 0
    converted_rows = 0

    with (
        source_path.open(encoding="utf-8") as source_handle,
        planner_path.open("x", encoding="utf-8") as planner_handle,
        body_path.open("x", encoding="utf-8") as body_handle,
        ledger_path.open("x", encoding="utf-8") as ledger_handle,
    ):
        for line_number, raw_line in enumerate(source_handle, start=1):
            if not raw_line.strip():
                continue
            ordinal = total_rows
            total_rows += 1
            try:
                source_record = json.loads(raw_line)
                if not isinstance(source_record, dict):
                    raise PlanGraphError("source row must be a JSON object")
                records = build_plangraph_sft_records(source_record)
                _validate_sft_record_shapes(records)
            except Exception as exc:  # noqa: BLE001 - denominator must retain all failures.
                category = _failure_category(exc)
                failure_categories[category] += 1
                if len(failure_examples) < int(max_failure_examples):
                    failure_examples.append(
                        {
                            "split": split,
                            "ordinal": ordinal,
                            "line_number": line_number,
                            "category": category,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                continue

            planner = records["planner"]
            body = records["body"]
            identity = str(body["training_pair_sha256"])
            planner_handle.write(_canonical_json(planner) + "\n")
            body_handle.write(_canonical_json(body) + "\n")
            ledger_handle.write(
                _canonical_json(
                    {
                        "ordinal": ordinal,
                        "training_pair_sha256": identity,
                    }
                )
                + "\n"
            )
            identities.append(identity)
            answer_length_counts[int(body["answer_token_count"])] += 1
            num_atom_counts[int(body["num_atoms"])] += 1
            converted_rows += 1

    duplicate_count = converted_rows - len(set(identities))
    return {
        "split": split,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "total_rows": total_rows,
        "converted_rows": converted_rows,
        "failed_rows": total_rows - converted_rows,
        "conversion_rate": (
            float(converted_rows) / total_rows if total_rows else 0.0
        ),
        "all_rows_converted": total_rows > 0 and converted_rows == total_rows,
        "training_pair_identities": identities,
        "ordered_training_pair_sha256": _ordered_identity_sha256(identities),
        "unique_training_pair_count": len(set(identities)),
        "within_split_duplicate_count": duplicate_count,
        "answer_token_count_distribution": {
            str(key): int(answer_length_counts[key])
            for key in sorted(answer_length_counts)
        },
        "num_atoms_distribution": {
            str(key): int(num_atom_counts[key]) for key in sorted(num_atom_counts)
        },
        "failure_categories": {
            key: int(failure_categories[key]) for key in sorted(failure_categories)
        },
        "failure_examples": failure_examples,
        "outputs": {
            "planner": str(planner_path.relative_to(stage)),
            "body": str(body_path.relative_to(stage)),
            "row_ledger": str(ledger_path.relative_to(stage)),
        },
    }


def _cross_split_overlaps(
    split_reports: Mapping[str, Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    split_names = list(split_reports)
    overlaps: list[Dict[str, Any]] = []
    for left_index, left in enumerate(split_names):
        left_ids = set(split_reports[left]["training_pair_identities"])
        for right in split_names[left_index + 1 :]:
            identities = sorted(
                left_ids & set(split_reports[right]["training_pair_identities"])
            )
            if identities:
                overlaps.append(
                    {
                        "left_split": left,
                        "right_split": right,
                        "count": len(identities),
                        "examples": identities[:20],
                    }
                )
    return overlaps


def build_plangraph_dataset(
    *,
    split_inputs: Mapping[str, str | Path],
    output_dir: str | Path,
    vocab_file: str | Path,
    project_root: str | Path | None = None,
    minimum_conversion_rate: float = 0.98,
    require_all_rows: bool = True,
    fail_on_cross_split_overlap: bool = True,
    max_failure_examples: int = 20,
) -> Dict[str, Any]:
    """Build and atomically publish one frozen PlanGraph-DLM dataset.

    Existing output directories are never reused or overwritten.  Conversion
    failures are audited across the complete denominator before the staging
    directory is discarded.
    """

    output = assert_writable_output_path(
        output_dir,
        project_root=project_root,
    )
    if output.exists():
        raise PlanGraphDatasetBuildError(
            f"refusing existing output path: {output}",
            report={"output_dir": str(output), "published": False},
        )
    if not split_inputs:
        raise PlanGraphDatasetBuildError("at least one split input is required")
    split_names = list(split_inputs)
    if len(split_names) != len(set(split_names)):
        raise PlanGraphDatasetBuildError("split names must be unique")
    for split in split_names:
        if not split or not split.replace("_", "").replace("-", "").isalnum():
            raise PlanGraphDatasetBuildError(f"unsafe split name: {split!r}")

    minimum_rate = float(minimum_conversion_rate)
    if not 0.0 <= minimum_rate <= 1.0:
        raise PlanGraphDatasetBuildError(
            "minimum_conversion_rate must be between 0 and 1"
        )
    resolved_inputs = {
        split: Path(path).expanduser().resolve()
        for split, path in split_inputs.items()
    }
    for split, source_path in resolved_inputs.items():
        if not source_path.is_file():
            raise PlanGraphDatasetBuildError(
                f"source split {split!r} is not a file: {source_path}"
            )
    vocab = Path(vocab_file).expanduser().resolve()
    if not vocab.is_file():
        raise PlanGraphDatasetBuildError(f"vocab file is not a file: {vocab}")

    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{output.name}.staging-"
    with tempfile.TemporaryDirectory(prefix=prefix, dir=output.parent) as temp_dir:
        stage = Path(temp_dir)
        for directory in ("planner", "body", "row_ledger"):
            (stage / directory).mkdir()

        split_reports: Dict[str, Dict[str, Any]] = {}
        for split, source_path in resolved_inputs.items():
            split_reports[split] = _prepare_split(
                split=split,
                source_path=source_path,
                stage=stage,
                max_failure_examples=max_failure_examples,
            )

        overlaps = _cross_split_overlaps(split_reports)
        total_rows = sum(report["total_rows"] for report in split_reports.values())
        converted_rows = sum(
            report["converted_rows"] for report in split_reports.values()
        )
        failed_rows = total_rows - converted_rows
        aggregate_rate = float(converted_rows) / total_rows if total_rows else 0.0
        coverage_gate_passed = (
            total_rows > 0
            and aggregate_rate >= minimum_rate
            and all(report["total_rows"] > 0 for report in split_reports.values())
        )
        all_rows_converted = total_rows > 0 and failed_rows == 0
        overlap_gate_passed = not overlaps or not fail_on_cross_split_overlap
        publication_gate_passed = (
            coverage_gate_passed
            and (all_rows_converted or not require_all_rows)
            and overlap_gate_passed
        )

        public_split_reports: Dict[str, Dict[str, Any]] = {}
        for split, report in split_reports.items():
            public_split_reports[split] = {
                key: value
                for key, value in report.items()
                if key != "training_pair_identities"
            }
        report: Dict[str, Any] = {
            "dataset_version": PLANGRAPH_DATASET_VERSION,
            "plangraph_version": PLANGRAPH_VERSION,
            "output_dir": str(output),
            "published": False,
            "minimum_conversion_rate": minimum_rate,
            "require_all_rows": bool(require_all_rows),
            "fail_on_cross_split_overlap": bool(fail_on_cross_split_overlap),
            "total_rows": total_rows,
            "converted_rows": converted_rows,
            "failed_rows": failed_rows,
            "conversion_rate": aggregate_rate,
            "coverage_gate_passed": coverage_gate_passed,
            "all_rows_converted": all_rows_converted,
            "cross_split_identity_overlaps": overlaps,
            "cross_split_overlap_gate_passed": overlap_gate_passed,
            "publication_gate_passed": publication_gate_passed,
            "splits": public_split_reports,
            "source_metadata_copied": False,
            "sample_ids_copied": False,
            "retry": False,
            "replacement": False,
            "repair": False,
            "filter": False,
            "rerank": False,
        }
        if not publication_gate_passed:
            reasons: list[str] = []
            if not coverage_gate_passed:
                reasons.append("conversion coverage gate failed")
            if require_all_rows and not all_rows_converted:
                reasons.append("one or more source rows failed conversion")
            if not overlap_gate_passed:
                reasons.append("cross-split training-pair overlap detected")
            raise PlanGraphDatasetBuildError(
                "; ".join(reasons) or "dataset publication gate failed",
                report=report,
            )

        shutil.copyfile(vocab, stage / "body" / "vocab_tokens.txt")
        max_answer_token_count = max(
            (
                int(length)
                for split_report in split_reports.values()
                for length in split_report["answer_token_count_distribution"]
            ),
            default=0,
        )
        stats = {
            "dataset_version": PLANGRAPH_DATASET_VERSION,
            "representation": "dynamic_v1",
            "r5_representation": "plangraph_v1_dynamic_body",
            "plangraph_version": PLANGRAPH_VERSION,
            "total_rows": total_rows,
            "split_rows": {
                split: int(split_report["converted_rows"])
                for split, split_report in split_reports.items()
            },
            "max_answer_token_count": max_answer_token_count,
            "answer_token_count": max_answer_token_count,
            "tokenizer_preflight_required": True,
            "source_metadata_copied": False,
        }
        (stage / "body" / "stats.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        output_files = sorted(
            path
            for path in stage.rglob("*")
            if path.is_file() and path.name not in {"manifest.json", "_SUCCESS"}
        )
        output_hashes = {
            str(path.relative_to(stage)): sha256_file(path) for path in output_files
        }
        manifest = {
            **report,
            "published": True,
            "vocab_source_path": str(vocab),
            "vocab_source_sha256": sha256_file(vocab),
            "output_file_sha256": output_hashes,
            "fixed_validation_panel": {
                "split": "val" if "val" in split_reports else None,
                "ordered_training_pair_sha256": (
                    split_reports["val"]["ordered_training_pair_sha256"]
                    if "val" in split_reports
                    else None
                ),
                "row_count": (
                    split_reports["val"]["converted_rows"]
                    if "val" in split_reports
                    else 0
                ),
            },
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_sha256 = sha256_file(manifest_path)
        (stage / "_SUCCESS").write_text(
            json.dumps(
                {
                    "dataset_version": PLANGRAPH_DATASET_VERSION,
                    "manifest_sha256": manifest_sha256,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        stage.rename(output)
        manifest["manifest_sha256"] = manifest_sha256
        manifest["success_marker"] = str(output / "_SUCCESS")
        return manifest


__all__ = [
    "BODY_RECORD_FIELDS",
    "PLANGRAPH_DATASET_VERSION",
    "PLANNER_RECORD_FIELDS",
    "PlanGraphDatasetBuildError",
    "build_plangraph_dataset",
    "sha256_file",
]
