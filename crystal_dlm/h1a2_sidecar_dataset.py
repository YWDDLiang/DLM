"""Build R5-C body supervision with a non-visible PlanGraph sidecar.

The failed PlanGraph v1 experiment changed both the Planner language and the
R5-C body prompt.  That made it impossible to attribute a result to planned
corruption.  This module enforces the stricter H1-A2 comparison:

* ``prompt`` and ``answer`` remain byte-for-byte identical to R5-C;
* PlanGraph is deterministically compiled from the structure-derived teacher
  ``plan_state``;
* the graph is stored only as collator metadata and is never appended to the
  model-visible prompt;
* source metadata, IDs, energy, stability, and S.U.N. fields are not copied;
* every source row is retained, or publication fails atomically.

The teacher ``plan_state`` is not a sampled Planner output.  This dataset is
therefore eligible only for body-DLM training and likelihood diagnostics.
Fully-de-novo evidence must sample a Plan from the frozen or candidate H1-A2
Planner first, then compile the same sidecar from that realized model Plan.
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
from crystal_dlm.plangraph_dataset import sha256_file
from crystal_dlm.planned_corruption import plan_condition_sha256
from crystal_dlm.plangraph_v1 import (
    PLANGRAPH_VERSION,
    PlanGraphError,
    plangraph_from_record,
    plangraph_to_json,
)


H1A2_SIDECAR_DATASET_VERSION = "h1a2_r5c_plangraph_sidecar_v2"
H1A2_R5C_REPRESENTATION = "r5_exact_dynamic_v1"

H1A2_SIDECAR_RECORD_FIELDS = frozenset(
    {
        "task",
        "representation",
        "r5_representation",
        "prompt",
        "answer",
        "text",
        "loss_profile",
        "sample_weight",
        "num_atoms",
        "answer_semantic_length",
        "answer_token_count",
        "plangraph_version",
        "plangraph",
        "plan_condition_sha256",
        "model_visible_sha256",
        "training_pair_sha256",
    }
)


class H1A2SidecarBuildError(RuntimeError):
    """Raised when H1-A2 parity or atomic publication fails."""

    def __init__(self, message: str, *, report: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.report = dict(report or {})


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ordered_sha256(identities: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for ordinal, identity in enumerate(identities):
        digest.update(f"{ordinal}\t{identity}\n".encode("utf-8"))
    return digest.hexdigest()


def model_visible_text(source_record: Mapping[str, Any]) -> str:
    """Return the exact text consumed by ``JsonlSftDataset``."""

    prompt = source_record.get("prompt")
    answer = source_record.get("answer")
    if not isinstance(prompt, str) or not isinstance(answer, str):
        raise PlanGraphError("H1-A2 sidecar source requires string prompt/answer")
    visible = prompt.rstrip() + "\n" + answer
    source_text = source_record.get("text")
    if source_text is not None and source_text != visible:
        raise PlanGraphError(
            "source text differs from the R5-C training prompt/answer boundary"
        )
    return visible


def build_h1a2_sidecar_record(
    source_record: Mapping[str, Any],
) -> Dict[str, Any]:
    """Create one metadata-safe record without changing model-visible bytes."""

    if source_record.get("representation") != "dynamic_v1":
        raise PlanGraphError("H1-A2 sidecar source must use dynamic_v1")
    r5_representation = str(
        source_record.get("r5_representation") or H1A2_R5C_REPRESENTATION
    )
    if r5_representation != H1A2_R5C_REPRESENTATION:
        raise PlanGraphError(
            "H1-A2 sidecar source must use the frozen R5-C exact representation"
        )
    task = str(source_record.get("task") or "")
    if task != "r5_exact_dynamic_body":
        raise PlanGraphError(
            "H1-A2 sidecar source must use task='r5_exact_dynamic_body'"
        )
    loss_profile = str(source_record.get("loss_profile") or "")
    if loss_profile != "fixed_slot":
        raise PlanGraphError(
            "H1-A2 sidecar source must use loss_profile='fixed_slot'"
        )
    try:
        source_weight = float(source_record.get("sample_weight", 1.0))
    except (TypeError, ValueError) as exc:
        raise PlanGraphError("H1-A2 source sample_weight is malformed") from exc
    if source_weight != 1.0:
        raise PlanGraphError(
            "H1-A2 sidecar refuses non-unit source sample weights"
        )

    visible = model_visible_text(source_record)
    prompt = str(source_record["prompt"])
    answer = str(source_record["answer"])
    graph = plangraph_from_record(source_record)
    graph_json = plangraph_to_json(graph)
    num_atoms = int(graph["composition"]["N"])
    answer_length = 7 + 4 * num_atoms
    for field in ("num_atoms", "answer_semantic_length", "answer_token_count"):
        declared = source_record.get(field)
        expected = num_atoms if field == "num_atoms" else answer_length
        if declared is not None and int(declared) != expected:
            raise PlanGraphError(
                f"source {field}={declared} disagrees with expected {expected}"
            )

    model_visible_sha256 = _sha256_text(visible)
    condition_sha256 = plan_condition_sha256(
        prompt=prompt,
        graph=graph,
    )
    training_pair_sha256 = _sha256_text(
        model_visible_sha256 + "\n" + graph_json
    )
    record: Dict[str, Any] = {
        "task": task,
        "representation": "dynamic_v1",
        "r5_representation": H1A2_R5C_REPRESENTATION,
        "prompt": prompt,
        "answer": answer,
        "text": visible,
        "loss_profile": loss_profile,
        "sample_weight": 1.0,
        "num_atoms": num_atoms,
        "answer_semantic_length": answer_length,
        "answer_token_count": answer_length,
        "plangraph_version": PLANGRAPH_VERSION,
        "plangraph": graph,
        "plan_condition_sha256": condition_sha256,
        "model_visible_sha256": model_visible_sha256,
        "training_pair_sha256": training_pair_sha256,
    }
    if set(record) != set(H1A2_SIDECAR_RECORD_FIELDS):
        raise AssertionError("H1-A2 sidecar record field contract changed")
    if record["text"] != model_visible_text(record):
        raise AssertionError("H1-A2 model-visible text changed during conversion")
    return record


def _prepare_split(
    *,
    split: str,
    source_path: Path,
    stage: Path,
) -> Dict[str, Any]:
    output_path = stage / f"{split}.jsonl"
    ledger_path = stage / "row_ledger" / f"{split}.jsonl"
    total_rows = 0
    converted_rows = 0
    failures: list[Dict[str, Any]] = []
    identities: list[str] = []
    condition_identities: list[str] = []
    visible_identities: list[str] = []
    answer_lengths: Counter[int] = Counter()

    with (
        source_path.open(encoding="utf-8") as source_handle,
        output_path.open("x", encoding="utf-8") as output_handle,
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
                record = build_h1a2_sidecar_record(source_record)
            except Exception as exc:  # noqa: BLE001 - retain denominator evidence.
                if len(failures) < 20:
                    failures.append(
                        {
                            "ordinal": ordinal,
                            "line_number": line_number,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                continue

            output_handle.write(_canonical_json(record) + "\n")
            ledger_handle.write(
                _canonical_json(
                    {
                        "ordinal": ordinal,
                        "model_visible_sha256": record["model_visible_sha256"],
                        "plan_condition_sha256": record[
                            "plan_condition_sha256"
                        ],
                        "training_pair_sha256": record["training_pair_sha256"],
                    }
                )
                + "\n"
            )
            identities.append(str(record["training_pair_sha256"]))
            condition_identities.append(str(record["plan_condition_sha256"]))
            visible_identities.append(str(record["model_visible_sha256"]))
            answer_lengths[int(record["answer_token_count"])] += 1
            converted_rows += 1

    return {
        "split": split,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "total_rows": total_rows,
        "converted_rows": converted_rows,
        "failed_rows": total_rows - converted_rows,
        "all_rows_converted": total_rows > 0 and total_rows == converted_rows,
        "failure_examples": failures,
        "ordered_training_pair_sha256": _ordered_sha256(identities),
        "ordered_plan_condition_sha256": _ordered_sha256(
            condition_identities
        ),
        "ordered_model_visible_sha256": _ordered_sha256(visible_identities),
        "unique_training_pair_count": len(set(identities)),
        "duplicate_training_pair_count": converted_rows - len(set(identities)),
        "answer_token_count_distribution": {
            str(key): int(answer_lengths[key]) for key in sorted(answer_lengths)
        },
        "output": f"{split}.jsonl",
        "row_ledger": f"row_ledger/{split}.jsonl",
        "_training_pair_identities": identities,
    }


def _cross_split_overlaps(
    reports: Mapping[str, Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    names = list(reports)
    overlaps: list[Dict[str, Any]] = []
    for left_index, left in enumerate(names):
        left_ids = set(reports[left]["_training_pair_identities"])
        for right in names[left_index + 1 :]:
            shared = sorted(
                left_ids & set(reports[right]["_training_pair_identities"])
            )
            if shared:
                overlaps.append(
                    {
                        "left_split": left,
                        "right_split": right,
                        "count": len(shared),
                        "examples": shared[:20],
                    }
                )
    return overlaps


def build_h1a2_sidecar_dataset(
    *,
    split_inputs: Mapping[str, str | Path],
    output_dir: str | Path,
    vocab_file: str | Path,
    project_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Atomically publish a full-denominator H1-A2 sidecar dataset."""

    output = assert_writable_output_path(
        output_dir,
        project_root=project_root,
    )
    if output.exists():
        raise H1A2SidecarBuildError(f"refusing existing output path: {output}")
    if not split_inputs:
        raise H1A2SidecarBuildError("at least one split input is required")
    resolved_inputs = {
        str(split): Path(path).expanduser().resolve()
        for split, path in split_inputs.items()
    }
    for split, path in resolved_inputs.items():
        if not split or not split.replace("_", "").replace("-", "").isalnum():
            raise H1A2SidecarBuildError(f"unsafe split name: {split!r}")
        if not path.is_file():
            raise H1A2SidecarBuildError(f"missing source split {split}: {path}")
    vocab = Path(vocab_file).expanduser().resolve()
    if not vocab.is_file():
        raise H1A2SidecarBuildError(f"missing R5-C vocabulary: {vocab}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.staging-",
        dir=output.parent,
    ) as temp_dir:
        stage = Path(temp_dir)
        (stage / "row_ledger").mkdir()
        reports = {
            split: _prepare_split(
                split=split,
                source_path=path,
                stage=stage,
            )
            for split, path in resolved_inputs.items()
        }
        overlaps = _cross_split_overlaps(reports)
        total_rows = sum(int(report["total_rows"]) for report in reports.values())
        converted_rows = sum(
            int(report["converted_rows"]) for report in reports.values()
        )
        gate_passed = (
            total_rows > 0
            and total_rows == converted_rows
            and not overlaps
            and all(
                report["all_rows_converted"] is True for report in reports.values()
            )
        )
        public_reports = {
            split: {
                key: value
                for key, value in report.items()
                if key != "_training_pair_identities"
            }
            for split, report in reports.items()
        }
        if not gate_passed:
            raise H1A2SidecarBuildError(
                "H1-A2 sidecar publication gate failed",
                report={
                    "total_rows": total_rows,
                    "converted_rows": converted_rows,
                    "cross_split_overlaps": overlaps,
                    "splits": public_reports,
                },
            )

        shutil.copyfile(vocab, stage / "vocab_tokens.txt")
        max_answer_token_count = max(
            (
                int(length)
                for report in reports.values()
                for length in report["answer_token_count_distribution"]
            ),
            default=0,
        )
        stats = {
            "dataset_version": H1A2_SIDECAR_DATASET_VERSION,
            "representation": "dynamic_v1",
            "r5_representation": H1A2_R5C_REPRESENTATION,
            "plangraph_version": PLANGRAPH_VERSION,
            "plangraph_visibility": "collator_sidecar_only",
            "prompt_answer_byte_identity": True,
            "total_rows": total_rows,
            "split_rows": {
                split: int(report["converted_rows"])
                for split, report in reports.items()
            },
            "max_answer_token_count": max_answer_token_count,
            "answer_token_count": max_answer_token_count,
        }
        (stage / "stats.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        output_files = sorted(
            path
            for path in stage.rglob("*")
            if path.is_file() and path.name not in {"manifest.json", "_SUCCESS"}
        )
        output_hashes = {
            str(path.relative_to(stage)): sha256_file(path)
            for path in output_files
        }
        validation_split = "val" if "val" in reports else None
        manifest: Dict[str, Any] = {
            "dataset_version": H1A2_SIDECAR_DATASET_VERSION,
            "published": True,
            "output_dir": str(output),
            "total_rows": total_rows,
            "converted_rows": converted_rows,
            "failed_rows": 0,
            "all_rows_converted": True,
            "cross_split_training_pair_overlaps": overlaps,
            "prompt_answer_byte_identity": True,
            "plangraph_visibility": "collator_sidecar_only",
            "planner_schema": "H1-A2 rich seven-line unchanged",
            "source_plan_provenance": "structure_derived_teacher_plan_state",
            "model_proposed_plan": False,
            "eligible_as_end_to_end_planner_evidence": False,
            "intended_use": "body_dlm_training_and_likelihood_diagnostics",
            "source_metadata_copied": False,
            "sample_ids_copied": False,
            "retry": False,
            "replacement": False,
            "repair": False,
            "filter": False,
            "rerank": False,
            "splits": public_reports,
            "vocab_source_path": str(vocab),
            "vocab_source_sha256": sha256_file(vocab),
            "output_file_sha256": output_hashes,
            "fixed_validation_panel": {
                "split": validation_split,
                "ordered_model_visible_sha256": (
                    reports[validation_split]["ordered_model_visible_sha256"]
                    if validation_split is not None
                    else None
                ),
                "row_count": (
                    reports[validation_split]["converted_rows"]
                    if validation_split is not None
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
                    "dataset_version": H1A2_SIDECAR_DATASET_VERSION,
                    "manifest_sha256": manifest_sha256,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        stage.rename(output)
        return {
            **manifest,
            "manifest_sha256": manifest_sha256,
            "success_marker": str(output / "_SUCCESS"),
        }


__all__ = [
    "H1A2_R5C_REPRESENTATION",
    "H1A2_SIDECAR_DATASET_VERSION",
    "H1A2_SIDECAR_RECORD_FIELDS",
    "H1A2SidecarBuildError",
    "build_h1a2_sidecar_dataset",
    "build_h1a2_sidecar_record",
    "model_visible_text",
]
