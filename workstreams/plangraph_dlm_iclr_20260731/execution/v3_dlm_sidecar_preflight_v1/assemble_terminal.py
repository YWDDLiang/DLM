#!/usr/bin/env python3
"""Assemble the fail-closed terminal report for V3 DLM sidecar preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SPLIT_ROWS = {
    "train": 27_136,
    "val": 9_047,
    "test": 9_046,
}
EXPECTED_PREFLIGHT_ROWS = EXPECTED_SPLIT_ROWS["train"] + EXPECTED_SPLIT_ROWS["val"]
EXPECTED_TOTAL_ROWS = sum(EXPECTED_SPLIT_ROWS.values())
EXPECTED_DATASET_VERSION = "h1a2_r5c_plangraph_sidecar_v2"
EXPECTED_PREFLIGHT_VERSION = "plangraph_dlm_tokenizer_mask_preflight_v1"
EXPECTED_PROVENANCE = "structure_derived_teacher_plan_state"
EXPECTED_VOCAB_SHA256 = (
    "71d3145913eb3496c82f28c654989e37b0031105f52291483788aca3a8bee56b"
)
EXPECTED_TOKENIZER = Path(
    "/public/home/jiaosz/ywliang/ai4s/"
    "diffsion_language_model_meets_diffusion/"
    "runs/20260529_212834-r5c-exactlen-256/"
    "outputs/r5c_exact_sft/final"
)


class TerminalAssemblyError(RuntimeError):
    """Raised when any frozen sidecar/preflight invariant is violated."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TerminalAssemblyError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TerminalAssemblyError(f"expected JSON object at {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalAssemblyError(message)


def _verify_sidecar(dataset_root: Path) -> tuple[dict[str, Any], str]:
    manifest_path = dataset_root / "manifest.json"
    success_path = dataset_root / "_SUCCESS"
    manifest = _load_json(manifest_path)
    success = _load_json(success_path)
    manifest_sha256 = _sha256(manifest_path)

    _require(
        success.get("manifest_sha256") == manifest_sha256,
        "sidecar _SUCCESS disagrees with manifest SHA-256",
    )
    _require(
        manifest.get("dataset_version") == EXPECTED_DATASET_VERSION,
        "unexpected sidecar dataset version",
    )
    _require(manifest.get("published") is True, "sidecar is not published")
    _require(
        manifest.get("total_rows") == EXPECTED_TOTAL_ROWS,
        "sidecar total denominator changed",
    )
    _require(
        manifest.get("converted_rows") == EXPECTED_TOTAL_ROWS,
        "sidecar converted-row denominator changed",
    )
    _require(manifest.get("failed_rows") == 0, "sidecar has failed rows")
    _require(
        manifest.get("all_rows_converted") is True,
        "sidecar publication is incomplete",
    )
    _require(
        manifest.get("cross_split_training_pair_overlaps") == [],
        "sidecar has cross-split training-pair overlap",
    )
    _require(
        manifest.get("prompt_answer_byte_identity") is True,
        "R5-C prompt/answer byte identity was not retained",
    )
    _require(
        manifest.get("plangraph_visibility") == "collator_sidecar_only",
        "PlanGraph became model-visible",
    )
    _require(
        manifest.get("planner_schema") == "H1-A2 rich seven-line unchanged",
        "H1-A2 Planner schema changed",
    )
    _require(
        manifest.get("source_plan_provenance") == EXPECTED_PROVENANCE,
        "teacher-Plan provenance changed",
    )
    _require(
        manifest.get("model_proposed_plan") is False,
        "teacher Plans were mislabeled as model-proposed",
    )
    _require(
        manifest.get("eligible_as_end_to_end_planner_evidence") is False,
        "teacher Plans became eligible as Planner evidence",
    )
    _require(
        manifest.get("intended_use")
        == "body_dlm_training_and_likelihood_diagnostics",
        "sidecar intended use changed",
    )
    _require(
        manifest.get("source_metadata_copied") is False
        and manifest.get("sample_ids_copied") is False,
        "source metadata or sample IDs leaked into the sidecar",
    )
    for prohibited in ("retry", "replacement", "repair", "filter", "rerank"):
        _require(
            manifest.get(prohibited) is False,
            f"prohibited sidecar operation enabled: {prohibited}",
        )

    split_reports = manifest.get("splits")
    _require(isinstance(split_reports, dict), "sidecar split reports are missing")
    for split, expected_rows in EXPECTED_SPLIT_ROWS.items():
        report = split_reports.get(split)
        _require(isinstance(report, dict), f"missing sidecar split: {split}")
        _require(
            report.get("total_rows") == expected_rows,
            f"{split} total-row denominator changed",
        )
        _require(
            report.get("converted_rows") == expected_rows,
            f"{split} converted-row denominator changed",
        )
        _require(
            report.get("failed_rows") == 0
            and report.get("all_rows_converted") is True,
            f"{split} did not convert completely",
        )
    return manifest, manifest_sha256


def _verify_mask_smoke(report: dict[str, Any], policy: str) -> None:
    smoke = report.get("mask_smoke")
    _require(isinstance(smoke, dict), f"{policy} mask smoke is missing")
    _require(smoke.get("device") == "cpu", f"{policy} smoke used a non-CPU device")
    _require(smoke.get("smoke_row_count") == 32, f"{policy} smoke row count changed")
    _require(
        smoke.get("deterministic_repeat_rows") == 32,
        f"{policy} stateless-repeat denominator changed",
    )
    _require(
        smoke.get("all_invariants_passed") is True,
        f"{policy} mask invariants failed",
    )
    planned = smoke.get("planned_only")
    fixed_iid = smoke.get("fixed_stateless_iid")
    mixture = smoke.get("iid_to_planned_2_to_1")
    _require(isinstance(planned, dict), f"{policy} planned-only counts missing")
    _require(isinstance(fixed_iid, dict), f"{policy} fixed-iid counts missing")
    _require(isinstance(mixture, dict), f"{policy} mixture counts missing")
    _require(
        planned.get("samples") == 32
        and planned.get("planned_samples") == 32
        and planned.get("iid_samples") == 0,
        f"{policy} planned-only smoke denominator changed",
    )
    _require(
        fixed_iid.get("samples") == 32
        and fixed_iid.get("planned_samples") == 0
        and fixed_iid.get("iid_samples") == 32,
        f"{policy} fixed-iid smoke denominator changed",
    )
    _require(
        mixture.get("samples") == 32
        and mixture.get("planned_samples", 0) + mixture.get("iid_samples", 0) == 32,
        f"{policy} 2:1 mixture smoke denominator changed",
    )


def _verify_preflight(
    *,
    path: Path,
    policy: str,
    dataset_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    report = _load_json(path)
    _require(
        report.get("preflight_version") == EXPECTED_PREFLIGHT_VERSION,
        f"{policy} preflight version changed",
    )
    _require(report.get("policy") == policy, f"{policy} report policy changed")
    _require(report.get("max_length") == 382, f"{policy} max length changed")
    _require(
        report.get("corruption_mix")
        == {"iid_fraction": 2.0, "planned_fraction": 1.0},
        f"{policy} corruption mixture changed",
    )
    _require(
        report.get("validation_corruption") == "stateless_iid_fixed_panel_v1",
        f"{policy} validation corruption changed",
    )
    _require(
        report.get("total_rows") == EXPECTED_PREFLIGHT_ROWS
        and report.get("passed_rows") == EXPECTED_PREFLIGHT_ROWS
        and report.get("failed_rows") == 0
        and report.get("all_rows_passed") is True,
        f"{policy} full train+validation denominator failed",
    )
    _require(
        report.get("preflight_gate_passed") is True,
        f"{policy} preflight gate failed",
    )
    _require(
        report.get("gpu_used") is False and report.get("model_loaded") is False,
        f"{policy} preflight loaded a model or used a GPU",
    )
    verification = report.get("manifest_verification")
    _require(
        isinstance(verification, dict)
        and verification.get("manifest_sha256") == dataset_manifest_sha256,
        f"{policy} verified a different sidecar manifest",
    )
    tokenizer = report.get("tokenizer")
    _require(isinstance(tokenizer, dict), f"{policy} tokenizer report missing")
    _require(
        tokenizer.get("vocab_file_sha256") == EXPECTED_VOCAB_SHA256,
        f"{policy} data vocabulary changed",
    )
    _require(
        Path(str(tokenizer.get("tokenizer_path"))).resolve()
        == EXPECTED_TOKENIZER.resolve(),
        f"{policy} tokenizer/checkpoint identity changed",
    )
    split_reports = report.get("splits")
    _require(isinstance(split_reports, dict), f"{policy} split reports missing")
    for split in ("train", "val"):
        split_report = split_reports.get(split)
        expected_rows = EXPECTED_SPLIT_ROWS[split]
        _require(
            isinstance(split_report, dict),
            f"{policy} missing split report: {split}",
        )
        _require(
            split_report.get("total_rows") == expected_rows
            and split_report.get("passed_rows") == expected_rows
            and split_report.get("failed_rows") == 0
            and split_report.get("all_rows_passed") is True,
            f"{policy} {split} tokenizer denominator failed",
        )
        _require(
            int(split_report.get("max_model_length", 0)) <= 382,
            f"{policy} {split} exceeds frozen max length",
        )
        _require(
            split_report.get("duplicate_corruption_key_rows") == 0,
            f"{policy} {split} has duplicate corruption keys",
        )
    _verify_mask_smoke(report, policy)
    return report, _sha256(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-identity", type=Path, required=True)
    parser.add_argument("--d1-report", type=Path, required=True)
    parser.add_argument("--d2-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise TerminalAssemblyError(f"refusing existing output: {output}")

    source_identity = _load_json(args.source_identity.expanduser().resolve())
    _require(
        source_identity.get("status") == "complete"
        and source_identity.get("total_rows") == EXPECTED_TOTAL_ROWS,
        "frozen source identity gate failed",
    )
    manifest, manifest_sha256 = _verify_sidecar(
        args.dataset_root.expanduser().resolve()
    )
    d1, d1_sha256 = _verify_preflight(
        path=args.d1_report.expanduser().resolve(),
        policy="d1",
        dataset_manifest_sha256=manifest_sha256,
    )
    d2, d2_sha256 = _verify_preflight(
        path=args.d2_report.expanduser().resolve(),
        policy="d2",
        dataset_manifest_sha256=manifest_sha256,
    )

    terminal = {
        "schema": "h1a2_dlm_sidecar_preflight_terminal_v1",
        "status": "complete",
        "sidecar_gate_passed": True,
        "d1_preflight_gate_passed": True,
        "d2_preflight_gate_passed": True,
        "total_sidecar_rows": EXPECTED_TOTAL_ROWS,
        "preflight_train_validation_rows": EXPECTED_PREFLIGHT_ROWS,
        "split_rows": EXPECTED_SPLIT_ROWS,
        "source_plan_provenance": EXPECTED_PROVENANCE,
        "model_proposed_plan": False,
        "prompt_answer_byte_identity": True,
        "model_visible_plangraph": False,
        "max_length": 382,
        "observed_max_model_length": {
            "d1_train": d1["splits"]["train"]["max_model_length"],
            "d1_val": d1["splits"]["val"]["max_model_length"],
            "d2_train": d2["splits"]["train"]["max_model_length"],
            "d2_val": d2["splits"]["val"]["max_model_length"],
        },
        "source_split_sha256": {
            split: manifest["splits"][split]["source_sha256"]
            for split in EXPECTED_SPLIT_ROWS
        },
        "dataset_manifest_sha256": manifest_sha256,
        "source_identity_sha256": _sha256(args.source_identity.resolve()),
        "d1_preflight_report_sha256": d1_sha256,
        "d2_preflight_report_sha256": d2_sha256,
        "retry": False,
        "replacement": False,
        "repair": False,
        "filter": False,
        "rerank": False,
        "sample_ids_copied": False,
        "source_metadata_copied": False,
        "gpu_used": False,
        "model_loaded": False,
        "crystal_generation": False,
        "sun_evaluation": False,
        "automatic_downstream": False,
        "scientific_training_authorized": False,
        "next_gate": "targeted_2xa800_b1_b2_engineering_smoke",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(terminal, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(terminal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
