from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.crysllmgen.epoch_selection import (
    paired_bootstrap_difference_ci,
    select_epoch_checkpoint,
)
from crystal_dlm.wqcodiff.crysllmgen.epoch_training import (
    load_mixed_edit_epoch_contract,
    sha256_file,
)
from crystal_dlm.wqcodiff.crysllmgen.protocol import load_protocol_v4


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiments/wyckoff_codiffusion/protocol_v4.yaml"
CONTRACT = (
    ROOT
    / "configs/experiments/wyckoff_codiffusion/mixed_edit_three_epoch_v1.json"
)
BASE_SOURCE_SHA256 = "6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908"
EXECUTION_PATCH_SHA256 = "1" * 64
ADAPTER_TRAINING_PATCH_SHA256 = "2" * 64
REFINER_TRAINING_PATCH_SHA256 = "3" * 64
EVALUATION_PATCH_SHA256 = "4" * 64


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


class CrysLLMGenEpochSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        protocol = load_protocol_v4(PROTOCOL)
        self.contract = load_mixed_edit_epoch_contract(
            CONTRACT,
            base_protocol_name=protocol.name,
            base_protocol_sha256=protocol.sha256,
        )

    def _candidate(
        self,
        root: Path,
        *,
        epoch: int,
        strict_count: int,
        meta_count: int,
        metrics: dict[str, float],
        adapter_training_patch_sha256: str = EXECUTION_PATCH_SHA256,
        refiner_training_patch_sha256: str = EXECUTION_PATCH_SHA256,
        evaluation_patch_sha256: str = EXECUTION_PATCH_SHA256,
    ) -> dict[str, object]:
        directory = root / f"epoch-{epoch}"
        directory.mkdir()
        adapter_sha = hashlib.sha256(f"adapter-{epoch}".encode()).hexdigest()
        training = directory / "training.json"
        _write_json(
            training,
            {
                "schema": "crysllmgen_lora_training_report_v1",
                "logical_epoch": epoch,
                "training_seed": 11,
                "training_stage": "mixed_edit",
                "source_bundle_sha256": BASE_SOURCE_SHA256,
                "execution_patch_sha256": adapter_training_patch_sha256,
                "training_amendment": {"sha256": self.contract.sha256},
                "model": {"adapter_sha256": adapter_sha},
            },
        )
        nll = directory / "nll.json"
        _write_json(
            nll,
            {
                "schema": "crysllmgen_lora_validation_nll_v1",
                "ok": True,
                "training_seed": 11,
                "training_stage": "mixed_edit",
                "adapter_nll": 0.6 - 0.02 * epoch,
                "adapter": {
                    "adapter_model_sha256": adapter_sha,
                    "training_report_sha256": sha256_file(training),
                },
                "adapter_training_execution_patch_sha256": (
                    adapter_training_patch_sha256
                ),
                "evaluation_execution_patch_sha256": (
                    evaluation_patch_sha256
                ),
            },
        )
        generation = directory / "generation.jsonl"
        with generation.open("w", encoding="utf-8") as handle:
            for index in range(256):
                handle.write(
                    json.dumps(
                        {
                            "schema": "wqcodiff_generation_attempt_v1",
                            "attempt_id": f"attempt-{index:03d}",
                            "pair_id": f"pair-{index:03d}",
                            "paired_seed": 1000 + index,
                            "training_seed": 11,
                            "method": "C-WQ-HANDOFF",
                            "status": "succeeded",
                            "retry_or_replacement_used": False,
                            "adapter_training_execution_patch_sha256": (
                                adapter_training_patch_sha256
                            ),
                            "refiner_training_execution_patch_sha256": (
                                refiner_training_patch_sha256
                            ),
                            "evaluation_execution_patch_sha256": (
                                evaluation_patch_sha256
                            ),
                            "model_identity": {
                                "refiner_training_execution_patch_sha256": (
                                    refiner_training_patch_sha256
                                ),
                                "evaluation_execution_patch_sha256": (
                                    evaluation_patch_sha256
                                ),
                                "adapter_training": {
                                    "adapter_model_sha256": adapter_sha,
                                    "execution_patch_sha256": (
                                        adapter_training_patch_sha256
                                    ),
                                }
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        metric_report = directory / "metrics.json"
        _write_json(
            metric_report,
            {
                "schema": "crysllmgen_generation_metrics_report_v1",
                "ok": True,
                "attempts": 256,
                "denominator": "all_generation_attempts",
                "generation_jsonl_sha256": sha256_file(generation),
                "retry_or_replacement_used": False,
                "metrics_unchanged_upstream": metrics,
            },
        )
        r5c = directory / "a100.jsonl"
        with r5c.open("w", encoding="utf-8") as handle:
            for index in range(256):
                is_strict = index < strict_count
                is_meta = index < meta_count
                handle.write(
                    json.dumps(
                        {
                            "schema": "crysllmgen_r5c_a100_sun_attempt_v1",
                            "attempt_id": f"attempt-{index:03d}",
                            "method": "C-WQ-HANDOFF",
                            "metrics": {
                                "strict_full_sun": is_strict,
                                "meta_full_sun": is_meta,
                                "novel_unique": is_meta,
                            },
                            "base_source_bundle_sha256": BASE_SOURCE_SHA256,
                            "execution_patch_sha256": evaluation_patch_sha256,
                            "retry_or_replacement_used": False,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        summary = directory / "a100-summary.json"
        a100_contract = self.contract.data["evaluation"]["selection"][
            "r5c_a100_sun"
        ]
        exact = a100_contract["exact_executor"]
        references = a100_contract["frozen_references"]
        _write_json(
            summary,
            {
                "schema": "crysllmgen_r5c_a100_sun_summary_v1",
                "ok": True,
                "method": "C-WQ-HANDOFF",
                "base_source_bundle_sha256": BASE_SOURCE_SHA256,
                "execution_patch_sha256": evaluation_patch_sha256,
                "denominator": "all_generation_attempts",
                "coverage_adjusted_selection_role": "report_only_never_checkpoint_selection",
                "counts": {"total_attempts": 256},
                "rates": {
                    "attempt_strict_full_sun_lower_bound": strict_count / 256,
                    "attempt_meta_full_sun_lower_bound": meta_count / 256,
                    "attempt_novel_unique": meta_count / 256,
                },
                "exact_legacy_r5c_a100": {
                    "strict": {"stable": {"numerator": strict_count}},
                    "meta_like": {"stable": {"numerator": meta_count}},
                    "selection_role": "report_only",
                },
                "assets": {
                    "eval_sun_py": {"sha256": exact["eval_sun_py_sha256"]},
                    "eval_sun_resumable_py": {
                        "sha256": exact["eval_sun_resumable_py_sha256"]
                    },
                    "chgnet_model_asset": {
                        "sha256": exact["chgnet_model_sha256"]
                    },
                    "chgnet_runtime_checkpoint": {
                        "sha256": exact["chgnet_model_sha256"]
                    },
                    "train_csv": {
                        "sha256": references["mp20_train_csv_sha256"]
                    },
                    "training_index_cache": {
                        "sha256": references[
                            "mp20_training_index_cache_sha256"
                        ]
                    },
                    "mp_hull_cache": {
                        "sha256": references["mp_hull_cache_sha256"]
                    },
                    "chgnet_relax_cache": {
                        "sha256": references["chgnet_relax_cache_sha256"]
                    },
                },
                "attempt_results": {"sha256": sha256_file(r5c)},
                "retry_or_replacement_used": False,
            },
        )
        return {
            "logical_epoch": epoch,
            "training_report": str(training),
            "nll_report": str(nll),
            "generation_jsonl": str(generation),
            "crysllmgen_metrics_report": str(metric_report),
            "a100_attempt_jsonl": str(r5c),
            "a100_summary": str(summary),
        }

    def test_sun_equivalence_uses_direct_metrics_and_excludes_collapse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            epochs = [
                self._candidate(
                    root,
                    epoch=1,
                    strict_count=80,
                    meta_count=120,
                    metrics={
                        "comp_valid": 90.0,
                        "struct_valid": 99.0,
                        "valid": 90.0,
                        "cov_recall": 91.0,
                        "cov_precision": 96.0,
                        "wdist_density": 0.4,
                        "wdist_num_elems": 0.2,
                    },
                ),
                self._candidate(
                    root,
                    epoch=2,
                    strict_count=81,
                    meta_count=121,
                    metrics={
                        "comp_valid": 89.0,
                        "struct_valid": 98.8,
                        "valid": 88.5,
                        "cov_recall": 92.0,
                        "cov_precision": 97.0,
                        "wdist_density": 0.38,
                        "wdist_num_elems": 0.19,
                    },
                ),
                self._candidate(
                    root,
                    epoch=3,
                    strict_count=140,
                    meta_count=150,
                    metrics={
                        "comp_valid": 65.0,
                        "struct_valid": 94.0,
                        "valid": 60.0,
                        "cov_recall": 70.0,
                        "cov_precision": 85.0,
                        "wdist_density": 1.7,
                        "wdist_num_elems": 0.7,
                    },
                ),
            ]
            manifest = root / "manifest.json"
            _write_json(
                manifest,
                {
                    "schema": "crysllmgen_epoch_selection_evidence_v1",
                    "training_seed": 11,
                    "training_amendment_sha256": self.contract.sha256,
                    "base_source_bundle_sha256": BASE_SOURCE_SHA256,
                    "execution_patch_sha256": EXECUTION_PATCH_SHA256,
                    "epochs": epochs,
                },
            )
            output = root / "selection.json"
            report = select_epoch_checkpoint(
                contract=self.contract,
                evidence_manifest_path=manifest,
                output=output,
            )
            self.assertEqual(report["strict_sun_point_best_epoch"], 2)
            self.assertFalse(report["strict_sun_point_best_decisive"])
            self.assertEqual(report["strict_sun_equivalent_epochs"], [1, 2])
            self.assertEqual(
                report["meta_sun_point_best_epoch_within_strict_equivalent"], 2
            )
            self.assertFalse(report["meta_sun_point_best_decisive"])
            self.assertEqual(report["meta_sun_equivalent_epochs"], [1, 2])
            self.assertEqual(report["selected_epoch"], 1)
            self.assertFalse(all(report["absolute_guard_checks"]["3"].values()))
            self.assertEqual(json.loads(output.read_text()), report)

    def test_v2_binds_training_refiner_and_evaluation_patches_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = {
                "comp_valid": 90.0,
                "struct_valid": 99.0,
                "valid": 89.0,
                "cov_recall": 92.0,
                "cov_precision": 96.0,
                "wdist_density": 0.4,
                "wdist_num_elems": 0.2,
            }
            epochs = [
                self._candidate(
                    root,
                    epoch=epoch,
                    strict_count=80 + epoch,
                    meta_count=120 + epoch,
                    metrics=metrics,
                    adapter_training_patch_sha256=(
                        ADAPTER_TRAINING_PATCH_SHA256
                    ),
                    refiner_training_patch_sha256=(
                        REFINER_TRAINING_PATCH_SHA256
                    ),
                    evaluation_patch_sha256=EVALUATION_PATCH_SHA256,
                )
                for epoch in (1, 2, 3)
            ]
            manifest = root / "manifest-v2.json"
            _write_json(
                manifest,
                {
                    "schema": "crysllmgen_epoch_selection_evidence_v2",
                    "training_seed": 11,
                    "training_amendment_sha256": self.contract.sha256,
                    "base_source_bundle_sha256": BASE_SOURCE_SHA256,
                    "execution_patch_sha256": EVALUATION_PATCH_SHA256,
                    "adapter_training_execution_patch_sha256": (
                        ADAPTER_TRAINING_PATCH_SHA256
                    ),
                    "refiner_training_execution_patch_sha256": (
                        REFINER_TRAINING_PATCH_SHA256
                    ),
                    "evaluation_execution_patch_sha256": (
                        EVALUATION_PATCH_SHA256
                    ),
                    "execution_supersession": {
                        "supersedes_epoch_evaluation_array_job_id": "27104",
                        "reason": (
                            "pre-scientific execution-identity contract correction"
                        ),
                    },
                    "retry_or_replacement_used": True,
                    "attempt_retry_or_replacement_used": False,
                    "epochs": epochs,
                },
            )
            output = root / "selection-v2.json"
            report = select_epoch_checkpoint(
                contract=self.contract,
                evidence_manifest_path=manifest,
                output=output,
            )
            self.assertTrue(report["ok"])
            self.assertEqual(
                report["adapter_training_execution_patch_sha256"],
                ADAPTER_TRAINING_PATCH_SHA256,
            )
            self.assertEqual(
                report["refiner_training_execution_patch_sha256"],
                REFINER_TRAINING_PATCH_SHA256,
            )
            self.assertEqual(
                report["evaluation_execution_patch_sha256"],
                EVALUATION_PATCH_SHA256,
            )
            self.assertEqual(
                report["execution_supersession"][
                    "supersedes_epoch_evaluation_array_job_id"
                ],
                "27104",
            )
            self.assertTrue(report["retry_or_replacement_used"])
            self.assertFalse(report["attempt_retry_or_replacement_used"])
            self.assertEqual(len(report["candidates"]), 3)

    def test_paired_bootstrap_detects_a_decisive_difference(self) -> None:
        report = paired_bootstrap_difference_ci(
            [True] * 200 + [False] * 56,
            [True] * 50 + [False] * 206,
            draws=1000,
            seed=47,
        )
        self.assertGreater(report["ci95_lower"], 0.0)


if __name__ == "__main__":
    unittest.main()
