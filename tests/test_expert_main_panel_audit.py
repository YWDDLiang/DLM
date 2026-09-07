"""Fail-closed integration boundaries for the autonomous 1200-request panel."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from scripts import export_r03_evaluation_inputs as exporter
from scripts.refine_dlm_with_crysllmgen import empty_payload
from scripts.run_r03_integrated_body import (
    B0_ADAPTER_SHA256,
    EDITOR_PANEL_SCHEMA,
    validate_editor_panel_holdout,
)


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_fixture(path, *, parent=None, development_composition=None):
    path.mkdir()
    files = {}
    for name in (
        "adapter_model.safetensors", "adapter_config.json", "expert_edit_modules.pt",
        "expert_edit_config.json", "periodic_state.pt", "periodic_state_config.json",
        "EXPERT_EDITOR.json", "roundtrip_probe.pt",
    ):
        (path / name).write_bytes(("fixture:" + name).encode())
        files[name] = digest(path / name)
    initializer = (
        {"kind": "original_B0", "adapter_sha256": B0_ADAPTER_SHA256}
        if parent is None else
        {"kind": "editor_checkpoint", "path": str(parent),
         "receipt_sha256": digest(parent / "CHECKPOINT_FINAL.json")}
    )
    contract = {"train_files": [], "dev_files": [], "initialization": initializer}
    if development_composition is not None:
        data = path / "data"
        data.mkdir()
        write_json(data / "DATA_FINAL.json", {})
        write_json(data / "dev.jsonl", {
            "source_split": "dev", "composition_key": development_composition,
        })
        contract["dev_files"].append({
            "path": str(data / "dev.jsonl"), "sha256": digest(data / "dev.jsonl"),
            "report_sha256": digest(data / "DATA_FINAL.json"),
        })
    write_json(path / "CHECKPOINT_FINAL.json", {
        "model_files_sha256": files, "contract": contract,
    })
    (path / "_CHECKPOINT_SUCCESS").touch()
    return path


def cohort(symbol):
    return [{"body_eligible": True, "plan_state": {
        "N": 1, "elements": [symbol], "counts": [1],
    }}]


class ExpertMainPanelAuditTests(unittest.TestCase):
    def test_holdout_checks_development_exposure_in_warmstart_ancestor(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ancestor = checkpoint_fixture(root / "ancestor", development_composition="Na:1")
            latest = checkpoint_fixture(root / "latest", parent=ancestor)
            clean = validate_editor_panel_holdout(latest, cohort("Cl"))
            self.assertEqual(len(clean["lineage"]), 2)
            self.assertEqual(len(clean["files"]), 1)
            with self.assertRaisesRegex(ValueError, "model selection"):
                validate_editor_panel_holdout(latest, cohort("Na"))

    def test_holdout_receipt_is_bound_to_actual_model_bytes(self):
        with TemporaryDirectory() as directory:
            checkpoint = checkpoint_fixture(Path(directory) / "checkpoint")
            self.assertTrue(validate_editor_panel_holdout(checkpoint, cohort("Cl"))[
                "evaluation_compositions_absent_from_train_and_dev"
            ])
            with (checkpoint / "expert_edit_modules.pt").open("ab") as stream:
                stream.write(b"changed-side-module")
            with self.assertRaisesRegex(ValueError, "file identity changed"):
                validate_editor_panel_holdout(checkpoint, cohort("Cl"))

    def test_editor_directory_cannot_omit_config_to_fall_back_to_legacy_rules(self):
        with TemporaryDirectory() as directory:
            body = Path(directory)
            write_json(body / "body_attempts.jsonl", {
                "schema": EDITOR_PANEL_SCHEMA, "ordinal": 0, "sample_idx": 0,
                "attempt_id": "original:0", "purpose": "evaluation",
                "body_generation_complete": False, "body_plan_match": False,
                "body_graph_complete": False,
            })
            write_json(body / "sample_metrics.json", {
                "schema": EDITOR_PANEL_SCHEMA, "requested_samples": 1, "denominator": 1,
                "decoded_samples": 0, "parse_success": 0, "graph_success": 0,
            })
            write_json(body / "run_config.json", {
                "schema": EDITOR_PANEL_SCHEMA, "expected_requests": 1,
            })
            (body / "_SUCCESS").touch()
            self.assertEqual(len(exporter.load_body_directory(body, 1)[0]), 1)
            (body / "run_config.json").unlink()
            with self.assertRaisesRegex(ValueError, "formal run configuration"):
                exporter.load_body_directory(body, 1)

    def test_formal_tau800_rejects_missing_frozen_seeding_flag(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            body, refined = root / "body", root / "refined"
            body.mkdir()
            refined.mkdir()
            write_json(body / "run_config.json", {
                "schema": EDITOR_PANEL_SCHEMA, "expected_requests": 1,
            })
            write_json(body / "attempt_ledger.jsonl", {
                "sample_idx": 0, "refiner_noise_seed": 71,
            })
            torch.save([], body / "proposal_graphs.pt")
            output = refined / "empty_refined.pt"
            torch.save(empty_payload(1), output)
            model_file = root / "model494-fixture.pt"
            model_file.write_bytes(b"fixture")
            config = {
                "proposal_graphs": str(body / "proposal_graphs.pt"),
                "diff_steps": 800, "num_evals": 1, "timesteps": 1000,
                "frozen_single_request_seeding": True,
                "seed_mode": "frozen_h1_ordinal_refiner_noise_seed",
                "seed_before_dataloader": True, "seed_from_graph_field": "refiner_noise_seed",
                "seed_by_sample_index": False, "batch_size": 1, "effective_batch_size": 1,
                "checkpoint": str(model_file), "checkpoint_sha256": exporter.MODEL494_SHA256,
                "proposal_graphs_sha256": digest(body / "proposal_graphs.pt"),
                "frozen_seed_ledger_sha256": digest(body / "attempt_ledger.jsonl"),
            }
            write_json(refined / "run_config.json", config)
            write_json(refined / "refinement_metrics.json", {
                "diff_steps": 800, "num_evals": 1, "assigned_proposals": 0,
                "output_file": str(output),
            })
            real_digest = exporter.sha256_file
            # Only the multi-GB model's digest is substituted. Every request,
            # graph, config and tensor is read through the real export code.
            with patch.object(exporter, "sha256_file", side_effect=lambda path: (
                exporter.MODEL494_SHA256 if Path(path) == model_file else real_digest(path)
            )):
                payload, _ = exporter.load_tau800(output, body_dir=body)
                self.assertEqual(payload["sample_indices"].numel(), 0)
                del config["frozen_single_request_seeding"]
                write_json(refined / "run_config.json", config)
                with self.assertRaisesRegex(ValueError, "exact frozen request-seeding"):
                    exporter.load_tau800(output, body_dir=body)


if __name__ == "__main__":
    unittest.main()
