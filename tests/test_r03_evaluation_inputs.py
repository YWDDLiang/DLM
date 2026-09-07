"""CPU-only artifact accounting; no generated structures or physical scoring."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from pymatgen.core import Structure

from crystal_dlm.dynamic_crystal import parse_dynamic_answer
from scripts.export_r03_evaluation_inputs import (
    MODEL494_SHA256, P0_ADAPTER_SHA256,
    export_components, export_legacy_panel, export_records, load_body_directory, load_tau800, main,
    validate_refined_shapes,
)


def body_text(species):
    head = "<N_002><LA_040><LB_040><LC_040><AA_090><AB_090><AG_090>"
    return head + f"<E_{species[0]}><X_000><Y_000><Z_000><E_{species[1]}><X_050><Y_050><Z_050>"


def request(ordinal, sample_idx, species=("O", "Li"), *, failure=False):
    text = body_text(species)
    row = {"schema": "r03_integrated_body_v1", "ordinal": ordinal, "sample_idx": sample_idx,
           "attempt_id": f"original-request-{sample_idx}", "body_eligible": not failure,
           "status": "failed" if failure else "succeeded", "parsed": not failure,
           "attempt_status": "planner_failure" if failure else "complete",
           "body_generation_complete": not failure, "body_plan_match": not failure,
           "body_graph_complete": not failure, "body_noise_seed": 101 + sample_idx,
           "reason": "original_planner_failure" if failure else None,
           "plan_state": {"N": 2, "elements": list(species), "counts": [1, 1],
                          "charge_bucket": "recorded_value_not_recomputed"},
           "planner_record": {"attempt_id": f"original-request-{sample_idx}"},
           "body_prompt": "the actually recorded rich prompt\n",
           "raw_body_text": text, "text": text, "arrays": parse_dynamic_answer(text, strict=True)}
    return row


def ledger():
    return [request(0, 17), request(1, 91, failure=True), request(2, 300, ("Na", "Cl"))]


def refined_payload():
    # Deliberately reversed global request order and reversed species order.
    return {"sample_indices": torch.tensor([300, 17], dtype=torch.long),
            "num_atoms": torch.tensor([[2, 2]], dtype=torch.long),
            "atom_types": torch.tensor([[17, 11, 3, 8]], dtype=torch.long),
            "frac_coords": torch.tensor([[[0.123456789012, 0.2, 0.3], [0.6, 0.7, 0.8],
                                           [0.234567890123, 0.25, 0.3], [0.7, 0.75, 0.8]]], dtype=torch.float64),
            "lengths": torch.tensor([[[5.123456789012, 5.0, 6.0], [4.123456789012, 4.0, 4.0]]], dtype=torch.float64),
            "angles": torch.tensor([[[90.0, 90.0, 96.0], [90.0, 90.0, 93.0]]], dtype=torch.float64)}


def load_common_script(name):
    spec = importlib.util.spec_from_file_location(f"_r03_test_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R03EvaluationInputsTests(unittest.TestCase):
    def test_native_keeps_all_global_requests_and_original_attempts(self):
        source = ledger()
        rows, report = export_records(source, endpoint="native", expected_requests=3, method_id="R03-G")
        self.assertEqual([r["sample_idx"] for r in rows], [17, 91, 300])
        self.assertEqual([r["evaluation_ordinal"] for r in rows], [0, 1, 2])
        self.assertEqual([r["attempt_id"] for r in rows], [r["attempt_id"] for r in source])
        self.assertEqual(report["requests"], 3)
        self.assertEqual(report["successful"], 2)
        self.assertEqual(rows[0]["plan_state"], source[0]["plan_state"])
        self.assertTrue(all(r["source_split"] == "evaluation" and not r["trainable_teacher"] for r in rows))
        self.assertIsNone(rows[1]["body"])
        self.assertIsNone(rows[1]["structure"])
        self.assertNotIn("trace", rows[1])
        self.assertFalse(report["neural_trace_synthesized"])

    def test_tau_tensor_join_is_global_not_position_and_does_not_assume_site_order(self):
        rows, report = export_records(ledger(), endpoint="tau800", expected_requests=3,
                                      method_id="R03-G", refined_payload=refined_payload())
        self.assertTrue(rows[0]["success"])
        self.assertFalse(rows[1]["success"])
        self.assertTrue(rows[2]["success"])
        first = Structure.from_dict(rows[0]["structure"])
        last = Structure.from_dict(rows[2]["structure"])
        self.assertEqual(list(first.atomic_numbers), [3, 8])
        self.assertEqual(list(last.atomic_numbers), [17, 11])
        self.assertAlmostEqual(first.lattice.a, 4.123456789012, places=10)
        self.assertAlmostEqual(first.frac_coords[0, 0], 0.234567890123, places=11)
        self.assertFalse(rows[0]["site_order_mapping_verified"])
        self.assertIsNone(rows[0]["body"])
        self.assertEqual(report["tensor_join_key"], "global_sample_idx")

    def test_missing_refined_graph_ready_request_is_an_engineering_error(self):
        payload = refined_payload()
        payload = {"sample_indices": payload["sample_indices"][:1], "num_atoms": payload["num_atoms"][:, :1],
                   "atom_types": payload["atom_types"][:, :2], "frac_coords": payload["frac_coords"][:, :2],
                   "lengths": payload["lengths"][:, :1], "angles": payload["angles"][:, :1]}
        with self.assertRaisesRegex(ValueError, "source graph ledger"):
            export_records(ledger(), endpoint="tau800", expected_requests=3,
                            method_id="R03-P", refined_payload=payload)

    def test_one_nonfinite_refined_geometry_fails_only_its_request(self):
        payload = refined_payload()
        payload["frac_coords"][0, 0, 0] = float("nan")
        rows, report = export_records(ledger(), endpoint="tau800", expected_requests=3,
                                      method_id="R03-P", refined_payload=payload)
        self.assertTrue(rows[0]["success"])
        self.assertFalse(rows[2]["success"])
        self.assertIsNone(rows[2]["structure"])
        self.assertEqual(report["requests"], 3)

    def test_refined_species_change_does_not_get_silently_relabelled(self):
        payload = refined_payload()
        payload["atom_types"][0, 2] = 11  # Na replaces original Li at sample17.
        rows, _ = export_records(ledger(), endpoint="tau800", expected_requests=3,
                                  method_id="R03-G", refined_payload=payload)
        self.assertFalse(rows[0]["success"])
        self.assertIn("species multiset", rows[0]["artifact_error"])
        self.assertTrue(rows[2]["success"])

    def test_out_of_ledger_or_duplicate_tensor_ids_are_rejected(self):
        payload = refined_payload()
        payload["sample_indices"][0] = 999
        with self.assertRaises(ValueError):
            export_records(ledger(), endpoint="tau800", expected_requests=3, method_id="R03-G", refined_payload=payload)
        payload["sample_indices"] = torch.tensor([17, 17])
        with self.assertRaises(ValueError):
            validate_refined_shapes(payload)

    def test_tensor_integer_and_atom_boundaries_cannot_be_truncated_or_guessed(self):
        payload = refined_payload()
        payload["sample_indices"] = torch.tensor([300.5, 17.0])
        with self.assertRaises(ValueError):
            validate_refined_shapes(payload)
        payload = refined_payload()
        payload["frac_coords"] = payload["frac_coords"][:, :-1]
        with self.assertRaises(ValueError):
            validate_refined_shapes(payload)

    def test_controller_failure_never_rescues_the_stale_original_body(self):
        source = ledger()
        source[0].update(status="failed", parsed=False, attempt_status="controller_failure", reason="controller_failure")
        rows, _ = export_records(source, endpoint="native", expected_requests=3, method_id="R03-G")
        self.assertFalse(rows[0]["success"])
        self.assertIsNone(rows[0]["body"])
        self.assertIsNone(rows[0]["structure"])

    def test_pure_graph_failure_keeps_native_endpoint_and_original_execution_failure(self):
        source = ledger()
        source[0].update(status="failed", parsed=False, attempt_status="body_failure",
                          body_graph_complete=False, earliest_failure_stage="body_graph", reason="graph_failed")
        rows, _ = export_records(source, endpoint="native", expected_requests=3, method_id="R03-G")
        self.assertTrue(rows[0]["success"])
        self.assertFalse(rows[0]["native_execution_success"])
        payload = refined_payload()
        payload = {"sample_indices": payload["sample_indices"][:1], "num_atoms": payload["num_atoms"][:, :1],
                   "atom_types": payload["atom_types"][:, :2], "frac_coords": payload["frac_coords"][:, :2],
                   "lengths": payload["lengths"][:, :1], "angles": payload["angles"][:, :1]}
        refined, _ = export_records(source, endpoint="tau800", expected_requests=3,
                                     method_id="R03-G", refined_payload=payload)
        self.assertFalse(refined[0]["success"])

    def test_common_evaluator_and_label_input_helpers_need_no_fake_neural_trace(self):
        rows, _ = export_records(ledger(), endpoint="native", expected_requests=3, method_id="R03-R")
        evaluator = load_common_script("evaluate_programmed_paths")
        labels = load_common_script("label_programmed_paths")
        self.assertEqual(evaluator.read_input_structure(rows[0]).composition, labels.structure_from_record(rows[0]).composition)
        with self.assertRaises(ValueError):
            evaluator.read_input_structure(rows[1])
        failed = labels.label_record(rows[1], model=object(), optimizer=object())
        self.assertEqual(failed["status"], "generation_failure")
        self.assertIsNone(failed["raw_energy"])

    def test_source_rows_are_not_renumbered_and_method_endpoint_labels_cannot_mix(self):
        source = ledger()
        swapped = [source[2], source[1], source[0]]
        with self.assertRaises(ValueError):
            export_records(swapped, endpoint="native", expected_requests=3, method_id="R03-R")
        native, _ = export_records(source, endpoint="native", expected_requests=3, method_id="R03-R")
        tau, _ = export_records(source, endpoint="tau800", expected_requests=3, method_id="R03-R", refined_payload=refined_payload())
        self.assertNotEqual(native[0]["trajectory_id"], tau[0]["trajectory_id"])
        self.assertEqual(native[0]["attempt_id"], tau[0]["attempt_id"])

    def test_cli_artifact_has_all_rows_and_no_scoring_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            body = base / "body"
            body.mkdir()
            rows = ledger()
            raw = "".join(json.dumps(row) + "\n" for row in rows)
            (body / "raw_generations.jsonl").write_text(raw)
            (body / "body_attempts.jsonl").write_text(raw)
            (body / "sample_metrics.json").write_text(json.dumps({"schema": "r03_integrated_body_v1",
                                                                  "requested_samples": 3, "denominator": 3}))
            (body / "_SUCCESS").touch()
            output = base / "evaluation"
            main(["--body-dir", str(body), "--endpoint", "native", "--output-dir", str(output),
                  "--expected-requests", "3", "--method-id", "R03-R"])
            report = json.loads((output / "EVALUATION_INPUTS_FINAL.json").read_text())
            self.assertEqual(report["requests"], 3)
            self.assertEqual(report["new_physics_calls"], 0)
            self.assertTrue((output / "_SUCCESS").exists())
            (body / "raw_generations.jsonl").write_text(raw.replace("original-request-17", "changed-request-17"))
            with self.assertRaises(ValueError):
                load_body_directory(body, 3)

    def test_refined_terminal_settings_and_graph_source_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            body, refined = base / "body", base / "refined"
            body.mkdir()
            refined.mkdir()
            path = refined / "dlm_refined_mp_2.pt"
            torch.save(refined_payload(), path)
            config = {"diff_steps": 800, "num_evals": 1, "proposal_graphs": str(body / "proposal_graphs.pt")}
            metrics = {"diff_steps": 800, "num_evals": 1, "assigned_proposals": 2, "output_file": str(path)}
            (refined / "run_config.json").write_text(json.dumps(config))
            (refined / "refinement_metrics.json").write_text(json.dumps(metrics))
            payload, report = load_tau800(path, body_dir=body)
            self.assertEqual(payload["sample_indices"].tolist(), [300, 17])
            self.assertEqual(report["refined_metrics"]["assigned_proposals"], 2)
            config["proposal_graphs"] = str(base / "different_method" / "proposal_graphs.pt")
            (refined / "run_config.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_tau800(path, body_dir=body)
            config.update(proposal_graphs=str(body / "proposal_graphs.pt"), diff_steps=200)
            (refined / "run_config.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_tau800(path, body_dir=body)


def component_fixture(base, component_id, global_idx, planner_seed, *, endpoint="native"):
    body = base / component_id / "body"
    body.mkdir(parents=True)
    row = request(0, global_idx)
    row["planner_record"].update(seed=planner_seed, c3fd_enabled=False, prompt_input_ids=[1, 2, 3])
    raw = json.dumps(row) + "\n"
    for name in ("body_attempts.jsonl", "raw_generations.jsonl"):
        (body / name).write_text(raw)
    (body / "sample_metrics.json").write_text(json.dumps({"schema": "r03_integrated_body_v1", "requested_samples": 1, "denominator": 1}))
    config = {"schema": "r03_integrated_body_v1", "expected_requests": 1, "seed": planner_seed + 1000,
              "post_construction_repair": False, "b0": {"sha256": "fixture_B0"}, "temperature": 0.7,
              "frozen_runtime": {"fixture": "same_runtime"}, "base_model": "same_fixture_base",
              "cfg_scale": 0.0, "remasking": "low_confidence", "geometry_support_scope": "original_constructor_masks_only",
              "repair_checkpoint": None}
    (body / "run_config.json").write_text(json.dumps(config))
    (body / "_SUCCESS").touch()
    spec = {"component_id": component_id, "method_id": "R03-R", "body_dir": str(body),
            "expected_requests": 1, "sample_idx_start": global_idx, "planner_seed": planner_seed}
    prompt = base / component_id / "native_prompt.txt"
    prompt.write_text("unchanged fixture prompt", encoding="utf-8")
    planner_config = {
        "planner_identity": {"adapter_sha256": P0_ADAPTER_SHA256, "adapter_path": "fixture_P0", "base_model_path": "fixture_Llama",
                             "base_config_sha256": "1" * 64, "llama_hidden_size": 4096, "tokenizer_source": "fixture_tokenizer",
                             "tokenizer_size": 128256, "tokenizer_files": {"tokenizer.json": "2" * 64}, "p0_trainable_parameters": 0},
        "native_prompt_file": str(prompt), "native_prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
        "native_prompt_input_ids_sha256": hashlib.sha256(b"[1,2,3]").hexdigest(), "c3fd_domain": None,
        "sampling": {"max_new_tokens": 96, "temperature": 0.9, "top_p": 0.95, "top_k": 50, "do_sample": True},
        "seed": planner_seed, "num_samples": 1, "sample_index_offset": global_idx, "batch_size": 4,
        "seed_mode": "fixture_original_seed_rule", "prompt_style": "fixture_rich", "include_sample_id": False,
        "stop_after_plan_marker": True, "truncate_after_plan_marker": True, "eos_token_id": 0, "pad_token_id": 0,
        "planner_weights_changed": False, "tokenizer_resized": False, "formula_prefill_added": False,
        "external_formula_composer": False, "formula_bridge_source_sha256": "3" * 64, "sampler_source_sha256": "4" * 64,
    }
    planner_file = base / component_id / "planner_run_config.json"
    planner_file.write_text(json.dumps(planner_config))
    spec["planner_run_config"] = {"path": str(planner_file), "sha256": hashlib.sha256(planner_file.read_bytes()).hexdigest()}
    if endpoint == "tau800":
        refined = base / component_id / "refined"
        refined.mkdir()
        path = refined / "dlm_refined_mp_1.pt"
        payload = refined_payload()
        payload = {"sample_indices": torch.tensor([global_idx]), "num_atoms": payload["num_atoms"][:, 1:],
                   "atom_types": payload["atom_types"][:, 2:], "frac_coords": payload["frac_coords"][:, 2:],
                   "lengths": payload["lengths"][:, 1:], "angles": payload["angles"][:, 1:]}
        torch.save(payload, path)
        (refined / "run_config.json").write_text(json.dumps({"diff_steps": 800, "num_evals": 1,
                                                             "checkpoint": "same_frozen_model494",
                                                             "proposal_graphs": str(body / "proposal_graphs.pt")}))
        (refined / "refinement_metrics.json").write_text(json.dumps({"diff_steps": 800, "num_evals": 1,
                                                                      "assigned_proposals": 1, "output_file": str(path)}))
        spec["refined_pt"] = str(path)
        identity = base / component_id / "REFINER_IDENTITY.json"
        identity.write_text(json.dumps({"checkpoint": "same_frozen_model494", "checkpoint_sha256": MODEL494_SHA256, "tau": 800}))
        spec["refiner_identity"] = {"path": str(identity), "sha256": hashlib.sha256(identity.read_bytes()).hexdigest()}
    return spec


class R03ComponentPoolingTests(unittest.TestCase):
    def manifest(self, components, endpoint="native"):
        return {"schema": "r03_evaluation_components_v1", "method_id": "R03-R", "endpoint": endpoint,
                "expected_requests": 2, "components": components}

    def update_planner(self, component, change):
        path = Path(component["planner_run_config"]["path"])
        config = json.loads(path.read_text())
        change(config)
        path.write_text(json.dumps(config))
        component["planner_run_config"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def test_missing_or_conflicting_sampler_identity_is_not_equal_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first, second = component_fixture(base, "seed17", 0, 17), component_fixture(base, "seed18", 1, 18)
            self.update_planner(second, lambda cfg: cfg.update(sampler_source_sha256="5" * 64))
            with self.assertRaisesRegex(ValueError, "identities conflict"):
                export_components(self.manifest([first, second]), endpoint="native", expected_requests=2, method_id="R03-R")
            for component in (first, second):
                self.update_planner(component, lambda cfg: cfg.pop("sampler_source_sha256"))
            with self.assertRaisesRegex(ValueError, "fields are missing"):
                export_components(self.manifest([first, second]), endpoint="native", expected_requests=2, method_id="R03-R")

    def test_enabled_c3fd_domain_bytes_are_verified_not_only_compared(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first, second = component_fixture(base, "seed17", 0, 17), component_fixture(base, "seed18", 1, 18)
            domain_path = base / "C3FD_DOMAIN.json"
            domain_path.write_text(json.dumps({"schema": "fixture_domain", "nodes": {"Li": [1], "O": [-2]},
                                               "max_atoms": 20, "max_species": 2, "allowed_strata": [["oxide", 2, 2]]}))
            domain = {"path": str(domain_path), "sha256": hashlib.sha256(domain_path.read_bytes()).hexdigest(),
                      "schema": "fixture_domain", "max_atoms": 20, "max_species": 2, "symbols": 2, "declared_strata": 1}
            for component in (first, second):
                self.update_planner(component, lambda cfg: cfg.update(c3fd_domain=domain))
                body = Path(component["body_dir"])
                row = json.loads((body / "body_attempts.jsonl").read_text())
                row["planner_record"]["c3fd_enabled"] = True
                for name in ("body_attempts.jsonl", "raw_generations.jsonl"):
                    (body / name).write_text(json.dumps(row) + "\n")
            output, _ = export_components(self.manifest([first, second]), endpoint="native", expected_requests=2, method_id="R03-R")
            self.assertEqual(len(output), 2)
            domain_path.write_text(domain_path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "actual C3FD domain"):
                export_components(self.manifest([first, second]), endpoint="native", expected_requests=2, method_id="R03-R")

    def test_refiner_actual_weight_identity_must_be_frozen_model494(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first = component_fixture(base, "seed17", 0, 17, endpoint="tau800")
            second = component_fixture(base, "seed18", 1, 18, endpoint="tau800")
            path = Path(second["refiner_identity"]["path"])
            record = json.loads(path.read_text())
            record["checkpoint_sha256"] = "7" * 64
            path.write_text(json.dumps(record))
            second["refiner_identity"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "actual refiner identity"):
                export_components(self.manifest([first, second], "tau800"), endpoint="tau800", expected_requests=2, method_id="R03-R")

    def test_two_seed_components_pool_once_in_global_order(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first = component_fixture(base, "seed17", 0, 17)
            second = component_fixture(base, "seed18", 1, 18)
            rows, report = export_components(self.manifest([second, first]), endpoint="native", expected_requests=2, method_id="R03-R")
            self.assertEqual([r["sample_idx"] for r in rows], [0, 1])
            self.assertEqual([r["evaluation_ordinal"] for r in rows], [0, 1])
            self.assertEqual([r["source_component_ordinal"] for r in rows], [0, 0])
            self.assertEqual([r["planner_seed"] for r in rows], [17, 18])
            self.assertEqual([r["original_attempt_id"] for r in rows], ["original-request-0", "original-request-1"])
            self.assertFalse(report["per_component_NU_sums_used"])
            self.assertTrue(all(r["source_files_sha256"] for r in report["components"]))

    def test_two_refiner_payloads_are_joined_by_their_own_global_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first = component_fixture(base, "seed17", 0, 17, endpoint="tau800")
            second = component_fixture(base, "seed18", 1, 18, endpoint="tau800")
            rows, report = export_components(self.manifest([second, first], "tau800"), endpoint="tau800", expected_requests=2, method_id="R03-R")
            self.assertEqual([r["refined_sample_idx"] for r in rows], [0, 1])
            self.assertEqual(report["successful"], 2)
            payload = torch.load(second["refined_pt"], weights_only=False)
            payload["sample_indices"] = torch.tensor([0])  # Forbidden local renumbering of shard2.
            torch.save(payload, second["refined_pt"])
            with self.assertRaises(ValueError):
                export_components(self.manifest([first, second], "tau800"), endpoint="tau800", expected_requests=2, method_id="R03-R")

    def test_cross_arm_and_global_gaps_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first = component_fixture(base, "seed17", 0, 17)
            second = component_fixture(base, "seed18", 2, 18)  # Missing global request1.
            with self.assertRaisesRegex(ValueError, "global pooled"):
                export_components(self.manifest([first, second]), endpoint="native", expected_requests=2, method_id="R03-R")
            second["method_id"] = "R03-P"
            with self.assertRaisesRegex(ValueError, "cross-arm"):
                export_components(self.manifest([first, second]), endpoint="native", expected_requests=2, method_id="R03-R")

    def test_actual_body_policy_identity_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first = component_fixture(base, "seed17", 0, 17)
            second = component_fixture(base, "seed18", 1, 18)
            path = Path(second["body_dir"]) / "run_config.json"
            config = json.loads(path.read_text())
            config["post_construction_repair"] = True
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "identities conflict"):
                export_components(self.manifest([first, second]), endpoint="native", expected_requests=2, method_id="R03-R")


def legacy_panel_fixture(base, *, endpoint="native", arm="control"):
    """Four historical occurrences: two graphs, a Planner failure, a graph failure."""
    cohort, body, seeds, refinement_attempts = [], [], [], []
    policy = "d1" if arm == "control" else "d2_safe_axis"
    for index in range(4):
        eligible, graph_complete = index != 1, index in (0, 2)
        row = request(index, index, ("Na", "Cl") if index == 2 else ("O", "Li"), failure=not eligible)
        plan_state = copy.deepcopy(row["plan_state"]) if eligible else None
        if plan_state is not None:
            plan_state.update(lattice_system="orthorhombic", spacegroup_bucket="sg_016_074",
                              volume_per_atom_bin="volpa_020_024")
        state_text = json.dumps(plan_state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        state_sha = hashlib.sha256(state_text.encode()).hexdigest()
        prompt = "recorded original rich prompt\nplan_state: " + state_text + "\n" if eligible else None
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest() if eligible else None
        cohort.append({
            "schema": "h1a2_epoch2_exactplan_paired_cohort_v1", "cohort_ordinal": index,
            "global_raw_ordinal": index, "planner_attempt_id": row["attempt_id"],
            "planner_rank": 0, "planner_sampling_seed": 17, "repeat": 0,
            "plan_state": plan_state, "plan_state_sha256": state_sha,
            "body_prompt": prompt, "body_prompt_sha256": prompt_sha,
            "body_eligible": eligible, "retry_or_replacement_used": False,
        })
        row.pop("raw_body_text")
        row.pop("parsed")
        row.pop("attempt_status")
        row.pop("planner_record")
        row.pop("plan_state")
        row.pop("body_prompt")
        row.update(schema="h1_body_safeaxis256_attempt_v1", generation_policy=policy,
                   planner_arm="P0", body_checkpoint_arm="B0",
                   schedule_arm="D1" if arm == "control" else "D2_SAFE_AXIS",
                   evaluation_order=index, body_prompt_sha256=prompt_sha, plan_state_sha256=state_sha,
                   filter_used=False, repair_used=False, replacement_used=False, rerank_used=False,
                   retry_used=False, body_graph_complete=graph_complete,
                   status="succeeded" if graph_complete else "failed",
                   earliest_failure_stage=None if graph_complete else "body_graph" if eligible else "planner")
        row["raw_body_text_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest() if eligible else None
        if not eligible:
            row.update(text=None, arrays=None)
        body.append(row)
        seeds.append({
            "schema": "h1a2_epoch2_exactplan1200_paired_seed_ledger_v1",
            "ordinal": index, "raw_ordinal": index, "sample_idx": index, "seed_derivation_ordinal": index,
            "repeat": 0, "paired_across_arms": True, "sampling_seed_root": 17029,
            "body_noise_seed": row["body_noise_seed"], "refiner_noise_seed": 9001 + index,
        })
        refinement_attempts.append({
            "schema": "h1_r03e_refinement_attempt_v1", "ordinal": index, "repeat": 0,
            "attempt_id": row["attempt_id"], "sample_idx": index, "body_noise_seed": row["body_noise_seed"],
            "body_graph_complete": graph_complete, "refiner_complete": graph_complete,
            "refiner_sampling_seed": seeds[-1]["refiner_noise_seed"],
        })
    manifest = {"schema": "h1a2_frozen_legacy_evaluation_v1", "method_id": "historical-" + arm,
                "endpoint": endpoint, "expected_requests": 4, "arm": arm, "files": {}}

    def pin(name, value, *, jsonl=False, binary=False):
        path = base / (name + (".pt" if binary else ".jsonl" if jsonl else ".json"))
        if binary:
            torch.save(value, path)
        else:
            content = "".join(json.dumps(row) + "\n" for row in value) if jsonl else json.dumps(value)
            path.write_text(content, encoding="utf-8")
        manifest["files"][name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return path

    pin("cohort", cohort, jsonl=True)
    body_path = pin("body", body, jsonl=True)
    seed_path = pin("seed_ledger", seeds, jsonl=True)
    if endpoint == "tau800":
        graphs = pin("proposal_graphs", [{"sample_idx": i} for i in (0, 2)], binary=True)
        payload = refined_payload()
        payload.pop('sample_indices')
        payload["sample_idx"] = torch.tensor([2, 0], dtype=torch.long)
        tensor = pin("refined_pt", payload, binary=True)
        common = {"arm": arm, "diff_steps": 800, "num_evals": 1, "effective_batch_size": 1,
                  "filter": False, "repair": False, "replacement": False, "rerank": False,
                  "retry": False, "new_scientific_seed_per_repeat": False, "repeat": 0}
        pin("refiner_config", dict(common, schema="h1_r03e_refiner_run_v1", num_samples=4,
            timesteps=1000, seed_mode="frozen_h1_ordinal_refiner_noise_seed",
            checkpoint_sha256_recorded=MODEL494_SHA256, proposal_graphs=str(graphs),
            body_attempts=str(body_path), attempt_ledger=str(seed_path)))
        pin("refiner_metrics", dict(common, schema="h1_r03e_refiner_metrics_v1", all_attempt_denominator=4,
            status="complete", output_file=str(tensor), refiner_complete=2, body_complete=2))
        pin("refinement_attempts", refinement_attempts, jsonl=True)
    return manifest


class R03LegacyPanelTests(unittest.TestCase):
    def run_export(self, manifest):
        return export_legacy_panel(manifest, endpoint=manifest["endpoint"],
                                   expected_requests=4, method_id=manifest["method_id"])

    def mutate_and_repin(self, manifest, name, mutate):
        path = Path(manifest["files"][name]["path"])
        if path.suffix == ".jsonl":
            value = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            mutate(value)
            text = "".join(json.dumps(row) + "\n" for row in value)
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            mutate(value)
            text = json.dumps(value)
        path.write_text(text, encoding="utf-8")
        manifest["files"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def test_legacy_native_retains_planner_and_graph_failures_without_schema_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = legacy_panel_fixture(Path(directory))
            rows, report = self.run_export(manifest)
            self.assertEqual([r["sample_idx"] for r in rows], [0, 1, 2, 3])
            self.assertEqual([r["evaluation_ordinal"] for r in rows], [0, 1, 2, 3])
            self.assertEqual(report["requests"], 4)
            self.assertFalse(report["source_schema_rewritten"])
            self.assertTrue(report["retrospective_frozen_panel"])
            self.assertTrue(all(r["source_artifact_schema"] == "h1_body_safeaxis256_attempt_v1" for r in rows))
            self.assertFalse(rows[1]["success"])
            self.assertIsNone(rows[1]["structure"])
            self.assertTrue(rows[3]["success"])
            self.assertFalse(rows[3]["native_execution_success"])
            self.assertEqual(rows[1]["refiner_noise_seed"], 9002)
            self.assertTrue(all(not r["trainable_teacher"] for r in rows))

    def test_legacy_refined_tensor_joins_by_global_index_and_retains_all_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = legacy_panel_fixture(Path(directory), endpoint="tau800", arm="candidate")
            rows, report = self.run_export(manifest)
            self.assertEqual(report["successful"], 2)
            self.assertEqual([r["success"] for r in rows], [True, False, True, False])
            self.assertEqual([r["sample_idx"] for r in rows], [0, 1, 2, 3])
            self.assertEqual(list(Structure.from_dict(rows[0]["structure"]).atomic_numbers), [3, 8])
            self.assertEqual(list(Structure.from_dict(rows[2]["structure"]).atomic_numbers), [17, 11])
            self.assertTrue(all(r["body"] is None for r in rows))
            self.assertEqual([r["refiner_noise_seed"] for r in rows], [9001, 9002, 9003, 9004])

    def test_legacy_unrepinned_file_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = legacy_panel_fixture(Path(directory))
            path = Path(manifest["files"]["body"]["path"])
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                self.run_export(manifest)

    def test_legacy_repin_cannot_hide_wrong_policy_or_body_seed(self):
        for field, value in (("generation_policy", "d2_safe_axis"), ("body_noise_seed", 999999)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                manifest = legacy_panel_fixture(Path(directory))
                self.mutate_and_repin(manifest, "body", lambda rows: rows[0].update({field: value}))
                with self.assertRaises(ValueError):
                    self.run_export(manifest)

    def test_legacy_repin_cannot_hide_rich_plan_content_change(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = legacy_panel_fixture(Path(directory))
            self.mutate_and_repin(manifest, "cohort", lambda rows: rows[0]["plan_state"].update(lattice_system="triclinic"))
            with self.assertRaises(ValueError):
                self.run_export(manifest)

    def test_legacy_seed_derivation_and_repeat_identity_cannot_drift(self):
        mutations = (("raw_ordinal", 99), ("seed_derivation_ordinal", 99), ("repeat", 1),
                     ("sampling_seed_root", 17030), ("paired_across_arms", False))
        for field, value in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                manifest = legacy_panel_fixture(Path(directory))
                self.mutate_and_repin(manifest, "seed_ledger", lambda rows: rows[0].update({field: value}))
                with self.assertRaises(ValueError):
                    self.run_export(manifest)

    def test_legacy_matching_raw_ordinals_must_still_bind_the_request_ordinal(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = legacy_panel_fixture(Path(directory))
            self.mutate_and_repin(manifest, "cohort", lambda rows: rows[0].update(global_raw_ordinal=99))
            self.mutate_and_repin(manifest, "seed_ledger", lambda rows: rows[0].update(raw_ordinal=99, seed_derivation_ordinal=99))
            with self.assertRaises(ValueError):
                self.run_export(manifest)

    def test_legacy_refiner_noise_on_failed_request_is_still_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = legacy_panel_fixture(Path(directory), endpoint="tau800")
            self.mutate_and_repin(manifest, "refinement_attempts", lambda rows: rows[1].update(refiner_sampling_seed=99))
            with self.assertRaises(ValueError):
                self.run_export(manifest)

    def test_legacy_refiner_protocol_cannot_gain_repeat_or_batch(self):
        for field, value in (("effective_batch_size", 2), ("num_evals", 2), ("diff_steps", 200), ("repeat", 1)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                manifest = legacy_panel_fixture(Path(directory), endpoint="tau800")
                self.mutate_and_repin(manifest, "refiner_config", lambda cfg: cfg.update({field: value}))
                with self.assertRaises(ValueError):
                    self.run_export(manifest)

    def test_legacy_cohort_reordering_or_lost_failure_cannot_change_denominator(self):
        for name in ("cohort", "body", "seed_ledger"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                manifest = legacy_panel_fixture(Path(directory))
                self.mutate_and_repin(manifest, name, lambda rows: rows.reverse())
                with self.assertRaises(ValueError):
                    self.run_export(manifest)
        with tempfile.TemporaryDirectory() as directory:
            manifest = legacy_panel_fixture(Path(directory), endpoint="tau800")
            self.mutate_and_repin(manifest, "refinement_attempts", lambda rows: rows.pop(1))
            with self.assertRaises(ValueError):
                self.run_export(manifest)

    def test_legacy_graph_success_cannot_contradict_original_generation_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = legacy_panel_fixture(Path(directory))
            self.mutate_and_repin(manifest, "body", lambda rows: rows[0].update(status="failed", earliest_failure_stage="body"))
            with self.assertRaises(ValueError):
                self.run_export(manifest)


if __name__ == "__main__":
    unittest.main()
