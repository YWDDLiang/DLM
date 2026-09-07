"""Synthetic CPU ledger tests; external scoring is replaced with an explicit fake."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_r03_trial_scoring", ROOT / "operations/r03_c3fd_main_20260907/evaluate_trial.py")
trial_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trial_module)


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def component_fixture(base, role, *, count=16, seed=202609070):
    component = base / role
    component.mkdir()
    final = {"role": role, "method_id": "R03_" + role, "requests": count, "planner_seed": seed,
             "body_seed": seed + 100, "refiner_seed": seed + 200, "sample_index_offset": 0,
             "labels_purpose": "evaluation", "pooled_NU_scored_here": False}
    write_json(component / "COMPONENT_FINAL.json", final)
    (component / "_SUCCESS").touch()
    body_dir = component / "body"
    body_dir.mkdir()
    (body_dir / "_SUCCESS").touch()
    write_json(body_dir / "run_config.json", {"max_batch_size": 8, "geometry_support_scope": "post_construction_repair_only"})
    for endpoint in ("native", "tau800"):
        directory, label_dir, direct_dir = component / endpoint, component / (endpoint + "_labels"), component / (endpoint + "_direct")
        for child in (directory, label_dir, direct_dir):
            child.mkdir()
            (child / "_SUCCESS").touch()
        paths, labels, direct = [], [], []
        for i in range(count):
            structure = {"lattice": {"matrix": [[4, 0, 0], [0, 4, 0], [0, 0, 4]]},
                         "sites": [{"species": [{"element": "Li", "occu": 1}], "abc": [i / 100, 0, 0]},
                                   {"species": [{"element": "O", "occu": 1}], "abc": [0.5, 0.5, 0.5]}]}
            row = {"method_id": "R03_" + role, "source_split": "evaluation", "purpose": "evaluation",
                   "endpoint": endpoint, "trajectory_id": f"r03-eval:R03_{role}:{endpoint}:{i}",
                   "attempt_id": f"original:{seed}:{i}", "sample_idx": i, "evaluation_ordinal": i,
                   "group_id": f"eval:{i}", "source_row_idx": i, "planner_seed": seed,
                   "plan_state": {"N": 2, "elements": ["Li", "O"], "counts": [1, 1]},
                   "prompt": "same actually recorded Plan prompt", "body_noise_seed": seed + i,
                   "planner_record": {"seed": seed, "species_program": ["Li", "O"]},
                   "success": True, "parseable": True, "structure": structure}
            label = {"trajectory_id": row["trajectory_id"], "source_split": "evaluation", "endpoint": endpoint,
                     "group_id": row["group_id"], "source_row_idx": i, "status": "verified", "verified": True,
                     "endpoint_cache_key": hashlib.sha256(json.dumps(structure, sort_keys=True).encode()).hexdigest(), "error": None}
            paths.append(row)
            labels.append(label)
            direct.append({"ordinal": i, "attempt_id": row["attempt_id"], "comp_valid": True, "struct_valid": True, "valid": True})
        write_rows(directory / "paths.jsonl", paths)
        write_rows(label_dir / "labels.jsonl", labels)
        write_json(label_dir / "LABEL_FINAL.json", {"purpose": "evaluation", "requested": count, "completed": count,
                                                     "statuses": {"verified": count}, "protocol": {"fixture": "same"},
                                                     "verification_protocol": {"fixture": "same_verification"}})
        write_rows(direct_dir / "attempt_metrics.jsonl", direct)
        write_json(direct_dir / "report.json", {"attempts": count, "generation_succeeded": count,
                                                "comp_valid_count": count, "struct_valid_count": count, "valid_count": count})
    return {"method_id": "R03_" + role, "role": role, "component_dir": str(component), "planner_seed": seed}


def geometry_evidence_fixture(method, *, no_support_first=False):
    component = Path(method["component_dir"])
    final = trial_module.read_json(component / "COMPONENT_FINAL.json")
    enabled = method["role"] in ("G", "P")
    final.update(construction_geometry=enabled, body_batch_size=1)
    write_json(component / "COMPONENT_FINAL.json", final)
    write_json(component / "body/run_config.json", {"construction_geometry_enabled": enabled, "max_batch_size": 1,
                                                     "geometry_support_scope": "construction_and_repair" if enabled else "original_constructor_masks_only"})
    sources = trial_module.read_rows(component / "native/paths.jsonl")
    originals, records = [], []
    for row in sources:
        tokens = [100 + row["sample_idx"]] * 15  # Synthetic identity-only token canvas.
        source = {key: row[key] for key in ("sample_idx", "attempt_id", "body_noise_seed", "plan_state")}
        source.update(body_eligible=True, body_generation_complete=True, raw_body_token_ids=tokens,
                      status="succeeded", attempt_status="complete")
        if enabled:
            source["construction_geometry"] = {"enabled": True}
        original = copy.deepcopy(source)
        if enabled:
            source.update(construction_raw_body_token_ids=tokens.copy(), construction_status="succeeded")
        records.append(source)
        originals.append(original)
    if no_support_first:
        failure = {"failure_class": "construction_constraint_no_legal_support", "reason": "pbc_no_legal_completion",
                   "partial_body_token_ids": [[1, 2, 126336]], "failure_positions": [10]}
        for row in (records[0], originals[0]):
            row.pop("raw_body_token_ids", None)
            row.pop("construction_raw_body_token_ids", None)
            row.update(body_generation_complete=False, attempt_status="construction_constraint_failure",
                       construction_geometry={"enabled": True, "status": "no_legal_support", "failure": copy.deepcopy(failure)},
                       repair_skip_reason="construction_incomplete", repair_used=False)
        for endpoint in ("native", "tau800"):
            paths_path = component / endpoint / "paths.jsonl"
            paths = trial_module.read_rows(paths_path)
            paths[0].update(success=False, parseable=False, structure=None, body=None)
            write_rows(paths_path, paths)
            label_dir = component / (endpoint + "_labels")
            labels = trial_module.read_rows(label_dir / "labels.jsonl")
            labels[0].update(status="generation_failure", verified=False, endpoint_cache_key=paths[0]["trajectory_id"])
            write_rows(label_dir / "labels.jsonl", labels)
            receipt = trial_module.read_json(label_dir / "LABEL_FINAL.json")
            receipt["statuses"] = dict(Counter(row["status"] for row in labels))
            write_json(label_dir / "LABEL_FINAL.json", receipt)
            direct_dir = component / (endpoint + "_direct")
            direct = trial_module.read_rows(direct_dir / "attempt_metrics.jsonl")
            direct[0].update(comp_valid=False, struct_valid=False, valid=False)
            write_rows(direct_dir / "attempt_metrics.jsonl", direct)
            receipt = trial_module.read_json(direct_dir / "report.json")
            for key in ("generation_succeeded", "comp_valid_count", "struct_valid_count", "valid_count"):
                receipt[key] -= 1
            write_json(direct_dir / "report.json", receipt)
    write_rows(component / "body/raw_generations.jsonl", records)
    write_rows(component / "body/construction_raw_generations.jsonl", originals)
    write_json(component / "body/batch_partition.json", [[row["sample_idx"]] for row in sources])


def basic_validity_fixture(method):
    component = Path(method["component_dir"])
    final = trial_module.read_json(component / "COMPONENT_FINAL.json")
    final.update(validity_metrics=["comp_valid", "struct_valid"], joint_valid_reported=False, direct_suite_run=False)
    write_json(component / "COMPONENT_FINAL.json", final)
    for endpoint in ("native", "tau800"):
        legacy = component / (endpoint + "_direct")
        target = component / (endpoint + "_validity")
        target.mkdir()
        (target / "_SUCCESS").touch()
        rows = trial_module.read_rows(legacy / "attempt_metrics.jsonl")
        for row in rows:
            row.pop("valid", None)
        write_rows(target / "attempt_metrics.jsonl", rows)
        report = trial_module.read_json(legacy / "report.json")
        report.pop("valid_count", None)
        report.update(schema="crysllmgen_basic_validity_v1", reported_metrics=["comp_valid", "struct_valid"],
                      omitted_metrics=["joint_valid"])
        write_json(target / "report.json", report)


def trial_fixture(base):
    methods = [component_fixture(base, role) for role in ("G", "P")]
    frozen = base / "frozen_config.json"
    write_json(frozen, {"frozen_code": {"eval_sun_sha256": "f" * 64}})
    manifest = {"schema": "r03_trial_evaluation_manifest_v1", "phase": "canary_repair", "expected_requests": 16,
                "frozen_config": str(frozen), "hull_run_root": str(base / "hull"), "methods": methods}
    path = base / "trial.json"
    write_json(path, manifest)
    return path, manifest


class FakeScorer:
    """Exercise command sequencing and receipts without executing any evaluator."""
    def __init__(self, *, coverage=True):
        self.coverage, self.commands = coverage, []

    def __call__(self, command, name, output):
        self.commands.append(command)
        def argument(flag):
            return command[command.index(flag) + 1]
        if "verify-endpoints" in command:
            write_json(argument("--output-report"), {"coverage_accounted": self.coverage, "is_subset": self.coverage,
                                                       "not_covered": 0 if self.coverage else 1})
            return
        paths = trial_module.read_rows(argument("--paths-jsonl"))
        labels = trial_module.read_json(Path(argument("--labels-jsonl")).parent / "LABEL_FINAL.json")
        role = paths[0]["method_id"][-1]
        strict, meta = (2, 4) if role == "P" else (1, 3)
        rows = []
        for i, row in enumerate(paths):
            rows.append({"trajectory_id": row["trajectory_id"], "sample_idx": row["sample_idx"], "group_id": row["group_id"],
                         "strict_sun": i < strict, "meta_sun": i < meta, "verified_strict_sun": i < strict,
                         "verified_meta_sun": i < meta, "terminal_verified": True, "reconstructed": True,
                         "novel": True, "unique_representative": True, "novel_unique": True,
                         "official_hull_status": "known", "terminal_energy_eV_atom": 0.0,
                         "e_above_hull_eV_atom": 0.0})
        keys = ("strict_sun", "meta_sun", "verified_strict_sun", "verified_meta_sun", "terminal_verified", "reconstructed",
                "novel", "unique_representative", "novel_unique")
        report = {"endpoint": argument("--endpoint"), "cohort_role": argument("--cohort-role"), "policy_stage": argument("--policy-stage"),
                  "counts": {"requests": len(rows), **{key: sum(row[key] for row in rows) for key in keys}},
                  "frozen_nu_source_sha256": "f" * 64, "terminal_protocol": labels["protocol"],
                  "verification_protocol": labels["verification_protocol"], "official_cache": argument("--official-cache")}
        destination = Path(argument("--output-dir"))
        destination.mkdir()
        write_json(destination / "EVALUATION_FINAL.json", report)
        write_rows(destination / "attempt_results.jsonl", rows)
        (destination / "_SUCCESS").touch()


class TrialLedgerTests(unittest.TestCase):
    def test_optional_matched_interface_canary_requires_exact_I_G_P_set(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            path, manifest = trial_fixture(base)
            manifest["include_matched_interface_reference"] = True
            with self.assertRaisesRegex(ValueError, "missing"):
                trial_module.validate_manifest(manifest, path)
            manifest["methods"].append(component_fixture(base, "I"))
            normalized = trial_module.validate_manifest(manifest, path)
            self.assertEqual([row["role"] for row in normalized["methods"]], ["I", "G", "P"])
            del manifest["include_matched_interface_reference"]
            with self.assertRaisesRegex(ValueError, "outside the registered trial"):
                trial_module.validate_manifest(manifest, path)

    def test_matched_interface_geometry_canary_scores_all_six_cells_without_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            path, manifest = trial_fixture(base)
            manifest.update(include_matched_interface_reference=True, registered_construction_geometry=True)
            manifest["methods"].append(component_fixture(base, "I"))
            for method in manifest["methods"]:
                geometry_evidence_fixture(method)
            write_json(path, manifest)
            fake = FakeScorer()
            result = trial_module.evaluate_trial(path, base / "scores", command_runner=fake)
            self.assertEqual(len(fake.commands), 7)  # Coverage once, I/G/P x native/tau800.
            self.assertEqual([(r["role"], r["endpoint"]) for r in result["methods"]],
                             [(role, endpoint) for role in ("I", "G", "P") for endpoint in ("native", "tau800")])
            self.assertTrue(result["include_matched_interface_reference"])
            self.assertTrue(result["G_P_construction_exact_match_checked"])
            self.assertFalse(result["adoption"]["enabled"])
            self.assertIsNone(result["adoption"]["selected_role"])

    def test_missing_methods_cannot_be_scored_as_a_complete_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = trial_fixture(Path(directory))
            manifest["methods"] = manifest["methods"][:1]
            with self.assertRaisesRegex(ValueError, "missing"):
                trial_module.validate_manifest(manifest, path)
            manifest["expected_requests"] = 256
            manifest["validity_artifact"] = "basic_comp_struct_only"
            with self.assertRaisesRegex(ValueError, "missing"):
                trial_module.validate_manifest(manifest, path)

    def test_label_geometry_identity_and_engineering_errors_are_checked_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = trial_fixture(Path(directory))
            trial = trial_module.validate_manifest(manifest, path)
            label_path = Path(manifest["methods"][0]["component_dir"]) / "native_labels/labels.jsonl"
            rows = trial_module.read_rows(label_path)
            rows[0]["endpoint_cache_key"] = "wrong_geometry"
            write_rows(label_path, rows)
            with self.assertRaisesRegex(ValueError, "exact endpoint geometry"):
                trial_module.preflight_components(trial)
            source = trial_module.read_rows(Path(manifest["methods"][0]["component_dir"]) / "native/paths.jsonl")[0]
            rows[0]["endpoint_cache_key"] = trial_module.endpoint_cache_key(source)
            for message in ("RuntimeError: CUDA out of memory", "MemoryError", "CUBLAS_STATUS_ALLOC_FAILED"):
                rows[0]["error"] = message
                write_rows(label_path, rows)
                with self.assertRaisesRegex(ValueError, "OOM"):
                    trial_module.preflight_components(trial)

    def test_nonconverged_scientific_labels_remain_in_the_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = trial_fixture(Path(directory))
            label_dir = Path(manifest["methods"][0]["component_dir"]) / "native_labels"
            rows = trial_module.read_rows(label_dir / "labels.jsonl")
            rows[0].update(status="not_converged", verified=False)
            write_rows(label_dir / "labels.jsonl", rows)
            report = trial_module.read_json(label_dir / "LABEL_FINAL.json")
            report["statuses"] = dict(Counter(row["status"] for row in rows))
            write_json(label_dir / "LABEL_FINAL.json", report)
            result = trial_module.preflight_components(trial_module.validate_manifest(manifest, path))
            self.assertEqual(len(result[0]["cells"]["native"]["rows"]), 16)

    def test_unaccounted_hull_coverage_prevents_every_score_call(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            path, _ = trial_fixture(base)
            fake = FakeScorer(coverage=False)
            with self.assertRaisesRegex(ValueError, "hull coverage"):
                trial_module.evaluate_trial(path, base / "scores", command_runner=fake)
            self.assertEqual(len(fake.commands), 1)
            self.assertIn("verify-endpoints", fake.commands[0])
            self.assertTrue((base / "scores/_FAILED").exists())

    def test_canary_scores_each_endpoint_but_never_adopts_the_better_fake_P(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            path, _ = trial_fixture(base)
            fake = FakeScorer()
            result = trial_module.evaluate_trial(path, base / "scores", command_runner=fake)
            self.assertEqual(len(fake.commands), 5)  # One coverage check, four separate arm/endpoint calls.
            self.assertEqual(len(result["methods"]), 4)
            self.assertFalse(result["adoption"]["enabled"])
            self.assertIsNone(result["adoption"]["selected_role"])
            for command in fake.commands[1:]:
                self.assertIn("evaluate_programmed_paths.py", command[1])
                self.assertNotIn("--selection-json", command)
                self.assertNotIn("label_programmed_paths.py", " ".join(command))
                self.assertEqual(command[command.index("--expected-requests") + 1], "16")
            self.assertTrue((base / "scores/_SUCCESS").exists())

    def test_G_and_P_cannot_change_the_shared_scientific_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = trial_fixture(Path(directory))
            component = Path(manifest["methods"][1]["component_dir"])
            for endpoint in ("native", "tau800"):
                file = component / endpoint / "paths.jsonl"
                rows = trial_module.read_rows(file)
                rows[0]["plan_state"]["counts"] = [2, 1]
                write_rows(file, rows)
            with self.assertRaisesRegex(ValueError, "share the registered Planner request"):
                trial_module.preflight_components(trial_module.validate_manifest(manifest, path))

    def test_new_geometry_registration_rejects_old_method_names_and_changed_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = trial_fixture(Path(directory))
            manifest["registered_construction_geometry"] = True
            with self.assertRaisesRegex(ValueError, "construction geometry"):
                trial_module.preflight_components(trial_module.validate_manifest(manifest, path))
            for method in manifest["methods"]:
                geometry_evidence_fixture(method)
            result = trial_module.preflight_components(trial_module.validate_manifest(manifest, path))
            self.assertEqual(len(result[0]["construction_evidence"]["snapshots"]), 16)
            component = Path(manifest["methods"][1]["component_dir"])
            originals = trial_module.read_rows(component / "body/construction_raw_generations.jsonl")
            rows = trial_module.read_rows(component / "body/raw_generations.jsonl")
            originals[0]["raw_body_token_ids"][2] += 1
            rows[0]["construction_raw_body_token_ids"][2] += 1
            write_rows(component / "body/construction_raw_generations.jsonl", originals)
            write_rows(component / "body/raw_generations.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "G/P actual construction"):
                trial_module.preflight_components(trial_module.validate_manifest(manifest, path))

    def test_common_no_support_failure_keeps_and_compares_actual_partial_canvas(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = trial_fixture(Path(directory))
            manifest["registered_construction_geometry"] = True
            for method in manifest["methods"]:
                geometry_evidence_fixture(method, no_support_first=True)
            result = trial_module.preflight_components(trial_module.validate_manifest(manifest, path))
            self.assertEqual(result[0]["construction_evidence"]["snapshots"][0]["state"], "no_legal_support")
            component = Path(manifest["methods"][1]["component_dir"])
            for name in ("raw_generations.jsonl", "construction_raw_generations.jsonl"):
                rows = trial_module.read_rows(component / "body" / name)
                rows[0]["construction_geometry"]["failure"]["partial_body_token_ids"][0][0] += 1
                write_rows(component / "body" / name, rows)
            with self.assertRaisesRegex(ValueError, "common no-support failures differ"):
                trial_module.preflight_components(trial_module.validate_manifest(manifest, path))

    def test_pilot_requires_all_roles_batch1_and_only_GP_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            methods = [component_fixture(base, role, count=256) for role in ("R", "I", "G", "P")]
            for method in methods:
                geometry_evidence_fixture(method)
                basic_validity_fixture(method)
            manifest = {"schema": trial_module.SCHEMA, "phase": "pilot256", "expected_requests": 256,
                        "validity_artifact": "basic_comp_struct_only", "frozen_config": str(base / "config.json"),
                        "hull_run_root": str(base / "hull"), "methods": methods}
            trial = trial_module.validate_manifest(manifest, base / "manifest.json")
            self.assertTrue(trial["registered_construction_geometry"])
            checked = trial_module.preflight_components(trial)
            self.assertEqual([item["construction_evidence"]["geometry_enabled"] for item in checked], [False, False, True, True])
            config = base / "I/body/run_config.json"
            value = trial_module.read_json(config)
            value["max_batch_size"] = 8
            write_json(config, value)
            with self.assertRaisesRegex(ValueError, "batch1"):
                trial_module.preflight_components(trial)


class AdoptionRuleTests(unittest.TestCase):
    def summaries(self):
        return [{"role": role, "endpoint": endpoint, "validity": {"comp_valid_count": 200, "struct_valid_count": 200},
                 "headline": {"strict_sun": 20, "meta_sun": 110}}
                for role in ("G", "P") for endpoint in ("native", "tau800")]

    def test_only_registered_tau_sun_gates_selection(self):
        rows = self.summaries()
        rows[2]["headline"] = {"strict_sun": 1, "meta_sun": 2}  # Native SUN is diagnostic, not a hidden veto.
        rows[3]["validity"]["struct_valid_count"] = 190  # Validity is reported, not a selection gate.
        rows[3]["headline"]["meta_sun"] = 111
        result = trial_module.adoption_rule("pilot", rows)
        self.assertEqual(result["selected_role"], "P")
        self.assertFalse(result["statistical_significance_claimed"])
        rows[2]["validity"]["struct_valid_count"] = 199
        self.assertEqual(trial_module.adoption_rule("pilot", rows)["selected_role"], "P")

    def test_tie_or_tau_strict_decline_cannot_select_P(self):
        rows = self.summaries()
        self.assertEqual(trial_module.adoption_rule("pilot", rows)["selected_role"], "G")
        rows[3]["headline"].update(strict_sun=19, meta_sun=130)
        self.assertEqual(trial_module.adoption_rule("pilot", rows)["selected_role"], "G")


if __name__ == "__main__":
    unittest.main()
