"""CPU fixtures for hull-reference accounting; no API/model calls occur."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("r03_hull_union_fixture_module", ROOT / "operations/r03_c3fd_main_20260907/prepare_hull_union.py")
H = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = H
SPEC.loader.exec_module(H)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(H.canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def planner_row(index, elements, *, eligible=True):
    return {"sample_idx": index, "body_eligible": eligible, "parsed": eligible,
            "attempt_status": "complete" if eligible else "planner_parse_failure",
            "plan_state": {"N": len(elements), "elements": elements, "counts": [1] * len(elements)}}


def input_manifest(root, sources, *, phase="pilot256", filename="sources.json"):
    inputs = []
    for index, (arm, kind, rows) in enumerate(sources):
        path = root / f"{filename}.{arm}.{index}.jsonl"
        write_rows(path, rows)
        inputs.append({"cell_id": f"{arm}_seed{index}", "arm": arm, "seed": index,
                       "type": kind, "path": str(path), "expected_requests": len(rows)})
    path = root / filename
    write_json(path, {"schema": H.INPUT_SCHEMA, "purpose": "evaluation", "phase": phase, "inputs": inputs})
    return path


def cache_fixture(root, resolved=(), errors=None, *, database="2026.04.13", query_inputs=None):
    root.mkdir(parents=True)
    errors = errors or {}
    entries = {name: [{"entry_id": f"fixture-{symbol}", "composition": {symbol: 1}, "energy": -1.0}
                      for symbol in name.split("-")] for name in resolved}
    all_systems = sorted(set(resolved) | set(errors))
    audits, unknown = [], []
    for index, name in enumerate(all_systems):
        row = {"schema": "h1_official_mp_gga_u_chemsys_v1", "query_index": index, "query_total": len(all_systems),
               "chemsys": name, "elements": name.split("-"), **H.THERMO, "manual_transport_attempts": 1}
        if name in entries:
            row.update(query_status="resolved", entry_count=len(entries[name]),
                       slim_entries_sha256=H.canonical_sha256(entries[name]), phase_diagram_constructed=True,
                       unary_reference_elements=name.split("-"), error=None)
        else:
            row.update(query_status="query_error", entry_count=None, slim_entries_sha256=None,
                       phase_diagram_constructed=False, error=errors[name])
            unknown.append({"chemsys": name, "elements": name.split("-"),
                            "reason": "fresh_official_query_unresolved", "error": errors[name]})
        audits.append(row)
    write_rows(root / "official_slim_cache.jsonl", [{"chemsys": name, "entries": entries[name]} for name in sorted(entries)])
    write_rows(root / "unresolved_chemsys.jsonl", unknown)
    write_rows(root / "query_audit.jsonl", audits)
    manifest = {"schema": H.CLEAN_SCHEMA, **H.THERMO, "database_version": database,
                "query_status": "complete_with_explicit_unresolved", "fresh_empty_cache": True,
                "query_count": len(all_systems), "query_resolved": len(entries), "query_unresolved": len(unknown),
                "all_resolved_phase_diagrams_constructed": True, "all_resolved_elemental_references_present": True,
                "outputs": {key: H.identity(root / filename) for key, filename in H.CACHE_OUTPUTS.items()}}
    if query_inputs is not None:
        manifest["inputs"] = {"wanted_chemsys": H.identity(query_inputs / "wanted_chemsys.jsonl"),
                              "input_manifest": H.identity(query_inputs / "input_manifest.json")}
    write_json(root / "completion_manifest.json", manifest)
    (root / "completion_SUCCESS").touch()
    return root


class HullUnionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = self.root / "query_config.json"
        write_json(self.config, {"thermo": {**H.THERMO, "fresh_empty_cache": True,
                    "reuse_any_historical_or_august_cache": False},
                    "runtime": {"official_mp_python": "/fixture/official/python"}})

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, sources, caches=(), *, name="union", phase="pilot256"):
        source_manifest = input_manifest(self.root, sources, phase=phase, filename=f"{name}.sources.json")
        run = self.root / name
        report = H.prepare(inputs_manifest=source_manifest, known_caches=caches, query_config=self.config,
                           query_source=ROOT / "eval_runtime", run_root=run)
        return run, report

    def test_planner_upper_bound_retains_failures_and_uses_both_arms(self):
        source = input_manifest(self.root, [
            ("R", "planner", [planner_row(0, ["Na", "Cl"]), planner_row(1, ["Li", "F"], eligible=False)]),
            ("M", "planner", [planner_row(10, ["Li", "F"]), planner_row(11, ["Na", "Cl"])]),
        ])
        wanted, report = H.collect_inputs(source)
        self.assertEqual([row["chemsys"] for row in wanted], ["Cl-Na", "F-Li"])
        self.assertEqual(sum(cell["observed_requests"] for cell in report["cells"]), 4)
        self.assertEqual(report["cells"][0]["composition_counts"]["planner_body_ineligible"], 1)
        self.assertTrue(report["planner_upper_bound_present"])
        self.assertFalse(report["generated_energy_or_SUN_used_for_selection"])

    def test_cohort_sizes_are_declared_and_not_hardcoded(self):
        for count in (16, 128, 256, 1000):
            with self.subTest(count=count):
                source = input_manifest(self.root, [("R", "planner", [planner_row(i, ["Na", "Cl"]) for i in range(count)])],
                                        filename=f"sources{count}.json")
                _, report = H.collect_inputs(source)
                self.assertEqual(report["cells"][0]["observed_requests"], count)

    def test_input_count_or_training_scope_mismatch_is_rejected(self):
        source = input_manifest(self.root, [("R", "planner", [planner_row(0, ["Na", "Cl"])])])
        manifest = H.read_json(source)
        manifest["inputs"][0]["expected_requests"] = 2
        write_json(source, manifest)
        with self.assertRaisesRegex(H.HullUnionError, "denominator"):
            H.collect_inputs(source)
        with self.assertRaisesRegex(H.HullUnionError, "training"):
            H.extract_chemsys({**planner_row(0, ["Na", "Cl"]), "source_split": "train"}, "planner")

    def test_official_reference_absence_is_distinct_from_transport_failure(self):
        official = {"type": "ContractError", "http_status": None, "message": "missing unary references: ['Yb']"}
        transport = {"type": "ReadTimeout", "http_status": None, "message": "timed out"}
        cached = cache_fixture(self.root / "known", ["Cl-Na"], {"Li-Yb": official, "F-Li": transport})
        loaded = H.load_cache(cached)
        self.assertEqual(set(loaded["official_unresolved"]), {"Li-Yb"})
        self.assertEqual(set(loaded["query_errors"]), {"F-Li"})
        run, report = self.prepare([("R", "planner", [planner_row(0, ["Na", "Cl"]), planner_row(1, ["Li", "F"]),
                                                       planner_row(2, ["Li", "Yb"])])], [cached])
        self.assertEqual(report["missing_chemsys"], ["F-Li"])
        self.assertEqual(report["known_official_unresolved"], 1)
        self.assertEqual(report["historical_query_error_records_excluded"], 1)
        self.assertEqual(H.read_jsonl(run / "missing_query/inputs/wanted_chemsys.jsonl"),
                         [{"query_index": 0, "chemsys": "F-Li", "elements": ["F", "Li"]}])

    def test_cache_hash_thermo_and_record_bindings_are_validated(self):
        cached = cache_fixture(self.root / "known", ["Cl-Na"])
        manifest = H.read_json(cached / "completion_manifest.json")
        manifest["thermo_type"] = "R2SCAN"
        write_json(cached / "completion_manifest.json", manifest)
        with self.assertRaisesRegex(H.HullUnionError, "fixed official"):
            H.load_cache(cached)
        manifest["thermo_type"] = H.THERMO["thermo_type"]
        write_json(cached / "completion_manifest.json", manifest)
        with (cached / "official_slim_cache.jsonl").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(H.HullUnionError, "bytes/SHA"):
            H.load_cache(cached)

    def test_missing_query_files_are_accepted_by_unchanged_protocol_manifest_reader(self):
        run, report = self.prepare([("R", "planner", [planner_row(0, ["Na", "Cl"])])])
        source = run / "query_source"
        with patch.dict(os.environ, {"H1_ACTIVE_DENOMINATOR": "256"}):
            spec = importlib.util.spec_from_file_location("hull_query_protocol_fixture", source / "protocol.py")
            protocol = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(protocol)
        protocol.require_source_manifest(source, report["query_command"]["source_manifest_sha256"])
        query_inputs = run / "missing_query/inputs"
        manifest = H.read_json(query_inputs / "input_manifest.json")
        rows = H.read_jsonl(query_inputs / "wanted_chemsys.jsonl")
        self.assertEqual(manifest["wanted_chemsys_sha256"], protocol.canonical_sha256(rows))
        self.assertEqual((source / "query_official_mp.py").read_bytes(), (ROOT / "eval_runtime/query_official_mp.py").read_bytes())
        self.assertEqual(report["query_command"]["environment"], {"H1_ACTIVE_DENOMINATOR": "256"})

    def test_finalize_combines_known_fresh_and_official_unresolved_at_one_database_version(self):
        absent = {"type": "ContractError", "http_status": None, "message": "missing unary references: ['Yb']"}
        cached = cache_fixture(self.root / "known", ["Cl-Na"], {"Li-Yb": absent})
        run, _ = self.prepare([("R", "planner", [planner_row(0, ["Na", "Cl"])]),
                               ("M", "planner", [planner_row(0, ["Li", "F"]), planner_row(1, ["Li", "Yb"])])], [cached])
        cache_fixture(run / "missing_query/official_mp_cache", ["F-Li"], query_inputs=run / "missing_query/inputs")
        result = H.finalize(run_root=run)
        self.assertTrue(result["coverage_accounted"])
        self.assertEqual(result["not_covered"], 0)
        self.assertEqual(result["database_version"], "2026.04.13")
        self.assertFalse(result["fresh_empty_cache"])
        self.assertTrue(result["actual_endpoint_subset_check_required"])
        loaded = H.load_cache(run / "official_mp_cache")
        self.assertEqual(set(loaded["resolved"]), {"Cl-Na", "F-Li"})
        self.assertEqual(set(loaded["official_unresolved"]), {"Li-Yb"})
        self.assertEqual(loaded["resolved"]["Cl-Na"], H.load_cache(cached)["resolved"]["Cl-Na"])

    def test_transport_failure_cannot_publish_completion_or_fake_official_unknown(self):
        run, _ = self.prepare([("R", "planner", [planner_row(0, ["Li", "F"])])])
        cache_fixture(run / "missing_query/official_mp_cache", [], {"F-Li": {"type": "MPRestError", "http_status": 503,
                      "message": "upstream unavailable"}}, query_inputs=run / "missing_query/inputs")
        result = H.finalize(run_root=run)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["not_covered"], 1)
        self.assertEqual(result["transport_errors_promoted_to_official_unresolved"], 0)
        self.assertFalse((run / "official_mp_cache").exists())

    def test_database_version_drift_blocks_mixed_reference_cache(self):
        cached = cache_fixture(self.root / "known", ["Cl-Na"])
        run, _ = self.prepare([("R", "planner", [planner_row(0, ["Na", "Cl"]), planner_row(1, ["Li", "F"])])], [cached])
        cache_fixture(run / "missing_query/official_mp_cache", ["F-Li"], database="2026.09.01",
                      query_inputs=run / "missing_query/inputs")
        result = H.finalize(run_root=run)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("entire_registered_union", result["database_version_conflict"]["required_action"])
        self.assertFalse((run / "official_mp_cache/completion_SUCCESS").exists())

    def test_actual_endpoints_must_be_subset_of_registered_planner_union(self):
        cached = cache_fixture(self.root / "known", ["Cl-Na"])
        run, _ = self.prepare([("R", "planner", [planner_row(0, ["Na", "Cl"])])], [cached])
        H.finalize(run_root=run)
        actual = input_manifest(self.root, [("R", "eval_paths", [
            {"sample_idx": 0, "source_split": "evaluation", "endpoint": "native", "parseable": True,
             "structure": {"sites": [{"species": [{"element": "Na", "occu": 1} ]},
                                      {"species": [{"element": "Cl", "occu": 1}]}]}}
        ])], filename="actual.json")
        report = H.verify_endpoints(run_root=run, actual_inputs=actual, output_report=self.root / "check.json")
        self.assertTrue(report["coverage_accounted"])
        records = H.read_jsonl(Path(H.read_json(actual)["inputs"][0]["path"]))
        records[0]["structure"]["sites"][0]["species"][0]["element"] = "Li"
        write_rows(Path(H.read_json(actual)["inputs"][0]["path"]), records)
        report = H.verify_endpoints(run_root=run, actual_inputs=actual, output_report=self.root / "outside.json")
        self.assertFalse(report["coverage_accounted"])
        self.assertEqual(report["outside_registered_union"], ["Cl-Li"])

    def test_missing_refined_structure_never_falls_back_to_native_body(self):
        name, reason = H.extract_chemsys({"sample_idx": 0, "source_split": "evaluation", "endpoint": "tau800",
                                         "body": "<N_002><E_Na><E_Cl>"}, "eval_paths")
        self.assertIsNone(name)
        self.assertIn("no_native_substitution", reason)

    def test_finalize_counts_actual_endpoint_escape_as_uncovered(self):
        cached = cache_fixture(self.root / "known", ["Cl-Na"])
        run, _ = self.prepare([("R", "planner", [planner_row(0, ["Na", "Cl"])])], [cached])
        actual = input_manifest(self.root, [("R", "body", [{"sample_idx": 0, "arrays": {"species": ["Li", "F"]}}])],
                                filename="escaped_actual.json")
        result = H.finalize(run_root=run, actual_inputs=actual)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["not_covered"], 1)
        self.assertEqual(result["uncovered_chemsys"], ["F-Li"])
        self.assertFalse((run / "official_mp_cache").exists())

    def test_prep_rejects_inputs_modified_after_freeze(self):
        cached = cache_fixture(self.root / "known", ["Cl-Na"])
        run, prepared = self.prepare([("R", "planner", [planner_row(0, ["Na", "Cl"])])], [cached])
        path = Path(prepared["input_sources"]["cells"][0]["source"]["path"])
        with path.open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(H.HullUnionError, "bytes/SHA"):
            H.finalize(run_root=run)


if __name__ == "__main__":
    unittest.main()
