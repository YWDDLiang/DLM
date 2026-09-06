#!/usr/bin/env python3
"""Freeze composition-only official inputs for a previously frozen cohort."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-code", type=Path, required=True)
    parser.add_argument("--planner-records", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_code / "eval_runtime"))
    import protocol
    spec = importlib.util.spec_from_file_location("collect_planner_compositions", args.source_code / "scripts/collect_plan_official_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = protocol.read_jsonl(args.planner_records)
    wanted, failures = module.collect(rows, expected_requested=1200)
    args.run_root.mkdir(parents=True, exist_ok=False)
    inputs = args.run_root / "inputs"
    inputs.mkdir()
    systems = [{"query_index": i, "chemsys": name, "elements": name.split("-")}
               for i, name in enumerate(sorted(wanted))]
    protocol.write_jsonl_exclusive(inputs / "wanted_chemsys.jsonl", systems)
    protocol.write_json_exclusive(inputs / "input_manifest.json", {
        "schema": "frozen_planner_cohort_official_coverage_repair_v1",
        "requested": 1200, "planner_failures_retained": failures,
        "wanted_chemsys_count": len(systems), "wanted_chemsys_sha256": protocol.canonical_sha256(systems),
        "planner_records": protocol.identity(args.planner_records),
        "selection": "all_valid_compositions_in_frozen_Planner_records",
        "timing": "cache_coverage_repair_after_initial_evaluation",
        "generated_structure_or_energy_read": False, "outcome_selection": False,
    })
    (inputs / "inputs_SUCCESS").touch()
    protocol.write_source_manifest(inputs, ("wanted_chemsys.jsonl", "input_manifest.json", "inputs_SUCCESS"))
    source = args.run_root / "source"
    source.mkdir()
    for name in ("protocol.py", "query_official_mp.py"):
        shutil.copyfile(args.source_code / "eval_runtime" / name, source / name)
    shutil.copyfile(args.config, source / "CONFIG.json")
    shutil.copyfile(Path(__file__).with_name("launch_official_query.py"), source / "launch_official_query.py")
    manifest = protocol.write_source_manifest(source, ("protocol.py", "query_official_mp.py", "CONFIG.json", "launch_official_query.py"))
    (args.run_root / "SOURCE_MANIFEST_SHA256").write_text(protocol.sha256_file(manifest) + "\n")
    (args.run_root / "WAITING_FOR_CREDENTIAL").touch()
    print(json.dumps({"run_root": str(args.run_root), "requested": 1200,
                      "wanted_chemsys": len(wanted), "planner_failures": failures,
                      "credential_read": False, "generated_geometry_or_energy_read": False}))


if __name__ == "__main__":
    main()
