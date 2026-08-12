#!/usr/bin/env python3
"""Freeze V8's all-reconstructed chemical systems for login-node MP prequery."""

from __future__ import annotations

import argparse
from pathlib import Path

from protocol import (
    ContractError,
    canonical_sha256,
    identity,
    read_json,
    read_jsonl,
    require_source_manifest,
    write_json_exclusive,
    write_jsonl_exclusive,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(source / "CONFIG.json")
    spec = config["prequery_all_reconstructed"]
    source_path = Path(spec["source_path"]).resolve()
    rows = read_jsonl(source_path)
    chemsys = [str(row.get("chemsys")) for row in rows]
    if (
        len(rows) != int(spec["count"])
        or len(set(chemsys)) != len(rows)
        or canonical_sha256(rows) != spec["canonical_sha256"]
        or [int(row.get("query_index", -1)) for row in rows]
        != list(range(len(rows)))
        or any(row.get("elements") != str(row["chemsys"]).split("-") for row in rows)
    ):
        raise ContractError("V8 all-reconstructed prequery inventory changed")

    run_root = args.run_root.resolve()
    workspace = run_root / "prequery_workspace"
    if workspace.exists():
        raise FileExistsError(workspace)
    (workspace / "inputs").mkdir(parents=True)
    (workspace / "status").mkdir()
    wanted_path = workspace / "inputs/wanted_chemsys.jsonl"
    write_jsonl_exclusive(wanted_path, rows)
    manifest = {
        "schema": "h1a2_v8_all_reconstructed_official_prequery_input_v1",
        "status": "complete",
        "source_manifest_sha256": args.source_manifest_sha256,
        "evaluated_stage": "post_model494_only",
        "pre_refine_evaluated": False,
        "wanted_chemsys_count": len(rows),
        "wanted_chemsys_sha256": canonical_sha256(rows),
        "wanted_chemsys": identity(wanted_path),
        "v8_source": identity(source_path),
        "role": spec["role"],
    }
    write_json_exclusive(workspace / "inputs/input_manifest.json", manifest)
    (workspace / "status/preliminary_assembly_SUCCESS").touch(exist_ok=False)
    write_json_exclusive(run_root / "status/prequery_inputs_report.json", manifest)
    (run_root / "status/prequery_inputs_SUCCESS").touch(exist_ok=False)
    print({"prequery_inputs": "PASS", "wanted_chemsys": len(rows)})


if __name__ == "__main__":
    main()
