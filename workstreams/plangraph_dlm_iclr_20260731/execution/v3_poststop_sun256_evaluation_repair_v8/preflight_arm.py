#!/usr/bin/env python3
"""Validate the frozen v7 input and isolated S.U.N. runtime before evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from protocol import (
    require_source_manifest,
    sha256_file,
    validate_arm,
    verify_frozen_arm,
    write_json_exclusive,
)


REQUIRED_ORIGINS = {
    "runner": "scripts/a800/run_crysllmgen_a100_sun.py",
    "crystal_dlm": "crystal_dlm/__init__.py",
    "crystal_dlm.wqcodiff": "crystal_dlm/wqcodiff/__init__.py",
    "crystal_dlm.wqcodiff.contracts": "crystal_dlm/wqcodiff/contracts.py",
    "crystal_dlm.wqcodiff.crysllmgen": (
        "crystal_dlm/wqcodiff/crysllmgen/__init__.py"
    ),
    "crystal_dlm.wqcodiff.crysllmgen.epoch_training": (
        "crystal_dlm/wqcodiff/crysllmgen/epoch_training.py"
    ),
    "crystal_dlm.wqcodiff.crysllmgen.a100_sun": (
        "crystal_dlm/wqcodiff/crysllmgen/a100_sun.py"
    ),
}


def isolated_runtime_preflight(runtime_root: Path) -> dict[str, object]:
    runtime = runtime_root.resolve()
    runner = runtime / "scripts/a800/run_crysllmgen_a100_sun.py"
    bootstrap = """
import runpy
import sys
from pathlib import Path

runtime = Path(sys.argv[1]).resolve()
runner = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(runtime))
sys.argv = [str(runner), "--preflight-runtime-root", str(runtime)]
runpy.run_path(str(runner), run_name="__main__")
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            bootstrap,
            str(runtime),
            str(runner),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "isolated runtime preflight failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"unexpected runtime preflight output: {completed.stdout!r}")
    report = json.loads(lines[0])
    if report.get("status") != "pass" or report.get("origins") != REQUIRED_ORIGINS:
        raise RuntimeError(f"runtime import origins changed: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()

    arm = validate_arm(args.arm)
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    frozen = verify_frozen_arm(args.input_manifest.resolve(), arm)
    runtime_report = isolated_runtime_preflight(source / "runtime")
    credential_names = ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY")
    present_credentials = [
        name for name in credential_names if bool(os.environ.get(name))
    ]
    if present_credentials:
        raise RuntimeError(
            f"MP credentials must be absent for frozen-cache evaluation: "
            f"{present_credentials}"
        )
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    report = {
        "schema": "h1a2_v3_sun_evaluation_repair_preflight_v1",
        "status": "pass",
        "arm": arm,
        "method": frozen["method"],
        "attempts": 256,
        "generation_succeeded": frozen["counts"]["generation_succeeded"],
        "all_successes_diffusion_refined": True,
        "diffusion_steps": 800,
        "frozen_input_manifest_sha256": sha256_file(args.input_manifest.resolve()),
        "runtime_import_preflight": runtime_report,
        "mp_credentials_present": False,
        "mp_api_enabled": False,
        "source_manifest_sha256": args.source_manifest_sha256,
        "generation_or_refinement_rerun": False,
        "direct_metrics_rerun": False,
    }
    write_json_exclusive(output / "preflight_report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
