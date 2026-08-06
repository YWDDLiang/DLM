#!/usr/bin/env python3
"""Stage and verify the complete isolated runtime for the A100 S.U.N. runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


COPIED_RUNTIME_FILES = (
    "crystal_dlm/__init__.py",
    "crystal_dlm/wqcodiff/contracts.py",
    "crystal_dlm/wqcodiff/crysllmgen/__init__.py",
    "crystal_dlm/wqcodiff/crysllmgen/epoch_training.py",
    "crystal_dlm/wqcodiff/crysllmgen/a100_sun.py",
    "scripts/a800/run_crysllmgen_a100_sun.py",
)
WQCODIFF_PACKAGE_MARKER = '''"""Minimal isolated package marker for S.U.N. evaluation."""

__all__: list[str] = []
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _install_exact(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or source.read_bytes() != destination.read_bytes():
            raise FileExistsError(
                f"refusing to replace non-identical runtime file: {destination}"
            )
        return
    shutil.copy2(source, destination)


def stage_runtime(project_root: Path, runtime_root: Path) -> dict[str, object]:
    project = project_root.resolve()
    runtime = runtime_root.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    for relative in COPIED_RUNTIME_FILES:
        _install_exact(project / relative, runtime / relative)

    marker = runtime / "crystal_dlm/wqcodiff/__init__.py"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker_bytes = WQCODIFF_PACKAGE_MARKER.encode("utf-8")
    if marker.exists():
        if not marker.is_file() or marker.read_bytes() != marker_bytes:
            raise FileExistsError(
                f"refusing to replace non-identical runtime marker: {marker}"
            )
    else:
        marker.write_bytes(marker_bytes)

    verification = verify_runtime(runtime)
    files = {
        relative: _sha256(runtime / relative)
        for relative in (
            *COPIED_RUNTIME_FILES,
            "crystal_dlm/wqcodiff/__init__.py",
        )
    }
    return {
        "schema": "crysllmgen_a100_sun_runtime_stage_v1",
        "status": "pass",
        "runtime_root": str(runtime),
        "files": files,
        "import_preflight": verification,
    }


def verify_runtime(runtime_root: Path) -> dict[str, object]:
    runtime = runtime_root.resolve()
    runner = runtime / "scripts/a800/run_crysllmgen_a100_sun.py"
    required = (
        *COPIED_RUNTIME_FILES,
        "crystal_dlm/wqcodiff/__init__.py",
    )
    missing = [relative for relative in required if not (runtime / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"runtime dependency closure is incomplete: {missing}")

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
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "isolated runtime import preflight failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"unexpected import preflight output: {completed.stdout!r}")
    report = json.loads(lines[0])
    if report.get("status") != "pass":
        raise RuntimeError(f"runtime import preflight did not pass: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify_runtime(args.runtime_root)
    else:
        if args.project_root is None:
            parser.error("--project-root is required unless --verify-only is used")
        result = stage_runtime(args.project_root, args.runtime_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
