#!/usr/bin/env python3
"""Offline contract test for the clean stability repair source."""

from __future__ import annotations

import argparse
import inspect
import platform
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry

from protocol import ContractError, cell_specs, read_json, require_source_manifest
from query_official_mp import (
    THERMO_CRITERIA,
    MPRester,
    slim_entries,
    validate_reference_set,
)
from assemble import exact_mcnemar
from reevaluate_cell import exact_hull


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    for path in sorted(source.glob("*.py")):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

    config = read_json(source / "CONFIG.json")
    runtime = config.get("runtime") or {}
    expected_python = str(runtime.get("official_mp_python") or "")
    if sys.executable != expected_python:
        raise ContractError(
            f"official MP interpreter changed: {sys.executable!r} != {expected_python!r}"
        )
    expected_python_version = str(runtime.get("official_mp_python_version") or "")
    if platform.python_version() != expected_python_version:
        raise ContractError(
            "official MP Python version changed: "
            f"{platform.python_version()!r} != {expected_python_version!r}"
        )
    expected_packages = runtime.get("official_mp_packages") or {}
    for package, expected in expected_packages.items():
        try:
            observed = version(str(package))
        except PackageNotFoundError as exc:
            raise ContractError(f"official MP package is missing: {package}") from exc
        if observed != str(expected):
            raise ContractError(
                f"official MP package changed: {package}={observed!r} != {expected!r}"
            )
    cells = cell_specs(config)
    if len(cells) != 16 or len({row["cell_id"] for row in cells}) != 16:
        raise ContractError("expected exactly 16 distinct V1 cells")
    serialized = "\n".join(
        (source / name).read_text(encoding="utf-8")
        for name in (
            "CONFIG.json",
            "query_official_mp.py",
            "collect_inputs.py",
            "reevaluate_cell.py",
        )
    )
    forbidden = (
        "mp_hull_entries_cache_merged_slim_plus_mpapi",
        "completed_mp_hull_cache",
        "MaterialsProject2020Compatibility",
        "/materials/thermo/",
    )
    if any(value in serialized for value in forbidden):
        raise ContractError("operational source references a forbidden legacy path")
    signature = inspect.signature(MPRester.get_entries_in_chemsys)
    if not {"compatible_only", "additional_criteria"}.issubset(signature.parameters):
        raise ContractError("installed official MPRester signature is incompatible")
    if THERMO_CRITERIA != {"thermo_types": ["GGA_GGA+U"]}:
        raise ContractError("legacy-parity official thermo criterion changed")

    entries = [
        ComputedEntry("Li", -1.0, entry_id="Li"),
        ComputedEntry("O2", -4.0, entry_id="O"),
        ComputedEntry("Li2O", -7.0, entry_id="Li2O"),
    ]
    result = validate_reference_set(entries, ["Li", "O"])
    if not result["phase_diagram_constructed"] or len(slim_entries(entries)) != 3:
        raise ContractError("synthetic complete-reference test failed")
    diagram = PhaseDiagram(entries)
    if exact_hull(diagram, entries[2].composition, -3.5) != 0.0:
        raise ContractError("synthetic exact-hull test failed")
    try:
        validate_reference_set(entries[1:], ["Li", "O"])
    except ContractError as exc:
        if "missing unary" not in str(exc):
            raise
    else:
        raise ContractError("missing-unary self-test did not fail")

    large = exact_mcnemar(
        [True] * 1500 + [False] * 1500,
        [False] * 1500 + [True] * 1500,
    )
    if large["discordant"] != 3000 or not (0.999999 <= large["two_sided_exact_p"] <= 1.0):
        raise ContractError("large exact-McNemar self-test failed")

    # Ensure exclusive temporary creation works on the active filesystem.
    with tempfile.TemporaryDirectory() as directory:
        marker = Path(directory) / "ok"
        marker.touch(exist_ok=False)
    print("official_mp_clean_self_test=PASS")


if __name__ == "__main__":
    main()
