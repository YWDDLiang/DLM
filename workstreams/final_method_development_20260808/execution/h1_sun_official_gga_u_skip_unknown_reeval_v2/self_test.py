#!/usr/bin/env python3
"""Offline contract test for the skip-unknown S.U.N. repair source."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry

from assemble import exact_mcnemar
from protocol import ContractError, cell_specs, read_json, require_source_manifest
from reevaluate_cell import exact_hull, load_unresolved


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
    cells = cell_specs(config)
    if len(cells) != 16 or len({row["cell_id"] for row in cells}) != 16:
        raise ContractError("expected exactly 16 distinct V2 cells")
    adoption = config.get("cache_adoption") or {}
    if (
        int(adoption.get("expected_query_count", -1)) != 2630
        or int(adoption.get("expected_resolved_count", -1)) != 2550
        or int(adoption.get("expected_unresolved_count", -1)) != 80
        or int(adoption.get("new_mp_queries", -1)) != 0
        or adoption.get("accepted_unresolved_error_message")
        != "missing unary references: ['Yb']"
    ):
        raise ContractError("frozen cache-adoption counts changed")
    if config["thermo"].get("unresolved_policy") != "explicit_hull_unknown_excluded_from_skip_unknown_denominators":
        raise ContractError("skip-unknown denominator policy changed")

    serialized = "\n".join(
        (source / name).read_text(encoding="utf-8")
        for name in (
            "CONFIG.json",
            "adopt_failed_spool.py",
            "collect_inputs.py",
            "reevaluate_cell.py",
            "prepare_and_submit_once.sh",
        )
    )
    forbidden = (
        "MaterialsProject2020Compatibility",
        "/materials/thermo/",
        "MP_API_KEY",
        "PMG_MAPI_KEY",
        "MAPI_KEY",
        "get_entries_in_chemsys(",
    )
    if any(value in serialized for value in forbidden):
        raise ContractError("repair source can issue a query or use a legacy path")

    entries = [
        ComputedEntry("Li", -1.0, entry_id="Li"),
        ComputedEntry("O2", -4.0, entry_id="O"),
        ComputedEntry("Li2O", -7.0, entry_id="Li2O"),
    ]
    diagram = PhaseDiagram(entries)
    if exact_hull(diagram, entries[2].composition, -3.5) != 0.0:
        raise ContractError("synthetic exact-hull test failed")

    large = exact_mcnemar(
        [True] * 1500 + [False] * 1500,
        [False] * 1500 + [True] * 1500,
    )
    if large["discordant"] != 3000 or not (0.999999 <= large["two_sided_exact_p"] <= 1.0):
        raise ContractError("large exact-McNemar self-test failed")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "unknown.jsonl"
        path.write_text(
            json.dumps(
                {
                    "query_index": 0,
                    "chemsys": "Li-Yb",
                    "elements": ["Li", "Yb"],
                    "reason": "official_gga_gga_u_missing_yb_unary_reference",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        if set(load_unresolved(path)) != {"Li-Yb"}:
            raise ContractError("unresolved-cache self-test failed")
    print("official_mp_skip_unknown_self_test=PASS")


if __name__ == "__main__":
    main()
