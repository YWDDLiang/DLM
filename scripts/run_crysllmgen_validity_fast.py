#!/usr/bin/env python3
"""Predicted-only CrysLLMGen validity screen without redundant GT fingerprints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({str(row["attempt_id"]) for row in rows}) != len(rows):
        raise ValueError("generation attempts must be nonempty with unique attempt_id")
    return rows


def structure_array(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from pymatgen.core import Structure

    structure = Structure.from_dict(payload)
    return {
        "frac_coords": np.asarray(structure.frac_coords, dtype=float),
        "atom_types": np.asarray([int(site.specie.Z) for site in structure], dtype=int),
        "lengths": np.asarray(structure.lattice.abc, dtype=float),
        "angles": np.asarray(structure.lattice.angles, dtype=float),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-jsonl", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.snapshot_root.resolve()))
    from compute_metrics import Crystal

    rows = read_rows(args.generation_jsonl)
    started = time.time()
    attempts = []
    for row in rows:
        comp_valid = struct_valid = valid = False
        reason = str(row.get("reason") or "")
        if row.get("status") == "succeeded" and isinstance(row.get("structure"), dict):
            try:
                crystal = Crystal(structure_array(row["structure"]))
                comp_valid = bool(crystal.comp_valid)
                struct_valid = bool(crystal.struct_valid)
                valid = bool(crystal.valid)
                reason = str(getattr(crystal, "invalid_reason", "") or "")
            except Exception as exc:  # preserve the requested denominator
                reason = f"{type(exc).__name__}:{exc}"
        attempts.append(
            {
                "schema": "crysllmgen_metric_attempt_v1",
                "attempt_id": str(row["attempt_id"]),
                "method": str(row.get("method") or ""),
                "comp_valid": comp_valid,
                "struct_valid": struct_valid,
                "valid": valid,
                "reason": reason,
            }
        )
    with (args.output_dir / "attempt_metrics.jsonl").open("x", encoding="utf-8") as handle:
        for row in attempts:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "schema": "crysllmgen_predicted_only_validity_v1",
        "attempts": len(attempts),
        "generation_succeeded": sum(row.get("status") == "succeeded" for row in rows),
        "comp_valid_count": sum(row["comp_valid"] for row in attempts),
        "struct_valid_count": sum(row["struct_valid"] for row in attempts),
        "valid_count": sum(row["valid"] for row in attempts),
        "unchanged_upstream_component": "compute_metrics.Crystal",
        "omitted": "GT fingerprints and distribution-level coverage/diversity metrics",
        "walltime_s": time.time() - started,
        "retry_or_replacement_used": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
