#!/usr/bin/env python3
"""Run unchanged CrysLLMGen generation metrics on the full attempt denominator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


def _require_runtime() -> int:
    values = []
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        value = int(os.environ.get(name, "0"))
        if value not in (4, 8, 16):
            raise RuntimeError(f"{name} must be one of 4, 8, or 16")
        values.append(value)
    if len(set(values)) != 1:
        raise RuntimeError("metric numerical thread settings must agree")
    if values[0] > int(os.environ.get("SLURM_CPUS_PER_TASK", "0")):
        raise RuntimeError("metric threads exceed allocated Slurm CPUs")
    if os.environ.get("CRYSLLMGEN_METRICS_NUM_CPUS", "1") != "1":
        raise RuntimeError("the frozen CrysLLMGen metric worker count is one")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("registered metrics must run through Slurm CPU")
    return values[0]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    result = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "wqcodiff_generation_attempt_v1":
                raise ValueError(f"line {line_number}: invalid generation schema")
            attempt_id = str(row.get("attempt_id", ""))
            if not attempt_id or attempt_id in seen:
                raise ValueError(f"line {line_number}: missing/duplicate attempt ID")
            seen.add(attempt_id)
            result.append(row)
    if not result or len({str(row["method"]) for row in result}) != 1:
        raise ValueError("metric input must contain exactly one nonempty method")
    return result


def _invalid_crystal(reason: str) -> Any:
    return SimpleNamespace(
        constructed=False,
        invalid_reason=reason,
        comp_valid=False,
        struct_valid=False,
        valid=False,
        comp_fp=None,
        struct_fp=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-jsonl", type=Path, required=True)
    parser.add_argument("--gt-csv", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-model-name", choices=("mp20",), default="mp20")
    args = parser.parse_args()
    runtime_threads = _require_runtime()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = _rows(args.generation_jsonl.resolve())
    snapshot = args.snapshot_root.resolve()
    os.environ["CRYSLLMGEN_METRICS_NUM_CPUS"] = "1"
    sys.path.insert(0, str(snapshot))
    try:
        from compute_metrics import Crystal, GenEval, get_gt_crys_ori
    finally:
        sys.path.pop(0)
    from pymatgen.core import Structure
    import pandas as pd

    started = time.monotonic()
    crystals = []
    attempt_records = []
    for row in rows:
        reason = ""
        if row.get("status") != "succeeded" or not isinstance(
            row.get("structure"), Mapping
        ):
            reason = "upstream_generation:" + str(row.get("reason", row.get("status")))
            crystal = _invalid_crystal(reason)
        else:
            try:
                structure = Structure.from_dict(dict(row["structure"]))
                crystal = Crystal(
                    {
                        "frac_coords": structure.frac_coords,
                        "atom_types": __import__("numpy").array(
                            structure.atomic_numbers
                        ),
                        "lengths": __import__("numpy").array(structure.lattice.abc),
                        "angles": __import__("numpy").array(
                            structure.lattice.angles
                        ),
                    }
                )
                if not crystal.valid:
                    reason = str(getattr(crystal, "invalid_reason", "crysllmgen_invalid"))
            except Exception as exc:
                reason = f"{type(exc).__name__}:{exc}"
                crystal = _invalid_crystal(reason)
        crystals.append(crystal)
        attempt_records.append(
            {
                "schema": "crysllmgen_metric_attempt_v1",
                "attempt_id": row["attempt_id"],
                "method": row["method"],
                "comp_valid": bool(crystal.comp_valid),
                "struct_valid": bool(crystal.struct_valid),
                "valid": bool(crystal.valid),
                "reason": reason,
            }
        )
    dataframe = pd.read_csv(args.gt_csv.resolve())
    if "cif" not in dataframe:
        raise ValueError("CrysLLMGen ground-truth CSV has no cif column")
    gt_crystals = [get_gt_crys_ori(cif) for cif in dataframe["cif"]]
    evaluator = GenEval(
        crystals,
        gt_crystals,
        eval_model_name=args.eval_model_name,
    )
    metrics = evaluator.get_metrics()
    attempts_path = output / "attempt_metrics.jsonl"
    with attempts_path.open("x", encoding="utf-8") as handle:
        for record in attempt_records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    denominator = len(rows)
    report = {
        "schema": "crysllmgen_generation_metrics_report_v1",
        "ok": True,
        "method": rows[0]["method"],
        "denominator": "all_generation_attempts",
        "attempts": denominator,
        "generation_succeeded": sum(row.get("status") == "succeeded" for row in rows),
        "comp_valid_count": sum(record["comp_valid"] for record in attempt_records),
        "struct_valid_count": sum(record["struct_valid"] for record in attempt_records),
        "valid_count": sum(record["valid"] for record in attempt_records),
        "metrics_unchanged_upstream": metrics,
        "metric_worker_count": 1,
        "numerical_threads": runtime_threads,
        "gt_csv": str(args.gt_csv.resolve()),
        "gt_csv_sha256": _sha(args.gt_csv.resolve()),
        "gt_structures": len(gt_crystals),
        "generation_jsonl": str(args.generation_jsonl.resolve()),
        "generation_jsonl_sha256": _sha(args.generation_jsonl.resolve()),
        "attempt_metrics": str(attempts_path),
        "attempt_metrics_sha256": _sha(attempts_path),
        "upstream_compute_metrics_sha256": _sha(snapshot / "compute_metrics.py"),
        "upstream_eval_utils_sha256": _sha(snapshot / "eval_utils.py"),
        "walltime_s": time.monotonic() - started,
        "retry_or_replacement_used": False,
    }
    report_path = output / "report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
