#!/usr/bin/env python3
"""Report missing MP hull-cache chemsys for A100 eval_sun generated files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--run", action="append", required=True, help="LABEL=generated.pt")
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args()


def load_cache_chemsys(path: Path) -> set[str]:
    chemsys = set()
    if not path.exists():
        return chemsys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                chemsys.add(str(json.loads(line)["chemsys"]))
            except Exception:
                continue
    return chemsys


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.eval_dir).resolve()))
    from eval_sun import compute_novelty, compute_uniqueness, load_generated_structures, load_training_index

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    train_structures, train_formula_idx = load_training_index(args.train_csv)
    existing = load_cache_chemsys(Path(args.cache_path))
    summary = {
        "cache_path": args.cache_path,
        "cache_chemsys": len(existing),
        "runs": {},
    }
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"--run must be LABEL=path, got {spec!r}")
        label, gen_file = spec.split("=", 1)
        gen_structures, n_generated_total = load_generated_structures(gen_file)
        novel_mask = compute_novelty(gen_structures, train_structures, train_formula_idx, matcher)
        eq_class, _ = compute_uniqueness(gen_structures, matcher)
        seen_classes = set()
        novel_unique_mask = np.zeros(len(gen_structures), dtype=bool)
        for i in range(len(gen_structures)):
            cls = eq_class[i]
            if cls in seen_classes:
                continue
            seen_classes.add(cls)
            if novel_mask[i]:
                novel_unique_mask[i] = True
        grouped = defaultdict(int)
        for structure, keep in zip(gen_structures, novel_unique_mask):
            if not keep:
                continue
            chemsys = "-".join(sorted(el.symbol for el in structure.composition.elements))
            grouped[chemsys] += 1
        missing = {chemsys: count for chemsys, count in grouped.items() if chemsys not in existing}
        summary["runs"][label] = {
            "generated": gen_file,
            "n_generated_total": n_generated_total,
            "n_reconstructed": len(gen_structures),
            "n_novel": int(novel_mask.sum()),
            "n_novel_unique": int(novel_unique_mask.sum()),
            "n_grouped_chemsys": len(grouped),
            "missing_chemsys": len(missing),
            "missing_structures": int(sum(missing.values())),
            "top_missing": sorted(missing.items(), key=lambda item: (-item[1], item[0]))[:50],
        }
    out = Path(args.summary_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
