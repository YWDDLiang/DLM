#!/usr/bin/env python3
"""Enrich an A100 eval_sun MP hull cache for a generated .pt file.

This is a network/cache preparation helper. It computes the same novel+unique
set as eval_sun_resumable.py, identifies chemsys not already present in the
cache, and appends MP entries for those chemsys using a key read from a local
secret file. The key value is never printed.
"""

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
from pymatgen.ext.matproj import MPRester
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--gen-file", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep-s", type=int, default=2)
    return parser.parse_args()


def load_cache_chemsys(path: Path) -> set[str]:
    chemsys = set()
    if not path.exists():
        return chemsys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                chemsys.add(json.loads(line)["chemsys"])
            except Exception:
                continue
    return chemsys


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    sys.path.insert(0, str(eval_dir.resolve()))

    from eval_sun import (  # pylint: disable=import-error,import-outside-toplevel
        _get_mp_entries_for_chemsys,
        compute_novelty,
        compute_uniqueness,
        load_generated_structures,
        load_training_index,
    )

    key = Path(args.key_file).read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError("MP API key file is empty")

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    gen_structures, n_generated_total = load_generated_structures(args.gen_file)
    train_structures, train_formula_idx = load_training_index(args.train_csv)
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
        elements = tuple(sorted(el.symbol for el in structure.composition.elements))
        grouped[elements] += 1

    cache_path = Path(args.cache_path)
    existing = load_cache_chemsys(cache_path)
    missing = [
        elements
        for elements in sorted(grouped)
        if "-".join(sorted(elements)) not in existing
    ]
    if args.limit is not None:
        missing = missing[: args.limit]

    queried = 0
    successful = 0
    empty = 0
    failed = 0

    with MPRester(key) as mpr:
        cache = {chemsys: True for chemsys in existing}
        for elements in tqdm(missing, desc="MP cache enrichment"):
            queried += 1
            entries = _get_mp_entries_for_chemsys(
                mpr,
                list(elements),
                cache,
                cache_path=cache_path,
                retries=args.retries,
                sleep_s=args.sleep_s,
            )
            if entries is None:
                failed += 1
            elif len(entries) == 0:
                empty += 1
            else:
                successful += 1

    final_existing = load_cache_chemsys(cache_path)
    summary = {
        "generated": args.gen_file,
        "n_generated_total": n_generated_total,
        "n_reconstructed": len(gen_structures),
        "n_novel": int(novel_mask.sum()),
        "n_novel_unique": int(novel_unique_mask.sum()),
        "n_grouped_chemsys": len(grouped),
        "cache_path": str(cache_path),
        "cache_chemsys_before": len(existing),
        "cache_chemsys_after": len(final_existing),
        "missing_before": len([e for e in grouped if "-".join(sorted(e)) not in existing]),
        "queried": queried,
        "successful": successful,
        "empty": empty,
        "failed": failed,
        "key_file_present": bool(args.key_file),
    }
    out = Path(args.summary_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
