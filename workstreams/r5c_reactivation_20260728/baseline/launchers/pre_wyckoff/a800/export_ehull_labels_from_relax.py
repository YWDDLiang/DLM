#!/usr/bin/env python3
"""Export per-sample e_hull labels from existing A100 relax_results.jsonl.

This intentionally does not re-run CHGNet relaxation and does not recompute
novelty/uniqueness from a generated .pt.  It mirrors the hull call used by the
A100 eval_sun_resumable.py script, then writes per-record strict/meta labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--mp-cache-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=path/to/relax_results.jsonl",
    )
    return parser.parse_args()


def load_relax_records(path: Path, composition_cls):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            comp = composition_cls.from_dict(item["composition"])
            records.append(
                {
                    "local_index": int(item["local_index"]),
                    "energy_per_atom": item.get("energy_per_atom"),
                    "composition": comp,
                    "composition_dict": item["composition"],
                }
            )
    return records


def composition_family(comp) -> dict:
    symbols = sorted(el.symbol for el in comp.elements)
    counts = [float(comp[el]) for el in comp.elements]
    n_atoms = int(round(sum(counts)))
    arity = len(symbols)
    anions = {"O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P", "As"}
    chalcogens = {"O", "S", "Se", "Te"}
    halogens = {"F", "Cl", "Br", "I"}
    return {
        "formula": comp.reduced_formula,
        "chemsys": "-".join(symbols),
        "elements": symbols,
        "N": n_atoms,
        "arity": arity,
        "contains_oxygen": "O" in symbols,
        "contains_chalcogen": bool(chalcogens.intersection(symbols)),
        "contains_halogen": bool(halogens.intersection(symbols)),
        "contains_common_anion": bool(anions.intersection(symbols)),
        "all_metal_proxy": not bool(anions.intersection(symbols)),
    }


def load_training_formula_set(path: Path) -> set[str]:
    formulas = set()
    if not path.exists():
        return formulas
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key in ("pretty_formula", "formula", "reduced_formula"):
                value = row.get(key)
                if value:
                    formulas.add(value)
    return formulas


def summarize(records: list[dict], strict_threshold: float = 0.0, meta_threshold: float = 0.1):
    total = len(records)
    evaluated = [r for r in records if r["e_hull"] is not None]
    unknown = total - len(evaluated)
    strict = [r for r in evaluated if r["strict_stable"]]
    meta = [r for r in evaluated if r["meta_stable"]]
    counters = {
        "N": Counter(str(r["N"]) for r in records),
        "arity": Counter(str(r["arity"]) for r in records),
        "chemsys": Counter(r["chemsys"] for r in records),
        "elements": Counter(el for r in records for el in r["elements"]),
        "flags": Counter(
            key
            for r in records
            for key in [
                "contains_oxygen" if r["contains_oxygen"] else None,
                "contains_chalcogen" if r["contains_chalcogen"] else None,
                "contains_halogen" if r["contains_halogen"] else None,
                "contains_common_anion" if r["contains_common_anion"] else None,
                "all_metal_proxy" if r["all_metal_proxy"] else None,
                "strict_stable" if r["strict_stable"] else None,
                "meta_stable" if r["meta_stable"] else None,
                "unknown_hull" if r["unknown_hull"] else None,
            ]
            if key
        ),
    }

    def pct(n, d=total):
        return 100.0 * n / d if d else 0.0

    def mean(values):
        return sum(values) / len(values) if values else None

    e_values = [r["e_hull"] for r in evaluated]
    energy_values = [r["energy_per_atom"] for r in records if r["energy_per_atom"] is not None]
    return {
        "total_relax_records": total,
        "evaluated": len(evaluated),
        "unknown": unknown,
        "strict_threshold": strict_threshold,
        "meta_threshold": meta_threshold,
        "strict_stable": len(strict),
        "meta_stable": len(meta),
        "strict_stable_per_evaluated_pct": pct(len(strict), len(evaluated)),
        "meta_stable_per_evaluated_pct": pct(len(meta), len(evaluated)),
        "strict_lower_bound_over_records_pct": pct(len(strict)),
        "meta_lower_bound_over_records_pct": pct(len(meta)),
        "unknown_pct": pct(unknown),
        "mean_N": mean([r["N"] for r in records]),
        "mean_arity": mean([r["arity"] for r in records]),
        "mean_e_hull": mean(e_values),
        "median_e_hull": sorted(e_values)[len(e_values) // 2] if e_values else None,
        "mean_energy_per_atom": mean(energy_values),
        "N_hist": dict(sorted(counters["N"].items(), key=lambda kv: int(kv[0]))),
        "arity_hist": dict(sorted(counters["arity"].items(), key=lambda kv: int(kv[0]))),
        "flag_counts": dict(counters["flags"]),
        "top_chemsys": counters["chemsys"].most_common(30),
        "top_elements": counters["elements"].most_common(30),
    }


def grouped_enrichment(records: list[dict], key: str, label: str) -> list[dict]:
    total_counter = Counter(str(r[key]) for r in records)
    label_counter = Counter(str(r[key]) for r in records if r[label])
    out = []
    label_total = sum(label_counter.values())
    total = len(records)
    for value, count in total_counter.items():
        labelled = label_counter[value]
        base_rate = count / total if total else 0.0
        label_rate = labelled / label_total if label_total else 0.0
        out.append(
            {
                "value": value,
                "total": count,
                "label_count": labelled,
                "total_pct": 100.0 * base_rate,
                "label_pct": 100.0 * label_rate,
                "enrichment": (label_rate / base_rate) if base_rate else None,
            }
        )
    return sorted(out, key=lambda row: (-(row["enrichment"] or 0), -row["label_count"], row["value"]))[:30]


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.eval_dir).resolve()))

    from eval_sun import compute_e_hull_batch
    from pymatgen.core import Composition

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_formula_set = load_training_formula_set(Path(args.train_csv))

    all_summary = {
        "eval_dir": args.eval_dir,
        "train_csv": args.train_csv,
        "mp_cache_path": args.mp_cache_path,
        "runs": {},
    }

    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"--run must be LABEL=path, got {spec!r}")
        label, relax_path = spec.split("=", 1)
        relax_path = Path(relax_path)
        raw_records = load_relax_records(relax_path, Composition)
        energies_comps = [
            (r["energy_per_atom"], r["composition"])
            for r in raw_records
            if r["energy_per_atom"] is not None
        ]
        e_hull = compute_e_hull_batch(
            energies_comps,
            args.train_csv,
            mp_api_key=None,
            mp_cache_path=args.mp_cache_path,
        )

        records = []
        for r in raw_records:
            fam = composition_family(r["composition"])
            energy = r["energy_per_atom"]
            hull = None
            if energy is not None:
                hull = e_hull.get((energy, r["composition"].reduced_formula))
            record = {
                "local_index": r["local_index"],
                "energy_per_atom": energy,
                "composition": r["composition_dict"],
                **fam,
                "e_hull": hull,
                "unknown_hull": hull is None,
                "strict_stable": hull is not None and hull <= 0.0,
                "meta_stable": hull is not None and hull <= 0.1,
                "formula_in_train_csv_proxy": fam["formula"] in train_formula_set,
            }
            records.append(record)

        run_dir = out_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        labels_path = run_dir / "ehull_labels.jsonl"
        labels_path.write_text(
            "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records),
            encoding="utf-8",
        )
        summary = summarize(records)
        summary["relax_results_path"] = str(relax_path)
        summary["labels_path"] = str(labels_path)
        summary["enrichment"] = {
            "strict_by_N": grouped_enrichment(records, "N", "strict_stable"),
            "meta_by_N": grouped_enrichment(records, "N", "meta_stable"),
            "strict_by_arity": grouped_enrichment(records, "arity", "strict_stable"),
            "meta_by_arity": grouped_enrichment(records, "arity", "meta_stable"),
            "strict_by_chemsys": grouped_enrichment(records, "chemsys", "strict_stable"),
            "meta_by_chemsys": grouped_enrichment(records, "chemsys", "meta_stable"),
        }
        (run_dir / "ehull_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        all_summary["runs"][label] = summary

    (out_dir / "all_ehull_summary.json").write_text(
        json.dumps(all_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(all_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
