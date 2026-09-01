#!/usr/bin/env python3
"""Finalize the registered G2 raw/refined Direct and CHGNet endpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Iterable


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def describe(values: Iterable[float]) -> dict:
    items = [float(value) for value in values]
    return {
        "known": len(items),
        "mean_eV_per_atom": None if not items else statistics.fmean(items),
        "median_eV_per_atom": None if not items else statistics.median(items),
        "q10_eV_per_atom": quantile(items, 0.10),
        "q90_eV_per_atom": quantile(items, 0.90),
    }


def bootstrap_mean_ci(values: list[float], *, seed: int, replicates: int = 10000) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    size = len(values)
    means = [statistics.fmean(values[rng.randrange(size)] for _ in range(size)) for _ in range(replicates)]
    return [quantile(means, 0.025), quantile(means, 0.975)]


def load_energy(path: Path) -> dict[int, float]:
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        energy = row.get("energy_per_atom")
        if energy is not None:
            result[int(row["local_index"])] = float(energy)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not (args.eval_run / "_OFFLINE_SUCCESS").is_file():
        raise ValueError("offline run is not terminal")

    raw_direct_path = args.eval_run / "raw/direct/report.json"
    refined_direct_path = args.eval_run / "refined/direct/report.json"
    raw_energy_path = args.eval_run / "raw/full_reconstructed/all_reconstructed_relax_results.jsonl"
    refined_energy_path = args.eval_run / "refined/full_reconstructed/all_reconstructed_relax_results.jsonl"
    raw_summary_path = args.eval_run / "raw/full_reconstructed/summary.json"
    refined_summary_path = args.eval_run / "refined/full_reconstructed/summary.json"
    raw_direct = json.loads(raw_direct_path.read_text())
    refined_direct = json.loads(refined_direct_path.read_text())
    raw_summary = json.loads(raw_summary_path.read_text())
    refined_summary = json.loads(refined_summary_path.read_text())
    raw_energy = load_energy(raw_energy_path)
    refined_energy = load_energy(refined_energy_path)
    paired_indices = sorted(set(raw_energy) & set(refined_energy))
    paired_delta = [refined_energy[index] - raw_energy[index] for index in paired_indices]

    for report in (raw_direct, refined_direct):
        assert report["attempts"] == 256
        assert report["denominator"] == "all_generation_attempts"
        assert report["retry_or_replacement_used"] is False
    assert raw_summary["chgnet_relaxation_known"] == len(raw_energy)
    assert refined_summary["chgnet_relaxation_known"] == len(refined_energy)

    result = {
        "schema": "compact_v2_periodic_relation_g2_full_offline_final_v1",
        "requested": 256,
        "official_query": False,
        "raw": {
            "body": int(raw_direct["generation_succeeded"]),
            "composition_valid": int(raw_direct["comp_valid_count"]),
            "structural_valid": int(raw_direct["struct_valid_count"]),
            "direct": int(raw_direct["valid_count"]),
            "chgnet": describe(raw_energy.values()),
            "chgnet_unknown": 248 - len(raw_energy),
        },
        "refined": {
            "body": int(refined_direct["generation_succeeded"]),
            "composition_valid": int(refined_direct["comp_valid_count"]),
            "structural_valid": int(refined_direct["struct_valid_count"]),
            "direct": int(refined_direct["valid_count"]),
            "chgnet": describe(refined_energy.values()),
            "chgnet_unknown": 248 - len(refined_energy),
        },
        "paired_refined_minus_raw": {
            **describe(paired_delta),
            "pairs": len(paired_indices),
            "mean_95pct_bootstrap_CI": bootstrap_mean_ci(paired_delta, seed=20260901),
            "bootstrap_seed": 20260901,
            "bootstrap_replicates": 10000,
        },
        "base_raw_direct": 106,
        "G2_raw_direct_delta_count": int(raw_direct["valid_count"]) - 106,
        "inputs": {
            str(path): sha256(path)
            for path in (raw_direct_path, refined_direct_path, raw_energy_path, refined_energy_path, raw_summary_path, refined_summary_path)
        },
    }
    ci = result["paired_refined_minus_raw"]["mean_95pct_bootstrap_CI"]
    result["energy_direction"] = (
        "refined_lower_supported" if ci[1] is not None and ci[1] < 0 else
        "refined_higher_supported" if ci[0] is not None and ci[0] > 0 else
        "uncertain"
    )

    args.output_dir.mkdir(parents=True)
    json_path = args.output_dir / "G2_FULL_OFFLINE_FINAL.json"
    csv_path = args.output_dir / "G2_FULL_OFFLINE_FINAL.csv"
    md_path = args.output_dir / "G2_FULL_OFFLINE_FINAL.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("endpoint", "body", "composition_valid", "structural_valid", "direct", "chgnet_known", "mean_eV_per_atom", "median_eV_per_atom"))
        writer.writeheader()
        for endpoint in ("raw", "refined"):
            row = result[endpoint]
            writer.writerow({"endpoint": endpoint, "body": row["body"], "composition_valid": row["composition_valid"], "structural_valid": row["structural_valid"], "direct": row["direct"], "chgnet_known": row["chgnet"]["known"], "mean_eV_per_atom": row["chgnet"]["mean_eV_per_atom"], "median_eV_per_atom": row["chgnet"]["median_eV_per_atom"]})
    md_path.write_text(
        "# G2 full offline final\n\n"
        f"- raw body/comp/Direct: `{result['raw']['body']}/{result['raw']['composition_valid']}/{result['raw']['direct']}` of 256\n"
        f"- refined body/comp/Direct: `{result['refined']['body']}/{result['refined']['composition_valid']}/{result['refined']['direct']}` of 256\n"
        f"- raw/refined CHGNet known: `{result['raw']['chgnet']['known']}/{result['refined']['chgnet']['known']}` of 248\n"
        f"- paired refined-minus-raw mean: `{result['paired_refined_minus_raw']['mean_eV_per_atom']:.6f}` eV/atom, 95% CI `{ci}`\n"
        f"- energy direction: `{result['energy_direction']}`\n"
        "- official query: `false`\n"
    )
    outputs = (json_path, csv_path, md_path)
    (args.output_dir / "OUTPUTS.sha256").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in outputs))
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
