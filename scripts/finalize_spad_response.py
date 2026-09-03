#!/usr/bin/env python3
"""Finalize the fixed SPAD model494-response development result."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
EVAL_RUNTIME = ROOT / "eval_runtime"
if str(EVAL_RUNTIME) not in sys.path:
    sys.path.insert(0, str(EVAL_RUNTIME))
import protocol  # noqa: E402


def load_common() -> Any:
    path = EVAL_RUNTIME / "finalize_official.py"
    spec = importlib.util.spec_from_file_location("spad_response_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def distribution(values: list[float]) -> dict[str, Any]:
    values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not values:
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": values[0],
        "maximum": values[-1],
    }


def paired_energy(
    baseline_by_stream: Mapping[int, list[dict[str, Any]]],
    response_by_stream: Mapping[int, list[dict[str, Any]]],
    *,
    label: str,
) -> dict[str, Any]:
    stream_deltas: dict[int, dict[int, float]] = {}
    for stream in (17, 18):
        base = {int(row["ordinal"]): row for row in baseline_by_stream[stream]}
        response = {int(row["ordinal"]): row for row in response_by_stream[stream]}
        deltas = {}
        for index in range(256):
            left = base[index].get("chgnet_energy_per_atom")
            right = response[index].get("chgnet_energy_per_atom")
            if left is not None and right is not None:
                deltas[index] = float(right) - float(left)
        stream_deltas[stream] = deltas
    values = [
        statistics.fmean(
            stream_deltas[stream][index]
            for stream in (17, 18)
            if index in stream_deltas[stream]
        )
        for index in range(256)
        if any(index in stream_deltas[stream] for stream in (17, 18))
    ]
    rng = random.Random(
        int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    )
    means = sorted(
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(10_000)
    )
    return {
        "direction": "negative_is_favorable",
        "composition_count": len(values),
        "mean_delta_eV_per_atom": statistics.fmean(values),
        "median_delta_eV_per_atom": statistics.median(values),
        "fraction_negative": sum(value < 0 for value in values) / len(values),
        "bootstrap95": [means[249], means[9749]],
        "per_stream": {
            str(stream): distribution(list(stream_deltas[stream].values()))
            for stream in (17, 18)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-eval", type=Path, required=True)
    parser.add_argument("--baseline-eval-s17", type=Path, required=True)
    parser.add_argument("--baseline-eval-s18", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--baseline-final", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not (args.response_eval / "_SUCCESS").is_file():
        raise FileNotFoundError("response fast evaluation is incomplete")
    common = load_common()
    cache = args.official_run / "official_mp_cache"
    phase_diagrams = common._phase_diagrams(cache / "official_slim_cache.jsonl")
    unresolved = {
        str(row["chemsys"])
        for row in protocol.read_jsonl(cache / "unresolved_chemsys.jsonl")
    }
    rows_by_endpoint_stream: dict[tuple[str, int], list[dict[str, Any]]] = {}
    aggregates = []
    for endpoint in ("raw", "refined"):
        all_rows = []
        for stream in (17, 18):
            labels = protocol.read_jsonl(
                args.response_eval
                / f"stream{stream}/{endpoint}/full_reconstructed/attempt_labels_preofficial.jsonl"
            )
            if len(labels) != 256 or {int(row["ordinal"]) for row in labels} != set(
                range(256)
            ):
                raise ValueError("response labels changed fixed denominator")
            evaluated = []
            for row in labels:
                e_hull = None
                status = "not_reconstructed"
                if row.get("reconstructed") is True and row.get(
                    "chgnet_energy_per_atom"
                ) is None:
                    status = "chgnet_energy_unknown"
                elif row.get("reconstructed") is True and str(row.get("chemsys")) in unresolved:
                    status = "official_hull_unknown"
                elif row.get("reconstructed") is True:
                    pd = phase_diagrams[str(row["chemsys"])]
                    e_hull = common._e_above_hull(
                        pd,
                        row["chgnet_composition"],
                        float(row["chgnet_energy_per_atom"]),
                    )
                    status = "known"
                nu = row.get("novel") is True and row.get("unique_representative") is True
                evaluated.append(
                    {
                        **row,
                        "official_hull_status": status,
                        "official_e_above_hull": e_hull,
                        "strict_stable": e_hull is not None and e_hull <= 0.0,
                        "meta_stable": e_hull is not None and e_hull <= 0.1,
                        "strict_sun": bool(nu and e_hull is not None and e_hull <= 0.0),
                        "meta_sun": bool(nu and e_hull is not None and e_hull <= 0.1),
                    }
                )
            rows_by_endpoint_stream[(endpoint, stream)] = evaluated
            all_rows.extend(evaluated)
        aggregates.append(
            {
                "endpoint": endpoint,
                "requested": 512,
                "reconstructed": sum(row.get("reconstructed") is True for row in all_rows),
                "novel": sum(row.get("novel") is True for row in all_rows),
                "unique": sum(row.get("unique_representative") is True for row in all_rows),
                "novel_unique": sum(
                    row.get("novel") is True and row.get("unique_representative") is True
                    for row in all_rows
                ),
                "hull_known": sum(row["official_hull_status"] == "known" for row in all_rows),
                "strict_stable": sum(row["strict_stable"] for row in all_rows),
                "meta_stable": sum(row["meta_stable"] for row in all_rows),
                "strict_sun": sum(row["strict_sun"] for row in all_rows),
                "meta_sun": sum(row["meta_sun"] for row in all_rows),
                "strict_sun_rate": sum(row["strict_sun"] for row in all_rows) / 512,
                "meta_sun_rate": sum(row["meta_sun"] for row in all_rows) / 512,
                "direct": "DEFERRED_COST",
            }
        )
    baseline = json.loads(args.baseline_final.read_text(encoding="utf-8"))
    baseline_aggregates = {
        (row["arm"], row["endpoint"]): row for row in baseline["aggregates"]
    }
    comparisons = {}
    for endpoint in ("raw", "refined"):
        baseline_paths = {
            17: args.baseline_eval_s17,
            18: args.baseline_eval_s18,
        }
        base_rows = {
            stream: protocol.read_jsonl(
                path
                / f"BS/stream{stream}/{endpoint}/evaluation/full_reconstructed/attempt_labels_preofficial.jsonl"
            )
            for stream, path in baseline_paths.items()
        }
        response_rows = {
            stream: rows_by_endpoint_stream[(endpoint, stream)] for stream in (17, 18)
        }
        current = next(row for row in aggregates if row["endpoint"] == endpoint)
        old = baseline_aggregates[("BS", endpoint)]
        comparisons[endpoint] = {
            "energy": paired_energy(
                base_rows, response_rows, label=f"response-minus-BS:{endpoint}"
            ),
            "strict_sun_delta_count": int(current["strict_sun"]) - int(old["strict_sun"]),
            "meta_sun_delta_count": int(current["meta_sun"]) - int(old["meta_sun"]),
            "novel_unique_delta_count": int(current["novel_unique"])
            - int(old["novel_unique"]),
        }
    report = {
        "schema": "spad_model494_response_final_v1",
        "development_only": True,
        "same_frozen_plans_streams": True,
        "official_query_reused_not_repeated": True,
        "direct": "DEFERRED_COST",
        "aggregates": aggregates,
        "response_minus_BS": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "SPAD_RESPONSE_FINAL.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# SPAD model494-response development result",
        "",
        "The same frozen Plans and streams are retained. Full Direct is DEFERRED_COST.",
        "",
        "| Endpoint | Reconstructed | N/U/NU | Strict stable/S.U.N. | Meta stable/S.U.N. |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['endpoint']} | {row['reconstructed']}/512 | "
            f"{row['novel']}/{row['unique']}/{row['novel_unique']} | "
            f"{row['strict_stable']}/{row['strict_sun']} ({100*row['strict_sun_rate']:.2f}%) | "
            f"{row['meta_stable']}/{row['meta_sun']} ({100*row['meta_sun_rate']:.2f}%) |"
        )
    for endpoint, value in comparisons.items():
        energy = value["energy"]
        lines.extend(
            [
                "",
                f"- {endpoint} response-minus-BS CHGNet: {energy['mean_delta_eV_per_atom']:+.6f} "
                f"eV/atom, 95% bootstrap [{energy['bootstrap95'][0]:+.6f}, "
                f"{energy['bootstrap95'][1]:+.6f}]; Strict/Meta S.U.N. count deltas "
                f"{value['strict_sun_delta_count']:+d}/{value['meta_sun_delta_count']:+d}; "
                f"NU delta {value['novel_unique_delta_count']:+d}."
            ]
        )
    (args.output_dir / "SPAD_RESPONSE_FINAL.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
