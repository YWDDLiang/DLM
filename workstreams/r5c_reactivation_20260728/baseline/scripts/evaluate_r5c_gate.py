#!/usr/bin/env python3
"""Evaluate R5-C exact-length dynamic DLM smoke gates."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


N_TOKEN_RE = re.compile(r"<N_(\d{3})>")


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def rate(numerator: Any, denominator: Any) -> float:
    try:
        return float(numerator) / max(1.0, float(denominator))
    except Exception:
        return 0.0


def nested_summary(payload: Mapping[str, Any], preferred: str | None) -> Mapping[str, Any]:
    if preferred and isinstance(payload.get(preferred), Mapping):
        return payload[preferred]  # type: ignore[index]
    for key in ("raw_jsonl", "raw_pt", "refined_pt"):
        item = payload.get(key)
        if isinstance(item, Mapping):
            return item
    return payload


def load_raw_generation_stats(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "decoded_rows": 0,
            "n_token_parse_success": 0,
            "n_token_match_plan": 0,
            "plan_n_histogram": {},
            "token_n_histogram": {},
            "n_token_max_fraction": 0.0,
            "n_hist_l1_to_plan": 1.0,
        }

    decoded_rows = 0
    n_token_parse_success = 0
    n_token_match_plan = 0
    plan_n_hist: Counter[str] = Counter()
    token_n_hist: Counter[str] = Counter()
    mismatch_examples: list[dict[str, Any]] = []
    missing_examples: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            decoded_rows += 1
            plan = record.get("plan_state") or {}
            try:
                plan_n = int(plan.get("N"))
                plan_n_hist[str(plan_n)] += 1
            except Exception:
                plan_n = None

            text = str(record.get("text") or "")
            match = N_TOKEN_RE.search(text)
            if match is None:
                if len(missing_examples) < 10:
                    missing_examples.append(
                        {
                            "row_idx": row_idx,
                            "sample_idx": record.get("sample_idx", row_idx),
                            "plan_N": plan_n,
                            "text_prefix": text[:160],
                        }
                    )
                continue
            token_n = int(match.group(1))
            n_token_parse_success += 1
            token_n_hist[str(token_n)] += 1
            if plan_n is not None and token_n == plan_n:
                n_token_match_plan += 1
            elif len(mismatch_examples) < 10:
                mismatch_examples.append(
                    {
                        "row_idx": row_idx,
                        "sample_idx": record.get("sample_idx", row_idx),
                        "plan_N": plan_n,
                        "token_N": token_n,
                        "text_prefix": text[:160],
                    }
                )

    token_max_fraction = 0.0
    if decoded_rows:
        token_max_fraction = max(token_n_hist.values() or [0]) / decoded_rows

    keys = set(plan_n_hist) | set(token_n_hist)
    l1 = 0.0
    for key in keys:
        l1 += abs(plan_n_hist.get(key, 0) / max(1, decoded_rows) - token_n_hist.get(key, 0) / max(1, decoded_rows))

    return {
        "decoded_rows": decoded_rows,
        "n_token_parse_success": n_token_parse_success,
        "n_token_parse_rate": n_token_parse_success / max(1, decoded_rows),
        "n_token_match_plan": n_token_match_plan,
        "n_token_match_plan_rate": n_token_match_plan / max(1, decoded_rows),
        "plan_n_histogram": dict(plan_n_hist.most_common()),
        "token_n_histogram": dict(token_n_hist.most_common()),
        "n_token_max_fraction": token_max_fraction,
        "n_hist_l1_to_plan": l1,
        "n_missing_examples": missing_examples,
        "n_mismatch_examples": mismatch_examples,
    }


def write_markdown(result: Mapping[str, Any], path: Path) -> None:
    metrics = result["metrics"]
    lines = [
        "# R5-C Exact-Length 256 Gate",
        "",
        f"- passed: `{result['passed']}`",
        f"- parse_rate: `{metrics['parse_rate']:.6f}`",
        f"- plan_match_rate: `{metrics['plan_match_rate']:.6f}`",
        f"- n_token_parse_rate: `{metrics['n_token_parse_rate']:.6f}`",
        f"- n_token_match_plan_rate: `{metrics['n_token_match_plan_rate']:.6f}`",
        f"- graph_acceptance: `{metrics['graph_acceptance']:.6f}`",
        f"- comp_valid: `{metrics['comp_valid']:.6f}`",
        f"- single_element: `{metrics['single_element']:.6f}`",
        f"- all_metal: `{metrics['all_metal']:.6f}`",
        f"- pbc_duplicate: `{metrics['pbc_duplicate']:.6f}`",
        f"- n_token_max_fraction: `{metrics['n_token_max_fraction']:.6f}`",
        f"- n_hist_l1_to_plan: `{metrics['n_hist_l1_to_plan']:.6f}`",
        "",
    ]
    if result["failures"]:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {item}" for item in result["failures"])
        lines.append("")
    lines.extend(
        [
            "## N Distribution",
            "",
            "```json",
            json.dumps(result["n_distribution"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Composition Reasons",
            "",
            "```json",
            json.dumps(result["composition_reason_counts"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-metrics", type=Path, required=True)
    parser.add_argument("--raw-generations-jsonl", type=Path, required=True)
    parser.add_argument("--composition-summary", type=Path, required=True)
    parser.add_argument("--composition-key", default="raw_jsonl")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--min-n-parse", type=float, default=0.99)
    parser.add_argument("--min-body-parse", type=float, default=0.95)
    parser.add_argument("--min-plan-match", type=float, default=0.95)
    parser.add_argument("--min-graph-acceptance", type=float, default=0.85)
    parser.add_argument("--min-comp-valid", type=float, default=0.918)
    parser.add_argument("--max-single-element", type=float, default=0.01)
    parser.add_argument("--max-all-metal", type=float, default=1.0)
    parser.add_argument("--max-pbc-duplicate", type=float, default=0.0)
    parser.add_argument("--max-n-fraction", type=float, default=0.35)
    parser.add_argument("--max-n-l1-to-plan", type=float, default=0.02)
    args = parser.parse_args()

    sample = load_json(args.sample_metrics)
    composition_payload = load_json(args.composition_summary)
    composition = nested_summary(composition_payload, args.composition_key)
    raw_stats = load_raw_generation_stats(args.raw_generations_jsonl)

    decoded = sample.get("decoded_samples") or raw_stats.get("decoded_rows") or sample.get("requested_samples") or 0
    parse_rate = float(sample.get("parse_rate") or rate(sample.get("parse_success", 0), decoded))
    plan_match_rate = float(sample.get("plan_match_rate") or rate(sample.get("plan_match_success", 0), decoded))
    graph_acceptance = float(
        sample.get("graph_acceptance_rate")
        or sample.get("graph_rate")
        or rate(sample.get("graph_success", 0), decoded)
    )
    comp_valid = float(composition.get("comp_valid_rate") or 0.0)
    reason_counts = composition.get("reason_counts") or {}
    count = float(composition.get("count") or sum(reason_counts.values()) or 1.0)
    single_element = float(reason_counts.get("single_element_shortcut", 0)) / max(1.0, count)
    all_metal = float(reason_counts.get("all_metal_shortcut", 0)) / max(1.0, count)
    pbc_duplicate = float(composition.get("pbc_equivalent_duplicate_fraction") or 0.0)

    failures: list[str] = []
    if raw_stats["n_token_parse_rate"] < args.min_n_parse:
        failures.append(f"n_token_parse_rate {raw_stats['n_token_parse_rate']:.4f} < {args.min_n_parse:.4f}")
    if parse_rate < args.min_body_parse:
        failures.append(f"body_parse_rate {parse_rate:.4f} < {args.min_body_parse:.4f}")
    if plan_match_rate < args.min_plan_match:
        failures.append(f"plan_match_rate {plan_match_rate:.4f} < {args.min_plan_match:.4f}")
    if raw_stats["n_token_match_plan_rate"] < args.min_n_parse:
        failures.append(
            f"n_token_match_plan_rate {raw_stats['n_token_match_plan_rate']:.4f} < {args.min_n_parse:.4f}"
        )
    if graph_acceptance < args.min_graph_acceptance:
        failures.append(f"graph_acceptance {graph_acceptance:.4f} < {args.min_graph_acceptance:.4f}")
    if comp_valid < args.min_comp_valid:
        failures.append(f"comp_valid {comp_valid:.4f} < {args.min_comp_valid:.4f}")
    if single_element > args.max_single_element:
        failures.append(f"single_element {single_element:.4f} > {args.max_single_element:.4f}")
    if all_metal > args.max_all_metal:
        failures.append(f"all_metal {all_metal:.4f} > {args.max_all_metal:.4f}")
    if pbc_duplicate > args.max_pbc_duplicate:
        failures.append(f"pbc_duplicate {pbc_duplicate:.4f} > {args.max_pbc_duplicate:.4f}")
    if raw_stats["n_token_max_fraction"] > args.max_n_fraction:
        failures.append(f"n_token_max_fraction {raw_stats['n_token_max_fraction']:.4f} > {args.max_n_fraction:.4f}")
    if raw_stats["n_hist_l1_to_plan"] > args.max_n_l1_to_plan:
        failures.append(f"n_hist_l1_to_plan {raw_stats['n_hist_l1_to_plan']:.4f} > {args.max_n_l1_to_plan:.4f}")

    result = {
        "passed": not failures,
        "failures": failures,
        "metrics": {
            "decoded_samples": int(decoded),
            "parse_rate": parse_rate,
            "plan_match_rate": plan_match_rate,
            "n_token_parse_rate": raw_stats["n_token_parse_rate"],
            "n_token_match_plan_rate": raw_stats["n_token_match_plan_rate"],
            "graph_acceptance": graph_acceptance,
            "comp_valid": comp_valid,
            "single_element": single_element,
            "all_metal": all_metal,
            "pbc_duplicate": pbc_duplicate,
            "n_token_max_fraction": raw_stats["n_token_max_fraction"],
            "n_hist_l1_to_plan": raw_stats["n_hist_l1_to_plan"],
        },
        "thresholds": {
            "min_n_parse": args.min_n_parse,
            "min_body_parse": args.min_body_parse,
            "min_plan_match": args.min_plan_match,
            "min_graph_acceptance": args.min_graph_acceptance,
            "min_comp_valid": args.min_comp_valid,
            "max_single_element": args.max_single_element,
            "max_all_metal": args.max_all_metal,
            "max_pbc_duplicate": args.max_pbc_duplicate,
            "max_n_fraction": args.max_n_fraction,
            "max_n_l1_to_plan": args.max_n_l1_to_plan,
        },
        "sample_metrics": sample,
        "composition_reason_counts": reason_counts,
        "n_distribution": {
            "plan_n_histogram": raw_stats["plan_n_histogram"],
            "token_n_histogram": raw_stats["token_n_histogram"],
            "n_missing_examples": raw_stats["n_missing_examples"],
            "n_mismatch_examples": raw_stats["n_mismatch_examples"],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, args.output_md)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
