#!/usr/bin/env python3
"""Audit whether CCFD can wrap the frozen Planner tokenizer without resizing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.ccfd_text import (  # noqa: E402
    analyze_formula_prefix,
    formula_term_boundaries,
    token_prefix_alignment,
)
from crystal_dlm.h1_llm_planner import (  # noqa: E402
    H1_PLANNER_PROMPT_STYLE_RICH_PLAN_PREFILL,
    format_planner_prompt,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"non-object row in {path}")
                yield value


def plan_from_row(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = row.get("plan_state") or row.get("r5_plan_state") or row.get("parsed_plan")
    return value if isinstance(value, Mapping) else None


def quantile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return float(ordered[index])


def audit_dataset(
    name: str,
    path: Path,
    tokenizer: object,
    *,
    max_atoms: int,
    prefill_prompt: str,
) -> dict[str, Any]:
    rows = formulas = syntax_terminal = roundtrip = incremental = prefix_safe = unk_free = 0
    prefill_boundary_exact = 0
    newline_crossing = 0
    token_counts: list[int] = []
    term_counts: list[int] = []
    failures: dict[str, int] = {}
    unsafe_prefix_examples: list[dict[str, Any]] = []
    unk_id = getattr(tokenizer, "unk_token_id", None)
    prompt_ids = tokenizer.encode(prefill_prompt, add_special_tokens=False)  # type: ignore[attr-defined]
    for row in iter_jsonl(path):
        rows += 1
        plan = plan_from_row(row)
        if plan is None or not plan.get("formula"):
            failures["missing_formula"] = failures.get("missing_formula", 0) + 1
            continue
        formula = str(plan["formula"])
        formulas += 1
        syntax = analyze_formula_prefix(formula, max_atoms=max_atoms)
        if not syntax.terminal:
            failures[f"syntax:{syntax.reason}"] = failures.get(f"syntax:{syntax.reason}", 0) + 1
            continue
        syntax_terminal += 1
        target = " " + formula + "\nanion:"
        combined_ids = tokenizer.encode(prefill_prompt + target, add_special_tokens=False)  # type: ignore[attr-defined]
        boundary_exact = list(combined_ids[: len(prompt_ids)]) == list(prompt_ids)
        prefill_boundary_exact += int(boundary_exact)
        token_ids = (
            list(combined_ids[len(prompt_ids) :])
            if boundary_exact
            else tokenizer.encode(target, add_special_tokens=False)  # type: ignore[attr-defined]
        )
        token_counts.append(len(token_ids))
        term_counts.append(len(formula_term_boundaries(formula)))
        result = token_prefix_alignment(tokenizer, token_ids, target, max_atoms=max_atoms)
        roundtrip += int(result["roundtrip_exact"] is True)
        incremental += int(result["incremental_prefix_exact"] is True)
        prefix_safe += int(result["formula_prefix_valid"] is True)
        if result["formula_prefix_valid"] is not True and len(unsafe_prefix_examples) < 20:
            unsafe_prefix_examples.append(
                {
                    "formula": formula,
                    "prefix": result.get("first_invalid_formula_prefix"),
                    "reason": result.get("first_invalid_reason"),
                    "token_ids": [int(value) for value in token_ids],
                }
            )
        unk_free += int(unk_id is None or int(unk_id) not in set(int(value) for value in token_ids))
        newline_step = result.get("newline_step")
        if isinstance(newline_step, int):
            decoded_at_newline = tokenizer.decode(  # type: ignore[attr-defined]
                token_ids[:newline_step],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            newline_crossing += int("\n" in decoded_at_newline and not decoded_at_newline.endswith("\n"))

    denominator = max(1, formulas)
    return {
        "name": name,
        "path": str(path.resolve()),
        "rows": rows,
        "formulas": formulas,
        "counts": {
            "syntax_terminal": syntax_terminal,
            "roundtrip_exact": roundtrip,
            "incremental_prefix_exact": incremental,
            "formula_prefix_safe": prefix_safe,
            "unk_free": unk_free,
            "newline_crossing_token": newline_crossing,
            "prefill_boundary_exact": prefill_boundary_exact,
        },
        "rates": {
            "syntax_terminal": syntax_terminal / denominator,
            "roundtrip_exact": roundtrip / denominator,
            "incremental_prefix_exact": incremental / denominator,
            "formula_prefix_safe": prefix_safe / denominator,
            "unk_free": unk_free / denominator,
            "newline_crossing_token": newline_crossing / denominator,
            "prefill_boundary_exact": prefill_boundary_exact / denominator,
        },
        "token_lengths": {
            "mean": statistics.fmean(token_counts) if token_counts else None,
            "q50": quantile(token_counts, 0.50),
            "q90": quantile(token_counts, 0.90),
            "max": max(token_counts) if token_counts else None,
        },
        "semantic_terms": {
            "mean": statistics.fmean(term_counts) if term_counts else None,
            "max": max(term_counts) if term_counts else None,
        },
        "failures": dict(sorted(failures.items())),
        "unsafe_prefix_examples": unsafe_prefix_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--dataset", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-atoms", type=int, default=20)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    checkpoint = Path(args.checkpoint_path) if args.checkpoint_path else None
    source = checkpoint if checkpoint and (checkpoint / "tokenizer_config.json").exists() else Path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(source), trust_remote_code=True)
    prefill_prompt = format_planner_prompt(
        tokenizer,
        sample_idx=0,
        prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_PLAN_PREFILL,
    )

    datasets: list[dict[str, Any]] = []
    for value in args.dataset:
        if "=" not in value:
            parser.error(f"invalid --dataset {value!r}; expected NAME=PATH")
        name, raw_path = value.split("=", 1)
        datasets.append(
            audit_dataset(
                name.strip(),
                Path(raw_path),
                tokenizer,
                max_atoms=int(args.max_atoms),
                prefill_prompt=prefill_prompt,
            )
        )
    gates = {
        "all_syntax_terminal": all(row["rates"]["syntax_terminal"] == 1.0 for row in datasets),
        "all_roundtrip_exact": all(row["rates"]["roundtrip_exact"] == 1.0 for row in datasets),
        "all_incremental_prefix_exact": all(
            row["rates"]["incremental_prefix_exact"] == 1.0 for row in datasets
        ),
        "all_formula_prefix_safe": all(row["rates"]["formula_prefix_safe"] == 1.0 for row in datasets),
        "all_unk_free": all(row["rates"]["unk_free"] == 1.0 for row in datasets),
        "all_prefill_boundaries_exact": all(
            row["rates"]["prefill_boundary_exact"] == 1.0 for row in datasets
        ),
    }
    gates["same_tokenizer_phase1_feasible"] = all(gates.values())
    report = {
        "schema": "h1a2_ccfd_tokenizer_interface_audit_v1",
        "model_path": str(Path(args.model_path).resolve()),
        "checkpoint_path": None if checkpoint is None else str(checkpoint.resolve()),
        "tokenizer_source": str(source.resolve()),
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "unk_token_id": tokenizer.unk_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "added_tokens": 0,
        },
        "interface": {
            "weights_changed": False,
            "tokenizer_resized": False,
            "matched_arm_prompt": "rich Planner prompt with formula-value prefill in both F0 and F1",
            "constraint_scope": "formula value only; rich fields remain unconstrained",
            "dead_end_policy": "one trajectory; fail without repair or replacement",
        },
        "datasets": datasets,
        "gates": gates,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "CCFD_TOKENIZER_INTERFACE_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = (
        "name",
        "formulas",
        "syntax_terminal",
        "roundtrip_exact",
        "incremental_prefix_exact",
        "formula_prefix_safe",
        "unk_free",
        "newline_crossing_token",
        "prefill_boundary_exact",
        "token_mean",
        "token_q90",
        "token_max",
    )
    with (output / "CCFD_TOKENIZER_INTERFACE_AUDIT.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in datasets:
            writer.writerow(
                {
                    "name": row["name"],
                    "formulas": row["formulas"],
                    **row["rates"],
                    "token_mean": row["token_lengths"]["mean"],
                    "token_q90": row["token_lengths"]["q90"],
                    "token_max": row["token_lengths"]["max"],
                }
            )
    lines = [
        "# CCFD frozen-tokenizer interface audit",
        "",
        f"Same-tokenizer Phase 1 feasible: **{gates['same_tokenizer_phase1_feasible']}**",
        "",
        "| Dataset | Formulas | Syntax | Round-trip | Incremental | Prefix-safe | UNK-free | Prefill boundary | Newline-crossing | Tokens mean/q90/max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in datasets:
        rates = row["rates"]
        lengths = row["token_lengths"]
        lines.append(
            f"| {row['name']} | {row['formulas']} | {rates['syntax_terminal']:.2%} | "
            f"{rates['roundtrip_exact']:.2%} | {rates['incremental_prefix_exact']:.2%} | "
            f"{rates['formula_prefix_safe']:.2%} | {rates['unk_free']:.2%} | "
            f"{rates['prefill_boundary_exact']:.2%} | "
            f"{rates['newline_crossing_token']:.2%} | "
            f"{lengths['mean']:.2f}/{lengths['q90']:.0f}/{lengths['max']} |"
        )
    lines.extend(["", "## Gates", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in gates.items())
    (output / "CCFD_TOKENIZER_INTERFACE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
