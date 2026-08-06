#!/usr/bin/env python3
"""Build a legacy SEAL-style success buffer from raw crystal rollouts.

This script is intentionally guarded because raw DLM generations should not
enter the default SFT buffer. Use ``build_refined_seal_success_buffer.py`` for
the production MP-20 loop: raw generations are diagnostics/RL evidence, while
positive SFT buffer rows must come from diffusion-refined structures.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.composition_validity import composition_record
from crystal_dlm.fixed_slot import CANONICAL_PROMPT, parse_fixed_slot_answer, tokenize_answer_text, write_json
from scripts.sample_llada_crystals import graph_from_arrays, import_process_one


FAILURE_REASONS_FOR_EDIT = {
    "single_element_shortcut",
    "charge_neutrality_fail",
    "pauling_fail_or_ratio_rejected",
    "oxidation_state_missing",
    "all_metal_shortcut",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            count += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return count


def pbc_duplicate_record(frac_coords: Iterable[Iterable[float]]) -> dict[str, Any]:
    exact: Counter[tuple[int, int, int]] = Counter()
    pbc: Counter[tuple[int, int, int]] = Counter()
    for coord in frac_coords:
        bins = tuple(int(round(float(value) * 100.0)) for value in coord)
        exact[bins] += 1
        pbc[tuple(value % 100 for value in bins)] += 1
    exact_dup = sum(count - 1 for count in exact.values() if count > 1)
    pbc_dup = sum(count - 1 for count in pbc.values() if count > 1)
    return {
        "exact_duplicate_site_count": exact_dup,
        "pbc_equivalent_duplicate_site_count": pbc_dup,
        "has_pbc_equivalent_duplicate": pbc_dup > 0,
    }


def classify_answer(answer: str) -> dict[str, Any]:
    arrays = parse_fixed_slot_answer(answer)
    comp = composition_record(arrays["atom_types"])
    pbc = pbc_duplicate_record(arrays["frac_coords"])
    reason = str(comp["reason"])
    return {
        "arrays": arrays,
        "reason": reason,
        "strict_valid": reason == "charge_neutral_pauling_valid",
        "comp": comp,
        **pbc,
    }


def expand_atom_symbols(row: dict[str, Any]) -> list[str]:
    atom_types = row.get("atom_types") or []
    if atom_types:
        from crystal_dlm.fixed_slot import Z_TO_SYMBOL

        return [Z_TO_SYMBOL[int(value)] for value in atom_types]
    symbols = row.get("symbols") or []
    counts = row.get("counts") or []
    expanded: list[str] = []
    for symbol, count in zip(symbols, counts):
        expanded.extend([str(symbol)] * int(count))
    return expanded


def answer_with_replaced_elements(answer: str, symbols: list[str]) -> str:
    tokens = tokenize_answer_text(answer)
    if len(tokens) < 107:
        raise ValueError("source answer does not contain 107 fixed-slot tokens")
    num_atoms = int(tokens[0][3:6])
    if len(symbols) != num_atoms:
        raise ValueError(f"replacement has {len(symbols)} atoms, expected {num_atoms}")
    for slot_idx, symbol in enumerate(symbols):
        tokens[8 + 5 * slot_idx] = f"<E_{symbol}>"
    return "".join(tokens[:107])


def reference_pool(cache_rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in cache_rows:
        if not row.get("strict_valid"):
            continue
        if row.get("has_pbc_equivalent_duplicate"):
            continue
        num_atoms = int(row.get("num_atoms") or 0)
        if num_atoms > 0:
            grouped[num_atoms].append(row)
    return grouped


def should_edit(diagnosis: dict[str, Any]) -> bool:
    if diagnosis.get("has_pbc_equivalent_duplicate"):
        return True
    return str(diagnosis.get("reason")) in FAILURE_REASONS_FOR_EDIT


def maybe_validate_graph(arrays: dict[str, Any], process_one: Any | None) -> bool:
    if process_one is None:
        return True
    graph_from_arrays(arrays, process_one)
    return True


def accepted_row(
    *,
    prompt: str,
    answer: str,
    row: dict[str, Any],
    row_idx: int,
    source_diag: dict[str, Any],
    accepted_diag: dict[str, Any],
    candidate_idx: int | None,
    source_kind: str,
    reference_sample_id: Any = None,
) -> dict[str, Any]:
    formula = str(accepted_diag["comp"]["formula"])
    if source_kind == "source_success":
        mask_policy = "n_active_element"
    else:
        mask_policy = "active_element"
    return {
        "task": "unconditional",
        "prompt": prompt,
        "answer": answer,
        "text": prompt.rstrip() + "\n" + answer,
        "mask_policy": mask_policy,
        "seal_buffer": True,
        "seal_source_kind": source_kind,
        "source_sample_idx": row.get("sample_idx", row_idx),
        "candidate_idx": candidate_idx,
        "reason_before": source_diag["reason"],
        "reason_after": accepted_diag["reason"],
        "formula_after": formula,
        "reference_sample_id": reference_sample_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen")
    parser.add_argument("--candidates-per-source", type=int, default=16)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--max-accepted", type=int, default=None)
    parser.add_argument("--max-formula-repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--skip-graph-validation", action="store_true")
    parser.add_argument(
        "--allow-raw-buffer",
        action="store_true",
        help=(
            "Explicitly allow building an SFT buffer from raw rollouts. "
            "Default MP-20 training should use build_refined_seal_success_buffer.py instead."
        ),
    )
    parser.add_argument(
        "--include-source-successes",
        action="store_true",
        help="Also accept model rollouts that are already strict-valid, graph OK, and non-duplicate.",
    )
    args = parser.parse_args()
    if not args.allow_raw_buffer:
        raise SystemExit(
            "Refusing to build an SFT buffer from raw generations. "
            "Use scripts/build_refined_seal_success_buffer.py with --refined-pt, "
            "or pass --allow-raw-buffer only for a deliberate ablation."
        )

    rng = random.Random(args.seed)
    cache_rows = read_jsonl(args.reference_cache)
    pool_by_n = reference_pool(cache_rows)
    process_one = None if args.skip_graph_validation else import_process_one(args.crysllmgen_dir)
    accepted: list[dict[str, Any]] = []
    formula_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    accepted_kind_counts: Counter[str] = Counter()
    source_seen = source_eligible = 0
    for row_idx, row in enumerate(read_jsonl(args.source_jsonl)):
        if args.max_sources is not None and source_seen >= int(args.max_sources):
            break
        if args.max_accepted is not None and len(accepted) >= int(args.max_accepted):
            break
        text = str(row.get("response") or row.get("answer") or row.get("text") or "")
        if not text:
            continue
        prompt = row.get("prompt") or CANONICAL_PROMPT
        source_seen += 1
        try:
            source_diag = classify_answer(text)
        except Exception as exc:
            failure_counts[f"source_parse:{type(exc).__name__}"] += 1
            continue
        if args.include_source_successes and source_diag["strict_valid"] and not source_diag["has_pbc_equivalent_duplicate"]:
            try:
                formula = str(source_diag["comp"]["formula"])
                if formula_counts[formula] >= int(args.max_formula_repeats):
                    failure_counts["source_success_formula_repeat_cap"] += 1
                else:
                    maybe_validate_graph(source_diag["arrays"], process_one)
                    formula_counts[formula] += 1
                    accepted.append(
                        accepted_row(
                            prompt=prompt,
                            answer=text,
                            row=row,
                            row_idx=row_idx,
                            source_diag=source_diag,
                            accepted_diag=source_diag,
                            candidate_idx=None,
                            source_kind="source_success",
                        )
                    )
                    accepted_kind_counts["source_success"] += 1
                    if args.max_accepted is not None and len(accepted) >= int(args.max_accepted):
                        break
            except Exception as exc:
                failure_counts[f"source_success:{type(exc).__name__}"] += 1
        if not should_edit(source_diag):
            continue
        source_eligible += 1
        num_atoms = int(source_diag["arrays"]["num_atoms"])
        candidates = list(pool_by_n.get(num_atoms) or [])
        if not candidates:
            failure_counts["no_reference_same_n"] += 1
            continue
        for cand_idx in range(int(args.candidates_per_source)):
            ref = rng.choice(candidates)
            symbols = expand_atom_symbols(ref)
            rng.shuffle(symbols)
            try:
                edited_answer = answer_with_replaced_elements(text, symbols)
                edited_diag = classify_answer(edited_answer)
                if not edited_diag["strict_valid"]:
                    failure_counts[f"not_strict:{edited_diag['reason']}"] += 1
                    continue
                if edited_diag["has_pbc_equivalent_duplicate"]:
                    failure_counts["pbc_duplicate"] += 1
                    continue
                formula = str(edited_diag["comp"]["formula"])
                if formula_counts[formula] >= int(args.max_formula_repeats):
                    failure_counts["formula_repeat_cap"] += 1
                    continue
                maybe_validate_graph(edited_diag["arrays"], process_one)
                formula_counts[formula] += 1
                accepted.append(
                    accepted_row(
                        prompt=prompt,
                        answer=edited_answer,
                        row=row,
                        row_idx=row_idx,
                        source_diag=source_diag,
                        accepted_diag=edited_diag,
                        candidate_idx=cand_idx,
                        source_kind="edited_success",
                        reference_sample_id=ref.get("sample_id"),
                    )
                )
                accepted_kind_counts["edited_success"] += 1
                if args.max_accepted is not None and len(accepted) >= int(args.max_accepted):
                    break
            except Exception as exc:
                failure_counts[f"candidate:{type(exc).__name__}"] += 1
        if args.max_accepted is not None and len(accepted) >= int(args.max_accepted):
            break
    written = write_jsonl(args.output_jsonl, accepted)
    summary = {
        "source_jsonl": str(args.source_jsonl),
        "reference_cache": str(args.reference_cache),
        "output_jsonl": str(args.output_jsonl),
        "source_seen": source_seen,
        "source_eligible": source_eligible,
        "accepted": written,
        "accepted_kind_counts": dict(accepted_kind_counts.most_common()),
        "accepted_per_eligible_source": written / max(1, source_eligible),
        "formula_top20": dict(formula_counts.most_common(20)),
        "failure_counts": dict(failure_counts.most_common()),
    }
    write_json(str(args.summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
