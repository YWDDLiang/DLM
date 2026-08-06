#!/usr/bin/env python3
"""Build H2 rich-plan conditioned CrysLLMGen plain-text DLM SFT data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.crysllmgen_text import (  # noqa: E402
    CRYSLLMGEN_MODULE_TO_ID,
    parse_crysllmgen_text,
    structure_to_crysllmgen_text,
    write_json,
)
from crystal_dlm.fixed_slot import metadata_from_csv_row  # noqa: E402
from crystal_dlm.h2_plaintext_dlm import (  # noqa: E402
    H2_PLAINTEXT_DLM_PROMPT_VERSION,
    H2_PLAINTEXT_DLM_REPRESENTATION,
    build_h2_plaintext_prompt,
    proposal_matches_plan_composition,
)
from crystal_dlm.r5_plan_body import H1_RICH_PLAN_FORMAT, format_composition_plan  # noqa: E402
from crystal_dlm.r5_plan_state import PLAN_STATE_VERSION, plan_state_from_arrays  # noqa: E402
from scripts.build_crysllmgen_text_sft_data import (  # noqa: E402
    histogram_add,
    import_crysllmgen_process_one,
    load_tokenizer,
    read_rows,
    token_len,
    validate_graph,
)


def build_split(
    *,
    split: str,
    input_csv: Path,
    output_jsonl: Path,
    tokenizer,
    process_one,
    seed: int,
    limit: int | None,
    skip_graph_validation: bool,
    progress_every: int,
    append_eos: bool,
    train_origin_shift: bool,
    train_permute_sites: bool,
) -> Dict[str, Any]:
    from pymatgen.core import Structure

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    eos_text = tokenizer.eos_token if append_eos and tokenizer is not None and tokenizer.eos_token else ""
    stats: Dict[str, Any] = {
        "split": split,
        "input_csv": str(input_csv),
        "rows_seen": 0,
        "rows_written": 0,
        "failures": 0,
        "representation": "crysllmgen_text",
        "h2_representation": H2_PLAINTEXT_DLM_REPRESENTATION,
        "prompt_model_lengths": {},
        "answer_model_lengths": {},
        "atom_count_histogram": {},
        "num_elements_histogram": {},
        "element_histogram": {},
        "anion_framework_histogram": {},
        "charge_bucket_histogram": {},
        "lattice_system_histogram": {},
        "spacegroup_bucket_histogram": {},
        "volume_per_atom_bin_histogram": {},
    }
    failure_path = output_jsonl.with_suffix(".failure_cases.jsonl")
    with output_jsonl.open("w", encoding="utf-8") as out, failure_path.open("w", encoding="utf-8") as failures:
        for row_idx, row in enumerate(read_rows(input_csv, limit=limit)):
            stats["rows_seen"] += 1
            try:
                structure = Structure.from_str(row["cif"], fmt="cif")
                split_offset = {"train": 0, "val": 1, "test": 2}[split]
                row_rng = random.Random(int(seed) + split_offset * 1_000_003 + row_idx)
                answer_core, arrays = structure_to_crysllmgen_text(
                    structure,
                    rng=row_rng,
                    origin_shift=split == "train" and train_origin_shift,
                    permute_sites=split == "train" and train_permute_sites,
                )
                parsed_answer = parse_crysllmgen_text(answer_core)
                metadata = metadata_from_csv_row(row)
                plan = plan_state_from_arrays(parsed_answer, metadata=metadata)
                if not proposal_matches_plan_composition(parsed_answer, plan):
                    raise ValueError("teacher CrysLLMGen text composition does not match teacher plan")
                if not skip_graph_validation:
                    validate_graph(parsed_answer, process_one)
                prompt = build_h2_plaintext_prompt(plan)
                answer = answer_core + eos_text
                plan_text = format_composition_plan(plan, plan_style=H1_RICH_PLAN_FORMAT)
                record = {
                    "task": "h2_plaintext_dlm_proposal",
                    "module": "full",
                    "module_id": CRYSLLMGEN_MODULE_TO_ID["full"],
                    "representation": "crysllmgen_text",
                    "h2_representation": H2_PLAINTEXT_DLM_REPRESENTATION,
                    "prompt_version": H2_PLAINTEXT_DLM_PROMPT_VERSION,
                    "plan_state_version": PLAN_STATE_VERSION,
                    "plan_format": H1_RICH_PLAN_FORMAT,
                    "prompt": prompt,
                    "answer": answer,
                    "text": prompt.rstrip() + "\n" + answer,
                    "plan_text": plan_text,
                    "plan_state": dict(plan),
                    "num_atoms": int(parsed_answer["num_atoms"]),
                    "num_elements": len(set(parsed_answer["species"])),
                    "metadata": metadata,
                    "loss_profile": "text",
                    "sample_weight": 1.0,
                    "prompt_length": token_len(tokenizer, prompt.rstrip() + "\n"),
                    "answer_model_length": token_len(tokenizer, answer),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1
                if record["prompt_length"] is not None:
                    histogram_add(stats["prompt_model_lengths"], record["prompt_length"])
                if record["answer_model_length"] is not None:
                    histogram_add(stats["answer_model_lengths"], record["answer_model_length"])
                histogram_add(stats["atom_count_histogram"], record["num_atoms"])
                histogram_add(stats["num_elements_histogram"], record["num_elements"])
                for symbol, count in Counter(parsed_answer["species"]).items():
                    histogram_add(stats["element_histogram"], symbol, count)
                histogram_add(stats["anion_framework_histogram"], plan.get("anion_framework", "unknown"))
                histogram_add(stats["charge_bucket_histogram"], plan.get("charge_bucket", "unknown"))
                histogram_add(stats["lattice_system_histogram"], plan.get("lattice_system", "unknown"))
                histogram_add(stats["spacegroup_bucket_histogram"], plan.get("spacegroup_bucket", "sg_unknown"))
                histogram_add(stats["volume_per_atom_bin_histogram"], plan.get("volume_per_atom_bin", "volpa_unknown"))
            except Exception as exc:  # noqa: BLE001
                stats["failures"] += 1
                failures.write(
                    json.dumps(
                        {
                            "split": split,
                            "row_idx": row_idx,
                            "material_id": row.get("material_id"),
                            "reason": type(exc).__name__,
                            "message": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if progress_every > 0 and stats["rows_seen"] % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "h2_plaintext_dlm_builder_progress",
                            "split": split,
                            "rows_seen": stats["rows_seen"],
                            "rows_written": stats["rows_written"],
                            "failures": stats["failures"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return stats


def max_hist_key(histogram: Dict[str, int]) -> int:
    if not histogram:
        return 0
    return max(int(float(key)) for key in histogram)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20_h2_plaintext_dlm")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--crysllmgen-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen")
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--skip-graph-validation", action="store_true")
    parser.add_argument("--append-eos", action="store_true", default=True)
    parser.add_argument("--no-append-eos", dest="append_eos", action="store_false")
    parser.add_argument("--train-origin-shift", dest="train_origin_shift", action="store_true", default=True)
    parser.add_argument("--no-train-origin-shift", dest="train_origin_shift", action="store_false")
    parser.add_argument("--train-permute-sites", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_path)
    process_one = None if args.skip_graph_validation else import_crysllmgen_process_one(args.crysllmgen_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        splits[split] = build_split(
            split=split,
            input_csv=args.input_dir / f"{split}.csv",
            output_jsonl=args.output_dir / f"{split}.jsonl",
            tokenizer=tokenizer,
            process_one=process_one,
            seed=args.seed,
            limit=args.limit,
            skip_graph_validation=bool(args.skip_graph_validation),
            progress_every=int(args.progress_every),
            append_eos=bool(args.append_eos),
            train_origin_shift=bool(args.train_origin_shift),
            train_permute_sites=bool(args.train_permute_sites),
        )

    max_answer = max(max_hist_key(item["answer_model_lengths"]) for item in splits.values())
    max_prompt = max(max_hist_key(item["prompt_model_lengths"]) for item in splits.values())
    summary = {
        "representation": "crysllmgen_text",
        "h2_representation": H2_PLAINTEXT_DLM_REPRESENTATION,
        "prompt_version": H2_PLAINTEXT_DLM_PROMPT_VERSION,
        "plan_format": H1_RICH_PLAN_FORMAT,
        "answer_layout": "CrysLLMGen plain text conditioned on ordinary-text rich plan",
        "splits": splits,
        "answer_token_count": max_answer + 8,
        "max_answer_model_length": max_answer,
        "max_prompt_model_length": max_prompt,
        "max_length_recommended": max_prompt + max_answer + 16,
        "special_token_count": 0,
        "append_eos": bool(args.append_eos),
        "train_origin_shift": bool(args.train_origin_shift),
        "train_permute_sites": bool(args.train_permute_sites),
    }
    write_json(str(args.output_dir / "stats.json"), summary)
    write_json(
        str(args.output_dir / "_SUCCESS"),
        {
            "representation": "crysllmgen_text",
            "h2_representation": H2_PLAINTEXT_DLM_REPRESENTATION,
            "prompt_version": H2_PLAINTEXT_DLM_PROMPT_VERSION,
            "complete": True,
            "splits": {
                split: {
                    "rows_seen": split_stats["rows_seen"],
                    "rows_written": split_stats["rows_written"],
                    "failures": split_stats["failures"],
                }
                for split, split_stats in splits.items()
            },
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
