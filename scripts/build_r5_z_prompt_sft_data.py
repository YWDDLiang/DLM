#!/usr/bin/env python3
"""Build R5 prompt-side z-conditioned fixed-slot SFT data.

The answer remains the original 107-token fixed-slot body.  Physical state z is
written only in the prompt.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.fixed_slot import ANSWER_TOKEN_COUNT, parse_fixed_slot_answer, write_json  # noqa: E402
from crystal_dlm.r5_conditioning import (  # noqa: E402
    answer_has_only_fixed_slot_body,
    build_r5_prompt,
    build_z_payload_from_answer,
    bucket_counts,
    formula_cap_weight,
    validate_z_matches_answer,
    tier_sample_weight,
)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            count += 1
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return count


def read_csv_rows(csv_dir: Path, split: str) -> list[dict[str, Any]]:
    path = csv_dir / f"{split}.csv"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def csv_by_material(csv_rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in csv_rows:
        material_id = row.get("material_id")
        if material_id:
            out[str(material_id)] = dict(row)
    return out


def coerce_csv_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "material_id",
        "formation_energy_per_atom",
        "band_gap",
        "pretty_formula",
        "e_above_hull",
        "elements",
        "spacegroup.number",
        "spacegroup.number.conv",
    ]
    metadata: dict[str, Any] = {}
    for key in keys:
        if key not in row or row[key] in ("", None):
            continue
        value = row[key]
        try:
            text = str(value)
            metadata[key] = float(text) if "." in text or "e" in text.lower() else int(text)
        except Exception:
            metadata[key] = value
    return metadata


def merge_metadata(
    row: Mapping[str, Any],
    *,
    split: str,
    idx: int,
    csv_rows: list[dict[str, Any]],
    csv_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    material_id = metadata.get("material_id")
    csv_row = csv_by_id.get(str(material_id)) if material_id else None
    if csv_row is None and idx < len(csv_rows):
        csv_row = csv_rows[idx]
    if csv_row:
        for key, value in coerce_csv_metadata(csv_row).items():
            metadata.setdefault(key, value)
    metadata.setdefault("source_split", split)
    metadata.setdefault("source_row_idx", idx)
    return metadata


def maybe_load_tokenizer(tokenizer_path: str | None, vocab_file: Path):
    if not tokenizer_path:
        return None
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if vocab_file.exists():
        with vocab_file.open(encoding="utf-8") as handle:
            tokens = [line.strip() for line in handle if line.strip()]
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    return tokenizer


def token_len(tokenizer, text: str) -> int | None:
    if tokenizer is None:
        return None
    return int(len(tokenizer(text, add_special_tokens=False)["input_ids"]))


def stable_sort_key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    tier = str(row.get("ehull_tier", "unknown_ehull"))
    rank = {"strict_anchor": 0, "meta_anchor": 1}.get(tier, 2)
    return (rank, str(row.get("prototype_key", "")), str(row.get("source_id", "")))


def prototype_id(z: Mapping[str, Any], split: str, idx: int) -> str:
    source = f"{split}:{idx}:{z.get('material_id')}:{z.get('prototype_key')}:{z.get('full_formula')}"
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return f"r5proto_{digest}"


def make_conditioned_row(
    row: Mapping[str, Any],
    *,
    z: Mapping[str, Any],
    tokenizer,
    sample_weight: float,
) -> dict[str, Any]:
    answer = str(row["answer"])
    validate_z_matches_answer(z, answer)
    if not answer_has_only_fixed_slot_body(answer):
        raise ValueError("answer is not exactly one 107-token fixed-slot body")
    prompt = build_r5_prompt(z)
    metadata = dict(row.get("metadata") or {})
    metadata["r5_z"] = dict(z)
    prompt_text = prompt.rstrip() + "\n"
    out = dict(row)
    out.update(
        {
            "task": "r5_z_prompt_fixed_slot",
            "representation": "fixed_slot",
            "prompt": prompt,
            "answer": answer,
            "text": prompt_text + answer,
            "prompt_length": token_len(tokenizer, prompt_text),
            "answer_model_length": token_len(tokenizer, answer),
            "answer_token_count": ANSWER_TOKEN_COUNT,
            "loss_profile": "fixed_slot",
            "selection_role": "r5_z_conditioned",
            "sample_weight": float(sample_weight),
            "r5_z": dict(z),
            "metadata": metadata,
        }
    )
    return out


def annotate_replay_row(row: Mapping[str, Any], z: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    metadata = dict(row.get("metadata") or {})
    metadata["r5_z_reference"] = dict(z)
    out.update(
        {
            "representation": "fixed_slot",
            "answer_token_count": ANSWER_TOKEN_COUNT,
            "selection_role": "r5_unconditional_replay",
            "r5_z": dict(z),
            "metadata": metadata,
        }
    )
    return out


def build_split(
    *,
    split: str,
    input_rows: list[dict[str, Any]],
    csv_rows: list[dict[str, Any]],
    tokenizer,
    rng: random.Random,
    replay_fraction: float,
    replay_train_only: bool,
    formula_weight_cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    csv_id = csv_by_material(csv_rows)
    formula_counts: Counter[str] = Counter()
    z_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(input_rows):
        try:
            metadata = merge_metadata(row, split=split, idx=idx, csv_rows=csv_rows, csv_by_id=csv_id)
            answer = str(row["answer"])
            arrays = parse_fixed_slot_answer(answer, strict=True)
            z = build_z_payload_from_answer(answer, metadata=metadata, strict=True)
            z["source_split"] = split
            z["source_row_idx"] = idx
            z["source_id"] = metadata.get("material_id") or f"{split}:{idx}"
            z["prototype_id"] = prototype_id(z, split, idx)
            formula_counts[str(z["full_formula"])] += 1
            z_rows.append({"row": row, "z": z, "arrays": arrays})
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "split": split,
                    "row_idx": idx,
                    "stage": "z_payload",
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )

    for item in z_rows:
        row = item["row"]
        z = item["z"]
        base_weight = float(row.get("sample_weight", 1.0) or 1.0) * tier_sample_weight(str(z["ehull_tier"]))
        weight = formula_cap_weight(base_weight, formula_counts[str(z["full_formula"])], formula_weight_cap)
        try:
            output_rows.append(make_conditioned_row(row, z=z, tokenizer=tokenizer, sample_weight=weight))
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "split": split,
                    "row_idx": z["source_row_idx"],
                    "stage": "conditioned_row",
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue

        if str(z["ehull_tier"]) in {"strict_anchor", "meta_anchor"}:
            prototype_rows.append(
                {
                    "prototype_id": z["prototype_id"],
                    "prototype_key": z["prototype_key"],
                    "source_split": split,
                    "source_row_idx": z["source_row_idx"],
                    "source_id": z.get("source_id"),
                    "ehull_tier": z["ehull_tier"],
                    "e_above_hull": z.get("e_above_hull"),
                    "full_formula": z["full_formula"],
                    "reduced_formula": z["reduced_formula"],
                    "chemsys": z["chemsys"],
                    "anion_framework": z["anion_framework"],
                    "lattice_system": z["lattice_system"],
                    "count_pattern": z["count_pattern"],
                    "volume_per_atom_token": z["volume_per_atom_token"],
                    "high_symmetry_token": z["high_symmetry_token"],
                    "spacegroup_number": z.get("spacegroup_number"),
                    "spacegroup_number_conv": z.get("spacegroup_number_conv"),
                    "z": z,
                    "prompt": build_r5_prompt(z),
                }
            )

        add_replay = (split == "train") or not replay_train_only
        if add_replay and rng.random() < replay_fraction:
            output_rows.append(annotate_replay_row(row, z))

    conditioned_count = sum(1 for row in output_rows if row.get("selection_role") == "r5_z_conditioned")
    replay_count = sum(1 for row in output_rows if row.get("selection_role") == "r5_unconditional_replay")
    stats = {
        "split": split,
        "input_rows": len(input_rows),
        "output_rows": len(output_rows),
        "conditioned_rows": conditioned_count,
        "unconditional_replay_rows": replay_count,
        "conditioned_fraction": conditioned_count / max(1, len(output_rows)),
        "failures": len(failures),
        "ehull_tier_counts": bucket_counts((item["z"] for item in z_rows), "ehull_tier"),
        "anion_counts": bucket_counts((item["z"] for item in z_rows), "anion_framework"),
        "lattice_system_counts": bucket_counts((item["z"] for item in z_rows), "lattice_system"),
        "top_formula_counts": dict(formula_counts.most_common(30)),
    }
    return output_rows, prototype_rows, {"stats": stats, "failures": failures}


def copy_sidecars(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "vocab_tokens.txt",
        "vocab.json",
        "tokenizer_report.json",
        "ehull_weight_summary.json",
    ):
        src = input_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)


def write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# R5 z-prompt fixed-slot data",
        "",
        f"- input_dir: `{summary['input_dir']}`",
        f"- output_dir: `{summary['output_dir']}`",
        f"- answer_token_count: `{ANSWER_TOKEN_COUNT}`",
        f"- replay_fraction: `{summary['replay_fraction']}`",
        f"- formula_weight_cap: `{summary['formula_weight_cap']}`",
        "",
        "| split | input | output | z-conditioned | replay | conditioned fraction | failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, payload in summary["splits"].items():
        lines.append(
            f"| {split} | {payload['input_rows']} | {payload['output_rows']} | "
            f"{payload['conditioned_rows']} | {payload['unconditional_replay_rows']} | "
            f"{payload['conditioned_fraction']:.3f} | {payload['failures']} |"
        )
    lines.extend(
        [
            "",
            "## Prototype Library",
            "",
            f"- stable prototype rows: `{summary['prototype_rows']}`",
            f"- prompt pool rows: `{summary['prompt_pool_rows']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20")
    parser.add_argument("--csv-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20_r5_z_prompt_v0")
    parser.add_argument(
        "--prototype-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data/prototypes/mp20_stable_prototype_library.jsonl",
    )
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--replay-fraction", type=float, default=0.25)
    parser.add_argument("--replay-train-only", action="store_true", default=True)
    parser.add_argument("--formula-weight-cap", type=int, default=64)
    parser.add_argument("--prompt-pool-max", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.prototype_jsonl.parent.mkdir(parents=True, exist_ok=True)
    copy_sidecars(args.input_dir, args.output_dir)
    tokenizer = maybe_load_tokenizer(args.tokenizer_path, args.output_dir / "vocab_tokens.txt")

    split_summaries: dict[str, Any] = {}
    all_prototypes: list[dict[str, Any]] = []
    for offset, split in enumerate(("train", "val", "test")):
        input_rows = read_jsonl(args.input_dir / f"{split}.jsonl", limit=args.limit)
        csv_rows = read_csv_rows(args.csv_dir, split)
        rows, prototypes, payload = build_split(
            split=split,
            input_rows=input_rows,
            csv_rows=csv_rows,
            tokenizer=tokenizer,
            rng=random.Random(int(args.seed) + offset * 10_000),
            replay_fraction=float(args.replay_fraction),
            replay_train_only=bool(args.replay_train_only),
            formula_weight_cap=int(args.formula_weight_cap),
        )
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
        write_jsonl(args.output_dir / f"{split}.failure_cases.jsonl", payload["failures"])
        split_summaries[split] = payload["stats"]
        all_prototypes.extend(prototypes)

    all_prototypes = sorted(all_prototypes, key=stable_sort_key)
    write_jsonl(args.prototype_jsonl, all_prototypes)
    write_jsonl(args.output_dir / "prototype_library.jsonl", all_prototypes)
    prompt_pool = [row for row in all_prototypes if row.get("source_split") == "train"] or all_prototypes
    prompt_pool = prompt_pool[: int(args.prompt_pool_max)]
    write_jsonl(args.output_dir / "prototype_prompt_pool.jsonl", prompt_pool)

    summary = {
        "representation": "fixed_slot",
        "conditioning": "r5_prompt_side_v0",
        "answer_token_count": ANSWER_TOKEN_COUNT,
        "input_dir": str(args.input_dir),
        "csv_dir": str(args.csv_dir),
        "output_dir": str(args.output_dir),
        "prototype_jsonl": str(args.prototype_jsonl),
        "replay_fraction": float(args.replay_fraction),
        "formula_weight_cap": int(args.formula_weight_cap),
        "prototype_rows": len(all_prototypes),
        "prompt_pool_rows": len(prompt_pool),
        "max_length_recommended": 512,
        "splits": split_summaries,
    }
    write_json(str(args.output_dir / "r5_z_prompt_summary.json"), summary)
    write_json(str(args.output_dir / "stats.json"), summary)
    write_json(
        str(args.output_dir / "prompt_pool.json"),
        {
            "prompt_pool": ["r5_z_prompt_fixed_slot", "original_fixed_slot_replay"],
            "canonical_prompt": "r5_z_prompt_fixed_slot",
            "prototype_prompt_pool_jsonl": "prototype_prompt_pool.jsonl",
        },
    )
    write_markdown(summary, args.output_dir / "result.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

