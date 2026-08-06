#!/usr/bin/env python3
"""Paired, field-matched direct PlanGraph dependency likelihood screen."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


PANEL_ROWS = 100
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_801


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object row")
                rows.append(value)
    return rows


def row_num_atoms(row: Mapping[str, Any]) -> int:
    graph = row.get("plangraph")
    if not isinstance(graph, Mapping):
        raise ValueError("validation row is missing PlanGraph")
    composition = graph.get("composition")
    if not isinstance(composition, Mapping):
        raise ValueError("validation PlanGraph is missing composition")
    value = int(composition["N"])
    if value < 1 or value > 20:
        raise ValueError(f"invalid PlanGraph N={value}")
    return value


def select_donor_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    panel_rows: int = PANEL_ROWS,
) -> list[int]:
    """Select the next same-N, different-answer donor, wrapping once."""

    if len(rows) < int(panel_rows):
        raise ValueError("validation data is shorter than the frozen panel")
    by_n: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_n[row_num_atoms(row)].append(index)

    donors: list[int] = []
    for target_index in range(int(panel_rows)):
        target = rows[target_index]
        target_answer = str(target.get("answer", ""))
        target_identity = str(target.get("training_pair_sha256", ""))
        candidates = [
            index
            for index in by_n[row_num_atoms(target)]
            if index != target_index
            and str(rows[index].get("answer", "")) != target_answer
            and (
                not target_identity
                or str(rows[index].get("training_pair_sha256", ""))
                != target_identity
            )
        ]
        after = [index for index in candidates if index > target_index]
        if after:
            donors.append(min(after))
        elif candidates:
            donors.append(min(candidates))
        else:
            raise ValueError(
                f"no deterministic same-N donor for panel ordinal {target_index}"
            )
    if len(set(zip(range(int(panel_rows)), donors))) != int(panel_rows):
        raise AssertionError("target/donor pairs are not unique")
    return donors


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("bootstrap requires at least one row margin")
    rng = random.Random(int(seed))
    size = len(samples)
    draws = [
        mean(samples[rng.randrange(size)] for _ in range(size))
        for _ in range(int(replicates))
    ]
    return {
        "mean": mean(samples),
        "ci95": [quantile(draws, 0.025), quantile(draws, 0.975)],
        "replicates": int(replicates),
        "seed": int(seed),
        "unit": "validation_row",
    }


def _load_model(model_path: Path, checkpoint_path: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    from crystal_dlm.llada_resize import ensure_llada_vocab_size
    from crystal_dlm.transformers_compat import (
        ensure_create_bidirectional_mask,
        ensure_llada2_rope_parameters,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ensure_create_bidirectional_mask()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    ensure_llada2_rope_parameters(config)
    model_class = (
        AutoModelForCausalLM
        if getattr(config, "model_type", None) == "llada2_moe"
        else AutoModel
    )
    model = model_class.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))
    ensure_llada_vocab_size(model, len(tokenizer))
    model = PeftModel.from_pretrained(
        model,
        str(checkpoint_path),
        is_trainable=False,
    )
    model.to(torch.device("cuda", 0)).eval()
    return tokenizer, model


def _tokenize_row(tokenizer, row: Mapping[str, Any]) -> tuple[list[int], list[int], list[int]]:
    prompt = str(row["prompt"]).rstrip() + "\n"
    answer = str(row["answer"])
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    answer_ids = list(tokenizer(answer, add_special_tokens=False)["input_ids"])
    full_ids = list(
        tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]
    )
    if full_ids != [*prompt_ids, *answer_ids]:
        raise ValueError("prompt+answer tokenization is not additive")
    expected = 7 + 4 * row_num_atoms(row)
    if len(answer_ids) != expected:
        raise ValueError(
            f"dynamic-v1 answer token count changed: {len(answer_ids)} != {expected}"
        )
    return prompt_ids, answer_ids, full_ids


def build_pair_records(
    rows: Sequence[Mapping[str, Any]],
    tokenizer,
) -> tuple[list[dict[str, Any]], list[int]]:
    from crystal_dlm.fixed_slot import MASK_TOKEN_ID
    from crystal_dlm.planned_corruption import plangraph_dependency_groups

    donors = select_donor_indices(rows)
    records: list[dict[str, Any]] = []
    for target_index in range(PANEL_ROWS):
        target = rows[target_index]
        donor_index = donors[target_index]
        donor = rows[donor_index]
        prompt_ids, answer_ids, full_ids = _tokenize_row(tokenizer, target)
        _, donor_answer_ids, _ = _tokenize_row(tokenizer, donor)
        if len(donor_answer_ids) != len(answer_ids):
            raise ValueError("same-N donor changed answer token length")
        groups = plangraph_dependency_groups(target["plangraph"])
        if len(groups) < 2:
            raise ValueError("PlanGraph must expose at least one dependent group")

        for active_index in range(1, len(groups)):
            target_positions = tuple(groups[active_index].positions)
            prerequisite_positions = tuple(
                position
                for group in groups[:active_index]
                for position in group.positions
            )
            future_positions = tuple(
                position
                for group in groups[active_index + 1 :]
                for position in group.positions
            )
            matched = list(full_ids)
            counterfactual = list(full_ids)
            prompt_length = len(prompt_ids)
            for relative in prerequisite_positions:
                counterfactual[prompt_length + relative] = donor_answer_ids[relative]
            for relative in (*target_positions, *future_positions):
                absolute = prompt_length + relative
                matched[absolute] = int(MASK_TOKEN_ID)
                counterfactual[absolute] = int(MASK_TOKEN_ID)
            absolute_targets = tuple(
                prompt_length + relative for relative in target_positions
            )
            records.append(
                {
                    "panel_ordinal": target_index,
                    "donor_validation_ordinal": donor_index,
                    "num_atoms": row_num_atoms(target),
                    "active_group_index": active_index,
                    "active_group": groups[active_index].name,
                    "target_positions": absolute_targets,
                    "target_token_ids": tuple(
                        answer_ids[relative] for relative in target_positions
                    ),
                    "prerequisite_token_count": len(prerequisite_positions),
                    "target_token_count": len(target_positions),
                    "matched_input_ids": matched,
                    "counterfactual_input_ids": counterfactual,
                }
            )
    return records, donors


def score_records(
    records: Sequence[Mapping[str, Any]],
    *,
    tokenizer,
    model,
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    device = torch.device("cuda", 0)
    scored: list[dict[str, Any]] = []
    for offset in range(0, len(records), int(batch_size)):
        batch = list(records[offset : offset + int(batch_size)])
        sequences = [
            sequence
            for record in batch
            for sequence in (
                list(record["matched_input_ids"]),
                list(record["counterfactual_input_ids"]),
            )
        ]
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), max_length),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(sequences), max_length),
            dtype=torch.long,
            device=device,
        )
        for sequence_index, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[sequence_index, :length] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=device,
            )
            attention_mask[sequence_index, :length] = 1
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
        for pair_index, record in enumerate(batch):
            positions = torch.tensor(
                record["target_positions"],
                dtype=torch.long,
                device=device,
            )
            targets = torch.tensor(
                record["target_token_ids"],
                dtype=torch.long,
                device=device,
            )
            matched_nll = float(
                functional.cross_entropy(
                    logits[2 * pair_index, positions].float(),
                    targets,
                    reduction="mean",
                )
                .detach()
                .cpu()
            )
            counterfactual_nll = float(
                functional.cross_entropy(
                    logits[2 * pair_index + 1, positions].float(),
                    targets,
                    reduction="mean",
                )
                .detach()
                .cpu()
            )
            scored.append(
                {
                    "panel_ordinal": int(record["panel_ordinal"]),
                    "donor_validation_ordinal": int(
                        record["donor_validation_ordinal"]
                    ),
                    "num_atoms": int(record["num_atoms"]),
                    "active_group_index": int(record["active_group_index"]),
                    "active_group": str(record["active_group"]),
                    "prerequisite_token_count": int(
                        record["prerequisite_token_count"]
                    ),
                    "target_token_count": int(record["target_token_count"]),
                    "matched_nll": matched_nll,
                    "counterfactual_nll": counterfactual_nll,
                    "dependency_margin": counterfactual_nll - matched_nll,
                }
            )
        del logits, input_ids, attention_mask
    return scored


def aggregate(scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_row: dict[int, list[float]] = defaultdict(list)
    by_group: dict[str, list[float]] = defaultdict(list)
    for record in scored:
        margin = float(record["dependency_margin"])
        if not math.isfinite(margin):
            raise ValueError("dependency margin is non-finite")
        by_row[int(record["panel_ordinal"])].append(margin)
        by_group[str(record["active_group"])].append(margin)
    if sorted(by_row) != list(range(PANEL_ROWS)):
        raise ValueError("dependency screen did not retain all 100 panel rows")
    row_records = [
        {
            "panel_ordinal": ordinal,
            "group_pairs": len(by_row[ordinal]),
            "dependency_margin": mean(by_row[ordinal]),
        }
        for ordinal in range(PANEL_ROWS)
    ]
    row_margins = [float(record["dependency_margin"]) for record in row_records]
    return {
        "pair_count": len(scored),
        "panel_rows": PANEL_ROWS,
        "row_records": row_records,
        "arm_margin": bootstrap_mean_ci(row_margins),
        "group_margin_means": {
            key: mean(values) for key, values in sorted(by_group.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("B1", "B2"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if int(args.batch_size) < 1:
        raise ValueError("batch size must be positive")
    rows = load_jsonl(args.validation_jsonl)
    tokenizer, model = _load_model(args.model_path, args.checkpoint_path)
    import torch

    torch.cuda.reset_peak_memory_stats(0)
    records, donors = build_pair_records(rows, tokenizer)
    scored = score_records(
        records,
        tokenizer=tokenizer,
        model=model,
        batch_size=int(args.batch_size),
    )
    result = aggregate(scored)

    report = {
        "schema": "h1a2_v3_paired_direct_dependency_margin_arm_v1",
        "status": "complete",
        "arm": args.arm,
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_adapter_sha256": args.checkpoint_sha256,
        "execution_manifest_sha256": args.execution_manifest_sha256,
        "validation_jsonl_sha256": sha256_file(args.validation_jsonl),
        "data_manifest_sha256": sha256_file(args.data_manifest),
        "panel_ordinals": list(range(PANEL_ROWS)),
        "donor_validation_ordinals": donors,
        "donor_rule": "next_same_N_different_answer_wrap_once",
        "matched_field_and_mask_budget": True,
        "target_groups": "all_non_root_compiled_plangraph_groups",
        "nll_reduction": "unweighted_mean_token_cross_entropy_per_active_group",
        "result": result,
        "pair_records": scored,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_reserved(0)),
        "generation_sun_energy_or_hull_used": False,
        "shuffle_training_arm_used": False,
        "automatic_promotion": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "arm": args.arm,
                "pair_count": result["pair_count"],
                "panel_rows": result["panel_rows"],
                "arm_margin": result["arm_margin"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
