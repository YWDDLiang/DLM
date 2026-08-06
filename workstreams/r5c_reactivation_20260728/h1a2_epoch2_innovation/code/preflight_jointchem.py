#!/usr/bin/env python3
"""Real-tokenizer and deterministic Torch gate for JointChem target-only loss."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from jointchem_loss import (
    combine_active_losses,
    pairwise_preference_loss,
    supervised_target_positions,
    target_only_mean_nll,
)
from train_h1a2_jointchem import tokenize_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-one-a800", action="store_true")
    args = parser.parse_args()

    gpu_count = int(torch.cuda.device_count())
    gpu_names = [str(torch.cuda.get_device_name(index)) for index in range(gpu_count)]
    cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if args.require_one_a800:
        if gpu_count != 1 or "A800" not in gpu_names[0] or cpu_count > 8:
            raise RuntimeError(
                f"resource gate failed: gpu_count={gpu_count}, gpu_names={gpu_names}, cpu={cpu_count}"
            )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.use_deterministic_algorithms(True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_source,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    answer = "\n".join(
        (
            "formula: Li2O",
            "anion: oxide",
            "charge: neutral_plausible",
            "lattice: cubic",
            "spacegroup: sg_195_230",
            "volume: volpa_010_014",
            "end: plan",
        )
    )
    tokenized_a = tokenize_plan(tokenizer, "User: produce a Plan\nAssistant:", answer, 768)
    tokenized_b = tokenize_plan(
        tokenizer,
        "System: chemistry\nUser: produce the same Plan without an identifier\nAssistant:",
        answer,
        768,
    )
    target_a = tokenized_a["labels"][tokenized_a["labels"] != -100].tolist()
    target_b = tokenized_b["labels"][tokenized_b["labels"] != -100].tolist()
    if target_a != target_b or not target_a:
        raise AssertionError("real tokenizer changed supervised target tokens across prompt perturbation")
    prompt_tokens_a = int((tokenized_a["labels"] == -100).sum().item())
    prompt_tokens_b = int((tokenized_b["labels"] == -100).sum().item())
    if prompt_tokens_a <= 0 or prompt_tokens_b <= 0:
        raise AssertionError("real-tokenizer fixture has no masked prompt tokens")
    if len(supervised_target_positions(tokenized_a["labels"].tolist())) != len(target_a):
        raise AssertionError("causal supervised positions do not match answer-token count")

    labels = torch.tensor([[-100, -100, 2, 3, 4]], dtype=torch.long, device=device)
    logits = torch.tensor(
        [
            [
                [9.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0],
                [-1.0, -2.0, 4.0, 0.0, -1.0, -2.0, -3.0],
                [-2.0, -1.0, 0.0, 5.0, 1.0, -2.0, -3.0],
                [-3.0, -2.0, -1.0, 0.0, 6.0, 1.0, -4.0],
                [1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    first = target_only_mean_nll(logits, labels)
    second = target_only_mean_nll(logits.clone(), labels)
    if not torch.equal(first, second):
        raise AssertionError("target-only loss is not bitwise deterministic on the fixed GPU fixture")

    prompt_perturbed = logits.clone()
    prompt_perturbed[:, 0, :] = torch.tensor(
        [100.0, -100.0, 50.0, -50.0, 25.0, -25.0, 0.0],
        device=device,
    )
    prompt_loss = target_only_mean_nll(prompt_perturbed, labels)
    if not torch.equal(first, prompt_loss):
        raise AssertionError("a prompt-only logit changed target-only loss")

    target_perturbed = logits.clone()
    target_perturbed[:, 1, 2] = -20.0
    changed = target_only_mean_nll(target_perturbed, labels)
    if torch.equal(first, changed):
        raise AssertionError("a supervised target logit did not change target-only loss")

    positive = torch.tensor([0.2, 0.4], dtype=torch.float32, device=device)
    negative = torch.tensor([0.5, 0.6], dtype=torch.float32, device=device)
    observed_pairwise = pairwise_preference_loss(positive, negative, margin=0.10)
    expected_pairwise = F.softplus(0.10 + positive - negative).mean()
    if not torch.equal(observed_pairwise, expected_pairwise):
        raise AssertionError("pairwise loss does not match the preregistered formula")
    combined = combine_active_losses(((0.75, first.mean()), (0.15, observed_pairwise), (0.10, None)))
    expected_combined = (0.75 * first.mean() + 0.15 * observed_pairwise) / 0.90
    if not torch.equal(combined, expected_combined):
        raise AssertionError("active-loss renormalization is incorrect")

    report = {
        "schema": "h1a2_jointchem_loss_preflight_v1",
        "ok": True,
        "device": str(device),
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "cpu_count": cpu_count,
        "real_tokenizer_prompt_tokens": [prompt_tokens_a, prompt_tokens_b],
        "real_tokenizer_target_tokens": len(target_a),
        "target_only_prompt_perturbation_invariant": True,
        "target_logit_sensitivity": True,
        "pairwise_formula_exact": True,
        "active_loss_renormalization_exact": True,
        "deterministic_repeat_bitwise_equal": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
