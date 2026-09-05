"""Batch materialization shared by state warmup and attempted-path training."""
from __future__ import annotations

from typing import Any, Sequence

import torch

from crystal_dlm.spad_program import program_from_element_order
from crystal_dlm.state_conditioned_model import CrystalStateContext


def materialize_state_batch(
    examples: Sequence[dict[str, Any]], tokenizer: Any, *, device: torch.device,
    max_length: int = 382, max_sites: int = 20,
) -> dict[str, Any]:
    prefixes = [example.get("prompt_token_ids") for example in examples]
    for i, prefix in enumerate(prefixes):
        if prefix is None:
            prefixes[i] = tokenizer(examples[i]["prompt"].rstrip() + "\n", add_special_tokens=False)["input_ids"]
    lengths = [len(prefix) + len(row["input_body"]) for prefix, row in zip(prefixes, examples)]
    width = max(lengths)
    if width > max_length:
        raise ValueError(f"state length {width} exceeds {max_length}; truncation is not allowed")
    pad = int(tokenizer.pad_token_id)
    batch = len(examples)
    tokens = torch.full((batch, width), pad, dtype=torch.long, device=device)
    old = tokens.clone()
    attention = torch.zeros_like(tokens)
    active = torch.zeros_like(tokens, dtype=torch.bool)
    prompt_lengths = torch.tensor([len(x) for x in prefixes], device=device)
    counts = torch.tensor([int(row["num_atoms"]) for row in examples], device=device)
    ranks = torch.full((batch, max_sites), max_sites, dtype=torch.long, device=device)
    positions, targets = [], []
    for row, (example, prefix, length) in enumerate(zip(examples, prefixes, lengths)):
        prompt = len(prefix)
        expected = 7 + 4 * int(example["num_atoms"])
        if len(example["input_body"]) != expected or len(example["old_body"]) != expected:
            raise ValueError("state changed its exact native body length")
        tokens[row, :length] = torch.tensor(prefix + example["input_body"], device=device)
        old[row, :length] = torch.tensor(prefix + example["old_body"], device=device)
        attention[row, :length] = 1
        active[row, [prompt + p for p in example["transaction_positions"]]] = True
        program = program_from_element_order(
            example["plan_state"], example["species_program"],
            order_source=str(example.get("species_program_source", "recorded_program")),
        )
        for rank, entry in enumerate(program.entries):
            ranks[row, list(entry.slot_indices)] = rank
        positions.append(prompt + int(example["position"]))
        targets.append(int(example["target_token"]))
    return {
        "input_ids": tokens, "attention_mask": attention,
        "geometry_context": CrystalStateContext(old, prompt_lengths, counts, ranks, active),
        "positions": torch.tensor(positions, device=device),
        "targets": torch.tensor(targets, device=device), "examples": examples,
    }


def enable_native_checkpointing(base_model: torch.nn.Module) -> list[str]:
    enabled = []
    for name, module in base_model.named_modules():
        method = getattr(module, "set_activation_checkpointing", None)
        if callable(method) and hasattr(module, "transformer"):
            method("whole_layer")
            enabled.append(name)
    return enabled
