"""Target-only sequence losses for H1-A2C JointChem.

Torch imports are deliberately deferred so the data and masking contracts can
be unit-tested in a dependency-light environment.
"""

from __future__ import annotations

from typing import Sequence


def build_target_only_token_lists(tokenizer, prompt: str, answer: str, max_length: int):
    """Tokenize prompt+answer while supervising only the complete answer."""

    eos = tokenizer.eos_token or ""
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    answer_ids = list(
        tokenizer(str(answer).strip() + eos, add_special_tokens=False)["input_ids"]
    )
    if len(answer_ids) >= int(max_length):
        raise ValueError(
            f"answer has {len(answer_ids)} tokens and does not fit max_length={max_length}"
        )
    max_prompt_tokens = int(max_length) - len(answer_ids)
    if len(prompt_ids) > max_prompt_tokens:
        prompt_ids = prompt_ids[-max_prompt_tokens:]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    if any(value != -100 for value in labels[: len(prompt_ids)]):
        raise AssertionError("prompt token entered target supervision")
    return {
        "input_ids": input_ids,
        "labels": labels,
        "prompt_token_count": len(prompt_ids),
        "target_token_count": len(answer_ids),
    }


def supervised_target_positions(labels: Sequence[int]) -> tuple[int, ...]:
    """Return causal-LM target positions after the standard one-token shift."""

    return tuple(index for index, value in enumerate(labels[1:], start=1) if int(value) != -100)


def target_only_mean_nll(logits, labels):
    """Return one length-normalized answer NLL per batch element."""

    import torch
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    mask = (shift_labels != -100).to(dtype=losses.dtype)
    supervised = mask.sum(dim=1)
    if bool(torch.any(supervised <= 0)):
        raise ValueError("a sequence has no supervised answer tokens")
    return (losses * mask).sum(dim=1) / supervised


def pairwise_preference_loss(positive_nll, negative_nll, *, margin: float = 0.10):
    """Soft margin preferring lower normalized NLL for the positive Plan."""

    import torch.nn.functional as F

    if positive_nll.shape != negative_nll.shape:
        raise ValueError("positive and negative NLL tensors must have the same shape")
    return F.softplus(float(margin) + positive_nll - negative_nll).mean()


def combine_active_losses(weighted_losses):
    """Renormalize a list of ``(weight, scalar_loss)`` active components."""

    active = [(float(weight), loss) for weight, loss in weighted_losses if loss is not None]
    if not active:
        raise ValueError("at least one loss component is required")
    total_weight = sum(weight for weight, _ in active)
    if total_weight <= 0:
        raise ValueError("active loss weights must sum to a positive value")
    return sum(weight * loss for weight, loss in active) / total_weight
