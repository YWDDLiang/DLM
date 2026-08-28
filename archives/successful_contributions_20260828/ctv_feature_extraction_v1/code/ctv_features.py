"""Frozen feature helpers for CTV value heads."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


CTV_GEOMETRY_PREFIXES = (
    "<LA_",
    "<LB_",
    "<LC_",
    "<AA_",
    "<AB_",
    "<AG_",
    "<X_",
    "<Y_",
    "<Z_",
)


def exact_prompt_length(total_tokens: int, num_atoms: int) -> int:
    total = int(total_tokens)
    atoms = int(num_atoms)
    if not 1 <= atoms <= 20:
        raise ValueError("CTV feature extraction requires N in 1..20")
    body = 7 + 4 * atoms
    prompt = total - body
    if prompt <= 0:
        raise ValueError("CTV state is shorter than its exact 7+4N body")
    return prompt


def geometry_token_family(token: str) -> int:
    text = str(token)
    for index, prefix in enumerate(CTV_GEOMETRY_PREFIXES):
        if text.startswith(prefix) and text.endswith(">"):
            return index
    raise ValueError(f"token is not a CTV geometry token: {text}")


def selected_probability_error(
    *,
    selected_token_ids: Sequence[int],
    selected_probabilities: Sequence[float],
    legal_token_ids: Sequence[int],
    legal_probabilities: Sequence[float],
) -> float:
    if len(selected_token_ids) != len(selected_probabilities):
        raise ValueError("selected CTV token/probability lengths differ")
    if len(legal_token_ids) != len(legal_probabilities):
        raise ValueError("legal CTV token/probability lengths differ")
    probability_by_token: Mapping[int, float] = {
        int(token): float(probability)
        for token, probability in zip(legal_token_ids, legal_probabilities)
    }
    errors = []
    for token, expected in zip(selected_token_ids, selected_probabilities):
        if int(token) not in probability_by_token:
            raise ValueError("selected CTV action is absent from reproduced legal support")
        errors.append(abs(probability_by_token[int(token)] - float(expected)))
    maximum = max(errors, default=0.0)
    if not math.isfinite(maximum):
        raise ValueError("CTV probability reproduction produced a non-finite error")
    return maximum


__all__ = [
    "CTV_GEOMETRY_PREFIXES",
    "exact_prompt_length",
    "geometry_token_family",
    "selected_probability_error",
]
