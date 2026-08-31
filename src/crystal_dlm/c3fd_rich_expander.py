"""C3FD-conditioned rich-Plan expansion for the H1-A2 Llama Planner.

Route F conditions on an immutable C3FD formula only. Route M keeps the same
visible prompt and target, while a small trainable projector converts frozen
C3FD semantic features into soft prefix embeddings for the Llama backbone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from crystal_dlm.h1_llm_planner import H1_PLANNER_SYSTEM_PROMPT
from crystal_dlm.r5_plan_body import (
    H1_RICH_PLAN_FORMAT,
    composition_plan_from_state,
    format_composition_plan,
    parse_composition_plan,
)


C3FD_RICH_EXPANDER_VERSION = "c3fd_rich_expander_v1"
C3FD_SOFT_PREFIX_FEATURE_VERSION = "c3fd_soft_prefix_features_v1"
ROUTE_FORMULA = "F"
ROUTE_SOFT_PREFIX = "M"
ROUTES = (ROUTE_FORMULA, ROUTE_SOFT_PREFIX)
MAX_SPECIES = 7
CHECKPOINT_ORDER = ("seed17", "seed18")
PREDICTED_FIELDS = (
    "lattice_system",
    "spacegroup_bucket",
    "volume_per_atom_bin",
)
RICH_SUFFIX_LABELS = (
    "anion",
    "charge",
    "lattice",
    "spacegroup",
    "volume",
    "end",
)
HEADER_FEATURES = 8
SLOT_FEATURES = 4
PREDICTION_FEATURES = 3
FEATURE_DIM = (
    HEADER_FEATURES
    + MAX_SPECIES * SLOT_FEATURES
    + len(CHECKPOINT_ORDER) * len(PREDICTED_FIELDS) * PREDICTION_FEATURES
)


def rich_suffix_from_plan_state(plan_state: Mapping[str, Any]) -> str:
    """Return the six generated lines after the immutable formula line."""

    lines = format_composition_plan(
        plan_state, plan_style=H1_RICH_PLAN_FORMAT
    ).splitlines()
    if len(lines) != 7 or not lines[0].startswith("formula: "):
        raise ValueError("historical rich Plan serialization changed")
    return "\n".join(lines[1:])


def expander_messages(plan_state: Mapping[str, Any]) -> list[dict[str, str]]:
    plan = composition_plan_from_state(plan_state)
    user = (
        "Expand the fixed C3FD composition into the remaining H1-A2 rich Plan "
        "fields. The assistant formula line is already fixed and must not be "
        "changed. Return exactly six additional lines: anion, charge, lattice, "
        "spacegroup, volume, and end: plan. Use only the allowed historical "
        "H1-A2 categorical values and add no prose, coordinates, candidates, "
        "or markdown.\n\n"
        f"fixed_formula: {plan['formula']}\n"
        f"fixed_N: {plan['N']}"
    )
    return [
        {"role": "system", "content": H1_PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_expander_prompt(tokenizer: Any, plan_state: Mapping[str, Any]) -> str:
    """Build one train/serve-identical prompt with an assistant formula prefill."""

    messages = expander_messages(plan_state)
    if hasattr(tokenizer, "apply_chat_template") and getattr(
        tokenizer, "chat_template", None
    ):
        prompt = str(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    else:
        prompt = (
            f"System: {messages[0]['content']}\n\n"
            f"User: {messages[1]['content']}\n\nAssistant:"
        )
    plan = composition_plan_from_state(plan_state)
    return prompt.rstrip() + f" formula: {plan['formula']}\n"


def assemble_expanded_plan(
    plan_state: Mapping[str, Any], generated_suffix: str
) -> dict[str, Any]:
    """Strictly parse a generated suffix without repairing or filling fields."""

    plan = composition_plan_from_state(plan_state)
    suffix = validate_generated_suffix_shape(generated_suffix)
    full_text = f"formula: {plan['formula']}\n{suffix}"
    parsed = parse_composition_plan(
        full_text, plan_style=H1_RICH_PLAN_FORMAT, max_atoms=20
    )
    observed = composition_plan_from_state(parsed)
    if observed != plan:
        raise ValueError("rich Expander changed the immutable C3FD composition")
    return {
        "plan_text": format_composition_plan(
            parsed, plan_style=H1_RICH_PLAN_FORMAT
        ),
        "plan_state": parsed,
    }


def validate_generated_suffix_shape(generated_suffix: str) -> str:
    """Require exactly the six historical rich suffix lines in fixed order."""

    lines = str(generated_suffix).strip().splitlines()
    if len(lines) != len(RICH_SUFFIX_LABELS):
        raise ValueError("rich Expander suffix must contain exactly six lines")
    for expected, line in zip(RICH_SUFFIX_LABELS, lines):
        label, separator, value = line.partition(":")
        if not separator or label.strip().lower() != expected or not value.strip():
            raise ValueError(f"rich Expander suffix line is not {expected}: <value>")
    if lines[-1].strip().lower() != "end: plan":
        raise ValueError("rich Expander suffix must terminate with end: plan")
    return "\n".join(line.strip() for line in lines)


def _species_lookup(vocabulary: Mapping[str, Any]) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for row in vocabulary.get("species") or ():
        result[int(row["id"])] = (
            int(row["atomic_number"]),
            int(row["oxidation_state"]),
        )
    return result


def _prediction_values(
    predicted_row: Mapping[str, Any] | None,
    checkpoint: str,
    field: str,
) -> tuple[str | None, float | None]:
    if predicted_row is None:
        return None, None
    by_checkpoint = predicted_row.get("predictions_by_checkpoint") or {}
    record = (by_checkpoint.get(checkpoint) or {}).get(field) or {}
    prediction = record.get("prediction")
    confidence = record.get("confidence")
    return (
        None if prediction is None else str(prediction),
        None if confidence is None else float(confidence),
    )


def pack_soft_prefix_features(
    semantic_row: Mapping[str, Any],
    vocabulary: Mapping[str, Any],
    predicted_row: Mapping[str, Any] | None = None,
) -> list[float]:
    """Pack only frozen C3FD state/predictions into one fixed-size vector."""

    proposal = semantic_row.get("proposal_targets") or {}
    species = [int(value) for value in semantic_row.get("species_labels") or ()]
    counts = [int(value) for value in semantic_row.get("count_targets") or ()]
    composition_supervision = semantic_row.get("composition_supervision") is True
    if len(species) != len(counts):
        if composition_supervision:
            raise ValueError("C3FD semantic species/count trace is malformed")
        species = []
        counts = []
    if len(species) > MAX_SPECIES:
        raise ValueError("C3FD semantic species trace exceeds MAX_SPECIES")
    lookup = _species_lookup(vocabulary)
    ledger = list(semantic_row.get("ledger_steps") or ())
    terminal = ledger[-1] if ledger else {}
    n_value = int(proposal.get("N") or semantic_row.get("N_target") or 0)
    arity = int(proposal.get("arity") or len(species))
    family = int(proposal.get("family") or 0)
    certificate = str(semantic_row.get("certificate_class") or "")
    branch = str(terminal.get("branch") or "unset")
    branch_code = {"unset": 0.0, "ionic": 0.5, "alloy": 1.0}.get(
        branch, 0.0
    )
    features: list[float] = [
        n_value / 20.0,
        arity / float(MAX_SPECIES),
        1.0 if composition_supervision else 0.0,
        1.0 if semantic_row.get("proposal_supervision") is True else 0.0,
        1.0 if certificate == "benchmark_compatible" else 0.0,
        family / 6.0,
        float(terminal.get("net_charge") or 0) / 20.0,
        branch_code,
    ]
    for slot in range(MAX_SPECIES):
        if slot < len(species):
            species_id = species[slot]
            if species_id not in lookup:
                raise ValueError(f"unknown C3FD species id {species_id}")
            atomic_number, oxidation = lookup[species_id]
            features.extend(
                [
                    1.0,
                    atomic_number / 118.0,
                    max(-1.0, min(1.0, oxidation / 8.0)),
                    counts[slot] / 20.0,
                ]
            )
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
    soft_vocabulary = vocabulary.get("soft_vocabulary") or {}
    for checkpoint in CHECKPOINT_ORDER:
        for field in PREDICTED_FIELDS:
            prediction, confidence = _prediction_values(
                predicted_row, checkpoint, field
            )
            choices = [str(value) for value in soft_vocabulary.get(field) or ()]
            if prediction is None or prediction not in choices or not choices:
                features.extend([0.0, 0.0, 0.0])
            else:
                denominator = max(1, len(choices) - 1)
                features.extend(
                    [
                        1.0,
                        choices.index(prediction) / float(denominator),
                        max(0.0, min(1.0, float(confidence or 0.0))),
                    ]
                )
    if len(features) != FEATURE_DIM:
        raise AssertionError(f"soft-prefix feature width changed: {len(features)}")
    return features


@dataclass(frozen=True)
class SoftPrefixProjectorConfig:
    input_dim: int = FEATURE_DIM
    prefix_length: int = 4
    model_hidden_dim: int = 4096
    projector_hidden_dim: int = 1024
    version: str = C3FD_SOFT_PREFIX_FEATURE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


try:
    import torch
    from torch import nn

    class SoftPrefixProjector(nn.Module):
        """Map frozen C3FD features to trainable Llama prefix embeddings."""

        def __init__(self, config: SoftPrefixProjectorConfig) -> None:
            super().__init__()
            self.config = config
            self.network = nn.Sequential(
                nn.LayerNorm(int(config.input_dim)),
                nn.Linear(
                    int(config.input_dim), int(config.projector_hidden_dim)
                ),
                nn.GELU(),
                nn.Linear(
                    int(config.projector_hidden_dim),
                    int(config.prefix_length) * int(config.model_hidden_dim),
                ),
            )

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            if features.ndim != 2 or features.shape[-1] != int(
                self.config.input_dim
            ):
                raise ValueError("soft-prefix features have the wrong shape")
            projected = self.network(features)
            return projected.view(
                features.shape[0],
                int(self.config.prefix_length),
                int(self.config.model_hidden_dim),
            )

except ImportError:  # pragma: no cover - pure helpers remain importable.
    SoftPrefixProjector = None  # type: ignore[assignment,misc]


__all__ = [
    "C3FD_RICH_EXPANDER_VERSION",
    "C3FD_SOFT_PREFIX_FEATURE_VERSION",
    "CHECKPOINT_ORDER",
    "FEATURE_DIM",
    "PREDICTED_FIELDS",
    "ROUTE_FORMULA",
    "ROUTE_SOFT_PREFIX",
    "ROUTES",
    "SoftPrefixProjector",
    "SoftPrefixProjectorConfig",
    "assemble_expanded_plan",
    "build_expander_prompt",
    "expander_messages",
    "pack_soft_prefix_features",
    "rich_suffix_from_plan_state",
    "validate_generated_suffix_shape",
]
