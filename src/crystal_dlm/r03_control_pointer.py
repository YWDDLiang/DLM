"""A head-only H1A2 controller for post-construction R03 revisions.

The controller sees a deterministic replay of the original Planner prompt and
the completed composition, never generated/teacher rich fields.  Replay is an
additional frozen-model read; it does not modify the Planner generation or its
token sequence.  The network reuses the SPAD masked-permutation architecture,
with its Typed-Planner soft-field embeddings removed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from crystal_dlm.composition_identity import canonical_symbol_counts, formula_from_symbol_counts
from crystal_dlm.fixed_slot import SYMBOL_TO_Z
from crystal_dlm.h1_llm_planner import (
    H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
    H1_PLANNER_PROMPT_STYLE_RICH_PLAN_PREFILL,
    format_planner_prompt,
)
from crystal_dlm.species_program_pointer import (
    PlanConditionedSpeciesPointer,
    SpeciesPointerConfig,
    species_pointer_loss,
)
from crystal_dlm.spad_program import program_from_element_order


P0_ADAPTER_SHA256 = "65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a"
FEATURE_SOURCE = "deterministic_formula_prefix_replay"
FEATURE_SCHEMA = "r03_control_formula_features_v1"
POINTER_STATE_SCHEMA = "r03_control_pointer_state_v1"
PROGRAM_SCHEMA = "r03_control_revision_program_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class FormulaFeatureSpec:
    prompt_style: str
    include_sample_id: bool
    feature_source: str = FEATURE_SOURCE

    def __post_init__(self) -> None:
        if self.prompt_style not in (
            H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
            H1_PLANNER_PROMPT_STYLE_RICH_PLAN_PREFILL,
        ):
            raise ValueError("R03 control features require an explicit historical rich prompt style")
        if type(self.include_sample_id) is not bool:
            raise TypeError("include_sample_id must be explicitly boolean")
        if self.feature_source != FEATURE_SOURCE:
            raise ValueError("control feature source must be deterministic formula-prefix replay")


def plan_composition(plan: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    """Return the head candidate set; never mutate or reorder the actual Plan."""
    values = canonical_symbol_counts(plan.get("elements") or (), plan.get("counts") or ())
    size = sum(count for _symbol, count in values)
    if not 1 <= size <= 20 or int(plan.get("N", size)) != size:
        raise ValueError("control Plan violates exact 1..20 atom/count conservation")
    return values


def formula_replay_text(
    tokenizer: Any,
    plan: Mapping[str, Any],
    *,
    spec: FormulaFeatureSpec,
    sample_idx: int | None = None,
    exact_prompt_text: str | None = None,
) -> str:
    """Build a control-only prefix ending at the formula, without a newline.

    The whole replay text is tokenized identically at train and export time.
    It deliberately does not claim to reproduce sampled BPE trajectories.
    ``exact_prompt_text`` can freeze the historical prompt from its runtime;
    this is supported only when the historical prompt has no sample ID.
    """
    formula = formula_from_symbol_counts(plan_composition(plan))
    if spec.include_sample_id and sample_idx is None:
        raise ValueError("the historical prompt requires sample_idx")
    if exact_prompt_text is not None:
        if spec.include_sample_id:
            raise ValueError("a constant exact prompt cannot include varying sample IDs")
        if not exact_prompt_text:
            raise ValueError("exact historical prompt must not be empty")
        prompt = exact_prompt_text
    else:
        prompt = format_planner_prompt(
            tokenizer,
            sample_idx=sample_idx if spec.include_sample_id else None,
            prompt_style=spec.prompt_style,
        )
    if spec.prompt_style == H1_PLANNER_PROMPT_STYLE_RICH_PLAN_PREFILL:
        if not prompt.endswith("formula:"):
            raise ValueError("historical formula-prefill prompt does not end with formula:")
        return prompt + " " + formula
    return prompt + "formula: " + formula


def compile_revision_program(
    plan: Mapping[str, Any],
    element_order: Sequence[str],
) -> dict[str, Any]:
    """Compile <=2 distinct species anchors, one reverse sweep, after R03."""
    program = program_from_element_order(
        plan, element_order, order_source="h1a2_frozen_formula_pointer"
    )
    selected = list(program.element_order[:2])
    # Unlike SPAD's canonical canvas, frozen R03 prefill follows the actual
    # stored Plan.elements order.  Canonicalization is only a head-candidate
    # convention; it must not silently change native anchor locations.
    native_slots = [str(symbol) for symbol, count in zip(plan["elements"], plan["counts"])
                    for _ in range(int(count))]
    revision_slots = [native_slots.index(symbol) for symbol in reversed(selected)]
    return {
        "schema": PROGRAM_SCHEMA,
        "scope": "post_construction_only",
        "initial_schedule": "unchanged_r03_safe_axis",
        "species_order": list(program.element_order),
        "selected_anchor_species": selected,
        "revision_species": list(reversed(selected)),
        "revision_slots": revision_slots,
        "slot_mapping": "original_plan_element_order_counts",
        "anchor_selection": "first_native_slot_of_each_selected_species",
        "sweeps": 1,
        "composition_mutable": False,
    }


def pointer_batch(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate only composition and optional contact labels, not rich fields."""
    import torch

    if not rows:
        raise ValueError("cannot collate empty controller rows")
    compositions = [plan_composition(row["plan_state"]) for row in rows]
    width = max(map(len, compositions))
    atomic = torch.zeros(len(rows), width, dtype=torch.long)
    counts = torch.zeros_like(atomic)
    valid = torch.zeros_like(atomic, dtype=torch.bool)
    teacher = torch.zeros_like(atomic)
    has_teacher = ["contact_tree_order_indices" in row for row in rows]
    if any(has_teacher) and not all(has_teacher):
        raise ValueError("a pointer batch cannot mix labelled and unlabelled rows")
    for index, (row, composition) in enumerate(zip(rows, compositions)):
        size = len(composition)
        atomic[index, :size] = torch.tensor([SYMBOL_TO_Z[symbol] for symbol, _ in composition])
        counts[index, :size] = torch.tensor([count for _, count in composition])
        valid[index, :size] = True
        if all(has_teacher):
            order = [int(value) for value in row["contact_tree_order_indices"]]
            if sorted(order) != list(range(size)):
                raise ValueError("contact target is not an exact candidate permutation")
            teacher[index, :size] = torch.tensor(order)
    result = {"atomic_numbers": atomic, "counts": counts, "valid_mask": valid}
    if all(has_teacher):
        result["teacher_order"] = teacher
    return result


@dataclass(frozen=True)
class R03ControlPointerConfig:
    llama_hidden_size: int
    pointer_size: int = 256
    max_elements: int = 20
    max_count: int = 20


if PlanConditionedSpeciesPointer is not None:
    import torch
    from torch import Tensor, nn

    class R03ControlPointer(PlanConditionedSpeciesPointer):
        """The existing pointer architecture without any soft-field inputs."""

        def __init__(self, config: R03ControlPointerConfig) -> None:
            super().__init__(SpeciesPointerConfig(
                llama_hidden_size=config.llama_hidden_size,
                pointer_size=config.pointer_size,
                max_elements=config.max_elements,
                max_count=config.max_count,
                num_lattice_systems=1,
                num_spacegroup_buckets=1,
                num_volume_per_atom_bins=1,
            ))
            self.soft_embeddings = nn.ModuleList()
            self.config = config

        def permutation_logits(
            self,
            formula_hidden: Tensor,
            atomic_numbers: Tensor,
            counts: Tensor,
            valid_mask: Tensor,
            *,
            teacher_order: Tensor | None = None,
        ) -> Tensor:
            if atomic_numbers.shape != counts.shape or valid_mask.shape != counts.shape:
                raise ValueError("composition tensor shapes must agree")
            if valid_mask.dtype is not torch.bool:
                raise TypeError("valid_mask must be boolean")
            if not bool(torch.isfinite(formula_hidden).all()):
                raise ValueError("formula replay hidden state must be finite")
            if bool(((atomic_numbers[~valid_mask] != 0) | (counts[~valid_mask] != 0)).any()):
                raise ValueError("padded composition entries must be zero")
            if bool((counts.sum(dim=1) > 20).any()):
                raise ValueError("composition exceeds MP20 atom count")
            for numbers, active in zip(atomic_numbers, valid_mask):
                values = numbers[active]
                if values.numel() != values.unique().numel():
                    raise ValueError("candidate species must be unique")
            if teacher_order is not None and bool(
                ((teacher_order < 0) | (teacher_order >= atomic_numbers.shape[1])).any()
            ):
                raise ValueError("teacher permutation index is outside candidate width")
            # The base implementation accepts this argument but has no soft
            # embeddings left to consume it.  No teacher field enters the head.
            unused = torch.zeros(formula_hidden.shape[0], 3, dtype=torch.long, device=formula_hidden.device)
            return super().permutation_logits(
                formula_hidden, atomic_numbers, counts, valid_mask, unused,
                teacher_order=teacher_order,
            )

        def decode(
            self,
            formula_hidden: Tensor,
            atomic_numbers: Tensor,
            counts: Tensor,
            valid_mask: Tensor,
        ) -> Tensor:
            return self.permutation_logits(
                formula_hidden, atomic_numbers, counts, valid_mask
            ).argmax(dim=-1)


    @torch.no_grad()
    def formula_hidden_batch(
        model: Any,
        tokenizer: Any,
        rows: Sequence[Mapping[str, Any]],
        *,
        spec: FormulaFeatureSpec,
        exact_prompt_text: str | None = None,
    ) -> Tensor:
        """Read frozen P0 decoder features without allocating vocabulary logits."""
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("P0 must be completely frozen before feature extraction")
        model.eval()
        texts = [formula_replay_text(
            tokenizer, row["plan_state"], spec=spec,
            sample_idx=row.get("sample_idx", row.get("source_row_idx")),
            exact_prompt_text=exact_prompt_text,
        ) for row in rows]
        encoded = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt")
        device = next(model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        position_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        decoder = getattr(base, "model", None)
        if decoder is None:
            raise TypeError("P0 must expose its Llama decoder as model")
        output = decoder(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state
        positions = torch.arange(mask.shape[1], device=device).unsqueeze(0).expand_as(mask)
        terminal = positions.masked_fill(mask == 0, -1).max(dim=1).values
        if bool((terminal < 0).any()):
            raise ValueError("empty formula replay input")
        features = hidden[torch.arange(hidden.shape[0], device=device), terminal]
        return features.detach().float().cpu().clone()

else:  # pragma: no cover - pure metadata use on hosts without torch
    R03ControlPointer = None
    formula_hidden_batch = None


__all__ = [
    "FEATURE_SCHEMA", "FEATURE_SOURCE", "P0_ADAPTER_SHA256", "POINTER_STATE_SCHEMA",
    "PROGRAM_SCHEMA", "FormulaFeatureSpec", "R03ControlPointer", "R03ControlPointerConfig",
    "compile_revision_program", "formula_hidden_batch", "formula_replay_text",
    "plan_composition", "pointer_batch", "sha256_file", "species_pointer_loss", "stable_digest",
]
