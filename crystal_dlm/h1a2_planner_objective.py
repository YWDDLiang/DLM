"""Fail-closed field accounting for the H1-A2 seven-line Planner objective."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


PLAN_FIELDS = (
    "formula",
    "anion",
    "charge",
    "lattice",
    "spacegroup",
    "volume",
    "end",
)
LOOKAHEAD_FIELDS = ("anion", "charge", "lattice", "spacegroup", "volume")
FIELD_GROUPS = {
    "formula": "formula",
    "anion": "chemistry",
    "charge": "chemistry",
    "lattice": "geometry",
    "spacegroup": "geometry",
    "volume": "geometry",
    "end": "terminator",
}
FIELD_GROUP_WEIGHTS = {
    "formula": 0.35,
    "chemistry": 0.25,
    "geometry": 0.35,
    "terminator": 0.05,
}
FIELD_GROUP_NAMES = tuple(FIELD_GROUP_WEIGHTS)
FIELD_GROUP_IDS = {
    group: index for index, group in enumerate(FIELD_GROUP_NAMES)
}
LOOKAHEAD_HEAD_SPECS = (
    ("formula", "anion"),
    ("formula", "charge"),
    ("formula", "lattice"),
    ("formula", "spacegroup"),
    ("formula", "volume"),
    ("lattice", "spacegroup"),
    ("lattice", "volume"),
)


@dataclass(frozen=True)
class PlanFieldSpan:
    field: str
    value: str
    start: int
    end: int


def parse_seven_line_plan(answer: str) -> tuple[PlanFieldSpan, ...]:
    """Parse only the exact H1-A2 visible schema and retain character spans."""

    text = str(answer).replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines(keepends=True)
    if len(lines) != len(PLAN_FIELDS):
        raise ValueError(
            f"expected exactly {len(PLAN_FIELDS)} Plan lines, observed {len(lines)}"
        )
    spans: list[PlanFieldSpan] = []
    cursor = 0
    for expected_field, line_with_end in zip(PLAN_FIELDS, lines, strict=True):
        line = line_with_end[:-1] if line_with_end.endswith("\n") else line_with_end
        prefix = f"{expected_field}:"
        if not line.startswith(prefix):
            raise ValueError(
                f"expected line {expected_field!r}, observed {line!r}"
            )
        value = line[len(prefix) :].strip()
        if not value:
            raise ValueError(f"field {expected_field!r} has an empty value")
        if expected_field == "end" and value != "plan":
            raise ValueError(f"end field must equal 'plan', observed {value!r}")
        spans.append(
            PlanFieldSpan(
                field=expected_field,
                value=value,
                start=cursor,
                end=cursor + len(line_with_end),
            )
        )
        cursor += len(line_with_end)
    if cursor != len(text):
        raise AssertionError("seven-line span accounting changed")
    return tuple(spans)


def plan_values(answer: str) -> dict[str, str]:
    return {span.field: span.value for span in parse_seven_line_plan(answer)}


def build_lookahead_vocabs(answers: Iterable[str]) -> dict[str, tuple[str, ...]]:
    values = {field: set() for field in LOOKAHEAD_FIELDS}
    count = 0
    for answer in answers:
        count += 1
        parsed = plan_values(answer)
        for field in LOOKAHEAD_FIELDS:
            values[field].add(parsed[field])
    if count == 0:
        raise ValueError("cannot build look-ahead vocabularies from zero Plans")
    output = {field: tuple(sorted(values[field])) for field in LOOKAHEAD_FIELDS}
    if any(not labels for labels in output.values()):
        raise ValueError("one or more look-ahead vocabularies are empty")
    return output


def encode_lookahead_labels(
    answer: str,
    vocabs: Mapping[str, Sequence[str]],
) -> dict[str, int]:
    parsed = plan_values(answer)
    encoded: dict[str, int] = {}
    for field in LOOKAHEAD_FIELDS:
        labels = tuple(str(value) for value in vocabs.get(field, ()))
        if len(labels) != len(set(labels)) or labels != tuple(sorted(labels)):
            raise ValueError(f"vocabulary for {field!r} is not unique and sorted")
        try:
            encoded[field] = labels.index(parsed[field])
        except ValueError as exc:
            raise ValueError(
                f"unknown {field!r} label {parsed[field]!r}"
            ) from exc
    return encoded


def token_fields_from_offsets(
    answer: str,
    offsets: Sequence[Sequence[int]],
    *,
    allow_terminal_special_tokens: bool = True,
) -> tuple[str, ...]:
    """Map fast-tokenizer offsets to fields without retokenizing substrings."""

    spans = parse_seven_line_plan(answer)
    text = str(answer).replace("\r\n", "\n").replace("\r", "\n")
    assigned: list[str] = []
    saw_special = False
    for token_index, raw_offset in enumerate(offsets):
        if len(raw_offset) != 2:
            raise ValueError(f"token offset {token_index} is not a pair")
        start, end = (int(raw_offset[0]), int(raw_offset[1]))
        if start == end == 0:
            if not allow_terminal_special_tokens:
                raise ValueError("zero-length special token has no registered field")
            saw_special = True
            assigned.append("end")
            continue
        if saw_special:
            raise ValueError("non-special token appears after a terminal special token")
        if start < 0 or end <= start or end > len(text):
            raise ValueError(f"invalid token offset {(start, end)}")
        token_text = text[start:end]
        scores: list[tuple[int, int, str]] = []
        for span_index, span in enumerate(spans):
            overlap_start = max(start, span.start)
            overlap_end = min(end, span.end)
            if overlap_end <= overlap_start:
                continue
            overlap = text[overlap_start:overlap_end]
            nonspace = sum(1 for char in overlap if not char.isspace())
            scores.append((nonspace, -span_index, span.field))
        if not scores:
            raise ValueError(f"token offset {(start, end)} is outside all Plan lines")
        best_nonspace = max(score[0] for score in scores)
        candidates = [score for score in scores if score[0] == best_nonspace]
        if best_nonspace == 0:
            # A newline-only token belongs to the preceding visible line.
            candidates.sort(key=lambda score: score[1], reverse=True)
            assigned.append(candidates[0][2])
            continue
        winners = {score[2] for score in candidates}
        if len(winners) != 1:
            raise ValueError(
                f"token {token_text!r} overlaps multiple Plan fields equally: "
                f"{sorted(winners)}"
            )
        assigned.append(next(iter(winners)))
    return tuple(assigned)


def field_balanced_mean(
    token_losses: Sequence[float],
    token_fields: Sequence[str],
) -> tuple[float, dict[str, float]]:
    """Reference scalar implementation used by CPU fixtures."""

    if len(token_losses) != len(token_fields):
        raise ValueError("token_losses and token_fields must have identical length")
    grouped: dict[str, list[float]] = {
        group: [] for group in FIELD_GROUP_WEIGHTS
    }
    for loss, field in zip(token_losses, token_fields, strict=True):
        if field not in FIELD_GROUPS:
            raise ValueError(f"unknown Plan field {field!r}")
        grouped[FIELD_GROUPS[field]].append(float(loss))
    missing = [group for group, losses in grouped.items() if not losses]
    if missing:
        raise ValueError(f"missing supervised field groups: {missing}")
    means = {
        group: sum(losses) / len(losses) for group, losses in grouped.items()
    }
    total = sum(FIELD_GROUP_WEIGHTS[group] * means[group] for group in means)
    return total, means


def encode_field_group_ids(token_fields: Sequence[str]) -> tuple[int, ...]:
    encoded: list[int] = []
    for field in token_fields:
        if field not in FIELD_GROUPS:
            raise ValueError(f"unknown Plan field {field!r}")
        encoded.append(FIELD_GROUP_IDS[FIELD_GROUPS[field]])
    return tuple(encoded)


def torch_field_balanced_loss(
    token_losses: Any,
    field_group_ids: Any,
    supervision_mask: Any,
    *,
    sample_weights: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Differentiable counterpart of :func:`field_balanced_mean`.

    Torch is imported lazily so the schema/data utilities remain usable in
    CPU-only planning environments that do not install the training stack.
    All four registered groups must occur in every sample; prompt positions
    use group id ``-1`` and an unset supervision mask.
    """

    import torch

    if token_losses.ndim != 2:
        raise ValueError("token_losses must have shape [batch, tokens]")
    if field_group_ids.shape != token_losses.shape:
        raise ValueError("field_group_ids shape does not match token_losses")
    if supervision_mask.shape != token_losses.shape:
        raise ValueError("supervision_mask shape does not match token_losses")
    mask = supervision_mask.to(device=token_losses.device, dtype=torch.bool)
    groups = field_group_ids.to(device=token_losses.device, dtype=torch.long)
    batch_size = int(token_losses.shape[0])
    per_sample_totals: list[Any] = []
    per_group_means: dict[str, list[Any]] = {
        group: [] for group in FIELD_GROUP_NAMES
    }
    for batch_index in range(batch_size):
        sample_total = token_losses.new_zeros(())
        for group in FIELD_GROUP_NAMES:
            group_id = FIELD_GROUP_IDS[group]
            group_mask = mask[batch_index] & (groups[batch_index] == group_id)
            if not bool(torch.any(group_mask).item()):
                raise ValueError(
                    f"sample {batch_index} has no supervised tokens for {group!r}"
                )
            group_mean = token_losses[batch_index][group_mask].mean()
            per_group_means[group].append(group_mean)
            sample_total = (
                sample_total + float(FIELD_GROUP_WEIGHTS[group]) * group_mean
            )
        per_sample_totals.append(sample_total)
    stacked = torch.stack(per_sample_totals)
    if sample_weights is None:
        weights = torch.ones_like(stacked)
    else:
        if sample_weights.ndim != 1 or int(sample_weights.shape[0]) != batch_size:
            raise ValueError("sample_weights must have shape [batch]")
        weights = sample_weights.to(device=stacked.device, dtype=stacked.dtype)
        if bool(torch.any(weights <= 0).item()):
            raise ValueError("sample_weights must be finite and positive")
        if not bool(torch.all(torch.isfinite(weights)).item()):
            raise ValueError("sample_weights must be finite and positive")
    total = (stacked * weights).sum() / weights.sum()
    diagnostics = {
        group: torch.stack(values).mean()
        for group, values in per_group_means.items()
    }
    diagnostics["field_balanced"] = total
    return total, diagnostics


def lookahead_head_name(boundary: str, target: str) -> str:
    spec = (str(boundary), str(target))
    if spec not in LOOKAHEAD_HEAD_SPECS:
        raise ValueError(f"unregistered look-ahead head {spec!r}")
    return f"{spec[0]}__{spec[1]}"


def build_lookahead_heads(
    hidden_size: int,
    vocab_sizes: Mapping[str, int],
    *,
    seed: int = 17,
) -> Any:
    """Create the seven registered affine heads with deterministic init."""

    import torch

    hidden = int(hidden_size)
    if hidden < 1:
        raise ValueError("hidden_size must be positive")
    normalized: dict[str, int] = {}
    for field in LOOKAHEAD_FIELDS:
        size = int(vocab_sizes.get(field, 0))
        if size < 1:
            raise ValueError(f"look-ahead vocabulary {field!r} is empty")
        normalized[field] = size
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        heads = torch.nn.ModuleDict(
            {
                lookahead_head_name(boundary, target): torch.nn.Linear(
                    hidden,
                    normalized[target],
                )
                for boundary, target in LOOKAHEAD_HEAD_SPECS
            }
        )
    return heads


def torch_lookahead_loss(
    hidden_states: Any,
    boundary_positions: Mapping[str, Any],
    labels: Mapping[str, Any],
    heads: Any,
    *,
    sample_weights: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Average the seven registered boundary-to-field cross-entropies."""

    import torch

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, tokens, hidden]")
    batch_size, token_count, _ = (int(value) for value in hidden_states.shape)
    if batch_size < 1 or token_count < 1:
        raise ValueError("hidden_states cannot be empty")
    if sample_weights is None:
        weights = hidden_states.new_ones((batch_size,), dtype=torch.float32)
    else:
        if sample_weights.ndim != 1 or int(sample_weights.shape[0]) != batch_size:
            raise ValueError("sample_weights must have shape [batch]")
        weights = sample_weights.to(
            device=hidden_states.device,
            dtype=torch.float32,
        )
        if bool(torch.any(weights <= 0).item()):
            raise ValueError("sample_weights must be finite and positive")
        if not bool(torch.all(torch.isfinite(weights)).item()):
            raise ValueError("sample_weights must be finite and positive")
    row_indices = torch.arange(batch_size, device=hidden_states.device)
    head_losses: list[Any] = []
    diagnostics: dict[str, Any] = {}
    for boundary, target in LOOKAHEAD_HEAD_SPECS:
        if boundary not in boundary_positions:
            raise ValueError(f"missing boundary positions for {boundary!r}")
        if target not in labels:
            raise ValueError(f"missing look-ahead labels for {target!r}")
        positions = boundary_positions[boundary].to(
            device=hidden_states.device,
            dtype=torch.long,
        )
        targets = labels[target].to(
            device=hidden_states.device,
            dtype=torch.long,
        )
        if positions.ndim != 1 or int(positions.shape[0]) != batch_size:
            raise ValueError(f"{boundary!r} positions must have shape [batch]")
        if targets.ndim != 1 or int(targets.shape[0]) != batch_size:
            raise ValueError(f"{target!r} labels must have shape [batch]")
        if bool(torch.any((positions < 0) | (positions >= token_count)).item()):
            raise ValueError(f"{boundary!r} positions are outside hidden states")
        head_name = lookahead_head_name(boundary, target)
        if head_name not in heads:
            raise ValueError(f"missing registered head {head_name!r}")
        head = heads[head_name]
        head_dtype = next(head.parameters()).dtype
        features = hidden_states[row_indices, positions].to(dtype=head_dtype)
        logits = head(features)
        per_sample = torch.nn.functional.cross_entropy(
            logits.float(),
            targets,
            reduction="none",
        )
        head_loss = (per_sample * weights).sum() / weights.sum()
        head_losses.append(head_loss)
        diagnostics[f"{head_name}_loss"] = head_loss
        diagnostics[f"{head_name}_accuracy"] = (
            (logits.argmax(dim=-1) == targets).to(torch.float32) * weights
        ).sum() / weights.sum()
    total = torch.stack(head_losses).mean()
    diagnostics["lookahead"] = total
    return total, diagnostics


def validate_objective_constants() -> None:
    if set(FIELD_GROUPS) != set(PLAN_FIELDS):
        raise AssertionError("field-to-group mapping is incomplete")
    if abs(sum(FIELD_GROUP_WEIGHTS.values()) - 1.0) > 1e-12:
        raise AssertionError("field group weights must sum to one")
    if {target for _, target in LOOKAHEAD_HEAD_SPECS} != set(LOOKAHEAD_FIELDS):
        raise AssertionError("look-ahead head coverage changed")


validate_objective_constants()


__all__ = [
    "FIELD_GROUPS",
    "FIELD_GROUP_IDS",
    "FIELD_GROUP_NAMES",
    "FIELD_GROUP_WEIGHTS",
    "LOOKAHEAD_HEAD_SPECS",
    "LOOKAHEAD_FIELDS",
    "PLAN_FIELDS",
    "PlanFieldSpan",
    "build_lookahead_vocabs",
    "build_lookahead_heads",
    "encode_field_group_ids",
    "encode_lookahead_labels",
    "field_balanced_mean",
    "parse_seven_line_plan",
    "plan_values",
    "lookahead_head_name",
    "torch_field_balanced_loss",
    "torch_lookahead_loss",
    "token_fields_from_offsets",
    "validate_objective_constants",
]
