"""Fail-closed token preparation for H1-A2 P-control/P* training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from crystal_dlm.h1a2_planner_objective import (
    encode_field_group_ids,
    encode_lookahead_labels,
    parse_seven_line_plan,
    token_fields_from_offsets,
)


@dataclass(frozen=True)
class PreparedPlannerExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    field_group_ids: tuple[int, ...]
    formula_boundary: int
    lattice_boundary: int
    lookahead_labels: dict[str, int]
    prompt_tokens: int
    answer_tokens: int
    source_line_sha256: str | None


def format_training_prompt(tokenizer: Any, row: Mapping[str, Any]) -> str:
    prompt = row.get("prompt")
    if prompt is not None and str(prompt):
        return str(prompt)
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Planner row has neither a persisted prompt nor messages")
    if (
        hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
    ):
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return (
        f"System: {messages[0]['content']}\n\n"
        f"User: {messages[1]['content']}\n\n"
        "Assistant:"
    )


def _input_ids(encoded: Any) -> tuple[int, ...]:
    try:
        values = encoded["input_ids"]
    except (KeyError, TypeError) as exc:
        raise ValueError("tokenizer output has no input_ids") from exc
    if values and isinstance(values[0], Sequence):
        raise ValueError("batched tokenizer output is not allowed for one row")
    return tuple(int(value) for value in values)


def prepare_planner_example(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
    lookahead_vocabs: Mapping[str, Sequence[str]],
) -> PreparedPlannerExample:
    """Prepare one row while preserving the historical additive tokenization."""

    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ValueError("a fast tokenizer with exact offset mapping is required")
    if "answer" not in row:
        raise ValueError("Planner row has no answer")
    answer = str(row["answer"])
    if answer != answer.strip():
        raise ValueError("Planner answer has leading or trailing whitespace")
    parse_seven_line_plan(answer)
    prompt = format_training_prompt(tokenizer, row)

    prompt_ids = _input_ids(
        tokenizer(prompt, add_special_tokens=False)
    )
    answer_encoding = tokenizer(
        answer,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    answer_ids_without_eos = _input_ids(answer_encoding)
    offsets = answer_encoding.get("offset_mapping")
    if offsets is None or len(offsets) != len(answer_ids_without_eos):
        raise ValueError("answer tokenizer offsets are absent or misaligned")
    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not eos_token or eos_token_id is None:
        raise ValueError("Planner tokenizer has no EOS token identity")
    registered_answer_ids = answer_ids_without_eos + (int(eos_token_id),)
    historical_answer_ids = _input_ids(
        tokenizer(answer + str(eos_token), add_special_tokens=False)
    )
    if historical_answer_ids != registered_answer_ids:
        raise ValueError(
            "answer+EOS tokenization differs from the historical additive contract"
        )

    answer_fields = token_fields_from_offsets(
        answer,
        offsets,
        allow_terminal_special_tokens=False,
    ) + ("end",)
    if len(answer_fields) != len(registered_answer_ids):
        raise AssertionError("answer field accounting changed")
    answer_group_ids = encode_field_group_ids(answer_fields)

    maximum = int(max_length)
    if maximum < 1 or len(registered_answer_ids) >= maximum:
        raise ValueError(
            f"answer uses {len(registered_answer_ids)} tokens and does not fit "
            f"max_length={maximum}"
        )
    max_prompt_tokens = maximum - len(registered_answer_ids)
    if len(prompt_ids) > max_prompt_tokens:
        prompt_ids = prompt_ids[-max_prompt_tokens:]
    prompt_length = len(prompt_ids)
    input_ids = prompt_ids + registered_answer_ids
    labels = (-100,) * prompt_length + registered_answer_ids
    group_ids = (-1,) * prompt_length + answer_group_ids

    def boundary_position(field: str) -> int:
        relative = [
            index for index, observed in enumerate(answer_fields) if observed == field
        ]
        if not relative:
            raise ValueError(f"no answer token belongs to boundary {field!r}")
        return prompt_length + relative[-1]

    stream = row.get("v3_planner_stream")
    source_sha = None
    if isinstance(stream, Mapping):
        value = stream.get("source_line_sha256")
        source_sha = None if value is None else str(value)
    return PreparedPlannerExample(
        input_ids=input_ids,
        labels=labels,
        field_group_ids=group_ids,
        formula_boundary=boundary_position("formula"),
        lattice_boundary=boundary_position("lattice"),
        lookahead_labels=encode_lookahead_labels(answer, lookahead_vocabs),
        prompt_tokens=prompt_length,
        answer_tokens=len(registered_answer_ids),
        source_line_sha256=source_sha,
    )


__all__ = [
    "PreparedPlannerExample",
    "format_training_prompt",
    "prepare_planner_example",
]
