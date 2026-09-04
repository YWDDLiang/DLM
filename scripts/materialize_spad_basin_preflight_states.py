#!/usr/bin/env python3
"""Materialize deterministic, outcome-blind SPAD basin-closure states from logs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


ROW_COUNT = 128
DEFAULT_EXPECTED_GROUPS = ROW_COUNT
MIN_HIGH_N_PER_TYPE = 12
MIN_HIGH_MULTIPLICITY_PER_TYPE = 16
STATE_SCHEMA = "spad_basin_preflight_state_v1"
CURSOR_BUCKETS = ("early", "middle", "late", "terminal")
MASK_DEFAULT = "<|mdm_mask|>"
TOKEN_RE = re.compile(
    r"<(?:N_\d{3}|L[ABC]_\d{3}|A[ABG]_\d{3}|S\d{2}|"
    r"E_[A-Z][a-z]?|[XYZ]_\d{3}|EMPTY|[XYZ]_PAD)>"
)
NUMERIC_TOKEN_RE = re.compile(
    r"^<(N|LA|LB|LC|AA|AB|AG|X|Y|Z)_(\d{3})>$"
)
FAMILY_RANGES = {
    "N": (1, 20),
    "LA": (0, 500),
    "LB": (0, 500),
    "LC": (0, 500),
    "AA": (1, 179),
    "AB": (1, 179),
    "AG": (1, 179),
    "X": (0, 100),
    "Y": (0, 100),
    "Z": (0, 100),
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            yield value


def _int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"{name} must be a list of integers")
    return [int(item) for item in value]


def _body_text(tokens: Sequence[str]) -> str:
    return "".join(tokens)


def _expected_family(position: int) -> str:
    if position == 0:
        return "N"
    if 1 <= position <= 6:
        return ("LA", "LB", "LC", "AA", "AB", "AG")[position - 1]
    remainder = (position - 7) % 4
    return ("E", "X", "Y", "Z")[remainder]


def _token_family(token: str) -> str:
    match = NUMERIC_TOKEN_RE.fullmatch(token)
    if match:
        return match.group(1)
    if re.fullmatch(r"<E_[A-Z][a-z]?>", token):
        return "E"
    if re.fullmatch(r"<S\d{2}>", token):
        return "S"
    if token in {"<EMPTY>", "<X_PAD>", "<Y_PAD>", "<Z_PAD>"}:
        return "PAD"
    raise ValueError(f"not a dynamic crystal token: {token!r}")


class TokenResolver:
    """Resolve logged action IDs without loading a tokenizer or model."""

    def __init__(self, vocabulary: Mapping[str, int] | None = None) -> None:
        self._explicit_vocab = vocabulary is not None
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.family_bases: dict[str, int] = {}
        if vocabulary is not None:
            for token, token_id in vocabulary.items():
                if not isinstance(token, str) or isinstance(token_id, bool) or not isinstance(
                    token_id, int
                ):
                    raise ValueError("tokenizer vocabulary must map strings to integer IDs")
                if token_id in self.id_to_token and self.id_to_token[token_id] != token:
                    raise ValueError(f"tokenizer vocabulary repeats ID {token_id}")
                self.token_to_id[token] = int(token_id)
                self.id_to_token[int(token_id)] = token

    def observe(self, token: str, token_id: int, *, position: int) -> None:
        expected = _expected_family(position)
        if _token_family(token) != expected:
            raise ValueError(
                f"position {position} expected {expected}, observed token {token!r}"
            )
        known_token = self.id_to_token.get(int(token_id))
        if known_token is not None and known_token != token:
            raise ValueError(
                f"token ID {token_id} maps to both {known_token!r} and {token!r}"
            )
        known_id = self.token_to_id.get(token)
        if known_id is not None and known_id != int(token_id):
            raise ValueError(f"token {token!r} maps to both {known_id} and {token_id}")
        self.id_to_token[int(token_id)] = token
        self.token_to_id[token] = int(token_id)
        if self._explicit_vocab:
            return
        numeric = NUMERIC_TOKEN_RE.fullmatch(token)
        if numeric:
            family, value = numeric.group(1), int(numeric.group(2))
            base = int(token_id) - value
            prior = self.family_bases.get(family)
            if prior is not None and prior != base:
                raise ValueError(
                    f"non-affine {family} token IDs; provide --tokenizer-vocab-json"
                )
            self.family_bases[family] = base

    def resolve(self, token_id: int, *, position: int) -> str:
        expected = _expected_family(position)
        known = self.id_to_token.get(int(token_id))
        if known is not None:
            if _token_family(known) != expected:
                raise ValueError(
                    f"ID {token_id} resolves to {known!r}, not {expected} at {position}"
                )
            return known
        base = self.family_bases.get(expected)
        if base is None or expected not in FAMILY_RANGES:
            raise ValueError(
                f"cannot resolve token ID {token_id} at position {position}; "
                "provide --tokenizer-vocab-json"
            )
        value = int(token_id) - base
        low, high = FAMILY_RANGES[expected]
        if not low <= value <= high:
            raise ValueError(
                f"token ID {token_id} is outside inferred {expected} range at {position}"
            )
        token = f"<{expected}_{value:03d}>"
        self.observe(token, int(token_id), position=position)
        return token


def load_vocabulary(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("tokenizer vocabulary JSON must be an object")
    return {str(token): int(token_id) for token, token_id in value.items()}


def _plan_state(row: Mapping[str, Any]) -> Mapping[str, Any]:
    plan = row.get("plan_state")
    if not isinstance(plan, Mapping):
        raise ValueError("plan row lacks plan_state")
    return plan


def _validate_plan_row(row: Mapping[str, Any]) -> None:
    if row.get("outcomes_read") is not False:
        raise ValueError("plan source is not explicitly outcome-blind")
    split = row.get("source_split", row.get("split"))
    if split is not None and str(split).lower() not in {
        "train",
        "mp20_train_only",
        "mp20-train-only",
    }:
        raise ValueError(f"preflight source is not MP20 train: {split!r}")
    plan = _plan_state(row)
    elements = [str(value) for value in plan.get("elements") or ()]
    counts = [int(value) for value in plan.get("counts") or ()]
    n = int(plan.get("N") or 0)
    if (
        not 1 <= n <= 20
        or not elements
        or len(elements) != len(counts)
        or len(elements) != len(set(elements))
        or any(value <= 0 for value in counts)
        or sum(counts) != n
    ):
        raise ValueError("plan composition is malformed")
    program = [str(value) for value in row.get("species_program") or ()]
    if len(program) != len(elements) or set(program) != set(elements):
        raise ValueError("species_program does not permute Plan elements")


def _validate_body(tokens: Sequence[str], plan: Mapping[str, Any]) -> None:
    n = int(plan["N"])
    if len(tokens) != 7 + 4 * n:
        raise ValueError(f"body has {len(tokens)} tokens, expected {7 + 4 * n}")
    if tokens[0] != f"<N_{n:03d}>":
        raise ValueError("body atom-count token does not match Plan")
    for position, token in enumerate(tokens):
        expected = _expected_family(position)
        if _token_family(token) != expected:
            raise ValueError(
                f"body position {position} expected {expected}, got {token!r}"
            )
    observed = Counter(tokens[7 + 4 * slot][3:-1] for slot in range(n))
    expected = Counter(
        {
            str(element): int(count)
            for element, count in zip(plan["elements"], plan["counts"], strict=True)
        }
    )
    if observed != expected:
        raise ValueError("body element counts do not match Plan")


def _coordinate_positions(slot: int) -> list[int]:
    return [8 + 4 * int(slot), 9 + 4 * int(slot), 10 + 4 * int(slot)]


def _closure_logs(
    rollout: Mapping[str, Any], n: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if rollout.get("spad_basin_closure") is not True:
        raise ValueError("rollout is not a SPAD basin-closure rollout")
    cell_value = rollout.get("spad_basin_closure_cell_revision_log")
    block_value = rollout.get("spad_basin_closure_species_block_revision_log")
    if not isinstance(cell_value, Mapping) or not isinstance(block_value, list):
        raise ValueError("rollout lacks basin-closure logs")
    cell = dict(cell_value)
    blocks = [dict(value) for value in block_value]
    if _int_list(cell.get("generation_positions"), "cell positions") != list(
        range(1, 7)
    ):
        raise ValueError("cell transaction must occupy generation positions 1..6")
    for name in ("previous_token_ids", "proposed_token_ids", "new_token_ids"):
        if len(_int_list(cell.get(name), f"cell {name}")) != 6:
            raise ValueError(f"cell {name} must contain six IDs")
    seen_slots: set[int] = set()
    for block_ordinal, block in enumerate(blocks):
        if int(block.get("block_index", -1)) != block_ordinal:
            raise ValueError("block indices must follow emitted order")
        slots = _int_list(block.get("slot_indices"), "block slot_indices")
        if not slots or len(slots) != len(set(slots)) or seen_slots.intersection(slots):
            raise ValueError("species blocks must contain disjoint nonempty slots")
        if any(slot < 0 or slot >= n for slot in slots):
            raise ValueError("species block slot lies outside dynamic body")
        seen_slots.update(slots)
        positions = _int_list(block.get("generation_positions"), "block positions")
        expected_positions = [value for slot in slots for value in _coordinate_positions(slot)]
        if positions != expected_positions:
            raise ValueError("block positions do not match exact dynamic 7+4N slots")
        width = len(positions)
        previous = _int_list(block.get("previous_token_ids"), "block previous IDs")
        proposed = _int_list(block.get("proposed_token_ids"), "block proposed IDs")
        new = _int_list(block.get("new_token_ids"), "block new IDs")
        if any(len(values) != width for values in (previous, proposed, new)):
            raise ValueError("block action widths do not match block positions")
        sites_value = block.get("site_revisions")
        if not isinstance(sites_value, list) or len(sites_value) != len(slots):
            raise ValueError("block site trace does not match slot count")
        site_new_flat: list[int] = []
        for site_ordinal, site_value in enumerate(sites_value):
            if not isinstance(site_value, Mapping):
                raise ValueError("site revision must be an object")
            site = dict(site_value)
            slot = slots[site_ordinal]
            expected_site_positions = _coordinate_positions(slot)
            if (
                int(site.get("block_index", -1)) != block_ordinal
                or int(site.get("site_order_index", -1)) != site_ordinal
                or int(site.get("slot_index", -1)) != slot
                or _int_list(site.get("generation_positions"), "site positions")
                != expected_site_positions
            ):
                raise ValueError("site trace cursor metadata is inconsistent")
            site_previous = _int_list(
                site.get("previous_token_ids"), "site previous IDs"
            )
            site_new = _int_list(site.get("new_token_ids"), "site new IDs")
            if len(site_previous) != 3 or len(site_new) != 3:
                raise ValueError("XYZ actions must contain exactly three IDs")
            start = 3 * site_ordinal
            if site_previous != previous[start : start + 3]:
                raise ValueError("site previous IDs disagree with block entry")
            site_new_flat.extend(site_new)
        if site_new_flat != proposed:
            raise ValueError("site outcomes do not reproduce block proposal")
        restored = bool(block.get("restored_complete_block"))
        expected_new = previous if restored else proposed
        if new != expected_new:
            raise ValueError("whole-block fallback outcome is inconsistent")
    if seen_slots != set(range(n)):
        raise ValueError("species-block trace does not cover every atom exactly once")
    return cell, blocks


def _observe_final_actions(
    final_tokens: Sequence[str],
    cell: Mapping[str, Any],
    blocks: Sequence[Mapping[str, Any]],
    resolver: TokenResolver,
) -> None:
    positions = _int_list(cell["generation_positions"], "cell positions")
    for position, token_id in zip(
        positions, _int_list(cell["new_token_ids"], "cell new IDs"), strict=True
    ):
        resolver.observe(final_tokens[position], token_id, position=position)
    for block in blocks:
        for position, token_id in zip(
            _int_list(block["generation_positions"], "block positions"),
            _int_list(block["new_token_ids"], "block new IDs"),
            strict=True,
        ):
            resolver.observe(final_tokens[position], token_id, position=position)


def _apply_action(
    body: list[str],
    positions: Sequence[int],
    token_ids: Sequence[int],
    resolver: TokenResolver,
) -> None:
    if len(positions) != len(token_ids):
        raise ValueError("action position/token widths differ")
    for position, token_id in zip(positions, token_ids, strict=True):
        if position < 0 or position >= len(body):
            raise ValueError("action position lies outside body")
        body[position] = resolver.resolve(int(token_id), position=int(position))


def _assert_action(
    body: Sequence[str],
    positions: Sequence[int],
    token_ids: Sequence[int],
    resolver: TokenResolver,
) -> None:
    expected = [
        resolver.resolve(int(token_id), position=int(position))
        for position, token_id in zip(positions, token_ids, strict=True)
    ]
    observed = [body[int(position)] for position in positions]
    if observed != expected:
        raise ValueError("logged action does not match body tokens")


def reconstruct_predictor(
    final_tokens: Sequence[str],
    cell: Mapping[str, Any],
    blocks: Sequence[Mapping[str, Any]],
    resolver: TokenResolver,
) -> list[str]:
    body = list(final_tokens)
    for block in reversed(blocks):
        positions = _int_list(block["generation_positions"], "block positions")
        _assert_action(
            body,
            positions,
            _int_list(block["new_token_ids"], "block new IDs"),
            resolver,
        )
        _apply_action(
            body,
            positions,
            _int_list(block["previous_token_ids"], "block previous IDs"),
            resolver,
        )
    cell_positions = _int_list(cell["generation_positions"], "cell positions")
    _assert_action(
        body,
        cell_positions,
        _int_list(cell["new_token_ids"], "cell new IDs"),
        resolver,
    )
    _apply_action(
        body,
        cell_positions,
        _int_list(cell["previous_token_ids"], "cell previous IDs"),
        resolver,
    )
    return body


def _forward_snapshots(
    predictor: Sequence[str],
    cell: Mapping[str, Any],
    blocks: Sequence[Mapping[str, Any]],
    resolver: TokenResolver,
) -> tuple[list[str], list[list[str]], list[str]]:
    body = list(predictor)
    _apply_action(
        body,
        _int_list(cell["generation_positions"], "cell positions"),
        _int_list(cell["new_token_ids"], "cell new IDs"),
        resolver,
    )
    post_cell = list(body)
    entries: list[list[str]] = []
    for block in blocks:
        entries.append(list(body))
        _apply_action(
            body,
            _int_list(block["generation_positions"], "block positions"),
            _int_list(block["new_token_ids"], "block new IDs"),
            resolver,
        )
    return post_cell, entries, body


def _n_bin(value: int) -> str:
    if value <= 4:
        return "n01_04"
    if value <= 8:
        return "n05_08"
    if value <= 12:
        return "n09_12"
    return "n13_20"


def _multiplicity_bin(value: int) -> str:
    if value <= 2:
        return "m01_02"
    if value <= 5:
        return "m03_05"
    return "m06_plus"


def stratum(row: Mapping[str, Any]) -> tuple[str, str, str]:
    plan = _plan_state(row)
    return (
        _n_bin(int(plan["N"])),
        _multiplicity_bin(max(int(value) for value in plan["counts"])),
        "species3plus" if len(plan["elements"]) >= 3 else "species1_2",
    )


def _stratum_key(row: Mapping[str, Any]) -> str:
    return "|".join(stratum(row))


def _source_index(row: Mapping[str, Any]) -> int:
    value = row.get("mp20_train_source_row_idx", row.get("source_row_idx"))
    if value is None:
        raise ValueError("plan row lacks MP20-train source index")
    return int(value)


def _row_order(record: Mapping[str, Any]) -> tuple[int, int]:
    return (_source_index(record["plan"]), int(record["sample_idx"]))


def _state_type_targets(expected_groups: int) -> dict[str, int]:
    if expected_groups <= 0:
        raise ValueError("expected_groups must be positive")
    return {
        "cell": (expected_groups + 1) // 2,
        "xyz": expected_groups // 2,
    }


def assign_state_types(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_groups: int = DEFAULT_EXPECTED_GROUPS,
) -> dict[int, str]:
    """Split every stratum as evenly as possible under aggregate constraints."""

    targets = _state_type_targets(int(expected_groups))
    if len(records) != int(expected_groups):
        raise ValueError(
            f"state assignment requires exactly {expected_groups} records"
        )

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_stratum_key(record["plan"])].append(record)
    group_items = [(key, sorted(groups[key], key=_row_order)) for key in sorted(groups)]
    total_high_n = sum(int(_plan_state(record["plan"])["N"]) >= 13 for record in records)
    total_high_m = sum(
        max(int(value) for value in _plan_state(record["plan"])["counts"]) >= 6
        for record in records
    )
    minimum_high_n = MIN_HIGH_N_PER_TYPE
    minimum_high_m = MIN_HIGH_MULTIPLICITY_PER_TYPE
    if total_high_n < 2 * minimum_high_n or total_high_m < 2 * minimum_high_m:
        raise ValueError("source cannot satisfy balanced high-N/high-multiplicity minima")

    # DP values are the per-stratum number assigned to cell.
    dp: dict[tuple[int, int, int], tuple[int, ...]] = {(0, 0, 0): ()}
    for key, values in group_items:
        size = len(values)
        choices = sorted({size // 2, size - size // 2})
        high_n = key.startswith("n13_20|")
        high_m = "|m06_plus|" in f"|{key}|"
        next_dp: dict[tuple[int, int, int], tuple[int, ...]] = {}
        for (count, count_n, count_m), history in sorted(dp.items()):
            for choice in choices:
                state = (
                    count + choice,
                    count_n + (choice if high_n else 0),
                    count_m + (choice if high_m else 0),
                )
                candidate = history + (choice,)
                if state not in next_dp or candidate < next_dp[state]:
                    next_dp[state] = candidate
        dp = next_dp

    feasible = [
        (state, history)
        for state, history in dp.items()
        if state[0] == targets["cell"]
        and minimum_high_n <= state[1] <= total_high_n - minimum_high_n
        and minimum_high_m <= state[2] <= total_high_m - minimum_high_m
    ]
    if not feasible:
        raise ValueError("balanced state assignment is mathematically infeasible")
    _, choices = min(
        feasible,
        key=lambda item: (
            abs(2 * item[0][1] - total_high_n),
            abs(2 * item[0][2] - total_high_m),
            item[1],
        ),
    )
    assignment: dict[int, str] = {}
    for (_, values), cell_count in zip(group_items, choices, strict=True):
        # Alternation avoids making one type a contiguous source-index prefix.
        alternating = values[::2] + values[1::2]
        cell_ids = {int(value["sample_idx"]) for value in alternating[:cell_count]}
        for value in values:
            sample_idx = int(value["sample_idx"])
            assignment[sample_idx] = "cell" if sample_idx in cell_ids else "xyz"
    if Counter(assignment.values()) != Counter(targets):
        raise RuntimeError("state assignment did not produce exact target counts")
    return assignment


def assign_cursor_buckets(
    xyz_records: Sequence[Mapping[str, Any]],
    *,
    expected_groups: int = DEFAULT_EXPECTED_GROUPS,
) -> dict[int, str]:
    expected_xyz = _state_type_targets(int(expected_groups))["xyz"]
    if len(xyz_records) != expected_xyz:
        raise ValueError(
            f"cursor assignment requires exactly {expected_xyz} xyz rows"
        )
    quotient, remainder = divmod(expected_xyz, len(CURSOR_BUCKETS))
    capacities = {
        name: quotient + int(index < remainder)
        for index, name in enumerate(CURSOR_BUCKETS)
    }
    global_counts: Counter[str] = Counter()
    assignment: dict[int, str] = {}
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in xyz_records:
        groups[_stratum_key(record["plan"])].append(record)
    for group_ordinal, key in enumerate(sorted(groups)):
        local_counts: Counter[str] = Counter()
        for row_ordinal, record in enumerate(sorted(groups[key], key=_row_order)):
            offset = (group_ordinal + row_ordinal) % len(CURSOR_BUCKETS)
            priority = CURSOR_BUCKETS[offset:] + CURSOR_BUCKETS[:offset]
            available = [
                name
                for name in priority
                if global_counts[name] < capacities[name]
            ]
            if not available:
                raise RuntimeError("cursor bucket capacity exhausted early")
            bucket = min(
                available,
                key=lambda name: (
                    local_counts[name],
                    global_counts[name],
                    priority.index(name),
                ),
            )
            assignment[int(record["sample_idx"])] = bucket
            local_counts[bucket] += 1
            global_counts[bucket] += 1
    if global_counts != Counter(capacities):
        raise RuntimeError("cursor buckets are not exactly balanced")
    return assignment


def _flatten_trace(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for block_ordinal, block in enumerate(blocks):
        sites = block["site_revisions"]
        for site_ordinal, site_value in enumerate(sites):
            trace.append(
                {
                    "flat_index": len(trace),
                    "block_ordinal": block_ordinal,
                    "site_ordinal": site_ordinal,
                    "block": block,
                    "site": site_value,
                }
            )
    return trace


def _cursor_index(size: int, bucket: str) -> int:
    if size <= 0:
        raise ValueError("cannot select a cursor from an empty trace")
    last = size - 1
    numerators = {"early": 1, "middle": 4, "late": 6, "terminal": 8}
    if bucket not in numerators:
        raise ValueError(f"unknown cursor bucket {bucket!r}")
    return (last * numerators[bucket]) // 8


def _mask_body(tokens: Sequence[str], positions: Sequence[int], mask_token: str) -> list[str]:
    body = list(tokens)
    for position in positions:
        body[int(position)] = mask_token
    return body


def _resolved_tokens(
    positions: Sequence[int], token_ids: Sequence[int], resolver: TokenResolver
) -> list[str]:
    return [
        resolver.resolve(int(token_id), position=int(position))
        for position, token_id in zip(positions, token_ids, strict=True)
    ]


def _common_state(record: Mapping[str, Any]) -> dict[str, Any]:
    plan_row = record["plan"]
    rollout = record["rollout"]
    plan = _plan_state(plan_row)
    metadata = rollout.get("spad_basin_closure_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("rollout lacks basin-closure metadata")
    for field in ("cell_sampling_seed", "species_block_sampling_seed"):
        if isinstance(metadata.get(field), bool) or not isinstance(
            metadata.get(field), int
        ):
            raise ValueError(f"rollout metadata lacks integer {field}")
    final_invalid = rollout.get("parsed") is not True or metadata.get(
        "final_geometry_supported"
    ) is not True
    return {
        "schema": STATE_SCHEMA,
        "preflight_idx": int(plan_row.get("preflight_idx", record["sample_idx"])),
        "sample_idx": int(record["sample_idx"]),
        "mp20_train_source_row_idx": _source_index(plan_row),
        "source_split": "train",
        "plan_state": dict(plan),
        "prompt": str(plan_row.get("prompt") or rollout.get("conditioning_prompt") or ""),
        "species_program": [str(value) for value in plan_row["species_program"]],
        "continuation_seeds": {
            "cell": int(metadata["cell_sampling_seed"]),
            "species_blocks": int(metadata["species_block_sampling_seed"]),
        },
        "stratum": {
            "key": _stratum_key(plan_row),
            "n": stratum(plan_row)[0],
            "multiplicity": stratum(plan_row)[1],
            "species": stratum(plan_row)[2],
        },
        "N": int(plan["N"]),
        "predictor_body": _body_text(record["predictor"]),
        "predictor_body_tokens": list(record["predictor"]),
        "final_body": _body_text(record["final_tokens"]),
        "final_body_tokens": list(record["final_tokens"]),
        "final_endpoint": {
            "parsed": rollout.get("parsed") is True,
            "final_geometry_supported": metadata.get("final_geometry_supported") is True,
            "invalid": bool(final_invalid),
        },
        "outcomes_read": False,
        "selection": False,
        "replacement": False,
    }


def materialize_cell_state(
    record: Mapping[str, Any], resolver: TokenResolver, mask_token: str
) -> dict[str, Any]:
    cell = record["cell"]
    positions = _int_list(cell["generation_positions"], "cell positions")
    previous_ids = _int_list(cell["previous_token_ids"], "cell previous IDs")
    new_ids = _int_list(cell["new_token_ids"], "cell new IDs")
    provisional = list(record["predictor"])
    state_body = _mask_body(provisional, positions, mask_token)
    replay = list(record["predictor"])
    _apply_action(replay, positions, new_ids, resolver)
    for block in record["blocks"]:
        _apply_action(
            replay,
            _int_list(block["generation_positions"], "block positions"),
            _int_list(block["new_token_ids"], "block new IDs"),
            resolver,
        )
    state = _common_state(record)
    state.update(
        {
            "state_type": "cell",
            "cursor_bucket": None,
            "cursor": None,
            "active_generation_positions": positions,
            "context_masked_generation_positions": [],
            "provisional_complete_body": _body_text(provisional),
            "provisional_complete_body_tokens": provisional,
            "state_body": _body_text(state_body),
            "state_body_tokens": state_body,
            "block_entry_snapshot": None,
            "reference_action": {
                "kind": "cell",
                "generation_positions": positions,
                "previous_token_ids": previous_ids,
                "previous_tokens": _resolved_tokens(positions, previous_ids, resolver),
                "token_ids": new_ids,
                "tokens": _resolved_tokens(positions, new_ids, resolver),
                "restored_complete_noop": bool(cell.get("restored_complete_noop")),
            },
            "reference_log_replay_matches_final": replay == record["final_tokens"],
        }
    )
    return state


def materialize_xyz_state(
    record: Mapping[str, Any],
    resolver: TokenResolver,
    mask_token: str,
    cursor_bucket: str,
) -> dict[str, Any]:
    trace = record["trace"]
    selected = trace[_cursor_index(len(trace), cursor_bucket)]
    block_ordinal = int(selected["block_ordinal"])
    site_ordinal = int(selected["site_ordinal"])
    block = selected["block"]
    site = selected["site"]
    block_entry = list(record["block_entries"][block_ordinal])
    provisional = list(block_entry)
    sites = block["site_revisions"]
    for prior in sites[:site_ordinal]:
        _apply_action(
            provisional,
            _int_list(prior["generation_positions"], "site positions"),
            _int_list(prior["new_token_ids"], "site new IDs"),
            resolver,
        )
    active = _int_list(site["generation_positions"], "site positions")
    context = [
        position
        for future in sites[site_ordinal + 1 :]
        for position in _int_list(future["generation_positions"], "future site positions")
    ]
    state_body = _mask_body(provisional, active + context, mask_token)
    previous_ids = _int_list(site["previous_token_ids"], "site previous IDs")
    new_ids = _int_list(site["new_token_ids"], "site new IDs")

    replay = list(provisional)
    _apply_action(replay, active, new_ids, resolver)
    for future in sites[site_ordinal + 1 :]:
        _apply_action(
            replay,
            _int_list(future["generation_positions"], "future site positions"),
            _int_list(future["new_token_ids"], "future site new IDs"),
            resolver,
        )
    _apply_action(
        replay,
        _int_list(block["generation_positions"], "block positions"),
        _int_list(block["new_token_ids"], "block new IDs"),
        resolver,
    )
    for later in record["blocks"][block_ordinal + 1 :]:
        _apply_action(
            replay,
            _int_list(later["generation_positions"], "later block positions"),
            _int_list(later["new_token_ids"], "later block new IDs"),
            resolver,
        )

    block_positions = _int_list(block["generation_positions"], "block positions")
    state = _common_state(record)
    state.update(
        {
            "state_type": "xyz",
            "cursor_bucket": cursor_bucket,
            "cursor": {
                "bucket": cursor_bucket,
                "block_index": int(site["block_index"]),
                "site_order_index": int(site["site_order_index"]),
                "slot_index": int(site["slot_index"]),
                "block_generation_positions": block_positions,
                "flattened_cursor_depth": int(selected["flat_index"]),
                "flattened_cursor_ordinal": int(selected["flat_index"]) + 1,
                "flattened_trace_length": len(trace),
            },
            "active_generation_positions": active,
            "context_masked_generation_positions": context,
            "provisional_complete_body": _body_text(provisional),
            "provisional_complete_body_tokens": provisional,
            "state_body": _body_text(state_body),
            "state_body_tokens": state_body,
            "block_entry_snapshot": {
                "body": _body_text(block_entry),
                "body_tokens": block_entry,
                "block_generation_positions": block_positions,
                "previous_token_ids": _int_list(
                    block["previous_token_ids"], "block previous IDs"
                ),
                "restored_complete_block": bool(block.get("restored_complete_block")),
            },
            "reference_action": {
                "kind": "xyz",
                "generation_positions": active,
                "previous_token_ids": previous_ids,
                "previous_tokens": _resolved_tokens(active, previous_ids, resolver),
                "token_ids": new_ids,
                "tokens": _resolved_tokens(active, new_ids, resolver),
                "restored_site_no_legal_z": bool(
                    site.get("restored_site_no_legal_z")
                ),
                "whole_block_restored_after_action": bool(
                    block.get("restored_complete_block")
                ),
            },
            "reference_log_replay_matches_final": replay == record["final_tokens"],
        }
    )
    return state


def prepare_records(
    plans: Sequence[dict[str, Any]],
    rollouts: Sequence[dict[str, Any]],
    resolver: TokenResolver,
    *,
    expected_groups: int = DEFAULT_EXPECTED_GROUPS,
) -> list[dict[str, Any]]:
    expected_groups = int(expected_groups)
    if expected_groups <= 0:
        raise ValueError("expected_groups must be positive")
    if len(plans) != expected_groups or len(rollouts) != expected_groups:
        raise ValueError(
            "materializer requires exactly "
            f"{expected_groups} plans and {expected_groups} rollouts"
        )
    plan_by_idx: dict[int, dict[str, Any]] = {}
    for plan in plans:
        _validate_plan_row(plan)
        sample_idx = int(plan["sample_idx"])
        if sample_idx in plan_by_idx:
            raise ValueError(f"duplicate plan sample_idx {sample_idx}")
        plan_by_idx[sample_idx] = plan
    rollout_by_idx: dict[int, dict[str, Any]] = {}
    for rollout in rollouts:
        sample_idx = int(rollout["sample_idx"])
        if sample_idx in rollout_by_idx:
            raise ValueError(f"duplicate rollout sample_idx {sample_idx}")
        rollout_by_idx[sample_idx] = rollout
    if set(plan_by_idx) != set(rollout_by_idx):
        raise ValueError("plan and rollout sample_idx sets differ")
    if set(plan_by_idx) != set(range(expected_groups)):
        raise ValueError(
            f"sample_idx must cover 0..{expected_groups - 1} exactly once"
        )
    source_indices = [_source_index(row) for row in plans]
    if len(set(source_indices)) != expected_groups:
        raise ValueError("MP20-train sources are not unique")

    records: list[dict[str, Any]] = []
    for sample_idx in sorted(plan_by_idx):
        plan_row = plan_by_idx[sample_idx]
        rollout = rollout_by_idx[sample_idx]
        plan = _plan_state(plan_row)
        prompt_record = rollout.get("prompt_record")
        if isinstance(prompt_record, Mapping):
            recorded_source = prompt_record.get(
                "mp20_train_source_row_idx", prompt_record.get("source_row_idx")
            )
            if recorded_source is not None and int(recorded_source) != _source_index(
                plan_row
            ):
                raise ValueError(f"sample {sample_idx} rollout source row changed")
            recorded_split = prompt_record.get(
                "source_split", prompt_record.get("split")
            )
            if recorded_split is not None and str(recorded_split).lower() not in {
                "train",
                "mp20_train_only",
                "mp20-train-only",
            }:
                raise ValueError(f"sample {sample_idx} rollout source is not train")
        rollout_plan = rollout.get("plan_state")
        if isinstance(rollout_plan, Mapping) and dict(rollout_plan) != dict(plan):
            raise ValueError(f"sample {sample_idx} rollout Plan changed")
        tokens = TOKEN_RE.findall(str(rollout.get("text") or ""))
        _validate_body(tokens, plan)
        cell, blocks = _closure_logs(rollout, int(plan["N"]))
        _observe_final_actions(tokens, cell, blocks, resolver)
        records.append(
            {
                "sample_idx": sample_idx,
                "plan": plan_row,
                "rollout": rollout,
                "final_tokens": tokens,
                "cell": cell,
                "blocks": blocks,
            }
        )

    for record in records:
        predictor = reconstruct_predictor(
            record["final_tokens"], record["cell"], record["blocks"], resolver
        )
        post_cell, entries, replay = _forward_snapshots(
            predictor, record["cell"], record["blocks"], resolver
        )
        if replay != record["final_tokens"]:
            raise ValueError(
                f"sample {record['sample_idx']} full logged replay mismatches final body"
            )
        record["predictor"] = predictor
        record["post_cell"] = post_cell
        record["block_entries"] = entries
        record["trace"] = _flatten_trace(record["blocks"])
    return records


def build_states(
    plans: Sequence[dict[str, Any]],
    rollouts: Sequence[dict[str, Any]],
    *,
    mask_token: str = MASK_DEFAULT,
    vocabulary: Mapping[str, int] | None = None,
    expected_groups: int = DEFAULT_EXPECTED_GROUPS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolver = TokenResolver(vocabulary)
    records = prepare_records(
        plans, rollouts, resolver, expected_groups=int(expected_groups)
    )
    type_assignment = assign_state_types(
        records, expected_groups=int(expected_groups)
    )
    xyz_records = [
        record
        for record in records
        if type_assignment[int(record["sample_idx"])] == "xyz"
    ]
    cursor_assignment = assign_cursor_buckets(
        xyz_records, expected_groups=int(expected_groups)
    )
    states: list[dict[str, Any]] = []
    for record in records:
        sample_idx = int(record["sample_idx"])
        if type_assignment[sample_idx] == "cell":
            state = materialize_cell_state(record, resolver, mask_token)
        else:
            state = materialize_xyz_state(
                record, resolver, mask_token, cursor_assignment[sample_idx]
            )
        states.append(state)
    replay_mismatches = sum(
        not state["reference_log_replay_matches_final"] for state in states
    )
    if replay_mismatches:
        raise ValueError(f"{replay_mismatches} selected reference replays mismatch final")
    type_counts = Counter(state["state_type"] for state in states)
    cursor_counts = Counter(
        state["cursor_bucket"] for state in states if state["state_type"] == "xyz"
    )
    strata: dict[str, dict[str, int]] = {}
    for key in sorted({state["stratum"]["key"] for state in states}):
        subset = [state for state in states if state["stratum"]["key"] == key]
        strata[key] = {
            "total": len(subset),
            "cell": sum(state["state_type"] == "cell" for state in subset),
            "xyz": sum(state["state_type"] == "xyz" for state in subset),
        }
    high_n = {
        name: sum(state["state_type"] == name and state["N"] >= 13 for state in states)
        for name in ("cell", "xyz")
    }
    high_m = {
        name: sum(
            state["state_type"] == name
            and max(int(value) for value in state["plan_state"]["counts"]) >= 6
            for state in states
        )
        for name in ("cell", "xyz")
    }
    if (
        min(high_n.values()) < MIN_HIGH_N_PER_TYPE
        or min(high_m.values()) < MIN_HIGH_MULTIPLICITY_PER_TYPE
    ):
        raise RuntimeError("aggregate state-type coverage constraints were not met")
    manifest = {
        "schema": "spad_basin_preflight_state_manifest_v1",
        "state_schema": STATE_SCHEMA,
        "rows": len(states),
        "source_rows_retained": len(states),
        "unique_source_rows": len({state["mp20_train_source_row_idx"] for state in states}),
        "source_split": "MP20_train_only",
        "preliminary_preflight_state_type_ignored": True,
        "state_types": dict(sorted(type_counts.items())),
        "cursor_buckets": {name: cursor_counts[name] for name in CURSOR_BUCKETS},
        "strata": strata,
        "high_N_rows": {"total": sum(high_n.values()), **high_n},
        "high_multiplicity_rows": {"total": sum(high_m.values()), **high_m},
        "invalid_final_endpoint_count": sum(
            state["final_endpoint"]["invalid"] for state in states
        ),
        "reference_log_replay_mismatches": replay_mismatches,
        "outcomes_read": False,
        "selection": False,
        "replacement": False,
        "selection_or_replacement": False,
        "tokenizer_vocabulary_supplied": vocabulary is not None,
    }
    if int(expected_groups) != DEFAULT_EXPECTED_GROUPS:
        manifest["expected_groups"] = int(expected_groups)
    return states, manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    plans_path = args.plans_jsonl.resolve()
    rollouts_path = args.rollouts_jsonl.resolve()
    vocabulary = load_vocabulary(getattr(args, "tokenizer_vocab_json", None))
    expected_groups = int(
        getattr(args, "expected_groups", DEFAULT_EXPECTED_GROUPS)
    )
    states, manifest = build_states(
        list(iter_jsonl(plans_path)),
        list(iter_jsonl(rollouts_path)),
        mask_token=str(args.mask_token),
        vocabulary=vocabulary,
        expected_groups=expected_groups,
    )
    manifest.update(
        {
            "plans_jsonl": str(plans_path),
            "rollouts_jsonl": str(rollouts_path),
        }
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "states.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for state in states:
            handle.write(json.dumps(state, sort_keys=True) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "_SUCCESS").touch()
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans-jsonl", type=Path, required=True)
    parser.add_argument("--rollouts-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-token", default=MASK_DEFAULT)
    parser.add_argument("--tokenizer-vocab-json", type=Path)
    parser.add_argument(
        "--expected-groups", type=int, default=DEFAULT_EXPECTED_GROUPS
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
