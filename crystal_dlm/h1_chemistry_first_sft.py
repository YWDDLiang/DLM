"""Pure contracts for the Evidence-First chemistry-first Planner SFT.

The inference representation remains the existing six-line no-charge H1 Plan.
Ion and oxidation witnesses are training-only arithmetic supervision.  This
module intentionally has no model dependency so ordering, weighting, and
one-epoch accumulation contracts can be audited on CPU.
"""

from __future__ import annotations

from collections import Counter, deque
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


H1_CHEMISTRY_FIRST_SFT_SCHEMA = "h1_chemistry_first_sft_v1"
H1_CHEMISTRY_FIRST_SFT_SEED = 26080817
H1_CHEMISTRY_FIRST_AUX_TASKS = (
    "atoms_to_formula",
    "ions_to_charge_sum_formula",
    "masked_oxidation",
    "formula_to_elements_counts_n",
)
H1_CHEMISTRY_FIRST_TRAIN_TASKS = (
    "conditional_structural_anchor",
    "direct_nocharge_plan",
    *H1_CHEMISTRY_FIRST_AUX_TASKS,
)
H1_CHEMISTRY_FIRST_PROMPT_STYLE = "rich_nocharge_plan_v1"
H1_CHEMISTRY_FIRST_INFERENCE_MESSAGES_SHA256 = (
    "b6231c94cda80ee9762ea247827ef98c69c4226d8130ea85be9fee0fc2f53bcd"
)
H1_CHEMISTRY_FIRST_GRAD_ACCUM = 8
H1_CHEMISTRY_FIRST_LR = 2e-6
H1_CHEMISTRY_FIRST_WEIGHT_DECAY = 0.0
H1_CHEMISTRY_FIRST_CURRICULUM_PREFIX_FRACTION = 0.10


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def record_content(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return order-independent record content used for multiset identity."""

    excluded = {"ledger_ordinal", "candidate_order"}
    return {str(key): value for key, value in record.items() if str(key) not in excluded}


def record_multiset_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    rows = sorted(
        (record_content(record) for record in records),
        key=lambda value: str(value.get("record_id", "")),
    )
    return canonical_json_sha256(rows)


def record_order_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_sha256([str(record["record_id"]) for record in records])


def _hash_key(seed: int, role: str, value: str) -> bytes:
    return hashlib.sha256(f"{int(seed)}|{role}|{value}".encode("utf-8")).digest()


def hash_shuffle(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = H1_CHEMISTRY_FIRST_SFT_SEED,
    role: str = "base_hash_shuffle",
) -> list[dict[str, Any]]:
    """Stateless deterministic permutation with record-id tie breaking."""

    rows = [dict(record) for record in records]
    ids = [str(row.get("record_id", "")) for row in rows]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("records require unique non-empty record_id values")
    return sorted(
        rows,
        key=lambda row: (
            _hash_key(int(seed), str(role), str(row["record_id"])),
            str(row["record_id"]),
        ),
    )


def assign_auxiliary_tasks(
    positive_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = H1_CHEMISTRY_FIRST_SFT_SEED,
) -> dict[int, str]:
    """Cycle the four auxiliary tasks over one deterministic POS ordering."""

    rows = sorted(
        positive_rows,
        key=lambda row: (
            _hash_key(
                int(seed),
                "auxiliary_cycle",
                f"{int(row['row_idx'])}|{row.get('material_id', '')}",
            ),
            int(row["row_idx"]),
        ),
    )
    result: dict[int, str] = {}
    for rank, row in enumerate(rows):
        row_idx = int(row["row_idx"])
        if row_idx in result:
            raise ValueError(f"duplicate POS row_idx {row_idx}")
        result[row_idx] = H1_CHEMISTRY_FIRST_AUX_TASKS[
            rank % len(H1_CHEMISTRY_FIRST_AUX_TASKS)
        ]
    return result


def deterministic_mask_cursor(
    atom_count: int,
    *,
    row_idx: int,
    material_id: str,
    seed: int = H1_CHEMISTRY_FIRST_SFT_SEED,
) -> int:
    count = int(atom_count)
    if count <= 0:
        raise ValueError("atom_count must be positive")
    digest = _hash_key(
        int(seed),
        "masked_oxidation_cursor",
        f"{int(row_idx)}|{str(material_id)}",
    )
    return int.from_bytes(digest[:8], "big") % count


def _deficit_round_robin_ids(
    queues: Mapping[str, Sequence[str]],
    *,
    category_order: Sequence[str],
) -> list[str]:
    """Deterministically interleave queues by largest proportional deficit.

    At output position ``t`` each non-empty category has ideal cumulative
    service ``t * category_count / total``.  The category with the largest
    ideal-minus-actual deficit is served; ties use ``category_order``.  This
    produces an exact no-replacement weighted round robin without randomness.
    """

    order = [str(value) for value in category_order]
    if set(order) != set(str(key) for key in queues):
        raise ValueError("category_order must name every queue exactly once")
    pending = {key: deque(str(value) for value in queues[key]) for key in order}
    initial = {key: len(pending[key]) for key in order}
    total = sum(initial.values())
    emitted = Counter()
    result: list[str] = []
    for position in range(1, total + 1):
        candidates = [key for key in order if pending[key]]
        if not candidates:
            raise RuntimeError("deficit round robin exhausted early")
        chosen = max(
            candidates,
            key=lambda key: (
                position * initial[key] / total - emitted[key],
                -order.index(key),
            ),
        )
        result.append(pending[chosen].popleft())
        emitted[chosen] += 1
    if any(pending[key] for key in order):
        raise RuntimeError("deficit round robin left unconsumed records")
    return result


def curriculum_order(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = H1_CHEMISTRY_FIRST_SFT_SEED,
    prefix_fraction: float = H1_CHEMISTRY_FIRST_CURRICULUM_PREFIX_FRACTION,
) -> list[dict[str, Any]]:
    """Create the frozen SFT-v2-C ordering while preserving record content."""

    rows = [dict(record) for record in records]
    by_id = {str(row["record_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("curriculum records require unique record_id values")
    fraction = float(prefix_fraction)
    if not (0.0 <= fraction <= 1.0):
        raise ValueError("prefix_fraction must be in [0, 1]")
    prefix_count = math.floor(len(rows) * fraction)

    direct = hash_shuffle(
        [row for row in rows if row.get("task") == "direct_nocharge_plan"],
        seed=int(seed),
        role="curriculum_prefix_direct",
    )
    auxiliary = hash_shuffle(
        [row for row in rows if row.get("task") in H1_CHEMISTRY_FIRST_AUX_TASKS],
        seed=int(seed),
        role="curriculum_prefix_auxiliary",
    )
    direct_ids = deque(str(row["record_id"]) for row in direct)
    auxiliary_ids = deque(str(row["record_id"]) for row in auxiliary)
    prefix: list[str] = []
    next_role = "direct"
    while len(prefix) < prefix_count:
        if next_role == "direct" and direct_ids:
            prefix.append(direct_ids.popleft())
            next_role = "auxiliary"
        elif next_role == "auxiliary" and auxiliary_ids:
            prefix.append(auxiliary_ids.popleft())
            next_role = "direct"
        elif direct_ids:
            prefix.append(direct_ids.popleft())
        elif auxiliary_ids:
            prefix.append(auxiliary_ids.popleft())
        else:
            raise ValueError(
                "the 10% curriculum prefix exceeds direct-plus-auxiliary support"
            )

    used = set(prefix)
    remainder_categories = {
        "anchor": [
            str(row["record_id"])
            for row in hash_shuffle(
                [
                    row
                    for row in rows
                    if row.get("task") == "conditional_structural_anchor"
                    and str(row["record_id"]) not in used
                ],
                seed=int(seed),
                role="curriculum_remainder_anchor",
            )
        ],
        "direct": [
            str(row["record_id"])
            for row in direct
            if str(row["record_id"]) not in used
        ],
        "auxiliary": [
            str(row["record_id"])
            for row in auxiliary
            if str(row["record_id"]) not in used
        ],
    }
    remainder = _deficit_round_robin_ids(
        remainder_categories,
        category_order=("anchor", "direct", "auxiliary"),
    )
    ordered_ids = prefix + remainder
    if len(ordered_ids) != len(rows) or len(set(ordered_ids)) != len(rows):
        raise RuntimeError("curriculum ordering did not consume each record once")
    if set(ordered_ids) != set(by_id):
        raise RuntimeError("curriculum ordering changed the record multiset")
    return [dict(by_id[record_id]) for record_id in ordered_ids]


def add_weight_span(
    answer: str,
    payload: str,
    *,
    label: str,
    weight: float = 2.0,
    occurrence: int = 0,
) -> dict[str, Any]:
    """Return one exact character span for a chemistry answer payload."""

    text = str(answer)
    needle = str(payload)
    if not needle:
        raise ValueError("weighted payload cannot be empty")
    start = -1
    cursor = 0
    for _ in range(int(occurrence) + 1):
        start = text.find(needle, cursor)
        if start < 0:
            raise ValueError(f"payload {needle!r} is absent from answer {text!r}")
        cursor = start + len(needle)
    return {
        "start": start,
        "end": start + len(needle),
        "weight": float(weight),
        "label": str(label),
    }


def validate_weight_spans(
    answer: str,
    spans: Sequence[Mapping[str, Any]],
    *,
    expected_weight: float = 2.0,
) -> None:
    text = str(answer)
    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        if not (0 <= start < end <= len(text)):
            raise ValueError(f"invalid weighted span {dict(span)}")
        if float(span["weight"]) != float(expected_weight):
            raise ValueError(f"non-frozen token weight {dict(span)}")


def optimizer_update_count(record_count: int, grad_accum: int = H1_CHEMISTRY_FIRST_GRAD_ACCUM) -> int:
    records = int(record_count)
    accumulation = int(grad_accum)
    if records <= 0 or accumulation <= 0:
        raise ValueError("record_count and grad_accum must be positive")
    return math.ceil(records / accumulation)


def warmup_step_count(total_updates: int) -> int:
    updates = int(total_updates)
    if updates <= 0:
        raise ValueError("total_updates must be positive")
    return max(25, math.floor(0.03 * updates + 0.5))


def accumulation_group_size(
    microbatch_index: int,
    *,
    total_microbatches: int,
    grad_accum: int = H1_CHEMISTRY_FIRST_GRAD_ACCUM,
) -> int:
    """Actual divisor for one microbatch, including the final partial group."""

    index = int(microbatch_index)
    total = int(total_microbatches)
    accumulation = int(grad_accum)
    if not (0 <= index < total) or accumulation <= 0:
        raise ValueError("invalid microbatch geometry")
    group_start = (index // accumulation) * accumulation
    return min(accumulation, total - group_start)


def accumulation_is_update_boundary(
    microbatch_index: int,
    *,
    total_microbatches: int,
    grad_accum: int = H1_CHEMISTRY_FIRST_GRAD_ACCUM,
) -> bool:
    index = int(microbatch_index)
    total = int(total_microbatches)
    accumulation = int(grad_accum)
    if not (0 <= index < total) or accumulation <= 0:
        raise ValueError("invalid microbatch geometry")
    return (index + 1) % accumulation == 0 or index + 1 == total


def order_pair_audit(
    base_records: Sequence[Mapping[str, Any]],
    curriculum_records: Sequence[Mapping[str, Any]],
    *,
    prefix_fraction: float = H1_CHEMISTRY_FIRST_CURRICULUM_PREFIX_FRACTION,
) -> dict[str, Any]:
    base_ids = [str(row["record_id"]) for row in base_records]
    curriculum_ids = [str(row["record_id"]) for row in curriculum_records]
    multiset_equal = (
        len(base_ids) == len(curriculum_ids)
        and Counter(base_ids) == Counter(curriculum_ids)
        and record_multiset_sha256(base_records)
        == record_multiset_sha256(curriculum_records)
    )
    prefix_count = math.floor(len(curriculum_records) * float(prefix_fraction))
    prefix_tasks = [str(row["task"]) for row in curriculum_records[:prefix_count]]
    prefix_roles = [
        "direct" if task == "direct_nocharge_plan" else "auxiliary"
        for task in prefix_tasks
    ]
    prefix_allowed = all(
        task == "direct_nocharge_plan" or task in H1_CHEMISTRY_FIRST_AUX_TASKS
        for task in prefix_tasks
    )
    prefix_alternating = all(
        left != right for left, right in zip(prefix_roles, prefix_roles[1:])
    )
    failures: list[str] = []
    if not multiset_equal:
        failures.append("record_multiset_mismatch")
    if not prefix_allowed:
        failures.append("curriculum_prefix_contains_anchor")
    if not prefix_alternating:
        failures.append("curriculum_prefix_not_alternating")
    if len(base_ids) != len(set(base_ids)):
        failures.append("base_duplicate_record_id")
    if len(curriculum_ids) != len(set(curriculum_ids)):
        failures.append("curriculum_duplicate_record_id")
    return {
        "passed": not failures,
        "failures": failures,
        "record_count": len(base_records),
        "record_multiset_equal": multiset_equal,
        "record_multiset_sha256": record_multiset_sha256(base_records),
        "base_order_sha256": record_order_sha256(base_records),
        "curriculum_order_sha256": record_order_sha256(curriculum_records),
        "orders_differ": base_ids != curriculum_ids,
        "curriculum_prefix_fraction": float(prefix_fraction),
        "curriculum_prefix_count": prefix_count,
        "curriculum_prefix_allowed": prefix_allowed,
        "curriculum_prefix_alternating": prefix_alternating,
    }


def task_census(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["task"]) for row in records).items()))


__all__ = [
    "H1_CHEMISTRY_FIRST_AUX_TASKS",
    "H1_CHEMISTRY_FIRST_CURRICULUM_PREFIX_FRACTION",
    "H1_CHEMISTRY_FIRST_GRAD_ACCUM",
    "H1_CHEMISTRY_FIRST_INFERENCE_MESSAGES_SHA256",
    "H1_CHEMISTRY_FIRST_LR",
    "H1_CHEMISTRY_FIRST_PROMPT_STYLE",
    "H1_CHEMISTRY_FIRST_SFT_SCHEMA",
    "H1_CHEMISTRY_FIRST_SFT_SEED",
    "H1_CHEMISTRY_FIRST_TRAIN_TASKS",
    "H1_CHEMISTRY_FIRST_WEIGHT_DECAY",
    "accumulation_group_size",
    "accumulation_is_update_boundary",
    "add_weight_span",
    "assign_auxiliary_tasks",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "curriculum_order",
    "deterministic_mask_cursor",
    "hash_shuffle",
    "optimizer_update_count",
    "order_pair_audit",
    "record_content",
    "record_multiset_sha256",
    "record_order_sha256",
    "task_census",
    "validate_weight_spans",
    "warmup_step_count",
]
