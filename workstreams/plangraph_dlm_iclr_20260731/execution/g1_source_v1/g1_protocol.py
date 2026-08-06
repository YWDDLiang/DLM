#!/usr/bin/env python3
"""Pure protocol helpers for the frozen PlanGraph-DLM G1 screen."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


G1_SCHEMA = "plangraph-dlm-g1@1"
G1_ATTEMPTS = 512
G1_SEED_BASE = 20260731
PLANGRAPH_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "source_plan_state_version",
        "site_group_strategy",
        "composition",
        "symmetry",
        "lattice",
        "site_groups",
        "constraints",
        "dependency_order",
    }
)


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"non-object JSONL row in {path}")
                yield payload


def ordinal_seed(ordinal: int, *, seed_base: int = G1_SEED_BASE) -> int:
    value = int(ordinal)
    if not 0 <= value < G1_ATTEMPTS:
        raise ValueError(f"ordinal {value} is outside the frozen G1 panel")
    return int(seed_base) + value


def build_seed_ledger(*, attempts: int = G1_ATTEMPTS, seed_base: int = G1_SEED_BASE) -> list[dict[str, int]]:
    if int(attempts) != G1_ATTEMPTS:
        raise ValueError(f"G1 freezes attempts={G1_ATTEMPTS}")
    return [
        {"ordinal": ordinal, "seed": ordinal_seed(ordinal, seed_base=seed_base)}
        for ordinal in range(int(attempts))
    ]


def ledger_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    encoded = "".join(canonical_json(dict(row)) + "\n" for row in rows)
    return sha256_bytes(encoded.encode("utf-8"))


def _content_rank(*, identity: str, seed: int, node: str) -> bytes:
    payload = f"plangraph_pg_shuffle_v1\0{int(seed)}\0{identity}\0{node}"
    return hashlib.sha256(payload.encode("utf-8")).digest()


def shuffle_dependency_links(
    graph: Mapping[str, Any],
    *,
    identity: str,
    seed: int = G1_SEED_BASE,
) -> dict[str, Any]:
    """Build the registered negative control without changing composition.

    The node order is a content-keyed SHA-256 permutation. Site-group
    prerequisites are replaced by the nodes preceding that group in the
    shuffled order. The result intentionally need not satisfy the proposed
    PlanGraph dependency semantics; it remains a complete JSON target with the
    same fields and composition.
    """

    payload = json.loads(canonical_json(dict(graph)))
    if set(payload) != set(PLANGRAPH_TOP_LEVEL_FIELDS):
        raise ValueError("PlanGraph top-level fields changed before shuffle")
    groups = payload.get("site_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("PlanGraph site_groups must be a non-empty list")
    group_ids = [str(group.get("group_id")) for group in groups]
    nodes = ["composition", "symmetry_lattice", *group_ids]
    shuffled = sorted(
        nodes,
        key=lambda node: (_content_rank(identity=str(identity), seed=int(seed), node=node), node),
    )
    if len(shuffled) > 1 and shuffled == nodes:
        rotation_digest = hashlib.sha256(
            f"plangraph_pg_shuffle_rotation_v1\0{int(seed)}\0{identity}".encode("utf-8")
        ).digest()
        rotation = 1 + int.from_bytes(rotation_digest[:8], "big") % (len(shuffled) - 1)
        shuffled = [*shuffled[rotation:], *shuffled[:rotation]]
    payload["dependency_order"] = shuffled
    positions = {name: index for index, name in enumerate(shuffled)}
    for group in groups:
        group_id = str(group["group_id"])
        group["depends_on"] = list(shuffled[: positions[group_id]])
    if payload.get("composition") != graph.get("composition"):
        raise AssertionError("PG-shuffle changed composition")
    if set(payload) != set(PLANGRAPH_TOP_LEVEL_FIELDS):
        raise AssertionError("PG-shuffle changed top-level fields")
    if payload["dependency_order"] == list(graph.get("dependency_order") or []):
        raise AssertionError("PG-shuffle produced an identity dependency order")
    return payload


def extract_first_json(text: str) -> tuple[dict[str, Any], int]:
    """Parse the first complete JSON object without repairing the continuation."""

    raw = str(text)
    start = raw.find("{")
    if start < 0:
        raise ValueError("no JSON object start")
    payload, end = json.JSONDecoder().raw_decode(raw[start:])
    if not isinstance(payload, dict):
        raise ValueError("generated JSON is not an object")
    return payload, start + end


def plangraph_completion(graph: Mapping[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    keys = set(graph)
    if keys != set(PLANGRAPH_TOP_LEVEL_FIELDS):
        errors.append(
            f"top_level_fields missing={sorted(PLANGRAPH_TOP_LEVEL_FIELDS - keys)} "
            f"extra={sorted(keys - PLANGRAPH_TOP_LEVEL_FIELDS)}"
        )
    composition = graph.get("composition")
    required_composition = {
        "N",
        "elements",
        "counts",
        "formula",
        "reduced_formula",
        "charge_bucket",
        "oxidation_candidates",
        "anion_framework",
    }
    if not isinstance(composition, Mapping):
        errors.append("composition is not an object")
    elif set(composition) != required_composition:
        errors.append("composition fields are incomplete")
    if not isinstance(graph.get("symmetry"), Mapping):
        errors.append("symmetry is not an object")
    if not isinstance(graph.get("lattice"), Mapping):
        errors.append("lattice is not an object")
    if not isinstance(graph.get("constraints"), Mapping):
        errors.append("constraints is not an object")
    if not isinstance(graph.get("site_groups"), list) or not graph.get("site_groups"):
        errors.append("site_groups is empty or malformed")
    if not isinstance(graph.get("dependency_order"), list) or not graph.get("dependency_order"):
        errors.append("dependency_order is empty or malformed")
    return not errors, errors


def composition_fields(graph: Mapping[str, Any]) -> tuple[list[str], list[int]]:
    composition = graph.get("composition")
    if not isinstance(composition, Mapping):
        raise ValueError("missing composition object")
    elements = composition.get("elements")
    counts = composition.get("counts")
    if not isinstance(elements, list) or not isinstance(counts, list):
        raise ValueError("composition elements/counts must be lists")
    if not elements or len(elements) != len(counts):
        raise ValueError("composition elements/counts arity mismatch")
    symbols = [str(value) for value in elements]
    values = [int(value) for value in counts]
    if any(value <= 0 for value in values):
        raise ValueError("composition counts must be positive")
    return symbols, values


def jsd_bits(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    keys = sorted(set(left) | set(right))
    left_total = float(sum(max(0, int(left.get(key, 0))) for key in keys))
    right_total = float(sum(max(0, int(right.get(key, 0))) for key in keys))
    if left_total <= 0 or right_total <= 0:
        return 1.0

    def term(probability: float, mixture: float) -> float:
        return 0.0 if probability <= 0.0 else probability * math.log2(probability / mixture)

    divergence = 0.0
    for key in keys:
        p = max(0, int(left.get(key, 0))) / left_total
        q = max(0, int(right.get(key, 0))) / right_total
        m = 0.5 * (p + q)
        divergence += 0.5 * term(p, m) + 0.5 * term(q, m)
    return float(divergence)


def rate(numerator: int | float, denominator: int | float = G1_ATTEMPTS) -> float:
    return float(numerator) / max(1.0, float(denominator))


def evaluate_g1_gate(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    max_jsd_bits: float = 0.15,
    max_shortcut_rate_delta: float = 0.05,
) -> dict[str, Any]:
    required = {"P0", "PG", "PG-shuffle"}
    if set(reports) != required:
        raise ValueError(f"G1 requires exactly {sorted(required)}")
    for arm, report in reports.items():
        if int(report.get("attempts") or 0) != G1_ATTEMPTS:
            raise ValueError(f"{arm} does not retain all {G1_ATTEMPTS} attempts")

    p0 = reports["P0"]
    pg = reports["PG"]
    shuffle = reports["PG-shuffle"]
    drift_fields = (
        "num_atoms_histogram",
        "element_arity_histogram",
        "lattice_system_histogram",
        "spacegroup_bucket_histogram",
        "anion_framework_histogram",
    )
    drift = {
        field: jsd_bits(
            p0.get(field) or {},
            pg.get(field) or {},
        )
        for field in drift_fields
    }
    max_observed_jsd = max(drift.values(), default=1.0)
    shortcut_deltas = {
        "single_element_rate": float(pg.get("single_element_rate") or 0.0)
        - float(p0.get("single_element_rate") or 0.0),
        "all_metal_rate": float(pg.get("all_metal_rate") or 0.0)
        - float(p0.get("all_metal_rate") or 0.0),
    }
    checks = {
        "pg_parse_rate_at_least_0_99": float(pg.get("parse_rate") or 0.0) >= 0.99,
        "pg_plan_completion_rate_at_least_0_98": float(
            pg.get("plan_completion_rate") or 0.0
        )
        >= 0.98,
        "pg_composition_valid_rate_at_least_0_95": float(
            pg.get("composition_valid_rate") or 0.0
        )
        >= 0.95,
        "pg_unique_formula_at_least_0_95_of_p0": int(
            pg.get("unique_formula_count") or 0
        )
        >= 0.95 * int(p0.get("unique_formula_count") or 0),
        "pg_beats_pg_shuffle_composition_validity": int(
            pg.get("composition_valid_count") or 0
        )
        > int(shuffle.get("composition_valid_count") or 0),
        "categorical_jsd_within_frozen_limit": max_observed_jsd
        <= float(max_jsd_bits),
        "single_element_inflation_within_frozen_limit": shortcut_deltas[
            "single_element_rate"
        ]
        <= float(max_shortcut_rate_delta),
        "all_metal_inflation_within_frozen_limit": shortcut_deltas[
            "all_metal_rate"
        ]
        <= float(max_shortcut_rate_delta),
    }
    return {
        "schema": G1_SCHEMA,
        "gate": "G1",
        "checks": checks,
        "passed": all(checks.values()),
        "distribution_drift": {
            "jsd_bits": drift,
            "maximum_observed_jsd_bits": max_observed_jsd,
            "maximum_allowed_jsd_bits": float(max_jsd_bits),
            "shortcut_rate_deltas": shortcut_deltas,
            "maximum_allowed_shortcut_rate_delta": float(
                max_shortcut_rate_delta
            ),
        },
    }


def histogram(values: Iterable[Any]) -> dict[str, int]:
    counter = Counter(str(value) for value in values)
    return {key: int(counter[key]) for key in sorted(counter)}

