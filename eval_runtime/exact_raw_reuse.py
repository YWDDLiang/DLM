#!/usr/bin/env python3
"""Exact, content-addressed planning for paired raw F/M relaxations.

This module deliberately knows nothing about StructureMatcher.  Its identity is
the SHA-256 of the lossless canonical JSON structure payload, so a numerical or
site-order change is a cache miss rather than a scientific equivalence claim.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, NamedTuple, TypeVar


ROLE_ORDER = ("F", "M")
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
T = TypeVar("T")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_structure_sha256(structure: Mapping[str, Any] | Any) -> str:
    """Return a lossless canonical identity without geometric tolerances."""

    payload = structure.as_dict() if hasattr(structure, "as_dict") else structure
    if not isinstance(payload, Mapping):
        raise TypeError("structure identity requires a mapping or as_dict()")
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def canonical_result_sha256(
    structure_sha256: str,
    energy_per_atom: float | None,
    composition: Mapping[str, Any],
) -> str:
    identity = _require_sha256(structure_sha256, "structure identity")
    payload = {
        "structure_sha256": identity,
        "energy_per_atom": energy_per_atom,
        "composition": dict(composition),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    observed = str(value)
    if HEX_SHA256.fullmatch(observed) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return observed


class ManifestIdentities(NamedTuple):
    total_attempts: int
    reconstructed: tuple[str, ...]


def manifest_identities(manifest: Mapping[str, Any]) -> ManifestIdentities:
    """Extract only reconstructed identities while retaining the denominator."""

    if manifest.get("schema") != "crysllmgen_r5c_a100_input_manifest_v1":
        raise ValueError("unexpected A100 input manifest schema")
    total_attempts = int(manifest.get("total_attempts", -1))
    records = manifest.get("attempt_records")
    if (
        total_attempts < 0
        or not isinstance(records, list)
        or len(records) != total_attempts
    ):
        raise ValueError("input manifest attempt denominator changed")
    if [int(record.get("generation_ordinal", -1)) for record in records] != list(
        range(total_attempts)
    ):
        raise ValueError("input manifest generation order changed")

    by_index: dict[int, str] = {}
    for record in records:
        raw_index = record.get("reconstructed_index")
        if raw_index is None:
            continue
        index = int(raw_index)
        if record.get("status") != "succeeded":
            raise ValueError("only successful reconstructed structures may be reused")
        if index in by_index:
            raise ValueError("duplicate reconstructed index")
        by_index[index] = _require_sha256(
            record.get("structure_sha256"), "canonical structure identity"
        )

    reconstructed_count = int(manifest.get("reconstructed_structures", -1))
    if set(by_index) != set(range(reconstructed_count)):
        raise ValueError("reconstructed identity coverage changed")
    return ManifestIdentities(
        total_attempts=total_attempts,
        reconstructed=tuple(by_index[index] for index in range(reconstructed_count)),
    )


class ExactPairPlan(NamedTuple):
    roles: tuple[str, str]
    unique_identities: tuple[str, ...]
    owners: tuple[str, ...]
    pair_attempts: int
    pair_reconstructed: int
    cache_hits: int
    cache_misses: int

    def owned_identities(self, role: str) -> tuple[str, ...]:
        if role not in self.roles:
            raise ValueError(f"unknown exact-reuse role: {role}")
        return tuple(
            identity
            for identity, owner in zip(self.unique_identities, self.owners)
            if owner == role
        )

    def owner_for(self, structure_sha256: str) -> str:
        identity = _require_sha256(structure_sha256, "structure identity")
        try:
            index = self.unique_identities.index(identity)
        except ValueError as exc:
            raise KeyError(identity) from exc
        return self.owners[index]

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": "h1_exact_raw_reuse_pair_plan_v1",
            "roles": list(self.roles),
            "unique_structures": [
                {"structure_sha256": identity, "owner": owner}
                for identity, owner in zip(self.unique_identities, self.owners)
            ],
            "pair_attempts": self.pair_attempts,
            "pair_reconstructed": self.pair_reconstructed,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }

    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.canonical_payload()).encode("utf-8")
        ).hexdigest()


def build_pair_plan(
    views: Mapping[str, ManifestIdentities],
) -> ExactPairPlan:
    """Assign exact shared identities once without moving cell-only inputs.

    Keeping F-only and M-only structures on their source worker is important:
    the frozen relaxation helper has a pre-existing rounded global-cache key.
    This planner therefore cannot introduce tolerance-based reuse across cells.
    Exact identities present in both cells are alternated to balance the saved
    work while still giving each identity exactly one owner.
    """

    if set(views) != set(ROLE_ORDER):
        raise ValueError("exact raw reuse requires one F and one M manifest")
    unique: list[str] = []
    seen: set[str] = set()
    for role in ROLE_ORDER:
        for identity in views[role].reconstructed:
            _require_sha256(identity, f"{role} structure identity")
            if identity not in seen:
                seen.add(identity)
                unique.append(identity)
    membership = {role: set(views[role].reconstructed) for role in ROLE_ORDER}
    owners_list: list[str] = []
    shared_index = 0
    for identity in unique:
        present = tuple(role for role in ROLE_ORDER if identity in membership[role])
        if len(present) == 1:
            owners_list.append(present[0])
        elif len(present) == len(ROLE_ORDER):
            owners_list.append(ROLE_ORDER[shared_index % len(ROLE_ORDER)])
            shared_index += 1
        else:
            raise AssertionError("exact reuse identity has no source cell")
    owners = tuple(owners_list)
    pair_attempts = sum(views[role].total_attempts for role in ROLE_ORDER)
    pair_reconstructed = sum(len(views[role].reconstructed) for role in ROLE_ORDER)
    cache_misses = len(unique)
    cache_hits = pair_reconstructed - cache_misses
    if cache_hits < 0:
        raise AssertionError("exact reuse accounting became negative")
    return ExactPairPlan(
        roles=ROLE_ORDER,
        unique_identities=tuple(unique),
        owners=owners,
        pair_attempts=pair_attempts,
        pair_reconstructed=pair_reconstructed,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )


def map_results_to_identities(
    ordered_identities: tuple[str, ...] | list[str],
    results: Mapping[str, T],
) -> list[T]:
    """Losslessly expand unique results back to every reconstructed row."""

    missing = sorted(set(ordered_identities) - set(results))
    if missing:
        raise ValueError(f"exact relaxation results miss {len(missing)} identities")
    return [results[identity] for identity in ordered_identities]
