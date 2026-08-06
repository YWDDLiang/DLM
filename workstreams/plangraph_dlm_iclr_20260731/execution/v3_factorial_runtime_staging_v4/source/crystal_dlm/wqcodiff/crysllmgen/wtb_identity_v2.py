"""Permutation-safe source identity for Wyckoff tangent-bridge evaluation.

The first WTB-256 execution expanded the in-memory proposal before serializing
it.  ``StratifiedState.to_dict()`` stores orbits in canonical order, so a later
round trip could legitimately permute the primitive atom sequence.  The v1
source contract nevertheless hashed that ordered atom sequence as
"composition", turning a representation-only permutation into a terminal
source failure.

This module keeps the scientifically meaningful invariants fail-closed:

* canonical proposal payload;
* exact topology hash, including species per Wyckoff orbit;
* exact element-count multiset and atom count.

The primitive atom order is still hashed and reported, but it is diagnostic
only.  Parent-model inputs are always built from the same canonicalized state,
so relaxing the legacy comparison does not introduce an ambiguous atom map.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter
from typing import Any, Mapping, Sequence

from ..state import StratifiedState


IDENTITY_SCHEMA = "wtb_permutation_safe_source_identity_v2"
COMPOSITION_SCHEMA = "wtb_composition_multiset_v2"
ORDERED_DIAGNOSTIC_SCHEMA = "wtb_ordered_atomic_numbers_diagnostic_v2"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _atomic_numbers(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 for value in result):
        raise ValueError("atomic numbers must be a non-empty positive sequence")
    return result


def composition_counts(values: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return an exact, permutation-invariant element-count multiset."""

    counts = Counter(_atomic_numbers(values))
    return tuple(sorted((int(number), int(count)) for number, count in counts.items()))


def composition_multiset_signature(values: Sequence[int]) -> str:
    """Hash composition without treating atom permutation as chemistry."""

    return hashlib.sha256(
        _canonical_json(
            {
                "schema": COMPOSITION_SCHEMA,
                "element_counts": composition_counts(values),
            }
        )
    ).hexdigest()


def ordered_atomic_numbers_signature(values: Sequence[int]) -> str:
    """Hash the primitive atom order for diagnostics, never for promotion."""

    return hashlib.sha256(
        _canonical_json(
            {
                "schema": ORDERED_DIAGNOSTIC_SCHEMA,
                "atomic_numbers": _atomic_numbers(values),
            }
        )
    ).hexdigest()


def canonicalize_proposal_payload(
    payload: Mapping[str, Any],
) -> tuple[StratifiedState, dict[str, Any]]:
    """Round-trip one proposal into the sole authoritative storage order."""

    state = StratifiedState.from_dict(dict(payload))
    canonical = state.to_dict(canonical_storage=True)
    return StratifiedState.from_dict(canonical), canonical


def source_signature_v2(
    *,
    proposal_state: Mapping[str, Any],
    topology_hash: str,
    atomic_numbers: Sequence[int],
) -> str:
    """Bind canonical discrete topology and exact composition, not atom order."""

    state, canonical = canonicalize_proposal_payload(proposal_state)
    observed_topology = state.topology_hash()
    if len(str(topology_hash)) != 64 or observed_topology != str(topology_hash):
        raise ValueError("source topology hash is not canonical and exact")
    numbers = _atomic_numbers(atomic_numbers)
    return hashlib.sha256(
        _canonical_json(
            {
                "schema": IDENTITY_SCHEMA,
                "proposal_state": canonical,
                "topology_hash": observed_topology,
                "composition_multiset_signature": (
                    composition_multiset_signature(numbers)
                ),
                "atom_count": len(numbers),
            }
        )
    ).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class SourceIdentityAuditV2:
    """Result of revalidating a persisted source in canonical representation."""

    ok: bool
    topology_identity: bool
    composition_multiset_identity: bool
    atom_count_identity: bool
    canonical_payload_identity: bool
    legacy_order_identity: bool
    legacy_order_mismatch_is_blocking: bool
    stored_atom_count: int
    canonical_atom_count: int
    stored_composition_multiset_signature: str
    canonical_composition_multiset_signature: str
    stored_ordered_atomic_numbers_signature: str
    canonical_ordered_atomic_numbers_signature: str
    canonical_topology_hash: str
    source_signature_v2: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def audit_legacy_source_row(
    row: Mapping[str, Any],
    *,
    canonical_atomic_numbers: Sequence[int],
) -> tuple[StratifiedState, dict[str, Any], SourceIdentityAuditV2]:
    """Validate a v1 source row without blocking on atom-order permutation.

    ``canonical_atomic_numbers`` must come from expanding the canonical state
    returned by :func:`canonicalize_proposal_payload`.  A changed element count,
    atom count, proposal payload, or topology remains a hard failure.
    """

    if row.get("status") != "succeeded":
        raise ValueError("source row is not successful")
    payload = row.get("proposal_state")
    if not isinstance(payload, Mapping):
        raise ValueError("successful source has no proposal state")
    stored_numbers = _atomic_numbers(row.get("atomic_numbers", ()))
    canonical_numbers = _atomic_numbers(canonical_atomic_numbers)
    state, canonical_payload = canonicalize_proposal_payload(payload)
    canonical_topology = state.topology_hash()
    topology_identity = canonical_topology == row.get("proposal_topology_hash")
    stored_multiset = composition_multiset_signature(stored_numbers)
    canonical_multiset = composition_multiset_signature(canonical_numbers)
    composition_identity = stored_multiset == canonical_multiset
    atom_count_identity = (
        len(stored_numbers) == len(canonical_numbers)
        and int(row.get("atom_count", -1)) == len(canonical_numbers)
    )
    canonical_payload_identity = canonical_payload == dict(payload)
    legacy_order_identity = stored_numbers == canonical_numbers
    signature = source_signature_v2(
        proposal_state=canonical_payload,
        topology_hash=canonical_topology,
        atomic_numbers=canonical_numbers,
    )
    ok = bool(
        topology_identity
        and composition_identity
        and atom_count_identity
        and canonical_payload_identity
    )
    audit = SourceIdentityAuditV2(
        ok=ok,
        topology_identity=topology_identity,
        composition_multiset_identity=composition_identity,
        atom_count_identity=atom_count_identity,
        canonical_payload_identity=canonical_payload_identity,
        legacy_order_identity=legacy_order_identity,
        legacy_order_mismatch_is_blocking=False,
        stored_atom_count=len(stored_numbers),
        canonical_atom_count=len(canonical_numbers),
        stored_composition_multiset_signature=stored_multiset,
        canonical_composition_multiset_signature=canonical_multiset,
        stored_ordered_atomic_numbers_signature=(
            ordered_atomic_numbers_signature(stored_numbers)
        ),
        canonical_ordered_atomic_numbers_signature=(
            ordered_atomic_numbers_signature(canonical_numbers)
        ),
        canonical_topology_hash=canonical_topology,
        source_signature_v2=signature,
    )
    if not ok:
        failed = [
            name
            for name, passed in (
                ("topology_identity", topology_identity),
                ("composition_multiset_identity", composition_identity),
                ("atom_count_identity", atom_count_identity),
                ("canonical_payload_identity", canonical_payload_identity),
            )
            if not passed
        ]
        raise ValueError(
            "permutation-safe source identity changed: " + ",".join(failed)
        )
    return state, canonical_payload, audit

