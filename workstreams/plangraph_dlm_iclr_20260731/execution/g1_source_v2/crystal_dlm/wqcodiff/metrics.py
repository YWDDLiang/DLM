"""Frozen StructureMatcher novelty, singleton uniqueness, and prototype metrics."""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .deadlines import WalltimeLimitExceeded, run_with_walltime_limit
from .state import StratifiedState


MATCHER_CONFIGS = {
    "strict": {"ltol": 0.1, "stol": 0.15, "angle_tol": 2.0},
    "standard": {"ltol": 0.2, "stol": 0.3, "angle_tol": 5.0},
    "lenient": {"ltol": 0.3, "stol": 0.5, "angle_tol": 10.0},
}
MATCHER_FIT_TIMEOUT_SECONDS = 5.0


@dataclasses.dataclass(frozen=True, slots=True)
class _BoundedMatcherOutcome:
    value: Any
    status: str
    elapsed_s: float


def _bounded_matcher_call(function: Any) -> _BoundedMatcherOutcome:
    started = time.monotonic()
    try:
        value = run_with_walltime_limit(function, MATCHER_FIT_TIMEOUT_SECONDS)
        status = "ok"
    except WalltimeLimitExceeded:
        value = None
        status = "timeout"
    except Exception:
        # Matcher errors are metric-level failures.  Callers apply the frozen
        # conservative duplicate/non-novel policy instead of dropping the
        # generated attempt or aborting the entire evaluator lane.
        value = None
        status = "error"
    return _BoundedMatcherOutcome(value, status, time.monotonic() - started)


def _record_diagnostic(
    diagnostics: Sequence[collections.Counter[str]] | None,
    indices: Sequence[int],
    key: str,
) -> None:
    if diagnostics is None:
        return
    for index in indices:
        diagnostics[index][key] += 1


def structure_matcher(sensitivity: str = "standard") -> Any:
    from pymatgen.analysis.structure_matcher import StructureMatcher

    try:
        values = MATCHER_CONFIGS[sensitivity]
    except KeyError as exc:
        raise ValueError(f"unknown matcher sensitivity: {sensitivity}") from exc
    return StructureMatcher(
        **values,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
    )


def matcher_contract_hash() -> str:
    payload = {
        "configs": MATCHER_CONFIGS,
        "primitive_cell": True,
        "scale": True,
        "attempt_supercell": False,
        "uniqueness": "duplicate_component_size_equals_one",
        "per_fit_timeout_seconds": MATCHER_FIT_TIMEOUT_SECONDS,
        "timeout_or_error_policy": {
            "duplicate": "conservative_duplicate",
            "full_novelty": "conservative_non_novel",
            "anonymous_prototype": "conservative_non_novel",
            "substitution_aware": "conservative_substitution_derived",
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def protostructure_key(structure: Any) -> str:
    """Species-Wyckoff multiset key redetected at the primary tolerance."""

    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    analyzer = SpacegroupAnalyzer(structure, symprec=0.01, angle_tolerance=5.0)
    conventional = analyzer.get_conventional_standard_structure(
        international_monoclinic=True
    )
    symmetrized = SpacegroupAnalyzer(
        conventional, symprec=0.01, angle_tolerance=5.0
    ).get_symmetrized_structure()
    orbits = []
    for indices, symbol in zip(
        symmetrized.equivalent_indices,
        symmetrized.wyckoff_symbols,
    ):
        species = conventional[indices[0]].specie.symbol
        orbits.append((str(symbol), species, len(indices)))
    payload = {
        "space_group": int(analyzer.get_space_group_number()),
        "orbits": sorted(orbits),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def state_protostructure_key(state_payload: Mapping[str, Any] | None, structure: Any) -> str:
    if state_payload is not None:
        return StratifiedState.from_dict(dict(state_payload)).topology_hash()
    return protostructure_key(structure)


def _composition_group(structure: Any) -> str:
    return str(structure.composition.reduced_formula)


def _anonymous_group(structure: Any) -> tuple[Any, ...]:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    reduced = structure.composition.reduced_composition
    amounts = sorted(float(value) for value in reduced.get_el_amt_dict().values())
    try:
        space_group = int(
            SpacegroupAnalyzer(
                structure,
                symprec=0.1,
                angle_tolerance=5.0,
            ).get_space_group_number()
        )
    except Exception:
        space_group = 0
    return (
        *(round(value, 8) for value in amounts),
        "sites",
        len(structure),
        "sg",
        space_group,
    )


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        left, right = self.find(first), self.find(second)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def duplicate_components(
    attempt_ids: Sequence[str],
    structures: Sequence[Any],
    *,
    sensitivity: str,
    diagnostics: Sequence[collections.Counter[str]] | None = None,
) -> tuple[tuple[str, ...], tuple[bool, ...]]:
    if len(attempt_ids) != len(structures):
        raise ValueError("attempt/structure length mismatch")
    if diagnostics is not None and len(diagnostics) != len(structures):
        raise ValueError("matcher diagnostics/structure length mismatch")
    matcher = structure_matcher(sensitivity)
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for index, structure in enumerate(structures):
        groups[_composition_group(structure)].append(index)
    union = _UnionFind(len(structures))
    for indices in groups.values():
        for offset, first in enumerate(indices):
            for second in indices[offset + 1 :]:
                outcome = _bounded_matcher_call(
                    lambda first=first, second=second: matcher.fit(
                        structures[first], structures[second]
                    )
                )
                matched = bool(outcome.value) if outcome.status == "ok" else True
                if outcome.status != "ok":
                    _record_diagnostic(
                        diagnostics,
                        (first, second),
                        f"duplicate_{sensitivity}_{outcome.status}",
                    )
                if matched:
                    union.union(first, second)
    members: dict[int, list[int]] = collections.defaultdict(list)
    for index in range(len(structures)):
        members[union.find(index)].append(index)
    cluster_ids = [""] * len(structures)
    unique = [False] * len(structures)
    for indices in members.values():
        labels = sorted(attempt_ids[index] for index in indices)
        cluster_id = "dup-" + hashlib.sha256("\n".join(labels).encode("utf-8")).hexdigest()[:20]
        for index in indices:
            cluster_ids[index] = cluster_id
            unique[index] = len(indices) == 1
    return tuple(cluster_ids), tuple(unique)


def full_structure_novelty(
    structures: Sequence[Any],
    train_structures: Sequence[Any],
    *,
    sensitivity: str,
    diagnostics: Sequence[collections.Counter[str]] | None = None,
) -> tuple[bool, ...]:
    if diagnostics is not None and len(diagnostics) != len(structures):
        raise ValueError("matcher diagnostics/structure length mismatch")
    matcher = structure_matcher(sensitivity)
    groups: dict[str, list[Any]] = collections.defaultdict(list)
    for structure in train_structures:
        groups[_composition_group(structure)].append(structure)
    results: list[bool] = []
    for index, structure in enumerate(structures):
        novel = True
        for reference in groups[_composition_group(structure)]:
            outcome = _bounded_matcher_call(
                lambda structure=structure, reference=reference: matcher.fit(
                    structure, reference
                )
            )
            if outcome.status != "ok":
                _record_diagnostic(
                    diagnostics,
                    (index,),
                    f"full_novelty_{sensitivity}_{outcome.status}",
                )
                novel = False
                break
            if bool(outcome.value):
                novel = False
                break
        results.append(novel)
    return tuple(results)


def anonymous_prototype_novelty(
    structures: Sequence[Any],
    train_structures: Sequence[Any],
    *,
    diagnostics: Sequence[collections.Counter[str]] | None = None,
) -> tuple[bool, ...]:
    if diagnostics is not None and len(diagnostics) != len(structures):
        raise ValueError("matcher diagnostics/structure length mismatch")
    matcher = structure_matcher("standard")
    groups: dict[tuple[Any, ...], list[Any]] = collections.defaultdict(list)
    for structure in train_structures:
        groups[_anonymous_group(structure)].append(structure)
    results: list[bool] = []
    for index, structure in enumerate(structures):
        novel = True
        for reference in groups[_anonymous_group(structure)]:
            outcome = _bounded_matcher_call(
                lambda reference=reference, structure=structure: matcher.fit_anonymous(
                    reference, structure
                )
            )
            if outcome.status != "ok":
                _record_diagnostic(
                    diagnostics,
                    (index,),
                    f"anonymous_prototype_{outcome.status}",
                )
                novel = False
                break
            if bool(outcome.value):
                novel = False
                break
        results.append(novel)
    return tuple(results)


def substitution_aware_novelty(
    structures: Sequence[Any],
    train_structures: Sequence[Any],
    *,
    full_novelty: Sequence[bool],
    diagnostics: Sequence[collections.Counter[str]] | None = None,
) -> tuple[bool, ...]:
    """Mark likely substitutions of a train prototype as non-novel.

    Mapping probability is the product of frozen pymatgen conditional
    substitution probabilities and must exceed the registered 1e-3 threshold.
    """

    from pymatgen.analysis.structure_prediction.substitution_probability import (
        SubstitutionPredictor,
    )

    predictor = SubstitutionPredictor(threshold=1.0e-3)
    if diagnostics is not None and len(diagnostics) != len(structures):
        raise ValueError("matcher diagnostics/structure length mismatch")
    matcher = structure_matcher("standard")
    groups: dict[tuple[Any, ...], list[Any]] = collections.defaultdict(list)
    for structure in train_structures:
        groups[_anonymous_group(structure)].append(structure)
    results: list[bool] = []
    for structure, is_full_novel in zip(structures, full_novelty):
        if not is_full_novel:
            results.append(False)
            continue
        substitution_derived = False
        for reference in groups[_anonymous_group(structure)]:
            outcome = _bounded_matcher_call(
                lambda reference=reference, structure=structure: matcher.get_all_anonymous_mappings(
                    reference, structure
                )
            )
            if outcome.status != "ok":
                _record_diagnostic(
                    diagnostics,
                    (len(results),),
                    f"substitution_aware_{outcome.status}",
                )
                substitution_derived = True
                break
            mappings = outcome.value
            if not mappings:
                continue
            for mapping in mappings:
                probability = 1.0
                try:
                    for source, target in mapping.items():
                        probability *= float(predictor.p.cond_prob(source, target))
                except Exception:
                    probability = 0.0
                if probability > 1.0e-3:
                    substitution_derived = True
                    break
            if substitution_derived:
                break
        results.append(not substitution_derived)
    return tuple(results)


@dataclasses.dataclass(frozen=True, slots=True)
class RelationalMetricRow:
    duplicate_cluster: Mapping[str, str]
    unique: Mapping[str, bool]
    full_novel: Mapping[str, bool]
    anonymous_prototype_novel: bool
    protostructure_novel: bool
    substitution_aware_novel: bool
    matcher_diagnostics: Mapping[str, int]


def compute_relational_metrics(
    attempt_ids: Sequence[str],
    structures: Sequence[Any],
    state_payloads: Sequence[Mapping[str, Any] | None],
    *,
    train_structures: Sequence[Any],
    train_protostructure_keys: frozenset[str],
) -> tuple[RelationalMetricRow, ...]:
    if not (len(attempt_ids) == len(structures) == len(state_payloads)):
        raise ValueError("relational metric input lengths differ")
    cluster_by_matcher: dict[str, tuple[str, ...]] = {}
    unique_by_matcher: dict[str, tuple[bool, ...]] = {}
    novelty_by_matcher: dict[str, tuple[bool, ...]] = {}
    matcher_diagnostics = [collections.Counter() for _ in structures]
    for sensitivity in MATCHER_CONFIGS:
        clusters, unique = duplicate_components(
            attempt_ids,
            structures,
            sensitivity=sensitivity,
            diagnostics=matcher_diagnostics,
        )
        cluster_by_matcher[sensitivity] = clusters
        unique_by_matcher[sensitivity] = unique
        novelty_by_matcher[sensitivity] = full_structure_novelty(
            structures,
            train_structures,
            sensitivity=sensitivity,
            diagnostics=matcher_diagnostics,
        )
    prototype_novel = anonymous_prototype_novelty(
        structures,
        train_structures,
        diagnostics=matcher_diagnostics,
    )
    proto_keys = [
        state_protostructure_key(payload, structure)
        for payload, structure in zip(state_payloads, structures)
    ]
    proto_novel = [value not in train_protostructure_keys for value in proto_keys]
    substitution_novel = substitution_aware_novelty(
        structures,
        train_structures,
        full_novelty=novelty_by_matcher["standard"],
        diagnostics=matcher_diagnostics,
    )
    return tuple(
        RelationalMetricRow(
            duplicate_cluster={
                sensitivity: cluster_by_matcher[sensitivity][index]
                for sensitivity in MATCHER_CONFIGS
            },
            unique={
                sensitivity: unique_by_matcher[sensitivity][index]
                for sensitivity in MATCHER_CONFIGS
            },
            full_novel={
                sensitivity: novelty_by_matcher[sensitivity][index]
                for sensitivity in MATCHER_CONFIGS
            },
            anonymous_prototype_novel=prototype_novel[index],
            protostructure_novel=proto_novel[index],
            substitution_aware_novel=substitution_novel[index],
            matcher_diagnostics=dict(sorted(matcher_diagnostics[index].items())),
        )
        for index in range(len(structures))
    )
