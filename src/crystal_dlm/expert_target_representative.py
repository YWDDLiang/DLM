"""Certified whole-target integer translations and species assignments for training.

The objective is periodic fractional squared representation distance in integer
bin units. It is not Cartesian displacement, an atom trajectory, or an energy.
Original records and physics-label provenance are retained without modification.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re

import numpy as np
from scipy.optimize import linear_sum_assignment


PERIOD = 100
SCHEMA = "expert_whole_target_representative_v1"


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value):
    return sha256_bytes(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8"))


def parse_body(tokens):
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("a token body must be a nonempty list")
    match = re.fullmatch(r"<N_(\d{3})>", tokens[0])
    if not match:
        raise ValueError("missing native count token")
    count = int(match[1])
    if not 1 <= count <= 20 or len(tokens) != 7 + 4 * count:
        raise ValueError("body violates the native 7+4N layout")
    for position, family in enumerate(("LA", "LB", "LC", "AA", "AB", "AG"), 1):
        if not re.fullmatch(rf"<{family}_(\d{{3}})>", tokens[position]):
            raise ValueError(f"wrong lattice token family at {position}")
    species, raw_bins = [], []
    for site in range(count):
        start = 7 + 4 * site
        element = re.fullmatch(r"<E_([A-Z][a-z]?)>", tokens[start])
        if not element:
            raise ValueError(f"wrong species token at site {site}")
        species.append(element[1])
        coordinates = []
        for axis, family in enumerate("XYZ"):
            coordinate = re.fullmatch(rf"<{family}_(\d{{3}})>", tokens[start + 1 + axis])
            if not coordinate or not 0 <= int(coordinate[1]) <= PERIOD:
                raise ValueError(f"wrong coordinate token at site {site}, axis {axis}")
            coordinates.append(int(coordinate[1]))
        raw_bins.append(coordinates)
    raw_bins = np.asarray(raw_bins, dtype=np.int64)
    return {
        "count": count, "species": species, "raw_bins": raw_bins,
        "bins": raw_bins % PERIOD, "alias_count": int((raw_bins == PERIOD).sum()),
    }


def render_coordinates(template, bins, *, alias_mask=None):
    result = list(template)
    for site, coordinates in enumerate(bins):
        for axis, value in enumerate(coordinates):
            value = int(value)
            if alias_mask is not None and bool(alias_mask[site, axis]):
                if value != 0:
                    raise AssertionError("inverse alias restoration did not recover bin zero")
                value = PERIOD
            result[8 + 4 * site + axis] = f"<{'XYZ'[axis]}_{value:03d}>"
    return result


def squared_distances(left, right):
    delta = np.abs(left[:, None, :] - right[None, :, :])
    delta = np.minimum(delta, PERIOD - delta)
    return np.sum(delta * delta, axis=-1, dtype=np.int64)


def paired_cost(left, right):
    delta = np.abs(left - right)
    delta = np.minimum(delta, PERIOD - delta)
    return int(np.sum(delta * delta, dtype=np.int64))


def minimum_assignment_cost(cost):
    if not len(cost):
        return 0
    rows, columns = linear_sum_assignment(cost)
    return int(cost[rows, columns].sum(dtype=np.int64))


def lexicographic_optimal_assignment(cost):
    """Smallest column-index tuple among assignments with minimum integer cost.

    Rows are in OLD slot order; callers sort columns by shifted XYZ and original
    target index. Repeated constrained Hungarian calls implement an exact tie
    rule without floating epsilon perturbations or solver-dependent ties.
    """
    count = len(cost)
    columns = list(range(count))
    remaining_cost = minimum_assignment_cost(cost)
    selected = []
    for row in range(count):
        for column in columns:
            rest = [value for value in columns if value != column]
            future = cost[np.ix_(list(range(row + 1, count)), rest)]
            future_cost = minimum_assignment_cost(future)
            if int(cost[row, column]) + future_cost == remaining_cost:
                selected.append(column)
                columns = rest
                remaining_cost = future_cost
                break
        else:
            raise AssertionError("no lexicographic optimal assignment continuation")
    if remaining_cost != 0:
        raise AssertionError("assignment cost accounting failed")
    return selected


def assignment_for_shift(old, target, species, shift, *, resolve_ties):
    shifted = (target + np.asarray(shift, dtype=np.int64)) % PERIOD
    mapping = [-1] * len(species)
    total = 0
    for element in sorted(set(species)):
        old_slots = [i for i, symbol in enumerate(species) if symbol == element]
        target_slots = sorted(
            old_slots, key=lambda j: (tuple(map(int, shifted[j])), j)
        )
        cost = squared_distances(old[old_slots], shifted[target_slots])
        total += minimum_assignment_cost(cost)
        if resolve_ties:
            chosen = lexicographic_optimal_assignment(cost)
            for row, column in enumerate(chosen):
                mapping[old_slots[row]] = target_slots[column]
    if not resolve_ties:
        return total
    aligned = shifted[mapping]
    if paired_cost(old, aligned) != total:
        raise AssertionError("assignment reconstruction does not attain its cost")
    return total, aligned, mapping


def align_target(old, target, species):
    shifts = {(0, 0, 0)}
    for i, element in enumerate(species):
        for j, other in enumerate(species):
            if element == other:
                shifts.add(tuple(map(int, (old[i] - target[j]) % PERIOD)))
    costs = [
        (assignment_for_shift(old, target, species, shift, resolve_ties=False), shift)
        for shift in sorted(shifts)
    ]
    best_cost = min(cost for cost, _ in costs)
    minimizers = [shift for cost, shift in costs if cost == best_cost]
    resolved = []
    for shift in minimizers:
        cost, aligned, mapping = assignment_for_shift(
            old, target, species, shift, resolve_ties=True
        )
        key = (cost, tuple(map(int, aligned.reshape(-1))), shift, tuple(mapping))
        resolved.append((key, aligned, mapping))
    key, aligned, mapping = min(resolved, key=lambda item: item[0])
    zero_cost, zero_aligned, zero_mapping = assignment_for_shift(
        old, target, species, (0, 0, 0), resolve_ties=True
    )
    return {
        "aligned": aligned, "permutation": mapping, "shift": list(key[2]),
        "cost": best_cost, "candidate_count": len(shifts),
        "min_cost_shift_count": len(minimizers),
        "zero_shift_cost": zero_cost, "zero_shift_aligned": zero_aligned,
        "zero_shift_permutation": zero_mapping,
    }


def element_xyz_multiset(species, bins):
    return Counter((symbol, *map(int, row)) for symbol, row in zip(species, bins))


def training_representative(record, tokenizer):
    """Return a certified content-training representative; retain original labels."""
    if (record.get('task') != 'S' or not record.get('content_supervision')
            or record['action']['mode'] not in ('full_cell', 'all_xyz')):
        return record
    vocabulary = tokenizer.get_vocab()
    inverse = {int(value): token for token, value in vocabulary.items()}
    old_tokens = [inverse[int(t)] for t in record['old_body']]
    target_tokens = [inverse[int(t)] for t in record['target_body']]
    old, target = parse_body(old_tokens), parse_body(target_tokens)
    if old['count'] != target['count'] or old['species'] != target['species']:
        raise ValueError('training representative cannot change ordered composition')
    aligned = align_target(old['bins'], target['bins'], old['species'])
    tokens = render_coordinates(target_tokens, aligned['aligned'])
    old_canonical = render_coordinates(old_tokens, old['bins'])
    target_canonical = render_coordinates(target_tokens, target['bins'])
    active = set(record['action']['positions'])
    for body in (target_canonical, tokens):
        if any(a != b and i not in active for i, (a, b) in enumerate(zip(old_canonical, body))):
            raise ValueError('whole-target representative escapes the original action')
    permutation = aligned['permutation']
    untranslated = (aligned['aligned'] - np.asarray(aligned['shift'], dtype=np.int64)) % PERIOD
    recovered = np.empty_like(target['bins'])
    for slot, original_slot in enumerate(permutation):
        recovered[original_slot] = untranslated[slot]
    restored = render_coordinates(target_tokens, recovered, alias_mask=target['raw_bins'] == PERIOD)
    if (sorted(permutation) != list(range(old['count'])) or restored != target_tokens
            or element_xyz_multiset(old['species'], untranslated) != element_xyz_multiset(target['species'], target['bins'])
            or tokens[:7] != target_tokens[:7]
            or any(tokens[7+4*i] != old_tokens[7+4*i] for i in range(old['count']))):
        raise ValueError('integer-grid representative failed exact equivalence checks')
    body = [int(vocabulary[t]) for t in tokens]
    result = dict(record)
    result['training_target_body'] = body
    result['training_representative_zero_edit'] = tokens == old_canonical
    result['training_target_certificate'] = {
        'schema': SCHEMA, 'scope': 'S full_cell/all_xyz content views only',
        'original_old_body_sha256': canonical_hash(record['old_body']),
        'original_target_body_sha256': canonical_hash(record['target_body']),
        'training_target_body_sha256': canonical_hash(body),
        'original_target_physics_id': record.get('target_physics_id'),
        'shift_bins_xyz': aligned['shift'], 'old_slot_to_original_target_slot': permutation,
        'inverse_recovers_original_tokens': True,
        'physical_label_modified': False, 'independently_physics_evaluated': False,
        'original_coordinate_token_changes': int((old['bins'] != target['bins']).sum()),
        'training_coordinate_token_changes': int((old['bins'] != aligned['aligned']).sum()),
        'representation_cost': 'sum of squared minimum-image fractional-coordinate bin distances',
        'original_representation_cost': paired_cost(old['bins'], target['bins']),
        'training_representation_cost': aligned['cost'],
    }
    return result
