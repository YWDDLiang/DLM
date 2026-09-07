"""Compile executable crystal edits with separate geometry and energy evidence.

This module never infers a physical negative from a missing teacher. Its
geometry certificate is a training-data admission rule, not an inference mask
or a replacement for the frozen benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import itertools
import json
import math
import copy
import random
from pathlib import Path
from typing import Mapping

import numpy as np

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens, parse_dynamic_answer, arrays_to_structure
from crystal_dlm.fixed_slot import FixedSlotConfig, Z_TO_SYMBOL
from crystal_dlm.terminal_energy_consistency import COMMON_RELAXATION_PROTOCOL, TERMINAL_VERIFICATION_PROTOCOL


SCHEMA = 'expert_crystal_edit_v1'
GEOMETRY_PROTOCOL = {'minimum_distance_A': .5, 'minimum_volume_A3': .1,
                     'image_bound': 'reciprocal_column_norm_complete_for_contact_cutoff',
                     'max_pair_images': 4_000_000, 'tolerance_A': 1e-8}


def read_rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_rows(path, rows):
    with Path(path).open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def composition_key(plan_or_species):
    if isinstance(plan_or_species, Mapping):
        if len(plan_or_species['elements']) != len(plan_or_species['counts']):
            raise ValueError('composition elements/counts lengths differ')
        counts = Counter()
        for element, count in zip(plan_or_species['elements'], plan_or_species['counts']):
            if type(count) is not int or count <= 0:
                raise ValueError('composition counts must be positive integers')
            counts[str(element)] += count
        if sum(counts.values()) != int(plan_or_species['N']):
            raise ValueError('composition does not match N')
    else:
        counts = Counter(map(str, plan_or_species))
    if not counts:
        raise ValueError('empty composition')
    divisor = math.gcd(*counts.values())
    return '|'.join(f'{key}:{counts[key] // divisor}' for key in sorted(counts))


def lattice_from_parameters(lengths, angles):
    lengths, angles = np.asarray(lengths, dtype=float), np.asarray(angles, dtype=float)
    if lengths.shape != (3,) or angles.shape != (3,) or not np.isfinite(np.r_[lengths, angles]).all():
        raise ValueError('invalid lattice values')
    if (lengths <= 0).any() or (angles <= 0).any() or (angles >= 180).any():
        raise ValueError('lattice parameters outside physical domain')
    alpha, beta, gamma = np.deg2rad(angles)
    a, b, c = lengths
    sg = math.sin(gamma)
    cx, cy = c * math.cos(beta), c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sg
    square = c * c - cx * cx - cy * cy
    if sg <= 1e-10 or square <= 1e-10:
        raise ValueError('nonpositive lattice metric')
    return np.asarray([[a, 0., 0.], [b * math.cos(gamma), b * sg, 0.], [cx, cy, math.sqrt(square)]])


def certify_geometry(arrays, *, cutoff=.5, max_pair_images=4_000_000):
    """Certify all periodic contacts below cutoff with an adaptive exact box.

    For a wrapped fractional difference d, any contact r=(d+n)L shorter
    than cutoff has |n_i| <= .5 + cutoff*||L^-1[:,i]||. Bounded chunks avoid
    a large all-image tensor; exceeding the explicit cap means unverified.
    """
    try:
        lattice = lattice_from_parameters(arrays['lengths'], arrays['angles'])
        coords = np.asarray(arrays['frac_coords'], dtype=float)
        n = len(arrays['species'])
        if not 1 <= n <= 20 or coords.shape != (n, 3) or not np.isfinite(coords).all():
            raise ValueError('invalid coordinate dimensions')
        volume = float(abs(np.linalg.det(lattice)))
        if volume < GEOMETRY_PROTOCOL['minimum_volume_A3']:
            raise ValueError('volume below target admission')
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
        return {'valid': False, 'certified': True, 'reason': str(error)}
    radii = np.ceil(.5 + cutoff * np.linalg.norm(np.linalg.inv(lattice), axis=0) + 1e-12).astype(int)
    count = math.prod(int(2 * radius + 1) for radius in radii)
    if n * n * count > max_pair_images:
        return {'valid': None, 'certified': False, 'reason': 'periodic_certificate_budget',
                'images': count, 'volume_A3': volume}
    delta = coords[:, None] - coords[None, :]
    delta -= np.round(delta)
    shifts = itertools.product(*(range(-int(r), int(r) + 1) for r in radii))
    minimum = float('inf')
    while True:
        block = list(itertools.islice(shifts, 256))
        if not block:
            break
        offsets = np.asarray(block, dtype=float)
        values = np.linalg.norm((delta[:, :, None, :] + offsets) @ lattice, axis=-1)
        zero = np.flatnonzero((offsets == 0).all(-1))
        if len(zero):
            values[np.arange(n), np.arange(n), int(zero[0])] = np.inf
        minimum = min(minimum, float(values.min()))
        if minimum < cutoff - GEOMETRY_PROTOCOL['tolerance_A']:
            return {'valid': False, 'certified': True, 'reason': 'periodic_short_contact',
                    'witness_distance_A': minimum, 'images': count, 'volume_A3': volume}
    return {'valid': True, 'certified': True, 'reason': None, 'images': count,
            'searched_minimum_A': minimum, 'volume_A3': volume}


def quantize_arrays(arrays, vocabulary):
    tokens, diagnostics = arrays_to_dynamic_tokens(
        arrays['lengths'], arrays['angles'], arrays['species'], arrays['frac_coords'], config=FixedSlotConfig())
    if diagnostics.length_clips or diagnostics.angle_clips or diagnostics.coord_clips:
        raise ValueError('expert target clipped by B0 quantization')
    tokens = [token.replace('_100>', '_000>') if token.startswith(('<X_', '<Y_', '<Z_')) else token
              for token in tokens]
    try:
        ids = [int(vocabulary[token]) for token in tokens]
    except KeyError as error:
        raise ValueError('target token is outside the preserved B0 vocabulary') from error
    decoded = parse_dynamic_answer(''.join(tokens), strict=True)
    return ids, decoded, asdict(diagnostics)


def decode_body(ids, inverse_vocabulary):
    return parse_dynamic_answer(''.join(inverse_vocabulary[int(i)] for i in ids), strict=True)


def canonical_body(ids, vocabulary, inverse_vocabulary=None):
    inverse = inverse_vocabulary or {int(value): key for key, value in vocabulary.items()}
    result = []
    for value in ids:
        token = inverse[int(value)]
        if token.startswith(('<X_', '<Y_', '<Z_')) and token.endswith('_100>'):
            token = token[:-4] + '000>'
        result.append(int(vocabulary[token]))
    return result


def numeric_positions(n):
    return list(range(1, 7)) + [8 + 4 * i + axis for i in range(n) for axis in range(3)]


def fixed_composition(old, target, n):
    positions = [0] + [7 + 4 * i for i in range(n)]
    return len(old) == len(target) == 7 + 4 * n and all(old[i] == target[i] for i in positions)


def full_action(old, target, n):
    if not fixed_composition(old, target, n):
        raise ValueError('expert edit changed exact N or ordered species slots')
    if old == target:
        return {'mode': 'identity', 'sites': [], 'positions': []}
    if old[1:7] == target[1:7]:
        return {'mode': 'all_xyz', 'sites': list(range(n)), 'positions': numeric_positions(n)[6:]}
    return {'mode': 'full_cell', 'sites': list(range(n)), 'positions': numeric_positions(n)}


def refined_arrays(path):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True)
    expected_dims = {'num_atoms': 2, 'frac_coords': 3, 'atom_types': 2, 'lengths': 3, 'angles': 3}
    if any(payload[key].ndim != dims or payload[key].shape[0] != 1 for key, dims in expected_dims.items()):
        raise ValueError('expert refinement requires exactly one evaluation and registered tensor ranks')
    if (payload['sample_indices'].ndim != 1
            or not torch.equal(payload['num_atoms'], payload['num_atoms'].long())
            or not torch.equal(payload['sample_indices'], payload['sample_indices'].long())
            or not torch.equal(payload['atom_types'], payload['atom_types'].long())):
        raise ValueError('refiner indices, counts and species must be exact integers')
    counts = payload['num_atoms'][0].long().tolist()
    indices = payload['sample_indices'].long().tolist()
    if len(counts) != len(indices) or len(indices) != len(set(indices)):
        raise ValueError('refined tensor has inconsistent global sample indices')
    if (any(not 1 <= n <= 20 for n in counts) or any(index < 0 for index in indices)
            or payload['lengths'].shape != (1, len(counts), 3)
            or payload['angles'].shape != (1, len(counts), 3)
            or payload['frac_coords'].shape != (1, sum(counts), 3)
            or payload['atom_types'].shape != (1, sum(counts))):
        raise ValueError('refiner tensor atom/lattice accounting differs')
    position, result = 0, {}
    for row, (index, n) in enumerate(zip(indices, counts)):
        atom_types = payload['atom_types'][0, position:position + n].long().tolist()
        result[int(index)] = {
            'lengths': payload['lengths'][0, row].double().tolist(),
            'angles': payload['angles'][0, row].double().tolist(),
            'species': [Z_TO_SYMBOL[int(z)] for z in atom_types],
            'frac_coords': payload['frac_coords'][0, position:position + n].double().tolist(),
        }
        position += n
    if position != payload['frac_coords'].shape[1]:
        raise ValueError('refined tensor atom accounting differs')
    return result


def align_complete_target(old_arrays, target_arrays):
    """Whole-cell rewrite: order target species into immutable old slots.

    This is a representation correspondence, not a claimed atom trajectory.
    Local physics edits require a separately verified alignment/hybrid.
    """
    if Counter(old_arrays['species']) != Counter(target_arrays['species']):
        raise ValueError('refiner changed fixed composition')
    buckets = defaultdict(list)
    for index, element in enumerate(target_arrays['species']):
        buckets[element].append(index)
    order = [buckets[element].pop(0) for element in old_arrays['species']]
    result = dict(target_arrays, species=list(old_arrays['species']),
                  frac_coords=[target_arrays['frac_coords'][i] for i in order])
    return result, order


def physics_input(record_id, group, source_idx, split, endpoint, arrays=None, body=None):
    structure = None
    if arrays is not None:
        try:
            structure = arrays_to_structure(arrays).as_dict()
        except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError):
            pass  # Keep an invalid raw body for explicit invalid_raw accounting.
    return {'trajectory_id': record_id, 'group_id': group, 'source_row_idx': source_idx,
            'source_split': split, 'purpose': 'expert_edit', 'endpoint': endpoint,
            'success': arrays is not None or bool(body), 'structure': structure, 'body': body}


def prepare_collection(manifest_path, heldout_cohort, tokenizer, output, *, limit=None, min_index=0):
    if min_index < 0 or (limit is not None and limit <= min_index):
        raise ValueError('invalid disjoint request interval')
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['purpose'] != 'train':
        raise ValueError('expert collection must be registered as training before generation')
    expected_cohort = manifest['evaluation']['panel_files']['cohort/cohort.jsonl']['sha256']
    if sha256(heldout_cohort) != expected_cohort:
        raise ValueError('heldout cohort changed after registration')
    forbidden = set()
    for row in read_rows(heldout_cohort):
        if row.get('plan_state'):
            try:
                forbidden.add(composition_key(row['plan_state']))
            except (ValueError, KeyError, TypeError):
                if row.get('body_eligible'):
                    raise ValueError('eligible heldout Plan has invalid composition')
    vocabulary = tokenizer.get_vocab()
    inverse = {int(v): k for k, v in vocabulary.items()}
    root, output = Path(manifest['run_root']), Path(output)
    selected_components = [c for c in manifest['components']
                           if c['sample_index_offset'] + c['requests'] > min_index
                           and (limit is None or c['sample_index_offset'] < limit)]
    if not selected_components or any(not (root / c['output_dir'] / '_SUCCESS').is_file() for c in selected_components):
        raise ValueError('required collection components have not completed')
    output.mkdir(parents=True, exist_ok=False)
    pending, old_inputs, target_inputs, rejected, provenance = [], [], [], [], []
    attempted = 0
    for component in selected_components:
        directory = root / component['output_dir']
        planner_dir, body_dir = directory / 'planner', directory / 'body'
        if json.loads((planner_dir / 'run_config.json').read_text()).get('purpose') != 'train':
            raise ValueError('actual planner output is not training provenance')
        plans = {int(row['sample_idx']): row for row in read_rows(planner_dir / 'plans_for_dlm.jsonl')}
        parents = read_rows(body_dir / 'raw_generations.jsonl')
        if len(parents) != component['requests'] or len(plans) != component['requests']:
            raise ValueError('collection lost requested rows')
        expected_ids = set(range(component['sample_index_offset'],
                                 component['sample_index_offset'] + component['requests']))
        if set(plans) != expected_ids or {int(row['sample_idx']) for row in parents} != expected_ids:
            raise ValueError('collection changed its registered request identities')
        refinement = json.loads((directory / 'teacher_refine/refinement_metrics.json').read_text())
        targets = refined_arrays(refinement['output_file'])
        if not set(targets).issubset(expected_ids) or refinement.get('num_evals') != 1 or refinement.get('diff_steps') != 800:
            raise ValueError('expert refinement changed its registered endpoint or request set')
        provenance.append({'component': component['id'], 'parents_sha256': sha256(body_dir / 'raw_generations.jsonl'),
                           'plans_sha256': sha256(planner_dir / 'plans_for_dlm.jsonl'),
                           'refined_sha256': sha256(refinement['output_file'])})
        for parent in parents:
            index = int(parent['sample_idx'])
            if index < min_index or (limit is not None and index >= limit):
                continue
            attempted += 1
            plan = plans[index]
            if parent.get('planner_record') != plan:
                raise ValueError('body was generated from different rich Plan metadata')
            if plan.get('body_eligible') and (
                    parent.get('body_prompt') != plan.get('body_prompt')
                    or parent.get('body_prompt_sha256') != hashlib.sha256(plan['body_prompt'].encode()).hexdigest()):
                raise ValueError('body used a different actual rich prompt')
            ancestor = f"{manifest['source_identity']['commit']}:{component['id']}:{index}"
            try:
                if not plan.get('body_eligible') or plan.get('source_split') != 'train':
                    raise ValueError('planner_failure_or_nontraining_source')
                key = composition_key(plan['plan_state'])
                if key in forbidden:
                    raise ValueError('evaluation_composition_excluded')
                original_ids = list(map(int, parent.get('raw_body_token_ids') or []))
                ids = canonical_body(original_ids, vocabulary, inverse)
                old_arrays = decode_body(ids, inverse)
                expected_counts = Counter()
                for element, count in zip(plan['plan_state']['elements'], plan['plan_state']['counts']):
                    expected_counts[element] += count
                if (Counter(old_arrays['species']) != expected_counts
                        or old_arrays['num_atoms'] != plan['plan_state']['N']):
                    raise ValueError('old body differs from actual Plan composition')
                split = 'dev' if int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 8 == 0 else 'train'
                old_id = 'expert-old:' + ancestor
                old_text = ''.join(inverse[i] for i in ids)
                old_inputs.append(physics_input(old_id, ancestor, index, split, 'native', old_arrays, old_text))
                row = {'schema': SCHEMA, 'ancestor_id': ancestor, 'source_kind': 'current_B0_full_rich',
                       'source_split': split, 'source_row_idx': index, 'composition_key': key,
                       'plan_state': plan['plan_state'], 'prompt': plan['body_prompt'], 'num_atoms': len(old_arrays['species']),
                       'old_body': ids, 'old_physics_id': old_id, 'old_geometry': certify_geometry(old_arrays),
                       'original_source_body': original_ids, 'periodic_aliases_canonicalized': True,
                       'teacher_available': False, 'target_physics_id': None}
                if index in targets:
                    try:
                        aligned, order = align_complete_target(old_arrays, targets[index])
                        row['prequant_geometry'] = certify_geometry(aligned)
                        target_ids, decoded, diagnostics = quantize_arrays(aligned, vocabulary)
                        certificate = certify_geometry(decoded)
                        row.update(target_geometry=certificate, quantization=diagnostics,
                                   alignment={'kind': 'whole_cell_species_representation', 'target_order': order})
                        if certificate['valid'] is True and certificate['certified']:
                            action = full_action(ids, target_ids, row['num_atoms'])
                            target_id = 'expert-target:' + ancestor
                            target_inputs.append(physics_input(target_id, ancestor, index, split, 'expert_quantized',
                                                               decoded, ''.join(inverse[i] for i in target_ids)))
                            row.update(teacher_available=True, target_body=target_ids, action=action,
                                       target_physics_id=target_id)
                    except (ValueError, KeyError, TypeError) as error:
                        row['teacher_failure'] = str(error)
                pending.append(row)
            except (ValueError, KeyError, TypeError) as error:
                rejected.append({'ancestor_id': ancestor, 'source_row_idx': index, 'reason': str(error)})
    write_rows(output / 'pairs_pending.jsonl', pending)
    write_rows(output / 'old_inputs.jsonl', old_inputs)
    write_rows(output / 'target_inputs.jsonl', target_inputs)
    write_rows(output / 'all_inputs.jsonl', old_inputs + target_inputs)
    write_rows(output / 'rejected.jsonl', rejected)
    report = {'schema': SCHEMA, 'requested_scope': limit, 'attempted_requests': attempted,
              'min_source_index': min_index,
              'admitted_old': len(old_inputs), 'quantized_teacher_targets': len(target_inputs),
              'rejections': dict(Counter(row['reason'] for row in rejected)), 'geometry_protocol': GEOMETRY_PROTOCOL,
              'heldout_cohort_sha256': sha256(heldout_cohort), 'manifest_sha256': sha256(manifest_path),
              'source_provenance': provenance, 'evaluation_chemistry_groups': len(forbidden)}
    report['teacher_unavailable'] = sum(not row['teacher_available'] for row in pending)
    report['teacher_rejections'] = dict(Counter(
        row.get('teacher_failure') or row.get('target_geometry', {}).get('reason') or 'missing_refiner_endpoint'
        for row in pending if not row['teacher_available']))
    report['files_sha256'] = {name: sha256(output / name) for name in (
        'pairs_pending.jsonl', 'old_inputs.jsonl', 'target_inputs.jsonl', 'all_inputs.jsonl', 'rejected.jsonl')}
    write_json(output / 'PREPARATION_FINAL.json', report)
    (output / '_SUCCESS').touch()
    return report


def endpoint_fingerprint(record):
    if not record['success']:
        return record['trajectory_id']
    value = json.dumps(record['structure'], sort_keys=True) if record.get('structure') is not None else str(record['body'])
    return hashlib.sha256(value.encode()).hexdigest()


def bound_labels(input_path, directory, *, exclude_worker_errors=False):
    directory = Path(directory)
    report = json.loads((directory / 'LABEL_FINAL.json').read_text())
    if report.get('purpose') != 'expert_edit':
        raise ValueError('editor labels require the explicit expert-edit purpose')
    if (report.get('protocol') != COMMON_RELAXATION_PROTOCOL
            or report.get('verification_protocol') != TERMINAL_VERIFICATION_PROTOCOL):
        raise ValueError('physics protocol differs from the frozen common relaxation')
    recovered = (exclude_worker_errors and report['statuses'].get('worker_error', 0)
                 and (directory / '_ENGINEERING_FAILED').is_file())
    if (not (directory / '_SUCCESS').is_file() or report['statuses'].get('worker_error', 0)) and not recovered:
        raise ValueError('physics labels are incomplete or contain engineering failures')
    if report.get('input_sha256') != sha256(input_path) or len(report.get('runtime_identities', [])) != 1:
        raise ValueError('physics input/runtime identity is not fully established')
    input_rows, label_rows = read_rows(input_path), read_rows(directory / 'labels.jsonl')
    inputs = {row['trajectory_id']: row for row in input_rows}
    labels = {row['trajectory_id']: row for row in label_rows}
    if (len(inputs) != len(input_rows) or len(labels) != len(label_rows) or set(inputs) != set(labels)
            or report['completed'] != len(inputs) or report.get('requested') != len(inputs)
            or dict(Counter(row['status'] for row in label_rows)) != report['statuses']):
        raise ValueError('physics labels do not cover exactly the registered inputs')
    for key, record in inputs.items():
        label = labels[key]
        if (label.get('endpoint_cache_key') != endpoint_fingerprint(record)
                or (label.get('versions') != report['runtime_identities'][0]
                    and not (recovered and label['status'] == 'worker_error'))
                or any(label.get(field) != record.get(field) for field in ('group_id', 'source_row_idx', 'source_split', 'endpoint'))):
            raise ValueError('physics label is bound to a different input geometry or occurrence')
    return labels, report


def arrays_from_structure(structure):
    lattice = structure['lattice']
    sites = structure['sites']
    species = []
    for site in sites:
        if len(site['species']) != 1 or abs(float(site['species'][0].get('occu', 1)) - 1) > 1e-8:
            raise ValueError('disordered terminal is outside the fixed-species edit contract')
        species.append(site['species'][0]['element'])
    return {'lengths': [lattice[k] for k in ('a', 'b', 'c')],
            'angles': [lattice[k] for k in ('alpha', 'beta', 'gamma')],
            'species': species, 'frac_coords': [site['abc'] for site in sites]}


def terminal_geometry(label):
    try:
        certificate = certify_geometry(arrays_from_structure(label['final_structure']))
    except (KeyError, ValueError, TypeError):
        return None
    return certificate['valid'] if certificate['certified'] else None


def reliable_terminal(label):
    if label.get('scale_uncertain'):
        return None
    geometry = terminal_geometry(label)
    if geometry is False:
        return False
    if label.get('verified') is not True:
        return False if label.get('status') not in ('worker_error', 'evaluation_error') else None
    return True if geometry is True else None


def derive_pair_supervision(pair, old_label, target_label, *, margin=.01):
    """Observed proposal labels; no absent-teacher or state-level S-stop labels."""
    result = {'G_reason': None, 'S_outcome': None, 'gain_eV_atom': None,
              'old_reliable': reliable_terminal(old_label), 'target_reliable': None,
              'old_terminal_geometry': terminal_geometry(old_label), 'target_terminal_geometry': None}
    if not pair.get('teacher_available') or target_label is None:
        return result
    result['target_reliable'] = reliable_terminal(target_label)
    result['target_terminal_geometry'] = terminal_geometry(target_label)
    if pair['old_body'] == pair['target_body']:
        result.update(S_outcome='identity', gain_eV_atom=0.)
        return result
    raw_old = pair['old_geometry'].get('valid')
    raw_target = pair['target_geometry'].get('valid')
    if raw_old is False and pair['old_geometry']['certified'] and raw_target is True:
        result['G_reason'] = 'raw_geometry_recovery'
    elif raw_old is True and result['target_reliable'] is True and result['old_reliable'] is False:
        if old_label.get('status') == 'invalid_terminal' or result['old_terminal_geometry'] is False:
            result['G_reason'] = 'relaxation_geometry_recovery'
        elif old_label.get('status') in ('not_converged', 'optimizer_stop_unverified', 'terminal_consistency_unverified'):
            result['G_reason'] = 'terminal_reliability_recovery'
    if (raw_old is True and raw_target is True and result['old_reliable'] is True
            and result['target_reliable'] is True):
        values = [old_label.get('terminal_energy'), target_label.get('terminal_energy')]
        if all(value is not None and math.isfinite(float(value)) for value in values):
            gain = float(values[0]) - float(values[1])
            result['gain_eV_atom'] = gain
            result['S_outcome'] = 'positive' if gain >= margin else 'negative' if gain <= -margin else 'neutral'
    return result


def compile_collection(prepared, labels_dir, output, *, margin=.01, exclude_worker_errors=False):
    prepared, output = Path(prepared), Path(output)
    if not (prepared / '_SUCCESS').is_file():
        raise ValueError('expert input compilation is incomplete')
    preparation = json.loads((prepared / 'PREPARATION_FINAL.json').read_text())
    for name, expected in preparation['files_sha256'].items():
        if Path(name).name != name or sha256(prepared / name) != expected:
            raise ValueError('prepared editing pairs or physics inputs changed')
    inputs = prepared / 'all_inputs.jsonl'
    labels, label_report = bound_labels(inputs, labels_dir, exclude_worker_errors=exclude_worker_errors)
    pending = read_rows(prepared / 'pairs_pending.jsonl')
    output.mkdir(parents=True, exist_ok=False)
    records, outcomes, counts = [], [], Counter()
    positive_groups = {'train': set(), 'dev': set()}
    comparable_groups = set()
    excluded_engineering = []
    for pair in pending:
        old = labels[pair['old_physics_id']]
        target = labels.get(pair['target_physics_id'])
        if any(label is not None and label.get('status') == 'worker_error' for label in (old, target)):
            excluded_engineering.append(pair['ancestor_id'])
            outcomes.append({'ancestor_id': pair['ancestor_id'], 'source_row_idx': pair['source_row_idx'],
                             'source_split': pair['source_split'], 'status': 'engineering_unknown_excluded'})
            continue
        supervision = derive_pair_supervision(pair, old, target, margin=margin)
        if pair['old_geometry'].get('valid') is not None or supervision['old_reliable'] is not None:
            old_only = {key: pair[key] for key in (
                'schema', 'ancestor_id', 'source_kind', 'source_split', 'source_row_idx', 'composition_key',
                'plan_state', 'prompt', 'num_atoms', 'old_body', 'old_geometry', 'old_physics_id',
                'original_source_body', 'periodic_aliases_canonicalized') if key in pair}
            records.append({**old_only, 'record_id': 'state:' + pair['ancestor_id'], 'task': 'G',
                            'state_only': True, 'content_supervision': False, 'accept_label': None,
                            'label_reason': 'observed_state_only', 'gain_eV_atom': None,
                            'old_reliable': supervision['old_reliable'], 'target_reliable': None})
            counts['state_observation'] += 1
        outcomes.append({'ancestor_id': pair['ancestor_id'], 'source_row_idx': pair['source_row_idx'],
                         'source_split': pair['source_split'], **supervision,
                         'old_status': old['status'], 'target_status': target['status'] if target else None})
        def emit(task, accept, content, reason, *, reverse=False):
            row = {**pair, 'record_id': task + ':' + reason + ':' + pair['ancestor_id'],
                   'task': task, 'content_supervision': content, 'accept_label': accept,
                   'label_reason': reason, 'gain_eV_atom': supervision['gain_eV_atom'],
                   'old_reliable': supervision['old_reliable'], 'target_reliable': supervision['target_reliable']}
            if reverse:
                row.update(old_body=pair['target_body'], target_body=pair['old_body'],
                           old_geometry=pair['target_geometry'], target_geometry=pair['old_geometry'],
                           old_reliable=supervision['target_reliable'], target_reliable=supervision['old_reliable'],
                           gain_eV_atom=-supervision['gain_eV_atom'])
                row.update(old_physics_id=pair['target_physics_id'], target_physics_id=pair['old_physics_id'],
                           alignment={'kind': 'reverse_of_verified_full_action'})
                row['action'] = full_action(row['old_body'], row['target_body'], row['num_atoms'])
            records.append(row)
            counts[f'{task}:{reason}'] += 1
        if supervision['G_reason']:
            emit('G', True, True, supervision['G_reason'])
        outcome = supervision['S_outcome']
        if outcome in ('positive', 'negative', 'neutral'):
            comparable_groups.add(pair['ancestor_id'])
            emit('S', outcome == 'positive', outcome == 'positive', outcome)
            if outcome == 'positive':
                positive_groups[pair['source_split']].add(pair['composition_key'])
                emit('S', False, False, 'reverse_verified_degradation', reverse=True)
        elif outcome == 'identity':
            emit('S', False, False, 'identity_zero_action')
    for split in ('train', 'dev'):
        write_rows(output / (split + '.jsonl'), [row for row in records if row['source_split'] == split])
    write_rows(output / 'outcomes.jsonl', outcomes)
    report = {'schema': SCHEMA, 'counts': dict(counts), 'records': len(records),
              'requested_sources': preparation['attempted_requests'], 'admitted_sources': len(pending),
              'comparable_sources': len(comparable_groups), 'S_positive_compositions': {k: len(v) for k, v in positive_groups.items()},
              'S_margin_eV_atom': margin, 'labels_protocol': label_report['protocol'],
              'labels_runtime': label_report['runtime_identities'], 'labels_sha256': sha256(Path(labels_dir) / 'labels.jsonl'),
              'preparation_sha256': sha256(prepared / 'PREPARATION_FINAL.json'),
              'excluded_engineering_ancestors': excluded_engineering,
              'unknown_is_negative': False, 'identity_teaches_S_stop': False}
    report['output_sha256'] = {name: sha256(output/name) for name in ('train.jsonl','dev.jsonl','outcomes.jsonl')}
    report['headroom_budget_gate'] = (len(comparable_groups) >= 32
                                      and sum(map(len, positive_groups.values())) >= 16)
    write_json(output / 'DATA_FINAL.json', report)
    (output / '_SUCCESS').touch()
    return report


def geometry_auxiliary_examples(clean, vocabulary, rng):
    """Local, cooperative, all-coordinate and lattice-coupled witnessed repairs."""
    target, target_arrays, _ = quantize_arrays(clean, vocabulary)
    target_geometry = certify_geometry(target_arrays)
    if target_geometry['valid'] is not True or not target_geometry['certified']:
        return []
    n, examples = len(clean['species']), []
    inverse_lattice = np.linalg.inv(lattice_from_parameters(target_arrays['lengths'], target_arrays['angles']))
    def near(point):
        if rng.random() < .3:
            return list(point)
        direction = np.asarray([rng.uniform(-1,1) for _ in range(3)])
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        delta = (direction*rng.uniform(.03,.13))@inverse_lattice
        return np.mod(np.asarray(point)+delta,1.).tolist()
    buckets = ('single_site','cooperative_sites','all_xyz','lattice_coupled') if n > 1 else ('lattice_coupled',)
    for bucket in buckets:
        damaged = copy.deepcopy(target_arrays)
        sites = []
        if bucket == 'single_site':
            selected, anchor = rng.sample(range(n),2)
            sites = [selected]
            damaged['frac_coords'][selected] = near(target_arrays['frac_coords'][anchor])
            mode = 'local_xyz'
        elif bucket in ('cooperative_sites','all_xyz'):
            choices = [k for k in (2,4,8) if k <= n]
            sites = sorted(rng.sample(range(n), rng.choice(choices))) if bucket == 'cooperative_sites' else list(range(n))
            point = rng.choice(target_arrays['frac_coords'])
            for site in sites:
                damaged['frac_coords'][site] = near(point)
            mode = 'all_xyz' if len(sites) == n else 'local_xyz'
        else:
            # Two different failures prevent the geometry branch from learning
            # only one artificial token signature. Nonselected coordinates are
            # exact copy targets, so the old state is useful for reconstruction.
            if rng.random() < .5:
                damaged['lengths'][rng.randrange(3)] = rng.choice((.1,.2,.3,.4))
            else:
                alpha, beta = rng.randint(30,80), rng.randint(30,80)
                damaged['angles'] = [float(alpha),float(beta),float(alpha+beta+rng.randint(1,8))]
            if n > 1:
                sites = sorted(rng.sample(range(n),2))
                damaged['frac_coords'][sites[0]] = near(target_arrays['frac_coords'][sites[1]])
            mode = 'full_cell'
        old, old_arrays, _ = quantize_arrays(damaged, vocabulary)
        old_geometry = certify_geometry(old_arrays)
        if old_geometry['valid'] is not False or not old_geometry['certified']:
            continue
        positions = ([1,2,3,4,5,6] if mode == 'full_cell' else [])
        active_sites = list(range(n)) if mode in ('full_cell','all_xyz') else sites
        positions += [8+4*site+axis for site in active_sites for axis in range(3)]
        changed = [i for i,(a,b) in enumerate(zip(old,target)) if a != b]
        if not changed or not set(changed).issubset(positions):
            raise ValueError('quantized auxiliary corruption escaped its declared correction scope')
        if mode == 'local_xyz' and len(sites) not in (1,2,4,8):
            raise ValueError('unregistered cooperative repair size')
        examples.append({'bucket': bucket, 'old_body': old, 'target_body': target,
                         'old_geometry': old_geometry, 'target_geometry': target_geometry,
                         'action': {'mode': mode, 'sites': active_sites, 'positions': positions,
                                    'changed_positions': changed, 'changed_sites': sorted({(i-8)//4 for i in changed if i>=8})}})
    return examples


def build_geometry_auxiliary(source_path, source_sha256, heldout_cohort, heldout_sha256, tokenizer,
                             output, *, source_limit=2048, seed=20260908):
    """Use only clean MP20 TRAIN token bodies; discard all GT-derived soft hints."""
    if source_limit < 1:
        raise ValueError('a positive independent auxiliary source count is required')
    if sha256(source_path) != source_sha256 or sha256(heldout_cohort) != heldout_sha256:
        raise ValueError('MP20 training source or excluded evaluation cohort changed')
    from crystal_dlm.r5_plan_state import build_hard_anchor_body_prompt
    forbidden = {composition_key(row['plan_state']) for row in read_rows(heldout_cohort) if row.get('plan_state')}
    candidates = read_rows(source_path)
    ids = [row['source_row_idx'] for row in candidates]
    if len(set(ids)) != len(ids) or any(row.get('source_split') != 'train' or
                  row.get('closure',{}).get('source_answer_is_clean_teacher') is not True for row in candidates):
        raise ValueError('auxiliary sources must be distinct declared clean training examples')
    candidates.sort(key=lambda row: hashlib.sha256(f'{seed}:{row["source_row_idx"]}'.encode()).hexdigest())
    vocabulary = tokenizer.get_vocab()
    records, exclusions, sources, splits = [], Counter(), 0, Counter()
    for row in candidates:
        if sources >= source_limit:
            break
        try:
            clean = parse_dynamic_answer(row['source_answer'], strict=True)
            key = composition_key(clean['species'])
            if key in forbidden:
                exclusions['evaluation_composition'] += 1
                continue
            seed_value = int(hashlib.sha256(f'{seed}:{row["source_row_idx"]}:corruption'.encode()).hexdigest()[:16],16)
            examples = geometry_auxiliary_examples(clean, vocabulary, random.Random(seed_value))
            if not examples:
                exclusions['no_certified_target_and_recovery'] += 1
                continue
        except (ValueError,KeyError,TypeError,IndexError) as error:
            exclusions[type(error).__name__] += 1
            continue
        counts = Counter(clean['species'])
        plan = {'N': len(clean['species']), 'elements': list(counts), 'counts': list(counts.values()),
                'formula': ''.join(symbol+(str(count) if count != 1 else '') for symbol,count in counts.items())}
        split = 'dev' if int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%8 == 0 else 'train'
        ancestor = f'mp20-geometry:{source_sha256[:16]}:{row["source_row_idx"]}'
        shared = {'schema': SCHEMA, 'ancestor_id': ancestor, 'source_kind': 'mp20_geometry_auxiliary',
                  'source_split': split, 'source_row_idx': row['source_row_idx'], 'composition_key': key,
                  'plan_state': plan, 'prompt': build_hard_anchor_body_prompt(plan).rstrip()+'\n',
                  'num_atoms': plan['N'], 'task': 'G', 'old_reliable': None, 'target_reliable': None,
                  'old_physics_id': None, 'target_physics_id': None, 'gain_eV_atom': None,
                  'GT_soft_hints_used': False, 'source_clean_body_sha256': hashlib.sha256(row['source_answer'].encode()).hexdigest()}
        for example in examples:
            records.append({**shared, **example, 'record_id': 'G:'+example['bucket']+':'+ancestor,
                            'content_supervision': True, 'accept_label': True,
                            'label_reason': 'certified_local_geometry_recovery', 'teacher_available': True})
        clean_body, clean_geometry = examples[0]['target_body'], examples[0]['target_geometry']
        records.append({**shared, 'record_id': 'state:'+ancestor, 'state_only': True,
                        'old_body': clean_body, 'old_geometry': clean_geometry,
                        'content_supervision': False, 'accept_label': None, 'label_reason': 'geometry_only_clean_state'})
        sources += 1
        splits[split] += 1
    if sources < source_limit:
        raise ValueError(f'only {sources}/{source_limit} independent certified auxiliary sources are available')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for split in ('train','dev'):
        write_rows(output/(split+'.jsonl'), [row for row in records if row['source_split']==split])
    write_rows(output/'outcomes.jsonl', [])
    report = {'schema': SCHEMA, 'records': len(records), 'admitted_sources': sources, 'splits': dict(splits),
              'counts': dict(Counter(row.get('bucket','state') for row in records)), 'exclusions': dict(exclusions),
              'source_sha256': source_sha256, 'heldout_sha256': heldout_sha256, 'seed': seed,
              'geometry_protocol': GEOMETRY_PROTOCOL, 'content_task': 'G_only', 'physical_energy_supervision': False,
              'GT_soft_hints_used': False, 'identity_teaches_S_stop': False,
              'output_sha256': {name: sha256(output/name) for name in ('train.jsonl','dev.jsonl','outcomes.jsonl')}}
    write_json(output/'DATA_FINAL.json', report)
    (output/'_SUCCESS').touch()
    return report


def feedback_decision(task, old_body, target_body, old_geometry, target_geometry, old_label, target_label, *, margin=.01):
    """Observed structural decisions; an unlabelled R endpoint stays unknown."""
    old_r = reliable_terminal(old_label) if old_label is not None else None
    new_r = reliable_terminal(target_label) if target_label is not None else None
    gain, accept, reason = None, None, 'unobserved_physical_outcome'
    old_g, new_g = old_geometry.get('valid'), target_geometry.get('valid')
    if old_body == target_body:
        gain, accept, reason = 0., False, 'observed_identity_zero_action'
    elif new_g is False:
        accept, reason = False, 'observed_invalid_proposal'
    elif task == 'G':
        if old_g is False and new_g is True:
            accept, reason = True, 'observed_raw_geometry_recovery'
        elif old_g is True and new_g is True and old_r is not None and new_r is not None:
            accept = old_r is False and new_r is True
            reason = 'observed_reliability_recovery' if accept else 'observed_no_G_recovery'
    elif task == 'S':
        if old_g is True and new_g is True and old_r is True and new_r is True:
            values = [old_label.get('terminal_energy'), target_label.get('terminal_energy')]
            if all(value is not None and math.isfinite(float(value)) for value in values):
                gain = float(values[0])-float(values[1])
                accept, reason = gain >= margin, 'observed_comparable_stability'
        elif old_r is True and new_r is False:
            accept, reason = False, 'observed_reliability_loss'
    return {'accept_label': accept, 'gain_eV_atom': gain, 'label_reason': reason,
            'old_reliable': old_r, 'target_reliable': new_r}


def replay_feedback_trace(output, initial_body, num_atoms):
    """Bind complete proposal supervision to the recorded autonomous commit chain."""
    if (output.get('scope_policy') != 'learned' or output.get('accept_all') is not False
            or output.get('S_admission_policy') != 'learned'):
        raise ValueError('feedback trace must use autonomous learned scope and acceptance')
    current, complete = list(initial_body), []
    for index, trace in enumerate(output['trace']):
        if trace.get('old_body') != current or trace.get('task') not in ('G', 'S'):
            raise ValueError('feedback trace does not follow its actual retained state')
        candidate = trace.get('proposal_body')
        if not fixed_composition(current, candidate, num_atoms):
            raise ValueError('feedback trace changed protected composition tokens')
        accepted = trace.get('accepted')
        applied = trace.get('applied', accepted)
        if type(accepted) is not bool or type(applied) is not bool or accepted != applied:
            raise ValueError('feedback trace acceptance differs from the applied action')
        if trace.get('mode') == 'none' or trace.get('reason') == 'insufficient_complete_proposal_budget':
            if candidate != current or applied:
                raise ValueError('a trace without a complete proposal changed the retained state')
            continue
        sites, positions = trace.get('sites'), trace.get('positions')
        if (trace.get('mode') not in ('local_xyz', 'all_xyz', 'full_cell')
                or not isinstance(sites, list) or not isinstance(positions, list)
                or any(type(site) is not int or not 0 <= site < num_atoms for site in sites)
                or len(set(sites)) != len(sites)):
            raise ValueError('feedback trace lacks its complete registered action scope')
        expected = ([1, 2, 3, 4, 5, 6] if trace['mode'] == 'full_cell' else []) + [
            8+4*site+axis for site in sites for axis in range(3)]
        if (positions != expected or not positions
                or (trace['mode'] != 'local_xyz' and sites != list(range(num_atoms)))
                or not {i for i, (a,b) in enumerate(zip(current,candidate)) if a != b}.issubset(positions)):
            raise ValueError('feedback trace action does not cover its actual edited fields')
        complete.append((index, trace))
        if applied:
            current = list(candidate)
    if current != output.get('canonical_body'):
        raise ValueError('feedback trace final state differs from its labelled endpoint')
    return complete


def compile_student_feedback(samples_directory, labels_directory, reference_prepared, reference_labels,
                             tokenizer, output, *, margin=.01):
    """Include every recorded training proposal, with masks for unobserved R.

    Actual intermediate/rejected proposals remain present even when only the
    final committed endpoint was labelled. Validated offline targets can teach
    a new correction from a complete student state; no bad partial prefix is
    paired with an assumed-valid teacher continuation.
    """
    samples_directory, labels_directory, output = map(Path,(samples_directory,labels_directory,output))
    summary = json.loads((samples_directory/'SAMPLE_FINAL.json').read_text())
    if (not (samples_directory/'_SUCCESS').is_file() or summary.get('split') != 'train'
            or summary.get('source_kind') != 'all_old_states' or summary.get('frozen_B0_control') is True):
        raise ValueError('feedback requires a completed training-only autonomous student sample')
    samples = read_rows(samples_directory/'samples.jsonl')
    inputs = read_rows(samples_directory/'physics.jsonl')
    if (len(samples) != summary['requested'] or len(inputs) != len(samples)
            or len({row['ancestor_id'] for row in samples}) != len(samples)
            or any(row.get('source_split') != 'train' for row in samples+inputs)):
        raise ValueError('student feedback lost/duplicated sources or contains development data')
    current, label_report = bound_labels(samples_directory/'physics.jsonl',labels_directory,exclude_worker_errors=True)
    by_group = {row['group_id']: row for row in inputs}
    references, pairs, provenance, heldout_identities = {}, {}, [], set()
    if not reference_prepared or len(reference_prepared) != len(reference_labels):
        raise ValueError('each reference label set requires its exact prepared input directory')
    scientific_keys = ('model','model_checkpoint_sha256','chgnet_package','ase_package','torch_package','pymatgen_package')
    for prepared, directory in zip(map(Path,reference_prepared),map(Path,reference_labels)):
        original, report = bound_labels(prepared/'all_inputs.jsonl',directory,exclude_worker_errors=True)
        if any(report['runtime_identities'][0].get(key) != label_report['runtime_identities'][0].get(key) for key in scientific_keys):
            raise ValueError('feedback physical weights/runtime differ from the reference R labels')
        preparation = json.loads((prepared/'PREPARATION_FINAL.json').read_text())
        heldout_identity = preparation.get('heldout_cohort_sha256')
        if not isinstance(heldout_identity,str) or len(heldout_identity) != 64:
            raise ValueError('feedback reference lacks its excluded evaluation-cohort identity')
        heldout_identities.add(heldout_identity)
        if len(heldout_identities) != 1:
            raise ValueError('feedback reference sources excluded different evaluation cohorts')
        if sha256(prepared/'pairs_pending.jsonl') != preparation['files_sha256']['pairs_pending.jsonl']:
            raise ValueError('offline teacher targets changed after physics input registration')
        for pair in read_rows(prepared/'pairs_pending.jsonl'):
            if pair['source_split'] != 'train':
                continue
            if pair['ancestor_id'] in pairs:
                raise ValueError('reference source intervals overlap')
            pairs[pair['ancestor_id']] = pair
        if set(references)&set(original):
            raise ValueError('reference physics IDs overlap')
        references.update(original)
        provenance.append({'prepared': str(prepared), 'pairs_sha256': sha256(prepared/'pairs_pending.jsonl'),
                           'labels_sha256': sha256(directory/'labels.jsonl'), 'runtime': report['runtime_identities'][0]})
    inverse = {int(value): key for key,value in tokenizer.get_vocab().items()}
    records, outcomes, ids, skipped = [], [], set(), []
    def emit(row):
        if row['record_id'] not in ids:
            ids.add(row['record_id'])
            records.append(row)
    for sample in samples:
        ancestor = sample['ancestor_id']
        if ancestor not in pairs or ancestor not in by_group:
            raise ValueError('feedback source does not belong to the registered training ancestors')
        pair, record = pairs[ancestor], by_group[ancestor]
        if (sample['old_body'] != pair['old_body'] or sample['prompt'] != pair['prompt']
                or sample['num_atoms'] != pair['num_atoms'] or sample['source_row_idx'] != pair['source_row_idx']):
            raise ValueError('student old-state/condition differs from its actual original source')
        complete_proposals = replay_feedback_trace(sample['output'], pair['old_body'], pair['num_atoms'])
        final_label = current[record['trajectory_id']]
        initial_label = references[pair['old_physics_id']]
        teacher_label = references.get(pair.get('target_physics_id'))
        if any(label is not None and label.get('status') == 'worker_error'
               for label in (initial_label, teacher_label, final_label)):
            skipped.append(ancestor)
            continue
        final_body = sample['output']['canonical_body']
        final_arrays = decode_body(final_body,inverse)
        rebound = physics_input(record['trajectory_id'],ancestor,pair['source_row_idx'],'train','expert_quantized',
                                final_arrays,''.join(inverse[token] for token in final_body))
        if endpoint_fingerprint(rebound) != endpoint_fingerprint(record):
            raise ValueError('labelled final geometry differs from the recorded student output')
        known = {tuple(pair['old_body']): initial_label, tuple(final_body): final_label}
        if pair.get('teacher_available'):
            known.setdefault(tuple(pair['target_body']),teacher_label)
        geometries = {}
        def geometry(body):
            key = tuple(body)
            if key not in geometries:
                geometries[key] = certify_geometry(decode_body(body,inverse))
            return geometries[key]
        shared = {key: pair[key] for key in ('ancestor_id','source_split','source_row_idx','composition_key',
                                           'plan_state','prompt','num_atoms')}
        shared.update(schema=SCHEMA,source_kind='student_proposal_feedback')
        def decision(old, candidate, task, action, tag, *, allow_content=False):
            if not fixed_composition(old,candidate,pair['num_atoms']):
                raise ValueError('student feedback changed protected composition tokens')
            changed = {i for i,(a,b) in enumerate(zip(old,candidate)) if a!=b}
            if not changed.issubset(action['positions']):
                raise ValueError('feedback action does not cover the actual proposal changes')
            old_label, target_label = known.get(tuple(old)),known.get(tuple(candidate))
            judgement = feedback_decision(task,old,candidate,geometry(old),geometry(candidate),old_label,target_label,margin=margin)
            rid = f'feedback:{samples_directory.name}:{ancestor}:{tag}:{task}'
            row = {**shared, **judgement, 'record_id': rid, 'task': task,
                   'old_body': list(old), 'target_body': list(candidate), 'action': action,
                   'old_geometry': geometry(old), 'target_geometry': geometry(candidate),
                   'old_physics_id': old_label.get('trajectory_id') if old_label else None,
                   'target_physics_id': target_label.get('trajectory_id') if target_label else None,
                   'content_supervision': allow_content and judgement['accept_label'] is True,
                   'original_teacher_reference': tag.startswith('teacher_from_student')}
            emit(row)
            outcomes.append({'record_id': rid, 'ancestor_id': ancestor, 'task': task, **judgement,
                             'gain_label_known': judgement['gain_eV_atom'] is not None,
                             'identity_zero_override': old == candidate})
        candidates = {}
        for index, trace in complete_proposals:
            old,candidate = trace['old_body'],trace['proposal_body']
            action = {key: trace[key] for key in ('mode','sites','positions')}
            decision(old,candidate,trace['task'],action,f'proposal_{index}')
            candidates[tuple(candidate)] = candidate
        # A value head can score an observed old/final pair regardless of how
        # many neural edits generated the final structure. It is not credited
        # to a particular intermediate action without that action's R labels.
        for task in ('G','S'):
            decision(pair['old_body'],final_body,task,full_action(pair['old_body'],final_body,pair['num_atoms']),
                     'observed_complete_transaction',allow_content=True)
        candidates[tuple(final_body)] = final_body
        for candidate in candidates.values():
            key = hashlib.sha256(json.dumps(candidate).encode()).hexdigest()[:20]
            label = known.get(tuple(candidate))
            emit({**shared,'record_id':f'feedback-state:{samples_directory.name}:{ancestor}:{key}',
                  'task':'G','state_only':True,'old_body':list(candidate),'old_geometry':geometry(candidate),
                  'old_reliable':reliable_terminal(label) if label is not None else None,
                  'content_supervision':False,'accept_label':None,'gain_eV_atom':None,
                  'label_reason':'observed_student_state'})
            if pair.get('teacher_available') and geometry(pair['target_body'])['valid'] is True:
                for task in ('G','S'):
                    decision(candidate,pair['target_body'],task,full_action(candidate,pair['target_body'],pair['num_atoms']),
                             'teacher_from_student_'+key,allow_content=True)
    output.mkdir(parents=True,exist_ok=False)
    write_rows(output/'train.jsonl',records)
    write_rows(output/'dev.jsonl',[])
    write_rows(output/'outcomes.jsonl',outcomes)
    report = {'schema':SCHEMA,'records':len(records),'requested_sources':len(samples),
              'admitted_sources':len(samples)-len(skipped),'excluded_engineering_ancestors':skipped,
              'counts':dict(Counter(f'{row["task"]}:{row.get("accept_label")}' for row in records if not row.get('state_only'))),
              'content_positive_sources': {task:len({row['ancestor_id'] for row in records if row['task']==task and row.get('content_supervision')}) for task in ('G','S')},
              'sample_sha256':sha256(samples_directory/'samples.jsonl'),
              'student_labels_sha256':sha256(labels_directory/'labels.jsonl'),
              'reference_inputs':provenance,'label_runtime':label_report['runtime_identities'],
              'heldout_cohort_sha256':next(iter(heldout_identities)),
              'unknown_is_negative':False,'development_records_used_for_training':False,
              'partial_student_prefix_is_assumed_valid':False,
              'output_sha256':{name:sha256(output/name) for name in ('train.jsonl','dev.jsonl','outcomes.jsonl')}}
    write_json(output/'DATA_FINAL.json',report)
    (output/'_SUCCESS').touch()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare', 'compile', 'geometry-auxiliary', 'student-feedback'), default='prepare')
    parser.add_argument('--collection-manifest', type=Path)
    parser.add_argument('--heldout-cohort', type=Path)
    parser.add_argument('--b0-checkpoint', type=Path)
    parser.add_argument('--prepared-dir', type=Path)
    parser.add_argument('--labels-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--min-index', type=int, default=0)
    parser.add_argument('--source-path', type=Path)
    parser.add_argument('--source-sha256')
    parser.add_argument('--heldout-cohort-sha256')
    parser.add_argument('--source-limit', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--samples-dir', type=Path)
    parser.add_argument('--reference-prepared', type=Path,nargs='+')
    parser.add_argument('--reference-labels', type=Path,nargs='+')
    parser.add_argument('--exclude-worker-errors', action='store_true',
                        help='Training only: exclude entire unresolved ancestors from a fully accounted label run')
    args = parser.parse_args(argv)
    if args.mode == 'student-feedback':
        if None in (args.samples_dir,args.labels_dir,args.reference_prepared,args.reference_labels,args.b0_checkpoint):
            parser.error('student feedback needs samples, labels, matching reference inputs/labels and B0 tokenizer')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.b0_checkpoint,trust_remote_code=True,local_files_only=True)
        print(json.dumps(compile_student_feedback(args.samples_dir,args.labels_dir,args.reference_prepared,
                         args.reference_labels,tokenizer,args.output_dir)),flush=True)
        return
    if args.mode == 'geometry-auxiliary':
        if None in (args.source_path,args.source_sha256,args.heldout_cohort,args.heldout_cohort_sha256,args.b0_checkpoint):
            parser.error('geometry auxiliary needs pinned clean source, heldout cohort, and the B0 tokenizer')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.b0_checkpoint, trust_remote_code=True, local_files_only=True)
        print(json.dumps(build_geometry_auxiliary(args.source_path,args.source_sha256,args.heldout_cohort,
                         args.heldout_cohort_sha256,tokenizer,args.output_dir,
                         source_limit=args.source_limit,seed=args.seed)), flush=True)
        return
    if args.mode == 'compile':
        if args.prepared_dir is None or args.labels_dir is None:
            parser.error('compile requires prepared-dir and labels-dir')
        print(json.dumps(compile_collection(args.prepared_dir, args.labels_dir, args.output_dir,
                                            exclude_worker_errors=args.exclude_worker_errors)), flush=True)
        return
    if args.collection_manifest is None or args.heldout_cohort is None or args.b0_checkpoint is None:
        parser.error('prepare requires collection-manifest, heldout-cohort and b0-checkpoint')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.b0_checkpoint, trust_remote_code=True, local_files_only=True)
    print(json.dumps(prepare_collection(args.collection_manifest, args.heldout_cohort, tokenizer,
                                        args.output_dir, limit=args.limit, min_index=args.min_index)), flush=True)


if __name__ == '__main__':
    main()
