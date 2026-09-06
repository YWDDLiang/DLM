"""Stdlib-only audit of row-independent inputs to the frozen typed collators.

Reads source JSONL files only. No torch, model calls, sampling, or new labels.
Hashes omit source identity, soft labels, contact targets, and audit metadata.
Goal strings and (family_id,N,arity) tuples are lossless substitutes for their
fixed injective embedding IDs; no checkpoint needs to be deserialized.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import struct
from typing import Any

SYMBOLS = ('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn '
           'Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce '
           'Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn '
           'Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og').split()
Z = {symbol: index + 1 for index, symbol in enumerate(SYMBOLS)}
SOFT_FIELDS = ('lattice_system', 'spacegroup_bucket', 'volume_per_atom_bin')


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return sha256(canonical_json(value).encode('utf-8')).hexdigest()


def f32(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError('Non-finite ledger input')
    return struct.unpack('<f', struct.pack('<f', value))[0]


def load_rows(path: Path) -> tuple[dict[int, dict], dict]:
    rows = {}
    hasher = sha256()
    with path.open('rb') as handle:
        for raw in handle:
            hasher.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            index = int(row['source_row_idx'])
            if index in rows:
                raise ValueError(f'Duplicate source_row_idx in {path}: {index}')
            rows[index] = row
    return rows, {'path': str(path), 'rows': len(rows), 'sha256': hasher.hexdigest()}


def composition(plan: dict) -> tuple[tuple[int, int], ...]:
    counts = Counter()
    if len(plan['elements']) != len(plan['counts']):
        raise ValueError('Element/count shape mismatch')
    for symbol, count in zip(plan['elements'], plan['counts']):
        count = int(count)
        if count <= 0:
            raise ValueError('Nonpositive count')
        counts[Z[str(symbol)]] += count
    answer = tuple(sorted(counts.items()))
    if sum(counts.values()) != int(plan['N']):
        raise ValueError('N differs from sum of counts')
    return answer


def normalized_ledger(raw: dict) -> list[float]:
    branch = str(raw.get('branch') or 'unset')
    branches = {'unset': [1., 0., 0.], 'ionic': [0., 1., 0.], 'alloy': [0., 0., 1.]}
    return [f32(float(raw['remaining_atoms']) / 20.),
            f32(float(raw['net_charge']) / 160.),
            f32(float(raw['remaining_species']) / 7.), *branches[branch]]


def actual_chemical_inputs(metadata: dict) -> dict:
    """Canonical form of the unpadded collator tensors, excluding soft labels.

    Matches collate_prediction_requests and collate_typed_rows on chemical
    inputs. Position 0 ledger is explicitly zeros even if its raw row differs.
    previous_n_values is consumed by frozen C3FD, not by typed_inputs_embeds.
    """
    species = [int(v) for v in metadata['species_ids']]
    counts = [int(v) for v in metadata['count_targets']]
    length = len(species) + 2
    if len(species) != len(counts) or len(metadata['ledger_steps']) != length:
        raise ValueError('Chemical sequence/ledger lengths disagree')
    proposal = metadata['proposal_target']
    stratum = [int(proposal['family_id']), int(proposal['N']), int(proposal['arity'])]
    if stratum[2] != len(species):
        raise ValueError('Arity differs from chemical sequence length')
    return {
        'goal_as_fixed_embedding_id_equivalent': str(metadata['stability_condition']),
        'proposal_state_ids_equivalent': [['query']] + [['stratum', *stratum] for _ in range(length - 1)],
        'previous_species_indices': [-1, -1, *species],
        'previous_count_values': [0, 0, *counts],
        'previous_n_values_for_c3fd': [0, int(proposal['N']), *([0] * len(species))],
        'ledger_features_float32': [[0.] * 6] + [normalized_ledger(v) for v in metadata['ledger_steps'][1:]],
        'attention_mask_unpadded': [1] * length,
        'soft_position_indices': length - 1,
    }


def keys_for(plan: dict, goal: str | None) -> dict[str, str]:
    comp = composition(plan)
    divisor = math.gcd(*(n for _, n in comp))
    exact = [[z, n] for z, n in comp]
    reduced = [[z, n // divisor] for z, n in comp]
    hard = {'N': int(plan['N']), 'composition': exact, 'anion_framework': str(plan['anion_framework'])}
    return {
        'reduced_formula_key': canonical_json(reduced),
        'canonical_exact_formula_key': canonical_json(exact),
        'dlm_hard_context_key': canonical_json(hard),
        'planner_condition_key_exact_composition_family_goal': canonical_json({**hard, 'goal': goal}),
    }


def group_summary(records: list[dict], field: str, max_examples: int) -> dict:
    output = {}
    for key_name in ('reduced_formula_key', 'canonical_exact_formula_key', 'dlm_hard_context_key',
                     'planner_condition_key_exact_composition_family_goal'):
        groups = defaultdict(list)
        for record in records:
            groups[record['keys'][key_name]].append(record)
        ambiguous = []
        repeated = 0
        distribution = Counter()
        for key, members in groups.items():
            repeated += len(members) > 1
            variants = defaultdict(list)
            for record in members:
                variants[record[field]].append({'split': record['split'], 'source_row_idx': record['source_row_idx']})
            distribution[len(variants)] += 1
            if len(variants) > 1:
                ambiguous.append({'key': json.loads(key), 'rows': len(members), 'signature_count': len(variants),
                                  'variants': [{'sha256': signature, 'rows': rows[:5]} for signature, rows in variants.items()]})
        ambiguous.sort(key=lambda r: (-r['signature_count'], -r['rows'], canonical_json(r['key'])))
        output[key_name] = {
            'rows': len(records), 'groups': len(groups), 'groups_with_multiple_rows': repeated,
            'groups_with_multiple_signatures': len(ambiguous),
            'rows_in_groups_with_multiple_signatures': sum(r['rows'] for r in ambiguous),
            'signature_count_histogram': dict(sorted(distribution.items())),
            'examples': ambiguous[:max_examples],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pointer-dir', type=Path, required=True)
    parser.add_argument('--sft-dir', type=Path, required=True)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-examples', type=int, default=12)
    args = parser.parse_args()
    receipts, data = [], {}
    for split in ('train', 'val'):
        data[split] = {}
        for name, directory in (('pointer', args.pointer_dir), ('sft', args.sft_dir), ('prepared', args.prepared_dir)):
            rows, receipt = load_rows(directory / (split + '.jsonl'))
            data[split][name] = rows
            receipts.append(receipt)
        if set(data[split]['sft']) != set(data[split]['prepared']):
            raise ValueError(f'{split}: prepared/source row identities differ')

    teacher_records, actual_records, pointer_records = [], [], []
    split_counts, signature_payloads = {}, {}
    for split, tables in data.items():
        counts = Counter()
        for index, source in tables['sft'].items():
            counts['all_sft_rows'] += 1
            plan = source['plan_state']
            prepared = tables['prepared'][index]
            if composition(plan) != composition(prepared['plan_state']):
                raise ValueError(f'{split}:{index}: prepared changed composition')
            if source['answer'] != prepared['answer']:
                raise ValueError(f'{split}:{index}: prepared changed clean answer')
            metadata = tables['pointer'].get(index)
            if metadata is not None:
                atomic = tuple(zip(map(int, metadata['canonical_atomic_numbers']), map(int, metadata['canonical_element_counts'])))
                if atomic != composition(plan):
                    raise ValueError(f'{split}:{index}: pointer/source composition mismatch')
                feed = actual_chemical_inputs(metadata)
                signature = digest(feed)
                signature_payloads.setdefault(signature, feed)
                teacher_records.append({'split': split, 'source_row_idx': index, 'keys': keys_for(plan, metadata['stability_condition']), 'M': signature})
                counts['teacher_metadata_rows'] += 1
            else:
                counts['missing_teacher_metadata_rows'] += 1

            evidence = prepared['condition_prediction']
            status = str(evidence['status'])
            counts['prepared_status:' + status] += 1
            if status == 'teacher_soft_fallback':
                counts['no_actual_planner_forward_rows'] += 1
                continue
            if status not in ('predicted_direct', 'predicted_metadata_reuse'):
                raise ValueError(f'Unexpected preparation status: {status}')
            donor_split = str(evidence['typed_metadata_source_split'])
            donor_index = int(evidence['typed_metadata_source_row_idx'])
            donor = data[donor_split]['pointer'][donor_index]
            feed = actual_chemical_inputs(donor)
            signature = digest(feed)
            signature_payloads.setdefault(signature, feed)
            record = {'split': split, 'source_row_idx': index, 'keys': keys_for(plan, donor['stability_condition']), 'M': signature}
            actual_records.append(record)
            pointer_feed = {'chemical_input_signature': signature,
                            'pointer_atomic_numbers': [Z[symbol] for symbol in plan['elements']],
                            'pointer_counts': [int(c) for c in plan['counts']],
                            'pointer_valid_mask': [True] * len(plan['elements']),
                            'actual_sampled_soft_ids': [int(evidence['soft_ids'][field]) for field in SOFT_FIELDS]}
            pointer_records.append({**record, 'pointer_input': digest(pointer_feed)})
            counts['actual_planner_forward_rows'] += 1
        split_counts[split] = dict(counts)

    analyses = {}
    for stage, records, field in (('teacher_M_in_metadata', teacher_records, 'M'),
                                   ('prepared_actual_M', actual_records, 'M'),
                                   ('prepared_pointer_with_sampled_soft', pointer_records, 'pointer_input')):
        analyses[stage] = {
            subset: group_summary([r for r in records if subset == 'pooled_train_val' or r['split'] == subset], field, args.max_examples)
            for subset in ('train', 'val', 'pooled_train_val')
        }
    varied_signatures = set()
    for stage in analyses.values():
        for subset in stage.values():
            for grouping in subset.values():
                for example in grouping['examples']:
                    for variant in example['variants']:
                        varied_signatures.add(variant['sha256'])
    payload = {
        'schema': 'upstream_actual_collator_signature_audit_v1', 'model_calls': 0, 'new_generation': False,
        'source_files': receipts, 'split_counts': split_counts, 'analyses': analyses,
        'example_M_payloads': {s: signature_payloads[s] for s in sorted(varied_signatures) if s in signature_payloads},
        'semantics': {
            'hashed_M': 'Unpadded actual chemical collator arrays, normalized to float32; includes fixed-ID-equivalent goal and stratum.',
            'not_hashed': ['source_row_idx', 'source_split', 'audit fields', 'teacher soft labels', 'contact-order labels', 'clean answer', 'sampling seed'],
            'constant_context': 'C3FD context and frozen model parameters are common to all rows and omitted.',
            'dlm_hard_context': 'N, canonical exact element counts, anion_framework; goal is not explicitly present in native DLM prompt.',
            'planner_condition': 'DLM hard context plus explicit goal.',
            'pointer_signature': 'Separately includes actual sampled soft IDs; its variation is not evidence that chemical M is non-deterministic.',
            'claim_limit': 'A unique M per observed C supports finite-table determinism; it does not prove out-of-sample determinism, causal irrelevance, or absence of useful computational features.'
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
        print(json.dumps({'output': str(args.output), 'split_counts': split_counts,
                          'pooled_actual_by_planner_condition': analyses['prepared_actual_M']['pooled_train_val']['planner_condition_key_exact_composition_family_goal']}, ensure_ascii=False))
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
