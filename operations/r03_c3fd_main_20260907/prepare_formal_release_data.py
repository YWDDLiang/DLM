"""Export exact Plan presets and composition-isolated formal training requests."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / 'src'))
from scripts.run_post_refine_cycle import read_rows, write_rows, write_json, file_hash
from crystal_dlm.post_refine_contract import derived_seed
from crystal_dlm.sun_feedback_contract import composition_counts, reduced_key


def composition(row):
    return reduced_key(composition_counts(row['plan_state']))


def export(root):
    previous = Path(json.loads((root / 'REGISTRATION.json').read_text())['previous'])
    assets = json.loads((previous / 'fit/RUN_SPEC.json').read_text())['assets']
    output = root / 'formal_preparation/portable'
    if (output / 'PREPARATION.json').exists():
        print((output / 'PREPARATION.json').read_text())
        return
    output.mkdir(parents=True, exist_ok=True)
    cohort = read_rows(assets['cohort'])
    seeds = {r['ordinal']: r for r in read_rows(assets['seed_ledger'])}
    presets = {}
    for preset, rows in [('H1A2_1200', cohort),
                         ('R03_256', read_rows(root / 'formal_preparation/R03_256_PLANS.jsonl'))]:
        plans = []
        for i, row in enumerate(rows):
            original = row.get('cohort_ordinal', row.get('original_ordinal', i))
            seed = seeds[original] if preset == 'H1A2_1200' else row
            plans.append(dict(schema='crystal_plan_v1', source_id=f'{preset}:{original:04d}',
                original_ordinal=original, plan_state=row.get('plan_state'),
                body_eligible=row.get('body_eligible', False), ineligible_reason=row.get('ineligible_reason'),
                body_prompt=row.get('body_prompt'), body_noise_seed=seed['body_noise_seed'],
                refiner_noise_seed=seed['refiner_noise_seed'], raw_plan_text=row.get('raw_plan_text'),
                provenance=dict(dataset_origin='planner_generated', original_split='generated',
                    usage_role='evaluation', preset=preset)))
        path = output / (preset + '.jsonl')
        write_rows(path, plans)
        presets[preset] = dict(rows=len(plans), legal=sum(r['body_eligible'] for r in plans),
                               sha256=file_hash(path))
    forbidden = set()
    exclusion_sources = {}
    def exclude(name, rows):
        keys = set()
        for row in rows:
            try:
                keys.add(composition(row))
            except (ValueError, KeyError, TypeError):
                if row.get('body_eligible'):
                    raise
        forbidden.update(keys)
        exclusion_sources[name] = len(keys)
    exclude('all_H1A2_1200', cohort)
    exclude('all_R03_256', read_rows(root / 'formal_preparation/R03_256_PLANS.jsonl'))
    fit = read_rows(previous / 'fit/cohort/plans.jsonl')
    roles = read_rows(previous / 'SOURCE_SPLIT.jsonl')
    if len(fit) != len(roles):
        raise ValueError('previous role and source tables differ')
    heldout = []
    for plan, role in zip(fit, roles, strict=True):
        if plan['ancestor_id'] != role['ancestor_id']:
            raise ValueError('previous role is bound to a different source')
        if role['split'] != 'train':
            heldout.append(plan)
    exclude('previous_dev_and_exploratory_final', heldout)
    exclude('previous_fresh_256', read_rows(previous / 'fresh/cohort/plans.jsonl'))
    parent = Path(assets['training_preparation']) / 'pairs_pending.jsonl'
    candidates = sorted(read_rows(parent), key=lambda r: (r['source_row_idx'], r['ancestor_id']))
    seen_sources, seen_compositions = set(), set()
    clean, excluded = [], Counter()
    for row in candidates:
        plan = row['plan_state']
        try:
            key = composition(row)
        except (ValueError, KeyError, TypeError):
            excluded['invalid_plan'] += 1
            continue
        if plan.get('rich_field_valid') is not True or plan.get('plan_end_marker_present') is not True:
            excluded['invalid_rich_plan'] += 1
            continue
        if key in forbidden:
            excluded['evaluation_or_validation_composition'] += 1
            continue
        if row['ancestor_id'] in seen_sources or key in seen_compositions:
            excluded['duplicate_source_or_reduced_composition'] += 1
            continue
        seen_sources.add(row['ancestor_id']); seen_compositions.add(key)
        clean.append(dict(schema='crystal_plan_v1', source_id=row['ancestor_id'],
            ancestor_id=row['ancestor_id'], original_ordinal=row['source_row_idx'],
            source_row_idx=row['source_row_idx'], plan_state=plan, body_prompt=row['prompt'],
            body_eligible=True, ineligible_reason=None,
            body_noise_seed=derived_seed(row['ancestor_id'], 'formal_train_G'),
            refiner_noise_seed=derived_seed(row['ancestor_id'], 'formal_train_F'),
            provenance=dict(dataset_origin='planner_generated', original_split='generated', usage_role='train',
                source_row_idx_semantics='synthetic_generation_request_index',
                collection='expanded_6144_requests')))
    write_rows(output / 'CLEAN_TRAIN_POOL.jsonl', clean)
    write_rows(output / 'CLEAN_TRAIN_1000.jsonl', clean[:1000])
    report = dict(presets=presets, synthetic_parent_sha256=file_hash(parent),
        original_cohort_sha256=file_hash(assets['cohort']), original_seed_ledger_sha256=file_hash(assets['seed_ledger']),
        source_requests=len(candidates), clean_unique_compositions=len(clean),
        selected_train_requests=len(clean[:1000]), selection='original_generation_order_after_source_and_composition_exclusion',
        no_physical_outcomes_used_for_request_selection=True, exclusion_sources=exclusion_sources,
        forbidden_reduced_compositions=len(forbidden), exclusions=dict(excluded),
        clean_train_pool_sha256=file_hash(output / 'CLEAN_TRAIN_POOL.jsonl'),
        clean_train_1000_sha256=file_hash(output / 'CLEAN_TRAIN_1000.jsonl'),
        train_eval_composition_overlap=len(seen_compositions & forbidden),
        supervised_pair_status='not_compiled; only physically reliable, exact-endpoint-bound training feedback will be used')
    write_json(output / 'PREPARATION.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    export(parser.parse_args().root)
