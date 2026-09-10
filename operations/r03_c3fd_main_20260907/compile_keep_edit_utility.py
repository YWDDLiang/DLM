"""Compile signed, source-balanced editor utility without historical U labels."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE/'src'))
from scripts.run_post_refine_cycle import read_rows, write_rows, write_json, file_hash
from scripts.run_rsi_stages import scores, score_directory
from crystal_dlm.post_refine_contract import fingerprint
from crystal_dlm.ranked_feedback import endpoint_quality, align_fixed_slots, target_action, validate_action_target


def utility(score):
    quality = endpoint_quality(score)
    if quality['known_failure']: return 0
    if not quality['reliable']: return None
    stable = score.get('strict_stable') is True
    meta = score.get('meta_stable') is True
    if not stable and not meta: return 0
    if score.get('novel') is None: return None
    return (int(stable)+int(meta)) if score['novel'] is True else 0


def physics_equivalence(old, previous):
    old_source = Path((old/'SOURCE_PATH').read_text().strip())
    new_pipe = json.loads((previous/'RAW0_PIPELINE.json').read_text())
    new_source = Path(new_pipe['source_root'])
    sources = [s/'src/crystal_dlm/ranked_feedback.py' for s in (old_source, new_source)]
    trees = []
    for path in sources:
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if not
            (isinstance(node, ast.FunctionDef) and node.name == 'assign_current_mode_targets')]
        trees.append(ast.dump(tree, include_attributes=False))
    if trees[0] != trees[1]: raise ValueError('historical physical helper logic differs')
    return dict(sources={str(p):file_hash(p) for p in sources},
        all_other_module_AST_equal=True, excluded_nonphysical_function='assign_current_mode_targets')


def compile_data(root):
    root = Path(root); reg = json.loads((root/'PREREGISTRATION.json').read_text())
    previous, old, trial = (Path(reg[k]) for k in ('previous_run','old256_run','previous_trial'))
    equivalence = physics_equivalence(old, previous)
    allowed_joint_hashes = set(equivalence['sources'].values())
    split = read_rows(root/'SOURCE_SPLIT.jsonl'); roles = {r['ancestor_id']:r for r in split}
    plans = read_rows(root/'fit/cohort/plans.jsonl'); frozen_plans = {p['ancestor_id']:p for p in plans}
    if file_hash(root/'SOURCE_SPLIT.jsonl') != reg['source_split_sha256']: raise ValueError('source split changed')
    fixed = Path(reg['fixed_input'])
    base_physics = json.loads((fixed/'current/labeling/result/LABEL_FINAL.json').read_text())
    base_score_path = next(score_directory(fixed,'current').glob('*FINAL.json'))
    base_score = json.loads(base_score_path.read_text())
    official = {r['chemsys']:fingerprint(r) for r in read_rows(previous/'fit_hull/official_mp_cache/official_slim_cache.jsonl')}
    cache_checks = {}
    def normalized(runtime): return {k:v for k,v in runtime.items() if k != 'joint_stop_source_sha256'}
    base_runtime = [normalized(x) for x in base_physics['runtime_identities']]
    def verified_stage(parent, name):
        stage = parent/name; inputs = stage/'inputs.jsonl'; label = stage/'labeling/result/LABEL_FINAL.json'
        report = json.loads(label.read_text()); measured = scores(parent, name)
        score_path = next(score_directory(parent, name).glob('*FINAL.json')); score_report = json.loads(score_path.read_text())
        if report['input_sha256'] != file_hash(inputs) or not (label.parent/'_SUCCESS').exists():
            raise ValueError('unbound historical endpoint labels')
        for key in ('protocol','verification_protocol','geometry_validation_protocol'):
            if report[key] != base_physics[key]: raise ValueError('historical protocol differs:'+key)
        if any(normalized(x) not in base_runtime or x['joint_stop_source_sha256'] not in allowed_joint_hashes
               for x in report['runtime_identities']): raise ValueError('historical physical runtime differs')
        for key in ('frozen_config_sha256','physical_model_sha256','frozen_nu_source_sha256','novelty_uniqueness_endpoint'):
            if score_report[key] != base_score[key]: raise ValueError('historical scoring identity differs:'+key)
        cache = Path(score_report['official_cache'])/'official_slim_cache.jsonl'
        if str(cache) not in cache_checks:
            if file_hash(cache) != score_report['official_cache_sha256']: raise ValueError('historical official cache changed')
            entries = read_rows(cache)
            if any(official.get(r['chemsys']) != fingerprint(r) for r in entries):
                raise ValueError('historical hull entries differ from the fixed current cache')
            cache_checks[str(cache)] = dict(sha256=file_hash(cache), exact_entries_subset=True, chemsystems=len(entries))
        pins = {str(p):file_hash(p) for p in (inputs,label,label.parent/'labels.jsonl',score_path,
                                             score_directory(parent,name)/'attempt_results.jsonl')}
        return read_rows(inputs), measured, pins
    old_parents = [old/'fit'] + [old/f'rounds/round{k}/fit' for k in range(1,4)]
    old_parents += [p/'editor_collection' for p in old_parents[1:]]
    new_parents = [previous/'fit',previous/'bootstrap_comparator',fixed,fixed/'editor_collection']
    parents = old_parents + sorted(p for p in (trial/'collections').iterdir() if p.is_dir()) + new_parents
    examples = {}; evidence = {}; exclusions = Counter(); identity_seen = set()
    for parent in parents:
        if not (parent/'current/inputs.jsonl').exists(): continue
        parent_plans = read_rows(parent/'cohort/plans.jsonl')
        for plan in parent_plans:
            frozen = frozen_plans[plan['ancestor_id']]
            if any(plan[k] != frozen[k] for k in ('plan_state','body_prompt','source_row_idx','source_split')):
                raise ValueError('same ancestor changed fixed Plan')
        before, before_scores, before_pins = verified_stage(parent,'current')
        evidence.update(before_pins)
        evidence[str(parent/'cohort/plans.jsonl')] = file_hash(parent/'cohort/plans.jsonl')
        for name in ('proposal','teacher'):
            if not (parent/name/'inputs.jsonl').exists(): continue
            after, after_scores, after_pins = verified_stage(parent,name); evidence.update(after_pins)
            for plan, a, b, x, y in zip(parent_plans,before,after,before_scores,after_scores,strict=True):
                source = plan['ancestor_id']; role = roles[source]
                if role['split'] != 'train': continue
                if (a['trajectory_id'] != x['trajectory_id'] or b['trajectory_id'] != y['trajectory_id'] or
                        not a['sample_idx'] == b['sample_idx'] == x['sample_idx'] == y['sample_idx']):
                    raise ValueError('historical paired source identities differ')
                left = a.get('body_token_ids'); right = b.get('body_token_ids')
                if not left or not right: exclusions['missing_token_view'] += 1; continue
                u, v = utility(x), utility(y)
                if u is None or v is None: exclusions['unknown_physics_or_needed_novelty'] += 1; continue
                try:
                    right, permutation = align_fixed_slots(right, left)
                    action = target_action(left, right)
                    record_path = parent/name/'records'/f"{plan['original_ordinal']:04d}.json"
                    wrapper = json.loads(record_path.read_text()) if record_path.exists() else {}
                    trace = wrapper.get('editor_trace',{}); actual_positions = trace.get('action',{}).get('positions')
                    positions = actual_positions if actual_positions is not None and permutation == list(range(len(permutation))) else action['positions']
                    validate_action_target(left,right,positions)
                except (ValueError,KeyError,TypeError) as error:
                    exclusions[str(error)] += 1; continue
                pair_id = fingerprint(dict(source=source,current=left,proposal=right))
                row = dict(pair_id=pair_id, source_id=source, source_split='train', focus_split='train',
                    source_ordinal=role['ordinal'], old_E_seen_source=role['old_E_seen_source'],
                    prompt=plan['body_prompt'], num_sites=plan['plan_state']['N'],
                    current_tokens=left, proposal_tokens=right, action_positions=positions,
                    utility_target=v-u, before_utility=u, after_utility=v, historical_U_reused=False,
                    source_trace=dict(current=str(parent/'current'), proposal=str(parent/name),
                        before_score_sha256=fingerprint(x), after_score_sha256=fingerprint(y)),
                    training_kind='observed_pair')
                if pair_id in examples:
                    if examples[pair_id]['utility_target'] != row['utility_target']:
                        raise ValueError('duplicate physical pair has conflicting verified utility')
                    exclusions['duplicate_pair'] += 1
                else: examples[pair_id] = row
                identity_id = fingerprint(dict(source=source,current=left,proposal=left))
                if identity_id not in examples:
                    examples[identity_id] = dict(row,pair_id=identity_id,proposal_tokens=left,
                        action_positions=[],utility_target=0,after_utility=u,training_kind='exact_KEEP_anchor')
    rows = [examples[k] for k in sorted(examples)]; multiplicity = Counter(r['source_id'] for r in rows)
    for row in rows: row['source_weight'] = 1./multiplicity[row['source_id']]
    if any(roles[r['source_id']]['split']!='train' for r in rows): raise ValueError('source leakage')
    path = root/'data/UTILITY_TRAIN.jsonl'; write_rows(path,rows)
    report = dict(schema='signed_editor_utility_training_v1', examples=len(rows),
        sources=len(multiplicity), source_split_sha256=file_hash(root/'SOURCE_SPLIT.jsonl'),
        data_sha256=file_hash(path), target_counts=dict(Counter(r['utility_target'] for r in rows)),
        old_E_seen_sources=sum(roles[s]['old_E_seen_source'] for s in multiplicity),
        old_E_unseen_train_sources=sum(not roles[s]['old_E_seen_source'] for s in multiplicity),
        historical_U_reused=False, complete_source_weight_sum=len(multiplicity),
        physics_equivalence=equivalence, official_cache_checks=cache_checks,
        evidence=evidence, exclusions=dict(exclusions))
    write_json(root/'data/UTILITY_DATA_FINAL.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('evidence','official_cache_checks','physics_equivalence')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    compile_data(parser.parse_args().root)
