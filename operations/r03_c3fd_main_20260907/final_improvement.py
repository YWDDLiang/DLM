"""Prepare and dispatch the user's final, feasibility-focused editor experiment."""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE/'src'), str(Path(__file__).resolve().parent)]
from scripts.run_post_refine_cycle import read_rows, write_rows, write_json, file_hash
from scripts.run_rsi_stages import scores
from crystal_dlm.post_refine_contract import fingerprint
from crystal_dlm.sun_ranker import endpoint_targets
from crystal_dlm.ranked_feedback import endpoint_quality


def quality(score):
    target, status = endpoint_targets(score)
    q = endpoint_quality(score)
    if target is None or not q['reliable']:
        return None
    hull = score.get('e_above_hull_eV_atom')
    return (target[0], float(score.get('strict_stable') is True),
            float(score.get('meta_stable') is True), -float(hull) if hull is not None else -1e6)


def improves(after, before):
    if after is None:
        return False
    if before is None:
        return True
    return after[:3] > before[:3] or (after[:3] == before[:3] and after[3] > before[3] + .01)


def prepare(root, previous):
    from run_component import verify_deployed_source
    reg = json.loads((previous/'PREREGISTRATION.json').read_text())
    origin = Path(reg['previous_run'])
    spec = json.loads((previous/'fit/RUN_SPEC.json').read_text())
    split = read_rows(previous/'SOURCE_SPLIT.jsonl')
    current = scores(origin/'fit', 'native')
    per_source = defaultdict(list)
    discarded = Counter()
    for stream in ('primary', 'rank1', 'rank2', 'rank3'):
        bank = previous/'fit/bank'/stream
        after = scores(origin/'fit', 'hybrid_proposal') if stream == 'primary' else scores(bank, 'candidate')
        for row in read_rows(bank/'FEATURE_ROWS.jsonl'):
            i = row['ordinal']
            if split[i]['split'] != 'train':
                continue
            bound = json.loads((bank/f'bound/{i:04d}.json').read_text())
            if not bound['commit_trace']['applied']:
                discarded['unchanged_or_invalid_edit'] += 1
                continue
            a, b = quality(current[i]), quality(after[i])
            if a is None or b is None:
                discarded['unreliable_physics'] += 1
                continue
            if row['source_id'] != split[i]['ancestor_id']:
                raise ValueError('source identity mismatch')
            per_source[i].append(dict(row, stream=stream, before_quality=a, after_quality=b,
                                     better=improves(b, a), before_score_sha256=fingerprint(current[i]),
                                     after_score_sha256=fingerprint(after[i])))
    data = []
    for i, candidates in sorted(per_source.items()):
        best = max(candidates, key=lambda r: (r['better'], r['after_quality'], r['pair_id']))
        # One current-condition mode and one content teacher per source. An
        # independently sampled proposal supplies positive or negative judge labels.
        judge = min(candidates, key=lambda r: fingerprint({'judge': r['pair_id']}))
        edit = best['better']
        positions = best['action_positions']
        sites = sorted({(p - 8) // 4 for p in positions if p >= 8})
        target = best['proposal_tokens'] if edit else best['current_tokens']
        data.append(dict(pair_id=fingerprint({'source': best['source_id'], 'experiment': 'final_content_v1'}),
            source_id=best['source_id'], source_split='train', ordinal=i, num_sites=best['num_sites'],
            prompt=best['prompt'], current_tokens=best['current_tokens'],
            content_target_tokens=target, content_positions=positions,
            proposal_tokens=judge['proposal_tokens'], action_positions=judge['action_positions'],
            mode_target=1 if edit else 0,
            site_targets=[int(site in sites) for site in range(best['num_sites'])],
            accept_target=float(judge['better']), known_sun=current[i].get('strict_sun') is True,
            objective_level='SUN' if edit and best['after_quality'][0] > best['before_quality'][0]
                else 'strict_stable' if edit and best['after_quality'][1] > best['before_quality'][1]
                else 'ordinary_improvement' if edit else 'identity_anchor',
            teacher_provenance=dict(stream=best['stream'], before_score_sha256=best['before_score_sha256'],
                after_score_sha256=best['after_score_sha256'], before_quality=best['before_quality'],
                after_quality=best['after_quality']),
            judge_provenance=dict(stream=judge['stream'], after_score_sha256=judge['after_score_sha256'])))
    root.mkdir(parents=True, exist_ok=True)
    data_path = root/'data/E.jsonl'
    write_rows(data_path, data)
    manifest = dict(source_split='train', sources=len(data), source_selection='existing composition-separated TRAIN only',
        files_sha256={'E.jsonl': file_hash(data_path)}, source_bank=str(previous),
        source_split_sha256=file_hash(previous/'SOURCE_SPLIT.jsonl'),
        objectives=dict(Counter(r['objective_level'] for r in data)), discarded=dict(discarded),
        validation_role='previously viewed DEV for feasibility; not a blind final test',
        formal_data_requirement='fresh data preparation with source/composition exclusions and reliable labels')
    write_json(root/'data/PAIRS_E_FINAL.json', manifest)
    write_json(root/'REGISTRATION.json', dict(previous=str(previous),
        start_utc='2026-09-10T13:02:01+00:00', default_deadline_utc='2026-09-10T17:02:01+00:00',
        feasibility='effective content and site learning plus real positive KEEP/EDIT examples and same-input comparison',
        methods=['minibatch content updates', 'correct content/decision learning rates',
                 'KL regularization without early stop', 'consistent atom permutations', 'categorical site supervision'],
        decisions='candidate absolute-state prediction; do not require nonnegative predicted MSUN increment',
        formal_evaluation='first 1050 legal H1A2 Plans, fixed original order, three improvement rounds',
        input_checkpoint=spec['assets']['editor_checkpoint'], training_data=manifest))
    pipe = json.loads((previous/'RAW0_PIPELINE.json').read_text())
    pipe.update(run_root=str(root), source_root=str(SOURCE), source_identity=verify_deployed_source(SOURCE))
    pipe['resources'].update(deadline_utc='2026-09-10T17:02:01+00:00',
        extra_gpu_until_utc='2026-09-10T17:02:01+00:00',gpus_before_extra_window=5,
        gpus_after_extra_window=5,max_submitted_slurm_jobs=3)
    pipe['components'] = []; pipe['jobs'] = {}
    write_json(root/'PIPELINE.json', pipe)
    for name, lr in [('mini_2e6', 2e-6), ('mini_5e7', 5e-7)]:
        cfg = dict(branch='E', base_model=spec['assets']['base_model'], checkpoint=spec['assets']['editor_checkpoint'],
            data=str(data_path), output_dir=str(root/'training'/name/'result'), seed=20260910,
            ranked_training=True, bounded_minibatch_training=True, editor_minibatch_content=True,
            permute_atoms=True, learning_rate=lr, head_learning_rate=1e-4, reference_kl_weight=1.,
            reference_KL_hard_stop=False, max_reference_kl=.02, mask_cuts=2, batch_size=16,
            epochs=8, max_training_seconds=1800, beta=.1, anchor_weight=1.)
        write_json(root/'configs'/f'{name}.json', cfg)
    print(json.dumps(manifest), flush=True)


def train(root, name):
    from run_component import verify_deployed_source
    from submit_stage import configured_dispatch
    pipe = json.loads((root/'PIPELINE.json').read_text())
    pipe.update(source_root=str(SOURCE), source_identity=verify_deployed_source(SOURCE))
    cfg = root/'configs'/f'{name}.json'
    job = 'final_' + name
    pipe['components'] = [dict(id=job, output_dir=f'training/{name}', gpus=1, stages=[
        dict(name='train', script='src/scripts/train_rsi_preferences.py', args=['--config', str(cfg)],
             inputs=[str(cfg), str(root/'data/E.jsonl'), str(root/'data/PAIRS_E_FINAL.json')],
             outputs=['{output}/TRAINING_FINAL.json', '{output}/checkpoint/RSI_TRAINING_DONE.json'])])]
    pipe['jobs'] = {job: dict(component_indices=[0],gpus_per_task=1,cpus_per_task=4,
        parallel_tasks=1,wall_minutes=40,memory='96G',partition='gpu')}
    path = root/f'{job}_PIPELINE.json'
    write_json(path, pipe)
    configured_dispatch(['--config', str(path), '--job', job])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'train'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--name', choices=['mini_2e6', 'mini_5e7'])
    args = parser.parse_args()
    prepare(args.root, args.previous) if args.mode == 'prepare' else train(args.root, args.name)
