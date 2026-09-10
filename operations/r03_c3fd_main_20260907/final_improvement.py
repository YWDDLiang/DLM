"""Prepare and dispatch the user's final, feasibility-focused editor experiment."""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import shutil
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
             outputs=['{output}/result/TRAINING_FINAL.json', '{output}/result/checkpoint/RSI_TRAINING_DONE.json'])])]
    pipe['jobs'] = {job: dict(component_indices=[0],gpus_per_task=1,cpus_per_task=4,
        parallel_tasks=1,wall_minutes=40,memory='96G',partition='gpu')}
    path = root/f'{job}_PIPELINE.json'
    write_json(path, pipe)
    configured_dispatch(['--config', str(path), '--job', job])


def probe(root):
    from run_component import verify_deployed_source
    from submit_stage import configured_dispatch
    reg = json.loads((root/'REGISTRATION.json').read_text())
    previous = Path(reg['previous'])
    panel = root/'failure_probe'
    spec = json.loads((previous/'fit/RUN_SPEC.json').read_text())
    plans = read_rows(previous/'fit/cohort/plans.jsonl')
    parents = read_rows(previous/'fit/cohort/parents.jsonl')
    selected = [dict(plans[i], evaluation_ordinal=j) for j, i in enumerate([549, 837])]
    write_rows(panel/'cohort/plans.jsonl', selected)
    write_rows(panel/'cohort/parents.jsonl', [parents[i] for i in [549, 837]])
    write_json(panel/'cohort/PREPARATION_FINAL.json',dict(files_sha256={
        'parents.jsonl':file_hash(panel/'cohort/parents.jsonl')}, source='fixed failure diagnostics 549 and 837'))
    spec.update(run_root=str(panel), run_id='final_failure_probe', requests=2,
                training_parent_root=str(panel/'cohort'))
    spec['policy']['adaptive_lattice_recovery'] = True
    cfg = panel/'RUN_SPEC.json'
    write_json(cfg, spec)
    pipe = json.loads((root/'PIPELINE.json').read_text())
    pipe.update(source_root=str(SOURCE), source_identity=verify_deployed_source(SOURCE))
    pipe['components'] = [dict(id='failure_probe', output_dir='failure_probe', gpus=1,
        stages=[dict(name='generate', script='src/scripts/run_post_refine_cycle.py',
            args=['--config',str(cfg),'--stage','construct'], inputs=[str(cfg)],
            outputs=['{output}/construction/records/0549.json','{output}/construction/records/0837.json',
                     '{output}/construction/worker_0_DONE.json'])])]
    pipe['jobs'] = {'failure_probe':dict(component_indices=[0],gpus_per_task=1,cpus_per_task=4,
        parallel_tasks=1,wall_minutes=15,memory='96G',partition='gpu')}
    path = root/'FAILURE_PROBE_PIPELINE.json'
    write_json(path, pipe)
    configured_dispatch(['--config',str(path),'--job','failure_probe'])


def variant(root, name):
    previous = Path(json.loads((root/'REGISTRATION.json').read_text())['previous'])
    destination = root/'variants'/name
    if (destination/'PREREGISTRATION.json').exists():
        return destination
    reg = json.loads((previous/'PREREGISTRATION.json').read_text())
    frozen=name in ('frozen_e3','balanced_e3','retained_e3','retained_delta_e3')
    checkpoint = Path(reg['old_editor']) if frozen else root/'training'/name/'result/checkpoint'
    receipt = checkpoint/'RSI_TRAINING_DONE.json'
    reg.update(old_editor=str(checkpoint), old_editor_receipt_sha256=file_hash(receipt),
               editor_content_changed=not frozen, parent_content_experiment=str(root),enforce_shared_forward_budget=True)
    if name=='balanced_e3':
        reg['head_training']=dict(reg['head_training'],balanced_keep_pairs=True,
            SUN_pairwise_coefficient=2.,ridge=.01,device='cpu')
    for folder in ('cohort','current','native'):
        shutil.copytree(previous/'fit'/folder, destination/'fit'/folder)
    shutil.copy2(previous/'SOURCE_SPLIT.jsonl', destination/'SOURCE_SPLIT.jsonl')
    shutil.copy2(previous/'PHYSICS_SOURCE_PIN.json', destination/'PHYSICS_SOURCE_PIN.json')
    spec = json.loads((previous/'fit/RUN_SPEC.json').read_text())
    spec.update(run_root=str(destination/'fit'), run_id=name, training_parent_root=str(destination/'fit/cohort'))
    spec['assets']['editor_checkpoint'] = str(checkpoint)
    spec['assets']['nu_cache'] = str(root/'nu_cache')
    write_json(destination/'fit/RUN_SPEC.json',spec)
    write_json(destination/'PREREGISTRATION.json',reg)
    write_json(destination/'LEARNED_KEEP_REGISTRATION.json',dict(schema='learned_KEEP_candidate_comparison_v1',
        decision='argmax predicted NS then NMS over learned KEEP and edit views',absolute_acceptance_floor=None,
        MS_gain_floor=None,known_SUN_hard_veto=False,role='feasibility'))
    if frozen:
        for stream in ('primary','rank1','rank2','rank3'):
            link=destination/'fit/bank'/stream
            link.parent.mkdir(parents=True,exist_ok=True)
            link.symlink_to(previous/'fit/bank'/stream,target_is_directory=True)
        if name in ('balanced_e3','retained_e3','retained_delta_e3'):
            (destination/'fit/bank/keep').symlink_to(root/'variants/frozen_e3/fit/bank/keep',target_is_directory=True)
    return destination


def prepare_scope(root):
    from pymatgen.core import Composition
    from pymatgen.analysis.phase_diagram import PDEntry,PhaseDiagram
    previous=Path(json.loads((root/'REGISTRATION.json').read_text())['previous'])
    spec=json.loads((previous/'fit/RUN_SPEC.json').read_text())
    prep=Path(spec['assets']['training_preparation'])
    roles=read_rows(previous/'SOURCE_SPLIT.jsonl')
    allowed={r['ancestor_id'] for r in roles if r['split']=='train' and r['ordinal'] not in (549,837)}
    parents=[r for r in read_rows(prep/'pairs_pending.jsonl') if r['ancestor_id'] in allowed]
    ids={r[k] for r in parents for k in ('old_physics_id','target_physics_id') if r.get(k)}
    labels={r['trajectory_id']:r for r in read_rows(prep.parent/'labels/labels.jsonl') if r['trajectory_id'] in ids}
    inputs={r['trajectory_id']:r for r in read_rows(prep/'all_inputs.jsonl') if r['trajectory_id'] in ids}
    tok=json.loads((Path(spec['assets']['b0_checkpoint'])/'tokenizer.json').read_text())
    inverse={i:t for t,i in tok['model']['vocab'].items()}
    inverse.update({r['id']:r['content'] for r in tok.get('added_tokens',[])})
    systems={'-'.join(sorted(r['plan_state']['elements'])) for r in parents}
    diagrams={}
    for r in read_rows(Path(spec['assets']['official_cache'])/'official_slim_cache.jsonl'):
        if r['chemsys'] in systems:
            diagrams[r['chemsys']]=PhaseDiagram([PDEntry(Composition(e['composition']),float(e['energy'])) for e in r['entries']])
    data=read_rows(root/'data/E.jsonl');added=[];excluded=Counter()
    for row in parents:
        before=labels.get(row['old_physics_id'],{});after=labels.get(row.get('target_physics_id'),{})
        system='-'.join(sorted(row['plan_state']['elements']))
        if (not row.get('target_body') or after.get('status')!='verified' or not after.get('verified')
                or system not in diagrams or not row.get('target_geometry',{}).get('valid')):
            excluded['unavailable_or_unverified_target']+=1;continue
        hull=float(diagrams[system].get_hull_energy_per_atom(Composition(dict(zip(row['plan_state']['elements'],row['plan_state']['counts'])))))
        target_hull=float(after['terminal_energy'])-hull
        if target_hull>.1:
            excluded['target_not_stable_or_metastable']+=1;continue
        good=(before.get('status') in ('invalid_raw','invalid_terminal','relaxation_energy_increased') or
              (before.get('status')=='verified' and after['terminal_energy']<before['terminal_energy']-.01))
        if not good:
            excluded['no_reliable_pair_improvement']+=1;continue
        for label_key,body_key in [('old_physics_id','old_body'),('target_physics_id','target_body')]:
            raw=inputs[row[label_key]]
            if raw.get('body')!=''.join(inverse[t] for t in row[body_key]):
                raise ValueError('cached physical label belongs to a different teacher token body')
        action=row['action'];mode=('none','local_xyz','all_xyz','full_cell').index(action['mode'])
        if mode not in (2,3):
            excluded['not_a_global_scope_teacher']+=1;continue
        item=dict(pair_id=fingerprint({'source':row['ancestor_id'],'kind':'trusted_global_scope_teacher_v1'}),
            source_id=row['ancestor_id'],source_split='train',num_sites=row['num_atoms'],prompt=row['prompt'],
            current_tokens=row['old_body'],content_target_tokens=row['target_body'],
            content_positions=action['positions'],proposal_tokens=row['target_body'],action_positions=action['positions'],
            mode_target=mode,site_targets=[0]*row['num_atoms'],accept_target=1.,known_sun=False,
            objective_level='strict_stable' if target_hull<=0 else 'meta_stable',
            teacher_provenance=dict(source='cached_B0_to_F800',old_status=before['status'],target_hull=target_hull,
                label_protocol_max_steps=500,target_physically_verified=True,
                before_label_sha256=fingerprint(before),after_label_sha256=fingerprint(after)))
        data.append(item);added.append(item)
    path=root/'data/E_SCOPE.jsonl';write_rows(path,data)
    manifest=dict(source_split='train',files_sha256={path.name:file_hash(path)},rows=len(data),global_teachers=len(added),
        global_stable=sum(r['objective_level']=='strict_stable' for r in added),
        global_source_ids=[r['source_id'] for r in added],exclusions=dict(excluded),
        explicitly_excluded_diagnostic_ordinals=[549,837],cached_target_labels_bound_to_exact_token_bodies=True,
        base_local_training_data_sha256=file_hash(root/'data/E.jsonl'),
        cached_label_receipt_sha256=file_hash(prep.parent/'labels/LABEL_FINAL.json'))
    write_json(root/'data/PAIRS_E_SCOPE_FINAL.json',manifest)
    cfg=json.loads((root/'configs/mini_2e6.json').read_text())
    cfg.update(data=str(path),output_dir=str(root/'training/scope_2e6/result'))
    write_json(root/'configs/scope_2e6.json',cfg)
    print(json.dumps(manifest),flush=True)


def run_variant(root, name, mode, stream=None, gpus=1):
    from run_component import verify_deployed_source
    from submit_stage import configured_dispatch
    destination = variant(root,name)
    pipe = json.loads((root/'PIPELINE.json').read_text())
    pipe.update(source_root=str(SOURCE), source_identity=verify_deployed_source(SOURCE))
    job = name + '_' + mode + ('_' + stream if stream else '')
    if mode in ('collect','collect_resume'):
        relative = f'variants/{name}/fit/'+('collection_resume' if mode=='collect_resume' else 'collection')
        stage = dict(name='collect',script='src/scripts/run_sun_rank_scope.py',
            args=['--root',str(destination),'--mode','collect','--panel','fit','--completion-dir','{output}'],
            inputs=[str(destination/'PREREGISTRATION.json')], outputs=['{output}/worker_0_DONE.json'])
        minutes = 30
    elif mode == 'physics':
        bank = destination/'fit/bank'/stream
        relative = f'variants/{name}/fit/bank/{stream}/candidate/labeling'
        origin = Path(json.loads((destination/'PREREGISTRATION.json').read_text())['previous_run'])
        stage = dict(name='label',script='scripts/label_rsi_cached_endpoints.py',args=[
            '--input-jsonl',str(bank/'candidate/inputs.jsonl'),'--output-dir','{output}/result',
            '--purpose','training_feedback','--gpu-count',str(gpus),'--workers-per-gpu','4',
            '--record-timeout','600','--deterministic','--feedback-manifest',str(bank/'candidate/FEEDBACK_MANIFEST.json'),
            '--reuse-endpoints',str(origin/'fit/native/labeling/result'),str(origin/'fit/hybrid_proposal/labeling/result'),
            '--joint-physical-stop','--max-steps','1000'],
            inputs=[str(bank/'candidate/inputs.jsonl'),str(bank/'candidate/FEEDBACK_MANIFEST.json')],
            outputs=['{output}/result/LABEL_FINAL.json','{output}/result/labels.jsonl'])
        minutes = 65
    elif mode == 'score':
        gpus=0
        bank=destination/'fit/bank'/stream
        relative=f'variants/{name}/fit/bank/{stream}/candidate/scoring'
        stage=dict(name='score',script='operations/r03_c3fd_main_20260907/final_improvement.py',
            args=['score_worker','--root',str(root),'--name',name,'--stream',stream],
            inputs=[str(bank/'candidate/inputs.jsonl'),str(bank/'candidate/labeling/result/labels.jsonl')],
            outputs=['{output}/result/FEEDBACK_FINAL.json','{output}/result/attempt_results.jsonl'])
        minutes=30
    else:
        relative = f'variants/{name}/{mode}_execution'
        stage = dict(name=mode, script='operations/r03_c3fd_main_20260907/final_improvement.py',
            args=[mode+'_worker','--root',str(root),'--name',name],inputs=[str(destination/'PREREGISTRATION.json')],
            outputs=['{output}/DONE.json'])
        minutes = 35
    pipe['components'] = [dict(id=job,output_dir=relative,gpus=gpus,stages=[stage])]
    pipe['jobs'] = {job:dict(component_indices=[0],gpus_per_task=gpus,cpus_per_task=4*gpus if gpus else 8,
        parallel_tasks=1,wall_minutes=minutes,memory=f'{96*gpus}G' if gpus else '64G',partition='gpu' if gpus else 'normal')}
    path = root/(job+'_PIPELINE.json')
    write_json(path,pipe)
    configured_dispatch(['--config',str(path),'--job',job])


def rank_worker(root,name):
    from scripts.train_sun_ranker import train
    destination=root/'variants'/name
    reg=json.loads((destination/'PREREGISTRATION.json').read_text())
    if name not in ('frozen_e3','balanced_e3'):
        for stream in ('primary','rank1','rank2','rank3'):
            score_bank(root,name,stream,nu_workers=3)
    if name!='balanced_e3':collect_keep_features(destination)
    train(destination,destination/'training/sun_ranker')
    training=destination/'training/sun_ranker/TRAINING_FINAL.json'
    report=json.loads(training.read_text())
    write_json(destination/'ABSOLUTE_STATE_REGISTRATION.json',dict(base_training_sha256=file_hash(training),
        base_train_rows_sha256=report['train_data_sha256'],sun_probability_thresholds=[.1],
        no_per_candidate_MS_gain_veto=True,feasibility_only=True))
    train(destination,destination/'training/nested_sun_ranker',nested=True)
    write_json(destination/'rank_execution/DONE.json',dict(complete=True))


def retained_worker(root,name):
    """Evaluate the retained state readout on KEEP without retraining it."""
    import torch
    from scripts.train_sun_ranker import features
    from crystal_dlm.sun_ranker import nested_probabilities
    torch.set_num_threads(1)
    destination=root/'variants'/name
    original=Path(json.loads((root/'REGISTRATION.json').read_text())['previous'])
    is_delta=name=='retained_delta_e3'
    model_path=original/'training'/('sun_ranker' if is_delta else 'nested_sun_ranker')/'SUN_RANKER.pt'
    state=torch.load(model_path,map_location='cpu',weights_only=False)
    write_json(destination/'RETAINED_READOUT_REGISTRATION.json',dict(
        model_path=str(model_path),model_sha256=file_hash(model_path),retrained=False,
        current_as_learned_KEEP=not is_delta,
        KEEP_reference='exact_zero_self_difference' if is_delta else 'model_current_view',
        score_rules=['2_delta_NS_plus_delta_NMS'] if is_delta else ['NS_then_NMS','2_NS_plus_NMS'],
        no_absolute_acceptance_floor=True,no_candidate_quality_labels_for_selection=True))
    banks,_,pins=features(destination,'fit',torch.device('cpu'))
    predictions=[]
    for stream,bank in banks.items():
        with torch.no_grad():
            logits=bank['x']@state['weight'].T+state['bias']
            values=(logits if is_delta else nested_probabilities(logits)).tolist()
        for row,value,op in zip(bank['rows'],values,bank['operational_utilities'],strict=True):
            trace=bank['traces'][row['ordinal']]
            value=[0.,0.] if is_delta and stream=='keep' else value
            predictions.append(dict(ordinal=row['ordinal'],stream=stream,sun_gain=value[0],ms_gain=value[1],
                operational_utility=op,valid=stream=='keep' or
                    (trace.get('proposal_generated',False) and trace['continuous_applied'])))
    write_rows(destination/'training/nested_sun_ranker/FIT_PREDICTIONS.jsonl',predictions)
    if not is_delta:policy_worker(root,name)
    policy_worker(root,name,utility=True)
    write_json(destination/'retained_execution/DONE.json',dict(complete=True,
        model_sha256=file_hash(model_path),feature_pins=pins,retrained=False))


def retained_fresh_worker(root,name,*,evaluate_only=False):
    """Apply the frozen gain utility to the previously viewed 256 sources."""
    import subprocess
    from evaluate_sun_ranker import build,evaluate,summary
    destination=root/'variants'/name
    original=Path(json.loads((root/'REGISTRATION.json').read_text())['previous'])
    panel=destination/'policies/fresh/zero_gain_keep'
    if not evaluate_only:
        fresh=destination/'fresh'
        if not fresh.exists():fresh.symlink_to(original/'fresh',target_is_directory=True)
        model_path=original/'training/sun_ranker/SUN_RANKER.pt'
        source=original/'fresh/ranker_inference/FRESH_PREDICTIONS.jsonl'
        write_json(destination/'FRESH_GAIN_POLICY_REGISTRATION.json',dict(model_sha256=file_hash(model_path),
            prediction_source_sha256=file_hash(source),score_weights=[2.,1.],KEEP_reference=[0.,0.],
            no_new_training=True,role='exploratory_replication_on_previously_viewed_sources',
            no_candidate_outcomes_used_for_choice=True))
        rows=read_rows(source)
        rows += [dict(ordinal=i,stream='keep',sun_gain=0.,ms_gain=0.,operational_utility=0.,
            valid=record['success'],feature_forward_calls=0)
            for i,record in enumerate(read_rows(fresh/'native/inputs.jsonl'))]
        prediction=destination/'retained_fresh_execution/FRESH_PREDICTIONS.jsonl'
        write_rows(prediction,rows)
        panel=build(destination,'fresh','zero_gain_keep',0.,0.,prediction_path=prediction,
            score_kind='signed_NS_NMS_gains',learned_keep=True,score_weights=(2.,1.))
        write_json(destination/'retained_fresh_execution/DONE.json',dict(complete=True,panel=str(panel),
            predictions_sha256=file_hash(prediction),candidate_physics_not_consulted=True))
        return
    import torch
    reuse=[original/'fresh/native/labeling/result']
    reuse += [p/'edited/labeling/result' for p in (original/'policies/fresh').glob('*')
              if (p/'edited/labeling/result/LABEL_FINAL.json').exists()]
    output=panel/'edited/labeling/result'
    command=[sys.executable,str(SOURCE/'scripts/label_rsi_cached_endpoints.py'),
        '--input-jsonl',str(panel/'edited/inputs.jsonl'),'--output-dir',str(output),
        '--purpose','training_feedback','--gpu-count',str(torch.cuda.device_count()),'--workers-per-gpu','4',
        '--record-timeout','600','--deterministic','--feedback-manifest',str(panel/'edited/FEEDBACK_MANIFEST.json'),
        '--reuse-endpoints',*[str(p) for p in reuse],'--joint-physical-stop','--max-steps','1000']
    subprocess.run(command,check=True)
    evaluate(destination,panel,cached=False,nu_workers=3)
    report=dict(complete=True,result=summary(destination,panel,'fresh'),
        role='exploratory_replication_on_previously_viewed_sources',selection='frozen_before_new_endpoint_labels')
    write_json(destination/'retained_fresh_evaluate_execution/DONE.json',report)
    print(json.dumps(report),flush=True)


def score_bank(root,name,stream,*,nu_workers=7):
    import subprocess
    from scripts.run_rsi_stages import validity
    panel=root/'variants'/name/'fit/bank'/stream
    spec=json.loads((panel/'RUN_SPEC.json').read_text())
    output=panel/'candidate/scoring/result'
    if (output/'_SUCCESS').exists():
        receipt=json.loads((output/'FEEDBACK_FINAL.json').read_text())
        if receipt['input_sha256']!=file_hash(panel/'candidate/inputs.jsonl'):
            raise ValueError('scored input changed')
        return
    command=[sys.executable,str(SOURCE/'scripts/evaluate_programmed_paths.py'),
        '--paths-jsonl',str(panel/'candidate/inputs.jsonl'),
        '--labels-jsonl',str(panel/'candidate/labeling/result/labels.jsonl'),
        '--frozen-config',spec['assets']['frozen_config'],'--official-cache',spec['assets']['official_cache'],
        '--output-dir',str(output),'--expected-requests',str(spec['requests']),
        '--endpoint','native','--cohort-role','training_feedback','--policy-stage','round0_diagnostic',
        '--sun-only','--nu-workers',str(nu_workers),'--nu-cache',str(root/'nu_cache'),
        '--feedback-manifest',str(panel/'candidate/FEEDBACK_MANIFEST.json'),'--joint-physical-stop']
    subprocess.run(command,check=True)
    validity(spec,'candidate')


def policy_worker(root,name,*,utility=False):
    from evaluate_sun_ranker import build,evaluate,summary
    destination=root/'variants'/name
    predictions=destination/'training/nested_sun_ranker/FIT_PREDICTIONS.jsonl'
    weights=(2.,1.) if utility else None
    tag='learned_keep_utility' if utility else 'learned_keep'
    if utility:
        write_json(destination/'UTILITY_POLICY_REGISTRATION.json',dict(score_weights=weights,
            rule='compare expected NS and NMS utility over learned KEEP and edits',
            candidate_MS_gain_veto=False,absolute_acceptance_threshold=None,reason='reduce weak edits while allowing SUN gains'))
    panel=build(destination,'fit',tag,0.,0.,prediction_path=predictions,
                score_kind='signed_NS_NMS_gains' if name=='retained_delta_e3' else 'absolute_NS_NMS_probabilities',
                learned_keep=True,score_weights=weights)
    evaluate(destination,panel,nu_workers=3)
    results={role:summary(destination,panel,role) for role in ('train','dev','final','all')}
    report=dict(complete=True,threshold=None,learned_keep=True,score_weights=weights,results=results,DEV_role='feasibility',old_FINAL_role='exploratory')
    write_json(destination/('utility_policy_execution' if utility else 'policy_execution')/'DONE.json',report)
    print(json.dumps(report),flush=True)


def collect_keep_features(destination,panel_name='fit'):
    import gc
    import torch
    from crystal_dlm.expert_edit import load_editor_model
    from scripts.train_keep_edit_utility import feature_rows
    panel=destination/panel_name
    spec=json.loads((panel/'RUN_SPEC.json').read_text())
    plans=read_rows(panel/'cohort/plans.jsonl')
    current=read_rows(panel/'current/inputs.jsonl')
    bank=panel/'bank/keep'
    model,tokenizer=load_editor_model(spec['assets']['base_model'],spec['assets']['editor_checkpoint'],torch.device('cuda',0))
    rows=[dict(ordinal=i,pair_id=fingerprint(dict(panel=panel_name,stream='keep',source=p['ancestor_id'])),
        source_id=p['ancestor_id'],prompt=p['body_prompt'],num_sites=p['plan_state']['N'],
        current_tokens=c['body_token_ids'],proposal_tokens=c['body_token_ids'],action_positions=[])
        for i,(p,c) in enumerate(zip(plans,current)) if c.get('body_token_ids') and c.get('success')]
    bank.mkdir(parents=True,exist_ok=True)
    write_rows(bank/'FEATURE_ROWS.jsonl',rows)
    feature_rows(model,tokenizer,rows,torch.device('cuda',0),bank,'CANDIDATE')
    write_json(bank/'COLLECTION_FINAL.json',dict(features_sha256=file_hash(bank/'CANDIDATE_FEATURES.pt'),
        feature_rows_sha256=file_hash(bank/'FEATURE_ROWS.jsonl'),rows=len(rows),kind='unchanged_current_as_model_candidate'))
    del model,tokenizer;gc.collect();torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare','scope','train','probe','collect','collect_resume','physics','score','rank','policy','utility_policy','retained','retained_fresh','retained_fresh_evaluate','score_worker','rank_worker','policy_worker','utility_policy_worker','retained_worker','retained_fresh_worker','retained_fresh_evaluate_worker'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--name', choices=['mini_2e6', 'mini_5e7','scope_2e6','frozen_e3','balanced_e3','retained_e3','retained_delta_e3'])
    parser.add_argument('--stream',choices=['primary','rank1','rank2','rank3'])
    parser.add_argument('--gpus',type=int,default=1)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.root, args.previous)
    elif args.mode == 'scope':
        prepare_scope(args.root)
    elif args.mode == 'train':
        train(args.root, args.name)
    elif args.mode == 'probe':
        probe(args.root)
    elif args.mode=='score_worker':
        score_bank(args.root,args.name,args.stream)
    elif args.mode=='utility_policy_worker':
        policy_worker(args.root,args.name,utility=True)
    elif args.mode=='retained_worker':
        retained_worker(args.root,args.name)
    elif args.mode in ('retained_fresh_worker','retained_fresh_evaluate_worker'):
        retained_fresh_worker(args.root,args.name,evaluate_only=args.mode=='retained_fresh_evaluate_worker')
    elif args.mode.endswith('_worker'):
        (rank_worker if args.mode=='rank_worker' else policy_worker)(args.root,args.name)
    else:
        run_variant(args.root,args.name,args.mode,args.stream,args.gpus)
