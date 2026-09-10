"""Train a regularized utility readout on historical and matched TRAIN pairs."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time

SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'src'))
sys.path.insert(0,str(SOURCE/'operations/r03_c3fd_main_20260907'))
from scripts.run_post_refine_cycle import read_rows,write_rows,write_json,file_hash,validate_rsi_checkpoint
from scripts.run_rsi_stages import scores,score_directory
from scripts.train_keep_edit_utility import tensor_hash
from compile_keep_edit_utility import utility
from crystal_dlm.post_refine_contract import fingerprint


def register(root):
    path=root/'READOUT_REGISTRATION.json'
    if path.exists() or (root/'FROZEN_SELECTION.json').exists():
        raise ValueError('readout needs a new registration and sealed FINAL')
    preceding={}
    for method in ('UTILITY','CONSENSUS'):
        prior=root/(method+'_DEV_SELECTION_FINAL.json');report=json.loads(prior.read_text())
        if report['admitted_to_FINAL'] or report['final_quality_consulted']:
            raise ValueError('readout requires both preceding DEV stages to fail with sealed FINAL')
        preceding[str(prior)]=file_hash(prior)
    write_json(path,dict(schema='matched_continuous_regularized_utility_readout_v1',
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),preceding_failed_DEV=preceding,
        plan_sha256=file_hash(SOURCE/'docs/r03_paper_story_20260907/KEEP_EDIT_MATCHED_READOUT_PLAN_20260910.md'),
        source_split_sha256=file_hash(root/'SOURCE_SPLIT.jsonl'),
        head_training=dict(epochs=64,snapshots=[8,16,32,64],thresholds=[0.,.05,.1,.2],
            learning_rate=1e-3,batch_size=256,seed=20260910,ridge_coefficient=.1,
            hidden_standard_deviation_floor=1e-4,historical_group_weight=.5,matched_group_weight=.5),
        optimized='quality_head_last_linear_acceptance_row_only',
        final_quality_consulted=False,matched_supervision_partition='train'))


def train(root):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from crystal_dlm.expert_edit import FloatMLP
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('readout training needs a GPU allocation')
    torch.cuda.set_device(0);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    reg=json.loads((root/'READOUT_REGISTRATION.json').read_text());settings=reg['head_training']
    torch.manual_seed(settings['seed']);device=torch.device('cuda',0);started=time.monotonic()
    if file_hash(root/'SOURCE_SPLIT.jsonl')!=reg['source_split_sha256']:
        raise ValueError('readout split changed')
    original=json.loads((root/'PREREGISTRATION.json').read_text());old=Path(original['old_editor'])
    validate_rsi_checkpoint(old,'E')
    previous=json.loads((root/'training/utility/result/TRAINING_FINAL.json').read_text())
    feature_receipt=json.loads((root/'evaluation_features/FEATURES_FINAL.json').read_text())
    history_path=root/'data/UTILITY_TRAIN.jsonl';train_features=root/'training/utility/result/TRAIN_FEATURES.pt'
    eval_features=root/'evaluation_features/EVAL_FEATURES.pt';eval_rows_path=root/'evaluation_features/EVAL_ROWS.jsonl'
    for path,expected in ((history_path,previous['data_sha256']),(train_features,previous['features_sha256']),
                           (eval_features,feature_receipt['features_sha256']),(eval_rows_path,feature_receipt['rows_sha256'])):
        if file_hash(path)!=expected:raise ValueError('readout frozen data/cache changed:'+str(path))
    historic=read_rows(history_path);eval_rows=read_rows(eval_rows_path)
    hf=torch.load(train_features,map_location='cpu',weights_only=False)
    ef=torch.load(eval_features,map_location='cpu',weights_only=False)
    if hf['pair_ids']!=[r['pair_id'] for r in historic] or ef['pair_ids']!=[r['pair_id'] for r in eval_rows]:
        raise ValueError('readout feature cache row identities differ')
    roles=read_rows(root/'SOURCE_SPLIT.jsonl');by_source={r['ancestor_id']:r for r in roles}
    if any(by_source[r['source_id']]['split']!='train' for r in historic):
        raise ValueError('historical readout supervision leaks held-out sources')
    before=scores(root/'fit','native');after=scores(root/'fit','hybrid_proposal')
    native_inputs_sha=file_hash(root/'fit/native/inputs.jsonl')
    proposal_inputs_sha=file_hash(root/'fit/hybrid_proposal/inputs.jsonl')
    selected=[];matched=[];excluded=Counter()
    for offset,row in enumerate(eval_rows):
        i=row['ordinal'];source=roles[i]
        if source['split']!='train':continue
        a,b=before[i],after[i]
        if a['sample_idx']!=b['sample_idx']:raise ValueError('matched readout source alignment differs')
        u,v=utility(a),utility(b)
        if u is None or v is None:
            excluded['matched_unknown_physics_or_novelty']+=1;continue
        selected.append(offset)
        matched.append(dict(pair_id=fingerprint(dict(domain='current_continuous',source=source['ancestor_id'],
            native_input=native_inputs_sha,proposal_input=proposal_inputs_sha,
            original_view_pair_id=row['pair_id'])),source_id=source['ancestor_id'],ordinal=i,
            utility_target=v-u,before_utility=u,after_utility=v,domain='current_continuous',
            cache_row=offset,before_score_sha256=fingerprint(a),after_score_sha256=fingerprint(b),focus_split='train'))
    if not matched:raise ValueError('no matched TRAIN supervision')
    rows=[dict(pair_id=r['pair_id'],source_id=r['source_id'],utility_target=r['utility_target'],
        before_utility=r['before_utility'],after_utility=r['after_utility'],domain='historical_token',
        cache_row=i,focus_split='train') for i,r in enumerate(historic)]+matched
    raw_weights=[]
    for group,group_weight in ((rows[:len(historic)],settings['historical_group_weight']),
                               (matched,settings['matched_group_weight'])):
        multiplicity=Counter(r['source_id'] for r in group)
        raw_weights.extend(group_weight/len(multiplicity)/multiplicity[r['source_id']] for r in group)
    for row,weight in zip(rows,raw_weights,strict=True):row['training_weight']=weight
    output=root/'training/readout_matched/result';output.mkdir(parents=True,exist_ok=True)
    data_path=root/'data/READOUT_TRAIN.jsonl';write_rows(data_path,rows)
    binding_paths=[history_path,train_features,eval_features,eval_rows_path,root/'SOURCE_SPLIT.jsonl']
    for stage in ('native','hybrid_proposal'):
        binding_paths.extend([root/f'fit/{stage}/inputs.jsonl',root/f'fit/{stage}/labeling/result/LABEL_FINAL.json',
            score_directory(root/'fit',stage)/'attempt_results.jsonl'])
    data_report=dict(rows=len(rows),historical_rows=len(historic),matched_rows=len(matched),
        sources=len({r['source_id'] for r in rows}),matched_sources=len({r['source_id'] for r in matched}),
        historical_target_counts=dict(Counter(r['utility_target'] for r in rows[:len(historic)])),
        matched_target_counts=dict(Counter(r['utility_target'] for r in matched)),
        data_sha256=file_hash(data_path),no_DEV_FINAL_supervision=True,exclusions=dict(excluded),
        representation_domains_kept_distinct=True,binding={str(p):file_hash(p) for p in binding_paths})
    write_json(root/'data/READOUT_DATA_FINAL.json',data_report)
    features=torch.cat((hf['features'],ef['features'][selected])).to(device)
    modules=torch.load(old/'expert_edit_modules.pt',map_location='cpu',weights_only=True)
    state=modules['quality_head'];head=FloatMLP(state['layers.0.weight'].shape[1],state['layers.0.weight'].shape[0],4).to(device)
    head.load_state_dict(state);head.eval()
    for parameter in head.parameters():parameter.requires_grad_(False)
    initial={k:v.detach().clone() for k,v in head.state_dict().items()}
    with torch.no_grad():
        hidden=torch.cat([head.layers[:2](features[i:i+256]) for i in range(0,len(rows),256)])
    weights=torch.tensor(raw_weights,device=device,dtype=torch.float32);weights/=weights.sum()
    targets=torch.tensor([r['utility_target'] for r in rows],device=device,dtype=torch.float32)
    mean=(hidden*weights[:,None]).sum(0)
    scale=((hidden-mean).square()*weights[:,None]).sum(0).sqrt().clamp_min(settings['hidden_standard_deviation_floor'])
    standardized=(hidden-mean)/scale
    linear=torch.nn.Linear(hidden.shape[1],1,dtype=torch.float32,device=device)
    with torch.no_grad():
        linear.weight.copy_(initial['layers.2.weight'][3:4]*scale)
        linear.bias.copy_(initial['layers.2.bias'][3:4]+(initial['layers.2.weight'][3]*mean).sum())
        equivalent=(linear(standardized)[:,0]-(hidden@initial['layers.2.weight'][3]+initial['layers.2.bias'][3])).abs().max().item()
    if equivalent>1e-5:raise ValueError('standardized readout initialization is not equivalent to old E3')
    optimizer=torch.optim.Adam(linear.parameters(),lr=settings['learning_rate'])
    generator=torch.Generator(device='cpu').manual_seed(settings['seed']);steps=0;snapshots={};history=[]
    exposure=torch.zeros(len(rows),dtype=torch.int64);evaluation=ef['features'].to(device);predictions={}
    for epoch in range(1,settings['epochs']+1):
        order=torch.randperm(len(rows),generator=generator)
        for start in range(0,len(rows),settings['batch_size']):
            cpu_indices=order[start:start+settings['batch_size']];indices=cpu_indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            error=linear(standardized[indices])[:,0]-targets[indices]
            data_loss=(error.square()*weights[indices]).sum()*len(rows)/len(indices)
            loss=data_loss+settings['ridge_coefficient']*linear.weight.square().sum()
            loss.backward();optimizer.step();steps+=1;exposure[cpu_indices]+=1
        with torch.no_grad():
            prediction=linear(standardized)[:,0]
            mse=float((weights*(prediction-targets).square()).sum())
        event=dict(epoch=epoch,optimizer_steps=steps,weighted_MSE=mse,
            ridge_penalty=float(settings['ridge_coefficient']*linear.weight.detach().square().sum()),
            seconds=time.monotonic()-started)
        history.append(event);write_json(output/'PROGRESS.json',event)
        if epoch not in settings['snapshots']:continue
        folded={k:v.detach().clone() for k,v in initial.items()}
        folded['layers.2.weight'][3]=linear.weight[0].detach()/scale
        folded['layers.2.bias'][3]=linear.bias[0].detach()-(linear.weight[0].detach()*mean/scale).sum()
        if any(not torch.equal(v,folded[k]) for k,v in initial.items() if k not in ('layers.2.weight','layers.2.bias')):
            raise ValueError('readout altered the frozen hidden head')
        if not torch.equal(folded['layers.2.weight'][:3],initial['layers.2.weight'][:3]) or not torch.equal(folded['layers.2.bias'][:3],initial['layers.2.bias'][:3]):
            raise ValueError('readout altered another quality output')
        head.load_state_dict(folded)
        with torch.no_grad():
            check=torch.cat([head(features[i:i+256])[:,3] for i in range(0,len(rows),256)])
            error=float((check-prediction).abs().max())
            predictions[str(epoch)]=torch.cat([head(evaluation[i:i+16])[:,3].cpu() for i in range(0,len(eval_rows),16)]).tolist()
        if error>1e-5:raise ValueError('folded readout differs from its standardized training predictor')
        path=output/f'quality_head_epoch{epoch}.pt';torch.save({k:v.cpu() for k,v in folded.items()},path)
        snapshots[str(epoch)]=dict(path=str(path),sha256=file_hash(path),optimizer_steps=steps,complete_passes=epoch,
            parameter_delta_squared=sum(float((v-initial[k]).double().square().sum()) for k,v in folded.items()),
            standardization_fold_max_abs_error=error)
        print(json.dumps(event),flush=True)
    if not bool((exposure==settings['epochs']).all()):raise ValueError('readout did not complete every data pass')
    torch.save(dict(optimizer=optimizer.state_dict(),mean=mean.cpu(),scale=scale.cpu(),exposures=exposure,
        sampler_state=generator.get_state()),output/'OPTIMIZER_NORMALIZATION_COVERAGE.pt')
    report=dict(schema='matched_continuous_frozen_quality_readout_v1',status='complete',train_rows=len(rows),
        train_sources=data_report['sources'],optimizer_steps=steps,complete_passes=settings['epochs'],settings=settings,
        data_sha256=file_hash(data_path),data_manifest_sha256=file_hash(root/'data/READOUT_DATA_FINAL.json'),
        snapshots=snapshots,history=history,minimum_row_visits=int(exposure.min()),maximum_row_visits=int(exposure.max()),
        optimized_parameters='quality_head.layers.2.weight[3] and quality_head.layers.2.bias[3] only',
        optimized_parameter_count=hidden.shape[1]+1,all_content_scope_and_quality_hidden_parameters_frozen=True,
        other_quality_outputs_unchanged=True,initial_old_head_max_abs_error=equivalent,
        original_checkpoint=str(old),training_seconds=time.monotonic()-started,no_DEV_FINAL_supervision=True)
    write_json(output/'TRAINING_FINAL.json',report);(output/'_SUCCESS').touch()
    original_predictions={r['ordinal']:r for r in read_rows(root/'evaluation_features/UTILITY_PREDICTIONS.jsonl')}
    values=[dict(original_predictions[row['ordinal']],learned_utilities={epoch:v[i] for epoch,v in predictions.items()})
        for i,row in enumerate(eval_rows)]
    evaluation_dir=root/'training/readout_matched/evaluation';write_rows(evaluation_dir/'UTILITY_PREDICTIONS.jsonl',values)
    write_json(evaluation_dir/'FEATURES_FINAL.json',dict(rows=len(eval_rows),rows_path=str(eval_rows_path),
        rows_sha256=file_hash(eval_rows_path),predictions_sha256=file_hash(evaluation_dir/'UTILITY_PREDICTIONS.jsonl'),
        features_sha256=file_hash(eval_features),features_path=str(eval_features),
        frozen_feature_receipt_sha256=file_hash(root/'evaluation_features/FEATURES_FINAL.json'),
        training_receipt_path=str(output/'TRAINING_FINAL.json'),training_receipt_sha256=file_hash(output/'TRAINING_FINAL.json'),
        heldout_features_unlabelled_for_training=True,no_DEV_FINAL_supervision=True))
    print(json.dumps({k:v for k,v in report.items() if k not in ('history','snapshots')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--register',action='store_true');args=parser.parse_args()
    (register if args.register else train)(args.root)
