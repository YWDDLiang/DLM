"""Train separate SUN/MS gain readouts on actual continuous TRAIN candidates."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import itertools
import json
import os
from pathlib import Path
import sys
import time

SOURCE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(SOURCE/'src'),str(SOURCE/'operations/r03_c3fd_main_20260907')]
from scripts.run_post_refine_cycle import read_rows,write_rows,write_json,file_hash
from scripts.run_rsi_stages import scores,score_directory
from scripts.run_sun_rank_scope import STREAMS
from crystal_dlm.sun_ranker import endpoint_targets,extra_features,EXTRA_FEATURES,nested_probabilities
from crystal_dlm.post_refine_contract import fingerprint


def features(root,panel_name,device):
    import torch
    from crystal_dlm.expert_edit import FloatMLP
    reg=json.loads((root/'PREREGISTRATION.json').read_text());previous=Path(reg['previous_run'])
    editor=Path(reg['old_editor'])
    if file_hash(editor/'RSI_TRAINING_DONE.json')!=reg['old_editor_receipt_sha256']:
        raise ValueError('frozen content checkpoint receipt changed')
    state=torch.load(editor/'expert_edit_modules.pt',map_location=device,weights_only=True)['quality_head']
    head=FloatMLP(state['layers.0.weight'].shape[1],state['layers.0.weight'].shape[0],4).to(device)
    head.load_state_dict(state);head.eval()
    panel=root/panel_name;current=read_rows(panel/'native/inputs.jsonl')
    before=scores(previous/'fit','native') if panel_name=='fit' else scores(panel,'native')
    if reg.get('editor_content_changed'):
        op_path=editor/'expert_edit_modules.pt'
        op_state=state
    else:
        operational=json.loads((previous/'operational/MODEL_DEFINITION.json').read_text())
        op_path=Path(operational['quality_head'])
        if file_hash(op_path)!=operational['quality_head_sha256']:raise ValueError('fixed operational comparator changed')
        op_state=torch.load(op_path,map_location=device,weights_only=True)
    if any(not torch.equal(op_state[k],state[k]) for k in ('layers.0.weight','layers.0.bias')):
        raise ValueError('operational comparator has a different hidden representation')
    source_current=previous/'fit/native' if panel_name=='fit' else panel/'native'
    bound_current=[source_current/'inputs.jsonl',source_current/'labeling/result/LABEL_FINAL.json',
        score_directory(source_current.parent,source_current.name)/'attempt_results.jsonl',editor/'expert_edit_modules.pt']
    result={};pins={str(p):file_hash(p) for p in [op_path,*bound_current]}
    learned_keep=(root/'LEARNED_KEEP_REGISTRATION.json').exists()
    if learned_keep:pins[str(root/'LEARNED_KEEP_REGISTRATION.json')]=file_hash(root/'LEARNED_KEEP_REGISTRATION.json')
    for name in [*STREAMS,*(['keep'] if learned_keep else [])]:
        bank=panel/'bank'/name;receipt=json.loads((bank/'COLLECTION_FINAL.json').read_text())
        fp=bank/'CANDIDATE_FEATURES.pt';rp=bank/'FEATURE_ROWS.jsonl'
        if file_hash(fp)!=receipt['features_sha256'] or file_hash(rp)!=receipt['feature_rows_sha256']:
            raise ValueError('ranker feature cache changed')
        payload=torch.load(fp,map_location='cpu',weights_only=False);rows=read_rows(rp)
        if payload['pair_ids']!=[r['pair_id'] for r in rows]:raise ValueError('ranker feature identities differ')
        candidate=current if name=='keep' else read_rows(bank/'candidate/inputs.jsonl');traces={};extra=[]
        for row in rows:
            i=row['ordinal']
            if name=='keep':
                trace=dict(action=dict(positions=[]),proposal_generated=False,continuous_applied=False)
                traces[i]=trace;extra.append(extra_features(before[i],current[i],current[i],trace,row['num_sites']))
                continue
            wrapper=json.loads((bank/f'candidate/records/{i:04d}.json').read_text())
            bound=json.loads((bank/f'bound/{i:04d}.json').read_text())
            if receipt['bindings'][i]['record_sha256']!=fingerprint(candidate[i]) or bound['record']!=candidate[i]:
                raise ValueError('ranker continuous input binding changed')
            trace=wrapper['editor_trace'];traces[i]=dict(trace,continuous_applied=bound['commit_trace']['applied'])
            extra.append(extra_features(before[i],current[i],candidate[i],trace,row['num_sites']))
        with torch.no_grad():
            hidden=torch.cat([head.layers[:2](payload['features'][i:i+16].to(device)) for i in range(0,len(rows),16)])
        x=torch.cat((hidden,torch.tensor(extra,dtype=torch.float32,device=device)),dim=1)
        with torch.no_grad():op_values=(hidden@op_state['layers.2.weight'][3]+op_state['layers.2.bias'][3]).cpu().tolist()
        result[name]=dict(x=x,rows=rows,traces=traces,operational_utilities=op_values)
        pins[str(fp)]=file_hash(fp);pins[str(rp)]=file_hash(rp)
    return result,before,pins


def train(root,output,*,nested=False):
    import torch
    reg=json.loads((root/'PREREGISTRATION.json').read_text());cfg=reg['head_training'];torch.manual_seed(cfg['seed'])
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('ranker training needs an allocation')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    device=torch.device(cfg.get('device','cuda'))
    if device.type=='cuda':torch.cuda.set_device(0)
    started=time.monotonic()
    if nested:
        supplement=json.loads((root/'ABSOLUTE_STATE_REGISTRATION.json').read_text())
        if supplement['base_training_sha256']!=file_hash(root/'training/sun_ranker/TRAINING_FINAL.json'):
            raise ValueError('absolute-state comparison changed base training')
    split=read_rows(root/'SOURCE_SPLIT.jsonl')
    if file_hash(root/'SOURCE_SPLIT.jsonl')!=reg['source_split_sha256']:raise ValueError('TRAIN roles changed')
    output.mkdir(parents=True,exist_ok=True)
    banks,before,pins=features(root,'fit',device)
    data=[];xs=[];exclusions=Counter();content=[]
    for stream,bank in banks.items():
        path=root/'fit/bank'/stream
        old_primary=stream=='primary' and not reg.get('editor_content_changed')
        observed=before if stream=='keep' else scores(Path(reg['previous_run'])/'fit','hybrid_proposal') if old_primary else scores(path,'candidate')
        measured=Path(reg['previous_run'])/'fit/native' if stream=='keep' else Path(reg['previous_run'])/'fit/hybrid_proposal' if old_primary else path/'candidate'
        for pin in (measured/'inputs.jsonl',measured/'labeling/result/LABEL_FINAL.json',
                    score_directory(measured.parent,measured.name)/'attempt_results.jsonl'):pins[str(pin)]=file_hash(pin)
        observed_inputs=read_rows(measured/'inputs.jsonl')
        for offset,row in enumerate(bank['rows']):
            i=row['ordinal'];role=split[i]
            if role['split']!='train':continue
            if role['ancestor_id']!=row['source_id']:raise ValueError('TRAIN source misalignment')
            if stream!='keep' and not bank['traces'][i]['continuous_applied']:
                exclusions['exact_KEEP_not_a_physical_edit']+=1;continue
            a,ak=endpoint_targets(before[i]);b,bk=endpoint_targets(observed[i])
            if a is None or b is None:
                exclusions['excluded_'+(ak if a is None else bk)]+=1;continue
            delta=[b[j]-a[j] for j in range(2)]
            trace=bank['traces'][i]
            data.append(dict(pair_id=row['pair_id'],source_id=row['source_id'],ordinal=i,stream=stream,
                target_sun_gain=delta[0],target_ms_gain=delta[1],before_targets=a,after_targets=b,
                before_status=ak,after_status=bk,focus_split='train',cache_offset=offset,
                before_score_sha256=fingerprint(before[i]),after_score_sha256=fingerprint(observed[i])))
            xs.append(bank['x'][offset])
            if delta[0]>0:
                content.append(dict(row,stream=stream,focus_split='train',SUN_gain=delta[0],MS_gain=delta[1],
                    actual_Stable_promotion=not bool(before[i].get('strict_stable')) and bool(observed[i].get('strict_stable')),
                    changed_sites=trace.get('action',{}).get('sites',[]),candidate_record_sha256=fingerprint(observed_inputs[i]),
                    physical_comparison=data[-1]))
    if not data:raise ValueError('no reliable TRAIN candidate comparisons')
    x=torch.stack(xs);target=torch.tensor([r['after_targets'] if nested else [r['target_sun_gain'],r['target_ms_gain']] for r in data],device=device)
    multiplicity=Counter(r['source_id'] for r in data)
    weights=torch.tensor([1/len(multiplicity)/multiplicity[r['source_id']] for r in data],device=device)
    mean=(x*weights[:,None]).sum(0);scale=((x-mean).square()*weights[:,None]).sum(0).sqrt().clamp_min(.05)
    z=(x-mean)/scale
    linear=torch.nn.Linear(x.shape[1],2,dtype=torch.float32,device=device)
    with torch.no_grad():linear.weight.zero_();linear.bias.zero_()
    optimizer=torch.optim.Adam(linear.parameters(),lr=cfg['learning_rate'])
    groups=defaultdict(list)
    for i,row in enumerate(data):groups[row['source_id']].append(i)
    edges=[];edge_classes=[]
    for source,indices in groups.items():
        comparisons=[]
        for a,b in itertools.combinations(indices if (root/'LEARNED_KEEP_REGISTRATION.json').exists() else [-1]+indices,2):
            if cfg.get('balanced_keep_pairs') and (data[a]['stream']=='keep')==(data[b]['stream']=='keep'):
                continue
            sa=0 if a==-1 else data[a]['target_sun_gain'];sb=0 if b==-1 else data[b]['target_sun_gain']
            if sa==sb:continue
            comparisons.append((a,b) if sa>sb else (b,a))
        for a,b in comparisons:
            edges.append((a,b,1/max(1,len(comparisons))))
            edge_classes.append('keep' if a<0 or data[a]['stream']=='keep' else 'edit')
    edge_a=torch.tensor([a+1 for a,b,w in edges],device=device,dtype=torch.long)
    edge_b=torch.tensor([b+1 for a,b,w in edges],device=device,dtype=torch.long)
    edge_w=torch.tensor([w for a,b,w in edges],device=device)
    if cfg.get('balanced_keep_pairs') and edges:
        totals={kind:sum(w for (_,_,w),k in zip(edges,edge_classes) if k==kind) for kind in set(edge_classes)}
        edge_w=torch.tensor([w/totals[k]/len(totals) for (_,_,w),k in zip(edges,edge_classes)],device=device)
    else:edge_w/=edge_w.sum().clamp_min(1)
    keep_values=torch.tensor([data[a if a>=0 else b]['before_targets'][0] for a,b,w in edges],device=device)
    generator=torch.Generator(device='cpu').manual_seed(cfg['seed']);visits=torch.zeros(len(data),dtype=torch.int64)
    steps=0;history=[]
    for epoch in range(1,cfg['epochs']+1):
        order=torch.randperm(len(data),generator=generator)
        for start in range(0,len(data),cfg['batch_size']):
            cpu=order[start:start+cfg['batch_size']];idx=cpu.to(device);optimizer.zero_grad(set_to_none=True)
            prediction=linear(z[idx])
            errors=torch.nn.functional.binary_cross_entropy(nested_probabilities(prediction),target[idx],reduction='none') if nested else (prediction-target[idx]).square()
            mse=(errors.mean(1)*weights[idx]).sum()*len(data)/len(idx)
            rank_loss=mse.new_zeros(())
            if edges:
                all_logits=linear(z);all_sun=torch.cat((mse.new_zeros(1),nested_probabilities(all_logits)[:,0] if nested else all_logits[:,0]))
                a_score=all_sun[edge_a];b_score=all_sun[edge_b]
                if nested:
                    a_score=torch.where(edge_a==0,keep_values,a_score);b_score=torch.where(edge_b==0,keep_values,b_score)
                rank_loss=(torch.nn.functional.softplus(-(a_score-b_score))*edge_w).sum()
            loss=mse+cfg['ridge']*linear.weight.square().sum()+cfg['SUN_pairwise_coefficient']*rank_loss
            loss.backward();optimizer.step();visits[cpu]+=1;steps+=1
        with torch.no_grad():
            error=torch.nn.functional.binary_cross_entropy(nested_probabilities(linear(z)),target,reduction='none') if nested else (linear(z)-target).square()
            mse=float((error.mean(1)*weights).sum())
        event=dict(epoch=epoch,optimizer_steps=steps,source_weighted_objective=mse,
            objective='absolute_state_BCE' if nested else 'delta_MSE',seconds=time.monotonic()-started)
        history.append(event);write_json(output/'PROGRESS.json',event)
        if epoch%8==0:print(json.dumps(event),flush=True)
    if not bool((visits==cfg['epochs']).all()):raise ValueError('incomplete actual supervised coverage')
    folded_weight=linear.weight.detach()/scale
    folded_bias=linear.bias.detach()-(folded_weight*mean).sum(1)
    with torch.no_grad():drift=float(((x@folded_weight.T+folded_bias)-linear(z)).abs().max())
    if drift>1e-5:raise ValueError('folded ranker parity failed')
    model=output/'SUN_RANKER.pt'
    torch.save(dict(weight=folded_weight.cpu(),bias=folded_bias.cpu(),mean=mean.cpu(),scale=scale.cpu(),
        extra_features=list(EXTRA_FEATURES),output_names=['P_NS','P_NMS'] if nested else ['sun_gain','ms_gain'],
        head_kind='nested_absolute_states' if nested else 'linear_gains',frozen_editor=reg['old_editor']),model)
    write_rows(output/'TRAIN_ROWS.jsonl',data);write_rows(output/'SUN_CONTENT_POSITIVES.jsonl',content)
    if nested and file_hash(output/'TRAIN_ROWS.jsonl')!=supplement['base_train_rows_sha256']:
        raise ValueError('absolute-state model did not use the exact original TRAIN comparisons')
    torch.save(dict(optimizer=optimizer.state_dict(),row_visits=visits,sampler_rng=generator.get_state()),output/'OPTIMIZER_COVERAGE.pt')
    prediction_rows=[]
    for stream,bank in banks.items():
        with torch.no_grad():
            logits=bank['x']@folded_weight.T+folded_bias
            values=(nested_probabilities(logits) if nested else logits).cpu().tolist()
        for row,value,op_value in zip(bank['rows'],values,bank['operational_utilities'],strict=True):
            prediction_rows.append(dict(ordinal=row['ordinal'],stream=stream,sun_gain=value[0],ms_gain=value[1],
                operational_utility=op_value,
                valid=stream=='keep' or (bank['traces'][row['ordinal']].get('proposal_generated',False) and bank['traces'][row['ordinal']]['continuous_applied'])))
    write_rows(output/'FIT_PREDICTIONS.jsonl',prediction_rows)
    report=dict(schema='SUN_nested_absolute_state_ranker_v1' if nested else 'SUN_specific_linear_gain_ranker_v1',
        head_kind='nested_absolute_states' if nested else 'linear_gains',status='complete',settings=cfg,train_rows=len(data),
        train_sources=len(multiplicity),optimizer_steps=steps,complete_passes=cfg['epochs'],
        minimum_row_visits=int(visits.min()),maximum_row_visits=int(visits.max()),
        SUN_ranking_pairs=len(edges),SUN_ranking_sources=len({data[a if a>=0 else b]['source_id'] for a,b,w in edges}),
        SUN_pairwise_updates_per_edge=steps,MSE_row_visits_semantics='actual_minibatch_updates; ranking term visits all registered edges each step',
        source_target_counts=dict(Counter(str((r['target_sun_gain'],r['target_ms_gain'])) for r in data)),
        target_definition=('absolute_candidate_NS_and_NMS_nested_probabilities' if nested else 'delta_verified_Stable_and_novel; delta_verified_MS_and_novel')+'; excludes_cohort_U',
        feature_schema=dict(quality_hidden_width=x.shape[1]-len(EXTRA_FEATURES),observable_extra_features=list(EXTRA_FEATURES)),
        SUN_content_positives=len(content),SUN_positive_sources=len({r['source_id'] for r in content}),
        true_Stable_promotion_rows=sum(r['actual_Stable_promotion'] for r in content),
        exclusions=dict(exclusions),optimized_parameter_count=sum(p.numel() for p in linear.parameters()),
        frozen_content_scope_and_quality_hidden=True,train_data_sha256=file_hash(output/'TRAIN_ROWS.jsonl'),
        model_path=str(model),model_sha256=file_hash(model),predictions_sha256=file_hash(output/'FIT_PREDICTIONS.jsonl'),
        feature_and_feedback_pins=pins,history=history,fold_max_abs_error=drift,
        no_DEV_FINAL_supervision=True,feature_normalization_TRAIN_only=True,training_seconds=time.monotonic()-started)
    write_json(output/'TRAINING_FINAL.json',report);(output/'_SUCCESS').touch()
    print(json.dumps({k:v for k,v in report.items() if k not in ('history','feature_and_feedback_pins')}),flush=True)


def infer(root,output,*,nested=False):
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise ValueError('fresh ranker inference needs GPU allocation')
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.use_deterministic_algorithms(True)
    frozen=json.loads((root/('NESTED_FROZEN_SELECTION.json' if nested else 'FROZEN_SELECTION.json')).read_text());model=Path(frozen['model_path'])
    if file_hash(model)!=frozen['model_sha256']:raise ValueError('frozen ranker changed')
    device=torch.device('cuda',0);state=torch.load(model,map_location=device,weights_only=False)
    banks,_,pins=features(root,'fresh',device);rows=[]
    for stream,bank in banks.items():
        with torch.no_grad():
            logits=bank['x']@state['weight'].T+state['bias']
            values=(nested_probabilities(logits) if nested else logits).cpu().tolist()
        for row,value,op_value in zip(bank['rows'],values,bank['operational_utilities'],strict=True):
            rows.append(dict(ordinal=row['ordinal'],stream=stream,sun_gain=value[0],ms_gain=value[1],
                operational_utility=op_value,
                valid=bank['traces'][row['ordinal']].get('proposal_generated',False) and bank['traces'][row['ordinal']]['continuous_applied']))
    write_rows(output/'FRESH_PREDICTIONS.jsonl',rows)
    write_json(output/'INFERENCE_FINAL.json',dict(rows=len(rows),model_sha256=frozen['model_sha256'],
        predictions_sha256=file_hash(output/'FRESH_PREDICTIONS.jsonl'),feature_pins=pins,no_candidate_labels_loaded=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--completion-dir',type=Path,required=True);p.add_argument('--mode',choices=['train','infer'],default='train')
    p.add_argument('--nested',action='store_true')
    a=p.parse_args();(train if a.mode=='train' else infer)(a.root,a.completion_dir,nested=a.nested)
