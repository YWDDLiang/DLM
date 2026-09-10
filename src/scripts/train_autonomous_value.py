"""Learn edit gains from model and geometry inputs; physics supplies labels only."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

SOURCE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(SOURCE/'src'),str(SOURCE/'operations/r03_c3fd_main_20260907')]
from scripts.run_post_refine_cycle import read_rows,write_rows,write_json,file_hash
from scripts.run_sun_rank_scope import candidate_streams
from crystal_dlm.sun_ranker import extra_features,EXTRA_FEATURES,endpoint_targets


def model_inputs(root,panel_name):
    """This loader never opens any physical label, score or novelty file."""
    import torch
    reg=json.loads((root/'PREREGISTRATION.json').read_text())
    panel=root/panel_name
    current=read_rows(panel/'native/inputs.jsonl')
    keep=panel/'bank/keep'
    keep_rows=read_rows(keep/'FEATURE_ROWS.jsonl')
    keep_payload=torch.load(keep/'CANDIDATE_FEATURES.pt',map_location='cpu',weights_only=False)
    if keep_payload['pair_ids']!=[row['pair_id'] for row in keep_rows]:raise ValueError('KEEP feature order differs')
    keep_by_id={row['ordinal']:i for i,row in enumerate(keep_rows)}
    banks={};pins={str(keep/'CANDIDATE_FEATURES.pt'):file_hash(keep/'CANDIDATE_FEATURES.pt')}
    for stream in candidate_streams(reg):
        bank=panel/'bank'/stream
        rows=read_rows(bank/'FEATURE_ROWS.jsonl')
        payload=torch.load(bank/'CANDIDATE_FEATURES.pt',map_location='cpu',weights_only=False)
        if payload['pair_ids']!=[row['pair_id'] for row in rows]:raise ValueError('candidate feature order differs')
        candidates=read_rows(bank/'candidate/inputs.jsonl')
        geometry=[];valid=[];baseline=[];base_geometry=[]
        for row in rows:
            i=row['ordinal'];trace={'action':{'positions':row['action_positions']}}
            # The first twelve legacy extras are forbidden label/MLIP features.
            geometry.append(extra_features({},current[i],candidates[i],trace,row['num_sites'])[12:])
            base_geometry.append(extra_features({},current[i],current[i],{'action':{'positions':[]}},row['num_sites'])[12:])
            baseline.append(keep_payload['features'][keep_by_id[i]])
            bound=json.loads((bank/f'bound/{i:04d}.json').read_text())
            valid.append(bool(bound['commit_trace']['applied']))
        banks[stream]=dict(rows=rows,raw=payload['features'].float(),baseline=torch.stack(baseline).float(),
            geometry=torch.tensor(geometry,dtype=torch.float32),base_geometry=torch.tensor(base_geometry,dtype=torch.float32),valid=valid)
        pins[str(bank/'CANDIDATE_FEATURES.pt')]=file_hash(bank/'CANDIDATE_FEATURES.pt')
    return banks,pins


def network(editor,kind,device):
    import torch
    state=torch.load(Path(editor)/'expert_edit_modules.pt',map_location='cpu',weights_only=True)['quality_head']
    width,inputs=state['layers.0.weight'].shape
    class Value(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden=torch.nn.Linear(inputs,width)
            self.head=torch.nn.Linear(width+9,2)
            self.register_buffer('mean',torch.zeros(width+9))
            self.register_buffer('scale',torch.ones(width+9))
            with torch.no_grad():
                self.hidden.weight.copy_(state['layers.0.weight']);self.hidden.bias.copy_(state['layers.0.bias'])
                self.head.weight.zero_();self.head.bias.zero_()
            if kind=='linear':
                for p in self.hidden.parameters():p.requires_grad_(False)
        def features(self,raw,geometry):return torch.cat((torch.nn.functional.silu(self.hidden(raw)),geometry),dim=1)
        def forward(self,raw,geometry):return self.head((self.features(raw,geometry)-self.mean)/self.scale)
    return Value().to(device)


def train(root,kind):
    import torch
    from scripts.run_rsi_stages import scores
    torch.set_num_threads(1);torch.manual_seed(20260911);torch.cuda.set_device(0)
    device=torch.device('cuda',0);started=time.monotonic()
    reg=json.loads((root/'PREREGISTRATION.json').read_text())
    output=root/('autonomous_'+kind);output.mkdir(parents=True,exist_ok=True)
    contract=dict(inputs=['DLM forward features from Plan/current/proposal/action','geometry computed from current and proposal'],
        geometry_features=list(EXTRA_FEATURES[12:]),current_physics_inputs=False,candidate_physics_inputs=False,
        current_stability_novelty_or_hull_inputs=False,physical_feedback_use='offline supervised targets only',
        KEEP_rule='subtract the learned value of the current view; identity gain is exactly zero')
    write_json(output/'INPUT_CONTRACT.json',contract)
    banks,pins=model_inputs(root,'fit')
    before=scores(Path(reg['previous_run'])/'fit','native')
    roles=read_rows(root/'SOURCE_SPLIT.jsonl')
    raw=[];baseline=[];geometry=[];base_geometry=[];metadata=[];targets=[];levels=[]
    for stream,bank in banks.items():
        after=scores(root/'fit/bank'/stream,'candidate') if reg.get('editor_content_changed') or stream!='primary' else scores(Path(reg['previous_run'])/'fit','hybrid_proposal')
        for j,row in enumerate(bank['rows']):
            i=row['ordinal']
            if roles[i]['split']!='train' or not bank['valid'][j]:continue
            a,_=endpoint_targets(before[i]);b,_=endpoint_targets(after[i])
            if a is None or b is None:continue
            if roles[i]['ancestor_id']!=row['source_id']:raise ValueError('supervision source misalignment')
            raw.append(bank['raw'][j]);baseline.append(bank['baseline'][j]);geometry.append(bank['geometry'][j]);base_geometry.append(bank['base_geometry'][j])
            targets.append([b[k]-a[k] for k in range(2)]);levels.append([a,b])
            metadata.append(dict(source_id=row['source_id'],ordinal=i,stream=stream,pair_id=row['pair_id'],target_gain=targets[-1]))
    tensors=[torch.stack(values).to(device) for values in (raw,baseline,geometry,base_geometry)]
    x,x0,g,g0=tensors;target=torch.tensor(targets,device=device);level=torch.tensor(levels,device=device)
    grouped=defaultdict(list)
    for i,row in enumerate(metadata):grouped[row['source_id']].append(i)
    groups=list(grouped.values());weights=torch.tensor([1/len(groups)/len(grouped[r['source_id']]) for r in metadata],device=device)
    model=network(reg['old_editor'],kind,device)
    with torch.no_grad():
        z=torch.cat([model.features(x[i:i+256],g[i:i+256]) for i in range(0,len(x),256)])
        model.mean.copy_((z*weights[:,None]).sum(0));model.scale.copy_(((z-model.mean).square()*weights[:,None]).sum(0).sqrt().clamp_min(.05))
    initial={n:p.detach().clone() for n,p in model.named_parameters()}
    parameter_groups=[dict(params=list(model.head.parameters()),lr=1e-3)]
    if kind=='mlp':parameter_groups.append(dict(params=list(model.hidden.parameters()),lr=1e-5))
    optimizer=torch.optim.AdamW(parameter_groups,weight_decay=0.)
    rng=torch.Generator().manual_seed(20260911);visits=torch.zeros(len(metadata),dtype=torch.int64);steps=0;history=[]
    for epoch in range(1,65):
        order=torch.randperm(len(groups),generator=rng).tolist()
        for start in range(0,len(order),32):
            selected=[groups[j] for j in order[start:start+32]];flat=[i for group in selected for i in group]
            idx=torch.tensor(flat,device=device);optimizer.zero_grad(set_to_none=True)
            after_value=model(x[idx],g[idx]);before_value=model(x0[idx],g0[idx]);gain=after_value-before_value
            losses=[];offset=0
            for group in selected:
                count=len(group);sl=slice(offset,offset+count);truth=target[idx[sl]];prediction=gain[sl]
                loss=(prediction-truth).square().mean()+.1*((before_value[sl]-level[idx[sl],0]).square().mean()+(after_value[sl]-level[idx[sl],1]).square().mean())
                values=torch.cat((prediction.new_zeros(1),prediction[:,0]));gold=torch.cat((truth.new_zeros(1),truth[:,0]))
                better=gold[:,None]>gold[None,:]
                if better.any():loss=loss+.2*torch.nn.functional.softplus(-(values[:,None]-values[None,:])[better]).mean()
                losses.append(loss);offset+=count
            loss=torch.stack(losses).mean()+.1*model.head.weight.square().sum()
            loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            optimizer.step();visits[flat]+=1;steps+=1
        event=dict(epoch=epoch,optimizer_steps=steps,last_loss=float(loss.detach()),seconds=time.monotonic()-started)
        history.append(event);write_json(output/'PROGRESS.json',event)
        if epoch%8==0:print(json.dumps(event),flush=True)
    drift={n:float((p.detach()-initial[n]).double().square().sum()) for n,p in model.named_parameters()}
    torch.save(dict(state_dict=model.cpu().state_dict(),kind=kind,editor=reg['old_editor'],contract=contract),output/'MODEL.pt')
    write_rows(output/'TRAIN_ROWS.jsonl',metadata)
    write_json(output/'TRAINING_FINAL.json',dict(complete=True,kind=kind,rows=len(metadata),sources=len(groups),
        optimizer_steps=steps,epochs=64,minimum_visits=int(visits.min()),maximum_visits=int(visits.max()),
        train_only=True,parameter_delta_squared=drift,model_sha256=file_hash(output/'MODEL.pt'),
        train_rows_sha256=file_hash(output/'TRAIN_ROWS.jsonl'),input_contract=contract,feature_pins=pins,history=history))


def infer(root,kind,panel_name='fit'):
    import torch
    torch.set_num_threads(1)
    output=root/('autonomous_'+kind)
    saved=torch.load(output/'MODEL.pt',map_location='cpu',weights_only=False)
    model=network(saved['editor'],kind,torch.device('cpu'));model.load_state_dict(saved['state_dict']);model.eval()
    banks,pins=model_inputs(root,panel_name);rows=[]
    for stream,bank in banks.items():
        values=[]
        with torch.no_grad():
            for i in range(0,len(bank['rows']),128):
                sl=slice(i,i+128)
                values.extend((model(bank['raw'][sl],bank['geometry'][sl])-model(bank['baseline'][sl],bank['base_geometry'][sl])).tolist())
        for row,value,valid in zip(bank['rows'],values,bank['valid'],strict=True):
            rows.append(dict(ordinal=row['ordinal'],stream=stream,sun_gain=value[0],ms_gain=value[1],operational_utility=0.,valid=valid))
    for i,current in enumerate(read_rows(root/panel_name/'native/inputs.jsonl')):
        rows.append(dict(ordinal=i,stream='keep',sun_gain=0.,ms_gain=0.,operational_utility=0.,valid=current['success'],feature_forward_calls=1))
    path=output/(panel_name.upper()+'_PREDICTIONS.jsonl');write_rows(path,rows)
    write_json(output/(panel_name.upper()+'_INFERENCE.json'),dict(complete=True,model_sha256=file_hash(output/'MODEL.pt'),
        predictions_sha256=file_hash(path),input_contract=saved['contract'],feature_pins=pins,no_physical_labels_loaded=True))
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['train','infer'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--kind',choices=['linear','mlp'],required=True)
    parser.add_argument('--panel',choices=['fit','fresh'],default='fit');args=parser.parse_args()
    if args.mode=='train':train(args.root,args.kind)
    else:infer(args.root,args.kind,args.panel)
