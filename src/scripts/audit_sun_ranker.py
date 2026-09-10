"""Reload the full frozen editor and exported SUN ranker for canonical parity."""
import argparse
import json
import os
from pathlib import Path
import sys

SOURCE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(SOURCE/'src'),str(SOURCE/'operations/r03_c3fd_main_20260907')]
from scripts.run_post_refine_cycle import read_rows,write_json,file_hash,validate_rsi_checkpoint
from scripts.run_rsi_stages import scores
from scripts.run_sun_rank_scope import STREAMS
from crystal_dlm.sun_ranker import extra_features,select_candidate,nested_probabilities
from editor_trial_analysis import flags


def audit(root,output,*,nested=False):
    import torch
    from crystal_dlm.expert_edit import load_editor_model
    from scripts.train_keep_edit_utility import feature_rows
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise ValueError('runtime parity needs GPU allocation')
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.use_deterministic_algorithms(True)
    device=torch.device('cuda',0);reg=json.loads((root/'PREREGISTRATION.json').read_text())
    frozen=json.loads((root/('NESTED_FROZEN_SELECTION.json' if nested else 'FROZEN_SELECTION.json')).read_text());spec=json.loads((root/'fit/RUN_SPEC.json').read_text())
    validate_rsi_checkpoint(reg['old_editor'],'E')
    if file_hash(frozen['model_path'])!=frozen['model_sha256']:raise ValueError('exported ranker changed')
    model,tokenizer=load_editor_model(spec['assets']['base_model'],reg['old_editor'],device)
    ranker=torch.load(frozen['model_path'],map_location=device,weights_only=False)
    results={}
    for name in ('fit','fresh'):
        data=root/name
        prediction_path=root/(('training/nested_sun_ranker/FIT_PREDICTIONS.jsonl' if nested else 'training/sun_ranker/FIT_PREDICTIONS.jsonl') if name=='fit' else ('fresh/nested_ranker_inference/FRESH_PREDICTIONS.jsonl' if nested else 'fresh/ranker_inference/FRESH_PREDICTIONS.jsonl'))
        expected={(r['stream'],r['ordinal']):r for r in read_rows(prediction_path)}
        current=read_rows(data/'native/inputs.jsonl');previous=Path(reg['previous_run'])
        before=scores(previous/'fit','native') if name=='fit' else scores(data,'native')
        actual={};maximum=0.;count=0
        for stream in STREAMS:
            bank=data/'bank'/stream;rows=read_rows(bank/'FEATURE_ROWS.jsonl')
            destination=output/name/stream;destination.mkdir(parents=True,exist_ok=True)
            payload=feature_rows(model,tokenizer,rows,device,destination,'RUNTIME')
            extra=[];valid=[]
            for row in rows:
                i=row['ordinal'];trace=json.loads((bank/f'candidate/records/{i:04d}.json').read_text())['editor_trace']
                bound=json.loads((bank/f'bound/{i:04d}.json').read_text())
                extra.append(extra_features(before[i],current[i],bound['record'],trace,row['num_sites']))
                valid.append(bool(trace.get('proposal_generated') and bound['commit_trace']['applied']))
            with torch.no_grad():
                hidden=torch.cat([model.quality_head.layers[:2](payload['features'][i:i+16].to(device)) for i in range(0,len(rows),16)])
                x=torch.cat((hidden,torch.tensor(extra,device=device,dtype=torch.float32)),dim=1)
                logits=x@ranker['weight'].T+ranker['bias']
                values=(nested_probabilities(logits) if nested else logits).cpu().tolist()
            for row,value,is_valid in zip(rows,values,valid,strict=True):
                i=row['ordinal'];wanted=expected[(stream,i)]
                maximum=max(maximum,abs(value[0]-wanted['sun_gain']),abs(value[1]-wanted['ms_gain']))
                actual[(stream,i)]=dict(stream=stream,valid=is_valid,sun_gain=value[0],ms_gain=value[1])
                if wanted['valid']!=is_valid:raise ValueError('runtime validity differs')
                count+=1
        mismatches=[]
        for i in range(len(current)):
            def choose(table):
                candidates=[table.get((s,i),dict(stream=s,valid=False,sun_gain=0.,ms_gain=0.)) for s in STREAMS]
                return select_candidate(candidates,sun_threshold=frozen['selected']['sun_threshold'],
                    ms_floor=frozen['selected']['ms_floor'],known_sun=flags(before[i])['SUN'])
            if choose(expected)!=choose(actual):mismatches.append(i)
        results[name]=dict(canonical_views=count,max_prediction_abs_difference=maximum,decision_mismatches=mismatches,
            model_sha256=frozen['model_sha256'],no_candidate_physical_labels_loaded=True)
        write_json(output/name/'PARITY.json',results[name])
        if maximum>1e-5 or mismatches:raise ValueError('exported SUN ranker changed actual full-model decisions')
    write_json(output/'AUDIT_FINAL.json',dict(status='complete',results=results,
        full_editor_checkpoint=reg['old_editor'],full_editor_receipt_sha256=reg['old_editor_receipt_sha256'],
        exported_ranker_sha256=frozen['model_sha256'],fixed_canonical_batch_size=16))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--completion-dir',type=Path,required=True);p.add_argument('--nested',action='store_true')
    a=p.parse_args();audit(a.root,a.completion_dir,nested=a.nested)
