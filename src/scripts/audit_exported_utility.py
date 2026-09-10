"""Persist full-model decisions and exact field differences for an exported E."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_post_refine_cycle import read_rows,write_rows,write_json,file_hash,validate_rsi_checkpoint


def differences(left,right,path=''):
    if isinstance(left,dict) and isinstance(right,dict):
        result=[]
        for key in sorted(set(left)|set(right)):
            if key not in left or key not in right:
                result.append(dict(path=path+'/'+key,left=left.get(key),right=right.get(key)))
            else:result.extend(differences(left[key],right[key],path+'/'+key))
        return result
    if isinstance(left,(list,tuple)) and isinstance(right,(list,tuple)):
        if len(left)!=len(right):return [dict(path=path,left=left,right=right)]
        return [item for i,(a,b) in enumerate(zip(left,right)) for item in differences(a,b,path+'/'+str(i))]
    if left==right:return []
    value=dict(path=path,left=left,right=right)
    if isinstance(left,(int,float)) and isinstance(right,(int,float)):
        value['absolute_difference']=abs(left-right)
    return [value]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--completion-dir',type=Path,required=True);args=parser.parse_args();root=args.root;output=args.completion_dir
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from crystal_dlm.expert_edit import load_editor_model
    from crystal_dlm.utility_acceptance import canonical_judgements,continuous_decision
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('export audit needs a GPU')
    torch.cuda.set_device(0);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);device=torch.device('cuda',0)
    frozen=json.loads((root/'FROZEN_SELECTION.json').read_text());chosen=frozen['selected'];epoch=str(chosen['epoch'])
    checkpoint=root/'models/selected_utility/checkpoint';receipt=validate_rsi_checkpoint(checkpoint,'E')
    policy=json.loads((checkpoint/'QUALITY_UTILITY_POLICY.json').read_text())
    if policy['selection_sha256']!=file_hash(root/'FROZEN_SELECTION.json'):raise ValueError('export selection differs')
    spec=json.loads((root/'fit/RUN_SPEC.json').read_text())
    model,tokenizer=load_editor_model(spec['assets']['base_model'],checkpoint,device)
    reference_head=None
    if policy.get('reference_acceptance_required'):
        reference_head=copy.deepcopy(model.quality_head)
        reference_head.load_state_dict(torch.load(checkpoint/'REFERENCE_QUALITY_HEAD.pt',map_location=device,weights_only=True))
    rows=read_rows(root/'evaluation_features/EVAL_ROWS.jsonl')
    actual,reference=canonical_judgements(model,tokenizer,rows,reference_head=reference_head)
    predicted={r['ordinal']:r for r in read_rows(Path(frozen['predictions_path']))}
    drift=max(abs(v-predicted[i]['learned_utilities'][epoch]) for i,v in actual.items())
    if drift>1e-5:raise ValueError('full-model utility changed')
    reference_drift=max((abs(v-predicted[i]['original_quality_logit']) for i,v in reference.items()),default=0.)
    if reference_drift>1e-5:raise ValueError('full-model reference acceptance changed')
    current=read_rows(root/'fit/current/inputs.jsonl');panel=Path(chosen['dev']['panel'])
    expected=read_rows(panel/'edited/inputs.jsonl');decisions=json.loads((panel/'DECISION_BINDING.json').read_text())['decisions']
    inverse={int(v):k for k,v in tokenizer.get_vocab().items()};mismatches=[];decision_differences=[];actual_decisions=[]
    fields=('structure','body','body_token_ids','success','reason')
    for i,before in enumerate(current):
        native=json.loads((root/f'fit/native/records/{i:04d}.json').read_text())
        trace=json.loads((root/f'fit/proposal/records/{i:04d}.json').read_text())['editor_trace']
        result,decision=continuous_decision(native,before.get('body_token_ids',[]),trace,inverse,actual.get(i),chosen['margin'],
            reference_logit=reference.get(i),reference_required=bool(policy.get('reference_acceptance_required')))
        write_json(output/f'actual_records/{i:04d}.json',dict(record=result,decision=decision))
        delta=differences({k:result.get(k) for k in fields},{k:expected[i].get(k) for k in fields})
        if delta:mismatches.append(dict(ordinal=i,actual_edit=decision['actual_edit'],differences=delta))
        if any(decision[k]!=decisions[i][k] for k in ('learned_accept','actual_edit')):decision_differences.append(i)
        actual_decisions.append(dict(ordinal=i,**decision))
    write_rows(output/'ACTUAL_SCORES.jsonl',[dict(ordinal=i,utility=v,reference_logit=reference.get(i)) for i,v in actual.items()])
    write_rows(output/'ACTUAL_DECISIONS.jsonl',actual_decisions)
    report=dict(status='audit_complete',requests=len(current),checkpoint=str(checkpoint),
        checkpoint_receipt_sha256=file_hash(checkpoint/'RSI_TRAINING_DONE.json'),
        selection_sha256=file_hash(root/'FROZEN_SELECTION.json'),strict_reload_probe_passed=True,
        maximum_utility_drift=drift,maximum_reference_drift=reference_drift,
        raw_field_mismatches=mismatches,decision_mismatches=decision_differences,
        actual_edits=sum(int(x['actual_edit']) for x in actual_decisions),
        fields_compared=list(fields),weight_or_policy_changes=False,
        original_export_receipt=receipt,slurm_job_id=os.environ['SLURM_JOB_ID'])
    write_json(output/'EXPORT_AUDIT_FINAL.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('raw_field_mismatches','original_export_receipt')}|
        dict(raw_field_mismatches_count=len(mismatches),first_mismatches=mismatches[:3])),flush=True)


if __name__=='__main__':main()
