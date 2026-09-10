"""Bind token proposals to one immutable continuous realization before selection."""
import argparse
import json
from pathlib import Path
import sys

SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'src'))
from scripts.run_post_refine_cycle import read_rows,write_json,file_hash
from crystal_dlm.utility_acceptance import materialize_continuous_patch
from crystal_dlm.post_refine_contract import fingerprint


def materialize_proposals(root,candidate,output,*,existing=None):
    from transformers import AutoTokenizer
    root,candidate,output=map(Path,(root,candidate,output))
    if (output/'MATERIALIZATION_FINAL.json').exists():raise ValueError('continuous candidate binding already exists')
    spec=json.loads((root/'fit/RUN_SPEC.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(spec['assets']['b0_checkpoint'],trust_remote_code=True)
    inverse={int(v):k for k,v in tokenizer.get_vocab().items()}
    current=read_rows(root/'fit/current/inputs.jsonl');proposed=read_rows(candidate/'proposal/inputs.jsonl')
    original=read_rows(Path(existing)/'inputs.jsonl') if existing else None
    changed=0;pins=[]
    for i,(before,after) in enumerate(zip(current,proposed,strict=True)):
        if before['original_ordinal']!=i or after['original_ordinal']!=i:raise ValueError('continuous proposal order differs')
        native=json.loads((root/f'fit/native/records/{i:04d}.json').read_text())
        bound=materialize_continuous_patch(native,before.get('body_token_ids') or [],after.get('body_token_ids') or [],inverse)
        bound['record']['trajectory_id']=original[i]['trajectory_id'] if original else after['trajectory_id']+':continuous'
        if original is not None:
            if json.dumps(bound['record'],sort_keys=True)!=json.dumps(original[i],sort_keys=True):
                raise ValueError('canonical continuous reconstruction differs from evaluated input:'+str(i))
            # Retain the actual evaluated serialization, including derived xyz.
            bound['record']=original[i]
        bound['record_sha256']=fingerprint(bound['record'])
        changed+=int(bound['commit_trace']['applied'])
        path=output/f'records/{i:04d}.json';write_json(path,bound);pins.append(file_hash(path))
    result=dict(requests=len(current),actual_continuous_patches=changed,
        native_inputs_sha256=file_hash(root/'fit/native/inputs.jsonl'),
        current_inputs_sha256=file_hash(root/'fit/current/inputs.jsonl'),
        proposal_inputs_sha256=file_hash(candidate/'proposal/inputs.jsonl'),
        exact_existing_evaluated_inputs=existing is not None,
        existing_inputs_sha256=file_hash(Path(existing)/'inputs.jsonl') if existing else None,
        bound_record_sha256=pins,source_commit=(SOURCE/'_CODE_READY').read_text().strip(),
        no_physical_labels_or_model_acceptance_used=True,realize_once_then_select=True)
    write_json(output/'MATERIALIZATION_FINAL.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='bound_record_sha256'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--existing',type=Path);args=parser.parse_args()
    materialize_proposals(args.root,args.candidate,args.output,existing=args.existing)
