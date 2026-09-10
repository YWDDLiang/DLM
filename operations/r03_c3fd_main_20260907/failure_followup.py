"""Physical follow-up of the two original failures, with explicit proposal scope."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

SOURCE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(SOURCE/'src'),str(SOURCE/'scripts')]
from scripts.run_post_refine_cycle import read_rows,write_json,write_rows,load_refiner,refine_one,structure_from_refined,physics_record
from crystal_dlm.post_refine_contract import derived_seed


def run(root):
    import torch
    from crystal_dlm.expert_edit import load_editor_model
    from crystal_dlm.rsi_minibatch import propose_ranked_batch
    from crystal_dlm.r03_physics_transfer import build_repair_constraints
    from crystal_dlm.utility_acceptance import materialize_continuous_patch
    output=root/'failure_followup'
    previous=Path(json.loads((root/'REGISTRATION.json').read_text())['previous'])
    spec=json.loads((root/'failure_probe/RUN_SPEC.json').read_text())
    plans={r['original_ordinal']:r for r in read_rows(root/'failure_probe/cohort/plans.jsonl')}
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.use_deterministic_algorithms(True)
    records=[];traces={}
    recovered=json.loads((root/'failure_probe/construction/records/0837.json').read_text())
    records.append(dict(recovered['record'],trajectory_id='failure_followup:837:raw'))
    model,Data,DataLoader=load_refiner(spec,torch.device('cuda',0))
    graph=torch.load(root/'failure_probe/construction/graphs/0837.pt',map_location='cpu',weights_only=False)
    value=refine_one(graph,model=model,Data=Data,DataLoader=DataLoader,seed=plans[837]['refiner_noise_seed'],steps=800)
    write_json(output/'837_F_OUTPUT.json',value)
    records.append(physics_record(plans[837],state_id='failure_followup:837:F',structure=structure_from_refined(value)))
    del model,Data,DataLoader;gc.collect();torch.cuda.empty_cache()
    model,tokenizer=load_editor_model(spec['assets']['base_model'],spec['assets']['editor_checkpoint'],torch.device('cuda',0))
    current=read_rows(previous/'fit/current/inputs.jsonl')[549]
    native=json.loads((previous/'fit/native/records/0549.json').read_text())
    request=dict(prompt=plans[549]['body_prompt'],body=current['body_token_ids'],n=plans[549]['plan_state']['N'],
        seed=derived_seed(str(plans[549]['body_noise_seed']),'final_failure_full_cell'),known_sun=False,
        force_proposal=True,exploration_mode='full_cell')
    trace=propose_ranked_batch(model,tokenizer,[request],support=build_repair_constraints(tokenizer),batch_size=1)[0]
    inverse={int(v):k for k,v in tokenizer.get_vocab().items()}
    bound=materialize_continuous_patch(native,current['body_token_ids'],trace['proposal_tokens'],inverse)
    write_json(output/'549_FULL_CELL.json',dict(trace=trace,bound=bound))
    record=dict(bound['record'],trajectory_id='failure_followup:549:full_cell')
    if not trace['proposal_generated'] or not bound['commit_trace']['applied']:
        record.update(success=False,body=None,structure=None,reason='global_repair_proposal_failed')
    records.append(record)
    del model,tokenizer;gc.collect();torch.cuda.empty_cache()
    write_rows(output/'inputs.jsonl',records)
    label_records(root,output)


def label_records(root,output):
    records=read_rows(root/'failure_followup/inputs.jsonl')
    recovered=json.loads((root/'failure_probe/construction/records/0837.json').read_text())
    trace=json.loads((root/'failure_followup/549_FULL_CELL.json').read_text())['trace']
    os.environ['RSI_JOINT_PHYSICAL_STOP']='1';os.environ['RSI_STRESS_TOLERANCE']='.5'
    os.environ['R03_DETERMINISTIC_LABELING']='1'
    import label_programmed_paths as physics
    physics.worker_init(0)
    labels=[]
    for record in records:
        label=physics.worker_label(record,.1,.5,1000)
        label=json.loads(json.dumps(label,default=physics.json_default))
        labels.append(label)
        write_json(output/(record['trajectory_id'].replace(':','_')+'.json'),label)
        print(json.dumps({k:label.get(k) for k in ['trajectory_id','status','verified','raw_energy','terminal_energy','actual_steps','error']}),flush=True)
    write_json(output/'DONE.json',dict(complete=True,diagnostic_only=True,
        global_scope_is_not_a_learned_policy_success=True,
        labels=labels,original_837_generation_calls=recovered['trace']['DLM_forward_calls'],
        original_549_edit_calls=trace['forward_calls']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--labels-only',action='store_true');args=parser.parse_args()
    if args.labels_only:
        label_records(args.root,args.root/'failure_physics')
    else:
        run(args.root)
