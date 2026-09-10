"""One declared same-Plan restart for the remaining physical failure."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE/'src'), str(SOURCE/'scripts')]
from scripts.run_post_refine_cycle import (read_rows, write_rows, write_json, constructor_api,
    load_refiner, refine_one, structure_from_refined, physics_record)
from crystal_dlm.post_refine_contract import derived_seed


def run(root):
    import torch
    from crystal_dlm import r03_geometry_bridge as bridge
    from crystal_dlm.r03_physics_transfer import build_repair_constraints, geometry_support_report
    from crystal_dlm.construction_recovery import construct_cascade
    output = root/'failure_restart'
    spec = json.loads((root/'failure_probe/RUN_SPEC.json').read_text())
    plan = next(r for r in read_rows(root/'failure_probe/cohort/plans.jsonl') if r['original_ordinal']==549)
    seed = derived_seed(str(plan['body_noise_seed']), 'physical_failure_same_plan_restart')
    write_json(output/'REGISTRATION.json', dict(original_ordinal=549, original_body_seed=plan['body_noise_seed'],
        restart_body_seed=seed, generator_checkpoint=spec['assets']['generator_checkpoint'],
        fixed_plan=plan['plan_state'], attempts=1, role='failure_recovery_diagnostic',
        no_training_on_failure_outcomes=True, no_selection_from_candidate_physical_labels=True))
    plan = dict(plan, body_noise_seed=seed)
    torch.set_num_threads(1); torch.cuda.set_device(0); torch.use_deterministic_algorithms(True)
    device = torch.device('cuda',0)
    native = constructor_api()
    runtime = native.load_frozen_runtime(Path(spec['assets']['frozen_runtime']))
    with native.frozen_imports(runtime):
        task = native.prepare_tasks([plan],runtime,seed=17029)[0]
        api = runtime.module
        model,tokenizer = api.load_model_and_tokenizer(spec['assets']['base_model'],
            spec['assets']['generator_checkpoint'],device)
        constraints = api.build_dynamic_lightweight_constraints(tokenizer,duplicate_coordinate_mask=True,
            lattice_volume_mask=True,min_lattice_rad=1e-4)
        process_one = api.import_process_one(Path(spec['assets']['crysllmgen']))
    support = build_repair_constraints(tokenizer)
    calls = [0]
    def count(_model,_inputs): calls[0] += 1
    hook = model.register_forward_pre_hook(count)
    with torch.no_grad(),native.frozen_imports(runtime):
        body,trace = construct_cascade(model,tokenizer,task,runtime,construct=native.construct_batch,
            constraints=constraints,repair_constraints=support,geometry_api=bridge,
            complete_geometry=lambda ids:geometry_support_report(ids,constraints=support),adaptive_lattice=True)
        generated,graph = native.materialize_record(task,body[0].tolist(),runtime=runtime,
            tokenizer=tokenizer,process_one=process_one)
    hook.remove()
    raw = physics_record(plan,state_id='failure_restart:549:raw',body=generated['text'])
    write_json(output/'RAW.json',dict(record=raw,trace=trace,DLM_forward_calls=calls[0],graph_available=graph is not None))
    del model,tokenizer; gc.collect(); torch.cuda.empty_cache()
    records = [raw]
    if graph is not None:
        model,Data,DataLoader = load_refiner(spec,device)
        value = refine_one(graph,model=model,Data=Data,DataLoader=DataLoader,seed=plan['refiner_noise_seed'],steps=800)
        write_json(output/'F_OUTPUT.json',value)
        records.append(physics_record(plan,state_id='failure_restart:549:F',structure=structure_from_refined(value)))
        del model,Data,DataLoader; gc.collect(); torch.cuda.empty_cache()
    write_rows(output/'inputs.jsonl',records)
    os.environ['RSI_JOINT_PHYSICAL_STOP']='1'; os.environ['RSI_STRESS_TOLERANCE']='.5'
    os.environ['R03_DETERMINISTIC_LABELING']='1'
    import label_programmed_paths as physics
    physics.worker_init(0)
    labels=[]
    for record in records:
        label=physics.worker_label(record,.1,.5,1000)
        label=json.loads(json.dumps(label,default=physics.json_default))
        labels.append(label)
        write_json(output/(record['trajectory_id'].replace(':','_')+'.json'),label)
        print(json.dumps({k:label.get(k) for k in ['trajectory_id','status','verified','terminal_energy','actual_steps','error']}),flush=True)
    write_json(output/'DONE.json',dict(complete=True,labels=labels,DLM_forward_calls=calls[0],
        learned_KEEP_EDIT_policy_success=False,scope='one_same_Plan_generator_restart_followed_by_F800'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
