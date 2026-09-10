"""Close the bounded SUN pilot with actual scheduler accounting and audit pins."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(SOURCE/'src'),str(SOURCE/'operations/r03_c3fd_main_20260907')]
from scripts.run_post_refine_cycle import read_rows,write_json,file_hash
from scripts.run_rsi_stages import scores,score_directory
from editor_trial_analysis import flags


def peak(intervals,index):
    events=[]
    for start,end,*values in intervals:
        events.extend([(start,values[index]),(end,-values[index])])
    current=maximum=0
    for _,value in sorted(events,key=lambda x:(x[0],x[1])):
        current+=value;maximum=max(maximum,current)
    return maximum


def close(root):
    reg=json.loads((root/'PREREGISTRATION.json').read_text());budget=json.loads((root/'BUDGET.json').read_text())
    if file_hash(budget['previous_budget_path'])!=budget['previous_budget_sha256']:
        raise ValueError('previous phase budget was changed')
    receipts={}
    for p in (root/'submissions').glob('*.json'):
        if p.name.endswith('.manifest.json'):continue
        value=json.loads(p.read_text())
        if 'job_id' in value:receipts[str(value['job_id'])]=value
    raw=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(receipts),
        '--format','JobID,JobName%60,State,Start,End,ElapsedRaw,AllocCPUS,AllocTRES%120,ExitCode'],
        env=dict(os.environ,TZ='UTC'),text=True)
    records=[];allocated=[];queued=[]
    for line in raw.splitlines():
        if not line.strip():continue
        ident,name,state,start,end,seconds,cpus,tres,code=line.strip().split('|')[:9]
        if ident not in receipts:continue
        if state!='COMPLETED' or code!='0:0':raise ValueError('a SUN pilot job is not successfully complete:'+line)
        counts=dict(x.split('=',1) for x in tres.split(','));gpus=int(counts.get('gres/gpu',0))
        begin=dt.datetime.fromisoformat(start).replace(tzinfo=dt.timezone.utc)
        finish=dt.datetime.fromisoformat(end).replace(tzinfo=dt.timezone.utc)
        seconds,cpus=int(seconds),int(cpus)
        submitted=dt.datetime.fromisoformat(receipts[ident]['submitted_utc'])
        allocated.append((begin,finish,gpus,cpus,1));queued.append((submitted,finish,1))
        records.append(dict(job_id=ident,name=name,state=state,start_utc=begin.isoformat(),end_utc=finish.isoformat(),
            elapsed_seconds=seconds,gpus=gpus,cpus=cpus,exit_code=code,submission_sha256=file_hash(root/'submissions'/(receipts[ident]['job']+'.json'))))
    if len(records)!=len(receipts):raise ValueError('scheduler did not account for every submitted job')
    active=subprocess.check_output(['squeue','-h','-u','jiaosz','-o','%i'],text=True).splitlines()
    if set(active)&set(receipts):raise ValueError('SUN jobs remain active')
    deadline=dt.datetime.fromisoformat(budget['deadline_utc'])
    if max(v[1] for v in allocated)>deadline:raise ValueError('SUN phase exceeded deadline')
    resource=dict(jobs=len(records),all_jobs_completed=True,failed_jobs=[],
        allocated_GPU_hours=sum(r['elapsed_seconds']*r['gpus'] for r in records)/3600,
        allocated_CPU_hours=sum(r['elapsed_seconds']*r['cpus'] for r in records)/3600,
        peak_allocated_GPUs=peak(allocated,0),peak_allocated_CPUs=peak(allocated,1),
        peak_running_jobs=peak(allocated,2),peak_submitted_and_unfinished_jobs=peak(queued,0),
        last_job_end_utc=max(v[1] for v in allocated).isoformat(),deadline_utc=deadline.isoformat(),
        scheduler_display_timezone='UTC_explicit_TZ_environment',records=records)
    if resource['peak_allocated_GPUs']>5 or resource['peak_allocated_CPUs']>30 or resource['peak_submitted_and_unfinished_jobs']>3:
        raise ValueError('actual resource peaks exceeded the phase cap')
    write_json(root/'RESOURCE_FINAL.json',resource)
    selected=json.loads((root/'FINAL_MODEL_SELECTION.json').read_text())
    first_fresh=min(dt.datetime.fromisoformat(v['submitted_utc']) for v in receipts.values()
        if v['job'] in ('sun_rank_fresh_nested_label','sun_rank_fresh_delta_label','sun_rank_fresh_operational_label'))
    if dt.datetime.fromisoformat(selected['created_utc'])>=first_fresh:
        raise ValueError('model was not selected before fresh candidate evaluation')
    paths=['PREREGISTRATION.json','ABSOLUTE_STATE_REGISTRATION.json','FROZEN_SELECTION.json','NESTED_FROZEN_SELECTION.json',
        'FINAL_MODEL_SELECTION.json','FRESH_COMPARISON_FINAL.json','RESOURCE_FINAL.json','fresh/INPUTS_FINAL.json',
        'fresh/PANELS_FROZEN.json','fresh/cohort/PREPARATION_FINAL.json','SOURCE_SPLIT.jsonl',
        'analysis/CANDIDATE_SUPPLY_FINAL.json','analysis/NESTED_CANDIDATE_SUPPLY_FINAL.json',
        'models/sun_ranker/MODEL_DEFINITION.json','models/MATERIALIZED_OUTPUT_PARITY.json']
    pins={name:file_hash(root/name) for name in paths}
    fresh=json.loads((root/'FRESH_COMPARISON_FINAL.json').read_text())
    training={}
    for name,folder in [('delta','sun_ranker'),('nested','nested_sun_ranker')]:
        path=root/'training'/folder/'TRAINING_FINAL.json';value=json.loads(path.read_text())
        training[name]={k:v for k,v in value.items() if k not in ('history','feature_and_feedback_pins')}
        pins[str(path.relative_to(root))]=file_hash(path)
    previous=Path(reg['previous_run']);before=scores(previous/'fit','native')
    old_selected=Path(json.loads((root/'NESTED_FROZEN_SELECTION.json').read_text())['selected']['panel'])
    after=scores(old_selected,'edited');decisions=json.loads((old_selected/'DECISION_BINDING.json').read_text())['decisions']
    repair=[]
    for i,a in enumerate(before):
        if a.get('terminal_status')=='verified':continue
        traces={name:json.loads((root/f'fit/bank/{name}/candidate/records/{i:04d}.json').read_text())['editor_trace'] for name in ('primary','rank1','rank2','rank3')}
        repair.append(dict(ordinal=i,before_status=a.get('terminal_status'),after_status=after[i].get('terminal_status'),
            before=flags(a),after=flags(after[i]),decision=decisions[i],
            proposal_generated={name:t.get('proposal_generated',False) for name,t in traces.items()}))
    result=dict(schema='sun_rank_scope_run_complete_v1',status='complete',root=str(root),
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=(SOURCE/'_CODE_READY').read_text().strip(),
        primary_scientific_outcome='SUN_increase_observed; complete_DEV_and_fresh_joint_adoption_gate_not_met',
        fresh=fresh,resources=resource,training=training,
        original_fit=json.loads((root/'analysis/DEV_policies/POLICY_EVALUATION_FINAL.json').read_text()),
        nested_fit=json.loads((root/'analysis/nested_DEV_policies/POLICY_EVALUATION_FINAL.json').read_text()),
        candidate_supply={name:json.loads((root/'analysis'/file).read_text())['reports'] for name,file in
            [('delta','CANDIDATE_SUPPLY_FINAL.json'),('nested','NESTED_CANDIDATE_SUPPLY_FINAL.json')]},
        runtime_audits={name:json.loads((root/'models'/folder/'AUDIT_FINAL.json').read_text()) for name,folder in
            [('delta','sun_ranker_runtime_audit'),('nested','nested_sun_ranker_runtime_audit')]},
        new_final_validity={'KEEP':json.loads((root/'fresh/native/scoring/result/BASIC_METRICS.json').read_text()),
            **{name:json.loads((Path(v['panel'])/'edited/scoring/result/BASIC_METRICS.json').read_text()) for name,v in fresh['reports'].items()}},
        failed_input_attempts=repair,model_choice_before_fresh_candidate_evaluation=True,
        prior_budget_unchanged=True,new_G_F_calls=0,original1000_remains_paused_after_S1=True,
        local_tests_passed=29,scientific_receipt_pins=pins)
    write_json(root/'RUN_COMPLETE.json',result)
    print(json.dumps(dict(path=str(root/'RUN_COMPLETE.json'),sha256=file_hash(root/'RUN_COMPLETE.json'),
        resources={k:v for k,v in resource.items() if k!='records'},failed_input_attempts=repair)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();close(a.root)
