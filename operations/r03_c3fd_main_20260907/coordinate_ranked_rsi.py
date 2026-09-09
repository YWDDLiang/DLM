#!/usr/bin/env python3
"""Resumeable coordinator for the approved three TRAIN G/E updates."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess as sp
import sys
import threading
import time

ORCHESTRATOR_SOURCE=Path(__file__).resolve().parents[2]
# A scheduling-only resume can keep every scientific command pinned to its
# original immutable deployment, including already completed local markers.
SOURCE=Path(os.environ.get('RANKED_EXECUTION_SOURCE',str(ORCHESTRATOR_SOURCE))).resolve()
sys.path.insert(0,str(SOURCE/'src'))
from scripts.run_post_refine_cycle import write_json,file_hash,validate_rsi_checkpoint
from run_component import verify_deployed_source


def dispatch_minutes(requested,remaining):
    if remaining < 180:
        raise RuntimeError('less than three minutes remain before the absolute deadline')
    return min(requested,int((remaining-90)//60))


def training_reserve_seconds(index,branch,requests=256,single_gpus=4):
    # At E training the current G/F and training feedback are already complete.
    # Reserve only its final editor/evaluation, plus all future complete rounds.
    if index not in (1,2,3) or branch not in ('G','E') or min(requests,single_gpus)<1:
        raise ValueError('invalid remaining-panel budget inputs')
    scale=(requests/256)*(4/single_gpus)
    # Preserve the measured 256/4-GPU allowance. Startup and reporting do not
    # scale with rows; generation/refinement/physics do. This is a reservation,
    # not a claim that the final runtime is known before the new panel runs.
    complete_round=20+45*scale
    editor_tail=5+15*scale
    return ((4-index)*complete_round+15)*60 if branch=='G' else ((3-index)*complete_round+editor_tail+15)*60


def collection_label_allocation(teacher,parallel_gpus,single_gpus):
    if not teacher.done():
        return parallel_gpus
    # Completed teacher scoring releases its allocation. Propagate failures
    # before starting more work; submitted collection jobs retain their receipts.
    teacher.result()
    return single_gpus


def runtime_allocations(policy,maximum=6):
    values={key:policy.get(key,default) for key,default in {
        'single_GPUs':4,'parallel_main_GPUs':4,'parallel_other_GPUs':2,'training_GPUs':4,
        'training_batch_size':None}.items()}
    for key in ('single_GPUs','parallel_main_GPUs','parallel_other_GPUs','training_GPUs'):
        if type(values[key]) is not int or not 1<=values[key]<=maximum:
            raise ValueError('invalid GPU allocation: '+key)
    if values['parallel_main_GPUs']+values['parallel_other_GPUs']>maximum:
        raise ValueError('parallel allocations exceed the shared GPU budget')
    batch=values['training_batch_size']
    if batch is not None and (type(batch) is not int or not 1<=batch<=64):
        raise ValueError('invalid source batch size')
    return values


class Coordinator:
    def __init__(self,root):
        self.root=Path(root).resolve();self.lock=threading.Lock()
        self.events=self.root/'driver_steps';self.events.mkdir(exist_ok=True)
        self.env={**os.environ,'PYTHONPATH':str(SOURCE/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                  'MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','CUDA_VISIBLE_DEVICES':''}
        budget=self.root/'BUDGET.json'
        if not budget.exists():
            now=dt.datetime.now(dt.timezone.utc)
            write_json(budget,{'dispatch_start_utc':now.isoformat(),'first_GPU_start_utc':None,
                'deadline_utc':(now+dt.timedelta(hours=12)).isoformat(),'wall_hours':12,'max_GPUs':6,
                'max_queued_and_running_jobs':3,'deadline_policy':'conservative_dispatch_plus_12h'})
        self.budget=json.loads(budget.read_text())
        self.deadline=dt.datetime.fromisoformat(self.budget['deadline_utc'])
        self.allocations()
        path=self.root/'RAW0_PIPELINE.json'
        if not path.exists():
            pipeline=json.loads((self.root/'RAW0_PIPELINE_TEMPLATE.json').read_text())
            pipeline.update(source_root=str(SOURCE),source_identity=verify_deployed_source(SOURCE))
            pipeline['resources'].update(deadline_utc=self.deadline.isoformat(),extra_gpu_until_utc=self.deadline.isoformat())
            write_json(path,pipeline)

    def remaining(self): return (self.deadline-dt.datetime.now(dt.timezone.utc)).total_seconds()

    def allocations(self):
        control=json.loads((self.root/'RUN_SPEC.json').read_text())
        return runtime_allocations(control.get('execution_policy',{}),self.budget.get('max_GPUs',6))

    @property
    def parallel_main_gpus(self): return self.allocations()['parallel_main_GPUs']

    @property
    def parallel_other_gpus(self): return self.allocations()['parallel_other_GPUs']

    @property
    def single_gpus(self): return self.allocations()['single_GPUs']

    def local(self,name,script,args):
        marker=self.events/(name+'_DONE.json')
        cmd=[sys.executable,str(SOURCE/script),*map(str,args)]
        if marker.exists():
            if json.loads(marker.read_text())['command']!=cmd: raise ValueError('completed local stage changed: '+name)
            return
        attempt=len(list(self.events.glob(name+'_attempt*.json')))+1
        log=self.events/f'{name}_attempt{attempt}.log';started=time.monotonic()
        with log.open('x') as stream: result=sp.run(cmd,env=self.env,stdout=stream,stderr=sp.STDOUT)
        report={'command':cmd,'seconds':time.monotonic()-started,'returncode':result.returncode,'log':str(log)}
        write_json(self.events/f'{name}_attempt{attempt}.json',report)
        if result.returncode: raise RuntimeError('local stage failed: '+name+'; '+str(log))
        write_json(marker,report)

    def rsi(self,name,config,action,stage=None):
        args=['--config',config,'--action',action]
        if stage: args+=['--stage',stage]
        self.local(name,'src/scripts/run_rsi_stages.py',args)

    def ranked(self,name,config,action,extra=()):
        self.local(name,'src/scripts/run_ranked_rsi.py',['--config',config,'--action',action,*extra])

    def job(self,name,config,action,stage='construction',gpus=4,minutes=90,extra=()):
        recovery=self.root/'JOB_RETRIES.json'
        if recovery.exists():
            name=json.loads(recovery.read_text()).get(name,name)
        receipt=self.root/'submissions'/f'{name}.json'
        while not receipt.exists():
            actual_minutes=dispatch_minutes(minutes,self.remaining())
            args=[sys.executable,str(ORCHESTRATOR_SOURCE/'operations/r03_c3fd_main_20260907/dispatch_rsi.py'),
                  '--root',str(self.root),'--config',str(config),'--job',name,'--action',action,
                  '--stage',stage,'--gpus',str(gpus),'--minutes',str(actual_minutes),*map(str,extra)]
            with self.lock:
                result=sp.run(args,text=True,capture_output=True,env=self.env)
            if result.returncode:
                message=result.stdout+result.stderr
                if any(s in message for s in ('resource budget is occupied','Slurm job count exceeds')):
                    time.sleep(20);continue
                write_json(self.events/(name+'_dispatch_failure.json'),{'command':args,'output':message})
                raise RuntimeError('dispatch failed: '+name+' '+message[-1000:])
        submitted=json.loads(receipt.read_text());job_id=submitted['job_id']
        manifest=json.loads(Path(submitted['manifest']).read_text())
        component=self.root/manifest['components'][0]['output_dir']
        failures=0
        while not (component/'_SUCCESS').exists():
            if self.remaining()<=0: raise RuntimeError('absolute deadline reached')
            state=sp.run(['squeue','-h','-j',job_id,'-o','%T'],text=True,capture_output=True).stdout.strip()
            if not state:
                failures+=1
                if failures>=3:
                    accounting=sp.run(['sacct','-n','-j',job_id,'-o','JobID,State,ExitCode'],text=True,capture_output=True).stdout
                    raise RuntimeError('job stopped without complete outputs: '+name+' '+accounting)
            else: failures=0
            if gpus and not self.budget.get('first_GPU_start_utc') and state.startswith('RUNNING'):
                with self.lock:
                    self.budget['first_GPU_start_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
                    self.budget['first_observed_GPU_job']=job_id
                    self.budget['start_time_precision']='first_running_observation; exact_sacct_time_recorded_in_final'
                    write_json(self.root/'BUDGET.json',self.budget)
            time.sleep(20)
        print(json.dumps({'event':'job_complete','name':name,'job_id':job_id,'remaining_seconds':int(self.remaining())}),flush=True)

    def evaluate(self,prefix,config,stage,gpus=4,caches=()):
        extra=['--reuse-endpoints',*caches] if caches else []
        self.job(prefix+'_label',config,'label',stage,gpus,90,extra)
        self.job(prefix+'_score',config,'score',stage,0,35)

    def body(self,tag,config,gpus=4):
        root=Path(json.loads(Path(config).read_text())['run_root'])
        self.job(tag+'_generate',config,'generate',gpus=gpus)
        self.rsi(tag+'_raw_inputs',config,'materialize','construction')
        self.evaluate(tag+'_raw',config,'construction',gpus)
        self.rsi(tag+'_gate',config,'gate')
        self.job(tag+'_refine',config,'refine',gpus=gpus)
        for stage in ('refined','tokenized'): self.rsi(tag+'_'+stage+'_inputs',config,'materialize',stage)
        self.evaluate(tag+'_token',config,'tokenized',gpus,[root/'construction/labeling/result'])
        self.rsi(tag+'_current_inputs',config,'select')
        self.rsi(tag+'_current_labels',config,'rebind','current')
        self.job(tag+'_current_score',config,'score','current',0,35)

    def final_editor(self,tag,config,gpus=4):
        root=Path(json.loads(Path(config).read_text())['run_root'])
        self.job(tag+'_edit',config,'edit',gpus=gpus)
        for stage in ('proposal','edited'): self.rsi(tag+'_'+stage+'_inputs',config,'materialize',stage)
        self.evaluate(tag+'_proposal',config,'proposal',gpus,[root/'current/labeling/result'])
        self.rsi(tag+'_edited_labels',config,'rebind','edited')
        self.job(tag+'_edited_score',config,'score','edited',0,35)

    def check_update(self,index,current):
        """A TRAIN admission check on the complete G->F outputs, not local loss."""
        fit=json.loads((self.root/'fit/RUN_SPEC.json').read_text())
        if not fit.get('training_policy',{}).get('bounded_minibatch_training'):return
        reports=[]
        for root in [self.root/'fit',current]:
            label=root/'construction/labeling/result/labels.jsonl'
            generation_failures=sum(json.loads(line)['status']=='generation_failure' for line in label.read_text().splitlines() if line)
            scores=json.loads((root/'tokenized/scoring/result/RANKED_METRICS.json').read_text())
            reports.append({'generation_failures':generation_failures,'G_plus_F':scores})
        baseline,actual=reports
        allowance=math.ceil(8*fit['requests']/256)
        accepted=(actual['generation_failures']<=baseline['generation_failures']+allowance and
                  actual['G_plus_F']['reliable_SUN']>=baseline['G_plus_F']['reliable_SUN'])
        report={'round':index,'accepted':accepted,'baseline':baseline,'actual':actual,
            'requests':fit['requests'],'generation_failure_allowance':allowance,
            'rule':'TRAIN same-condition complete G+F SUN must not fall below S0; generation failures may rise by ceil(requests*8/256)',
            'does_not_prove_heldout_generalization':True}
        write_json(self.root/f'G{index}_ADMISSION.json',report)
        if not accepted and not json.loads((self.root/'RUN_SPEC.json').read_text()).get('complete_flow_before_judgment'):
            raise RuntimeError('complete G+F regression: update rejected before downstream training')

    def training(self,index,branch,data,replay):
        overrides=self.root/'TRAINING_CONFIG_OVERRIDES.json'
        names=json.loads(overrides.read_text()) if overrides.exists() else {}
        name=names.get(f'{branch}{index}',f'TRAIN_{branch}{index}.json')
        if Path(name).name!=name: raise ValueError('training override must be a filename')
        config=self.root/'training_configs'/name
        fit=json.loads((self.root/'fit/RUN_SPEC.json').read_text())
        if not config.exists():
            allocation=self.allocations()
            remaining_updates=8-2*index-(branch=='E')
            # Reserve complete inference, physical feedback and reporting for
            # every remaining round. Allocation changes with actual elapsed work.
            reserve=training_reserve_seconds(index,branch,fit['requests'],allocation['single_GPUs'])
            seconds=int((self.remaining()-reserve)/remaining_updates)
            if seconds<180: raise RuntimeError('insufficient training budget after reserving complete panels')
            checkpoint=(fit['assets']['b0_checkpoint'] if branch=='G' else fit['assets']['editor_checkpoint']) if index==1 else str(
                self.root/'training'/f'round{index-1}'/branch/'result/checkpoint')
            if index>1: validate_rsi_checkpoint(checkpoint,branch)
            tuning=fit.get('training_policy',{})
            bounded=tuning.get('bounded_minibatch_training',False)
            if bounded:
                # Current G data already includes comparisons to historical candidates.
                # E trains on the present policy's current-state distribution.
                seconds=min(seconds,int(tuning.get('max_training_seconds',1800)))
            spec={'ranked_training':True,'branch':branch,'base_model':fit['assets']['base_model'],
                'checkpoint':checkpoint,'data':str(data),'replay_data':[] if bounded else [str(p) for p in replay],
                'output_dir':str(self.root/'training'/f'round{index}'/branch/'result'),
                'seed':20260909+100*index+(37 if branch=='E' else 0),'learning_rate':tuning.get('G_learning_rate',5e-6) if bounded and branch=='G' else 2e-5,
                'head_learning_rate':1e-4,'beta':.1,'anchor_weight':.2,'accumulation':2,
                'steps':4096,'mask_cuts':2,'max_training_seconds':seconds,'update_index':index,
                'training_gpus':allocation['training_GPUs'],
                'source_round':index-1 if branch=='G' else index,
                'time_allocation':{'remaining_seconds':self.remaining(),'reserved_panel_seconds':reserve,
                    'remaining_weight_updates':remaining_updates,'rule':'share_remaining_after_complete_panel_reserve'}}
            if bounded:
                spec.update(bounded_minibatch_training=True,
                    batch_size=allocation['training_batch_size'] or tuning.get('batch_size',8),
                    accumulation=1,epochs=tuning.get(branch+'_epochs',4 if branch=='G' else 8),
                    reference_kl_weight=tuning.get('reference_kl_weight',1.),
                    max_reference_kl=tuning.get('max_reference_kl',.02),
                    continue_heads_after_content_KL=tuning.get('continue_heads_after_content_KL',False))
            write_json(config,spec)
        training=json.loads(config.read_text())
        self.job(f'train_{branch}{index}',self.root/'fit/RUN_SPEC.json','train',gpus=training.get('training_gpus',4),
            minutes=math.ceil(training['max_training_seconds']/60)+15,extra=['--training-config',config])

    def run(self):
        fit=self.root/'fit';initial=fit/'RUN_SPEC.json';twin=self.root/'bootstrap_comparator'
        started=time.monotonic()
        # Four main GPUs and two throughput GPUs, always under the shared guard.
        with ThreadPoolExecutor(max_workers=2) as pool:
            other=pool.submit(self.body,'twin',twin/'RUN_SPEC.json',self.parallel_other_gpus)
            self.body('S0',initial,self.parallel_main_gpus)
            self.job('initialize_E0',initial,'initialize',gpus=1,minutes=30)
            self.final_editor('S0',initial,self.parallel_main_gpus)
            other.result()
        if not (self.root/'S0_COMPLETE.json').exists():
            write_json(self.root/'S0_COMPLETE.json',{'seconds':time.monotonic()-started,
                'config_sha256':file_hash(initial),'initial_E_seed':20260909,
                'timer_scope':'current_coordinator_attempt'})
        history=[twin,fit];g_data=[];e_data=[]
        self.ranked('S0_compile_G',initial,'compile',['--branch','G','--comparators',twin])
        g_data.append(fit/'pairs/G.jsonl')
        self.ranked('S0_archive',initial,'archive')
        for index in (1,2,3):
            self.training(index,'G',g_data[-1],g_data[:-1])
            self.local(f'round{index}_generation_register','src/scripts/prepare_rsi_round.py',
                ['--root',self.root,'--index',index,'--generation-only'])
            current=self.root/'rounds'/f'round{index}'/'fit';config=current/'RUN_SPEC.json'
            self.body(f'S{index}',config,self.single_gpus)
            if not json.loads((self.root/'RUN_SPEC.json').read_text()).get('complete_flow_before_judgment'):
                self.check_update(index,current)
            self.ranked(f'S{index}_compile_G',config,'compile',['--branch','G','--comparators',*history])
            g_data.append(current/'pairs/G.jsonl');history.append(current)
            previous=fit/'initialization/checkpoint' if index==1 else self.root/'training'/f'round{index-1}'/'E/result/checkpoint'
            self.ranked(f'S{index}_collection_register',config,'collection',['--checkpoint',previous])
            collection=current/'editor_collection';collect_config=collection/'RUN_SPEC.json'
            use_history=json.loads(initial.read_text()).get('training_policy',{}).get('historical_Stable_teachers',False)
            teacher_history=history[:-1] if use_history else []
            self.ranked(f'S{index}_teachers',collect_config,'teachers',
                        ['--comparators',*teacher_history] if teacher_history else [])
            teacher_caches=[current/'current/labeling/result',*[prior/stage/'labeling/result'
                for prior in teacher_history for stage in ('current','edited')
                if (prior/stage/'inputs.jsonl').exists()]]
            # Teacher endpoint scoring and old-E proposal collection are independent.
            with ThreadPoolExecutor(max_workers=2) as pool:
                teacher=pool.submit(self.evaluate,f'S{index}_teacher',collect_config,'teacher',self.parallel_main_gpus,
                                    teacher_caches)
                self.job(f'S{index}_collect',collect_config,'edit',gpus=self.parallel_other_gpus)
                self.rsi(f'S{index}_collect_inputs',collect_config,'materialize','proposal')
                label_gpus=collection_label_allocation(teacher,self.parallel_other_gpus,self.single_gpus)
                self.evaluate(f'S{index}_collect',collect_config,'proposal',label_gpus,[current/'current/labeling/result'])
                teacher.result()
            self.ranked(f'S{index}_compile_E',collect_config,'compile',['--branch','E'])
            data=collection/'pairs/E.jsonl'
            self.training(index,'E',data,e_data);e_data.append(data)
            self.ranked(f'S{index}_collection_archive',collect_config,'archive')
            self.local(f'round{index}_edit_register','src/scripts/prepare_rsi_round.py',
                ['--root',self.root,'--index',index])
            edited=current/'EDIT_SPEC.json'
            self.final_editor(f'S{index}',edited,self.single_gpus)
            if json.loads((self.root/'RUN_SPEC.json').read_text()).get('complete_flow_before_judgment'):
                self.check_update(index,current)
                write_json(self.root/f'S{index}_FULL_FLOW.json',{
                    'round':index,'completed_draft_refine_token_edit_keep':True,
                    'baseline':json.loads((fit/'edited/scoring/result/RANKED_METRICS.json').read_text()),
                    'actual':json.loads((current/'edited/scoring/result/RANKED_METRICS.json').read_text()),
                    'training_diagnostic_only':True,'G_plus_F_does_not_stop_full_flow':True})
            self.ranked(f'S{index}_archive',edited,'archive')
        reports=[]
        for index,current in enumerate([fit,*[self.root/'rounds'/f'round{i}'/'fit' for i in (1,2,3)]]):
            for stage in ('construction','tokenized','edited'):
                directory=current/stage/'scoring/result'
                reports.append({'round_index':index,'stage':stage,
                    'ranked':json.loads((directory/'RANKED_METRICS.json').read_text()),
                    'basic':json.loads((directory/'BASIC_METRICS.json').read_text())})
        write_json(self.root/'RANKED_RUN_FINAL.json',{'status':'complete','reports':reports,'budget':self.budget,
            'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'source':verify_deployed_source(SOURCE),
            'actual_weight_updates':{f'{b}{i}':validate_rsi_checkpoint(self.root/'training'/f'round{i}'/b/'result/checkpoint',b)
                                     for i in (1,2,3) for b in ('G','E')}})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    write_json(args.root/'COORDINATOR_PID.json',{'pid':os.getpid(),'source':str(SOURCE),
        'orchestrator_source':str(ORCHESTRATOR_SOURCE),'orchestrator_sha256':file_hash(Path(__file__)),
        'scientific_source_identity':verify_deployed_source(SOURCE)})
    try: Coordinator(args.root).run()
    except Exception as error:
        write_json(args.root/'COORDINATOR_FAILURE.json',{'error':repr(error),'utc':dt.datetime.now(dt.timezone.utc).isoformat()})
        raise
