#!/usr/bin/env python3
"""Exact endpoint cache reuse around the unchanged registered physics workers."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import label_programmed_paths as physics
from crystal_dlm.sun_feedback_contract import validate_training_feedback


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def grouped(records):
    result={}
    for record in records:
        geometry=json.dumps(record['structure'],sort_keys=True) if record.get('structure') is not None else str(record.get('body'))
        key=hashlib.sha256(geometry.encode()).hexdigest() if record['success'] else record['trajectory_id']
        result.setdefault(key,[]).append(record)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-jsonl',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--purpose',choices=['evaluation','training_feedback'],required=True)
    p.add_argument('--feedback-manifest',type=Path)
    p.add_argument('--gpu-count',type=int,default=2);p.add_argument('--workers-per-gpu',type=int,default=2)
    p.add_argument('--record-timeout',type=float,default=300.)
    p.add_argument('--worker-startup-timeout',type=float,default=120.)
    p.add_argument('--deterministic',action='store_true')
    p.add_argument('--reuse-endpoints',type=Path,nargs='*',default=[])
    args=p.parse_args()
    args.fmax=.1;args.stress_tolerance=.5;args.max_steps=500
    os.environ['R03_DETERMINISTIC_LABELING']='1' if args.deterministic else '0'
    physics.configure_deterministic_execution(args.deterministic)
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('cached labeling requires a registered allocation')
    if not 1<=args.gpu_count<=6 or args.gpu_count>len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')):
        raise ValueError('physics worker count exceeds allocated GPUs')
    if not 1<=args.workers_per_gpu<=4: raise ValueError('invalid worker count')
    records=rows(args.input_jsonl)
    if not records or len({r['trajectory_id'] for r in records})!=len(records): raise ValueError('invalid complete input ledger')
    for record in records:
        physics.validate_record_purpose(record,args.purpose)
        if args.purpose=='evaluation' and record.get('source_split')!='evaluation': raise ValueError('TRAIN/EVAL cache mixing forbidden')
    scope=validate_training_feedback(records,args.input_jsonl,args.feedback_manifest) if args.purpose=='training_feedback' else None
    by_endpoint=grouped(records);runtime=physics.runtime_identity();protocol=physics.COMMON_RELAXATION_PROTOCOL
    cached={};provenance=[]
    for directory in args.reuse_endpoints:
        report_path=directory/'LABEL_FINAL.json';report=json.loads(report_path.read_text())
        if not (directory/'_SUCCESS').is_file() or (directory/'_ENGINEERING_FAILED').exists(): raise ValueError('incomplete physics cache')
        original=Path(report['input_file'])
        if digest(original)!=report['input_sha256']: raise ValueError('cache input ledger changed')
        old_records=rows(original)
        if args.purpose=='training_feedback':
            old_scope=validate_training_feedback(old_records,original,original.parent/'FEEDBACK_MANIFEST.json')
            if old_scope!=report['training_feedback_scope']: raise ValueError('cache training provenance changed')
        elif any(r.get('source_split')!='evaluation' for r in old_records): raise ValueError('TRAIN/EVAL cache mixing forbidden')
        verified,pin=physics.reusable_labels(directory,grouped(old_records),input_sha256=digest(original),
                                             purpose=args.purpose,protocol=protocol,runtime=runtime)
        provenance.append(pin)
        for key,value in verified.items():
            if key not in by_endpoint: continue
            if key in cached and any(cached[key].get(f)!=value.get(f) for f in ('status','raw_energy','terminal_energy','verified')):
                raise ValueError('same physical endpoint has contradictory cached evidence')
            cached[key]=value
    # Failed generation requires no physical model call, and remains a failure.
    cheap_failures=0
    for key,occurrences in by_endpoint.items():
        if key not in cached and occurrences[0]['success'] is False:
            value=physics.label_record(occurrences[0],model=None,optimizer=None,
                        fmax=args.fmax,stress_tolerance=args.stress_tolerance,max_steps=args.max_steps)
            value['versions']=runtime;cached[key]=value;cheap_failures+=1
    pending={k:v for k,v in by_endpoint.items() if k not in cached}
    args.output_dir.mkdir(parents=True,exist_ok=False);(args.output_dir/'trajectories').mkdir()
    started=time.monotonic();counts=Counter();complete=0
    def results():
        for key,value in cached.items(): yield key,by_endpoint[key],dict(value)
        if pending: yield from physics.bounded_labels(pending,args)
    with (args.output_dir/'labels.jsonl').open('x') as stream:
        for key,occurrences,result in results():
            trajectory=result.pop('relaxation_trajectory',None)
            if trajectory is not None:
                path=args.output_dir/'trajectories'/f'{key}.json.gz'
                with gzip.open(path,'wt') as handle: json.dump(trajectory,handle)
                result['trajectory_file']=str(path)
            for occurrence in occurrences:
                labelled=dict(result,**{name:occurrence.get(name) for name in
                               ('trajectory_id','group_id','source_row_idx','source_split','endpoint')})
                labelled['endpoint_cache_key']=key
                stream.write(json.dumps(labelled,default=physics.json_default)+'\n')
                counts[result['status']]+=1;complete+=1
            stream.flush()
            if complete%32==0: print(json.dumps({'completed':complete,'requested':len(records),
                    'cached_endpoints':len(cached)-cheap_failures,'new_endpoint_evaluations':len(pending),
                    'seconds':time.monotonic()-started}),flush=True)
    report={'requested':len(records),'completed':complete,'statuses':dict(counts),'purpose':args.purpose,
        'protocol':protocol,'verification_protocol':physics.TERMINAL_VERIFICATION_PROTOCOL,
        'geometry_validation_protocol':physics.LABEL_GEOMETRY_PROTOCOL,'runtime_identities':[runtime],
        'input_file':str(args.input_jsonl.resolve()),'input_sha256':digest(args.input_jsonl),
        'training_feedback_scope':scope,'physical_reuse_sources':provenance,
        'cached_endpoints':len(cached)-cheap_failures,'generation_failures_without_model_calls':cheap_failures,
        'distinct_endpoint_evaluations':len(by_endpoint),'new_endpoint_evaluations':len(pending),
        'elapsed_seconds':time.monotonic()-started,'gpu_count':args.gpu_count,'workers_per_gpu':args.workers_per_gpu,
        'deterministic_algorithms_requested':args.deterministic,'cache_wrapper_sha256':digest(__file__),
        'engineering_deadlines':{'record_timeout_seconds':args.record_timeout,'worker_startup_timeout_seconds':args.worker_startup_timeout}}
    (args.output_dir/'LABEL_FINAL.json').write_text(json.dumps(report,indent=2)+'\n')
    if counts.get('worker_error',0):
        (args.output_dir/'_ENGINEERING_FAILED').touch();raise SystemExit(2)
    (args.output_dir/'_SUCCESS').touch();print(json.dumps(report),flush=True)


if __name__=='__main__': main()
