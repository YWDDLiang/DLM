#!/usr/bin/env python3
"""Collect complete, hash-bound RSI metrics without selecting favorable runs."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_post_refine_cycle import file_hash,read_rows,write_json,validate_rsi_checkpoint
from scripts.run_rsi_stages import scores,score_directory
from crystal_dlm.post_refine_contract import paired_change


def summarize(root):
    root=Path(root).resolve()
    result={'observed_utc':datetime.now(timezone.utc).isoformat(),'stages':[],
            'pending_stages':[],'stage_comparisons':[],'same_plan_weight_comparisons':[],
            'training':[],'complete_joint_weight_updates':0}
    panels={};flat=[]
    for index in range(4):
        for name,base in [('MAIN',root if index==0 else root/'rounds'/f'round{index}'),
                          ('FIT',root/'fit' if index==0 else root/'rounds'/f'round{index}'/'fit')]:
            plans=base/'cohort/plans.jsonl'
            if not plans.exists(): continue
            expected=root/('cohort/plans.jsonl' if name=='MAIN' else 'fit/cohort/plans.jsonl')
            if file_hash(plans)!=file_hash(expected): raise ValueError('Plan or seed bytes differ across weight versions')
            stages=['construction','refined','tokenized','edited']
            if name=='MAIN' and index==0:
                stages+=['baselines/H1A2_raw','baselines/H1A2','baselines/R03_raw','baselines/R03']
            local={}
            for stage in stages:
                directory=score_directory(base,stage);summary=directory/'BASIC_METRICS.json'
                if not summary.exists():
                    result['pending_stages'].append({'cohort':name,'theta':index,'stage':stage});continue
                metrics=json.loads(summary.read_text())
                for path,key in [(base/stage/'inputs.jsonl','inputs_sha256'),
                                 (directory/'attempt_results.jsonl','score_sha256'),
                                 (directory/'four_metrics.jsonl','metrics_sha256')]:
                    if file_hash(path)!=metrics[key]: raise ValueError('metric artifact changed: '+str(path))
                measured=scores(base,stage)
                if len(measured)!=len(read_rows(plans)): raise ValueError('metric denominator differs from complete fixed cohort')
                panels[name,index,stage]=local[stage]=measured
                row={'cohort':name,'theta':index,'stage':stage,'requested':metrics['requested'],
                    'comp_valid':metrics['comp_valid']['count'],'Struct_valid':metrics['Struct_valid']['count'],
                    'SUN':metrics['stability']['strict_sun']['count'],'MSUN':metrics['stability']['meta_sun']['count']}
                for metric in ('comp_valid','Struct_valid','SUN','MSUN'):
                    row[metric+'_percent']=None if row[metric] is None else 100*row[metric]/row['requested']
                flat.append(row)
                result['stages'].append({**row,'plans_sha256':file_hash(plans),'report':str(summary),
                    'report_sha256':file_hash(summary),'inputs_sha256':metrics['inputs_sha256'],
                    'score_sha256':metrics['score_sha256']})
            for before,after in [('construction','refined'),('refined','tokenized'),('tokenized','edited')]:
                if before in local and after in local:
                    result['stage_comparisons'].append({'cohort':name,'theta':index,'before':before,'after':after,
                                                       **paired_change(local[before],local[after])})
            if index:
                for stage in ['construction','refined','tokenized','edited']:
                    if (name,0,stage) in panels and stage in local:
                        result['same_plan_weight_comparisons'].append({'cohort':name,'theta_before':0,
                            'theta_after':index,'stage':stage,**paired_change(panels[name,0,stage],local[stage])})
    for index in range(1,4):
        complete=0
        for branch in ('G','E'):
            path=root/'training'/f'round{index}'/branch/'result/checkpoint'
            if not (path/'RSI_TRAINING_DONE.json').exists(): continue
            value=validate_rsi_checkpoint(path,branch);complete+=1
            result['training'].append({'update':index,'branch':branch,'checkpoint':str(path),
                'receipt_sha256':file_hash(path/'RSI_TRAINING_DONE.json'),
                **{k:value[k] for k in ['optimizer_steps','parameter_delta_squared','local_preference_examples',
                                        'local_decision_examples','training_seconds']}})
        result['complete_joint_weight_updates']+=int(complete==2)
    report=root/'RSI_PROGRESS.json';write_json(report,result)
    if flat:
        with (root/'RSI_METRICS.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(summarize(args.root)))
