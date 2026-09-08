"""Paired source-cluster analysis of fixed-checkpoint gauge repeats and deployment."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

def indexed(rows):
    result = {x['group_id']:x for x in rows}
    assert len(result) == len(rows) == 64
    assert all(x['versions']['deterministic_algorithms_enabled'] is True for x in rows)
    assert not any(x['status'] in ['worker_error','worker_timeout'] for x in rows)
    return result

def paired_interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(2026090837)
    draws = rng.integers(0,len(values),(20000,len(values)))
    return [float(v) for v in np.quantile(values[draws].mean(1)*100,[.025,.975])]

def main(directory):
    root = Path(directory)
    names = ['GAUGE64_COMPACT.json','GAUGE64_OLD_COMPACT.json','GAUGE64_REPEATS_COMPACT.json']
    first, old_capsule, repeated = [json.loads((root/name).read_text()) for name in names]
    old = indexed(old_capsule['dev']['rows'])
    seeds = [2026090813,2026090913,2026091013,2026091113]
    arms = ['control','aligned']
    panels = {(arm,seed):indexed(first[arm+':dev']['rows'] if seed==seeds[0]
              else repeated['labels'][arm+':'+str(seed)]['rows']) for arm in arms for seed in seeds}
    assert all(set(rows)==set(old) for rows in panels.values())
    tasks = {gid:x['trajectory_id'].split(':')[-1] for gid,x in panels[('control',seeds[0])].items()}
    def geo(x, before): return x['status'] != 'invalid_raw'
    def verified(x, before): return bool(x['verified'])
    def gain(x, before): return bool(x['verified'] and before['verified'] and before['terminal_energy']-x['terminal_energy'] >= .01)
    def worsens(x, before): return bool(x['verified'] and before['verified'] and x['terminal_energy']-before['terminal_energy'] >= .01)
    metrics = [('all','geometry_support',geo),('all','R_verified',verified),
               ('G','geometry_support',geo),('G','R_verified',verified),
               ('S','geometry_support',geo),('S','R_verified',verified),
               ('S','verified_gain_ge_0.01',gain),('S','verified_worsens_ge_0.01',worsens)]
    output = {'schema':'expert_gauge64_repeats_analysis_v1','seeds':seeds,
        'independent_sources':64,'observations':256,'S_independent_sources':32,
        'S_observations':128,'MAIN_or_SUN_claim':False,'paired_effects':{},
        'capsule_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},
        'bootstrap':'20000 paired resamples of original source clusters; all four generation seeds kept together',
        'autonomous':{}}
    for task, name, fn in metrics:
        gids = sorted(g for g in old if task=='all' or tasks[g]==task)
        values = {a:np.array([[fn(panels[(a,s)][g],old[g]) for s in seeds] for g in gids],dtype=float) for a in arms}
        diff = values['aligned']-values['control']
        output['paired_effects'][task+':'+name] = {
            'independent_sources':len(gids),'denominator':len(gids)*len(seeds),
            'control':int(values['control'].sum()),'aligned':int(values['aligned'].sum()),
            'aligned_minus_control_pp':float(diff.mean()*100),
            'paired_source_bootstrap_95pct_interval_pp':paired_interval(diff.mean(1)),
            'per_seed':{str(s):{a:int(values[a][:,i].sum()) for a in arms} for i,s in enumerate(seeds)},
            'paired_source_details':[{'group_id':g,**{a:[int(v) for v in values[a][i]] for a in arms}} for i,g in enumerate(gids)]}
    aold = indexed(repeated['labels']['autonomous_old']['rows'])
    output['autonomous']['old'] = {'sources':64,'geometry_support':sum(geo(x,None) for x in aold.values()),
                                   'R_verified':sum(verified(x,None) for x in aold.values())}
    for arm in arms:
        rows = indexed(repeated['labels'][arm+':autonomous']['rows'])
        samples = repeated['autonomous_samples'][arm]
        assert len(samples)==64 and set(rows)==set(aold)=={x['ancestor_id'] for x in samples}
        counts = Counter()
        failures = []
        for sample in samples:
            gid = sample['ancestor_id']
            result = sample['output']
            counts['forward_calls'] += result['forward_calls']
            changed = sample['old_body'] != result['body']
            counts['changed_sources'] += changed
            for trace in result['trace']:
                task = trace['task']
                counts[task+':reason:'+trace.get('reason','')] += 1
                counts[task+':mode:'+trace.get('mode','')] += 1
                counts[task+':accepted'] += bool(trace.get('accepted'))
                counts[task+':applied'] += bool(trace.get('applied'))
            if not changed and any(aold[gid].get(k)!=rows[gid].get(k) for k in ['status','verified','terminal_energy','actual_steps']):
                failures.append({'group_id':gid,'old':{k:aold[gid].get(k) for k in ['status','terminal_energy','actual_steps']},'new':{k:rows[gid].get(k) for k in ['status','terminal_energy','actual_steps']}})
        output['autonomous'][arm] = {'sources':64,'geometry_support':sum(geo(x,None) for x in rows.values()),
            'R_verified':sum(verified(x,None) for x in rows.values()),
            'verified_original_old_to_final_gain_ge_0.01':sum(gain(x,aold[g]) for g,x in rows.items()),
            'verified_original_old_to_final_worsens_ge_0.01':sum(worsens(x,aold[g]) for g,x in rows.items()),
            'action_counts':dict(counts),'unchanged_endpoint_label_mismatch':failures,
            'gain_attribution':'end-to-end ORIGINAL OLD to final committed body, not isolated S contribution'}
    (root/'GAUGE64_REPEATS_ANALYSIS.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({**output,'paired_effects':{k:{kk:vv for kk,vv in v.items() if kk!='paired_source_details'} for k,v in output['paired_effects'].items()}},indent=2))

if __name__=='__main__':
    main(sys.argv[1])
