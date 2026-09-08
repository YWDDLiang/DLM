"""Source-level, cross-noise analysis of a registered teacher-action menu."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(run_root):
    root = Path(run_root)
    generated, scored = root/'sun_v05_teacher64', root/'sun_v05_teacher64_scores'
    assert (generated/'_SUCCESS').is_file() and (scored/'_SUCCESS').is_file()
    study_path = generated/'refinement_prepared/STUDY.json'
    study = json.loads(study_path.read_text())
    component = json.loads((generated/'COMPONENT_CONFIG.json').read_text())
    sys.path.insert(0, str(Path(component['source'])/'src'))
    from scripts.generate_sun_feedback_candidates import validate_refinement_receipt
    actual_refined = read_rows(generated/'refined/refined_inputs.jsonl')
    validate_refinement_receipt(study, study_path, generated/'refined', actual_refined)
    refinement = json.loads((generated/'refined/REFINE_FINAL.json').read_text())
    fingerprints_path = generated/'refined/FINGERPRINTS.json'
    assert refinement['outputs_sha256']['FINGERPRINTS.json'] == sha(fingerprints_path)
    fingerprints = json.loads(fingerprints_path.read_text())
    keys = [(x['case_idx'],x['variant'],x['repeat']) for x in fingerprints]
    assert len(keys)==len(set(keys))
    expected_fingerprints = {(x['probe_case'],x['probe_variant'],x['technical_repeat']) for x in actual_refined if x['success']}
    assert set(keys)==expected_fingerprints
    for case in study['cases']:
        assert len(case['refiner_seeds'])==2 and len(set(case['refiner_seeds']))==2
        case_rows = [x for x in actual_refined if x['probe_case']==case['case_idx']]
        assert len({x['probe_gpu_rank'] for x in case_rows})==1
        before_states = []
        for seed in range(2):
            actual_seeds = {x['refiner_seed'] for x in case_rows if x['technical_repeat']==seed}
            assert actual_seeds == {case['refiner_seeds'][seed]}
            group = [x for x in fingerprints if x['case_idx']==case['case_idx'] and x['repeat']==seed]
            assert len({x['cuda_rng_before'] for x in group})<=1
            assert len({x['cuda_rng_after'] for x in group})<=1
            before_states.append({x['cuda_rng_before'] for x in group})
        assert not before_states[0] or not before_states[1] or before_states[0] != before_states[1]
    result = {'schema':'sun_teacher_headroom_capsule_v1', 'study':study, 'study_sha256':sha(study_path), 'arms':{},
              'refinement_report':refinement,'fingerprints':fingerprints,'fingerprints_sha256':sha(fingerprints_path),
              'actual_refined_schedule':[{k:x[k] for k in ['probe_job_index','probe_case','probe_variant',
                  'technical_repeat','refiner_noise_seed_index','refiner_seed','probe_gpu_rank','group_id','source_row_idx','success']} for x in actual_refined],
              'execution_integrity_verified':True}
    variants = study['variants']
    names = ['native_'+v for v in variants]+[f'ref{s}_'+v for s in range(2) for v in variants]
    common_identity, common_runtime = None, None
    for name in names:
        directory = scored/name
        assert (directory/'_SUCCESS').is_file()
        report = json.loads((directory/'FEEDBACK_FINAL.json').read_text())
        assert report['purpose'] == 'training_feedback' and report['status'] == 'complete'
        assert report['independent_evaluation'] is False and report['counts']['requests'] == study['sources']
        source_path = (generated/'candidates/native'/name.removeprefix('native_')/'inputs.jsonl' if name.startswith('native_')
            else generated/'refined_arms'/('seed'+name[3]) / name[5:] / 'inputs.jsonl')
        assert report['input_sha256'] == sha(source_path)
        values = read_rows(directory/'attempt_results.jsonl')
        assert len(values) == len({x['group_id'] for x in values}) == study['sources']
        assert {x['group_id'] for x in values} == {c['ancestor_id'] for c in study['cases']}
        identity_keys = ['physical_model_sha256','frozen_config_sha256','official_cache_sha256',
                         'official_unresolved_sha256','frozen_nu_source_sha256','terminal_protocol',
                         'geometry_validation_protocol','verification_protocol','novelty_uniqueness_endpoint']
        identity = {key:report[key] for key in identity_keys}
        if common_identity is None: common_identity = identity
        assert identity == common_identity
        for binding in report['label_bindings']:
            assert binding['runtime']['deterministic_algorithms_enabled'] is True
            if common_runtime is None: common_runtime = binding['runtime']
            assert binding['runtime'] == common_runtime
        result['arms'][name] = {'report':report, 'rows':values,
                               'rows_sha256':sha(directory/'attempt_results.jsonl'), 'report_sha256':sha(directory/'FEEDBACK_FINAL.json')}
    path = root/'SUN_TEACHER64_COMPACT.json'
    assert not path.exists()
    path.write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}))


def changed_tokens(case, variant):
    body = case['candidates'][variant]['body']
    if body is None:
        return 10**9
    if len(body) != len(case['old_body']):
        raise ValueError('candidate body length differs from OLD')
    return sum(a != b for a,b in zip(case['old_body'],body))


def require_resolved_feedback(row):
    """Formal zero accounting is not evidence that a missing energy is unstable."""
    if type(row['strict_sun']) is not bool:
        raise ValueError('an unresolved SUN predicate cannot create a training preference')
    status = row.get('terminal_status')
    if status in ('generation_failure', 'invalid_raw'):
        if row['strict_sun']:
            raise ValueError('an unusable endpoint cannot acquire positive SUN feedback')
        return
    resolved_statuses = {'verified', 'not_converged', 'optimizer_stop_unverified',
                         'relaxation_energy_increased', 'invalid_terminal'}
    values = [row.get(key) for key in ('terminal_energy_eV_atom', 'hull_energy_eV_atom')]
    if status not in resolved_statuses or any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
            for value in values):
        raise ValueError(f'unresolved physical feedback for {row.get("group_id")}: {status}')


def select_from_one_seed(case, variants, panels, seed):
    gid = case['ancestor_id']
    for variant in variants:
        require_resolved_feedback(panels[seed][variant][gid])
    return min(variants, key=lambda variant:(-int(panels[seed][variant][gid]['strict_sun']),
        0 if variant == 'keep' else 1, changed_tokens(case,variant), variants.index(variant)))


def bootstrap_interval(values, strata):
    rng = np.random.default_rng(2026090843)
    values = np.asarray(values,dtype=float)
    indices = [np.flatnonzero(np.asarray(strata)==value) for value in sorted(set(strata))]
    sampled = sum(values[group[rng.integers(0,len(group),(20000,len(group)))]].sum(1) for group in indices)/len(values)
    return [float(x) for x in np.quantile(sampled*100,[.025,.975])]


def energy_above_hull(row):
    values = [row.get(key) for key in ('terminal_energy_eV_atom', 'hull_energy_eV_atom')]
    if any(value is None or not math.isfinite(value) for value in values):
        return None
    return values[0] - values[1]


def summarize_arm(values):
    energies = [energy_above_hull(row) for row in values]
    finite = [value for value in energies if value is not None]
    return {'sources':len(values),
        **{key:sum(row[key] for row in values) for key in
           ['strict_sun','meta_sun','reconstructed','terminal_verified']},
        'terminal_statuses':dict(Counter(row['terminal_status'] for row in values)),
        'hull_distance_eV_atom':{'known':len(finite),
            'median':float(np.median(finite)) if finite else None,
            'stable_at_most_zero':sum(value<=0 for value in finite),
            'above_zero_to_0p05':sum(0<value<=.05 for value in finite),
            'above_0p05_to_0p1':sum(.05<value<=.1 for value in finite),
            'above_0p1':sum(value>.1 for value in finite)}}


def paired_change(values, baseline):
    previous = {row['group_id']:row for row in baseline}
    delta = [int(row['strict_sun'])-int(previous[row['group_id']]['strict_sun']) for row in values]
    energy_delta = []
    for row in values:
        a,b = energy_above_hull(row), energy_above_hull(previous[row['group_id']])
        if a is not None and b is not None: energy_delta.append(a-b)
    return {'SUN_gains':sum(x==1 for x in delta),'SUN_losses':sum(x==-1 for x in delta),'net_SUN':sum(delta),
        'both_hull_distances_known':len(energy_delta),
        'median_hull_distance_change_eV_atom':float(np.median(energy_delta)) if energy_delta else None}


def analyze(capsule):
    study = capsule['study']
    cases, variants = study['cases'], study['variants']
    expected = {case['ancestor_id'] for case in cases}
    assert len(expected)==len(cases)==study['sources']
    assert len({case['composition_key'] for case in cases})==len(cases)
    for arm in capsule['arms'].values():
        group_ids = [x['group_id'] for x in arm['rows']]
        if len(group_ids)!=len(cases) or len(set(group_ids))!=len(cases) or set(group_ids)!=expected:
            raise ValueError('an arm has duplicate, extra or missing source rows')
    for case in cases:
        for variant in variants: changed_tokens(case,variant)
    panels = {seed:{v:{x['group_id']:x for x in capsule['arms'][f'ref{seed}_{v}']['rows']} for v in variants} for seed in range(2)}
    assert all(set(rows)==expected for group in panels.values() for rows in group.values())
    for group in panels.values():
        for rows in group.values():
            for row in rows.values(): require_resolved_feedback(row)
    output = {'schema':'sun_teacher_headroom_analysis_v1', 'sources':len(cases), 'teacher_oracle_not_student':True,
        'scope':'unique-composition, reference-covered, geometry-stratified training headroom; not natural DEV',
        'within_cohort_uniqueness_is_trivial':True, 'arms':{}, 'repeatable_wins':[], 'cross_noise':{},
        'actual_actions':{}, 'identical_body_inconsistencies':[],
        'execution_integrity_verified':capsule.get('execution_integrity_verified') is True}
    fingerprint_map = {(x['case_idx'],x['variant'],x['repeat']):x for x in capsule.get('fingerprints',[])}
    geometry = {case['ancestor_id']:bool(case['old_geometry']['valid']) for case in cases}
    for name, arm in capsule['arms'].items():
        values = arm['rows']
        assert all(type(row['strict_sun']) is bool and type(row['meta_sun']) is bool for row in values)
        baseline = capsule['arms'][name.split('_',1)[0]+'_keep']['rows']
        output['arms'][name] = {**summarize_arm(values),'vs_KEEP':paired_change(values,baseline),
            'by_original_geometry':{str(valid).lower():summarize_arm([row for row in values if geometry[row['group_id']]==valid])
                                    for valid in [False,True]}}
    for case in cases:
        gid = case['ancestor_id']
        unique = {tuple(c['body']) for c in case['candidates'].values() if c['body'] is not None}
        identical_groups = {}
        for variant in variants:
            body = case['candidates'][variant]['body']
            if body is not None: identical_groups.setdefault(tuple(body),[]).append(variant)
        endpoint_groups = [(f'ref{s}',{v:panels[s][v] for v in variants}) for s in range(2)]
        if all('native_'+v in capsule['arms'] for v in variants):
            endpoint_groups.append(('native',{v:{x['group_id']:x for x in capsule['arms']['native_'+v]['rows']} for v in variants}))
        for aliases in identical_groups.values():
            first = aliases[0]
            for variant in aliases[1:]:
                for endpoint, values in endpoint_groups:
                    a,b = values[first][gid], values[variant][gid]
                    fields = ['terminal_status','terminal_energy_eV_atom','hull_energy_eV_atom','strict_sun','meta_sun','terminal_verified']
                    if any(a.get(key)!=b.get(key) for key in fields):
                        output['identical_body_inconsistencies'].append({'group_id':gid,'variants':[first,variant],'endpoint':endpoint,'kind':'physical_outcome'})
                if 'case_idx' in case and fingerprint_map:
                    for seed in range(2):
                        a,b = [fingerprint_map.get((case['case_idx'],v,seed)) for v in [first,variant]]
                        if (a is None)!=(b is None) or a is not None and a['output_tensors']!=b['output_tensors']:
                            output['identical_body_inconsistencies'].append({'group_id':gid,'variants':[first,variant],'endpoint':f'ref{seed}','kind':'F_output_fingerprint'})
        output['actual_actions'][gid] = {'distinct_available_bodies_including_KEEP':len(unique),
            'variants':{v:{'mode':case['candidates'][v]['mode'], 'sites':case['candidates'][v]['sites'],
                          'changed_tokens':changed_tokens(case,v), 'available':case['candidates'][v]['body'] is not None}
                        for v in variants}}
        winners = []
        for v in variants:
            body = case['candidates'][v]['body']
            if v != 'keep' and body is not None and body != case['old_body'] and all(panels[s][v][gid]['strict_sun'] and not panels[s]['keep'][gid]['strict_sun'] for s in range(2)):
                winners.append(v)
        if winners:
            local = [v for v in winners if case['candidates'][v]['mode']=='local_xyz'
                     and 0 < len(case['candidates'][v]['sites']) < case['num_atoms']]
            output['repeatable_wins'].append({'group_id':gid,'variants':winners,'proper_subset_local_variants':local,
                                              'original_geometry_valid':case['old_geometry']['valid']})
    strata = [bool(case['old_geometry']['valid']) for case in cases]
    for train_seed, test_seed in [(0,1),(1,0)]:
        details = []
        for case in cases:
            gid = case['ancestor_id']
            chosen = select_from_one_seed(case, variants, panels, train_seed)
            before, after = panels[test_seed]['keep'][gid]['strict_sun'], panels[test_seed][chosen][gid]['strict_sun']
            details.append({'group_id':gid,'selected_variant':chosen,'KEEP_SUN':before,'selected_SUN':after,
                            'delta':int(after)-int(before),'original_geometry_valid':case['old_geometry']['valid']})
        delta = [x['delta'] for x in details]
        output['cross_noise'][f'{train_seed}_to_{test_seed}'] = {'net_SUN':sum(delta),
            'gains':sum(x==1 for x in delta),'losses':sum(x==-1 for x in delta),
            'delta_pp':100*sum(delta)/len(delta), 'stratified_source_bootstrap_95pct_pp':bootstrap_interval(delta,strata),
            'selection_counts':dict(Counter(x['selected_variant'] for x in details)),
            'selection_seed_oracle_SUN':sum(panels[train_seed][x['selected_variant']][x['group_id']]['strict_sun'] for x in details),
            'evaluation_seed_selected_SUN':sum(x['selected_SUN'] for x in details),
            'evaluation_seed_KEEP_SUN':sum(x['KEEP_SUN'] for x in details),
            'by_original_geometry':{str(valid).lower():{
                'sources':sum(x['original_geometry_valid']==valid for x in details),
                'gains':sum(x['delta']==1 for x in details if x['original_geometry_valid']==valid),
                'losses':sum(x['delta']==-1 for x in details if x['original_geometry_valid']==valid),
                'net_SUN':sum(x['delta'] for x in details if x['original_geometry_valid']==valid)} for valid in [False,True]},
            'details':details}
    output['SUN_repeatability_by_variant'] = {v:{
        'both_seeds':sum(panels[0][v][gid]['strict_sun'] and panels[1][v][gid]['strict_sun'] for gid in expected),
        'exactly_one_seed':sum(panels[0][v][gid]['strict_sun'] != panels[1][v][gid]['strict_sun'] for gid in expected)}
        for v in variants}
    preservation = []
    for case in cases:
        gid = case['ancestor_id']
        old = [panels[s]['keep'][gid]['strict_sun'] for s in range(2)]
        if any(old):
            preservation.append({'group_id':gid,'KEEP_SUN_by_seed':old,
                'lost_by_variant':{v:[bool(old[s] and not panels[s][v][gid]['strict_sun']) for s in range(2)] for v in variants}})
    output['preservation'] = {'sources_with_KEEP_SUN_any_seed':len(preservation),
        'sources_with_KEEP_SUN_both_seeds':sum(all(x['KEEP_SUN_by_seed']) for x in preservation),
        'details':preservation,'acceptance_training_requires_separate_preservation_coverage_review':True}
    output['gate'] = {'integrity':not output['identical_body_inconsistencies'] and output['execution_integrity_verified'],
        'repeatable_source_wins_at_least_6':len(output['repeatable_wins'])>=6,
        'cross_noise_net_at_least_4_each_direction':all(x['net_SUN']>=4 for x in output['cross_noise'].values()),
        'proper_subset_local_winning_sources_at_least_2':sum(bool(x['proper_subset_local_variants']) for x in output['repeatable_wins'])>=2,
        'criterion_type':'predeclared practical support for bounded pilot, not a success probability'}
    output['gate']['global_content_pilot_supported'] = all(output['gate'][key] for key in
        ['integrity','repeatable_source_wins_at_least_6','cross_noise_net_at_least_4_each_direction'])
    output['gate']['local_content_pilot_supported'] = output['gate']['global_content_pilot_supported'] and output['gate']['proper_subset_local_winning_sources_at_least_2']
    return output


if __name__ == '__main__':
    if sys.argv[1] == '--collect':
        collect(sys.argv[2])
    else:
        path = Path(sys.argv[1])
        output = analyze(json.loads(path.read_text()))
        output['capsule_sha256'] = sha(path)
        target = path.with_name('SUN_TEACHER64_ANALYSIS.json')
        target.write_text(json.dumps(output,indent=2)+'\n')
        print(json.dumps({'path':str(target),'gate':output['gate'],'repeatable_source_wins':len(output['repeatable_wins']),
                          'cross_noise':{k:{kk:vv for kk,vv in v.items() if kk!='details'} for k,v in output['cross_noise'].items()},
                          'preservation':{k:v for k,v in output['preservation'].items() if k!='details'}},indent=2))
