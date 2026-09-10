"""Report registered and strictly composition-isolated FINAL without reselection."""
from collections import Counter,defaultdict
import argparse
import datetime as dt
import json
from pathlib import Path
import sys

SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'src'))
from scripts.run_post_refine_cycle import read_rows,write_json,file_hash
from scripts.run_rsi_stages import scores,score_directory
from editor_trial_analysis import flags


def composition_admission(root):
    from pymatgen.core import Composition
    rows=read_rows(root/'SOURCE_SPLIT.jsonl');groups=defaultdict(list)
    for row in rows:groups[Composition(row['reduced_formula']).reduced_formula].append(row)
    cross={k:v for k,v in groups.items() if len({x['split'] for x in v})>1}
    protected={formula for formula,group in groups.items() if any(x['split']!='final' for x in group)}
    excluded=[row['ordinal'] for row in rows if row['split']=='final'
        and Composition(row['reduced_formula']).reduced_formula in protected]
    admitted=[row['ordinal'] for row in rows if row['split']=='final' and row['ordinal'] not in excluded]
    return dict(schema='post_registration_canonical_composition_audit_v1',
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_split_sha256=file_hash(root/'SOURCE_SPLIT.jsonl'),
        original_FINAL=[x['ordinal'] for x in rows if x['split']=='final'],
        strict_composition_isolated_FINAL=admitted,excluded_FINAL=excluded,cross_partition_formulas=cross,
        original_split_and_frozen_selection_unchanged=True,exclusion_uses_composition_only=True,
        source_identity_unseen_and_composition_unseen_reported_separately=True,
        U_still_computed_on_original_full_1000_panel=True)


def compare(root,panel,stage,indices):
    before=scores(root/'fit','native');after=scores(panel,stage)
    decisions=json.loads((panel/'DECISION_BINDING.json').read_text())['decisions']
    names=('Stable','MS','SUN','MSUN');base=Counter();counts=Counter();gains={k:[] for k in names};losses={k:[] for k in names}
    for i in indices:
        a,b=flags(before[i]),flags(after[i])
        for name in names:
            base[name]+=int(a[name]);counts[name]+=int(b[name])
            if b[name] and not a[name]:gains[name].append(i)
            if a[name] and not b[name]:losses[name].append(i)
        counts['known_failure']+=int(b['known_failure']);counts['unknown']+=int(not b['reliable'] and not b['known_failure'])
        counts['actual_edits']+=int(decisions[i]['actual_edit'])
    return dict(requests=len(indices),KEEP=dict(base),counts=dict(counts),gains=gains,losses=losses,
        delta={name:counts[name]-base[name] for name in names},
        scores_sha256=file_hash(score_directory(panel,stage)/'attempt_results.jsonl'),
        decision_sha256=file_hash(panel/'DECISION_BINDING.json'))


def repeated_uncertainty(root,panels,indices):
    import numpy as np
    from pymatgen.core import Composition
    names=('Stable','MS','SUN','MSUN');before=scores(root/'fit','native')
    sources=read_rows(root/'SOURCE_SPLIT.jsonl');groups=defaultdict(list)
    for i in indices:groups[Composition(sources[i]['reduced_formula']).reduced_formula].append(i)
    ordered=sorted(groups);matrix=[]
    for key in ('primary','repeat1','repeat2'):
        panel,stage=panels[key];after=scores(panel,stage)
        matrix.append([[int(flags(after[i])[name])-int(flags(before[i])[name]) for name in names] for i in indices])
    values=np.asarray(matrix,dtype=float);average=values.mean(0);positions={i:j for j,i in enumerate(indices)}
    aggregate=np.asarray([average[[positions[i] for i in groups[g]]].sum(0) for g in ordered])
    sizes=np.asarray([len(groups[g]) for g in ordered])
    rng=np.random.default_rng(20260910);draws=rng.integers(len(ordered),size=(10000,len(ordered)))
    boot=100.*aggregate[draws].sum(1)/sizes[draws].sum(1)[:,None]
    intervals=np.quantile(boot,[.025,.975],axis=0)
    return dict(streams=['primary','repeat1','repeat2'],sources=len(indices),composition_clusters=len(ordered),
        mean_net_count={name:float(values[:,:,j].sum(1).mean()) for j,name in enumerate(names)},
        mean_percentage_point_delta={name:float(100.*average[:,j].mean()) for j,name in enumerate(names)},
        paired_composition_cluster_bootstrap_95pct_percentage_point_interval={name:intervals[:,j].tolist() for j,name in enumerate(names)},
        bootstrap_replicates=10000,bootstrap_seed=20260910,
        inference_scope='sampling uncertainty conditional on the fixed model, observed input cohort, physical protocol and three E seeds',
        does_not_quantify_CHGNet_or_hull_reference_uncertainty=True,
        all_three_SUN_and_MSUN_strictly_positive=bool(np.all(values[:,:,2].sum(1)>0) and np.all(values[:,:,3].sum(1)>0)))


def main(root):
    frozen=json.loads((root/'FROZEN_SELECTION.json').read_text())
    admission=composition_admission(root);path=root/'analysis/FINAL_COMPOSITION_ADMISSION.json'
    if path.exists():
        prior=json.loads(path.read_text())
        for field in ('source_split_sha256','strict_composition_isolated_FINAL','excluded_FINAL'):
            if prior[field]!=admission[field]:raise ValueError('composition audit admission changed')
    else:write_json(path,admission)
    panels={'primary':(Path(frozen['selected']['dev']['panel']),'edited'),
        'old_E3':(root/'policies/old_E3_canonical_continuous','edited')}
    pending=[]
    for index in (1,2):
        panel=root/f'repeats/repeat{index}'
        if (score_directory(panel,'continuous_edited')/'_SUCCESS').exists():panels[f'repeat{index}']=(panel,'continuous_edited')
        else:pending.append(f'repeat{index}')
    reports={name:{'registered_FINAL':compare(root,panel,stage,admission['original_FINAL']),
                  'strict_composition_FINAL':compare(root,panel,stage,admission['strict_composition_isolated_FINAL'])}
        for name,(panel,stage) in panels.items()}
    report=dict(schema='frozen_E_primary_and_repeated_FINAL_audit_v1',
        selection_sha256=file_hash(root/'FROZEN_SELECTION.json'),composition_admission_sha256=file_hash(path),
        excluded_original_FINAL=admission['excluded_FINAL'],reports=reports,pending=pending,
        no_weight_threshold_or_seed_reselection=True,
        original_cohort_metrics_preserved=True,metadata_based_strict_subset_is_an_additional_audit=True)
    if not pending:report['strict_subset_repeated_uncertainty']=repeated_uncertainty(root,panels,admission['strict_composition_isolated_FINAL'])
    output=root/'analysis'/('FINAL_AND_REPEATS_COMPLETE.json' if not pending else 'FINAL_AND_REPEATS_PROGRESS.json')
    write_json(output,report)
    print(json.dumps(dict(excluded=admission['excluded_FINAL'],pending=pending,reports={
        name:{kind:dict(counts=v['counts'],KEEP=v['KEEP'],delta=v['delta'],requests=v['requests'])
            for kind,v in values.items()} for name,values in reports.items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
