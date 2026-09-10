"""Reserve untouched cached G/F inputs outside prior studies and formal TRAIN."""
import argparse
import json
from pathlib import Path
import shutil
import sys

SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'src'))
from scripts.run_post_refine_cycle import read_rows,write_rows,write_json,file_hash
from scripts.run_sun_rank_scope import fresh_inputs
from crystal_dlm.post_refine_contract import derived_seed,fingerprint
from crystal_dlm.sun_feedback_contract import composition_counts,reduced_key


def prepare(root):
    previous=Path(json.loads((root/'REGISTRATION.json').read_text())['previous'])
    spec=json.loads((previous/'fresh/RUN_SPEC.json').read_text())
    assets=spec['assets'];study=root/'sealed_validation_v2';panel=study/'fresh'
    if (study/'DATA_FINAL.json').exists():
        return study
    def key(row):return reduced_key(composition_counts(row['plan_state']))
    forbidden=set();exclusion_counts={}
    sources={'previous_fit_1000':previous/'fit/cohort/plans.jsonl',
        'previous_fresh_256':previous/'fresh/cohort/plans.jsonl',
        'formal_train_1000':root/'formal_preparation/portable/CLEAN_TRAIN_1000.jsonl',
        'H1A2_all_1200':Path(assets['cohort']),
        'R03_all_256':root/'formal_preparation/R03_256_PLANS.jsonl'}
    for name,path in sources.items():
        values=set()
        for row in read_rows(path):
            try:values.add(key(row))
            except (KeyError,ValueError,TypeError):
                if row.get('body_eligible'):raise
        exclusion_counts[name]=len(values);forbidden.update(values)
    pool=Path(assets['training_preparation'])/'pairs_pending.jsonl'
    covered={row['chemsys'] for row in read_rows(Path(assets['official_cache'])/'official_slim_cache.jsonl')}
    eligible=[]
    for row in read_rows(pool):
        plan=row['plan_state']
        if (row['source_split']=='train' and plan.get('rich_field_valid') and plan.get('plan_end_marker_present')
            and key(row) not in forbidden and row.get('teacher_available') and row.get('target_body')
            and '-'.join(sorted(plan['elements'])) in covered):eligible.append(row)
    unique={}
    for row in sorted(eligible,key=lambda r:fingerprint(dict(domain='final_improvement_validation_source_v1',source=r['ancestor_id']))):
        unique.setdefault(key(row),row)
    order=sorted(unique,key=lambda k:fingerprint(dict(domain='final_improvement_validation_composition_v1',composition=k)))
    if not order:raise ValueError('no unused, covered cached input compositions')
    parents=[unique[k] for k in order[:256]]
    count=len(parents)
    parent_path=panel/'cohort/parents.jsonl';write_rows(parent_path,parents)
    plans=[dict(original_ordinal=i,evaluation_ordinal=i,sample_idx=i,body_eligible=True,
        ancestor_id=row['ancestor_id'],source_row_idx=row['source_row_idx'],source_split='train',
        dataset_origin='planner_generated',original_split='generated',original_assignment='synthetic_training_pool',
        usage_role='heldout_validation',E_evaluation_role='sealed_validation',
        plan_state=row['plan_state'],body_prompt=row['prompt'],
        body_noise_seed=derived_seed(row['ancestor_id'],'final_improvement_validation_E'),
        cached_teacher=True,canonical_composition=key(row)) for i,row in enumerate(parents)]
    plan_path=panel/'cohort/plans.jsonl';write_rows(plan_path,plans)
    report=dict(schema='composition_excluded_cached_GF_validation_v1',sources=count,requested_upper_bound=256,
        dataset_origin='planner_generated',original_split='generated',usage_role='heldout_validation',
        original_assignment='synthetic_training_pool',E_supervised_use=False,
        files_sha256={'parents.jsonl':file_hash(parent_path)},plans_sha256=file_hash(plan_path),
        heldout_cohort_sha256=file_hash(assets['cohort']),source_pool_sha256=file_hash(pool),
        exclusions=exclusion_counts,forbidden_compositions=len(forbidden),eligible_compositions=len(unique),
        selection='fixed source/composition hash before inspecting physical quality or novelty',
        conditioning='available cached B0/F800 input plus static thermodynamic-reference coverage',
        formal_train_overlap=0,no_endpoint_outcomes_loaded=True,
        exclusion_source_sha256={name:file_hash(path) for name,path in sources.items()})
    write_json(panel/'cohort/PREPARATION_FINAL.json',report)
    write_json(panel/'cohort/MANIFEST.json',report);(panel/'cohort/_SUCCESS').touch()
    spec.update(run_root=str(panel),run_id='final_improvement_sealed_validation',requests=count,
        training_parent_root=str(panel/'cohort'),E_evaluation_role='sealed_validation')
    spec['assets']['nu_cache']=str(root/'nu_cache')
    spec['resources'].update(deadline_utc='2026-09-10T19:16:40+00:00',
        budget_receipt=str(root/'WINDOW_AMENDMENT_20260910T161640Z.json'))
    write_json(panel/'RUN_SPEC.json',spec)
    shutil.copy2(previous/'PHYSICS_SOURCE_PIN.json',study/'PHYSICS_SOURCE_PIN.json')
    fresh_inputs(study)
    write_json(study/'DATA_FINAL.json',report)
    write_json(root/'formal_preparation/ADDITIONAL_VALIDATION_EXCLUSIONS.json',dict(
        validation_plans_sha256=file_hash(plan_path),compositions=sorted(key(p) for p in plans),
        already_fixed_formal_train_1000_changed=False,
        rule='exclude these compositions from any future expansion of formal training data'))
    print(json.dumps({k:v for k,v in report.items() if k!='exclusion_source_sha256'}),flush=True)
    return study


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    prepare(parser.parse_args().root)
