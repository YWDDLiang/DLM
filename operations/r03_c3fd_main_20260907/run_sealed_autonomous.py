"""Freeze the autonomous development choice, then validate once on unused sources."""
import argparse
import json
from pathlib import Path
import shutil
import sys

SOURCE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(SOURCE/'src'),str(SOURCE/'operations/r03_c3fd_main_20260907')]
from scripts.run_post_refine_cycle import read_rows,write_json,file_hash
from scripts.run_sun_rank_scope import collect
from final_improvement import collect_keep_features
from evaluate_autonomous_value import run as evaluate_autonomous


def generate(root):
    study=root/'sealed_validation_v2'
    candidates=[]
    for name,kinds in [('mini_2e6',['linear','mlp','distill']),('retained_e3',['linear','mlp','distill']),('mini_k8',['mlp','distill'])]:
        for kind in kinds:
            folder=root/'variants'/name/('autonomous_'+kind)
            path=folder/'fit_evaluation/RESULT.json'
            if not path.exists():continue
            report=json.loads(path.read_text())
            if not report['autonomous_inputs']:raise ValueError('development candidate lacks autonomous input audit')
            total={metric:sum(report['results'][role]['delta'][metric] for role in ['dev','final'])
                   for metric in ['SUN','MSUN','Stable']}
            candidates.append(dict(name=name,kind=kind,model=str(folder/'MODEL.pt'),
                model_sha256=file_hash(folder/'MODEL.pt'),development_delta=total,
                development_result_sha256=file_hash(path)))
    chosen=max(candidates,key=lambda row:(row['development_delta']['SUN'],row['development_delta']['MSUN'],
        row['development_delta']['Stable'],row['name'],row['kind']))
    parent=root/'variants'/chosen['name'];reg=json.loads((parent/'PREREGISTRATION.json').read_text())
    selection=dict(schema='autonomous_value_sealed_validation_selection_v1',selected=chosen,candidates=candidates,
        selection_basis='exploratory development selection over the already reviewed 247+246 source panels',
        selection_order=['combined_development_SUN_gain','combined_development_MSUN_gain','combined_development_Stable_gain'],
        unseen_validation_quality_consulted=False,unseen_sources=96,utility_weights=[2.,1.],
        runtime_inputs='Plan, current/proposed structures and learned model features only',
        future_policy_changes_on_this_validation=False)
    write_json(study/'FROZEN_SELECTION.json',selection)
    write_json(root/'AUTONOMOUS_FROZEN_SELECTION.json',selection)
    reg.update(autonomous_value_model=chosen['model'],autonomous_value_kind=chosen['kind'],
        sealed_validation_data_sha256=file_hash(study/'DATA_FINAL.json'))
    write_json(study/'PREREGISTRATION.json',reg)
    spec=json.loads((study/'fresh/RUN_SPEC.json').read_text())
    spec['assets']['editor_checkpoint']=reg['old_editor']
    write_json(study/'fresh/RUN_SPEC.json',spec)
    model_dir=study/('autonomous_'+chosen['kind']);model_dir.mkdir(exist_ok=True)
    shutil.copy2(chosen['model'],model_dir/'MODEL.pt')
    print(json.dumps(dict(event='autonomous_choice_frozen',selected=chosen)),flush=True)
    collect(study,'fresh',study/'generation_execution')
    collect_keep_features(study,'fresh')
    write_json(study/'generation_execution/DONE.json',dict(complete=True,selected=chosen,
        no_physical_or_novelty_labels_used_for_candidate_generation=True))


def evaluate(root):
    study=root/'sealed_validation_v2'
    chosen=json.loads((study/'FROZEN_SELECTION.json').read_text())['selected']
    if file_hash(study/('autonomous_'+chosen['kind'])/'MODEL.pt')!=chosen['model_sha256']:
        raise ValueError('selected autonomous model changed')
    evaluate_autonomous(study,chosen['kind'],'fresh',study/'final_evaluation',experiment_root=root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['generate','evaluate'])
    parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    (generate if args.mode=='generate' else evaluate)(args.root)
