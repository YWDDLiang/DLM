"""Select outputs with physical label-file access denied, then evaluate them."""
import argparse
import builtins
from contextlib import contextmanager
import json
from pathlib import Path
import sys

SOURCE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(SOURCE/'src'),str(SOURCE/'operations/r03_c3fd_main_20260907')]
from scripts.run_post_refine_cycle import write_json,file_hash
from scripts.train_autonomous_value import infer
from evaluate_sun_ranker import build,evaluate,summary


@contextmanager
def deny_physical_results():
    original_open,original_path_open=builtins.open,Path.open
    accesses=[]
    def check(path,mode):
        if isinstance(path,(str,bytes,Path)):
            text=str(path).replace('\\','/')
            if 'r' in str(mode) and ('/scoring/' in text or '/labeling/' in text or
                '/training/nested_sun_ranker/' in text or '/training/sun_ranker/' in text or '/state_utility_execution/' in text or
                Path(text).name in ('labels.jsonl','attempt_results.jsonl')):
                raise RuntimeError('physical result accessed during model selection: '+text)
            accesses.append(text)
    def checked_open(path,mode='r',*args,**kwargs):
        check(path,mode);return original_open(path,mode,*args,**kwargs)
    def checked_path_open(path,mode='r',*args,**kwargs):
        check(path,mode);return original_path_open(path,mode,*args,**kwargs)
    builtins.open=checked_open;Path.open=checked_path_open
    try:yield accesses
    finally:builtins.open=original_open;Path.open=original_path_open


def run(root,kind,panel_name,output,*,experiment_root=None):
    with deny_physical_results() as accesses:
        prediction=infer(root,kind,panel_name)
        panel=build(root,panel_name,'autonomous_'+kind,0.,0.,prediction_path=prediction,
            score_kind='autonomous_centered_learned_NS_NMS_values',learned_keep=True,score_weights=(2.,1.))
    write_json(output/'SELECTION_INPUT_AUDIT.json',dict(physical_result_files_denied=True,
        selection_completed=True,read_and_written_files=sorted(set(accesses)),
        predictions_sha256=file_hash(prediction),selected_inputs_sha256=file_hash(panel/'edited/inputs.jsonl')))
    if panel_name=='fresh':
        from final_improvement import label_fresh_policy
        phase=experiment_root or root.parents[1]
        original=Path(json.loads((phase/'REGISTRATION.json').read_text())['previous'])
        label_fresh_policy(phase,root,panel,original)
    evaluate(root,panel,cached=panel_name=='fit',nu_workers=3)
    roles=['fresh'] if panel_name=='fresh' else ['train','dev','final','all']
    report=dict(complete=True,kind=kind,autonomous_inputs=True,
        physical_feedback_only_after_selection=True,results={role:summary(root,panel,role) for role in roles})
    write_json(output/'RESULT.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--kind',choices=['linear','mlp','distill'],required=True);parser.add_argument('--panel',choices=['fit','fresh'],default='fit')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--experiment-root',type=Path)
    args=parser.parse_args()
    run(args.root,args.kind,args.panel,args.output,experiment_root=args.experiment_root)
