import json,os,subprocess as s
from pathlib import Path
g=Path('/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/proposal_realization_candidates_20260826/grounding')
r=g/'experiments/r03_c3fd_rich_20260907/run_20260907T033526Z'
out={'queue':s.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%T|%b|%C|%M'],text=True)}
for d in sorted(r.iterdir()):
 if not d.is_dir() or not d.name.startswith(('pointer_','canary_','physics_','development_','main_')):continue
 info={'success':(d/'_SUCCESS').exists()}
 names=['FAILURE.txt','STAGE_FINAL.json','train/results.json','train/TRAIN_FINAL.json','data/PREPARATION_FINAL.json']
 if not info['success']:names+=['train.out','train.err','prepare.out','prepare.err','R.out','R.err','I.out','I.err','R.FAILURE.txt','I.FAILURE.txt']
 for name in names:
  f=d/name
  if f.exists():info[name]=f.read_text(errors='replace')[-1800:]
 for f in sorted(d.glob('*/planner/sample_metrics.json'))+sorted(d.glob('*/body/progress.json'))+sorted(d.glob('*/body/sample_metrics.json')):
  info[str(f.relative_to(d))]=json.loads(f.read_text())
 out[d.name]=info
print(json.dumps(out))
