import json,os,subprocess as s
from pathlib import Path
g=Path('/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/proposal_realization_candidates_20260826/grounding')
r=g/'experiments/r03_c3fd_rich_20260907/run_20260907T033526Z'
out={'queue':s.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%T|%b|%C|%M'],text=True)}
for d in sorted(r.glob('pointer_*')):
 info={}
 for name in ['assets.out','assets.err','train.out','train.err','FAILURE.txt','assets/PREPARATION_FINAL.json','train/results.json']:
  f=d/name
  if f.exists():info[name]=f.read_text(errors='replace')[-3500:]
 out[d.name]=info
print(json.dumps(out))
