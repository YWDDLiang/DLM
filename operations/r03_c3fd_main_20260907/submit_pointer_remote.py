import json,os,re,subprocess as s
from pathlib import Path
g=Path('/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/proposal_realization_candidates_20260826/grounding')
r=g/'experiments/r03_c3fd_rich_20260907/run_20260907T033526Z'
f=r/'POINTER_SUBMISSION.json'
if f.exists():
 print(f.read_text())
else:
 source=Path((r/'SOURCE_PATH').read_text().strip())
 queue=s.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%b'],text=True).splitlines()
 assert len(queue)<2,queue
 used=0
 for line in queue:
  gres=line.split('|',1)[1].strip()
  if gres not in ('N/A','(null)',''):
   match=re.fullmatch(r'gpu(?::[^:]+)?:(\d+)',gres)
   assert match,gres
   used+=int(match.group(1))
 assert used+1<=4
 env=dict(os.environ,R03_SOURCE_ROOT=str(source),R03_RUN_ROOT=str(r))
 result=s.run(['sbatch','--parsable','--job-name=r03ctrl033526','--output='+str(r/'logs/pointer_%j.out'),'--error='+str(r/'logs/pointer_%j.err'),str(source/'operations/r03_c3fd_main_20260907/train_pointer.sbatch')],env=env,capture_output=True,text=True,check=True)
 job=result.stdout.strip().split(';')[0];assert job.isdigit(),result.stdout
 record={'job_id':job,'source':str(source),'run_root':str(r),'gpus':1,'cpus':4,'queue_before':queue,'stage':'assets_and_frozen_P0_control_head'}
 f.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
