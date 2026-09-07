import json,subprocess as s
from pathlib import Path
repo=Path('/public/home/jiaosz/ywliang/ai4s/.sscd_periodic_self_repair_20260906_v1')
g=Path('/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/proposal_realization_candidates_20260826/grounding')
r=g/'experiments/r03_c3fd_rich_20260907/run_20260907T033526Z'
commit='5b5193c315f1c8045e92f1f902f60b11e6fe89f3'
s.run(['git','-c','http.lowSpeedLimit=1','-c','http.lowSpeedTime=20','fetch','https://github.com/YWDDLiang/DLM.git','codex/r03-c3fd-rich-main'],cwd=repo,check=True,timeout=90)
assert s.check_output(['git','rev-parse',commit],cwd=repo,text=True).strip()==commit
d=r/'code'/commit
if not (d/'_CODE_READY').exists():
 d.mkdir(parents=True,exist_ok=False)
 a=s.Popen(['git','archive',commit],cwd=repo,stdout=s.PIPE)
 b=s.run(['tar','-xf','-','-C',str(d)],stdin=a.stdout)
 a.stdout.close();assert a.wait()==0 and b.returncode==0
 (d/'_CODE_READY').write_text(commit+'\n')
(r/'logs').mkdir(exist_ok=True)
(r/'SOURCE_PATH').write_text(str(d)+'\n')
print(json.dumps({'run_root':str(r),'source':str(d),'commit':commit,'ready':True}))
