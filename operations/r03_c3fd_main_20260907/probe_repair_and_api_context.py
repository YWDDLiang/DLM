import json,os,subprocess as s
from pathlib import Path
p=Path('/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion')
r=p/'workstreams/proposal_realization_candidates_20260826/grounding/experiments/r03_c3fd_rich_20260907/run_20260907T033526Z'
d=r/'canary_repair_40403'
out={'queue':s.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%T|%C|%M'],text=True),'credential_env_present':{k:bool(os.environ.get(k)) for k in ['MP_API_KEY','PMG_MAPI_KEY','MAPI_KEY','H1_MP_API_KEY','R03_MP_API_KEY']},'repair':{}}
names=['LANE_canary_repair/FAILURE.json','LANE_canary_repair/STAGE_FINAL.json','shared_I/control.stage.json']
for a in ['G','P']:
 names += [a+'.out',a+'.err',a+'/FAILURE.json',a+'/body.err',a+'/body/progress.json',a+'/body/sample_metrics.json',a+'/native_direct/report.json',a+'/native_labels/LABEL_FINAL.json',a+'/tau800_labels/LABEL_FINAL.json']
for name in names:
 f=d/name
 if f.is_file():out['repair'][name]=f.read_text(errors='replace')[-2200:]
print(json.dumps(out))
