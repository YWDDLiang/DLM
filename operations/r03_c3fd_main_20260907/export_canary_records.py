import base64,hashlib,json,zlib
from pathlib import Path
r=Path('/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/proposal_realization_candidates_20260826/grounding/experiments/r03_c3fd_rich_20260907/run_20260907T033526Z')
files={'I_body.jsonl':r/'canary_40400/I/body/raw_generations.jsonl','R_body.jsonl':r/'canary_40400/R/body/raw_generations.jsonl','G_body.jsonl':r/'canary_repair_40403/G/body/raw_generations.jsonl','P_body.jsonl':r/'canary_repair_40403/P/body/raw_generations.jsonl','PHYSICS_FINAL.json':r/'physics_40399/train/TRAIN_FINAL.json','PHYSICS_PREPARATION.json':r/'physics_40399/data/PREPARATION_FINAL.json','POINTER_RESULTS.json':r/'pointer_40395/train/results.json'}
out={}
for name,path in files.items():
 data=path.read_bytes();out[name]={'source':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'zlib_base64':base64.b64encode(zlib.compress(data)).decode()}
print(json.dumps(out))
