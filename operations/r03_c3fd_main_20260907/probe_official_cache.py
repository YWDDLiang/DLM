import hashlib,json
from pathlib import Path
p=Path('/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/proposal_realization_candidates_20260826/grounding/runs/spad_basin_closure_official_20260904_v1/official_mp_cache')
out={'directory':str(p),'success':(p/'completion_SUCCESS').is_file(),'files':{}}
for name in ['completion_manifest.json','official_slim_cache.jsonl','unresolved_chemsys.jsonl','query_audit.jsonl']:
 f=p/name
 out['files'][name]={'exists':f.is_file()}
 if f.is_file():
  b=f.read_bytes();out['files'][name].update(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
  if name=='completion_manifest.json':out['manifest']=json.loads(b)
  else:out['files'][name]['lines']=sum(bool(x.strip()) for x in b.splitlines())
print(json.dumps(out))
