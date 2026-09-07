import base64,hashlib,json
from pathlib import Path
p=Path('/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/proposal_realization_candidates_20260826/grounding/experiments/r03_c3fd_rich_20260907/run_20260907T033526Z/pointer_40395/assets')
files={}
for name in ['C3FD_DOMAIN.json','P0_NATIVE_PROMPT.txt','PHYSICS_SOURCES.json','ASSET_IDENTITIES.json','PREPARATION_FINAL.json']:
 b=(p/name).read_bytes();files[name]={'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
print(json.dumps({'artifact_bundle':'r03_existing_asset_metadata','files':files}))
