import hashlib, json, subprocess as s, sys, zlib
from pathlib import Path
r=Path('/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/proposal_realization_candidates_20260826/grounding/experiments/r03_c3fd_rich_20260907/run_20260907T033526Z')
src=r/'code/82050dd8a3f19297d4ae293399b02035c131933a'
directory=r/'canary_geometry_40457'
packed=directory/'TRANSFER_BUNDLE.zlib'
metadata=directory/'TRANSFER_BUNDLE_META.json'
if not packed.exists():
 plain=s.check_output([sys.executable,str(src/'operations/r03_c3fd_main_20260907/export_geometry_canary_records.py'),'--run-root',str(r)])
 data=zlib.compress(plain,9)
 with packed.open('xb') as f:f.write(data)
 metadata.write_text(json.dumps({'path':str(packed),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'plain_sha256':hashlib.sha256(plain).hexdigest(),'plain_bytes':len(plain)})+'\n')
meta=json.loads(metadata.read_text())
assert len(packed.read_bytes())==meta['bytes'] and hashlib.sha256(packed.read_bytes()).hexdigest()==meta['sha256']
print(json.dumps(meta))
