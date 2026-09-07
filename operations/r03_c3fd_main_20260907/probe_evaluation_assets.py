import hashlib,json
from pathlib import Path
p=Path('/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion')
c=p/'workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3/CONFIG.json'
d=p/'runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream'
cfg=json.loads(c.read_text())
items={'frozen_config':c,'direct_eval_utils':d/'eval_utils.py','direct_compute_metrics':d/'compute_metrics.py','eval_sun':Path(cfg['assets']['eval_sun_py'])}
out={'files':{k:{'path':str(v),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in items.items()},'train_csv':{'path':cfg['assets']['train_csv'],'exists':Path(cfg['assets']['train_csv']).is_file()},'frozen_code':cfg['frozen_code']}
assert out['files']['direct_eval_utils']['sha256']=='68e6d0a9703f412cfd3215e6d0ae687e5b153e941d16f3fa4f2fffeedb505cb6'
assert out['files']['eval_sun']['sha256']==cfg['frozen_code']['eval_sun_sha256']
print(json.dumps(out))
