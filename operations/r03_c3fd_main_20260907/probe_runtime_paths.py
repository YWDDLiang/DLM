import json
from pathlib import Path
p=Path('/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion')
g=p/'workstreams/proposal_realization_candidates_20260826/grounding'
c=json.loads((g/'runs/spad_state_path_train_39938/train/training_config.json').read_text())
t=Path(c['teacher_json'])
v=json.loads((g/'data/c3fd_semantic_v21_step1b_20260828/vocabulary.json').read_text())
old=p/'workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1'
print(json.dumps({'K8teacher':str(t),'K8teacher_exists':t.exists(),'vocabulary_keys':list(v),'family_values':v.get('soft_vocabulary',{}).get('anion_framework'),'first_species':v.get('species',[None])[0],'legacy_runtime':str(old),'legacy_exists':(old/'run_schedule256.py').exists(),'object_repo_exists':Path('/public/home/jiaosz/ywliang/ai4s/.sscd_periodic_self_repair_20260906_v1/.git').exists()}))
