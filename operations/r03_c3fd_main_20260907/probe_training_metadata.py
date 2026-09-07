import json
from pathlib import Path
g=Path('/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/proposal_realization_candidates_20260826/grounding')
k4=json.loads((g/'data/spad_state_teacher_round0_consistent_20260906_v2/teacher.json').read_text())
k8=json.loads((g/'runs/spad_state_path_train_39938/train/TRAIN_FINAL.json').read_text())
p=g/'data/spad_basin_closure_sft_v1_20260904/train.jsonl'
row=json.loads(p.open().readline()) if p.exists() else {}
print(json.dumps({'K4_provenance':k4['provenance'],'K8_final':k8,'full_teacher_path':str(p),'full_teacher_exists':p.exists(),'first_teacher_keys':list(row),'first_plan_keys':list((row.get('plan_state') or {}))}))
