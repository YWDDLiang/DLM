import json
from pathlib import Path
p=Path('/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion')
g=p/'workstreams/proposal_realization_candidates_20260826/grounding'
items={
'P0':p/'runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final/adapter_model.safetensors',
'B0':p/'runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final/adapter_model.safetensors',
'C3FD':g/'runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt',
'K4teacher':g/'data/spad_state_teacher_round0_consistent_20260906_v2/teacher.json',
'K8final':g/'runs/spad_state_path_train_39938/train/TRAIN_FINAL.json',
'pointer_data':g/'data/spad_species_pointer_v1_20260903/train.jsonl',
'MP20':p/'reference/crysllmgen/data/mp_20/train.csv'}
print(json.dumps({k:{'path':str(v),'exists':v.exists(),'bytes':v.stat().st_size if v.exists() else None} for k,v in items.items()}))
