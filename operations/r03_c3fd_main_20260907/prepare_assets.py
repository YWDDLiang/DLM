"""Bind existing R03/C3FD/physics assets; no crystal generation or physics call."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args();out=args.output_dir
    out.mkdir(parents=True,exist_ok=False)
    p=Path('/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion')
    g=p/'workstreams/proposal_realization_candidates_20260826/grounding'
    p0=p/'runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final'
    b0=p/'runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final'
    c3fd=g/'runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt'
    vocab_path=g/'data/c3fd_semantic_v21_step1b_20260828/vocabulary.json'
    ref=Path('/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt')
    frozen=p/'workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1'
    checks={
        'P0':(p0/'adapter_model.safetensors','65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a'),
        'B0':(b0/'adapter_model.safetensors','5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d'),
        'B0_tokenizer':(b0/'tokenizer.json','3a21588abca8e56155cc7b6cabb81df51992ccd2e89704aec770912f24e75509'),
        'C3FD':(c3fd,'87c1673f709c14488848196e3f466b0f4797cc3511108a07b6d498ca96d17d53'),
        'C3FD_vocabulary':(vocab_path,'3268dccfb8562a2936f9a81d928f547b4f75c2471d4b383c2e36bebc7b41a6e3'),
        'model494':(ref,'573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e')}
    receipts={}
    for name,(path,expected) in checks.items():
        value=digest(path)
        receipts[name]={'path':str(path),'bytes':path.stat().st_size,'sha256':value,'matches':value==expected}
        print(json.dumps({'event':'asset_identity','asset':name,**receipts[name]}),flush=True)
        if value!=expected: raise ValueError(f'protected asset identity changed: {name}')
    (out/'ASSET_IDENTITIES.json').write_text(json.dumps(receipts,indent=2)+'\n')
    if not (frozen/'run_schedule256.py').is_file(): raise FileNotFoundError('frozen R03 runtime missing')
    import torch
    from crystal_dlm.c3fd_calibration import StratumInteraction
    from crystal_dlm.family_reachability import FAMILY_FORBIDDEN,PaulingWitnessReachability
    from crystal_dlm.fixed_slot import Z_TO_SYMBOL
    vocabulary=json.loads(vocab_path.read_text())
    checkpoint=torch.load(c3fd,map_location='cpu',weights_only=False)
    interaction=StratumInteraction.from_dict(checkpoint['stratum_interaction'])
    family_values=vocabulary['soft_vocabulary']['anion_framework']
    nodes={}
    for item in vocabulary['species']:
        nodes.setdefault(Z_TO_SYMBOL[int(item['atomic_number'])],set()).add(int(item['oxidation_state']))
    atomic=sorted({int(item['atomic_number']) for item in vocabulary['species']})
    eneg,metals=PaulingWitnessReachability._load_smact_properties(atomic)
    strata=[];excluded=[]
    for family,n,k in interaction.strata:
        name=family_values[int(family)]
        if name in FAMILY_FORBIDDEN and 1<=int(k)<=min(7,int(n)) and 1<=int(n)<=20:
            strata.append([int(n),int(k),name])
        else: excluded.append([int(n),int(k),name])
    domain={'schema':'r03_c3fd_domain_v1','nodes':{k:sorted(v) for k,v in nodes.items()},
            'electronegativities':{Z_TO_SYMBOL[z]:eneg.get(z) for z in atomic},
            'metal_symbols':[Z_TO_SYMBOL[z] for z in sorted(metals)],'max_atoms':20,'max_species':7,
            'allowed_strata':strata,'excluded_nonchemical_strata':excluded,
            'source_assets':{k:receipts[k] for k in ('C3FD','C3FD_vocabulary')},
            'secondary_neural_composer_used':False,'external_N_family_arity_sampled':False}
    (out/'C3FD_DOMAIN.json').write_text(json.dumps(domain,indent=2)+'\n')
    # Tokenizer-only subprocess uses the original recovered H1A2 namespace.
    prompt_code="""from transformers import AutoTokenizer
from crystal_dlm.h1_llm_planner import format_planner_prompt
import pathlib,sys
t=AutoTokenizer.from_pretrained(sys.argv[1],trust_remote_code=True)
p=format_planner_prompt(t,sample_idx=None,prompt_style='h1_rich_plan_v1')
pathlib.Path(sys.argv[2]).write_text(p,encoding='utf-8')
"""
    env=dict(os.environ,PYTHONPATH=str(frozen/'runtime'))
    subprocess.run([sys.executable,'-c',prompt_code,str(p0),str(out/'P0_NATIVE_PROMPT.txt')],
                   cwd=frozen/'runtime',env=env,check=True)
    sources={'schema':'r03_physics_transfer_sources_v1'}
    k8config=json.loads((g/'runs/spad_state_path_train_39938/train/training_config.json').read_text())
    teachers={'k8':Path(k8config['teacher_json']),
              'k4':g/'data/spad_state_teacher_round0_consistent_20260906_v2/teacher.json'}
    for label,path in teachers.items():
        teacher=json.loads(path.read_text());prov=teacher['provenance']
        source={'teacher':{'path':str(path),'sha256':digest(path)},
                'paths':[{'path':v,'sha256':digest(v)} for v in prov['paths_jsonl']],
                'labels':[{'path':v,'sha256':digest(v)} for v in prov['labels_jsonl']],
                'source_tokenizer_path':prov['checkpoint'],
                'tokenizer_json_sha256':digest(Path(prov['checkpoint'])/'tokenizer.json')}
        if label=='k4': source['data_revision']='consistent_v2'
        sources[label]=source
    ce=g/'data/spad_basin_closure_sft_v1_20260904/train.jsonl'
    sources['mp20']={'path':str(ce),'sha256':digest(ce),'source_split':'train'}
    (out/'PHYSICS_SOURCES.json').write_text(json.dumps(sources,indent=2)+'\n')
    result={'status':'complete','domain_elements':len(nodes),'allowed_strata':len(strata),
            'excluded_strata':excluded,'prompt_sha256':digest(out/'P0_NATIVE_PROMPT.txt'),
            'domain_sha256':digest(out/'C3FD_DOMAIN.json'),'physics_sources_sha256':digest(out/'PHYSICS_SOURCES.json'),
            'new_training_structures':0,'new_physical_labels':0,'model_forward_calls':0}
    (out/'PREPARATION_FINAL.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'_SUCCESS').touch();print(json.dumps(result),flush=True)


if __name__=='__main__': main()
