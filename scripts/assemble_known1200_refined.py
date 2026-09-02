#!/usr/bin/env python3
"""Assemble accepted-CIF model494 outputs into main/remainder evaluator sets."""

from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch
from assemble_grounding_repeat import refined_structures


def read_jsonl(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def canon(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--accounting-jsonl',type=Path,required=True); p.add_argument('--refine-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--steps',type=int,required=True)
    a=p.parse_args(); out=a.output_dir.resolve()
    if out.exists(): raise FileExistsError(out)
    files=[x for x in a.refine_dir.glob('dlm_refined_mp_*.pt') if '.rank' not in x.name]
    if len(files)!=1: raise ValueError(files)
    structures=refined_structures(torch.load(files[0],map_location='cpu')); acc=read_jsonl(a.accounting_jsonl)
    accepted=[x for x in acc if x.get('parsed') is True]
    if len(accepted)!=1139 or set(structures)!=set(range(1139)): raise ValueError('refined accepted accounting changed')
    out.mkdir(parents=True); (out/'all_requested_accounting.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in acc))
    all_rows=[]
    partitions=(('main1000',accepted[:1000],1000),('remainder139',accepted[1000:],139))
    for split,selected,expected in partitions:
        if len(selected)!=expected: raise ValueError((split,len(selected)))
        folder=out/split; folder.mkdir(); rows=[]
        for ordinal,item in enumerate(selected):
            idx=int(item['accepted_index']); plan=item['plan_state']; source=int(item['source_sample_idx'])
            rows.append({'schema':'wqcodiff_generation_attempt_v1','attempt_id':f'plan1200-tau{a.steps}-{split}-{ordinal:04d}','method':f'G2-model494-tau{a.steps}','ordinal':ordinal,'sample_idx':ordinal,'requested_ordinal':int(item['split_requested_index']),'global_requested_index':int(item['requested_index']),'source_sample_idx':source,'pair_id':f'plan1200:{item["requested_index"]}','arm':'candidate','repeat':0,'status':'succeeded','reason':None,'structure':structures[idx],'plan_state':plan,'source_plan_state_sha256':canon(plan),'diffusion_refinement_applied':True,'diffusion_refinement_steps':a.steps,'refiner_noise_seed':101117+source,'retry_or_replacement_used':False})
        (folder/'generation.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows)); (folder/'_SUCCESS').touch(); all_rows.extend(rows)
    (out/'manifest.json').write_text(json.dumps({'schema':'known1200_refined_accepted_v2','steps':a.steps,'requested':1159,'default_denominator':1139,'main_denominator':1000,'remainder_denominator':139,'main_topup_from_later_valid':17,'invalid_cif_excluded':20,'partition_rule':'first1000 valid CIFs in original Plan order; remaining139 valid CIFs'},indent=2,sort_keys=True)+'\n'); (out/'_SUCCESS').touch(); print(out)


if __name__=='__main__': main()
