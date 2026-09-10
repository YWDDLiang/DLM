"""Reconstruct original mixed decoding batches without consulting physics labels."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_post_refine_cycle import read_rows,write_json,file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from crystal_dlm.expert_edit import load_editor_model,inference_view,materialize_edit_batch
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.use_deterministic_algorithms(True)
    device=torch.device('cuda',0);reg=json.loads((root/'PREREGISTRATION.json').read_text())
    spec=json.loads((root/'fit/RUN_SPEC.json').read_text());fit=root/'fit'
    model,tokenizer=load_editor_model(spec['assets']['base_model'],reg['old_editor'],device)
    plans=read_rows(fit/'cohort/plans.jsonl');current=read_rows(fit/'current/inputs.jsonl')
    traces={i:json.loads((fit/f'proposal/records/{i:04d}.json').read_text())['editor_trace'] for i in range(len(plans))}
    roles={r['ordinal']:r['split'] for r in read_rows(root/'SOURCE_SPLIT.jsonl')}
    rows=read_rows(root/'evaluation_features/EVAL_ROWS.jsonl');index={r['ordinal']:i for i,r in enumerate(rows)}
    predictions=read_rows(root/'evaluation_features/UTILITY_PREDICTIONS.jsonl')
    chosen=sorted([r for r in predictions if roles[r['ordinal']]!='final' and r['old_probability'] is not None],
        key=lambda r:-abs(r['old_probability']-1/(1+math.exp(-r['original_quality_logit']))))[:5]
    groups={}
    for shard in range(2):
        valid=[i for i in range(shard,len(plans),2) if current[i]['success'] and current[i].get('body_token_ids')]
        for start in range(0,len(valid),64):
            group=valid[start:start+64]
            for i in group:groups[i]=group
    def forward(views):
        batch=materialize_edit_batch(views,tokenizer,device)
        with torch.no_grad():out=model(batch['input_ids'],attention_mask=batch['attention_mask'],edit_context=batch['edit_context'])
        result=out.quality_logits[:,3].float().cpu();shape=list(batch['input_ids'].shape)
        return result,shape
    result=[]
    for target in chosen:
        i=target['ordinal'];step=traces[i]['forward_calls'];active=[j for j in groups[i] if traces[j]['forward_calls']>=step]
        views=[]
        for j in active:
            trace=traces[j];before=current[j]['body_token_ids'];order=trace['action']['positions']
            canvas=list(trace['initial_masked_tokens']);offset=min(step-2,len(trace['sampling_trace']))
            for item in trace['sampling_trace'][:offset]:canvas[item['position']]=item['token_id']
            views.append(inference_view(tokenizer(plans[j]['body_prompt'],add_special_tokens=False)['input_ids'],
                before,canvas,plans[j]['plan_state']['N'],1,order,remaining=80,reveal=offset/len(order)))
        mixed,old_shape=forward(views);mixed_probability=float(mixed[active.index(i)].sigmoid())
        start=index[i]//16*16;canonical_rows=rows[start:start+16]
        canonical_views=[inference_view(tokenizer(r['prompt'],add_special_tokens=False)['input_ids'],r['current_tokens'],
            r['proposal_tokens'],r['num_sites'],1,r['action_positions'],remaining=80,reveal=1.) for r in canonical_rows]
        canonical,new_shape=forward(canonical_views);raw=float(canonical[index[i]-start])
        single,single_shape=forward([canonical_views[index[i]-start]])
        result.append(dict(ordinal=i,split=roles[i],recorded_probability=target['old_probability'],
            reconstructed_mixed_probability=mixed_probability,original_batch_shape=old_shape,canonical_batch_shape=new_shape,
            cached_canonical_logit=target['original_quality_logit'],fresh_canonical_logit=raw,single_logit=float(single[0]),
            original_batch_reproduced_exactly=mixed_probability==target['old_probability'],
            canonical_batch_reproduced_exactly=raw==target['original_quality_logit'],original_forward_step=step))
    report=dict(schema='editor_judgement_batch_reconstruction_v1',cases=result,no_endpoint_labels_used=True,
        model_checkpoint_receipt_sha256=file_hash(Path(reg['old_editor'])/'RSI_TRAINING_DONE.json'),
        original_generation_partition=dict(shards=2,batch_size=64),canonical_judgement_partition=dict(batch_size=16,order='original_ordinal'),
        original_contexts_reproduced=all(r['original_batch_reproduced_exactly'] for r in result),
        canonical_contexts_reproduced=all(r['canonical_batch_reproduced_exactly'] for r in result))
    write_json(root/'analysis/JUDGEMENT_BATCH_AUDIT.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
