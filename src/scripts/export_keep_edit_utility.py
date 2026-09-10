"""Export the selected existing head into a reload-verified complete E checkpoint."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_post_refine_cycle import read_rows,write_json,file_hash,validate_rsi_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--completion-dir',type=Path,required=True);args=parser.parse_args();root=args.root;output=args.completion_dir
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from crystal_dlm.expert_edit import load_editor_model,materialize_edit_batch
    from crystal_dlm.utility_acceptance import canonical_judgements,continuous_decision
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
    torch.cuda.set_device(0);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);device=torch.device('cuda',0)
    reg=json.loads((root/'PREREGISTRATION.json').read_text());training=json.loads((root/'training/utility/result/TRAINING_FINAL.json').read_text())
    selection=root/'FROZEN_SELECTION.json'
    if not selection.exists():raise ValueError('export requires a DEV-admitted frozen policy')
    frozen=json.loads(selection.read_text());chosen=frozen['selected'];epoch=str(chosen['epoch']);margin=chosen['margin']
    consensus=frozen.get('method')=='consensus'
    snapshot=training['snapshots'][epoch]
    if file_hash(snapshot['path'])!=snapshot['sha256']:raise ValueError('selected head snapshot changed')
    old=Path(reg['old_editor']);old_receipt=validate_rsi_checkpoint(old,'E')
    spec=json.loads((root/'fit/RUN_SPEC.json').read_text())
    model,tokenizer=load_editor_model(spec['assets']['base_model'],old,device)
    reference_head=copy.deepcopy(model.quality_head) if consensus else None
    model.quality_head.load_state_dict(torch.load(snapshot['path'],map_location=device,weights_only=True))
    checkpoint=output/'checkpoint'
    if checkpoint.exists():raise ValueError('selected checkpoint export already exists')
    checkpoint.mkdir(parents=True)
    excluded={'expert_edit_modules.pt','roundtrip_probe.pt','RSI_TRAINING_DONE.json'}
    frozen_files={}
    for path in old.iterdir():
        if path.is_file() and path.name not in excluded:
            shutil.copy2(path,checkpoint/path.name)
            digest=file_hash(path)
            if file_hash(checkpoint/path.name)!=digest:raise ValueError('frozen checkpoint copy changed')
            frozen_files[path.name]=digest
    modules=torch.load(old/'expert_edit_modules.pt',map_location='cpu',weights_only=True)
    before=copy.deepcopy(modules)
    modules['quality_head']={k:v.detach().cpu() for k,v in model.quality_head.state_dict().items()}
    if any(not torch.equal(value,modules[name][key]) for name,values in before.items() if name!='quality_head'
           for key,value in values.items()):raise ValueError('export changed a nonquality module')
    torch.save(modules,checkpoint/'expert_edit_modules.pt')
    if consensus:
        torch.save({k:v.detach().cpu() for k,v in reference_head.state_dict().items()},checkpoint/'REFERENCE_QUALITY_HEAD.pt')
    probe=torch.load(old/'roundtrip_probe.pt',map_location='cpu',weights_only=False)
    batch=materialize_edit_batch(probe['examples'],tokenizer,device)
    with torch.no_grad():actual=model(batch['input_ids'],attention_mask=batch['attention_mask'],edit_context=batch['edit_context'])
    names=('logits','mode_logits','site_logits','count_logits','quality_logits')
    for name in names[:-1]:
        if not torch.equal(getattr(actual,name).cpu(),probe[name]):raise ValueError('selected head changed frozen probe output:'+name)
    torch.save(dict(examples=probe['examples'],**{name:getattr(actual,name).cpu() for name in names}),checkpoint/'roundtrip_probe.pt')
    policy=dict(schema='quality_utility_continuous_policy_v1',raw_utility_margin=margin,
        judgement_batch_size=16,judgement_order='original_ordinal_all_valid_token_views',remaining=80,reveal=1.,
        score_is_calibrated_probability=False,continuous_KEEP=True,continuous_EDIT='changed_numeric_fields_only',
        original_SUN_guard=True,selection_sha256=file_hash(selection),training_snapshot_sha256=snapshot['sha256'],epoch=int(epoch))
    policy['reference_acceptance_required']=consensus
    if consensus:policy.update(reference_raw_margin=0.,reference_head_sha256=file_hash(checkpoint/'REFERENCE_QUALITY_HEAD.pt'))
    write_json(checkpoint/'QUALITY_UTILITY_POLICY.json',policy)
    files={p.name:file_hash(p) for p in checkpoint.iterdir() if p.is_file()}
    receipt=dict(optimizer_steps=snapshot['optimizer_steps'],parameter_delta_squared=snapshot['parameter_delta_squared'],
        checkpoint_files=files,contract=dict(branch='E',objective='signed_novel_Stable_plus_novel_MS_utility',
            source_checkpoint=str(old),source_checkpoint_receipt_sha256=file_hash(old/'RSI_TRAINING_DONE.json'),
            training_data_sha256=training['data_sha256'],source_split_sha256=reg['source_split_sha256']),
        training_receipt_sha256=file_hash(root/'training/utility/result/TRAINING_FINAL.json'),complete_passes=int(epoch),
        frozen_files_identical=frozen_files,nonquality_extra_modules_identical=True,frozen_probe_outputs_identical=True,
        acceptance_policy=policy)
    write_json(checkpoint/'RSI_TRAINING_DONE.json',receipt)
    del model,actual,batch;torch.cuda.empty_cache()
    reloaded,tokenizer=load_editor_model(spec['assets']['base_model'],checkpoint,device)
    validate_rsi_checkpoint(checkpoint,'E')
    rows=read_rows(root/'evaluation_features/EVAL_ROWS.jsonl')
    if consensus:
        reference_head.load_state_dict(torch.load(checkpoint/'REFERENCE_QUALITY_HEAD.pt',map_location=device,weights_only=True))
    actual_scores,reference_scores=canonical_judgements(reloaded,tokenizer,rows,reference_head=reference_head)
    stored={r['ordinal']:r for r in read_rows(root/'evaluation_features/UTILITY_PREDICTIONS.jsonl')}
    drifts={i:abs(value-stored[i]['learned_utilities'][epoch]) for i,value in actual_scores.items()}
    max_drift=max(drifts.values(),default=0.)
    if max_drift>1e-5:raise ValueError('exported full-model utility differs from cached trained-head evaluation')
    reference_drift=max((abs(value-stored[i]['original_quality_logit']) for i,value in reference_scores.items()),default=0.)
    if reference_drift>1e-5:raise ValueError('exported reference head differs from the registered canonical old head')
    native=root/'fit/native/records';current=read_rows(root/'fit/current/inputs.jsonl')
    reference=read_rows(Path(chosen['dev']['panel'])/'edited/inputs.jsonl')
    inverse={int(v):k for k,v in tokenizer.get_vocab().items()};mismatches=[];decisions=[]
    for i,record in enumerate(current):
        wrapper=json.loads((native/f'{i:04d}.json').read_text())
        trace=json.loads((root/f'fit/proposal/records/{i:04d}.json').read_text())['editor_trace']
        selected,decision=continuous_decision(wrapper,record.get('body_token_ids',[]),trace,inverse,actual_scores.get(i),margin,
            reference_logit=reference_scores.get(i),reference_required=consensus)
        expected=reference[i]
        fields=('structure','body','body_token_ids','success','reason')
        if (json.dumps({k:selected.get(k) for k in fields},sort_keys=True) !=
                json.dumps({k:expected.get(k) for k in fields},sort_keys=True)):mismatches.append(i)
        decisions.append(dict(ordinal=i,**decision))
    if mismatches:raise ValueError('exported full-model continuous decisions differ:'+str(mismatches))
    report=dict(checkpoint=str(checkpoint),checkpoint_receipt_sha256=file_hash(checkpoint/'RSI_TRAINING_DONE.json'),
        strict_reload_probe_passed=True,all_1000_continuous_outputs_reproduced=True,
        full_model_vs_cached_head_max_abs_utility_drift=max_drift,actual_edits=sum(d['actual_edit'] for d in decisions),
        reference_acceptance_required=consensus,reference_head_max_abs_drift=reference_drift,
        selected_epoch=int(epoch),margin=margin,model_files_copied_exactly=frozen_files,
        classification='exported_candidate; adoption_requires_final_comparison',selection_sha256=file_hash(selection))
    write_json(output/'EXPORT_FINAL.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
