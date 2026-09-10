"""Verify persisted full-model scores against the once-realized candidate inputs."""
import argparse
import json
from pathlib import Path
import sys

SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'src'))
from scripts.run_post_refine_cycle import read_rows,write_json,file_hash,validate_rsi_checkpoint
from crystal_dlm.utility_acceptance import materialized_continuous_decision
from crystal_dlm.post_refine_contract import fingerprint


def main(root):
    import torch
    output=root/'models/selected_utility';checkpoint=output/'checkpoint';audit_path=output/'parity_audit/EXPORT_AUDIT_FINAL.json'
    if (output/'EXPORT_FINAL.json').exists():raise ValueError('export is already verified')
    audit=json.loads(audit_path.read_text());frozen=json.loads((root/'FROZEN_SELECTION.json').read_text())
    receipt=validate_rsi_checkpoint(checkpoint,'E');selection_sha=file_hash(root/'FROZEN_SELECTION.json')
    if audit['selection_sha256']!=selection_sha or audit['checkpoint_receipt_sha256']!=file_hash(checkpoint/'RSI_TRAINING_DONE.json'):
        raise ValueError('full-model audit no longer binds this checkpoint and selection')
    if not audit['strict_reload_probe_passed'] or audit['decision_mismatches'] or audit['maximum_utility_drift']>1e-5:
        raise ValueError('full-model score or decision parity failed')
    delta=[d for row in audit['raw_field_mismatches'] for d in row['differences']]
    if any(not row['actual_edit'] for row in audit['raw_field_mismatches']):
        raise ValueError('a KEEP record changed during recomputation')
    if any(len(d['path'].split('/'))!=6 or not d['path'].startswith('/structure/sites/')
           or d['path'].split('/')[4]!='xyz' or d.get('absolute_difference',1)>1e-12 for d in delta):
        raise ValueError('recomputed candidate has a primary-coordinate or non-roundoff difference')
    # The actual evaluated structures are independently reconstructed and bound
    # before this selection. No endpoint quality enters this operation.
    binding_path=root/'materialized_primary/MATERIALIZATION_FINAL.json'
    binding=json.loads(binding_path.read_text())
    if not binding['exact_existing_evaluated_inputs'] or binding['requests']!=1000:
        raise ValueError('canonical candidate realization is not verified')
    for field,path in (('native_inputs_sha256',root/'fit/native/inputs.jsonl'),
                       ('current_inputs_sha256',root/'fit/current/inputs.jsonl'),
                       ('proposal_inputs_sha256',root/'fit/proposal/inputs.jsonl'),
                       ('existing_inputs_sha256',root/'fit/hybrid_proposal/inputs.jsonl')):
        if file_hash(path)!=binding[field]:raise ValueError('canonical proposal source changed:'+field)
    reg=json.loads((root/'PREREGISTRATION.json').read_text());old=Path(reg['old_editor'])
    validate_rsi_checkpoint(old,'E')
    for name,digest in receipt['frozen_files_identical'].items():
        if file_hash(checkpoint/name)!=digest or file_hash(old/name)!=digest:
            raise ValueError('nonquality checkpoint file changed:'+name)
    left=torch.load(old/'expert_edit_modules.pt',map_location='cpu',weights_only=True)
    right=torch.load(checkpoint/'expert_edit_modules.pt',map_location='cpu',weights_only=True)
    if any(not torch.equal(v,right[name][key]) for name,values in left.items() if name!='quality_head' for key,v in values.items()):
        raise ValueError('nonquality editor tensor changed')
    training=json.loads(Path(frozen['training_receipt_path']).read_text());chosen=frozen['selected'];epoch=str(chosen['epoch'])
    snapshot=training['snapshots'][epoch]
    if file_hash(snapshot['path'])!=snapshot['sha256']:raise ValueError('selected snapshot changed')
    selected=torch.load(snapshot['path'],map_location='cpu',weights_only=True)
    if any(not torch.equal(v,right['quality_head'][key]) for key,v in selected.items()):
        raise ValueError('exported quality head is not the selected trained head')
    policy=json.loads((checkpoint/'QUALITY_UTILITY_POLICY.json').read_text())
    scores={r['ordinal']:r for r in read_rows(output/'parity_audit/ACTUAL_SCORES.jsonl')}
    before=read_rows(root/'fit/current/inputs.jsonl');panel=Path(chosen['dev']['panel'])
    expected=read_rows(panel/'edited/inputs.jsonl');expected_decisions=json.loads((panel/'DECISION_BINDING.json').read_text())['decisions']
    verified=[]
    for i,record in enumerate(before):
        native=json.loads((root/f'fit/native/records/{i:04d}.json').read_text())
        trace=json.loads((root/f'fit/proposal/records/{i:04d}.json').read_text())['editor_trace']
        bound_path=root/f'materialized_primary/records/{i:04d}.json'
        if file_hash(bound_path)!=binding['bound_record_sha256'][i]:raise ValueError('materialized record changed')
        bound=json.loads(bound_path.read_text());row=scores.get(i,{})
        value,decision=materialized_continuous_decision(native,record.get('body_token_ids') or [],trace,bound,
            row.get('utility'),chosen['margin'],reference_logit=row.get('reference_logit'),
            reference_required=bool(policy.get('reference_acceptance_required')))
        value['trajectory_id']=expected[i]['trajectory_id']
        if json.dumps(value,sort_keys=True)!=json.dumps(expected[i],sort_keys=True):
            raise ValueError('materialized full-model output differs from evaluated output:'+str(i))
        if any(decision[k]!=expected_decisions[i][k] for k in ('learned_accept','actual_edit')):
            raise ValueError('materialized full-model decision changed:'+str(i))
        verified.append(dict(ordinal=i,actual_edit=decision['actual_edit'],output_sha256=fingerprint(value)))
    parity=dict(requests=1000,all_output_fields_exact_after_trajectory_identity_binding=True,
        all_decisions_exact=True,all_KEEP_originals_exact=True,records=verified,
        materialization_receipt_sha256=file_hash(binding_path),full_model_audit_sha256=file_hash(audit_path))
    write_json(output/'MATERIALIZED_PARITY_FINAL.json',parity)
    report=dict(checkpoint=str(checkpoint),checkpoint_receipt_sha256=file_hash(checkpoint/'RSI_TRAINING_DONE.json'),
        strict_reload_probe_passed=True,all_1000_continuous_outputs_reproduced=True,
        full_model_vs_cached_head_max_abs_utility_drift=audit['maximum_utility_drift'],
        actual_edits=sum(x['actual_edit'] for x in verified),reference_acceptance_required=bool(policy.get('reference_acceptance_required')),
        reference_head_max_abs_drift=audit['maximum_reference_drift'],selected_epoch=int(epoch),margin=chosen['margin'],
        model_files_copied_exactly=receipt['frozen_files_identical'],selection_sha256=selection_sha,
        materialization='construct_once_then_select_bound_record_without_recomputing_derived_xyz',
        materialized_parity_sha256=file_hash(output/'MATERIALIZED_PARITY_FINAL.json'),
        original_recomputed_xyz_difference_records=len(audit['raw_field_mismatches']),
        original_recomputed_xyz_max_abs_difference=max((d['absolute_difference'] for d in delta),default=0.),
        scientific_inputs_weights_thresholds_or_geometry_rules_changed=False,
        original_export_job='42038',verification_full_model_job=audit['slurm_job_id'],
        classification='exported_candidate; stability_across_registered_streams_pending')
    write_json(output/'EXPORT_FINAL.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
