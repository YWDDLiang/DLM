"""Prepare TRAIN-only SUN-first comparisons from actual two-pool outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

SOURCE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(SOURCE/'src'))
from scripts.run_post_refine_cycle import read_rows,write_rows,write_json,file_hash
from scripts.run_rsi_stages import scores,score_directory
from crystal_dlm.post_refine_contract import fingerprint
from crystal_dlm.ranked_feedback import endpoint_quality


def value(row):
    q=endpoint_quality(row)
    if q['known_failure'] or (row.get('terminal_status')=='not_converged' and q['hull'] is not None):return (0,0)
    if not q['reliable']:return None
    if q['hull']>.1:return (0,0)
    if row.get('novel') is None:return None
    return (int(q['hull']<=0 and row['novel']),int(q['hull']<=.1 and row['novel']))


def main(root):
    roles=read_rows(root/'SOURCE_SPLIT.jsonl');current=read_rows(root/'fit/current/inputs.jsonl')
    model=json.loads((root/'operational/MODEL_DEFINITION.json').read_text())
    panels=[root/'operational/panels'/name for name in ('pool_A_K2','pool_B_K2')]
    observed=[scores(p,'edited') for p in panels]
    decisions=[json.loads((p/'DECISION_BINDING.json').read_text())['decisions'] for p in panels]
    records=[read_rows(p/'edited/inputs.jsonl') for p in panels];rows=[];excluded=Counter()
    for i,role in enumerate(roles):
        if role['split']!='train':continue
        a,b=(value(x[i]) for x in observed)
        if a is None or b is None:excluded['unknown_outcome']+=1;continue
        if a==b:excluded['tie']+=1;continue
        if not current[i].get('body_token_ids'):excluded['no_token_view']+=1;continue
        chosen,rejected=(0,1) if a>b else (1,0)
        def view(side):
            decision=decisions[side][i]
            if not decision['actual_edit']:return dict(tokens=current[i]['body_token_ids'],action_positions=[],kind='KEEP')
            origin=Path(model['sources'][decision['chosen_stream']]['proposals'])
            trace=json.loads((origin/f'proposal/records/{i:04d}.json').read_text())['editor_trace']
            return dict(tokens=trace['proposal_tokens'],action_positions=trace['action']['positions'],kind='EDIT')
        left,right=view(chosen),view(rejected)
        rows.append(dict(pair_id=fingerprint(dict(source=role['ancestor_id'],A=records[0][i],B=records[1][i])),
            source_id=role['ancestor_id'],ordinal=i,focus_split='train',current_tokens=current[i]['body_token_ids'],
            chosen=left,rejected=right,chosen_outcome=[a,b][chosen],rejected_outcome=[a,b][rejected],
            preference_basis='novel_Stable_priority' if a[0]!=b[0] else 'novel_MS_tiebreak',
            chosen_continuous_record_sha256=fingerprint(records[chosen][i]),
            rejected_continuous_record_sha256=fingerprint(records[rejected][i]),historical_U_reused=False))
    path=root/'data/SUN_FIRST_PAIRWISE_TRAIN.jsonl';write_rows(path,rows)
    report=dict(rows=len(rows),sources=len({r['source_id'] for r in rows}),basis_counts=dict(Counter(r['preference_basis'] for r in rows)),
        exclusions=dict(excluded),no_DEV_FINAL_supervision=True,training_performed=False,
        data_sha256=file_hash(path),source_split_sha256=file_hash(root/'SOURCE_SPLIT.jsonl'),
        physical_score_bindings={str(score_directory(p,'edited')/'attempt_results.jsonl'):file_hash(score_directory(p,'edited')/'attempt_results.jsonl') for p in panels},
        purpose='prepared_next_stage_supervision; no new ranking checkpoint is claimed')
    write_json(root/'data/SUN_FIRST_PAIRWISE_DATA_FINAL.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
