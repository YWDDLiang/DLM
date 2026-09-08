"""Reassess original positive teacher contracts with deterministic OLD/target R."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

def main(directory):
    root = Path(directory)
    names = ['TEACHER64_COMPACT.json','GAUGE64_OLD_COMPACT.json']
    teacher, old = [json.loads((root/n).read_text()) for n in names]
    output = {'schema':'expert_teacher64_reassessment_v1',
        'source_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},
        'selection':'same actual 32 G and 32 S positive teacher rows per gauge64 train/dev panel',
        'original_training_labels_unchanged':True,'deterministic_R_reassessment':True,'splits':{}}
    for split in ['train','dev']:
        initial = {x['group_id']:x for x in old[split]['rows']}
        target = {x['group_id']:x for x in teacher['labels'][split]['rows']}
        records = teacher['original_annotations'][split]
        assert len(records)==len(initial)==len(target)==64
        assert set(initial)==set(target)=={x['ancestor_id'] for x in records}
        assert all(x['versions']['deterministic_algorithms_enabled'] is True for x in list(initial.values())+list(target.values()))
        result = {}
        for task in ['G','S']:
            rows = [x for x in records if x['task']==task]
            details = []
            for row in rows:
                gid = row['ancestor_id']
                before, after = initial[gid], target[gid]
                both = before['verified'] and after['verified']
                gain = before['terminal_energy']-after['terminal_energy'] if both else None
                details.append({'group_id':gid,'original_gain_eV_atom':row['gain_eV_atom'],
                    'old_status':before['status'],'target_status':after['status'],
                    'old_verified':before['verified'],'target_verified':after['verified'],
                    'both_verified':both,'reassessed_gain_eV_atom':gain,
                    'S_positive_reconfirmed':bool(both and gain>=.01) if task=='S' else None,
                    'raw_geometry_recovery_reconfirmed':before['status']=='invalid_raw' and after['status']!='invalid_raw',
                    'G_reliability_recovery_reconfirmed':before['status'] in ['invalid_terminal','not_converged','optimizer_stop_unverified','terminal_consistency_unverified'] and after['verified']})
            result[task] = {'sources':len(rows),'target_geometry_support':sum(x['target_status']!='invalid_raw' for x in details),
                'target_R_verified':sum(x['target_verified'] for x in details),
                'both_verified':sum(x['both_verified'] for x in details),
                'target_statuses':dict(Counter(x['target_status'] for x in details)),
                'S_positive_reconfirmed':sum(x['S_positive_reconfirmed'] for x in details) if task=='S' else None,
                'S_positive_not_reconfirmed':sum(not x['S_positive_reconfirmed'] for x in details) if task=='S' else None,
                'S_both_verified_worsens_ge_0.01':sum(x['reassessed_gain_eV_atom'] is not None and x['reassessed_gain_eV_atom']<=-.01 for x in details) if task=='S' else None,
                'raw_geometry_recovery_reconfirmed':sum(x['raw_geometry_recovery_reconfirmed'] for x in details) if task=='G' else None,
                'G_reliability_recovery_reconfirmed':sum(x['G_reliability_recovery_reconfirmed'] for x in details) if task=='G' else None,
                'details':details}
        output['splits'][split] = result
    (root/'TEACHER64_ANALYSIS.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({**output,'splits':{s:{t:{k:v for k,v in row.items() if k!='details'} for t,row in d.items()} for s,d in output['splits'].items()}},indent=2))

if __name__=='__main__':
    main(sys.argv[1])
