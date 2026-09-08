"""Compare gauge-control proposals to each other and their exact common OLD inputs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics as st
import sys


def main(directory):
    root = Path(directory)
    data = json.loads((root/'GAUGE64_COMPACT.json').read_text())
    old_data = json.loads((root/'GAUGE64_OLD_COMPACT.json').read_text())
    result = {'schema': 'expert_gauge64_analysis_v1', 'generation_seed': 2026090813,
        'source_capsules_sha256': {n: hashlib.sha256((root/n).read_bytes()).hexdigest()
                                  for n in ['GAUGE64_COMPACT.json','GAUGE64_OLD_COMPACT.json']},
        'same_OLD_inputs_across_arms': data['old_input_equality'],
        'R_deterministic_for_all_arms_and_OLD': True, 'interpretation': '64-source mechanism diagnostic, not MAIN or SUN/MSUN',
        'common_CE_reference': 'original unaligned target, for both arms; representation dependent secondary measure',
        'arms': {}}
    for arm in ['control', 'aligned']:
        results = {'train_seconds': data[f'{arm}:TRAIN_FINAL.json']['train_seconds'],
                   'common_CE_at400': data[f'{arm}:TRAIN_FINAL.json']['metrics']}
        for split in ['train','dev']:
            old = {x['group_id']: x for x in old_data[split]['rows']}
            rows = data[f'{arm}:{split}']['rows']
            assert len(old) == len(rows) == 64 and set(old) == {x['group_id'] for x in rows}
            if any(x['versions'].get('deterministic_algorithms_enabled') is not True for x in rows + old_data[split]['rows']):
                raise ValueError('mixed physics execution settings')
            summary = {'sources': 64, 'statuses': dict(Counter(x['status'] for x in rows)),
                'common_R_input_support': sum(x['status'] != 'invalid_raw' for x in rows),
                'R_verified': sum(x['verified'] for x in rows), 'tasks': {}}
            for task in ['G','S']:
                selected = [x for x in rows if x['trajectory_id'].split(':')[-1] == task]
                pairs = [(old[x['group_id']],x) for x in selected]
                verified = [(a,b) for a,b in pairs if a['verified'] and b['verified']]
                gains = [a['terminal_energy'] - b['terminal_energy'] for a,b in verified]
                item = {'sources':len(selected), 'old_R_verified':sum(a['verified'] for a,b in pairs),
                    'proposal_R_verified':sum(b['verified'] for a,b in pairs),
                    'proposal_geometry_support':sum(b['status'] != 'invalid_raw' for a,b in pairs),
                    'both_verified_pairs':len(verified), 'S_gain_ge_0.01':sum(g >= .01 for g in gains) if task=='S' else None,
                    'both_verified_energy_worsens_by_ge_0.01':sum(g <= -.01 for g in gains),
                    'mean_verified_old_minus_proposal_energy_eV_atom':st.mean(gains) if gains else None,
                    'median_verified_old_minus_proposal_energy_eV_atom':st.median(gains) if gains else None,
                    'verified_pair_details':[{'group_id':b['group_id'], 'gain_eV_atom':a['terminal_energy']-b['terminal_energy']} for a,b in verified]}
                if task == 'G':
                    item['geometry_recovery_pairs'] = sum(a['status']=='invalid_raw' and b['status']!='invalid_raw' for a,b in pairs)
                    item['reliability_recovery_from_geometry_valid_old'] = sum(a['status']!='invalid_raw' and not a['verified'] and b['verified'] for a,b in pairs)
                summary['tasks'][task] = item
            results[split] = summary
        result['arms'][arm] = results
    certs = [x['training_target_certificate'] for x in data['representatives']]
    result['actual_training_representation_changes'] = {'records': len(certs),
        'old_to_original_target_coordinate_token_changes': sum(x['original_coordinate_token_changes'] for x in certs),
        'old_to_training_target_coordinate_token_changes': sum(x['training_coordinate_token_changes'] for x in certs),
        'zero_edits': sum(x['training_representative_zero_edit'] for x in data['representatives'])}
    result['common_initial_train_eval_equal'] = data['control:curves.jsonl'][0]['train'] == data['aligned:curves.jsonl'][0]['train']
    result['common_initial_dev_eval_equal'] = data['control:curves.jsonl'][0]['dev'] == data['aligned:curves.jsonl'][0]['dev']
    result['cumulative_conservative_training_seconds'] = 8647.386792445555 + sum(data[f'{a}:TRAIN_FINAL.json']['train_seconds'] for a in ['control','aligned'])
    (root/'GAUGE64_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'representative':result['actual_training_representation_changes'],
                     'results':{a:{s:{'geometry':x[s]['common_R_input_support'],'R_verified':x[s]['R_verified'],
                                     'G':{k:v for k,v in x[s]['tasks']['G'].items() if k!='verified_pair_details'},
                                     'S':{k:v for k,v in x[s]['tasks']['S'].items() if k!='verified_pair_details'}} for s in ['train','dev']} for a,x in result['arms'].items()}},indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
