"""Separate teacher-forced fitting, free generation on train, and fixed-panel DEV physics."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def source_interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(2026090837)
    draws = rng.integers(0, len(values), (20000, len(values)))
    return [float(v) for v in np.quantile(values[draws].mean(1) * 100, [.025, .975])]


def main(directory):
    root = Path(directory)
    names = ['FIT8_COMPACT.json', 'GAUGE64_OLD_COMPACT.json', 'TEACHER64_COMPACT.json',
             'RECERTIFY128_COMPACT.json']
    capsule, old_capsule, teacher, recert = [json.loads((root/n).read_text()) for n in names]
    for entry in capsule['files'].values():
        assert hashlib.sha256(entry['utf8'].encode()).hexdigest() == entry['sha256']
    checkpoint = json.loads(capsule['files']['checkpoint/CHECKPOINT_FINAL.json']['utf8'])
    assert checkpoint['global_step'] == 1600 and checkpoint['example_cursor'] == 38400
    assert checkpoint['contract']['smoke_sources'] == 8
    assert checkpoint['contract']['content_target_mode'] == 'next_token'
    assert checkpoint['contract']['m2t_probability'] == 1
    report = {'schema': 'expert_fit8_analysis_v1', 'MAIN_or_SUN_claim': False,
        'scope': 'capacity/memorization diagnostic; eight-versus-64 is not a source-count-only comparison',
        'source_sha256': {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},
        'training_final': json.loads(capsule['files']['train/TRAIN_FINAL.json']['utf8']),
        'checkpoint_identity': {k: checkpoint[k] for k in ['global_step', 'example_cursor', 'grouping', 'model_files_sha256']},
        'curves': [json.loads(line) for line in capsule['files']['train/curves.jsonl']['utf8'].splitlines()],
        'splits': {}}
    for split, seeds in [('train', [2026090813, 2026090913]),
                         ('dev', [2026090813, 2026090913, 2026091013, 2026091113])]:
        old = {x['group_id']: x for x in old_capsule[split]['rows']}
        target_labels = {x['group_id']: x for x in teacher['labels'][split]['rows']}
        certified_records = [json.loads(line) for line in recert['files']['compiled/'+split+'.jsonl']['utf8'].splitlines()]
        panels = {s: capsule['panels'][split+':'+str(s)] for s in seeds}
        rows_by_seed = {}
        for seed, panel in panels.items():
            for name in ['physics_input', 'old_physics_input']:
                assert hashlib.sha256(panel[name]['utf8'].encode()).hexdigest() == panel[name]['sha256']
            assert panel['label_report']['input_sha256'] == panel['physics_input']['sha256']
            labels = {x['group_id']: x for x in panel['labels']}
            details = []
            for sample in panel['samples']:
                gid = sample['ancestor_id']
                before, after = old[gid], labels[gid]
                traces = sample['output']['trace']
                assert len(traces) == 1 and traces[0]['mode'] == 'full_cell'
                trace = traces[0]
                proposed, target = trace['proposal_body'], sample['target_body']
                positions = trace['positions']
                expected = list(range(1, 7)) + [8+4*i+a for i in range(sample['num_atoms']) for a in range(3)]
                assert sorted(positions) == expected and len(proposed) == len(target) == len(sample['old_body'])
                both = bool(before['verified'] and after['verified'])
                gain = before['terminal_energy'] - after['terminal_energy'] if both else None
                target_label = target_labels[gid]
                # The recertified compiler preserves the original OLD and target bodies.
                annotated = next(x for x in certified_records if x['ancestor_id'] == gid
                    and x.get('target_physics_id') == sample['target_physics_id'] and not x.get('state_only'))
                assert sample['target_physics_id'] == target_label['trajectory_id']
                assert sample['target_body'] == annotated['target_body']
                assert sample['old_body'] == annotated['old_body']
                teacher_exact = proposed == target
                same_structure_mismatch = teacher_exact and any(after[k] != target_label[k]
                    for k in ['status', 'verified', 'terminal_energy', 'actual_steps'])
                correct = [proposed[p] == target[p] for p in positions]
                details.append({'group_id': gid, 'task': sample['task'], 'record_id': sample['record_id'],
                    'target_physics_id': sample['target_physics_id'], 'num_atoms': sample['num_atoms'],
                    'geometry_support': after['status'] != 'invalid_raw', 'R_verified': after['verified'],
                    'old_verified_lost': bool(before['verified'] and not after['verified']),
                    'newly_R_verified': bool(not before['verified'] and after['verified']),
                    'status': after['status'], 'old_status': before['status'],
                    'gain_eV_atom': gain, 'verified_gain_ge_0.01': bool(both and gain >= .01),
                    'verified_worsens_ge_0.01': bool(both and gain <= -.01),
                    'G_contract': bool((before['status'] == 'invalid_raw' and after['status'] != 'invalid_raw')
                         or (before['status'] != 'invalid_raw' and not before['verified'] and after['verified'])),
                    'target_body_exact': teacher_exact, 'mutable_correct': sum(correct),
                    'mutable_positions': len(positions), 'lattice_correct': sum(proposed[p] == target[p] for p in range(1,7)),
                    'coordinate_correct': sum(proposed[p] == target[p] for p in expected[6:]),
                    'first_error_reveal_index': next((i for i,v in enumerate(correct) if not v), None),
                    'proposal_changed_old': proposed != sample['old_body'], 'accepted': bool(trace['accepted']),
                    'accepted_verified_gain_ge_0.01': bool(trace['accepted'] and both and gain >= .01),
                    'accepted_verified_worsens_ge_0.01': bool(trace['accepted'] and both and gain <= -.01),
                    'rejected_verified_gain_ge_0.01': bool(not trace['accepted'] and both and gain >= .01),
                    'forward_calls': sample['output']['forward_calls'],
                    'exact_teacher_label_mismatch': bool(same_structure_mismatch)})
            rows_by_seed[seed] = {x['group_id']: x for x in details}
        tasks = {g: x['task'] for g, x in rows_by_seed[seeds[0]].items()}
        assert all(set(x) == set(tasks) for x in rows_by_seed.values())
        summary = {'seeds': seeds, 'independent_sources': len(tasks), 'observations': len(tasks)*len(seeds),
                   'tasks': {}, 'details_by_seed': {str(s): list(x.values()) for s,x in rows_by_seed.items()}}
        for task in ['G', 'S', 'all']:
            gids = sorted(g for g,t in tasks.items() if task == 'all' or t == task)
            all_rows = [rows_by_seed[s][g] for g in gids for s in seeds]
            result = {'sources': len(gids), 'observations': len(all_rows), 'statuses': dict(Counter(x['status'] for x in all_rows)),
                      'old_geometry_support': sum(old[g]['status'] != 'invalid_raw' for g in gids),
                      'old_R_verified': sum(old[g]['verified'] for g in gids),
                      'mutable_token_accuracy': sum(x['mutable_correct'] for x in all_rows)/sum(x['mutable_positions'] for x in all_rows),
                      'lattice_token_accuracy': sum(x['lattice_correct'] for x in all_rows)/(6*len(all_rows)),
                      'coordinate_token_accuracy': sum(x['coordinate_correct'] for x in all_rows)/sum(3*x['num_atoms'] for x in all_rows)}
            for metric in ['geometry_support', 'R_verified', 'verified_gain_ge_0.01', 'verified_worsens_ge_0.01',
                           'G_contract', 'target_body_exact', 'accepted', 'proposal_changed_old', 'exact_teacher_label_mismatch',
                           'accepted_verified_gain_ge_0.01', 'accepted_verified_worsens_ge_0.01', 'rejected_verified_gain_ge_0.01']:
                values = np.array([[rows_by_seed[s][g][metric] for s in seeds] for g in gids], dtype=float)
                result[metric] = {'count': int(values.sum()), 'per_seed': {str(s): int(values[:,i].sum()) for i,s in enumerate(seeds)}}
                if metric in ['geometry_support', 'R_verified', 'verified_gain_ge_0.01']:
                    result[metric]['source_bootstrap_95pct_rate_interval'] = source_interval(values.mean(1))
            delta = np.array([[int(rows_by_seed[s][g]['R_verified'])-int(old[g]['verified']) for s in seeds] for g in gids])
            result['R_verified_minus_OLD_pp'] = float(delta.mean()*100)
            result['paired_source_95pct_R_verified_change_interval_pp'] = source_interval(delta.mean(1))
            result['old_verified_lost'] = sum(x['old_verified_lost'] for x in all_rows)
            result['newly_R_verified'] = sum(x['newly_R_verified'] for x in all_rows)
            summary['tasks'][task] = result
        report['splits'][split] = summary
    out = root / 'FIT8_ANALYSIS.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'path': str(out), 'train_seconds': report['training_final']['train_seconds'],
        'splits': {s: {task: {k: v for k, v in values.items() if k in ['sources', 'observations',
            'geometry_support', 'R_verified', 'verified_gain_ge_0.01', 'accepted', 'old_verified_lost']}
            for task, values in x['tasks'].items() if task != 'all'} for s, x in report['splits'].items()}}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
