"""Collect completed official metrics and scalar per-request evidence without scoring."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    r = args.run_root
    sources = [(method, endpoint, r / 'formal_baselines_recovery2' / f'{method}_{endpoint}')
               for method in ('H1A2_D1', 'B0_D2') for endpoint in ('native', 'tau800')]
    sources += [('ExpertEdit_B0_D2', 'native', r / 'formal_editor_run/ExpertEdit_B0_D2_native'),
                ('ExpertEdit_B0_D2', 'tau800', r / 'formal_editor_refined_eval/ExpertEdit_B0_D2_tau800')]
    summary, source_files, table = [], {}, []
    for method, endpoint, prefix in sources:
        validity = Path(str(prefix) + '_validity')
        sun = Path(str(prefix) + '_SUN')
        assert (validity / '_SUCCESS').is_file() and (sun / '_SUCCESS').is_file()
        vr, sr = validity / 'report.json', sun / 'EVALUATION_FINAL.json'
        vp, sp = validity / 'attempt_metrics.jsonl', sun / 'attempt_results.jsonl'
        v, s = json.loads(vr.read_text()), json.loads(sr.read_text())
        assert s['status'] == 'complete' and v['attempts'] == s['counts']['requests'] == 1200
        a, b = read_rows(vp), read_rows(sp)
        assert len(a) == len(b) == 1200
        counts = {'comp_valid': v['comp_valid_count'], 'Struct_valid': v['struct_valid_count'],
                  'SUN': s['counts']['strict_sun'], 'MSUN': s['counts']['meta_sun']}
        expected = dict.fromkeys(counts, 0)
        for index, (x, y) in enumerate(zip(a, b)):
            assert x['ordinal'] == y['ordinal'] == y['sample_idx'] == index
            values = {'comp_valid': x['comp_valid'], 'Struct_valid': x['struct_valid'],
                      'SUN': y['strict_sun'], 'MSUN': y['meta_sun']}
            assert all(type(value) is bool for value in values.values())
            for key, value in values.items():
                expected[key] += int(value)
            energy, hull = y['terminal_energy_eV_atom'], y['hull_energy_eV_atom']
            above = energy - hull if all(isinstance(t, (int, float)) and math.isfinite(t)
                                         for t in (energy, hull)) else None
            table.append({'method': method, 'endpoint': endpoint, 'sample_idx': index,
                          **{key: int(value) for key, value in values.items()},
                          'terminal_verified': int(y['terminal_verified']),
                          'terminal_status': y['terminal_status'],
                          'terminal_energy_eV_atom': energy, 'hull_energy_eV_atom': hull,
                          'energy_above_hull_eV_atom': above})
        assert expected == counts
        summary.append({'method': method, 'endpoint': endpoint, 'requests': 1200,
                        'counts': counts, 'percent': {k: value / 12 for k, value in counts.items()}})
        for path in (vr, sr, vp, sp):
            source_files[str(path)] = sha(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = args.output_dir / 'per_request_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    report = {'schema': 'expert_editor_completed_evidence_v1', 'summary': summary,
              'rows': len(table), 'source_files_sha256': source_files,
              'csv_sha256': sha(csv_path), 'exporter_sha256': sha(Path(__file__)),
              'primary_metrics_recomputed': False, 'all_requests_retained': True,
              'energy_above_hull_column': 'stored terminal energy minus stored hull energy'}
    (args.output_dir / 'RESULTS.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    (args.output_dir / '_SUCCESS').touch()
    print(json.dumps({'summary': summary, 'rows': len(table), 'output': str(args.output_dir),
                      'csv_sha256': report['csv_sha256']}))


if __name__ == '__main__':
    main()
