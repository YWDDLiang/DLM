"""Read preserved geometry-canary progress and metrics; perform no model work."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess as sp


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--run-root', type=Path, required=True)
args = parser.parse_args()
submission = read(args.run_root / 'CANARY_GEOMETRY_SUBMISSION.json')
root = args.run_root / ('canary_geometry_' + submission['job_id'])
report = {'directory': str(root), 'submission': submission,
          'queue': sp.check_output(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i %t %M %j'], text=True).splitlines(),
          'complete': read(args.run_root / 'GEOMETRY_CANARY_COMPLETE.json'), 'components': {}}
for role in ('G', 'P', 'I_batch1_reference'):
    component = root / role
    body = component / 'body'
    actual = rows(body / 'raw_generations.jsonl')
    events = rows(body / 'body_progress.jsonl')
    stages = {path.name: read(path) for path in sorted(component.glob('*.stage.json'))}
    item = {'final': read(component / 'COMPONENT_FINAL.json'), 'failure': read(component / 'FAILURE.json'),
            'body_progress': read(body / 'progress.json'), 'body_attempts_recorded': len(events),
            'body_metrics': read(body / 'sample_metrics.json'), 'stages': stages,
            'native_direct': read(component / 'native_direct/report.json'),
            'tau800_direct': read(component / 'tau800_direct/report.json'),
            'native_labels': read(component / 'native_labels/LABEL_FINAL.json'),
            'tau800_labels': read(component / 'tau800_labels/LABEL_FINAL.json')}
    if actual:
        item.update(requests=len(actual), statuses=dict(Counter(row['attempt_status'] for row in actual)),
                    initial_supported=sum(row.get('repair_trace', {}).get('geometry_before', {}).get('supported') is True for row in actual),
                    final_supported=sum(row.get('repair_trace', {}).get('geometry_after', {}).get('supported') is True for row in actual),
                    construction_hash=digest([(row['sample_idx'], row.get('construction_raw_body_token_ids'),
                                               row.get('construction_geometry', {}).get('status')) for row in actual]),
                    failures=[{'sample_idx': row['sample_idx'], 'reason': row.get('reason'),
                               'geometry': row.get('construction_geometry', {}).get('failure')}
                              for row in actual if not row.get('body_generation_complete')])
    if item['failure']:
        item['body_error_tail'] = (component / 'body.err').read_text()[-5000:] if (component / 'body.err').exists() else None
    report['components'][role] = item
report['hull_canary_final'] = read(args.run_root / 'hull_execution_canary/STAGE_FINAL.json')
report['hull_canary_failure'] = read(args.run_root / 'hull_execution_canary/FAILURE.json')
print(json.dumps(report, sort_keys=True))
