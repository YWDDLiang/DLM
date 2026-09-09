"""Read Slurm allocation costs for exactly the jobs receipted by this run."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def collect(root):
    receipts = {}
    for path in sorted((root / 'submissions').glob('*.json')):
        record = json.loads(path.read_text())
        if 'job_id' not in record:
            continue
        job_id = str(record['job_id'])
        if not job_id.isdecimal() or job_id in receipts:
            raise ValueError('invalid or duplicate submission job ID')
        receipts[job_id] = (record, path)
    if not receipts:
        raise ValueError('no receipted jobs')
    fields = ['JobIDRaw', 'JobName', 'State', 'Start', 'End', 'ElapsedRaw', 'AllocTRES', 'TotalCPU']
    command = ['sacct', '-j', ','.join(receipts), '--parsable2', '--noheader',
               '--format=' + ','.join(fields)]
    raw = subprocess.check_output(command, env={**os.environ, 'TZ': 'UTC'}, text=True)
    jobs = []
    seen = set()
    for line in raw.splitlines():
        values = line.split('|')
        if len(values) != len(fields):
            raise ValueError('unexpected sacct field count')
        row = dict(zip(fields, values))
        job_id = row['JobIDRaw']
        if job_id not in receipts:
            continue  # Exclude .batch/.extern/.0 steps: parent accounts allocation.
        if job_id in seen:
            raise ValueError('duplicate sacct allocation')
        seen.add(job_id)
        receipt, path = receipts[job_id]
        match = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)', row['AllocTRES'])
        gpus = int(match.group(1)) if match else 0
        elapsed = int(row['ElapsedRaw'])
        row.update(stage=receipt['job'], allocated_gpus=gpus,
                   allocated_GPU_hours=gpus * elapsed / 3600,
                   receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        jobs.append(row)
    if seen != set(receipts):
        raise ValueError('sacct missing receipted allocations: ' + str(set(receipts) - seen))
    return {'observed_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'root': str(root), 'budget': json.loads((root / 'BUDGET.json').read_text()),
            'jobs': jobs, 'allocated_GPU_hours': sum(x['allocated_GPU_hours'] for x in jobs),
            'running_or_pending': [x['JobIDRaw'] for x in jobs
                                   if x['State'] in ('RUNNING', 'PENDING', 'COMPLETING', 'CONFIGURING')],
            'definition': 'Allocated GPU count times allocation elapsed seconds / 3600; not GPU busy time.',
            'sacct_timezone': 'UTC', 'sacct_raw': raw,
            'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.root)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps({key: report[key] for key in
                      ('allocated_GPU_hours', 'running_or_pending', 'observed_utc')}))
