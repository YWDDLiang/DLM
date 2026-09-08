"""Persist an exactly framed result returned by the existing outer SSH session."""
import argparse
import json
from pathlib import Path
from tmux_relay import STATE

parser = argparse.ArgumentParser()
parser.add_argument('receipt', type=Path)
args = parser.parse_args()
report = json.loads(args.receipt.read_text(encoding='utf-8'))
previous = json.loads(STATE.read_text())
if report['nonce'] != previous['nonce'] or report['status'] not in ('completed', 'running', 'delivery_unknown', 'wrong_pane'):
    raise ValueError('result does not match the outstanding relay action')
if report['status'] == 'completed':
    if type(report.get('returncode')) is not int:
        raise ValueError('completed command has no actual exit status')
    STATE.write_text(json.dumps({'nonce': report['nonce'], 'status': 'completed', 'returncode': report['returncode'],
                                 'pane': previous.get('pane', 'ssha800:1.0')}) + '\n')
print(json.dumps({'status': report['status'], 'returncode': report.get('returncode'), 'receipt': str(args.receipt.resolve())}))
