"""Fetch bounded read-only chunks through the existing A800 pane, preserving bytes."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess as sp
import sys
import time
import zlib

parser = argparse.ArgumentParser()
parser.add_argument('--metadata-receipt', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
receipt = json.loads(args.metadata_receipt.read_text(encoding='utf-8'))
assert receipt['status'] == 'completed' and receipt['returncode'] == 0 and not receipt.get('output_prefix_lost')
meta = json.loads(receipt['output'])
relay = Path(__file__).with_name('tmux_relay.py')
remote_python = '/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python'
args.output_dir.mkdir(parents=True, exist_ok=True)
chunk_dir = args.output_dir / '.transfer_chunks'
chunk_dir.mkdir(exist_ok=True)
receipts = []
packed = bytearray()
for offset in range(0, meta['bytes'], 24576):
    chunk_file = chunk_dir / (str(offset) + '.bin')
    expected_size = min(24576, meta['bytes'] - offset)
    if chunk_file.is_file():
        block = chunk_file.read_bytes()
        assert len(block) == expected_size
    else:
        code = ('import base64,hashlib,json;from pathlib import Path;'
                f'd=Path({meta["path"]!r}).read_bytes()[{offset}:{offset+expected_size}];'
                f'print(json.dumps({{"offset":{offset},"sha256":hashlib.sha256(d).hexdigest(),"data":base64.b64encode(d).decode()}}))')
        invocation = [sys.executable, str(relay), '--command', remote_python + ' -c ' + shlex.quote(code), '--quiet']
        began = time.monotonic()
        while True:
            result = sp.run(invocation, capture_output=True, text=True, encoding='utf-8', timeout=65)
            response = json.loads(result.stdout.splitlines()[-1])
            if response.get('status') == 'completed':
                record = json.loads(Path(response['receipt']).read_text(encoding='utf-8'))
                assert record['returncode'] == 0 and not record.get('output_prefix_lost'), record.get('status')
                value = json.loads(record['output'])
                block = base64.b64decode(value['data'])
                assert value['offset'] == offset and len(block) == expected_size
                assert hashlib.sha256(block).hexdigest() == value['sha256']
                receipts.append(str(Path(response['receipt']).resolve()))
                with chunk_file.open('xb') as stream:
                    stream.write(block)
                break
            if response.get('status') not in ('running', 'delivery_unknown', 'transport_timeout', 'transport_error'):
                raise RuntimeError('unexpected transfer status: ' + response.get('status', 'missing'))
            if time.monotonic() - began > 240:
                raise RuntimeError('read-only transfer remains unresolved; resume with status-only before any new command')
            invocation = [sys.executable, str(relay), '--status-only', '--quiet']
    packed.extend(block)
    print(json.dumps({'event': 'geometry_transfer', 'received': len(packed), 'total': meta['bytes']}), flush=True)
assert len(packed) == meta['bytes'] and hashlib.sha256(packed).hexdigest() == meta['sha256']
plain = zlib.decompress(packed)
assert len(plain) == meta['plain_bytes'] and hashlib.sha256(plain).hexdigest() == meta['plain_sha256']
files = json.loads(plain)
source_receipts = []
for name, row in files.items():
    if Path(name).name != name or not name.endswith(('.json', '.jsonl')):
        raise ValueError('export name is not a single JSON file')
    data = zlib.decompress(base64.b64decode(row['zlib_base64']))
    assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
    target = args.output_dir / name
    if target.exists():
        assert target.read_bytes() == data
    else:
        target.write_bytes(data)
    source_receipts.append({key: value for key, value in row.items() if key != 'zlib_base64'} | {'local': str(target.resolve())})
(args.output_dir / 'SOURCE_RECEIPTS.json').write_text(json.dumps(source_receipts, indent=2) + '\n', encoding='utf-8')
(args.output_dir / 'TRANSFER_RECEIPT.json').write_text(json.dumps({'remote': meta, 'receipts': receipts,
    'metadata_receipt': str(args.metadata_receipt.resolve()), 'exact_original_bytes_verified': True}, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'files': len(files), 'output_dir': str(args.output_dir.resolve()), 'verified': True}), flush=True)
