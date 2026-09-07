"""Export exact completed canary JSON bytes with content hashes for local analysis."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

parser = argparse.ArgumentParser()
parser.add_argument('--run-root', type=Path, required=True)
args = parser.parse_args()
marker = json.loads((args.run_root / 'GEOMETRY_CANARY_COMPLETE.json').read_text())
root = Path(marker['directory'])
files = {'GEOMETRY_CANARY_COMPLETE.json': args.run_root / 'GEOMETRY_CANARY_COMPLETE.json'}
for role in ('G', 'P', 'I_batch1_reference'):
    for name, relative in (('body.jsonl', 'body/raw_generations.jsonl'),
                           ('construction.jsonl', 'body/construction_raw_generations.jsonl'),
                           ('native_direct.json', 'native_direct/report.json'),
                           ('tau800_direct.json', 'tau800_direct/report.json'),
                           ('component.json', 'COMPONENT_FINAL.json')):
        if role != 'I_batch1_reference' or name != 'construction.jsonl':
            files[role + '_' + name] = root / role / relative
out = {}
for name, path in files.items():
    data = path.read_bytes()
    out[name] = {'source': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                 'zlib_base64': base64.b64encode(zlib.compress(data)).decode()}
print(json.dumps(out))
