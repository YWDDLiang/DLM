"""Run the unchanged official query using the authorized private provider in memory."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--hull-root', type=Path, required=True)
args = parser.parse_args()
spec = json.loads((args.hull_root / 'QUERY_COMMAND.json').read_text())
if not spec['needed']:
    raise SystemExit(0)
argv = spec['argv_without_credential']
if sys.executable != argv[0]:
    raise RuntimeError('official query interpreter differs from the registered runtime')
provider = Path('/public/home/jiaosz/ywliang/.config/ai4s/materials_project_api.key')
key = provider.read_text(encoding='ascii').strip()
if len(key) != 32 or any(character.isspace() for character in key):
    raise ValueError('authorized MP provider has an invalid format')
carrier = 'H1_OFFICIAL_QUERY_MEMORY_KEY'
for name in ('MP_API_KEY', 'PMG_MAPI_KEY', 'MAPI_KEY'):
    os.environ.pop(name, None)
os.environ.update(spec['environment'])
os.environ[carrier] = key
sys.path.insert(0, str(Path(argv[1]).parent))
sys.argv = argv[1:] + ['--key-env', carrier]
try:
    runpy.run_path(argv[1], run_name='__main__')
except BaseException as error:
    # The provider remains unchanged; never echo credentials from client errors.
    if isinstance(error, SystemExit) and error.code in (None, 0):
        pass
    else:
        report = {'type': type(error).__name__, 'message': str(error).replace(key, '[REDACTED]')[:1000],
                  'credential_serialized': False, 'provider_modified': False}
        (args.hull_root / 'QUERY_FAILURE.json').write_text(json.dumps(report) + '\n')
        print(json.dumps(report), flush=True)
        raise SystemExit(1) from None
finally:
    os.environ.pop(carrier, None)
    key = ''
(args.hull_root / 'QUERY_SUCCESS').touch()
