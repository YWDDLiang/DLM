"""Prepare the same tmux relay for an already-open outer SSH terminal."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import time
import uuid
from tmux_relay import REMOTE_PYTHON, STATE, PANE, outer_program

parser = argparse.ArgumentParser()
parser.add_argument('--command')
parser.add_argument('--python-code-file', type=Path)
parser.add_argument('--status-only', action='store_true')
parser.add_argument('--pane', help='Reuse a verified already-connected idle tmux pane; never opens SSH')
args = parser.parse_args()
previous = json.loads(STATE.read_text()) if STATE.exists() else None
pane = (previous.get('pane', PANE) if args.status_only and previous else args.pane or PANE)
if args.python_code_file:
    if args.command or args.status_only:
        parser.error('choose one relay action')
    args.command = REMOTE_PYTHON + ' -c ' + shlex.quote(args.python_code_file.read_text(encoding='utf-8'))
if args.status_only:
    if not previous:
        parser.error('no previous command')
    nonce, inner = previous['nonce'], None
else:
    if not args.command or (previous and previous.get('status') not in ('completed', 'abandoned_read_only')):
        parser.error('previous delivery must be resolved before a new command')
    nonce = uuid.uuid4().hex[:12]
    payload = base64.b64encode(args.command.encode()).decode()
    wrapper = ('import base64,subprocess;'
               f"print('R03_BEGIN_{nonce}',flush=True);"
               f"r=subprocess.run(base64.b64decode('{payload}').decode(),shell=True);"
               f"print('\\n\\x1b[2KR03_END_{nonce} '+str(r.returncode),flush=True)")
    inner = REMOTE_PYTHON + ' -c ' + shlex.quote(wrapper)
    if len(inner) >= 3000:
        parser.error('inner command must remain below 3000 characters')
    STATE.write_text(json.dumps({'nonce': nonce, 'status': 'delivery_unknown', 'submitted_unix': time.time(), 'pane': pane}) + '\n')
encoded = base64.b64encode(outer_program(nonce, inner, pane=pane).encode()).decode()
begin, end = 'R03_FRAME_BEGIN_' + nonce, 'R03_FRAME_END_' + nonce
wrapper = ('import base64,builtins,textwrap;'
           'emit=lambda value,*a,**k:builtins.print(' + repr(begin + '\n')
           + '+"\\n".join(textwrap.wrap(base64.b64encode(value.encode()).decode(),60))+'
           + repr('\n' + end) + ',flush=True);'
           'exec(base64.b64decode(' + repr(encoded) + '),{"print":emit})')
shell = 'python3 -u -c ' + shlex.quote(wrapper)
print(json.dumps({'nonce': nonce, 'outer_command': shell,
                  'command_sha256': hashlib.sha256(args.command.encode()).hexdigest() if args.command else None}))
