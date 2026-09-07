"""Read the existing tmux pane only; never type an inner command."""
import json
from pathlib import Path
import shlex
import subprocess as sp
import time

state = json.loads(Path(__file__).with_name('.relay_state.json').read_text())
nonce = state['nonce']
code = '''import json,re,subprocess as s
print(json.dumps({'outer_reached':True}),flush=True)
out=s.check_output(['tmux','capture-pane','-J','-p','-S','-80','-t','ssha800:1.0'],text=True,timeout=5)
command=s.check_output(['tmux','display-message','-p','-t','ssha800:1.0','#{pane_current_command}'],text=True,timeout=5).strip()
tail=[]
for line in out.splitlines()[-18:]:
 if re.search(r'api.?key|password|secret|Set-Cookie',line,re.I):line='[sensitive line omitted]'
 tail.append(line[:220])
end=re.search(r'^R03_END_'+re.escape(NONCE)+r' (\d+)\s*$',out,re.M)
print(json.dumps({'pane_command':command,'nonce_occurrences':out.count(NONCE),'begin_anywhere':('R03_BEGIN_'+NONCE) in out,'end_anywhere':('R03_END_'+NONCE) in out,'end_returncode':int(end.group(1)) if end else None,'matched_end':end.group(0) if end else None,'tail':tail}))
'''.replace('NONCE', repr(nonce))
try:
    result = sp.run(['ssh', '-4', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                     '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2', 'starteam5090',
                     'python3 -u -c ' + shlex.quote(code)], capture_output=True, text=True, timeout=45)
    print(result.stdout)
    if result.returncode == 0:
        observed = json.loads(result.stdout.splitlines()[-1])
        if type(observed.get('end_returncode')) is int and observed.get('matched_end', '').strip() == 'R03_END_' + nonce + ' ' + str(observed['end_returncode']):
            recovered = {'status': 'completed', 'returncode': observed['end_returncode'], 'nonce': nonce,
                         'output_prefix_lost': not observed['begin_anywhere'], 'diagnostic': observed,
                         'recorded_unix': time.time(), 'recovery_basis': 'exact terminal nonce observed read-only'}
            receipts = Path(__file__).resolve().parents[2] / 'docs/r03_paper_story_20260907/execution/remote_receipts'
            (receipts / (str(time.time_ns()) + '_' + nonce + '_recovered.json')).write_text(json.dumps(recovered, indent=2) + '\n')
            Path(__file__).with_name('.relay_state.json').write_text(json.dumps({'nonce': nonce, 'status': 'completed', 'returncode': observed['end_returncode']}) + '\n')
    if result.returncode:
        print(json.dumps({'returncode': result.returncode, 'error': result.stderr[-1000:]}))
except sp.TimeoutExpired as error:
    output = error.stdout or b''
    if isinstance(output, bytes):
        output = output.decode(errors='replace')
    print(json.dumps({'transport_timeout': True, 'partial_output': output}))
