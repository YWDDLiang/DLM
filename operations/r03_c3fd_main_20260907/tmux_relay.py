"""Use the already-open A800 SSH pane through starteam5090; never open a new inner SSH.

One recorded command at a time. An uncertain transport outcome requires a
status-only read before another dispatch. Output is framed and returned as JSON.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid


OUTER_HOST = "starteam5090"
PANE = "ssha800:1.0"
REMOTE_PYTHON = "/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python"
STATE = Path(__file__).with_name(".relay_state.json")


def outer_program(nonce: str, inner: str | None) -> str:
    values = {"nonce": nonce, "inner": inner, "pane": PANE}
    encoded = base64.b64encode(json.dumps(values).encode()).decode()
    return f'''import base64,json,re,subprocess,time
v=json.loads(base64.b64decode({encoded!r}))
def capture():
 return subprocess.check_output(['tmux','capture-pane','-J','-p','-S','-2500','-t',v['pane']],text=True)
if v['inner'] is not None:
 command=subprocess.check_output(['tmux','display-message','-p','-t',v['pane'],'#{{pane_current_command}}'],text=True).strip()
 if command!='ssh':
  print(json.dumps({{'status':'wrong_pane','command':command}}),flush=True);raise SystemExit(2)
 subprocess.run(['tmux','send-keys','-t',v['pane'],'-l',v['inner']],check=True)
 subprocess.run(['tmux','send-keys','-t',v['pane'],'Enter'],check=True)
begin='R03_BEGIN_'+v['nonce'];end='R03_END_'+v['nonce']
deadline=time.monotonic()+24
while True:
 output=capture()
 match=re.search(r'^'+re.escape(end)+r' (\\d+)\\s*$',output,re.M)
 # Legacy tqdm carriage returns can leave zero digits and a timer after the
 # printed zero exit status. Only recover this unambiguous zero case.
 if not match:
  match=re.search(r'^'+re.escape(end)+r' (0+)(?= \\[\\d{{2}}:\\d{{2}}<)',output,re.M)
 start=re.search(r'^'+re.escape(begin)+r'\\s*$',output,re.M)
 if match:
  captured=output[start.end():match.start()].strip() if start else output[:match.start()].strip()[-10000:]
  print(json.dumps({{'status':'completed','returncode':int(match.group(1)),'output':captured,'output_prefix_lost':start is None}},ensure_ascii=False),flush=True);break
 if time.monotonic()>=deadline:
  text=output[start.end():] if start else ''
  print(json.dumps({{'status':'running' if start else 'delivery_unknown','output_tail':text[-10000:]}},ensure_ascii=False),flush=True);break
 time.sleep(.4)
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command")
    parser.add_argument("--python-code-file", type=Path)
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.python_code_file:
        if args.command or args.status_only:
            parser.error("--python-code-file cannot be combined with other commands")
        args.command = REMOTE_PYTHON + " -c " + shlex.quote(args.python_code_file.read_text(encoding="utf-8"))
    previous = json.loads(STATE.read_text()) if STATE.exists() else None
    if args.status_only:
        if not previous:
            print(json.dumps({"status": "no_command"}));return 0
        nonce, inner = previous["nonce"], None
    else:
        if not args.command:
            parser.error("--command is required")
        if previous and previous.get("status") != "completed":
            parser.error("previous delivery is unresolved; use --status-only")
        nonce = uuid.uuid4().hex[:12]
        payload = base64.b64encode(args.command.encode()).decode()
        wrapped = (
            "import base64,subprocess;"
            f"print('R03_BEGIN_{nonce}',flush=True);"
            f"r=subprocess.run(base64.b64decode('{payload}').decode(),shell=True);"
            f"print('\\n\\x1b[2KR03_END_{nonce} '+str(r.returncode),flush=True)"
        )
        inner = f"{REMOTE_PYTHON} -c {shlex.quote(wrapped)}"
        if len(inner) >= 3000:
            parser.error("literal inner command must stay below 3000 characters")
        STATE.write_text(json.dumps({"nonce": nonce, "status": "delivery_unknown", "submitted_unix": time.time()})+"\n")
    remote = "python3 -u -c " + shlex.quote(outer_program(nonce, inner))
    argv = ["ssh", "-4", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2", OUTER_HOST, remote]
    try:
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", timeout=55)
    except subprocess.TimeoutExpired:
        print(json.dumps({"status": "transport_timeout", "nonce": nonce}));return 3
    lines = result.stdout.strip().splitlines()
    try:
        report = json.loads(lines[-1])
    except (ValueError, IndexError):
        report = {"status": "transport_error", "returncode": result.returncode, "stderr": result.stderr[-3000:], "stdout": result.stdout[-1000:]}
    report["nonce"] = nonce
    receipts = Path(__file__).resolve().parents[2] / "docs/r03_paper_story_20260907/execution/remote_receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    receipt = receipts / f"{time.time_ns()}_{nonce}.json"
    report["recorded_unix"] = time.time()
    if args.command:
        report["command_sha256"] = hashlib.sha256(args.command.encode()).hexdigest()
    receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if report.get("status") == "completed":
        STATE.write_text(json.dumps({"nonce": nonce, "status": "completed", "returncode": report["returncode"]})+"\n")
    printable = {"status":report.get("status"),"returncode":report.get("returncode"),"nonce":nonce,"receipt":str(receipt)} if args.quiet else report
    print(json.dumps(printable, ensure_ascii=False))
    return 0 if report.get("status") in ("completed", "running") else 3


if __name__ == "__main__":
    raise SystemExit(main())
