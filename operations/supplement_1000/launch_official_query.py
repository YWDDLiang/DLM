#!/usr/bin/env python3
"""Read an authorized private MP credential into a one-use process environment.

The credential provider file is never modified or copied to an experiment.
The frozen official query consumes the environment carrier before its requests.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--credential-file", type=Path, required=True)
    args = parser.parse_args()
    source = args.run_root / "source"
    config = json.loads((source / "CONFIG.json").read_text())
    if sys.executable != config["runtime"]["official_mp_python"]:
        raise RuntimeError("use the registered official MP interpreter")
    key = args.credential_file.read_text(encoding="ascii").strip()
    if len(key) != 32 or any(char.isspace() for char in key):
        raise ValueError("credential file must contain a valid raw MP API key")
    with (args.run_root / "QUERY_STARTED.json").open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
                   "credential_serialized": False}, handle)
    for name in ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY"):
        os.environ.pop(name, None)
    carrier = "H1_OFFICIAL_QUERY_MEMORY_KEY"
    os.environ[carrier] = key
    sys.path.insert(0, str(source))
    sys.argv = [str(source / "query_official_mp.py"), "--config", str(source / "CONFIG.json"),
                "--source-dir", str(source), "--source-manifest-sha256",
                (args.run_root / "SOURCE_MANIFEST_SHA256").read_text().strip(),
                "--run-root", str(args.run_root), "--key-env", carrier]
    try:
        runpy.run_path(str(source / "query_official_mp.py"), run_name="__main__")
    except BaseException as error:
        report = {"type": type(error).__name__, "error": str(error).replace(key, "[REDACTED]")[:1000],
                  "credential_serialized": False}
        (args.run_root / "QUERY_FAILED.json").write_text(json.dumps(report) + "\n")
        print(json.dumps(report), flush=True)
        raise SystemExit(1) from None
    finally:
        os.environ.pop(carrier, None)
        key = ""
    (args.run_root / "QUERY_SUCCESS").touch()


if __name__ == "__main__":
    main()
