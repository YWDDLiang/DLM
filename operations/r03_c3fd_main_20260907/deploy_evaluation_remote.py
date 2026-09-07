"""Deploy an exact Git commit as an immutable, content-verified run snapshot.

The historical filename remains the canonical deployment entry. Model assets
and experiment outputs are external references, never copied into this tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess as sp


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def deploy(repository, run_root, commit, *, remote=None, ref=None, bundle=None):
    repository, run_root = Path(repository).resolve(), Path(run_root).resolve()
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('deployment requires a full immutable commit ID')
    if bundle and remote:
        raise ValueError('choose either a verified bundle or a Git remote')
    if remote:
        if not ref:
            raise ValueError('remote fetch requires an explicit ref')
        sp.run(['git', '-c', 'http.lowSpeedLimit=1', '-c', 'http.lowSpeedTime=20',
                'fetch', remote, ref], cwd=repository, check=True, timeout=90)
    elif bundle:
        sp.run(['git', 'bundle', 'verify', str(bundle)], cwd=repository, check=True)
        sp.run(['git', 'fetch', str(bundle)], cwd=repository, check=True)
    resolved = sp.check_output(['git', 'rev-parse', commit + '^{commit}'], cwd=repository, text=True).strip()
    if resolved != commit:
        raise ValueError('requested commit did not resolve exactly')
    destination = run_root / 'code' / commit
    destination.parent.mkdir(parents=True, exist_ok=True)
    marker, manifest_path = destination / '_CODE_READY', destination / '_SOURCE_FILES.json'
    if not marker.exists():
        destination.mkdir(exist_ok=False)
        archive = sp.Popen(['git', 'archive', commit], cwd=repository, stdout=sp.PIPE)
        try:
            result = sp.run(['tar', '--warning=no-timestamp', '-xf', '-', '-C', str(destination)], stdin=archive.stdout)
        finally:
            archive.stdout.close()
        if archive.wait() or result.returncode:
            raise RuntimeError('archive deployment failed; incomplete directory preserved')
        files = {}
        for path in destination.rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.json', '.toml', '.yaml', '.yml'):
                relative = path.relative_to(destination)
                if relative.parts[0] in ('src', 'scripts', 'operations', 'eval_runtime') or len(relative.parts) == 1:
                    files[relative.as_posix()] = digest(path)
        if not files:
            raise RuntimeError('empty deployment code manifest')
        manifest_path.write_text(json.dumps(files, sort_keys=True, indent=2) + '\n')
        marker.write_text(commit + '\n')
        for path in destination.rglob('*'):
            if path.is_file():
                path.chmod(0o444)
        for path in sorted((p for p in destination.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            path.chmod(0o555)
        destination.chmod(0o555)
    if marker.read_text().strip() != commit:
        raise ValueError('existing deployment marker differs')
    files = json.loads(manifest_path.read_text())
    for relative, expected in files.items():
        path = (destination / relative).resolve()
        if destination not in path.parents or digest(path) != expected:
            raise ValueError('deployed source differs: ' + relative)
    return {'run_root': str(run_root), 'source_root': str(destination),
            'source_identity': {'commit': commit, 'manifest_sha256': digest(manifest_path)},
            'files_verified': len(files), 'ready': True}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--remote')
    parser.add_argument('--ref')
    parser.add_argument('--bundle', type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(deploy(args.repository, args.run_root, args.commit,
                           remote=args.remote, ref=args.ref, bundle=args.bundle)), flush=True)


if __name__ == '__main__':
    main()
