"""Archive committed root-cause evidence, source, tests and configs with per-file hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.repo.resolve()
    output = args.output.resolve()
    assert output.is_relative_to(root) and not output.exists()
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    study = 'docs/r03_paper_story_20260907/execution/expert_self_edit/mechanism_study/'
    historical = 'docs/h1a2_to_current_review_20260907/'
    paths = [p for p in tracked if p and (
        p.startswith(study) or p.startswith(historical) or p.startswith('src/') or
        p.startswith('operations/r03_c3fd_main_20260907/') or p.startswith('tests/') or
        p in ['pyproject.toml', 'requirements.txt', 'README.md'])
        and not p.endswith(('.zip', '.b64', '.pyc'))]
    assert paths and any(p.endswith('FIT8_ANALYSIS.json') for p in paths)
    changed = set(subprocess.check_output(['git', 'diff', '--name-only', '-z', 'HEAD'], cwd=root).decode().split('\0'))
    assert not changed.intersection(paths), 'commit evidence before packaging'
    metadata = {'schema': 'expert_root_cause_archive_v1',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
        'scope': 'Source, tests, historical reference documentation, run configs, per-source result capsules, analyses and figures.',
        'large_artifacts': 'Model checkpoints, original generated geometries and immutable historical runtime trees remain at the remote locations recorded in manifests; they are not embedded in this compact archive.',
        'frozen_original_MAIN': 'Original formal MAIN results are unchanged and are stored separately in the expert_self_edit/final_results archive.',
        'files': {}}
    for rel in sorted(paths):
        path = root / rel
        data = path.read_bytes()
        metadata['files'][rel] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for rel in sorted(paths):
            archive.write(root/rel, rel)
        archive.writestr('ARCHIVE_MANIFEST.json', json.dumps(metadata, indent=2)+'\n')
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read('ARCHIVE_MANIFEST.json'))
        for rel, record in manifest['files'].items():
            data = archive.read(rel)
            assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
    receipt = {'path': str(output), 'bytes': output.stat().st_size,
        'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'file_count': len(paths),
        'source_commit': metadata['source_commit'], 'all_member_hashes_verified': True}
    output.with_suffix('.manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
