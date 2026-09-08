"""Archive an idle experiment's data; prune only explicitly inventoried editor weights."""
import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile


WEIGHTS = {'adapter_model.safetensors', 'periodic_state.pt', 'expert_edit_modules.pt', 'training_state.pt'}


def sha(path):
    with Path(path).open('rb') as stream:
        return stream_sha(stream)


def stream_sha(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024**2), b''):
        result.update(block)
    return result.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def inside(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('unsafe relative archive path')
    path = root / rel
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError('path escapes the declared experiment')
    return path


def require_idle(root):
    query = ['squeue', '--array', '-h', '-u', getpass.getuser(), '-o', '%i']
    jobs = subprocess.check_output(query, text=True).split()
    own = os.environ.get('SLURM_JOB_ID')
    busy = []
    for job in jobs:
        if job == own:
            continue
        if not re.fullmatch(r'\d+(?:_\d+)?(?:\+\d+)?', job):
            raise ValueError('unsupported Slurm job identity')
        try:
            details = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            if job not in subprocess.check_output(query, text=True).split():
                continue
            raise
        fields = dict(re.findall(r'(?:^|\s)(Command|WorkDir|StdOut|StdErr)=(.*?)(?=\s\w+=|$)', details))
        if any(str(root) in value for value in fields.values()):
            busy.append({'job_id':job, 'paths':fields})
    if busy:
        raise ValueError('experiment is still referenced by active or pending Slurm jobs: ' + repr(busy))


def check_data(root, entries):
    for item in entries:
        path = inside(root, item['path'])
        if not path.is_file() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('retained data changed: ' + item['path'])


def walk_files(root):
    def failed(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=failed):
        if any((Path(directory)/name).is_symlink() for name in dirs+files):
            raise ValueError('experiment acquired a symlink')
        for name in sorted(files):
            path = Path(directory)/name
            if not path.is_file():
                raise ValueError('experiment contains a nonregular file')
            yield path


def check_inventory(root, expected):
    actual = {path.relative_to(root).as_posix() for path in walk_files(root)}
    if actual != set(expected):
        raise ValueError('experiment file inventory changed')


def archive(root, output, keep, required=(), *, check_idle=True):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('archive destination must be separate from the experiment tree')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'DATA_MANIFEST.json').exists() or (output/'experiment_data.tar.gz').exists():
        raise ValueError('archive already exists; refusing to overwrite it')
    if check_idle:
        require_idle(root)
    for relative in required:
        component = inside(root, relative)
        if not (component/'_SUCCESS').is_file() or json.loads((component/'COMPONENT_FINAL.json').read_text()).get('complete') is not True:
            raise ValueError('required experiment component is incomplete')
    paths = list(walk_files(root))
    checkpoint_dirs = {p.parent.relative_to(root).as_posix() for p in paths if p.name=='adapter_model.safetensors'}
    keep = set(keep)
    if not keep or not keep.issubset(checkpoint_dirs):
        raise ValueError('explicit retained checkpoints must exist')
    for relative in checkpoint_dirs:
        directory = inside(root, relative)
        if not (directory/'EXPERT_EDITOR.json').is_file() or not (directory/'adapter_config.json').is_file():
            raise ValueError('an unrecognized checkpoint cannot be pruned')
    data, weights = [], []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        parent = path.parent.relative_to(root).as_posix()
        status = path.stat()
        entry = {'path':relative, 'bytes':status.st_size, 'mtime_ns':status.st_mtime_ns}
        if parent in checkpoint_dirs and path.name in WEIGHTS:
            weights.append({**entry, 'retained':parent in keep})
        else:
            data.append({**entry, 'sha256':sha(path)})
    manifest = {'schema':'experiment_data_archive_v1', 'root':str(root), 'created_utc':utc(),
        'source_script_sha256':sha(__file__), 'required_complete_components':list(required),
        'retained_checkpoints':sorted(keep), 'data':sorted(data,key=lambda x:x['path']),
        'excluded_model_weights':sorted(weights,key=lambda x:x['path']),
        'data_bytes':sum(x['bytes'] for x in data),
        'prune_bytes':sum(x['bytes'] for x in weights if not x['retained']),
        'weight_archive_policy':'model weights stay outside this data archive; retained checkpoint weights remain in the experiment'}
    write(output/'DATA_MANIFEST.json', manifest)
    paths_file = output/'data_paths.null'
    paths_file.write_bytes(b''.join(x['path'].encode()+b'\0' for x in manifest['data']))
    partial = output/'experiment_data.tar.gz.partial'
    if partial.exists():
        raise ValueError('an earlier partial archive exists')
    subprocess.run(['tar', '--create', '--gzip', '--file', str(partial), '--directory', str(root),
                    '--null', '--files-from', str(paths_file)], check=True)
    expected = {x['path']:x for x in data}
    seen = set()
    with tarfile.open(partial, 'r|gz') as stream:
        for member in stream:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError('archive member inventory differs')
            item = expected[member.name]
            if member.size != item['bytes'] or stream_sha(stream.extractfile(member)) != item['sha256']:
                raise ValueError('archive member contents differ: ' + member.name)
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('archive omitted retained data')
    check_data(root, data)
    check_inventory(root, [x['path'] for x in data+weights])
    target = output/'experiment_data.tar.gz'
    partial.rename(target)
    receipt = {'schema':'experiment_data_archive_verified_v1', 'complete':True, 'verified_utc':utc(),
        'manifest_sha256':sha(output/'DATA_MANIFEST.json'), 'archive_sha256':sha(target),
        'archive_bytes':target.stat().st_size, 'retained_data_files':len(data), 'retained_data_bytes':manifest['data_bytes'],
        'excluded_weights':len(weights), 'prunable_weights':sum(not x['retained'] for x in weights),
        'prunable_bytes':manifest['prune_bytes']}
    write(output/'ARCHIVE_VERIFIED.json', receipt)
    print(json.dumps(receipt), flush=True)
    return manifest, receipt


def prune(output, *, check_idle=True):
    output = Path(output).resolve()
    receipt = json.loads((output/'ARCHIVE_VERIFIED.json').read_text())
    if (receipt.get('complete') is not True or sha(output/'DATA_MANIFEST.json') != receipt['manifest_sha256']
            or sha(output/'experiment_data.tar.gz') != receipt['archive_sha256']):
        raise ValueError('verified archive binding changed')
    manifest = json.loads((output/'DATA_MANIFEST.json').read_text())
    root = Path(manifest['root']).resolve()
    if check_idle:
        require_idle(root)
    check_data(root, manifest['data'])
    check_inventory(root, [x['path'] for x in manifest['data']+manifest['excluded_model_weights']])
    weights = []
    protected = set(manifest['retained_checkpoints'])
    for item in manifest['excluded_model_weights']:
        path = inside(root, item['path'])
        parent = path.parent.relative_to(root).as_posix()
        if path.name not in WEIGHTS or (parent in protected) != item['retained']:
            raise ValueError('weight deletion scope differs from retained checkpoint policy')
        status = path.stat()
        if status.st_size != item['bytes'] or status.st_mtime_ns != item['mtime_ns']:
            raise ValueError('checkpoint changed since archive: ' + item['path'])
        weights.append({**item, 'sha256':sha(path)})
    write(output/'CHECKPOINT_REMOVAL_PLAN.json', {'root':str(root), 'created_utc':utc(), 'weights':weights,
        'archive_sha256':receipt['archive_sha256'], 'manifest_sha256':receipt['manifest_sha256']})
    if check_idle:
        require_idle(root)
    removed = []
    with (output/'CHECKPOINT_REMOVAL_LOG.jsonl').open('x', encoding='utf-8') as log:
        for item in weights:
            if item['retained']:
                continue
            path = inside(root, item['path'])
            if path.stat().st_mtime_ns != item['mtime_ns'] or path.stat().st_size != item['bytes']:
                raise ValueError('checkpoint changed immediately before removal')
            path.unlink()
            log.write(json.dumps(item)+'\n')
            log.flush()
            os.fsync(log.fileno())
            removed.append(item)
    for relative in sorted({str(Path(x['path']).parent).replace('\\','/') for x in removed}):
        write(inside(root, relative)/'CHECKPOINT_REMOVED.json', {'removed_utc':utc(), 'reason':'user requested cleanup of unsuccessful experimental weights',
            'data_and_checkpoint_metadata_retained':True, 'archive':str(output),
            'removed_files':[x for x in removed if str(Path(x['path']).parent).replace('\\','/') == relative]})
    check_data(root, manifest['data'])
    for item in weights:
        path = inside(root, item['path'])
        if item['retained'] and sha(path) != item['sha256'] or not item['retained'] and path.exists():
            raise ValueError('checkpoint cleanup postcondition failed')
    check_inventory(root, [x['path'] for x in manifest['data']]
        +[x['path'] for x in weights if x['retained']]
        +[(Path(x['path']).parent/'CHECKPOINT_REMOVED.json').as_posix() for x in removed])
    result = {'schema':'experiment_checkpoint_cleanup_v1', 'complete':True, 'completed_utc':utc(),
        'removed_weight_files':len(removed), 'removed_bytes':sum(x['bytes'] for x in removed),
        'cleaned_checkpoints':len({str(Path(x['path']).parent) for x in removed}),
        'retained_checkpoints':sorted(protected), 'retained_data_files_verified':len(manifest['data']),
        'archive_sha256':receipt['archive_sha256'], 'manifest_sha256':receipt['manifest_sha256']}
    write(output/'CLEANUP_FINAL.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['archive','prune'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--keep-checkpoint', action='append', default=[])
    parser.add_argument('--require-complete', action='append', default=[])
    args = parser.parse_args()
    if args.mode == 'archive':
        if args.root is None:
            parser.error('--root is required for archive')
        archive(args.root, args.output, args.keep_checkpoint, args.require_complete)
    else:
        prune(args.output)
