"""Bind the 256 trial to its reviewed, completed geometry-canary source."""
import hashlib
import json
from pathlib import Path


def require_geometry_canary_acceptance(run_root, source):
    run_root, source = Path(run_root).resolve(), Path(source).resolve()
    accepted_path = run_root / 'GEOMETRY_CANARY_ACCEPTED.json'
    completed_path = run_root / 'GEOMETRY_CANARY_COMPLETE.json'
    accepted = json.loads(accepted_path.read_text())
    complete = json.loads(completed_path.read_text())
    for record in (accepted, complete):
        if any(not isinstance(record.get(key), str) or not record[key] or not Path(record[key]).is_absolute()
               for key in ('source', 'directory')):
            raise ValueError('canary source and directory must be explicit absolute paths')
    directory = Path(complete['directory']).resolve()
    if (accepted.get('accepted') is not True
            or accepted.get('geometry_canary_complete_sha256') != hashlib.sha256(completed_path.read_bytes()).hexdigest()
            or Path(accepted.get('source', '')).resolve() != source
            or Path(complete.get('source', '')).resolve() != Path(accepted.get('canary_source', str(source))).resolve()
            or Path(accepted.get('directory', '')).resolve() != directory
            or directory.parent != run_root or not directory.name.startswith('canary_geometry_')
            or complete.get('requests_per_arm') != 16 or complete.get('roles') != ['G', 'P']
            or not (directory / '_SUCCESS').is_file() or (directory / '_FAILED').exists()
            or not (source / '_CODE_READY').is_file()):
        raise ValueError('256 trial is not bound to the accepted complete geometry-canary source')
    canary_source = Path(complete['source']).resolve()
    if canary_source != source:
        pins = accepted.get('unchanged_generation_files')
        actual = {str(path.relative_to(canary_source)) for path in (canary_source / 'src').rglob('*.py')}
        if not isinstance(pins, dict) or not actual or set(pins) != actual:
            raise ValueError('evaluation-only source update must bind every generation Python source')
        for relative, expected in pins.items():
            for archive in (canary_source, source):
                if hashlib.sha256((archive / relative).read_bytes()).hexdigest() != expected:
                    raise ValueError('generation changed after the accepted canary: ' + relative)
    return accepted
