"""Explicit training-only endpoint rewards, separated from heldout evaluation."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = 'sun_training_feedback_inputs_v1'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def composition_counts(plan):
    elements, counts = plan['elements'], plan['counts']
    if len(elements) != len(counts) or len(set(elements)) != len(elements):
        raise ValueError('composition elements/counts are not aligned')
    if any(type(n) is not int or n <= 0 for n in counts) or sum(counts) != plan['N'] or not 1 <= plan['N'] <= 20:
        raise ValueError('invalid fixed MP20 composition')
    return dict(zip(elements, counts))


def reduced_key(counts):
    divisor = math.gcd(*counts.values())
    return '|'.join(f'{element}:{count//divisor}' for element, count in sorted(counts.items()))


def record_composition(record):
    values = []
    structure = record.get('structure')
    if structure is not None:
        elements = []
        for site in structure['sites']:
            species = site['species']
            if len(species) != 1 or species[0].get('occu', 1) != 1:
                raise ValueError('training feedback requires ordered integer-occupancy structures')
            elements.append(species[0]['element'])
        values.append(dict(Counter(elements)))
    body = record.get('body')
    if body:
        elements = re.findall(r'<E_([A-Z][a-z]?)>', body)
        count = re.search(r'<N_(\d{3})>', body)
        if count is None or len(elements) != int(count.group(1)):
            raise ValueError('feedback body has no complete fixed-composition canvas')
        values.append(dict(Counter(elements)))
    if not values:
        if record.get('success') is not False or not isinstance(record.get('declared_composition'), dict):
            raise ValueError('missing endpoint needs an explicit failed-input composition')
        values.append(record['declared_composition'])
    if any(x != values[0] for x in values):
        raise ValueError('body and structure compositions disagree')
    return values[0]


def validate_training_feedback(records, paths_file, manifest_file, *, endpoint=None):
    manifest_file = Path(manifest_file)
    spec = json.loads(manifest_file.read_text(encoding='utf-8'))
    if spec.get('schema') != SCHEMA or spec.get('purpose') != 'training_feedback':
        raise ValueError('an explicit training feedback manifest is required')
    def pinned(name):
        item = spec[name]
        path = Path(item['path'])
        if not path.is_file() or sha256(path) != item['sha256']:
            raise ValueError(f'{name} changed after registration')
        return path
    declared_paths = pinned('paths')
    if declared_paths.resolve() != Path(paths_file).resolve():
        raise ValueError('feedback manifest belongs to a different input file')
    prepared_path, parent_path, heldout_path = (pinned(k) for k in ['parent_preparation', 'parent_pairs', 'heldout_cohort'])
    prepared = json.loads(prepared_path.read_text(encoding='utf-8'))
    if (prepared.get('files_sha256', {}).get(parent_path.name) != sha256(parent_path)
            or prepared.get('heldout_cohort_sha256') != sha256(heldout_path)
            or not (prepared_path.parent/'_SUCCESS').is_file()):
        raise ValueError('parent preparation and heldout exclusion are not bound')
    parents = {x['ancestor_id']: x for x in rows(parent_path)}
    forbidden = set()
    for row in rows(heldout_path):
        if row.get('plan_state'):
            try:
                forbidden.add(reduced_key(composition_counts(row['plan_state'])))
            except (ValueError, KeyError, TypeError):
                if row.get('body_eligible'):
                    raise ValueError('eligible heldout condition has invalid composition')
    expected = spec.get('expected_requests')
    if type(expected) is not int or expected < 1 or len(records) != expected:
        raise ValueError('training feedback denominator changed')
    if len({x['trajectory_id'] for x in records}) != expected or len({x['group_id'] for x in records}) != expected:
        raise ValueError('a feedback arm needs one endpoint per original parent')
    if spec.get('endpoint') not in ['native', 'tau800'] or endpoint is not None and endpoint != spec['endpoint']:
        raise ValueError('feedback endpoint differs')
    for row in records:
        if (row.get('purpose') != 'training_feedback' or row.get('source_split') != 'train'
                or row.get('endpoint') != spec['endpoint']):
            raise ValueError('feedback cannot relabel evaluation/dev inputs as training')
        parent = parents[row['group_id']]
        counts = composition_counts(parent['plan_state'])
        if (parent.get('source_split') != 'train' or row.get('source_row_idx') != parent['source_row_idx']
                or record_composition(row) != counts or reduced_key(counts) in forbidden):
            raise ValueError('feedback violates fixed composition, source identity or heldout exclusion')
    return {'schema': SCHEMA, 'manifest_sha256': sha256(manifest_file),
            'parent_pairs_sha256': sha256(parent_path), 'heldout_cohort_sha256': sha256(heldout_path),
            'training_sources': expected, 'independent_evaluation': False}
