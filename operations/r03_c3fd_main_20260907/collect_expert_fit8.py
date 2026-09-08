"""Collect the fixed eight-source fit check, including actual free-running bodies."""
import hashlib
import json
from pathlib import Path
import sys


def main(directory):
    root = Path(directory)
    run = root / 'mechanism_fit8'
    read = lambda p: [json.loads(line) for line in p.open()]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    normalized = lambda p: {x['group_id']: {k: x[k] for k in
        ['body', 'structure', 'source_row_idx', 'source_split']} for x in read(p)}
    result = {'schema': 'expert_fit8_capsule_v1', 'files': {}, 'panels': {}}
    assert (run / 'train/_SUCCESS').is_file()
    for name in ['TRAIN_CONFIG.json', 'TRAIN_FINAL.json', 'curves.jsonl']:
        path = run / 'train' / name
        result['files']['train/' + name] = {'sha256': sha(path), 'utf8': path.read_text()}
    for name in ['CHECKPOINT_FINAL.json', 'EXPERT_EDITOR.json', 'expert_edit_config.json']:
        path = run / 'train/checkpoint' / name
        result['files']['checkpoint/' + name] = {'sha256': sha(path), 'utf8': path.read_text()}
    for split, seeds in [('train', [2026090813, 2026090913]),
                         ('dev', [2026090813, 2026090913, 2026091013, 2026091113])]:
        reference_path = root / ('mechanism_gauge64_control/how_' + split) / 'old_physics.jsonl'
        reference = normalized(reference_path)
        expected = 8 if split == 'train' else 64
        first_inputs = None
        for seed in seeds:
            key = split + ':' + str(seed)
            sample_dir = run / ('how_' + split + '_' + str(seed))
            label_dir = run / ('labels_' + split + '_' + str(seed))
            assert (sample_dir / '_SUCCESS').is_file() and (label_dir / '_SUCCESS').is_file()
            old_path = sample_dir / 'old_physics.jsonl'
            actual = normalized(old_path)
            assert len(actual) == expected
            assert all(actual[g] == reference[g] for g in actual)
            if first_inputs is not None:
                assert actual == first_inputs
            first_inputs = actual
            labels = read(label_dir / 'labels.jsonl')
            samples = read(sample_dir / 'samples.jsonl')
            assert len(labels) == len(samples) == expected
            assert set(actual) == {x['group_id'] for x in labels} == {x['ancestor_id'] for x in samples}
            assert all(x['versions']['deterministic_algorithms_enabled'] is True for x in labels)
            assert not any(x['status'] in ['worker_error', 'worker_timeout'] for x in labels)
            result['panels'][key] = {
                'input_identity': {'normalized_equal_to_common_old': True,
                    'common_reference_sha256': sha(reference_path), 'sample_old_sha256': sha(old_path)},
                'labels_sha256': sha(label_dir / 'labels.jsonl'),
                'samples_sha256': sha(sample_dir / 'samples.jsonl'),
                'sample_report': json.loads((sample_dir / 'SAMPLE_FINAL.json').read_text()),
                'label_report': json.loads((label_dir / 'LABEL_FINAL.json').read_text()),
                'physics_input': {'sha256': sha(sample_dir/'physics.jsonl'), 'utf8': (sample_dir/'physics.jsonl').read_text()},
                'old_physics_input': {'sha256': sha(old_path), 'utf8': old_path.read_text()},
                'samples': samples,
                'labels': [{k: x[k] for k in ['trajectory_id', 'group_id', 'status', 'verified',
                    'terminal_energy', 'actual_steps', 'error', 'versions']} for x in labels]}
    path = root / 'FIT8_COMPACT.json'
    assert not path.exists()
    path.write_text(json.dumps(result, separators=(',', ':')) + '\n')
    print(json.dumps({'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}))


if __name__ == '__main__':
    main(sys.argv[1])
