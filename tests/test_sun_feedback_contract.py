import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from crystal_dlm.sun_feedback_contract import SCHEMA, sha256, validate_training_feedback


def fixture(root):
    parent = {'ancestor_id': 'p0', 'source_row_idx': 7, 'source_split': 'train',
              'plan_state': {'N': 2, 'elements': ['Na', 'Cl'], 'counts': [1, 1]}}
    records = [{'trajectory_id': 'a0', 'group_id': 'p0', 'source_row_idx': 7, 'source_split': 'train',
                'purpose': 'training_feedback', 'endpoint': 'native', 'success': True,
                'body': '<N_002><E_Na><E_Cl>', 'structure': None}]
    (root/'parents.jsonl').write_text(json.dumps(parent)+'\n')
    (root/'heldout.jsonl').write_text(json.dumps({'body_eligible': True,
        'plan_state': {'N': 2, 'elements': ['K', 'Cl'], 'counts': [1, 1]}})+'\n')
    (root/'paths.jsonl').write_text(json.dumps(records[0])+'\n')
    (root/'PREPARATION_FINAL.json').write_text(json.dumps({'files_sha256': {'parents.jsonl': sha256(root/'parents.jsonl')},
         'heldout_cohort_sha256': sha256(root/'heldout.jsonl')}))
    (root/'_SUCCESS').touch()
    spec = {'schema': SCHEMA, 'purpose': 'training_feedback', 'endpoint': 'native', 'expected_requests': 1}
    for name, filename in [('paths', 'paths.jsonl'), ('parent_pairs', 'parents.jsonl'),
                           ('parent_preparation', 'PREPARATION_FINAL.json'), ('heldout_cohort', 'heldout.jsonl')]:
        spec[name] = {'path': str(root/filename), 'sha256': sha256(root/filename)}
    (root/'scope.json').write_text(json.dumps(spec))
    return records, spec


class FeedbackScopeTests(unittest.TestCase):
    def test_bound_training_input_is_accepted_and_cannot_be_evaluation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records, _ = fixture(root)
            result = validate_training_feedback(records, root/'paths.jsonl', root/'scope.json')
            self.assertFalse(result['independent_evaluation'])
            records[0]['source_split'] = 'evaluation'
            with self.assertRaisesRegex(ValueError, 'relabel'):
                validate_training_feedback(records, root/'paths.jsonl', root/'scope.json')

    def test_changed_parent_bytes_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records, _ = fixture(root)
            with (root/'parents.jsonl').open('a') as handle:
                handle.write(' ')
            with self.assertRaisesRegex(ValueError, 'parent_pairs changed'):
                validate_training_feedback(records, root/'paths.jsonl', root/'scope.json')

    def test_candidate_cannot_change_composition(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records, _ = fixture(root)
            records[0]['body'] = '<N_002><E_K><E_Cl>'
            with self.assertRaisesRegex(ValueError, 'fixed composition'):
                validate_training_feedback(records, root/'paths.jsonl', root/'scope.json')

    def test_heldout_overlap_is_rejected_even_with_matching_manifests(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records, spec = fixture(root)
            (root/'heldout.jsonl').write_text(json.dumps({'body_eligible': True,
                'plan_state': {'N': 4, 'elements': ['Na', 'Cl'], 'counts': [2, 2]}})+'\n')
            prepared = json.loads((root/'PREPARATION_FINAL.json').read_text())
            prepared['heldout_cohort_sha256'] = sha256(root/'heldout.jsonl')
            (root/'PREPARATION_FINAL.json').write_text(json.dumps(prepared))
            for name in ['parent_preparation', 'heldout_cohort']:
                spec[name]['sha256'] = sha256(spec[name]['path'])
            (root/'scope.json').write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, 'heldout exclusion'):
                validate_training_feedback(records, root/'paths.jsonl', root/'scope.json')

    def test_rewritten_dev_origin_cannot_become_training(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records, spec = fixture(root)
            parent = json.loads((root/'parents.jsonl').read_text())
            parent['source_split'] = 'dev'
            (root/'parents.jsonl').write_text(json.dumps(parent)+'\n')
            prepared = json.loads((root/'PREPARATION_FINAL.json').read_text())
            prepared['files_sha256']['parents.jsonl'] = sha256(root/'parents.jsonl')
            (root/'PREPARATION_FINAL.json').write_text(json.dumps(prepared))
            for name in ['parent_preparation', 'parent_pairs']:
                spec[name]['sha256'] = sha256(spec[name]['path'])
            (root/'scope.json').write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, 'source identity'):
                validate_training_feedback(records, root/'paths.jsonl', root/'scope.json')


if __name__ == '__main__':
    unittest.main()
