"""Adversarial feedback attribution checks without a live MLIP or GPU."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crystal_dlm import expert_edit_data as data
from crystal_dlm.fixed_slot import build_special_tokens


VERSIONS = {
    'model': 'CHGNet-0.3.0', 'model_checkpoint_sha256': 'c' * 64,
    'chgnet_package': 'fixture', 'ase_package': 'fixture', 'torch_package': 'fixture',
    'pymatgen_package': 'fixture', 'labeler_sha256': 'a' * 64,
}


class Tokenizer:
    def get_vocab(self):
        return {token: index for index, token in enumerate(build_special_tokens())}

    def __call__(self, text, **kwargs):
        return {'input_ids': [0]}


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def endpoint(record_id, group, source_idx, split, endpoint_name, arrays=None, body=None):
    structure = {
        'lattice': dict(zip(('a', 'b', 'c', 'alpha', 'beta', 'gamma'),
                            arrays['lengths'] + arrays['angles'])),
        'sites': [{'species': [{'element': symbol, 'occu': 1}], 'abc': coord}
                  for symbol, coord in zip(arrays['species'], arrays['frac_coords'])],
    }
    return {'trajectory_id': record_id, 'group_id': group, 'source_row_idx': source_idx,
            'source_split': split, 'purpose': 'expert_edit', 'endpoint': endpoint_name,
            'success': True, 'structure': structure, 'body': body}


class StudentFeedbackAttributionAudit(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.tokenizer = Tokenizer()
        self.vocabulary = self.tokenizer.get_vocab()
        self.inverse = {value: key for key, value in self.vocabulary.items()}

    def fixture(self, *, worker_error=None, finish='final', invalid_trace=False, accept_all=False,
                budget_stop=False):
        root = Path(tempfile.mkdtemp(dir=self.temporary.name))
        prepared, reference, samples, labels = [root / key for key in ('prepared', 'reference', 'samples', 'labels')]
        for path in (prepared, reference, samples, labels):
            path.mkdir()
        bodies, arrays = {}, {}
        for key, length in (('old', 4.), ('mid', 4.1), ('final', 4.2), ('teacher', 4.3)):
            raw = {'lengths': [length] * 3, 'angles': [90.] * 3, 'species': ['Na', 'Cl'],
                   'frac_coords': [[0., 0., 0.], [.5, .5, .5]]}
            bodies[key], arrays[key], _ = data.quantize_arrays(raw, self.vocabulary)
        pair = {
            'schema': data.SCHEMA, 'ancestor_id': 'source', 'source_split': 'train',
            'source_row_idx': 7, 'composition_key': 'Cl:1|Na:1',
            'plan_state': {'N': 2, 'elements': ['Na', 'Cl'], 'counts': [1, 1]},
            'prompt': 'actual immutable rich prompt', 'num_atoms': 2,
            'old_body': bodies['old'], 'target_body': bodies['teacher'],
            'teacher_available': True, 'old_physics_id': 'old', 'target_physics_id': 'teacher',
        }
        write_rows(prepared / 'pairs_pending.jsonl', [pair])
        refs = [endpoint(key, 'source', 7, 'train', kind, arrays[key],
                         ''.join(self.inverse[token] for token in bodies[key]))
                for key, kind in (('old', 'native'), ('teacher', 'expert_quantized'))]
        current = endpoint('final', 'source', 7, 'train', 'expert_quantized', arrays[finish],
                           ''.join(self.inverse[token] for token in bodies[finish]))
        write_rows(prepared / 'all_inputs.jsonl', refs)
        write_rows(samples / 'physics.jsonl', [current])
        (prepared / 'PREPARATION_FINAL.json').write_text(json.dumps({
            'heldout_cohort_sha256': 'b' * 64,
            'files_sha256': {name: data.sha256(prepared / name)
                             for name in ('pairs_pending.jsonl', 'all_inputs.jsonl')},
        }), encoding='utf-8')
        (prepared / '_SUCCESS').touch()

        def bundle(destination, input_path, inputs, energies):
            rows = []
            for record, energy in zip(inputs, energies):
                failed = record['trajectory_id'] == worker_error
                row = {key: record[key] for key in ('trajectory_id', 'group_id', 'source_row_idx', 'source_split', 'endpoint')}
                row.update(status='worker_error' if failed else 'verified', verified=not failed,
                           terminal_energy=None if failed else energy, versions=None if failed else VERSIONS,
                           final_structure=None if failed else record['structure'],
                           endpoint_cache_key=data.endpoint_fingerprint(record))
                rows.append(row)
            write_rows(destination / 'labels.jsonl', rows)
            report = {'purpose': 'expert_edit', 'protocol': data.COMMON_RELAXATION_PROTOCOL,
                      'verification_protocol': data.TERMINAL_VERIFICATION_PROTOCOL,
                      'requested': len(rows), 'completed': len(rows),
                      'statuses': dict(Counter(row['status'] for row in rows)),
                      'runtime_identities': [VERSIONS], 'input_sha256': data.sha256(input_path)}
            (destination / 'LABEL_FINAL.json').write_text(json.dumps(report), encoding='utf-8')
            (destination / ('_ENGINEERING_FAILED' if any(row['status'] == 'worker_error' for row in rows)
                            else '_SUCCESS')).touch()

        bundle(reference, prepared / 'all_inputs.jsonl', refs, [-1., -1.4])
        bundle(labels, samples / 'physics.jsonl', [current], [-1.2])
        first_old = bodies['teacher'] if invalid_trace else bodies['old']
        trace = [{'task': 'G', 'old_body': first_old, 'proposal_body': bodies['mid'],
                  'accepted': True, 'applied': True, 'reason': 'learned_accept',
                  **data.full_action(first_old, bodies['mid'], 2)}]
        if budget_stop:
            trace.append({'task': 'S', 'mode': 'full_cell', 'old_body': bodies['mid'],
                          'proposal_body': bodies['mid'], 'accepted': False, 'applied': False,
                          'reason': 'insufficient_complete_proposal_budget', 'calls': 1})
        else:
            trace.append({'task': 'S', 'old_body': bodies['mid'], 'proposal_body': bodies['final'],
                          'accepted': True, 'applied': True, 'reason': 'learned_accept',
                          **data.full_action(bodies['mid'], bodies['final'], 2)})
        sample = {key: pair[key] for key in ('ancestor_id', 'source_split', 'source_row_idx', 'prompt', 'num_atoms', 'old_body')}
        sample['output'] = {'canonical_body': bodies[finish], 'trace': trace, 'scope_policy': 'learned',
                            'accept_all': accept_all, 'S_admission_policy': 'learned'}
        write_rows(samples / 'samples.jsonl', [sample])
        (samples / 'SAMPLE_FINAL.json').write_text(json.dumps({
            'split': 'train', 'source_kind': 'all_old_states', 'requested': 1, 'frozen_B0_control': False,
        }), encoding='utf-8')
        (samples / '_SUCCESS').touch()
        return root, prepared, reference, samples, labels, bodies

    def compile(self, fixture):
        root, prepared, reference, samples, labels, _ = fixture
        with patch.object(data, 'physics_input', side_effect=endpoint):
            report = data.compile_student_feedback(samples, labels, [prepared], [reference],
                                                   self.tokenizer, root / 'compiled')
        return report, data.read_rows(root / 'compiled/train.jsonl')

    def test_compound_reward_does_not_label_unknown_intermediate_actions(self):
        fixture = self.fixture()
        report, rows = self.compile(fixture)
        mid = fixture[-1]['mid']
        proposals = [row for row in rows if ':proposal_' in row['record_id']]
        self.assertEqual(len(proposals), 2)
        self.assertTrue(all(row['accept_label'] is None and row['gain_eV_atom'] is None
                            and not row['content_supervision'] for row in proposals))
        transaction = [row for row in rows if ':observed_complete_transaction:S' in row['record_id']]
        self.assertEqual(len(transaction), 1)
        self.assertTrue(transaction[0]['accept_label'])
        self.assertTrue(transaction[0]['content_supervision'])
        self.assertAlmostEqual(transaction[0]['gain_eV_atom'], .2)
        from_mid = [row for row in rows if row.get('original_teacher_reference') and row['old_body'] == mid]
        self.assertTrue(from_mid)
        self.assertTrue(all(row['accept_label'] is None and not row['content_supervision'] for row in from_mid))
        self.assertEqual(report['content_positive_sources']['S'], 1)
        self.assertTrue(all('target_body' not in row for row in rows if row.get('state_only')))

    def test_any_reference_or_current_worker_error_excludes_the_whole_ancestor(self):
        for failed in ('old', 'teacher', 'final'):
            with self.subTest(failed=failed):
                report, rows = self.compile(self.fixture(worker_error=failed))
                self.assertEqual(rows, [])
                self.assertEqual(report['admitted_sources'], 0)
                self.assertEqual(report['excluded_engineering_ancestors'], ['source'])

    def test_trace_must_start_from_the_retained_source_and_compose_to_final(self):
        with self.assertRaises(ValueError):
            self.compile(self.fixture(invalid_trace=True))

    def test_accept_all_diagnostic_is_not_autonomous_feedback(self):
        with self.assertRaises(ValueError):
            self.compile(self.fixture(accept_all=True))

    def test_budget_stop_is_not_a_completed_proposal_or_keep_reward(self):
        _, rows = self.compile(self.fixture(finish='mid', budget_stop=True))
        proposals = [row for row in rows if ':proposal_' in row['record_id']]
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]['task'], 'G')

    def test_rejected_but_good_student_proposal_trains_judge_without_content(self):
        from crystal_dlm.expert_edit import ExpertEditDataset

        root = Path(tempfile.mkdtemp(dir=self.temporary.name))
        geometry = {'lengths': [4.] * 3, 'angles': [90.] * 3, 'species': ['Na', 'Cl'],
                    'frac_coords': [[0., 0., 0.], [.5, .5, .5]]}
        good, good_arrays, _ = data.quantize_arrays(geometry, self.vocabulary)
        geometry['frac_coords'][1] = [0., 0., 0.]
        bad, bad_arrays, _ = data.quantize_arrays(geometry, self.vocabulary)
        base = {
            'schema': data.SCHEMA, 'task': 'G', 'source_split': 'train', 'source_row_idx': 1,
            'composition_key': 'Cl:1|Na:1', 'prompt': 'actual immutable rich prompt',
            'num_atoms': 2, 'old_body': bad, 'target_body': good,
            'old_geometry': data.certify_geometry(bad_arrays),
            'target_geometry': data.certify_geometry(good_arrays),
            'old_reliable': None, 'target_reliable': None, 'gain_eV_atom': None,
            'action': data.full_action(bad, good, 2), 'accept_label': True,
        }
        offline = dict(base, record_id='offline_content', ancestor_id='offline_source',
                       source_kind='current_B0_full_rich', content_supervision=True)
        # The old gate rejected this proposal; completed geometry evidence now
        # makes it a judge positive, without making it a token-training target.
        student = dict(base, record_id='student_good_rejected', ancestor_id='student_source',
                       source_kind='student_proposal_feedback', content_supervision=False,
                       accepted_by_student=False)
        negative = dict(student, record_id='student_bad', ancestor_id='bad_source',
                        old_body=good, target_body=bad, old_geometry=base['target_geometry'],
                        target_geometry=base['old_geometry'], accept_label=False,
                        action=data.full_action(good, bad, 2))
        write_rows(root / 'train.jsonl', [offline, student, negative])
        (root / 'DATA_FINAL.json').write_text(json.dumps({
            'schema': data.SCHEMA, 'output_sha256': {'train.jsonl': data.sha256(root / 'train.jsonl')},
        }), encoding='utf-8')
        (root / '_SUCCESS').touch()
        dataset = ExpertEditDataset([root], self.tokenizer, seed=20260908, size=4096)
        sampled = [dataset[index] for index in range(len(dataset))]
        judge_positive = [view for view in sampled if view['kind'] == 'judge'
                          and view['record_id'] == student['record_id']]
        self.assertTrue(judge_positive)
        self.assertTrue(all(view['quality_targets'][3] is True for view in judge_positive))
        self.assertTrue(all(all(value == -100 for value in view['targets']) for view in judge_positive))
        self.assertFalse(any(view['kind'] == 'content' and view['record_id'] == student['record_id']
                             for view in sampled))


if __name__ == '__main__':
    unittest.main()
