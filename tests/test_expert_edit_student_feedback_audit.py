"""Adversarial feedback attribution checks without a live MLIP or GPU."""
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
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

    def additional_labels(self, fixture, *, mid_energy=-1.15, worker_error=False, real_mson=False):
        root, _, _, samples, _, _ = fixture
        prepared, labels = root / 'additional-prepared', root / 'additional-labels'
        with nullcontext() if real_mson else patch.object(data, 'physics_input', side_effect=endpoint):
            data.prepare_student_proposals(samples, self.tokenizer, prepared)
        inputs = data.read_rows(prepared / 'inputs.jsonl')
        labels.mkdir()
        rows = []
        for record in inputs:
            row = {key: record[key] for key in ('trajectory_id', 'group_id', 'source_row_idx', 'source_split', 'endpoint')}
            row.update(status='worker_error' if worker_error else 'verified', verified=not worker_error,
                       terminal_energy=None if worker_error else mid_energy,
                       final_structure=None if worker_error else record['structure'],
                       versions=None if worker_error else VERSIONS,
                       endpoint_cache_key=data.endpoint_fingerprint(record))
            rows.append(row)
        write_rows(labels / 'labels.jsonl', rows)
        (labels / 'LABEL_FINAL.json').write_text(json.dumps({
            'purpose': 'expert_edit', 'protocol': data.COMMON_RELAXATION_PROTOCOL,
            'verification_protocol': data.TERMINAL_VERIFICATION_PROTOCOL,
            'requested': len(rows), 'completed': len(rows),
            'statuses': dict(Counter(row['status'] for row in rows)),
            'runtime_identities': [VERSIONS], 'input_sha256': data.sha256(prepared / 'inputs.jsonl'),
        }), encoding='utf-8')
        (labels / ('_ENGINEERING_FAILED' if worker_error else '_SUCCESS')).touch()
        return prepared, labels

    def compile_with_additional(self, fixture, additional, *, real_mson=False):
        root, prepared, reference, samples, labels, _ = fixture
        with nullcontext() if real_mson else patch.object(data, 'physics_input', side_effect=endpoint):
            report = data.compile_student_feedback(samples, labels, [prepared], [reference], self.tokenizer,
                                                   root / 'compiled-additional', proposal_prepared=additional[0],
                                                   proposal_labels=additional[1])
        return report, data.read_rows(root / 'compiled-additional/train.jsonl')

    def priority_dataset(self, *, healthy_fraction=0.):
        from crystal_dlm.expert_edit import ExpertEditDataset

        fixture = self.fixture()
        root, *_, bodies = fixture
        directory = root / 'priority-dataset'
        directory.mkdir()
        good = bodies['teacher']
        bad = bodies['old'].copy()
        bad[12:15] = bad[8:11]
        base = {'schema': data.SCHEMA, 'task': 'G', 'source_split': 'train', 'source_row_idx': 1,
                'composition_key': 'Cl:1|Na:1', 'prompt': 'actual immutable rich prompt', 'num_atoms': 2,
                'old_body': bad, 'target_body': good, 'old_geometry': data.certify_geometry(data.decode_body(bad, self.inverse)),
                'target_geometry': data.certify_geometry(data.decode_body(good, self.inverse)),
                'old_reliable': None, 'target_reliable': None, 'gain_eV_atom': None,
                'action': data.full_action(bad, good, 2), 'accept_label': True}
        offline = dict(base, record_id='offline_content', ancestor_id='offline_source',
                       source_kind='current_B0_full_rich', content_supervision=True)
        rows, genuine = [offline], set()
        for index in range(4):
            record_id = f'feedback:refresh:source_A:proposal_{index}:G'
            rows.append(dict(base, record_id=record_id, ancestor_id='source_A',
                             source_kind='student_proposal_feedback', content_supervision=False,
                             original_teacher_reference=False))
            genuine.add(record_id)
        negative_id = 'feedback:refresh:source_B:proposal_0:G'
        rows.append(dict(rows[-1], record_id=negative_id, ancestor_id='source_B', accept_label=False,
                         old_body=good, target_body=bad, old_geometry=base['target_geometry'],
                         target_geometry=base['old_geometry'], action=data.full_action(good, bad, 2)))
        genuine.add(negative_id)
        # A directory name containing "proposal_" must not turn a whole
        # transaction into a real-proposal priority example.
        rows.extend([
            dict(rows[1], record_id='feedback:proposal_refresh:source_C:observed_complete_transaction:G', ancestor_id='source_C'),
            dict(rows[1], record_id='feedback:proposal_refresh:source_D:teacher_from_student_abc:G', ancestor_id='source_D',
                 original_teacher_reference=True),
            dict(rows[1], record_id='feedback:refresh:source_E:proposal_0:G', ancestor_id='source_E', accept_label=None),
            dict(rows[1], record_id='feedback:refresh:source_F:proposal_0:G', ancestor_id='source_F', accept_label=0),
        ])
        state = {key: base[key] for key in ('schema', 'task', 'source_split', 'source_row_idx', 'composition_key',
                                          'prompt', 'num_atoms', 'old_reliable', 'gain_eV_atom')}
        state.update(record_id='student_unknown_state', ancestor_id='state_source', state_only=True,
                     source_kind='student_proposal_feedback', old_body=good, old_geometry=base['target_geometry'],
                     accept_label=None, content_supervision=False)
        rows.append(state)
        if healthy_fraction:
            rows.append(dict(state, record_id='verified_healthy_state', ancestor_id='healthy_source',
                             task='S', old_reliable=True))
            rows.append(dict(state, record_id='invalid_but_reliable_flag', ancestor_id='invalid_source',
                             old_geometry={'valid': False, 'certified': True}, old_reliable=True))
        write_rows(directory / 'train.jsonl', rows)
        (directory / 'DATA_FINAL.json').write_text(json.dumps({
            'schema': data.SCHEMA, 'output_sha256': {'train.jsonl': data.sha256(directory / 'train.jsonl')},
        }), encoding='utf-8')
        (directory / '_SUCCESS').touch()
        dataset = ExpertEditDataset([directory], self.tokenizer, seed=20260908, size=4096,
                                   content_fraction=.2, inspect_fraction=.3, student_feedback_fraction=1.,
                                   healthy_state_fraction=healthy_fraction)
        return dataset, genuine

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

    def test_additional_mid_relaxation_sets_local_gain_not_compound_gain(self):
        for energy, expected_gain, expected_accept in ((-1.15, .05, True), (-1.3, -.1, False)):
            with self.subTest(mid_energy=energy):
                fixture = self.fixture()
                additional = self.additional_labels(fixture, mid_energy=energy)
                _, rows = self.compile_with_additional(fixture, additional)
                action = next(row for row in rows if ':proposal_1:S' in row['record_id'])
                self.assertAlmostEqual(action['gain_eV_atom'], expected_gain)
                self.assertIs(action['accept_label'], expected_accept)
                self.assertFalse(action['content_supervision'])
                self.assertTrue(action['old_physics_id'].startswith('expert-proposal:'))
                transaction = next(row for row in rows if ':observed_complete_transaction:S' in row['record_id'])
                self.assertAlmostEqual(transaction['gain_eV_atom'], .2)
                self.assertTrue(transaction['accept_label'])

    def test_real_mson_json_roundtrip_preserves_full_proposal_binding(self):
        fixture = self.fixture()
        _, prepared, reference, samples, labels, bodies = fixture
        # Replace the lightweight geometry fixtures with actual pymatgen MSON
        # for every endpoint. All subsequent preparation/rebinding is real.
        for input_path, label_dir in ((prepared / 'all_inputs.jsonl', reference),
                                      (samples / 'physics.jsonl', labels)):
            updated = []
            for record in data.read_rows(input_path):
                body = bodies[record['trajectory_id']]
                updated.append(data.physics_input(record['trajectory_id'], record['group_id'],
                    record['source_row_idx'], record['source_split'], record['endpoint'],
                    data.decode_body(body, self.inverse), ''.join(self.inverse[token] for token in body)))
            write_rows(input_path, updated)
            by_id = {row['trajectory_id']: row for row in updated}
            label_rows = data.read_rows(label_dir / 'labels.jsonl')
            for row in label_rows:
                source = by_id[row['trajectory_id']]
                row['final_structure'] = source['structure']
                row['endpoint_cache_key'] = data.endpoint_fingerprint(source)
            write_rows(label_dir / 'labels.jsonl', label_rows)
            report_path = label_dir / 'LABEL_FINAL.json'
            report = json.loads(report_path.read_text())
            report['input_sha256'] = data.sha256(input_path)
            report_path.write_text(json.dumps(report), encoding='utf-8')
        preparation_path = prepared / 'PREPARATION_FINAL.json'
        preparation = json.loads(preparation_path.read_text())
        preparation['files_sha256']['all_inputs.jsonl'] = data.sha256(prepared / 'all_inputs.jsonl')
        preparation_path.write_text(json.dumps(preparation), encoding='utf-8')

        additional = self.additional_labels(fixture, real_mson=True)
        expected, expected_map = data.student_proposal_inputs(data.read_rows(samples / 'samples.jsonl'), self.tokenizer)
        persisted = data.read_rows(additional[0] / 'inputs.jsonl')
        self.assertIsInstance(expected[0]['structure']['lattice']['pbc'], tuple)
        self.assertIsInstance(persisted[0]['structure']['lattice']['pbc'], list)
        self.assertNotEqual(expected, persisted)
        self.assertEqual(json.dumps(expected, sort_keys=True), json.dumps(persisted, sort_keys=True))
        self.assertEqual(expected_map, data.read_rows(additional[0] / 'proposal_map.jsonl'))
        _, rows = self.compile_with_additional(fixture, additional, real_mson=True)
        action = next(row for row in rows if ':proposal_1:S' in row['record_id'])
        self.assertAlmostEqual(action['gain_eV_atom'], .05)
        self.assertTrue(action['old_physics_id'].startswith('expert-proposal:'))

    def test_rejected_complete_proposals_are_included_once_without_quality_selection(self):
        fixture = self.fixture()
        sample = data.read_rows(fixture[3] / 'samples.jsonl')[0]
        old, mid, final = (fixture[-1][key] for key in ('old', 'mid', 'final'))
        sample['output']['trace'] = [
            {'task': 'G', 'old_body': old, 'proposal_body': mid, 'accepted': False, 'applied': False,
             'quality': [0., 0., 0., 0.], **data.full_action(old, mid, 2)},
            {'task': 'S', 'old_body': old, 'proposal_body': mid, 'accepted': False, 'applied': False,
             'quality': [1., 1., 1., 1.], **data.full_action(old, mid, 2)},
            {'task': 'S', 'old_body': old, 'proposal_body': final, 'accepted': True, 'applied': True,
             **data.full_action(old, final, 2)},
        ]
        with patch.object(data, 'physics_input', side_effect=endpoint):
            inputs, mapping = data.student_proposal_inputs([sample], self.tokenizer)
            altered = deepcopy(sample)
            altered['output']['trace'][0]['quality'] = [999.] * 4
            altered['output']['trace'][1]['terminal_energy'] = -1e9
            second = data.student_proposal_inputs([altered], self.tokenizer)
        self.assertEqual(len(inputs), 1)
        self.assertEqual(mapping[0]['body'], mid)
        self.assertEqual(mapping[0]['trace_indices'], [0, 1])
        self.assertEqual((inputs, mapping), second)

    def test_additional_proposal_cannot_be_omitted_by_rewriting_preparation_hashes(self):
        fixture = self.fixture()
        additional = self.additional_labels(fixture)
        prepared = additional[0]
        for name in ('inputs.jsonl', 'proposal_map.jsonl'):
            write_rows(prepared / name, [])
        report_path = prepared / 'PROPOSALS_FINAL.json'
        report = json.loads(report_path.read_text())
        report['additional_endpoints'] = 0
        report['files_sha256'] = {name: data.sha256(prepared / name) for name in ('inputs.jsonl', 'proposal_map.jsonl')}
        report_path.write_text(json.dumps(report), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'cover exactly'):
            self.compile_with_additional(fixture, additional)

    def test_additional_labels_are_bound_to_exact_sample_and_ancestor(self):
        for mutation in ('sample', 'ancestor'):
            with self.subTest(mutation=mutation):
                fixture = self.fixture()
                additional = self.additional_labels(fixture)
                prepared = additional[0]
                report_path = prepared / 'PROPOSALS_FINAL.json'
                report = json.loads(report_path.read_text())
                if mutation == 'sample':
                    report['sample_sha256'] = '0' * 64
                else:
                    mapping = data.read_rows(prepared / 'proposal_map.jsonl')
                    mapping[0]['ancestor_id'] = 'another_training_source'
                    write_rows(prepared / 'proposal_map.jsonl', mapping)
                    report['files_sha256']['proposal_map.jsonl'] = data.sha256(prepared / 'proposal_map.jsonl')
                report_path.write_text(json.dumps(report), encoding='utf-8')
                with self.assertRaises(ValueError):
                    self.compile_with_additional(fixture, additional)

    def test_proposal_preparation_rejects_dev_and_duplicate_ancestors(self):
        fixture = self.fixture()
        samples_directory = fixture[3]
        summary_path = samples_directory / 'SAMPLE_FINAL.json'
        summary = json.loads(summary_path.read_text())
        summary['split'] = 'dev'
        summary_path.write_text(json.dumps(summary), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'autonomous training sample'):
            data.prepare_student_proposals(samples_directory, self.tokenizer, fixture[0] / 'dev-proposals')
        sample = data.read_rows(samples_directory / 'samples.jsonl')[0]
        for rows in ([dict(sample, source_split='dev')], [sample, deepcopy(sample)]):
            with self.subTest(rows=len(rows)):
                with patch.object(data, 'physics_input', side_effect=endpoint), self.assertRaisesRegex(ValueError, 'training-only'):
                    data.student_proposal_inputs(rows, self.tokenizer)

    def test_development_proposals_preserve_split_and_use_an_untrainable_schema(self):
        fixture = self.fixture()
        samples = fixture[3]
        rows = data.read_rows(samples / 'samples.jsonl')
        rows[0]['source_split'] = 'dev'
        write_rows(samples / 'samples.jsonl', rows)
        summary = json.loads((samples / 'SAMPLE_FINAL.json').read_text())
        summary['split'] = 'dev'
        (samples / 'SAMPLE_FINAL.json').write_text(json.dumps(summary))
        output = fixture[0] / 'development-physics'
        report = data.prepare_development_proposals(samples, self.tokenizer, output)
        self.assertEqual(report['schema'], 'development_proposal_physics_v1')
        self.assertFalse(report['training_use_allowed'])
        self.assertEqual(report['additional_endpoints'], 1)
        self.assertEqual({x['source_split'] for x in data.read_rows(output / 'inputs.jsonl')}, {'dev'})
        with self.assertRaisesRegex(ValueError, 'training-only'):
            data.student_proposal_inputs(rows, self.tokenizer)

    def test_development_preparation_cannot_rename_training_samples(self):
        fixture = self.fixture()
        with self.assertRaisesRegex(ValueError, 'autonomous development sample'):
            data.prepare_development_proposals(fixture[3], self.tokenizer, fixture[0] / 'wrong-split')

    def test_additional_worker_error_excludes_all_supervision_for_ancestor(self):
        fixture = self.fixture()
        report, rows = self.compile_with_additional(fixture, self.additional_labels(fixture, worker_error=True))
        self.assertEqual(rows, [])
        self.assertEqual(report['admitted_sources'], 0)
        self.assertEqual(report['excluded_engineering_ancestors'], ['source'])

    def test_priority_judges_only_known_real_proposals_with_uniform_ancestors(self):
        dataset, genuine = self.priority_dataset()
        self.assertEqual({row['record_id'] for row in dataset.student_judgements}, genuine)
        views = [dataset[index] for index in range(len(dataset))]
        judges = [view for view in views if view['kind'] == 'judge']
        self.assertTrue(judges)
        self.assertTrue(all(view['record_id'] in genuine and type(view['quality_targets'][3]) is bool for view in judges))
        counts = Counter(view['ancestor_id'] for view in judges)
        self.assertEqual(set(counts), {'source_A', 'source_B'})
        self.assertAlmostEqual(counts['source_A'] / len(judges), .5, delta=.07)
        inspections = [view for view in views if view['kind'] == 'inspect']
        self.assertTrue(all(view['record_id'] == 'student_unknown_state' for view in inspections))
        self.assertTrue(all(view['quality_targets'][1] is None and view['mode_target'] == -100 for view in inspections))

    def test_healthy_priority_is_G_none_without_content_or_S_stop(self):
        dataset, _ = self.priority_dataset(healthy_fraction=1.)
        self.assertEqual({row['record_id'] for row in dataset.healthy_states}, {'verified_healthy_state'})
        views = [dataset[index] for index in range(len(dataset))]
        inspections = [view for view in views if view['kind'] == 'inspect']
        self.assertTrue(inspections)
        for view in inspections:
            self.assertEqual(view['record_id'], 'verified_healthy_state')
            self.assertEqual(view['task'], 0)
            self.assertEqual(view['mode_target'], 0)
            self.assertEqual(view['active'], [])
            self.assertEqual(view['quality_targets'], [True, True, None, None])
            self.assertTrue(all(value == -100 for value in view['targets']))


if __name__ == '__main__':
    unittest.main()
